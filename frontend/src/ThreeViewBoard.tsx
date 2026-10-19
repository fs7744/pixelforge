import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { api, Asset } from './api'
import { openPreview } from './Preview'

/**
 * 三视图流程画布。
 *
 * 之前是一次请求跑完「正面 → 侧面 → 背面 → 拼板」，整套 5-9 分钟，
 * 中间任何一张不满意都只能从头再来。这里改成无限画布式的四步流程：
 * 每步一个节点，看完再决定下一步，单步不满意就换种子重跑那一张。
 *
 * 依赖关系（重跑前面的会让后面的作废，因为造型变了）：
 *   front ──┬─→ side ──┐
 *           └─→ back ──┴─→ strip
 */

export type ThreeViewParams = {
  prompt: string
  styleId: string
  width: number
  height: number
  steps: number
  cfg: number
  seed: string
}

export type BoardHandle = { start: () => void }

type StepKey = 'front' | 'side' | 'back' | 'strip'
type Status = 'idle' | 'running' | 'done' | 'error'

type NodeState = {
  status: Status
  asset?: Asset
  seconds?: number
  seed?: number | null
  /** 生成时用的角色描述：用户改了描述但还没重跑时，节点上给出提示 */
  prompt?: string
  error?: string
  x: number
  y: number
}

type Props = {
  params: ThreeViewParams
  onAsset: (a: Asset) => void
  onChange?: (s: { hasFront: boolean; busy: boolean }) => void
  onSendToPost: (a: Asset) => void
}

const KEY = 'pf.threeview.board.v1'

const STEPS: { key: StepKey; no: string; title: string; hint: string }[] = [
  { key: 'front', no: '01', title: '正视图', hint: '文生图起点，后两张都从它推' },
  { key: 'side', no: '02', title: '侧视图', hint: '编辑指令转 90°，造型不动' },
  { key: 'back', no: '03', title: '背视图', hint: '编辑指令转 180°，脸完全不可见' },
  { key: 'strip', no: '04', title: '三视图拼板', hint: '抠底 → 统一暗绿底 → 基线对齐' },
]

// 画布几何：节点固定宽度，横向排开，连线走节点垂直中线
const W = 300
const H = 452
const GAP = 120
const TOP = 90
const LEFT = 70
const WORLD_W = LEFT * 2 + 4 * W + 3 * GAP
const WORLD_H = TOP * 2 + H

function layoutPos(i: number) {
  return { x: LEFT + i * (W + GAP), y: TOP }
}

function idleNodes(): Record<StepKey, NodeState> {
  const out = {} as Record<StepKey, NodeState>
  STEPS.forEach((s, i) => {
    out[s.key] = { status: 'idle', ...layoutPos(i) }
  })
  return out
}

function initNodes(): Record<StepKey, NodeState> {
  const base = idleNodes()
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || 'null')
    const saved = raw?.nodes || {}
    STEPS.forEach((s) => {
      const n = saved[s.key]
      // 只恢复真正出过图的节点；生成中的状态刷新后没有意义
      if (!n?.asset?.id) return
      base[s.key] = {
        ...base[s.key],
        status: 'done',
        asset: n.asset,
        seconds: n.seconds,
        seed: n.seed ?? null,
        prompt: n.prompt,
        x: typeof n.x === 'number' ? n.x : base[s.key].x,
        y: typeof n.y === 'number' ? n.y : base[s.key].y,
      }
    })
  } catch {
    // 存了坏值就当没有
  }
  return base
}

const ThreeViewBoard = forwardRef<BoardHandle, Props>(function ThreeViewBoard(
  { params, onAsset, onChange, onSendToPost },
  ref,
) {
  const [nodes, setNodes] = useState<Record<StepKey, NodeState>>(initNodes)
  const [view, setView] = useState({ x: 0, y: 0, k: 1 })
  const [dragging, setDragging] = useState(false)

  const wrapRef = useRef<HTMLDivElement>(null)
  const nodesRef = useRef(nodes)
  const paramsRef = useRef(params)
  const viewRef = useRef(view)
  const busyRef = useRef(false)
  const dragRef = useRef<{
    mode: 'pan' | 'node'
    key?: StepKey
    sx: number
    sy: number
    ox: number
    oy: number
  } | null>(null)

  useEffect(() => {
    nodesRef.current = nodes
  }, [nodes])
  useEffect(() => {
    paramsRef.current = params
  }, [params])
  useEffect(() => {
    viewRef.current = view
  }, [view])

  const busy = STEPS.some((s) => nodes[s.key].status === 'running')

  const fit = useCallback(() => {
    const el = wrapRef.current
    if (!el) return
    const cw = el.clientWidth
    const ch = el.clientHeight
    if (!cw || !ch) return
    // 按实际节点包围盒算，而不是固定的初始网格——用户把节点拖散了也能框住
    const ns = Object.values(nodesRef.current)
    const minX = Math.min(...ns.map((n) => n.x))
    const minY = Math.min(...ns.map((n) => n.y))
    const bw = Math.max(...ns.map((n) => n.x + W)) - minX
    const bh = Math.max(...ns.map((n) => n.y + H)) - minY
    const k = Math.min(Math.max(Math.min((cw - 48) / bw, (ch - 48) / bh), 0.3), 1.2)
    setView({
      k,
      x: (cw - bw * k) / 2 - minX * k,
      y: (ch - bh * k) / 2 - minY * k,
    })
  }, [])

  // 首次挂载时容器宽度才拿得到（flex 布局），延一帧再适应窗口
  useEffect(() => {
    const id = window.setTimeout(fit, 40)
    return () => window.clearTimeout(id)
  }, [fit])

  // 滚轮缩放：以指针位置为锚点。React 的 onWheel 是被动的，preventDefault
  // 会被忽略（页面跟着滚），所以走原生监听 + passive:false。
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = el.getBoundingClientRect()
      const mx = e.clientX - r.left
      const my = e.clientY - r.top
      setView((v) => {
        const k = Math.min(1.8, Math.max(0.3, v.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)))
        const ratio = k / v.k
        return { k, x: mx - (mx - v.x) * ratio, y: my - (my - v.y) * ratio }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  const zoomBy = (f: number) => {
    const el = wrapRef.current
    if (!el) return
    const cx = el.clientWidth / 2
    const cy = el.clientHeight / 2
    setView((v) => {
      const k = Math.min(1.8, Math.max(0.3, v.k * f))
      const ratio = k / v.k
      return { k, x: cx - (cx - v.x) * ratio, y: cy - (cy - v.y) * ratio }
    })
  }

  const beginDrag = (e: ReactPointerEvent, mode: 'pan' | 'node', key?: StepKey) => {
    if (e.button !== 0) return
    const v = viewRef.current
    const n = key ? nodesRef.current[key] : null
    dragRef.current = {
      mode,
      key,
      sx: e.clientX,
      sy: e.clientY,
      ox: mode === 'pan' ? v.x : n?.x ?? 0,
      oy: mode === 'pan' ? v.y : n?.y ?? 0,
    }
    setDragging(true)
    const onMove = (ev: PointerEvent) => {
      const d = dragRef.current
      if (!d) return
      const dx = ev.clientX - d.sx
      const dy = ev.clientY - d.sy
      if (d.mode === 'pan') {
        setView({ ...viewRef.current, x: d.ox + dx, y: d.oy + dy })
      } else if (d.key) {
        const k = viewRef.current.k
        const kk = d.key
        setNodes((prev) => ({
          ...prev,
          [kk]: { ...prev[kk], x: Math.round(d.ox + dx / k), y: Math.round(d.oy + dy / k) },
        }))
      }
    }
    const onUp = () => {
      dragRef.current = null
      setDragging(false)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const runStep = useCallback(
    async (key: StepKey) => {
      const p = paramsRef.current
      const cur = nodesRef.current
      // 单卡串行：并行两次生成会互相换出模型，比排队慢得多，还可能 OOM
      if (busyRef.current) return
      if (key !== 'front' && !cur.front.asset?.id) return
      if (key === 'strip' && !(cur.side.asset?.id && cur.back.asset?.id)) return
      if (!key || (key === 'front' && !p.prompt.trim())) return

      // 固定种子下重跑会得到同一张图（复现语义），空着则每次随机一版
      const fixed = p.seed.trim()
      const seed = fixed ? Number(fixed) : Math.floor(Math.random() * 2147483647)

      // 造型变了，下游成果就不再是同一个角色，直接作废
      const invalidate: StepKey[] =
        key === 'front' ? ['side', 'back', 'strip'] : key === 'strip' ? [] : ['strip']

      busyRef.current = true
      setNodes((prev) => {
        const next = { ...prev }
        invalidate.forEach((k) => {
          next[k] = {
            ...prev[k],
            status: 'idle',
            asset: undefined,
            seconds: undefined,
            error: undefined,
            prompt: undefined,
          }
        })
        next[key] = { ...prev[key], status: 'running', error: undefined, seed }
        return next
      })

      // 画布内部把最后一步叫 strip（节点键，也是 localStorage 的键，不轻易改），
      // 但接口的规范名是 compose。这里显式映射，别指望后端猜。
      const base = {
        view: key === 'strip' ? ('compose' as const) : key,
        prompt: p.prompt,
        style: p.styleId || undefined,
        width: p.width,
        height: p.height,
        steps: p.steps,
        cfg: p.cfg,
      }
      const body =
        key === 'strip'
          ? { ...base, viewAssetIds: [cur.front.asset!.id, cur.side.asset!.id, cur.back.asset!.id] }
          : key === 'front'
            ? { ...base, seed }
            : { ...base, seed, baseAssetId: cur.front.asset!.id }

      try {
        const data = await api.threeviewStep(body)
        const item: Asset = data.images[0]
        setNodes((prev) => ({
          ...prev,
          [key]: {
            ...prev[key],
            status: 'done',
            asset: item,
            seconds: data.meta?.seconds ?? prev[key].seconds,
            prompt: p.prompt,
            error: undefined,
          },
        }))
        onAsset(item)
      } catch (e: any) {
        setNodes((prev) => ({
          ...prev,
          [key]: { ...prev[key], status: 'error', error: e?.message || String(e) },
        }))
      } finally {
        busyRef.current = false
      }
    },
    [onAsset],
  )

  useImperativeHandle(ref, () => ({ start: () => void runStep('front') }), [runStep])

  useEffect(() => {
    onChange?.({ hasFront: !!nodes.front.asset, busy })
  }, [nodes.front.asset, busy, onChange])

  // 画布状态（出过的图 + 节点位置）落 localStorage：切去别的标签页再回来还在
  useEffect(() => {
    const id = window.setTimeout(() => {
      const slim: Record<string, unknown> = {}
      STEPS.forEach((s) => {
        const n = nodes[s.key]
        if (!n.asset) return
        slim[s.key] = {
          asset: n.asset,
          seconds: n.seconds,
          seed: n.seed,
          prompt: n.prompt,
          x: n.x,
          y: n.y,
        }
      })
      try {
        localStorage.setItem(KEY, JSON.stringify({ nodes: slim }))
      } catch {
        // 配额满就放弃，画布本身照常工作
      }
    }, 300)
    return () => window.clearTimeout(id)
  }, [nodes])

  const resetAll = () => {
    setNodes(idleNodes())
    try {
      localStorage.removeItem(KEY)
    } catch {
      // 忽略
    }
    fit()
  }

  const ready = (key: StepKey) =>
    key === 'front'
      ? true
      : key === 'strip'
        ? !!(nodes.side.asset && nodes.back.asset)
        : !!nodes.front.asset

  return (
    <div
      ref={wrapRef}
      className={`tv-wrap ${dragging ? 'on' : ''}`}
      onPointerDown={(e) => {
        // 点在空白画布上才平移；节点自己会 stopPropagation
        if (e.target === e.currentTarget || (e.target as HTMLElement).closest('.tv-bg')) {
          beginDrag(e, 'pan')
        }
      }}
    >
      <div className="tv-grid" />

      <div className="tv-toolbar">
        <button className="btn" onClick={() => zoomBy(1 / 1.15)} title="缩小">
          −
        </button>
        <span className="tv-zoom">{Math.round(view.k * 100)}%</span>
        <button className="btn" onClick={() => zoomBy(1.15)} title="放大">
          +
        </button>
        <button className="btn" onClick={fit}>
          适应窗口
        </button>
        <button className="btn btn-ghost" onClick={resetAll} title="清空画布上的四张图">
          清空
        </button>
      </div>

      <div className="tv-hint">
        拖空白处平移 · 滚轮缩放 · 拖节点标题移动 · 单步不满意的直接重跑
      </div>

      <div
        className="tv-world"
        style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})` }}
      >
        <svg className="tv-links" width={WORLD_W + 400} height={WORLD_H + 400}>
          {STEPS.slice(0, -1).map((s, i) => {
            const a = nodes[s.key]
            const b = nodes[STEPS[i + 1].key]
            const x1 = a.x + W
            const y1 = a.y + 150
            const x2 = b.x
            const y2 = b.y + 150
            const mx = (x1 + x2) / 2
            const live = a.status === 'done' && b.status !== 'idle'
            return (
              <path
                key={s.key}
                d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke={live ? '#00ff88' : 'rgba(255,255,255,0.22)'}
                strokeWidth={live ? 2 : 1.5}
                strokeDasharray={live ? undefined : '6 6'}
              />
            )
          })}
        </svg>

        {STEPS.map((s) => {
          const n = nodes[s.key]
          const ok = ready(s.key)
          // 角色描述改过但还没重跑：图还是旧的，得说出来
          const stale = s.key === 'front' && n.status === 'done' && n.prompt !== params.prompt
          const statusText =
            n.status === 'running'
              ? '生成中…'
              : n.status === 'error'
                ? '失败'
                : n.status === 'done'
                  ? `${n.seconds ?? '?'}s`
                  : '未生成'
          const actionText =
            s.key === 'front'
              ? n.asset
                ? '重生成'
                : '生成正视图'
              : s.key === 'strip'
                ? n.asset
                  ? '重新拼板'
                  : '拼板'
                : n.asset
                  ? '重生成'
                  : `生成${s.title}`

          return (
            <div
              key={s.key}
              className={`tv-node ${n.status} ${stale ? 'stale' : ''}`}
              style={{ left: n.x, top: n.y, width: W }}
            >
              <div
                className="tv-head"
                onPointerDown={(e) => {
                  e.stopPropagation()
                  beginDrag(e, 'node', s.key)
                }}
              >
                <span className="tv-idx">{s.no}</span>
                <span className="tv-title">{s.title}</span>
                <span className={`tv-status ${n.status}`}>{statusText}</span>
              </div>

              <div className="tv-body">
                {n.asset ? (
                  <img
                    src={n.asset.url}
                    alt={s.title}
                    draggable={false}
                    style={{ cursor: 'zoom-in' }}
                    onClick={() => openPreview(n.asset!.url, n.asset!.name)}
                  />
                ) : (
                  <div className="tv-ph">
                    {n.status === 'running' ? (
                      <>
                        <b>正在生成</b>
                        <br />
                        {s.key === 'front' ? '首次加载底座约 30s' : '编辑底座换入约 20-60s'}
                      </>
                    ) : n.status === 'error' ? (
                      <span style={{ color: 'var(--pink)' }}>这一步失败了</span>
                    ) : ok ? (
                      <>
                        <b>可以生成</b>
                        <br />
                        点下方按钮
                      </>
                    ) : (
                      <>
                        <b>等上一步</b>
                        <br />
                        {s.key === 'strip' ? '需要侧面和背面都出图' : '需要先有正视图'}
                      </>
                    )}
                  </div>
                )}
              </div>

              <div className="tv-foot">
                <div className="tv-meta">
                  {n.seed != null ? `seed ${n.seed}` : s.hint}
                  {n.asset && ` · ${n.asset.width}×${n.asset.height}`}
                </div>
                {stale && <div className="tv-warn">角色描述已改，重跑正视图后生效</div>}
                {n.error && <div className="tv-err">{n.error}</div>}
                <div className="tv-acts">
                  <button
                    className="btn"
                    disabled={busy || !ok || (s.key === 'front' && !params.prompt.trim())}
                    onClick={() => void runStep(s.key)}
                    onPointerDown={(e) => e.stopPropagation()}
                  >
                    {n.status === 'running' ? '生成中…' : actionText}
                  </button>
                  {n.asset && s.key === 'strip' && (
                    <button
                      className="btn btn-ghost"
                      onClick={() => onSendToPost(n.asset!)}
                      onPointerDown={(e) => e.stopPropagation()}
                    >
                      送后处理
                    </button>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
})

export default ThreeViewBoard
