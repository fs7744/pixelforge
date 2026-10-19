import { useEffect, useMemo, useRef, useState } from 'react'
import { api, Asset, AssetFacets, StyleInfo } from '../api'
import { RefPicker } from '../RefPicker'
import { openPreview } from '../Preview'
import { Splitter } from '../split'

type Props = {
  /** 从素材库点「转画风」进来时带的那张图 */
  incoming?: Asset | null
  onSendToPost?: (a: Asset) => void
}

/** 当前输入：本地文件走 objectURL，资产库的图直接用它的 url + id（不必下载再上传） */
type Src = { url: string; name: string; assetId?: string }
type Run = { asset: Asset; styleId: string; styleName: string }

const STEPS_DEFAULT = 40
const MAXSIDE = [768, 1024, 1280]

/**
 * 画风转换工作室：一张图 + 目标画风 → 编辑底座重绘。
 *
 * 和后处理里的「AI 画风重统一」是同一个后端端点，区别在这个页面把这件事
 * 做成了正经工作流：画风铺成带示例图的画廊（15 个画风平铺成 chips 根本
 * 分不清长什么样）、前后对比滑块、会话内多次转换留档、结果可以接着再转。
 * 转换单次 2-4 分钟，不给一个能反复对比的落点，用户就只能靠记忆判断好坏。
 */
export default function Restyle({ incoming, onSendToPost }: Props) {
  const [src, setSrc] = useState<Src | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [styles, setStyles] = useState<StyleInfo[]>([])
  const [styleId, setStyleId] = useState('')
  const [out, setOut] = useState<Asset | null>(null)
  const [history, setHistory] = useState<Run[]>([])
  const [steps, setSteps] = useState(STEPS_DEFAULT)
  const [seed, setSeed] = useState('')
  const [maxSide, setMaxSide] = useState(1024)
  const [pxOn, setPxOn] = useState(true)
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [split, setSplit] = useState(50)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [over, setOver] = useState(false)

  // 资产库取图
  const [libOpen, setLibOpen] = useState(false)
  const [library, setLibrary] = useState<Asset[]>([])
  const [libFacets, setLibFacets] = useState<AssetFacets | null>(null)
  const [libPick, setLibPick] = useState<string[]>([])

  const inputRef = useRef<HTMLInputElement>(null)
  const objectUrl = useRef('')
  const boxRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  useEffect(() => {
    api.styles()
      .then((r) => {
        setStyles(r.styles || [])
        setStyleId((cur) => cur || r.styles?.[0]?.id || '')
      })
      .catch(() => setStyles([]))
  }, [])

  // 本地图的 objectURL 自己回收，否则来回换几十张图会一直占着内存
  useEffect(() => () => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
  }, [])

  const clearOut = () => {
    setOut(null)
    setSplit(50)
    setError('')
    setNote('')
  }

  const acceptLocal = (f: File) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    const url = URL.createObjectURL(f)
    objectUrl.current = url
    setFile(f)
    setSrc({ url, name: f.name })
    clearOut()
  }

  const acceptAsset = (a: Asset) => {
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = ''
    }
    setFile(null)
    setSrc({ url: a.url, name: a.name, assetId: a.id })
    clearOut()
  }

  // 从素材库「转画风」跳进来：直接拿那张资产当输入
  useEffect(() => {
    if (incoming) acceptAsset(incoming)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incoming])

  const openLib = () => {
    const next = !libOpen
    setLibOpen(next)
    if (!next) return
    // 每次都重拉：刚生成的图不用刷新页面就能选到
    api.assets().then((d) => setLibrary(d.assets)).catch(() => setLibrary([]))
    api.assetFacets().then(setLibFacets).catch(() => setLibFacets(null))
  }

  const pickFromLibrary = (id: string) => {
    const a = library.find((x) => x.id === id)
    if (!a) return
    setLibPick([id])
    acceptAsset(a)
    setLibOpen(false)
    setLibPick([])
  }

  // 画廊按画风分组（后端给的 group），顺序沿用接口顺序
  const groups = useMemo(() => {
    const out: { name: string; items: StyleInfo[] }[] = []
    for (const s of styles) {
      const name = s.group || '其他'
      let g = out.find((x) => x.name === name)
      if (!g) {
        g = { name, items: [] }
        out.push(g)
      }
      g.items.push(s)
    }
    return out
  }, [styles])

  const cur = useMemo(() => styles.find((s) => s.id === styleId) || null, [styles, styleId])

  useEffect(() => {
    if (!busy) return
    setElapsed(0)
    const t = window.setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => window.clearInterval(t)
  }, [busy])

  // 转换中的每一步都很慢，先把按钮状态写在界面上，用户才知道没卡死
  const canRun = !!src && !!styleId && !busy

  const apply = async () => {
    if (!src || !styleId) return
    const seedNum = seed.trim() === '' ? null : Number(seed)
    if (seedNum != null && !Number.isFinite(seedNum)) {
      setError('种子要填整数，留空则每次不同')
      return
    }
    setBusy(true)
    clearOut()
    const t0 = performance.now()
    try {
      const res = await api.restyle(
        {
          style: styleId,
          assetId: src.assetId,
          steps,
          seed: seedNum,
          maxSide,
          pixelize: pxOn,
        },
        // 资产库的图交给后端按 id 读，不做 blob 往返
        src.assetId ? null : file,
      )
      if (!res.asset) throw new Error(res.meta?.reason || '转换失败')
      const item: Asset = res.asset
      setOut(item)
      setSplit(50)
      const px = res.meta?.pixelize
      const secs = Math.round((performance.now() - t0) / 1000)
      const pxText = px
        ? ` · 已像素化 ${px.grid?.[0]}×${px.grid?.[1]} 格 / ${px.colors} 色`
        : res.meta?.pixelizeSkipped
          ? ' · 按设置跳过像素化'
          : ''
      setNote(`${res.meta?.styleName || cur?.name} · ${secs}s · seed ${res.meta?.seed ?? '随机'}${pxText}`)
      setHistory((h) => [
        { asset: item, styleId, styleName: res.meta?.styleName || cur?.name || styleId },
        ...h.filter((x) => x.asset.id !== item.id),
      ].slice(0, 8))
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  // 以前后对比滑块：拖到哪就露出多少「转换后」
  const at = (clientX: number) => {
    const el = boxRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setSplit(Math.max(0, Math.min(100, ((clientX - r.left) / Math.max(1, r.width)) * 100)))
  }

  return (
    <div className="body">
      <aside className="panel">
        <div className="row between" style={{ alignItems: 'center' }}>
          <div className="section-title">
            图片输入 <span className="tag">INPUT</span>
          </div>
          <button
            className="btn btn-ghost"
            style={{ padding: '2px 8px', fontSize: 13 }}
            onClick={openLib}
          >
            {libOpen ? '收起资产库' : '从资产库选'}
          </button>
        </div>

        <div
          className={`drop ${over ? 'over' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            const f = e.dataTransfer.files?.[0]
            if (f) acceptLocal(f)
          }}
        >
          {src ? (
            <div>
              <div className="mono">{src.name}</div>
              <div className="hint">{src.assetId ? '素材库资产 · 点击替换' : '本地文件 · 点击替换'}</div>
            </div>
          ) : (
            <div>
              拖入或点击上传 PNG
              <div className="hint" style={{ marginTop: 6 }}>
                生成结果 / 旧素材 / 外部参考图都行
              </div>
            </div>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) acceptLocal(f)
            e.currentTarget.value = ''
          }}
        />

        {libOpen && (
          <RefPicker
            library={library}
            facets={libFacets}
            selected={libPick}
            usedSlots={0}
            maxRefs={1}
            single
            onToggle={pickFromLibrary}
          />
        )}

        <div>
          <div className="label">目标画风（{styles.length} 种）</div>
          {styles.length === 0 ? (
            <div className="hint" style={{ marginTop: 6 }}>画风库加载中…</div>
          ) : (
            <div className="style-gallery">
              {groups.map((g) => (
                <div key={g.name} className="style-group">
                  <div className="style-group-name">{g.name}</div>
                  <div className="style-cards">
                    {g.items.map((s) => (
                      <button
                        key={s.id}
                        type="button"
                        className={`style-card ${styleId === s.id ? 'on' : ''}`}
                        title={`${s.name}：${s.hint}`}
                        onClick={() => setStyleId(s.id)}
                      >
                        <span className="style-card-img checker">
                          {s.sample ? (
                            <img
                              src={s.sample}
                              alt={`${s.name}示例`}
                              loading="lazy"
                              style={{ imageRendering: s.pixelize ? 'pixelated' : 'auto' }}
                            />
                          ) : (
                            <span className="style-card-ph">{s.name}</span>
                          )}
                        </span>
                        <span className="style-card-name">{s.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <button className="btn btn-primary btn-block" disabled={!canRun} onClick={apply}>
          {busy ? `转换中 ${elapsed}s（约 2-4 分钟）` : out ? '再转一次' : '开始转换'}
        </button>
        <div className="hint">全部处理在本机完成 · 结果自动存入素材库</div>
        {note && <div className="ok">{note}</div>}
        {error && <div className="err">{error}</div>}
      </aside>

      <Splitter side="left" />

      <main className="stage">
        <div className="row between">
          <div className="section-title">
            画风转换 · {cur ? cur.name : '未选画风'} <span className="tag pink">AI-RESTYLE</span>
          </div>
          <div className="row">
            {out && (
              <>
                <button className="btn btn-ghost" onClick={() => openPreview(out.url, out.name)}>
                  放大
                </button>
                <a className="btn" href={`/api/assets/${out.id}/download`} download={`${out.name}.png`}>
                  下载 PNG
                </a>
                <button className="btn" onClick={() => api.exportZip([out.id])}>
                  导出 ZIP
                </button>
                {onSendToPost && (
                  <button className="btn" onClick={() => onSendToPost(out)}>
                    送后处理
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {out && src ? (
          <>
            <div
              className="cmp"
              ref={boxRef}
              onPointerDown={(e) => {
                dragging.current = true
                e.currentTarget.setPointerCapture(e.pointerId)
                at(e.clientX)
              }}
              onPointerMove={(e) => {
                if (dragging.current) at(e.clientX)
              }}
              onPointerUp={() => {
                dragging.current = false
              }}
              onPointerCancel={() => {
                dragging.current = false
              }}
            >
              <img className="cmp-img" src={src.url} alt="转换前" draggable={false} />
              <div className="cmp-after" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
                <img className="cmp-img" src={out.url} alt="转换后" draggable={false} />
              </div>
              <div className="cmp-line" style={{ left: `${split}%` }}>
                <i />
              </div>
              <span className="cmp-tag left">BEFORE</span>
              <span className="cmp-tag right">{out.meta?.styleName || 'AFTER'}</span>
            </div>
            <div className="row between">
              <span className="hint">
                拖动中间竖线对比前后 · 尺寸不变（{out.width}×{out.height}），主体与构图沿用原图
              </span>
              <div className="row">
                <button
                  className="btn btn-ghost"
                  onClick={() => {
                    acceptAsset(out)
                    setNote('已把转换结果设为新输入，可以继续叠加另一种画风（每多转一次细节会略有漂移）')
                  }}
                >
                  以此结果继续转换
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="cmp-wrap">
            {src ? (
              <div className="thumb checker" style={{ height: 420, borderColor: 'rgba(255,255,255,0.35)' }}>
                <img src={src.url} alt="待转换" style={{ cursor: 'zoom-in' }} onClick={() => openPreview(src.url, src.name)} />
              </div>
            ) : (
              <div className="empty" style={{ minHeight: 420 }}>
                <span className="pixel">NO INPUT</span>
                左侧上传一张图，或从资产库选一张旧素材。
                <div className="hint mt8">画风转换只改笔触、线条与上色，主体和构图沿用原图。</div>
              </div>
            )}
            {busy && (
              <div className="warn">
                AI 正在重绘第 {elapsed}s：40 步单次约 2-4 分钟。期间如驻留着 Z-Image 生成模型，
                会先换出显存再加载编辑底座，首次会多花几十秒。
              </div>
            )}
          </div>
        )}

        {history.length > 0 && (
          <div className="card">
            <div className="label">本次会话的转换结果（点一下回到对比视图）</div>
            <div className="hist-strip">
              {history.map((h) => (
                <button
                  key={h.asset.id}
                  className={`hist-item ${out?.id === h.asset.id ? 'on' : ''}`}
                  onClick={() => {
                    setOut(h.asset)
                    setSplit(50)
                  }}
                  title={`${h.styleName} · ${h.asset.width}×${h.asset.height}`}
                >
                  <span className="hist-img checker">
                    <img src={api.assetThumb(h.asset.id, 160)} alt={h.styleName} loading="lazy" />
                  </span>
                  <span className="hist-name">{h.styleName}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      <Splitter side="right" />

      <aside className="panel right">
        <div className="section-title">
          转换参数 <span className="tag pink">PIPELINE</span>
        </div>

        {cur ? (
          <div className="card">
            <div className="label">目标画风</div>
            <div style={{ fontSize: 15, fontWeight: 600 }}>{cur.name}</div>
            <div className="chips">
              {cur.tags.map((t) => (
                <span key={t} className="chip sm">
                  {t}
                </span>
              ))}
            </div>
            <div className="hint">{cur.hint}</div>
            {cur.pixelize && (
              <span className="tag" style={{ alignSelf: 'flex-start' }}>
                出图后建议像素化 {cur.pixelize.block}× · {cur.pixelize.colors} 色
              </span>
            )}
          </div>
        ) : (
          <div className="card">
            <div className="hint">左侧选一个目标画风。</div>
          </div>
        )}

        <div className="card">
          <div className="row">
            <div className="grow">
              <div className="label">步数</div>
              <input
                className="input"
                type="number"
                min={10}
                max={60}
                value={steps}
                onChange={(e) => setSteps(Math.max(10, Math.min(60, +e.target.value || STEPS_DEFAULT)))}
              />
            </div>
            <div className="grow">
              <div className="label">长边</div>
              <select className="select" value={maxSide} onChange={(e) => setMaxSide(+e.target.value)}>
                {MAXSIDE.map((n) => (
                  <option key={n} value={n}>
                    {n}px
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div>
            <div className="label">随机种子（可留空）</div>
            <input
              className="input"
              type="number"
              placeholder="留空则每次不同"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
          </div>
          <div className="hint">
            编辑前把长边压到 {maxSide}px 再推理，出图后放回原尺寸——尺寸越大越慢，
            细节收益在 1024 之后已经很小。同种子 + 同参数 + 同原图可复现同一张。
          </div>
        </div>

        {cur?.pixelize && (
          <div className="card">
            <div className="row between">
              <div className="label">AI 重绘后自动像素化</div>
              <div className={`toggle ${pxOn ? 'on' : 'off'}`} onClick={() => setPxOn(!pxOn)}>
                <span />
              </div>
            </div>
            <div className="hint">
              编辑底座出不了硬边像素，重绘后按画风建议的 {cur.pixelize.block}× 分块 /
              {' '}{cur.pixelize.colors} 色量化收尾。关掉可以先看 AI 的原始笔触，再自己去后处理调块大小。
            </div>
          </div>
        )}

        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 10, marginBottom: 4 }}>HOW IT WORKS</div>
          <div style={{ fontSize: 13, lineHeight: 1.7 }}>
            · 指令式重绘：Qwen-Image-Edit-2511，默认 40 步
            <br />· 指令以「重绘成什么画风」开头，主体与构图用「沿用原图」锚定
            <br />· 不写「保持完全不变」——那会把重绘压成原样输出
            <br />· 像素类画风：AI 重绘 → 确定性像素化收尾
            <br />· 结果按 kind=restyled / pipeline=restyle 入库
          </div>
        </div>

        {out?.meta?.instruction && (
          <div className="card">
            <div className="label">本次实际指令（可复现）</div>
            <pre
              className="mono"
              style={{ margin: 0, fontSize: 14, lineHeight: 1.7, color: 'var(--muted)', whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto' }}
            >
              {out.meta.instruction}
            </pre>
          </div>
        )}
      </aside>
    </div>
  )
}
