import { useEffect, useMemo, useState } from 'react'
import { api, Asset } from '../api'
import PartPainter, { PartDef } from './PartPainter'
import { AUTO_PARTS, DEFAULT_PARTS, EXTRA_PARTS } from '../parts'

/** 识别结果：色块 id → 建议的部位名与角色 */
type Detected = { [k: string]: { name: string; role: string; conf: number; reason: string } }

export type RigPart = {
  id: string
  name: string
  assetId: string
  bone: { x: number; y: number; len: number; parent: string | null }
  pivot: [number, number]
  scale: number
  isBody?: boolean
  role?: string
}

type Props = {
  assets: Asset[]
  canvas: [number, number]
  /** 把生成的部件装进工程（轨道 + 骨骼） */
  onApply: (parts: RigPart[], groupId: string) => void
}

/**
 * 角色部件 → 轨道 + 骨骼，一气呵成。
 *
 * 为什么要从后处理页搬进编辑器：原来要在「后处理」涂色、再切到「动画」
 * 导入、再手动摆轨道。跨页来回一次，人就懒得用了——「在编辑器里对着角色图
 * 涂完直接生成轨道」才是顺手路径。
 *
 * 骨架方案（骨骼位置 / pivot / 父子 / 缩放）由服务端算（parts.plan_rig），
 * 这里只负责显示与写入，不自己实现一遍数学。
 */
export default function PartRigBuilder({ assets, canvas, onApply }: Props) {
  // 候选角色图：宽高都够大、且不是缩略图那种小图
  const candidates = useMemo(
    () => assets.filter((a) => a.width >= 64 && a.height >= 64)
      .sort((a, b) => b.createdAt - a.createdAt).slice(0, 40),
    [assets])
  const [srcId, setSrcId] = useState('')
  // 命名模式：auto = 只涂色块、由服务端按几何识别；manual = 自己起名
  const [nameMode, setNameMode] = useState<'auto' | 'manual'>('auto')
  const [parts, setParts] = useState<PartDef[]>(AUTO_PARTS)
  const [withTail, setWithTail] = useState(false)
  // 识别结果：色块 → 建议的部位名，用户可改
  const [detected, setDetected] = useState<Detected | undefined>(undefined)

  const switchMode = (m: 'auto' | 'manual') => {
    setNameMode(m)
    setDetected(undefined)
    if (m === 'auto') {
      setParts(AUTO_PARTS)
      setWithTail(false)
      setActiveId(AUTO_PARTS[0].id)
    } else {
      setParts(DEFAULT_PARTS)
      setActiveId(DEFAULT_PARTS[0].id)
    }
  }
  const [activeId, setActiveId] = useState(DEFAULT_PARTS[0].id)
  const [tag, setTag] = useState<Blob | null>(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')

  useEffect(() => {
    if (!srcId && candidates.length) setSrcId(candidates[0].id)
  }, [candidates, srcId])

  const src = candidates.find((a) => a.id === srcId)

  const run = async () => {
    if (!src) { setErr('先选一张角色图'); return }
    if (!tag) { setErr('先在图上涂出部位（至少涂一处）'); return }
    setBusy(true)
    setErr('')
    setMsg('')
    try {
      // auto 模式不传 name —— 服务端按几何识别（头/躯干/四肢/尾巴）
      const payload = nameMode === 'auto'
        ? parts.map((p) => ({ id: p.id, name: '', color: p.color }))
        : parts
      const res = await api.parts(null, tag, payload, 40, canvas, src.id)
      if (res.detail) throw new Error(res.detail)
      const rig: RigPart[] = (res.rig || []).map((r: any) => ({
        id: r.id,
        name: r.name,
        assetId: r.id,
        bone: r.bone,
        pivot: r.pivot,
        scale: r.scale,
        isBody: r.isBody,
      }))
      if (!rig.length) {
        const missed = (res.parts || []).filter((x: any) => x.empty).map((x: any) => x.name)
        throw new Error(missed.length
          ? `没涂到：${missed.join('、')}`
          : '没有生成任何部件')
      }
      onApply(rig, res.groupId)
      // 记下识别结果：色块 id → 建议部位名，界面上显示出来
      const det: Detected = {}
      ;(res.parts || []).forEach((x: any, i: number) => {
        if (x.empty) return
        const src = parts[i]
        if (src) {
          det[src.id] = {
            name: x.name, role: x.role || 'extra',
            conf: typeof x.conf === 'number' ? x.conf : 1,
            reason: x.reason || '',
          }
        }
      })
      setDetected(det)
      const missed = (res.parts || []).filter((x: any) => x.empty).map((x: any) => x.name)
      const recognized = Object.values(det).map((d) => d.name).join('、')
      const lowCount = Object.values(det).filter((d) => d.conf < 0.55).length
      setMsg(`已生成 ${rig.length} 个部件并装好骨架`
        + (nameMode === 'auto' && recognized ? `；识别为：${recognized}` : '')
        + (lowCount ? `；${lowCount} 处把握不足，已标出待复核` : '')
        + (missed.length ? `；没涂到：${missed.join('、')}` : ''))
    } catch (e: any) {
      setErr(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel-inset">
      <div className="label">角色图</div>
      {candidates.length === 0 ? (
        <div className="hint">素材库里还没有可用的角色图（至少 64×64）</div>
      ) : (
        <>
          <div className="rig-assets">
            {candidates.map((a) => (
              <img key={a.id} src={api.assetThumb(a.id, 96)} alt={a.name}
                   title={`${a.name} ${a.width}×${a.height}`}
                   className={a.id === srcId ? 'on' : ''}
                   onClick={() => setSrcId(a.id)} />
            ))}
          </div>
          {src && (
            <div className="hint">
              {src.name} · {src.width}×{src.height}
            </div>
          )}
        </>
      )}

      {src && (
        <>
          <div className="row" style={{ gap: 8, marginTop: 8 }}>
            <div className="label grow">涂部位</div>
            <div className="seg">
              <button className={nameMode === 'auto' ? 'on' : ''}
                      onClick={() => switchMode('auto')}>自动识别</button>
              <button className={nameMode === 'manual' ? 'on' : ''}
                      onClick={() => switchMode('manual')}>自己起名</button>
            </div>
            {nameMode === 'manual' && (
              <label className="hint" style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={withTail}
                  onChange={(e) => {
                    setWithTail(e.target.checked)
                    setParts(e.target.checked
                      ? [...DEFAULT_PARTS, ...EXTRA_PARTS]
                      : DEFAULT_PARTS)
                    if (!e.target.checked) setActiveId(DEFAULT_PARTS[0].id)
                  }}
                />
                尾巴分节
              </label>
            )}
          </div>
          <div className="hint" style={{ marginTop: 2 }}>
            {nameMode === 'auto'
              ? '只管涂色块，涂完机器按几何判断这是头、那是尾巴（识别结果在下面，可改名）。'
              : '自己给每个部位起名。带数字的（尾巴1/2/3）会串成一条链。'}
          </div>
          <PartPainter
            src={api.assetThumb(src.id, 320)}
            parts={parts}
            activeId={activeId}
            onActive={setActiveId}
            onTag={setTag}
          />
          <div className="row" style={{ gap: 6, marginTop: 6 }}>
            <button className="btn btn-primary grow" onClick={run} disabled={busy}
                    style={{ height: 40 }}>
              {busy ? '生成中…' : '生成部件并装骨架'}
            </button>
          </div>
          {detected && Object.keys(detected).length > 0 && (
            <>
              <div className="label" style={{ marginTop: 8 }}>识别结果</div>
              <div className="chips">
                {Object.entries(detected).map(([pid, d]) => {
                  const low = d.conf < 0.55
                  return (
                    <span key={pid} className={`chip sm ${low ? 'low' : ''}`}
                          title={`${d.reason}（把握 ${Math.round(d.conf * 100)}%）`}>
                      <i className="dot" style={{
                        background: parts.find((x) => x.id === pid)?.color,
                      }} />
                      {d.name}
                      {low && ' ?'}
                    </span>
                  )
                })}
              </div>
              {(() => {
                const low = Object.entries(detected).filter(([, d]) => d.conf < 0.55)
                if (!low.length) {
                  return (
                    <div className="hint">
                      都比较有把握。认错了就点左栏的骨骼改名——
                      名字会决定旋转点（头绕脖子、臂绕肩）。
                    </div>
                  )
                }
                return (
                  <div className="hint" style={{ color: '#ffc14d' }}>
                    带 ? 的 {low.length} 处我不确定
                    （{low.map(([, d]) => d.name).join('、')}）——{low[0][1].reason}。
                    请人工核对一下，改名或调父骨骼都能修。
                  </div>
                )
              })()}
            </>
          )}
          {err && <div className="hint" style={{ color: '#ff6b6b' }}>{err}</div>}
          {msg && <div className="hint" style={{ color: 'var(--green)' }}>{msg}</div>}
          <div className="hint">
            部件会按原图位置整体缩放居中放进画布。
            {nameMode === 'auto'
              ? '旋转点由几何决定：头绕脖子、四肢绕肩胯、尾巴绕靠身体那端。'
              : '头与躯干的旋转点在下方（绕脖子/胯），四肢在上方（绕肩/胯）。'}
            {nameMode === 'manual' && (
              <>
                <br />
                勾了「尾巴分节」时，尾巴1/2/3 会串成一条链：
                转第一节，后面几节跟着摆。
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}
