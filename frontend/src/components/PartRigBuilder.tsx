import { useEffect, useMemo, useState } from 'react'
import { api, Asset } from '../api'
import PartPainter, { PartDef } from './PartPainter'
import { DEFAULT_PARTS, EXTRA_PARTS } from '../parts'

export type RigPart = {
  id: string
  name: string
  assetId: string
  bone: { x: number; y: number; len: number; parent: string | null }
  pivot: [number, number]
  scale: number
  isBody?: boolean
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
  const [parts, setParts] = useState<PartDef[]>(DEFAULT_PARTS)
  const [withTail, setWithTail] = useState(false)
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
      const res = await api.parts(null, tag, parts, 40, canvas, src.id)
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
      const missed = (res.parts || []).filter((x: any) => x.empty).map((x: any) => x.name)
      setMsg(`已生成 ${rig.length} 个部件并装好骨架`
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
          {err && <div className="hint" style={{ color: '#ff6b6b' }}>{err}</div>}
          {msg && <div className="hint" style={{ color: 'var(--green)' }}>{msg}</div>}
          <div className="hint">
            头与躯干的旋转点在下方（绕脖子/胯），四肢在上方（绕肩/胯）。
            部件会按原图位置整体缩放居中放进画布。
            <br />
            勾了「尾巴分节」时，尾巴1/2/3 会串成一条链：
            转第一节，后面几节跟着摆。
          </div>
        </>
      )}
    </div>
  )
}
