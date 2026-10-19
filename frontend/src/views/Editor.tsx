/**
 * 动画编辑器：骨骼 / 多轨 / 自由变形 / 特效笔刷 / 洋葱皮。
 *
 * 为什么预览走「后端渲染单帧 + <img>」而不是前端 canvas 合成：
 * 合成规则（骨骼级联、变形、混合模式、笔刷）必须和导出**完全一致**，
 * 前端另写一套必然出现「预览挺好、导出不一样」。代价是每次改动要请求一次，
 * 但一帧是十几毫秒的 PIL 操作，本地回环完全跟得上；给拖动过程加个节流即可。
 *
 * 交互约定：坐标换算只在一处（toCanvas），所有拖拽都换算成画布坐标再写回，
 * 避免「缩放 50% 时拖动变慢一半」这类只在特定缩放下出现的怪问题。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, Asset } from '../api'

type Mode = 'select' | 'bone' | 'warp' | 'brush'
type Pt = [number, number]

type Aff = { a: number; b: number; c: number; d: number; e: number; f: number }

const I: Aff = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }
const T = (x: number, y: number): Aff => ({ a: 1, b: 0, c: 0, d: 1, e: x, f: y })
const R = (deg: number): Aff => {
  const r = (deg * Math.PI) / 180
  return { a: Math.cos(r), b: Math.sin(r), c: -Math.sin(r), d: Math.cos(r), e: 0, f: 0 }
}
const S = (k: number): Aff => ({ a: k, b: 0, c: 0, d: k, e: 0, f: 0 })
const mul = (m: Aff, n: Aff): Aff => ({
  a: m.a * n.a + m.c * n.b,
  b: m.b * n.a + m.d * n.b,
  c: m.a * n.c + m.c * n.d,
  d: m.b * n.c + m.d * n.d,
  e: m.a * n.e + m.c * n.f + m.e,
  f: m.b * n.e + m.d * n.f + m.f,
})
const apply = (m: Aff, x: number, y: number): Pt => [m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f]
const chain = (...ms: Aff[]): Aff => ms.reduce(mul, I)

const ease = (t: number, kind: string) => {
  const c = Math.max(0, Math.min(1, t))
  if (kind === 'ease-in') return c * c
  if (kind === 'ease-out') return 1 - (1 - c) * (1 - c)
  return c
}

function sample(keys: any[], frame: number) {
  const base = { x: 0, y: 0, rot: 0, scale: 1, opacity: 1 }
  if (!keys?.length) return base
  const ks = [...keys].sort((p, q) => p.frame - q.frame)
  if (frame <= ks[0].frame) return { ...base, ...ks[0] }
  if (frame >= ks[ks.length - 1].frame) return { ...base, ...ks[ks.length - 1] }
  for (let i = 0; i < ks.length - 1; i++) {
    const a = ks[i], b = ks[i + 1]
    if (frame >= a.frame && frame <= b.frame) {
      const span = b.frame - a.frame || 1
      const t = ease((frame - a.frame) / span, b.ease || 'linear')
      return {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        rot: a.rot + (b.rot - a.rot) * t,
        scale: a.scale + (b.scale - a.scale) * t,
        opacity: a.opacity + (b.opacity - a.opacity) * t,
      }
    }
  }
  return base
}

function boneWorld(proj: any, frame: number): Record<string, Aff> {
  const bones: any[] = proj.bones || []
  const out: Record<string, Aff> = {}
  for (const b of bones) {
    const k = sample(b.keys || [], frame)
    const local = chain(T(k.x, k.y), R(k.rot), S(k.scale))
    const parent = b.parent ? bones.find((x) => x.id === b.parent) : null
    out[b.id] = parent && out[parent.id]
      ? chain(out[parent.id], T(b.x - parent.x, b.y - parent.y), local)
      : chain(T(b.x, b.y), local)
  }
  return out
}

function trackMatrix(proj: any, t: any, frame: number): Aff {
  const w = boneWorld(proj, frame)[t.bone || 'root'] || I
  const k = sample(t.keyframes || [], frame)
  const [px, py] = t.pivot || [0, 0]
  return chain(w, T(k.x, k.y), R(k.rot), S(k.scale), T(-px, -py))
}

export default function Editor() {
  const [projects, setProjects] = useState<any[]>([])
  const [pid, setPid] = useState('')
  const [proj, setProj] = useState<any>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [onion, setOnion] = useState(true)
  const [mode, setMode] = useState<Mode>('select')
  const [selTrack, setSelTrack] = useState('')
  const [selBone, setSelBone] = useState('root')
  const [bump, setBump] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [brush, setBrush] = useState({ brush: 'slash', color: '#ffcc33', size: 14 })
  const [assets, setAssets] = useState<Asset[]>([])
  const [picker, setPicker] = useState(false)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const [newName, setNewName] = useState('')
  const [size, setSize] = useState<[number, number]>([512, 512])

  const stageRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<any>(null)
  const dirtyRef = useRef(false)
  const saveTimer = useRef<number | null>(null)

  const track = useMemo(
    () => (proj?.tracks || []).find((t: any) => t.id === selTrack) || null,
    [proj, selTrack],
  )
  const assetOf = useCallback(
    (id: string) => assets.find((a) => a.id === id),
    [assets],
  )

  // ---------- 载入 ----------
  const loadList = useCallback(async (pick?: string) => {
    const r = await api.animList()
    setProjects(r.projects || [])
    const next = pick || pid || r.projects?.[0]?.id || ''
    if (next) await openProject(next)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pid])

  const openProject = async (id: string) => {
    setPid(id)
    setPlaying(false)
    const p = await api.animGet(id)
    setProj(p)
    setSize([p.canvas[0], p.canvas[1]])
    setFrame(0)
    setSelTrack(p.tracks?.[0]?.id || '')
    setSelBone(p.bones?.[0]?.id || 'root')
    setBump((b) => b + 1)
    dirtyRef.current = false
  }

  useEffect(() => {
    loadList()
    api.assets()
      .then((list: any) => setAssets(Array.isArray(list) ? list : (list.assets || list.items || [])))
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 自动保存：改完 400ms 落盘。拖动时每帧都发请求的话，键盘/鼠标事件会被
  // 网络往返拖慢，手感立刻变粘。
  useEffect(() => {
    if (!proj || !pid || !dirtyRef.current) return
    if (saveTimer.current) window.clearTimeout(saveTimer.current)
    saveTimer.current = window.setTimeout(async () => {
      try {
        await api.animSave(pid, proj)
        setBump((b) => b + 1)
      } catch (e: any) {
        setError(e.message || '保存失败')
      }
    }, 400)
  }, [proj, pid])

  const update = (fn: (p: any) => void) => {
    setProj((prev: any) => {
      if (!prev) return prev
      const next = JSON.parse(JSON.stringify(prev))
      fn(next)
      dirtyRef.current = true
      return next
    })
  }

  // ---------- 播放 ----------
  useEffect(() => {
    if (!playing || !proj) return
    const ms = 1000 / Math.max(1, proj.fps || 12)
    const id = window.setInterval(() => {
      setFrame((f) => (f + 1) % proj.frames)
    }, ms)
    return () => window.clearInterval(id)
  }, [playing, proj])

  // ---------- 坐标 ----------
  const toCanvas = (e: { clientX: number; clientY: number }): Pt => {
    const r = stageRef.current!.getBoundingClientRect()
    return [(e.clientX - r.left) / zoom, (e.clientY - r.top) / zoom]
  }

  // ---------- 关键帧编辑 ----------
  const upsertKey = (target: 'track' | 'bone', patch: Record<string, number>) => {
    update((p) => {
      const obj = target === 'track'
        ? p.tracks.find((t: any) => t.id === selTrack)
        : p.bones.find((b: any) => b.id === selBone)
      if (!obj) return
      const key = target === 'track' ? 'keyframes' : 'keys'
      if (!obj[key]) obj[key] = []
      let k = obj[key].find((x: any) => x.frame === frame)
      if (!k) {
        const cur = sample(obj[key], frame)
        k = { frame, x: cur.x, y: cur.y, rot: cur.rot, scale: cur.scale, opacity: 1, ease: 'linear' }
        obj[key].push(k)
      }
      Object.assign(k, patch)
      obj[key].sort((a: any, b: any) => a.frame - b.frame)
    })
  }

  const deleteKey = () => {
    update((p) => {
      const obj = p.tracks.find((t: any) => t.id === selTrack)
      if (obj) obj.keyframes = (obj.keyframes || []).filter((k: any) => k.frame !== frame)
    })
  }

  // ---------- 拖拽 ----------
  const onPointerDown = (e: React.PointerEvent) => {
    if (!proj) return
    const [x, y] = toCanvas(e)
    ;(e.target as Element).setPointerCapture?.(e.pointerId)

    if (mode === 'brush' && track) {
      dragRef.current = { kind: 'brush', points: [[x, y, 1]] as number[][] }
      return
    }
    if (mode === 'warp' && track) {
      const m = trackMatrix(proj, track, frame)
      const g = warpPoints(track, assetOf(track.assetId))
      let best = -1
      let bestD = 1e9
      g.forEach((p, i) => {
        const [gx, gy] = apply(m, p[0], p[1])
        const d = Math.hypot(gx - x, gy - y)
        if (d < bestD) { bestD = d; best = i }
      })
      if (bestD < 18 / zoom) {
        dragRef.current = { kind: 'warp', index: best, start: [x, y],
          base: (track.warp || Array(9).fill([0, 0])).map((p: number[]) => [...p]) }
      }
      return
    }
    if (mode === 'bone') {
      const w = boneWorld(proj, frame)[selBone]
      if (w) {
        const [bx, by] = apply(w, 0, 0)
        if (Math.hypot(bx - x, by - y) < 26 / zoom) {
          dragRef.current = { kind: 'bone', start: [x, y] }
        }
      }
      return
    }
    if (mode === 'select' && track) {
      dragRef.current = { kind: 'track', start: [x, y] }
    }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current
    if (!d || !proj) return
    const [x, y] = toCanvas(e)
    if (d.kind === 'brush') {
      d.points.push([x, y, 1])
      return
    }
    if (d.kind === 'bone' || d.kind === 'track') {
      // 位移按「相对上一次」累加：改成绝对位移的话，手一抖就跳回原点
      const dx = x - d.start[0]
      const dy = y - d.start[1]
      d.start = [x, y]
      if (d.kind === 'bone') {
        update((p) => {
          const b = p.bones.find((v: any) => v.id === selBone)
          if (!b) return
          b.keys = b.keys || []
          let k = b.keys.find((v: any) => v.frame === frame)
          if (!k) {
            const cur = sample(b.keys, frame)
            k = { frame, x: cur.x, y: cur.y, rot: cur.rot, scale: 1, ease: 'linear' }
            b.keys.push(k)
          }
          k.x += dx
          k.y += dy
        })
      } else {
        update((p) => {
          const t = p.tracks.find((v: any) => v.id === selTrack)
          if (!t) return
          t.keyframes = t.keyframes || []
          let k = t.keyframes.find((v: any) => v.frame === frame)
          if (!k) {
            const cur = sample(t.keyframes, frame)
            k = { frame, x: cur.x, y: cur.y, rot: cur.rot, scale: 1, opacity: 1, ease: 'linear' }
            t.keyframes.push(k)
          }
          k.x += dx
          k.y += dy
        })
      }
      return
    }
    if (d.kind === 'warp') {
      const m = trackMatrix(proj, track, frame)
      // 屏幕位移 → 源图像素位移：用矩阵线性部分的逆
      const det = m.a * m.d - m.b * m.c || 1
      const sx = ((x - d.start[0]) * m.d - (y - d.start[1]) * m.c) / det
      const sy = ((y - d.start[1]) * m.a - (x - d.start[0]) * m.b) / det
      d.start = [x, y]
      update((p) => {
        const t = p.tracks.find((v: any) => v.id === selTrack)
        if (!t) return
        t.warp = t.warp || Array(9).fill([0, 0])
        const cur = t.warp[d.index] || [0, 0]
        t.warp[d.index] = [cur[0] + sx, cur[1] + sy]
      })
    }
  }

  const onPointerUp = () => {
    const d = dragRef.current
    dragRef.current = null
    if (!d || d.kind !== 'brush' || !track) return
    const pts: number[][] = d.points
    if (pts.length < 2) return
    update((p) => {
      const t = p.tracks.find((v: any) => v.id === selTrack)
      if (!t) return
      t.fx = t.fx || []
      t.fx.push({ ...brush, points: pts, from: 0, to: p.frames - 1 })
    })
    setNote(`已添加 ${brush.brush} 笔迹（${pts.length} 个采样点）`)
  }

  // ---------- 操作 ----------
  const addTrack = (a: Asset) => {
    // id 先算好：写在 setState 更新函数里的话，React 严格模式重放会生成两个 id
    const tid = Math.random().toString(16).slice(2, 10)
    update((p) => {
      p.tracks = p.tracks || []
      p.tracks.push({
        id: tid,
        name: a.name || '轨道',
        assetId: a.id,
        bone: selBone || 'root',
        pivot: [Math.round(a.width / 2), Math.round(a.height / 2)],
        warp: Array(9).fill([0, 0]),
        keyframes: [{ frame: 0, x: 0, y: 0, rot: 0, scale: 1, opacity: 1, ease: 'linear' }],
        fx: [], opacity: 1, blend: 'normal', visible: true,
      })
    })
    setSelTrack(tid)
    setPicker(false)
    setNote(`已添加轨道：${a.name}`)
  }

  const createProject = async () => {
    const p = await api.animCreate({ name: newName || '未命名工程', canvas: size, frames: 12 })
    setNewName('')
    await loadList(p.id)
  }

  const removeProject = async () => {
    if (!pid || !confirm('删除这个工程？此操作不可撤销。')) return
    await api.animDelete(pid)
    setProj(null)
    setPid('')
    await loadList()
  }

  const doExport = async () => {
    if (!pid || !proj) return
    setBusy(true)
    setError('')
    try {
      await api.animRender(pid, {
        frames: '', scale: 1, cols: 4, normals: true,
        name: proj.name || 'anim',
      })
      setNote('已导出 ZIP：frames/ + sheet.png + normals/ + project.json')
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  const warpPoints = (t: any, a?: Asset): Pt[] => {
    const w = a?.width || 64
    const h = a?.height || 64
    const pts: Pt[] = []
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const base: Pt = [(c * w) / 2, (r * h) / 2]
        const off = t.warp?.[r * 3 + c] || [0, 0]
        pts.push([base[0] + off[0], base[1] + off[1]])
      }
    }
    return pts
  }

  const dims: [number, number] = (proj?.canvas as [number, number]) || size
  const bones = proj?.bones || []
  const worlds = proj ? boneWorld(proj, frame) : {}

  return (
    <div className="editor">
      <aside className="editor-left">
        <div className="section-title">工程</div>
        <div className="row" style={{ gap: 6 }}>
          <input className="input grow" placeholder="新工程名" value={newName}
                 onChange={(e) => setNewName(e.target.value)} />
          <button className="btn btn-ghost" onClick={createProject}>新建</button>
        </div>
        <div className="row" style={{ gap: 6, marginTop: 6 }}>
          <select className="input grow" value={size[0]}
                  onChange={(e) => {
                    const w = +e.target.value
                    setSize([w, w])
                  }}>
            {[256, 384, 512, 768].map((s) => <option key={s} value={s}>{s}×{s}</option>)}
          </select>
          <button className="btn btn-ghost" disabled={!pid} onClick={removeProject}>删除</button>
        </div>
        <div className="chips mt8">
          {projects.map((p) => (
            <button key={p.id} className={`chip ${p.id === pid ? 'on' : ''}`}
                    onClick={() => openProject(p.id)} title={`${p.tracks} 条轨道`}>
              {p.name}
            </button>
          ))}
          {!projects.length && <span className="hint">还没有工程</span>}
        </div>

        <div className="section-title" style={{ marginTop: 14 }}>骨骼</div>
        <div className="chips">
          {bones.map((b: any) => (
            <button key={b.id} className={`chip ${b.id === selBone ? 'on' : ''}`}
                    onClick={() => { setSelBone(b.id); setMode('bone') }}>
              {b.name}
            </button>
          ))}
          {proj && (
            <button className="chip" onClick={() => update((p) => {
              const id = `b${p.bones.length + 1}`
              const parent = p.bones.find((x: any) => x.id === selBone) || p.bones[0]
              p.bones.push({ id, name: `骨骼 ${p.bones.length}`, parent: parent?.id || null,
                x: (parent?.x || p.canvas[0] / 2) + 40, y: parent?.y || p.canvas[1] / 2, keys: [] })
              setSelBone(id)
            })}>+ 子骨骼</button>
          )}
        </div>

        <div className="section-title" style={{ marginTop: 14 }}>轨道</div>
        <button className="btn btn-ghost btn-block" onClick={() => setPicker((v) => !v)}>
          {picker ? '收起素材' : '+ 从素材库添加'}
        </button>
        {picker && (
          <div className="editor-picker">
            {assets.slice(0, 60).map((a) => (
              <img key={a.id} src={a.url} alt={a.name} title={a.name}
                   onClick={() => addTrack(a)} />
            ))}
          </div>
        )}
        <div className="editor-tracks">
          {(proj?.tracks || []).map((t: any, i: number) => (
            <div key={t.id} className={`editor-track ${t.id === selTrack ? 'on' : ''}`}
                 onClick={() => setSelTrack(t.id)}>
              <button className="mini" title="显示/隐藏"
                      onClick={(e) => { e.stopPropagation(); update((p) => {
                        const x = p.tracks.find((v: any) => v.id === t.id); x.visible = !x.visible
                      }) }}>
                {t.visible === false ? '○' : '●'}
              </button>
              <span className="grow">{t.name}</span>
              <select className="mini" value={t.bone} onClick={(e) => e.stopPropagation()}
                      onChange={(e) => update((p) => {
                        const x = p.tracks.find((v: any) => v.id === t.id); x.bone = e.target.value
                      })}>
                {bones.map((b: any) => <option key={b.id} value={b.id}>{b.name}</option>)}
              </select>
              <button className="mini" title="上移" disabled={i === 0}
                      onClick={(e) => { e.stopPropagation(); update((p) => {
                        const arr = p.tracks
                        ;[arr[i - 1], arr[i]] = [arr[i], arr[i - 1]]
                      }) }}>↑</button>
            </div>
          ))}
          {!proj?.tracks?.length && <div className="hint">从素材库添加一张图开始</div>}
        </div>
      </aside>

      <main className="editor-stage">
        <div className="editor-toolbar">
          <button className="btn btn-primary" onClick={() => setPlaying((v) => !v)}>
            {playing ? '暂停' : '播放'}
          </button>
          <div className="seg">
            {(['select', 'bone', 'warp', 'brush'] as Mode[]).map((m) => (
              <button key={m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}>
                {m === 'select' ? '移动' : m === 'bone' ? '骨骼' : m === 'warp' ? '变形' : '笔刷'}
              </button>
            ))}
          </div>
          <label className="hint" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input type="checkbox" checked={onion} onChange={(e) => setOnion(e.target.checked)} />
            洋葱皮
          </label>
          <div className="grow" />
          <span className="hint">帧 {frame + 1}/{proj?.frames || 0}</span>
          <select className="input" style={{ width: 76 }} value={zoom}
                  onChange={(e) => setZoom(+e.target.value)}>
            {[0.5, 0.75, 1, 1.5, 2].map((z) => <option key={z} value={z}>{Math.round(z * 100)}%</option>)}
          </select>
          <button className="btn" disabled={busy || !pid} onClick={doExport}>
            {busy ? '导出中…' : '导出 ZIP（含法线）'}
          </button>
        </div>

        <div className="editor-canvas-wrap">
          {!proj ? (
            <div className="hint">左侧新建或选一个工程</div>
          ) : (
            <div
              className="editor-canvas checker"
              ref={stageRef}
              style={{
                width: dims[0] * zoom,
                height: dims[1] * zoom,
                backgroundSize: `${24 * zoom}px ${24 * zoom}px`,
              }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
            >
              {pid && onion && [frame - 2, frame - 1, frame + 1].map((i) => (
                i >= 0 && i < proj.frames && (
                  <img key={i} className="onion" style={{ opacity: i === frame - 1 ? 0.28 : 0.14 }}
                       src={api.animFrameUrl(pid, i, 1, bump)} alt="" />
                )
              ))}
              {pid && (
                <img className="frame" src={api.animFrameUrl(pid, frame, 1, bump)} alt="frame" />
              )}

              <svg className="editor-overlay" width={dims[0]} height={dims[1]}
                   style={{ transform: `scale(${zoom})`, transformOrigin: '0 0' }}>
                {/* 骨骼：从父到子的连线 + 关节 */}
                {bones.map((b: any) => {
                  const w = worlds[b.id]
                  if (!w) return null
                  const [bx, by] = apply(w, 0, 0)
                  const p = b.parent ? worlds[b.parent] : null
                  const [px, py] = p ? apply(p, 0, 0) : [bx, by]
                  return (
                    <g key={b.id}>
                      <line x1={px} y1={py} x2={bx} y2={by}
                            stroke={b.id === selBone ? '#00FF88' : 'rgba(0,255,136,0.45)'}
                            strokeWidth={2 / zoom} />
                      <circle cx={bx} cy={by} r={(b.id === selBone ? 6 : 4) / zoom}
                              fill={b.id === selBone ? '#00FF88' : 'rgba(0,255,136,0.6)'} />
                    </g>
                  )
                })}
                {/* 变形网格 */}
                {mode === 'warp' && track && (() => {
                  const m = trackMatrix(proj, track, frame)
                  const pts = warpPoints(track, assetOf(track.assetId))
                  const scr = pts.map((p) => apply(m, p[0], p[1]))
                  const lines = []
                  for (let r = 0; r < 3; r++) {
                    lines.push(<polyline key={`r${r}`}
                      points={[0, 1, 2].map((c) => scr[r * 3 + c].join(',')).join(' ')}
                      fill="none" stroke="rgba(255,213,0,0.55)" strokeWidth={1 / zoom} />)
                  }
                  for (let c = 0; c < 3; c++) {
                    lines.push(<polyline key={`c${c}`}
                      points={[0, 1, 2].map((r) => scr[r * 3 + c].join(',')).join(' ')}
                      fill="none" stroke="rgba(255,213,0,0.55)" strokeWidth={1 / zoom} />)
                  }
                  return (
                    <g>
                      {lines}
                      {scr.map((p, i) => (
                        <rect key={i} x={p[0] - 4 / zoom} y={p[1] - 4 / zoom}
                              width={8 / zoom} height={8 / zoom} fill="#FFD500" />
                      ))}
                    </g>
                  )
                })()}
                {/* 笔迹预览线 */}
                {mode === 'brush' && dragRef.current?.kind === 'brush' && (
                  <polyline
                    points={dragRef.current.points.map((p: number[]) => `${p[0]},${p[1]}`).join(' ')}
                    fill="none" stroke={brush.color} strokeWidth={brush.size} strokeLinecap="round" />
                )}
              </svg>
            </div>
          )}
        </div>

        {proj && (
          <div className="editor-timeline">
            <div className="row" style={{ gap: 6 }}>
              <button className="btn btn-ghost" onClick={() => upsertKey('track', {})}>+ 轨道关键帧</button>
              <button className="btn btn-ghost" onClick={() => upsertKey('bone', {})}>+ 骨骼关键帧</button>
              <button className="btn btn-ghost" onClick={deleteKey}>删轨道关键帧</button>
              <div className="grow" />
              <span className="hint">帧率 {proj.fps} · 共 {proj.frames} 帧</span>
            </div>
            <div className="frames">
              {Array.from({ length: proj.frames }).map((_, i) => {
                const tk = (track?.keyframes || []).some((k: any) => k.frame === i)
                const bk = (bones.find((b: any) => b.id === selBone)?.keys || [])
                  .some((k: any) => k.frame === i)
                return (
                  <button key={i} className={`frame-cell ${i === frame ? 'on' : ''}`}
                          onClick={() => setFrame(i)}>
                    <span className="n">{i + 1}</span>
                    <span className="marks">
                      {tk && <i className="mk track" />}
                      {bk && <i className="mk bone" />}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {note && <div className="ok">{note}</div>}
        {error && <div className="err">{error}</div>}
      </main>

      <aside className="editor-right">
        <div className="section-title">当前帧</div>
        {proj && (() => {
          const k = sample(track?.keyframes || [], frame)
          const bk = sample(bones.find((b: any) => b.id === selBone)?.keys || [], frame)
          return (
            <>
              <div className="label">轨道位移</div>
              <div className="row" style={{ gap: 6 }}>
                <input className="input" type="number" value={Math.round(k.x)}
                       onChange={(e) => upsertKey('track', { x: +e.target.value })} />
                <input className="input" type="number" value={Math.round(k.y)}
                       onChange={(e) => upsertKey('track', { y: +e.target.value })} />
              </div>
              <div className="label" style={{ marginTop: 8 }}>旋转 {Math.round(k.rot)}°</div>
              <input type="range" min={-180} max={180} value={k.rot}
                     style={{ width: '100%' }}
                     onChange={(e) => upsertKey('track', { rot: +e.target.value })} />
              <div className="label" style={{ marginTop: 6 }}>缩放 {k.scale.toFixed(2)}</div>
              <input type="range" min={0.2} max={3} step={0.01} value={k.scale}
                     style={{ width: '100%' }}
                     onChange={(e) => upsertKey('track', { scale: +e.target.value })} />
              <div className="label" style={{ marginTop: 6 }}>不透明度 {Math.round(k.opacity * 100)}%</div>
              <input type="range" min={0} max={1} step={0.01} value={k.opacity}
                     style={{ width: '100%' }}
                     onChange={(e) => upsertKey('track', { opacity: +e.target.value })} />

              <div className="label" style={{ marginTop: 10 }}>骨骼「{selBone}」位移</div>
              <div className="row" style={{ gap: 6 }}>
                <input className="input" type="number" value={Math.round(bk.x)}
                       onChange={(e) => upsertKey('bone', { x: +e.target.value })} />
                <input className="input" type="number" value={Math.round(bk.y)}
                       onChange={(e) => upsertKey('bone', { y: +e.target.value })} />
              </div>
              <div className="label" style={{ marginTop: 8 }}>骨骼旋转 {Math.round(bk.rot)}°</div>
              <input type="range" min={-180} max={180} value={bk.rot} style={{ width: '100%' }}
                     onChange={(e) => upsertKey('bone', { rot: +e.target.value })} />
            </>
          )
        })()}

        {mode === 'brush' && (
          <>
            <div className="section-title" style={{ marginTop: 14 }}>笔刷</div>
            <div className="chips">
              {['slash', 'bristle', 'dry', 'spark', 'echo'].map((b) => (
                <button key={b} className={`chip sm ${brush.brush === b ? 'on' : ''}`}
                        onClick={() => setBrush({ ...brush, brush: b })}>{b}</button>
              ))}
            </div>
            <div className="row" style={{ gap: 6, marginTop: 6 }}>
              <input type="color" value={brush.color}
                     onChange={(e) => setBrush({ ...brush, color: e.target.value })} />
              <input className="input grow" type="range" min={2} max={80} value={brush.size}
                     onChange={(e) => setBrush({ ...brush, size: +e.target.value })} />
              <span className="hint">{brush.size}px</span>
            </div>
            <div className="hint">在画布上拖动画笔；松开后写入当前轨道的笔迹列表（全帧可见）。</div>
          </>
        )}

        {track && (
          <>
            <div className="section-title" style={{ marginTop: 14 }}>轨道属性</div>
            <div className="label">混合模式</div>
            <select className="input" value={track.blend}
                    onChange={(e) => update((p) => {
                      const t = p.tracks.find((v: any) => v.id === track.id); t.blend = e.target.value
                    })}>
              <option value="normal">正常</option>
              <option value="add">相加</option>
              <option value="multiply">正片叠底</option>
            </select>
            <button className="btn btn-ghost btn-block" style={{ marginTop: 8 }}
                    onClick={() => update((p) => {
                      const t = p.tracks.find((v: any) => v.id === track.id)
                      t.warp = Array(9).fill([0, 0])
                    })}>重置变形</button>
            <button className="btn btn-ghost btn-block" style={{ marginTop: 6 }}
                    onClick={() => update((p) => {
                      p.tracks = p.tracks.filter((v: any) => v.id !== track.id)
                      setSelTrack('')
                    })}>删除轨道</button>

            <div className="label" style={{ marginTop: 10 }}>笔迹 {track.fx?.length || 0} 条</div>
            <button className="btn btn-ghost btn-block" disabled={!track.fx?.length}
                    onClick={() => update((p) => {
                      const t = p.tracks.find((v: any) => v.id === track.id)
                      t.fx = t.fx.slice(0, -1)
                    })}>撤销最后一笔</button>
          </>
        )}

        <div className="hint" style={{ marginTop: 12 }}>
          拖动 = 在当前帧写关键帧；骨骼按父子级联，转父骨骼子骨骼跟着走。
          变形是 3×3 双线性网格，拖控制点即改形状。
        </div>
      </aside>
    </div>
  )
}
