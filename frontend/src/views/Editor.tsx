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

type Mode = 'select' | 'bone' | 'warp' | 'brush' | 'weight'
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

type IkOverride = Record<string, { x: number; y: number; ang: number; sc: number }>

function fkWorld(proj: any, frame: number, ov?: IkOverride): Record<string, Aff> {
  const bones: any[] = proj.bones || []
  const out: Record<string, Aff> = {}
  const seen = new Set<string>()
  const build = (b: any): Aff => {
    if (out[b.id]) return out[b.id]
    if (seen.has(b.id)) return (out[b.id] = I)   // 环：兜底，别让递归爆栈
    seen.add(b.id)
    const o = ov?.[b.id]
    let m: Aff
    if (o) {
      m = chain(T(o.x, o.y), R(o.ang), S(o.sc))
    } else {
      const k = sample(b.keys || [], frame)
      const local = chain(T(k.x, k.y), R(k.rot), S(k.scale))
      const p = b.parent ? bones.find((x) => x.id === b.parent) : null
      m = p ? chain(build(p), T(b.x - p.x, b.y - p.y), local) : chain(T(b.x, b.y), local)
    }
    out[b.id] = m
    return m
  }
  bones.forEach(build)
  return out
}

const matAngle = (m: Aff) => Math.atan2(m.b, m.a)
const matScale = (m: Aff) => Math.max(0.01, Math.hypot(m.a, m.b))

/** FABRIK：与后端 render._solve_ik 同一套算法，保证叠加层和成图一致 */
function solveIk(proj: any, frame: number, base: Record<string, Aff>,
                 ov: IkOverride, ik: any) {
  const bones: any[] = proj.bones || []
  const ids: string[] = []
  let cur = ik.end
  while (cur && ids.length < (ik.chain || 2)) {
    const b = bones.find((x) => x.id === cur)
    if (!b) break
    ids.push(cur)
    cur = b.parent
  }
  ids.reverse()
  if (ids.length < 2) return

  const pos = ids.map((id) => {
    const o = ov[id]
    if (o) return { x: o.x, y: o.y }
    const m = base[id] || I
    return { x: m.e, y: m.f }
  })
  const angs = ids.map((id) => (ov[id] ? (ov[id].ang * Math.PI) / 180 : matAngle(base[id] || I)))
  const lens = ids.map((id) => bones.find((x) => x.id === id)?.len || 60)
  const last = pos[pos.length - 1]
  const tip = {
    x: last.x + Math.cos(angs[angs.length - 1]) * lens[lens.length - 1],
    y: last.y + Math.sin(angs[angs.length - 1]) * lens[lens.length - 1],
  }
  const pts = [...pos, tip]
  const k = sample(ik.keys || [], frame)
  const root = { ...pts[0] }
  const total = lens.reduce((a, b) => a + b, 0)
  const dx = k.x - root.x
  const dy = k.y - root.y
  const reach = Math.hypot(dx, dy)

  if (reach >= total) {
    const ux = dx / (reach || 1)
    const uy = dy / (reach || 1)
    let cx = root.x
    let cy = root.y
    for (let i = 0; i < lens.length; i++) {
      cx += ux * lens[i]
      cy += uy * lens[i]
      pts[i + 1] = { x: cx, y: cy }
    }
  } else {
    for (let it = 0; it < 12; it++) {
      pts[pts.length - 1] = { x: k.x, y: k.y }
      for (let i = pts.length - 2; i >= 0; i--) {
        const vx = pts[i].x - pts[i + 1].x
        const vy = pts[i].y - pts[i + 1].y
        const n = Math.hypot(vx, vy) || 1
        pts[i] = { x: pts[i + 1].x + (vx / n) * lens[i], y: pts[i + 1].y + (vy / n) * lens[i] }
      }
      pts[0] = { ...root }
      for (let i = 0; i < pts.length - 1; i++) {
        const vx = pts[i + 1].x - pts[i].x
        const vy = pts[i + 1].y - pts[i].y
        const n = Math.hypot(vx, vy) || 1
        pts[i + 1] = { x: pts[i].x + (vx / n) * lens[i], y: pts[i].y + (vy / n) * lens[i] }
      }
      if (Math.hypot(pts[pts.length - 1].x - k.x, pts[pts.length - 1].y - k.y) < 0.5) break
    }
  }

  ids.forEach((id, i) => {
    const ang = (Math.atan2(pts[i + 1].y - pts[i].y, pts[i + 1].x - pts[i].x) * 180) / Math.PI
    const sc = ov[id]?.sc ?? matScale(base[id] || I)
    ov[id] = { x: pts[i].x, y: pts[i].y, ang, sc }
  })
}

/** 与后端 render._bezier 同一套求值，保证预览和导出曲线一致 */
function bezierEase(t: number, bez: number[]): number {
  const [x1, y1, x2, y2] = bez
  const bz = (u: number, a: number, b: number) => {
    const v = 1 - u
    return 3 * v * v * u * a + 3 * v * u * u * b + u * u * u
  }
  const slope = (u: number, a: number, b: number) => {
    const v = 1 - u
    return 3 * v * v * a + 6 * v * u * (b - a) + 3 * u * u * (1 - b)
  }
  let lo = 0
  let hi = 1
  let u = t
  for (let i = 0; i < 24; i++) {
    const x = bz(u, x1, x2)
    if (Math.abs(x - t) < 1e-5) break
    if (x < t) lo = u
    else hi = u
    u = (lo + hi) / 2
  }
  for (let i = 0; i < 8; i++) {
    const d = slope(u, x1, x2)
    if (Math.abs(d) < 1e-6) break
    u = Math.max(0, Math.min(1, u - (bz(u, x1, x2) - t) / d))
  }
  return bz(u, y1, y2)
}

/** 变形网格采样：有关键帧插值，没有就用静态 warp */
function sampleWarp(t: any, frame: number): number[][] {
  const keys = [...(t?.warpKeys || [])].sort((a: any, b: any) => a.frame - b.frame)
  const zero = Array.from({ length: 9 }, () => [0, 0])
  if (!keys.length) return (t?.warp || zero).map((q: number[]) => [q[0], q[1]])
  if (frame <= keys[0].frame) return keys[0].warp
  if (frame >= keys[keys.length - 1].frame) return keys[keys.length - 1].warp
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i]
    const b = keys[i + 1]
    if (frame >= a.frame && frame <= b.frame) {
      const span = b.frame - a.frame || 1
      let k = (frame - a.frame) / span
      k = b.bez
        ? bezierEase(k, b.bez)
        : b.ease === 'ease-in' ? k * k : b.ease === 'ease-out' ? 1 - (1 - k) * (1 - k) : k
      return a.warp.map((p: number[], j: number) => [
        p[0] + (b.warp[j][0] - p[0]) * k,
        p[1] + (b.warp[j][1] - p[1]) * k,
      ])
    }
  }
  return keys[0].warp
}

function boneWorld(proj: any, frame: number): Record<string, Aff> {
  const base = fkWorld(proj, frame)
  const iks = proj.ik || []
  if (!iks.length) return base
  const ov: IkOverride = {}
  iks.forEach((ik: any) => solveIk(proj, frame, base, ov, ik))
  return Object.keys(ov).length ? fkWorld(proj, frame, ov) : base
}

function trackMatrix(proj: any, t: any, frame: number): Aff {
  const w = boneWorld(proj, frame)[t.bone || 'root'] || I
  const k = sample(t.keyframes || [], frame)
  const [px, py] = t.pivot || [0, 0]
  return chain(w, T(k.x, k.y), R(k.rot), S(k.scale), T(-px, -py))
}

type CurveProps = { bez: number[]; onChange: (b: number[]) => void }

/** 关键帧缓动曲线：x=时间、y=进度，两个控制点可拖。y 允许出界（做回弹） */
function CurveEditor({ bez, onChange }: CurveProps) {
  const ref = useRef<SVGSVGElement>(null)
  const drag = useRef<number | null>(null)
  const W = 150
  const P = 14
  const sx = (x: number) => P + x * (W - 2 * P)
  const sy = (y: number) => P + (1 - y) * (W - 2 * P)
  const toUnit = (cx: number, cy: number): [number, number] => {
    const r = ref.current!.getBoundingClientRect()
    return [
      Math.max(0, Math.min(1, (cx - r.left - P) / (W - 2 * P))),
      Math.max(-0.6, Math.min(1.6, 1 - (cy - r.top - P) / (W - 2 * P))),
    ]
  }
  const move = (e: React.PointerEvent) => {
    if (drag.current == null) return
    const [x, y] = toUnit(e.clientX, e.clientY)
    const next = [...bez]
    next[drag.current * 2] = +x.toFixed(3)
    next[drag.current * 2 + 1] = +y.toFixed(3)
    onChange(next)
  }
  const [x1, y1, x2, y2] = bez
  return (
    <svg
      ref={ref}
      className="curve"
      width={W}
      height={W}
      viewBox={`0 0 ${W} ${W}`}
      onPointerMove={move}
      onPointerUp={() => { drag.current = null }}
      onPointerLeave={() => { drag.current = null }}
    >
      {[0.25, 0.5, 0.75].map((g) => (
        <g key={g}>
          <line x1={sx(g)} y1={P} x2={sx(g)} y2={W - P} stroke="rgba(255,255,255,0.06)" />
          <line x1={P} y1={sy(g)} x2={W - P} y2={sy(g)} stroke="rgba(255,255,255,0.06)" />
        </g>
      ))}
      <rect x={P} y={P} width={W - 2 * P} height={W - 2 * P} fill="none" stroke="var(--line)" />
      <line x1={P} y1={sy(0)} x2={sx(x1)} y2={sy(y1)} stroke="rgba(255,213,0,0.45)" />
      <line x1={W - P} y1={sy(1)} x2={sx(x2)} y2={sy(y2)} stroke="rgba(255,213,0,0.45)" />
      <path
        d={`M ${sx(0)} ${sy(0)} C ${sx(x1)} ${sy(y1)} ${sx(x2)} ${sy(y2)} ${sx(1)} ${sy(1)}`}
        fill="none"
        stroke="#00FF88"
        strokeWidth={2}
      />
      {[0, 1].map((i) => {
        const px = i === 0 ? x1 : x2
        const py = i === 0 ? y1 : y2
        return (
          <circle
            key={i}
            cx={sx(px)}
            cy={sy(py)}
            r={5}
            fill="#FFD500"
            style={{ cursor: 'grab' }}
            onPointerDown={(e) => {
              drag.current = i
              ;(e.target as Element).setPointerCapture?.(e.pointerId)
            }}
          />
        )
      })}
    </svg>
  )
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
  const [selIk, setSelIk] = useState('')
  const [curveOwner, setCurveOwner] = useState('track')
  const [bump, setBump] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [brush, setBrush] = useState({ brush: 'slash', color: '#ffcc33', size: 14 })
  const [assets, setAssets] = useState<Asset[]>([])
  const [picker, setPicker] = useState(false)
  const [busy, setBusy] = useState(false)
  const [godotBusy, setGodotBusy] = useState(false)
  // 权重笔刷：视图数据来自后端（数学只有一处实现），这里只负责显示与写入 paint
  const [wView, setWView] = useState<any>(null)
  const [wBone, setWBone] = useState('')
  const [wRadius, setWRadius] = useState(2)
  const [wStrength, setWStrength] = useState(0.5)
  const [wDir, setWDir] = useState(1)
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
    // 有 IK 就默认选中第一个、并把骨骼切到链末端：
    // 否则右侧曲线面板拿不到 IK 关键帧，看起来像「IK 不能调曲线」
    const firstIk = (p.ik || [])[0]
    setSelIk(firstIk?.id || '')
    setSelBone(firstIk?.end || p.bones?.[0]?.id || 'root')
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

  // 权重视图：轨道 / 帧 / 轨道改动（bump）变化时重拉
  useEffect(() => {
    if (mode !== 'weight' || !pid || !track?.skin) {
      setWView(null)
      return
    }
    let alive = true
    api.animWeights(pid, track.id, frame)
      .then((v) => {
        if (!alive) return
        setWView(v)
        setWBone((b) => (v.bones?.some((x: any) => x.id === b) ? b : v.bones?.[0]?.id || ''))
      })
      .catch(() => setWView(null))
    return () => { alive = false }
  }, [mode, pid, track?.id, track?.skin, frame, bump])

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

  const upsertIk = (ikId: string, x: number, y: number) => {
    update((p) => {
      const ik = (p.ik || []).find((v: any) => v.id === ikId)
      if (!ik) return
      ik.keys = ik.keys || []
      let k = ik.keys.find((v: any) => v.frame === frame)
      if (!k) {
        k = { frame, x, y, ease: 'linear', bez: null }
        ik.keys.push(k)
      } else {
        k.x = x
        k.y = y
      }
      ik.keys.sort((a: any, b: any) => a.frame - b.frame)
    })
  }

  const addIk = () => {
    const b = bones.find((v: any) => v.id === selBone)
    if (!b || !b.parent) {
      setNote('根骨骼没有父级，做不了 IK；先给它加一根子骨骼')
      return
    }
    const w = worlds[selBone]
    const id = Math.random().toString(16).slice(2, 8)
    update((p) => {
      p.ik = p.ik || []
      p.ik.push({
        id, name: `IK ${p.ik.length + 1}`, end: selBone, chain: 2,
        keys: [{ frame, x: (w?.e ?? b.x) + 50, y: (w?.f ?? b.y) - 30, ease: 'linear', bez: null }],
      })
    })
    setSelIk(id)
    setMode('bone')
  }

  const upsertIkWeight = (ikId: string, w: number) => {
    update((p) => {
      const ik = (p.ik || []).find((v: any) => v.id === ikId)
      if (!ik) return
      ik.keys = ik.keys || []
      let k = ik.keys.find((v: any) => v.frame === frame)
      if (!k) {
        const cur = sample(ik.keys, frame)
        k = { frame, x: cur.x, y: cur.y, w, ease: 'linear', bez: null }
        ik.keys.push(k)
      } else {
        k.w = w
      }
      ik.keys.sort((a: any, b: any) => a.frame - b.frame)
    })
  }

  const addWarpKey = () => {
    if (!track) return
    update((p) => {
      const t = p.tracks.find((v: any) => v.id === track.id)
      if (!t) return
      t.warpKeys = t.warpKeys || []
      const warp = sampleWarp(t, frame).map((q: number[]) => [q[0], q[1]])
      const exist = t.warpKeys.find((v: any) => v.frame === frame)
      if (exist) exist.warp = warp
      else t.warpKeys.push({ frame, warp, ease: 'linear', bez: null })
      t.warpKeys.sort((a: any, b: any) => a.frame - b.frame)
    })
    setMode('warp')
  }

  const deleteWarpKey = () => {
    if (!track) return
    update((p) => {
      const t = p.tracks.find((v: any) => v.id === track.id)
      if (!t) return
      t.warpKeys = (t.warpKeys || []).filter((v: any) => v.frame !== frame)
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

    if (mode === 'weight' && wView) {
      const g = wView.grid + 1
      let best = -1
      let bestD = 1e9
      wView.rest.forEach((p: number[], i: number) => {
        const d = Math.hypot(p[0] - x, p[1] - y)
        if (d < bestD) { bestD = d; best = i }
      })
      if (bestD < 18 / zoom) {
        dragRef.current = { kind: 'weight', idx: best, g }
      }
      return
    }
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
      for (const ik of proj.ik || []) {
        if (ik.id === selIk && ik.pole
            && Math.hypot(ik.pole[0] - x, ik.pole[1] - y) < 16 / zoom) {
          dragRef.current = { kind: 'pole', id: ik.id }
          return
        }
      }
      for (const ik of proj.ik || []) {
        const k = sample(ik.keys || [], frame)
        if (Math.hypot(k.x - x, k.y - y) < 16 / zoom) {
          dragRef.current = { kind: 'ik', id: ik.id }
          return
        }
      }
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
    if (d.kind === 'weight') {
      const g = d.g
      const ci = wView?.bones?.findIndex((b: any) => b.id === wBone) ?? -1
      if (ci < 0) return
      const ci0 = d.idx % g
      const rj0 = Math.floor(d.idx / g)
      const step = 0.04 * wStrength * wDir
      const next = wView.weights.map((row: number[]) => [...row])
      for (let j = rj0 - wRadius; j <= rj0 + wRadius; j++) {
        for (let i = ci0 - wRadius; i <= ci0 + wRadius; i++) {
          if (i < 0 || j < 0 || i >= g || j >= g) continue
          if (Math.hypot(i - ci0, j - rj0) > wRadius) continue
          const v = j * g + i
          next[v][ci] = Math.max(0, Math.min(1, (next[v][ci] || 0) + step))
        }
      }
      setWView({ ...wView, weights: next })
      update((p) => {
        const t = p.tracks.find((v: any) => v.id === track.id)
        if (!t?.skin) return
        t.skin.paint = t.skin.paint || {}
        const cells = t.skin.paint[wBone] || {}
        for (let j = rj0 - wRadius; j <= rj0 + wRadius; j++) {
          for (let i = ci0 - wRadius; i <= ci0 + wRadius; i++) {
            if (i < 0 || j < 0 || i >= g || j >= g) continue
            if (Math.hypot(i - ci0, j - rj0) > wRadius) continue
            cells[String(j * g + i)] = +next[j * g + i][ci].toFixed(3)
          }
        }
        t.skin.paint[wBone] = cells
      })
      return
    }
    if (d.kind === 'ik') {
      upsertIk(d.id, x, y)
      return
    }
    if (d.kind === 'pole') {
      update((p) => {
        const ik = (p.ik || []).find((v: any) => v.id === d.id)
        if (ik) ik.pole = [+x.toFixed(1), +y.toFixed(1)]
      })
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
        // 第一次拖变形就切到「关键帧模式」：把当前静态形状作为第 0 帧的基准，
        // 否则一动就把其它帧的形状抹平了
        if (!t.warpKeys?.length) {
          const seed = (t.warp || Array(9).fill([0, 0])).map((q: number[]) => [q[0], q[1]])
          t.warpKeys = frame === 0 ? [] : [{ frame: 0, warp: seed, ease: 'linear', bez: null }]
        }
        const base = sampleWarp(t, frame).map((q: number[]) => [q[0], q[1]])
        base[d.index] = [base[d.index][0] + sx, base[d.index][1] + sy]
        let k = t.warpKeys.find((v: any) => v.frame === frame)
        if (!k) {
          k = { frame, warp: base, ease: 'linear', bez: null }
          t.warpKeys.push(k)
        } else {
          k.warp = base
        }
        t.warpKeys.sort((a: any, b: any) => a.frame - b.frame)
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
    // pivot 默认取**底部中心**而不是几何中心：动画里锚点几乎总是「脚底/落地点」，
    // 几何中心会让角色一放进来就悬在半空还得自己往上拖。
    // 素材尺寸和画布差太多时明确提示——之前是默默出画布，用户一脸茫然。
    const canvasW = proj?.canvas?.[0] || dims[0]
    const canvasH = proj?.canvas?.[1] || dims[1]
    const hint = (a.width > canvasW || a.height > canvasH)
      ? ` 素材 ${a.width}×${a.height} 超出画布 ${canvasW}×${canvasH}，建议调大画布或用关键帧缩小。`
      : ''
    setNote(`已添加轨道：${a.name}（pivot 落在底部中心）${hint}`)
    // id 先算好：写在 setState 更新函数里的话，React 严格模式重放会生成两个 id
    const tid = Math.random().toString(16).slice(2, 10)
    update((p) => {
      p.tracks = p.tracks || []
      p.tracks.push({
        id: tid,
        name: a.name || '轨道',
        assetId: a.id,
        bone: selBone || 'root',
        pivot: [Math.round(a.width / 2), a.height],
        warp: Array(9).fill([0, 0]),
        keyframes: [{ frame: 0, x: 0, y: 0, rot: 0, scale: 1, opacity: 1, ease: 'linear' }],
        fx: [], opacity: 1, blend: 'normal', visible: true,
      })
    })
    setSelTrack(tid)
    setPicker(false)
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

  const doGodot = async () => {
    if (!pid || !proj) return
    setGodotBusy(true)
    setError('')
    try {
      await api.animGodot(pid, proj.name || 'anim')
      setNote('已导出 Godot 包：anim.json + textures/ + anim_player.gd + README')
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setGodotBusy(false)
    }
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

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const warpPoints = (t: any, a?: Asset): Pt[] => {
    const w = a?.width || 64
    const h = a?.height || 64
    const pts: Pt[] = []
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const base: Pt = [(c * w) / 2, (r * h) / 2]
        const off = sampleWarp(t, frame)[r * 3 + c] || [0, 0]
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
          <input className="input" type="number" style={{ width: 64 }} value={size[0]} min={16}
                 onChange={(e) => setSize([+e.target.value || 512, size[1]])} />
          <span className="hint">×</span>
          <input className="input" type="number" style={{ width: 64 }} value={size[1]} min={16}
                 onChange={(e) => setSize([size[0], +e.target.value || 512])} />
          <div className="chips">
            {[256, 384, 512, 768].map((v) => (
              <button key={v} className="chip sm"
                      onClick={() => setSize([v, v])}>{v}</button>
            ))}
          </div>
        </div>
        <div className="row" style={{ gap: 6, marginTop: 4 }}>
          <button className="btn btn-ghost grow" onClick={createProject}>按此尺寸新建</button>
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

        <div className="section-title" style={{ marginTop: 14 }}>IK 目标</div>
        <button className="btn btn-ghost btn-block" onClick={addIk}>
          + 给「{bones.find((b: any) => b.id === selBone)?.name || selBone}」加 IK
        </button>
        <div className="chips mt8">
          {(proj?.ik || []).map((ik: any) => (
            <button key={ik.id}
                    className={`chip ${ik.id === selIk ? 'on' : ''}`}
                    title={`${ik.chain} 节链`}
                    onClick={() => { setSelIk(ik.id); setSelBone(ik.end); setMode('bone') }}>
              {ik.name}
            </button>
          ))}
          {!proj?.ik?.length && <span className="hint">还没有 IK 目标</span>}
        </div>
        {(() => {
          const ik = (proj?.ik || []).find((v: any) => v.id === selIk)
          if (!ik) return null
          return (
            <div style={{ marginTop: 6 }}>
              <div className="label">链长 {ik.chain} 节</div>
              <input type="range" min={2} max={4} value={ik.chain} style={{ width: '100%' }}
                     onChange={(e) => update((p) => {
                       const v = (p.ik || []).find((x: any) => x.id === ik.id)
                       if (v) v.chain = +e.target.value
                     })} />
              <button className="btn btn-ghost btn-block" style={{ marginTop: 4 }}
                      onClick={() => {
                        update((p) => { p.ik = (p.ik || []).filter((x: any) => x.id !== ik.id) })
                        setSelIk('')
                      }}>删除 IK 目标</button>
            </div>
          )
        })()}

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
            {(['select', 'bone', 'warp', 'brush', 'weight'] as Mode[]).map((m) => (
              <button key={m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}
                      disabled={m === 'weight' && !track?.skin}>
                {m === 'select' ? '移动' : m === 'bone' ? '骨骼' : m === 'warp' ? '变形'
                  : m === 'brush' ? '笔刷' : '权重'}
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
          <button className="btn btn-ghost" disabled={godotBusy || !pid} onClick={doGodot}>
            {godotBusy ? '打包中…' : '导出 Godot 包'}
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
                {/* IK 目标：红叉手柄，虚线连到链末关节 */}
                {(proj.ik || []).map((ik: any) => {
                  const k = sample(ik.keys || [], frame)
                  const end = worlds[ik.end]
                  const ex = end ? end.e : 0
                  const ey = end ? end.f : 0
                  return (
                    <g key={ik.id}>
                      <line x1={ex} y1={ey} x2={k.x} y2={k.y}
                            stroke="rgba(255,90,90,0.55)" strokeDasharray="4 4"
                            strokeWidth={1 / zoom} />
                      <circle cx={k.x} cy={k.y} r={9 / zoom} fill="none"
                              stroke={ik.id === selIk ? '#ff5a5a' : 'rgba(255,90,90,0.75)'}
                              strokeWidth={2 / zoom} />
                      <line x1={k.x - 13 / zoom} y1={k.y} x2={k.x + 13 / zoom} y2={k.y}
                            stroke="#ff5a5a" strokeWidth={1 / zoom} />
                      <line x1={k.x} y1={k.y - 13 / zoom} x2={k.x} y2={k.y + 13 / zoom}
                            stroke="#ff5a5a" strokeWidth={1 / zoom} />
                    </g>
                  )
                })}
                {/* pole：蓝钻，决定肘/膝朝哪边 */}
                {(proj.ik || []).filter((ik: any) => ik.id === selIk && ik.pole).map((ik: any) => (
                  <g key={`pole-${ik.id}`}>
                    <line x1={ik.pole[0]} y1={ik.pole[1]}
                          x2={sample(ik.keys || [], frame).x}
                          y2={sample(ik.keys || [], frame).y}
                          stroke="rgba(91,87,235,0.45)" strokeDasharray="3 5"
                          strokeWidth={1 / zoom} />
                    <rect
                      x={ik.pole[0] - 7 / zoom}
                      y={ik.pole[1] - 7 / zoom}
                      width={14 / zoom}
                      height={14 / zoom}
                      transform={`rotate(45 ${ik.pole[0]} ${ik.pole[1]})`}
                      fill="#5B57EB"
                    />
                  </g>
                ))}
                {/* 权重热力：当前骨骼权重 0→红，1→蓝 */}
                {mode === 'weight' && wView && (() => {
                  const ci = wView.bones.findIndex((b: any) => b.id === wBone)
                  const g = wView.grid + 1
                  return (
                    <g>
                      {wView.rest.map((p: number[], i: number) => {
                        const w = ci >= 0 ? (wView.weights[i]?.[ci] ?? 0) : 0
                        const r = Math.round(255 * (1 - w))
                        const b = Math.round(255 * w)
                        const gch = Math.round(60 + 40 * (1 - Math.abs(w - 0.5) * 2))
                        return (
                          <g key={i}>
                            <circle cx={p[0]} cy={p[1]} r={6 / zoom}
                                    fill={`rgba(${r},${gch},${b},0.55)`}
                                    stroke="rgba(255,255,255,0.35)" strokeWidth={0.6 / zoom} />
                          </g>
                        )
                      })}
                    </g>
                  )
                })()}
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
              <button className="btn btn-ghost" onClick={addWarpKey}>+ 变形关键帧</button>
              <button className="btn btn-ghost" onClick={deleteWarpKey}>删变形关键帧</button>
              <div className="grow" />
              <span className="hint">帧率 {proj.fps} · 共 {proj.frames} 帧</span>
            </div>
            <div className="frames">
              {Array.from({ length: proj.frames }).map((_, i) => {
                const tk = (track?.keyframes || []).some((k: any) => k.frame === i)
                const bk = (bones.find((b: any) => b.id === selBone)?.keys || [])
                  .some((k: any) => k.frame === i)
                const wk = (track?.warpKeys || []).some((k: any) => k.frame === i)
                return (
                  <button key={i} className={`frame-cell ${i === frame ? 'on' : ''}`}
                          onClick={() => setFrame(i)}>
                    <span className="n">{i + 1}</span>
                    <span className="marks">
                      {tk && <i className="mk track" />}
                      {bk && <i className="mk bone" />}
                      {wk && <i className="mk warp" />}
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

        {(() => {
          // 曲线归属可以是轨道 / 骨骼 / IK 目标：三者都能打关键帧，
          // 只给轨道做曲线编辑器的话，IK 的缓动就没地方调了
          const owners: { id: string; label: string; keys: any[] }[] = []
          const tk = track?.keyframes || []
          if (tk.length) owners.push({ id: 'track', label: '轨道', keys: tk })
          const bk = bones.find((b: any) => b.id === selBone)?.keys || []
          if (bk.length) owners.push({ id: 'bone', label: '骨骼', keys: bk })
          const curIk = (proj?.ik || []).find((v: any) => v.id === selIk)
          if (curIk?.keys?.length) owners.push({ id: 'ik', label: 'IK', keys: curIk.keys })
          const usable = owners.filter((o) => o.keys.length >= 2)
          if (!usable.length) {
            return (
              <div className="hint" style={{ marginTop: 10 }}>
                至少要有两个关键帧才能调曲线（轨道 / 骨骼 / IK 任一有就行）
              </div>
            )
          }
          const owner = usable.find((o) => o.id === curveOwner) || usable[0]
          const list = [...owner.keys].sort((a: any, b: any) => a.frame - b.frame)
          const dest = list.find((k: any) => k.frame > frame) || list[list.length - 1]
          const prev = [...list].filter((k: any) => k.frame < dest.frame).pop()
          if (!prev) return null
          const bez = dest.bez || [0.42, 0, 0.58, 1]
          const setBez = (b: number[] | null) => update((p) => {
            let arr: any[] = []
            if (owner.id === 'track') {
              arr = p.tracks.find((x: any) => x.id === selTrack)?.keyframes || []
            } else if (owner.id === 'bone') {
              arr = p.bones.find((x: any) => x.id === selBone)?.keys || []
            } else {
              arr = (p.ik || []).find((x: any) => x.id === curIk?.id)?.keys || []
            }
            const k = arr.find((x: any) => x.frame === dest.frame)
            if (!k) return
            k.bez = b
            if (b) k.ease = 'linear'
          })
          return (
            <>
              <div className="section-title" style={{ marginTop: 14 }}>
                曲线 · 第 {prev.frame + 1}→{dest.frame + 1} 帧
              </div>
              {usable.length > 1 && (
                <div className="chips" style={{ marginBottom: 6 }}>
                  {usable.map((o) => (
                    <button key={o.id} className={`chip sm ${o.id === owner.id ? 'on' : ''}`}
                            onClick={() => setCurveOwner(o.id)}>{o.label}</button>
                  ))}
                </div>
              )}
              <CurveEditor bez={bez} onChange={setBez} />
              <div className="chips mt8">
                {([
                  ['线性', null],
                  ['缓入出', [0.42, 0, 0.58, 1]],
                  ['缓入', [0.42, 0, 1, 1]],
                  ['缓出', [0, 0, 0.58, 1]],
                  ['回弹', [0.34, 1.56, 0.64, 1]],
                ] as [string, number[] | null][]).map(([label, v]) => (
                  <button key={label} className={`chip sm ${!dest.bez && v === null ? 'on' : ''}`}
                          onClick={() => setBez(v)}>{label}</button>
                ))}
              </div>
              <div className="hint">
                拖两个黄点改曲线；y 可以拖出框外做过冲。曲线记在**目标帧**上，
                表示「走完这一段时怎么减速」。
              </div>
            </>
          )
        })()}

        {(() => {
          const ik = (proj?.ik || []).find((v: any) => v.id === selIk)
          if (!ik) return null
          const wv = sample(ik.keys || [], frame).w ?? 1
          return (
            <>
              <div className="label" style={{ marginTop: 10 }}>
                IK 权重 {Math.round(wv * 100)}%（0 = 跟 FK，1 = 跟目标）
              </div>
              <input type="range" min={0} max={1} step={0.01} value={wv}
                     style={{ width: '100%' }}
                     onChange={(e) => upsertIkWeight(ik.id, +e.target.value)} />
              <div className="hint">
                半权重是「手搭上去但没完全跟」的过渡，可以逐帧打点做权重动画；
                蓝菱形是 pole，决定肘/膝往哪边弯。
              </div>
            </>
          )
        })()}

        {mode === 'weight' && track?.skin && (
          <>
            <div className="section-title" style={{ marginTop: 14 }}>权重笔刷</div>
            {!wView ? (
              <div className="hint">正在读取权重…</div>
            ) : (
              <>
                <div className="label">正在刷的骨骼</div>
                <div className="chips">
                  {wView.bones.map((b: any) => (
                    <button key={b.id} className={`chip sm ${b.id === wBone ? 'on' : ''}`}
                            onClick={() => setWBone(b.id)}>{b.name}</button>
                  ))}
                </div>
                <div className="label" style={{ marginTop: 8 }}>方向</div>
                <div className="seg">
                  <button className={wDir > 0 ? 'on' : ''} onClick={() => setWDir(1)}>刷入</button>
                  <button className={wDir < 0 ? 'on' : ''} onClick={() => setWDir(-1)}>刷出</button>
                </div>
                <div className="label" style={{ marginTop: 8 }}>半径 {wRadius} 格</div>
                <input type="range" min={1} max={5} value={wRadius} style={{ width: '100%' }}
                       onChange={(e) => setWRadius(+e.target.value)} />
                <div className="label" style={{ marginTop: 6 }}>
                  强度 {Math.round(wStrength * 100)}%
                </div>
                <input type="range" min={0.05} max={1} step={0.05} value={wStrength}
                       style={{ width: '100%' }}
                       onChange={(e) => setWStrength(+e.target.value)} />
                <div className="hint" style={{ marginTop: 6 }}>
                  已手刷 {wView.painted || 0} 个顶点（红色 = 完全不跟该骨骼，蓝色 = 完全跟）。
                  网格密度改了要重刷——顶点下标会错位。
                </div>
                <button className="btn btn-ghost btn-block" style={{ marginTop: 6 }}
                        onClick={() => {
                          update((p) => {
                            const t = p.tracks.find((v: any) => v.id === track.id)
                            if (t?.skin) t.skin.paint = {}
                          })
                          setWView({ ...wView, painted: 0 })
                        }}>清除全部手刷（回到自动权重）</button>
              </>
            )}
          </>
        )}

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
            <div className="label" style={{ marginTop: 10 }}>骨骼蒙皮</div>
            <label className="hint" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={!!track.skin}
                onChange={(e) => update((p) => {
                  const t = p.tracks.find((v: any) => v.id === track.id)
                  if (!e.target.checked) {
                    t.skin = null
                    return
                  }
                  // 默认把「绑定的骨骼 + 它的父骨骼」放进来：一根骨骼带不动形状，
                  // 至少两节才能看出弯曲
                  const bind = p.bones.find((x: any) => x.id === t.bone)
                  const ids = [t.bone]
                  if (bind?.parent) ids.push(bind.parent)
                  t.skin = { bones: ids, falloff: 60, grid: 8 }
                })}
              />
              自动权重（按顶点到骨段的距离）
            </label>

            {track.skin && (
              <>
                <div className="label" style={{ marginTop: 6 }}>参与骨骼</div>
                <div className="chips">
                  {bones.map((b: any) => {
                    const on = track.skin.bones.includes(b.id)
                    return (
                      <button key={b.id} className={`chip sm ${on ? 'on' : ''}`}
                              onClick={() => update((p) => {
                                const t = p.tracks.find((v: any) => v.id === track.id)
                                const set = new Set<string>(t.skin.bones)
                                if (set.has(b.id)) set.delete(b.id)
                                else set.add(b.id)
                                t.skin = { ...t.skin, bones: [...set] }
                              })}>
                        {b.name}
                      </button>
                    )
                  })}
                </div>
                <div className="label" style={{ marginTop: 8 }}>
                  影响半径 {Math.round(track.skin.falloff)}px
                </div>
                <input type="range" min={10} max={200} value={track.skin.falloff}
                       style={{ width: '100%' }}
                       onChange={(e) => update((p) => {
                         const t = p.tracks.find((v: any) => v.id === track.id)
                         t.skin = { ...t.skin, falloff: +e.target.value }
                       })} />
                <div className="label" style={{ marginTop: 6 }}>网格 {track.skin.grid}×{track.skin.grid}</div>
                <input type="range" min={4} max={16} value={track.skin.grid}
                       style={{ width: '100%' }}
                       onChange={(e) => update((p) => {
                         const t = p.tracks.find((v: any) => v.id === track.id)
                         t.skin = { ...t.skin, grid: +e.target.value }
                       })} />
                <div className="hint">
                  网格越密越贴合骨骼，但片间折痕也越明显；12 以上一般够用。
                </div>
              </>
            )}

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
