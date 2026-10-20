import { useEffect, useRef, useState } from 'react'

export type PartDef = { id: string; name: string; color: string }

type Props = {
  /** 角色图地址（已去背最好） */
  src: string
  parts: PartDef[]
  activeId: string
  onActive: (id: string) => void
  /** 涂完把色标图回调出去（Blob，PNG） */
  onTag: (blob: Blob | null) => void
}

/**
 * 部位涂色板：在角色图上直接涂，每种部位一个纯色。
 *
 * 为什么不自动拆：头/身/四肢的语义本来就该由人给；自动方案猜出来的部件
 * 在关节处照样会撕，用户还得回来手改，不如把笔刷给他。
 *
 * canvas 的逻辑分辨率 = 图像自然尺寸，显示尺寸交给 CSS——这样涂的坐标和
 * 提交给服务端的像素一一对应，缩放窗口不会让涂歪。
 */
export default function PartPainter(props: Props) {
  const { src, parts, activeId, onActive, onTag } = props
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)
  const ref = useRef<HTMLCanvasElement>(null)
  const [brush, setBrush] = useState(18)
  const [mode, setMode] = useState<'draw' | 'box'>('draw')
  const painting = useRef(false)
  const last = useRef<{ x: number; y: number } | null>(null)
  // 框选模式：起点与当前点。
  // 当前点同时存 ref 和 state：ref 给 onUp 用（state 是异步的，
  // onUp 里读到的是上一次渲染的值，框会差一截），state 只给预览用。
  const boxStart = useRef<{ x: number; y: number } | null>(null)
  const boxNowRef = useRef<{ x: number; y: number } | null>(null)
  const [boxNow, setBoxNow] = useState<{ x: number; y: number } | null>(null)

  // 换图 / 换尺寸时清空涂色：旧色标和新图对不上，会切出莫名其妙的部件
  useEffect(() => {
    const cv = ref.current
    if (!cv || !size) return
    cv.width = size.w
    cv.height = size.h
    const ctx = cv.getContext('2d')
    if (ctx) ctx.clearRect(0, 0, cv.width, cv.height)
    onTag(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, size?.w, size?.h])

  const toLocal = (e: React.PointerEvent) => {
    const cv = ref.current!
    const r = cv.getBoundingClientRect()
    // 显示尺寸 ≠ 自然尺寸时必须按比例换算，否则缩放窗口后涂歪
    return {
      x: ((e.clientX - r.left) / r.width) * cv.width,
      y: ((e.clientY - r.top) / r.height) * cv.height,
    }
  }

  const stroke = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const cv = ref.current
    const ctx = cv?.getContext('2d')
    if (!cv || !ctx) return
    const part = parts.find((p) => p.id === activeId)
    if (!part) return
    ctx.strokeStyle = part.color
    ctx.fillStyle = part.color
    ctx.lineWidth = brush
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(from.x, from.y)
    ctx.lineTo(to.x, to.y)
    ctx.stroke()
    // 单击也要落一个点，否则点一下等于没涂
    ctx.beginPath()
    ctx.arc(to.x, to.y, brush / 2, 0, Math.PI * 2)
    ctx.fill()
  }

  const fillBox = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const cv = ref.current
    const ctx = cv?.getContext('2d')
    if (!cv || !ctx) return
    const part = parts.find((p) => p.id === activeId)
    if (!part) return
    ctx.fillStyle = part.color
    const x = Math.min(from.x, to.x)
    const y = Math.min(from.y, to.y)
    ctx.fillRect(x, y, Math.abs(to.x - from.x), Math.abs(to.y - from.y))
  }

  const onDown = (e: React.PointerEvent) => {
    painting.current = true
    const p = toLocal(e)
    last.current = p
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    if (mode === 'box') {
      boxStart.current = p
      boxNowRef.current = p
      setBoxNow(p)
    } else {
      stroke(p, p)
    }
  }
  const onMove = (e: React.PointerEvent) => {
    if (!painting.current) return
    const p = toLocal(e)
    if (mode === 'box') {
      boxNowRef.current = p
      setBoxNow(p)
      return
    }
    if (last.current) stroke(last.current, p)
    last.current = p
  }
  const onUp = () => {
    painting.current = false
    last.current = null
    if (mode === 'box') {
      const from = boxStart.current
      const to = boxNowRef.current
      boxStart.current = null
      boxNowRef.current = null
      setBoxNow(null)
      if (from && to) fillBox(from, to)
    }
    ref.current?.toBlob((b) => onTag(b), 'image/png')
  }

  const clear = () => {
    const cv = ref.current
    const ctx = cv?.getContext('2d')
    if (cv && ctx) ctx.clearRect(0, 0, cv.width, cv.height)
    onTag(null)
  }

  return (
    <div>
      <div className="chips" style={{ marginBottom: 8 }}>
        {parts.map((p) => (
          <button
            key={p.id}
            className={`chip sm ${p.id === activeId ? 'on' : ''}`}
            onClick={() => onActive(p.id)}
          >
            <i className="dot" style={{ background: p.color }} /> {p.name}
          </button>
        ))}
      </div>
      <div className="row" style={{ gap: 6, marginBottom: 6 }}>
        <div className="seg">
          <button className={mode === 'draw' ? 'on' : ''} onClick={() => setMode('draw')}>
            画笔
          </button>
          <button className={mode === 'box' ? 'on' : ''} onClick={() => setMode('box')}>
            框选
          </button>
        </div>
        {mode === 'draw' ? (
          <>
            <span className="hint">笔刷 {brush}px</span>
            <input
              type="range" min={4} max={60} value={brush}
              style={{ flex: 1 }}
              onChange={(e) => setBrush(+e.target.value)}
            />
          </>
        ) : (
          <span className="hint">拖一个矩形，整块填色（比描边省事）</span>
        )}
        <button className="btn btn-ghost" onClick={clear}>清除</button>
      </div>
      <div className="painter">
        <img
          src={src}
          alt="角色"
          onLoad={(e) => {
            const el = e.currentTarget
            if (el.naturalWidth) setSize({ w: el.naturalWidth, h: el.naturalHeight })
          }}
        />
        <canvas
          ref={ref}
          style={{ cursor: mode === 'box' ? 'cell' : 'crosshair' }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        />
        {mode === 'box' && boxNow && boxStart.current && (
          <div
            className="painter-box"
            style={{
              left: `${Math.min(boxStart.current.x, boxNow.x) / (size?.w || 1) * 100}%`,
              top: `${Math.min(boxStart.current.y, boxNow.y) / (size?.h || 1) * 100}%`,
              width: `${Math.abs(boxNow.x - boxStart.current.x) / (size?.w || 1) * 100}%`,
              height: `${Math.abs(boxNow.y - boxStart.current.y) / (size?.h || 1) * 100}%`,
              borderColor: parts.find((p) => p.id === activeId)?.color,
            }}
          />
        )}
      </div>
      <div className="hint">
        先选颜色，再涂。没涂到的部位不会生成图（结果里会标出来）。
        <br />
        框选模式拖一个矩形就能整块填色——描边容易漏，
        漏掉的像素切不出来，部件就会缺一块。
      </div>
    </div>
  )
}