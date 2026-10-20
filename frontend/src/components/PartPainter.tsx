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
  const painting = useRef(false)
  const last = useRef<{ x: number; y: number } | null>(null)

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

  const onDown = (e: React.PointerEvent) => {
    painting.current = true
    const p = toLocal(e)
    last.current = p
    ;(e.target as Element).setPointerCapture?.(e.pointerId)
    stroke(p, p)
  }
  const onMove = (e: React.PointerEvent) => {
    if (!painting.current) return
    const p = toLocal(e)
    if (last.current) stroke(last.current, p)
    last.current = p
  }
  const onUp = () => {
    painting.current = false
    last.current = null
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
      <div className="row" style={{ gap: 8, marginBottom: 8 }}>
        <span className="hint">笔刷 {brush}px</span>
        <input
          type="range" min={4} max={60} value={brush}
          style={{ flex: 1 }}
          onChange={(e) => setBrush(+e.target.value)}
        />
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
          style={{ cursor: 'crosshair' }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        />
      </div>
      <div className="hint">
        在图上拖动涂出部位：先选颜色，再涂。没涂到的部位不会生成图（结果里会标出来）。
      </div>
    </div>
  )
}