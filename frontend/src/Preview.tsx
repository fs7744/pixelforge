import { useCallback, useEffect, useRef, useState } from 'react'

export type PreviewItem = { url: string; name: string }

const EVT = 'pf:preview'

/** 任意视图里点图片都能唤起全屏预览（跨视图共享，不用层层传 props）。 */
export function openPreview(url: string, name: string) {
  window.dispatchEvent(new CustomEvent<PreviewItem>(EVT, { detail: { url, name } }))
}

const MIN = 0.2
const MAX = 16

export default function PreviewOverlay() {
  const [item, setItem] = useState<PreviewItem | null>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [size, setSize] = useState<{ w: number; h: number } | null>(null)
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null)
  const maskRef = useRef<HTMLDivElement | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  const clampPan = (x: number, y: number, z: number) => {
    const stage = stageRef.current
    const img = imgRef.current
    if (!stage || !img) return { x, y }
    const maxX = Math.max(0, (img.offsetWidth * z - stage.clientWidth) / 2)
    const maxY = Math.max(0, (img.offsetHeight * z - stage.clientHeight) / 2)
    return { x: Math.min(maxX, Math.max(-maxX, x)), y: Math.min(maxY, Math.max(-maxY, y)) }
  }

  const close = useCallback(() => {
    setItem(null)
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [])

  useEffect(() => {
    const onOpen = (e: Event) => {
      const d = (e as CustomEvent<PreviewItem>).detail
      setItem(d)
      setZoom(1)
      setPan({ x: 0, y: 0 })
    }
    window.addEventListener(EVT, onOpen as EventListener)
    return () => window.removeEventListener(EVT, onOpen as EventListener)
  }, [])

  useEffect(() => {
    if (!item) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
      else if (e.key === '0') {
        setZoom(1)
        setPan({ x: 0, y: 0 })
      } else if (e.key === '=' || e.key === '+') setZoom((z) => Math.min(MAX, z * 1.25))
      else if (e.key === '-') setZoom((z) => Math.max(MIN, z / 1.25))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [item, close])

  // 滚轮：原生监听 + passive:false 才拦得住页面滚动（React 的 onWheel 是 passive）
  useEffect(() => {
    const el = maskRef.current
    if (!item || !el) return
    const onWheelNative = (e: WheelEvent) => {
      e.preventDefault()
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      setZoom((z) => {
        const next = Math.min(MAX, Math.max(MIN, z * factor))
        if (next <= 1) setPan({ x: 0, y: 0 })
        return next
      })
    }
    el.addEventListener('wheel', onWheelNative, { passive: false })
    return () => el.removeEventListener('wheel', onWheelNative)
  }, [item])

  if (!item) return null

  const onPointerDown = (e: React.PointerEvent) => {
    if (zoom <= 1) return
    drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }
    ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const next = { x: d.px + (e.clientX - d.x), y: d.py + (e.clientY - d.y) }
    setPan(clampPan(next.x, next.y, zoom))
  }
  const onPointerUp = () => {
    drag.current = null
  }

  return (
    <div
      className="pv-mask"
      ref={maskRef}
      onClick={(e) => {
        if (e.target === e.currentTarget) close()
      }}
    >
      <div className="pv-bar" onClick={(e) => e.stopPropagation()}>
        <span className="pv-name">{item.name}</span>
        {size && (
          <span className="pv-size">
            {size.w}×{size.h}
          </span>
        )}
        <button
          className="btn btn-ghost"
          style={{ height: 26 }}
          onClick={() => setZoom((z) => Math.max(MIN, z / 1.25))}
        >
          －
        </button>
        <span className="pv-zoom">{Math.round(zoom * 100)}%</span>
        <button
          className="btn btn-ghost"
          style={{ height: 26 }}
          onClick={() => setZoom((z) => Math.min(MAX, z * 1.25))}
        >
          ＋
        </button>
        <button
          className="btn btn-ghost"
          style={{ height: 26 }}
          onClick={() => {
            setZoom(1)
            setPan({ x: 0, y: 0 })
          }}
        >
          适应窗口
        </button>
        <button className="btn btn-ghost" style={{ height: 26 }} onClick={close}>
          关闭 ✕
        </button>
      </div>
      <div
        className="pv-stage"
        ref={stageRef}
        onClick={(e) => e.stopPropagation()}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        style={{ cursor: zoom > 1 ? 'grab' : 'default' }}
      >
        <img
          ref={imgRef}
          src={item.url}
          alt={item.name}
          draggable={false}
          onLoad={(e) => {
            const el = e.currentTarget
            setSize({ w: el.naturalWidth, h: el.naturalHeight })
          }}
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            // 放大后按最近邻渲染：像素素材放大才不会糊成一团
            imageRendering: zoom >= 2 ? 'pixelated' : 'auto',
          }}
        />
      </div>
      <div className="pv-tip" onClick={(e) => e.stopPropagation()}>
        滚轮缩放 · 拖拽平移 · 0 复位 · Esc 关闭
      </div>
    </div>
  )
}
