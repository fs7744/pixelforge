import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from 'react'

export type Side = 'left' | 'right'
type Widths = Record<Side, number>

const KEY = 'pf.panel-widths'
// 生成参数那一栏控件多（模板列表 + 画风 chips + 尺寸/步数），320 太窄，默认给到 360
const DEFAULT: Widths = { left: 360, right: 320 }
const MIN: Widths = { left: 240, right: 230 }
const MAX: Widths = { left: 760, right: 640 }

function clamp(side: Side, v: number) {
  return Math.min(MAX[side], Math.max(MIN[side], Math.round(v)))
}

function load(): Widths {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || 'null')
    if (raw && typeof raw.left === 'number' && typeof raw.right === 'number') {
      return { left: clamp('left', raw.left), right: clamp('right', raw.right) }
    }
  } catch {
    // 存了坏值就当没有，走默认
  }
  return { ...DEFAULT }
}

type Ctx = Widths & {
  setWidth: (side: Side, v: number) => void
  reset: (side: Side) => void
  dragging: Side | null
  setDragging: (side: Side | null) => void
}

const SplitCtx = createContext<Ctx | null>(null)

export function SplitProvider({ children }: { children: ReactNode }) {
  const [w, setW] = useState<Widths>(load)
  const [dragging, setDragging] = useState<Side | null>(null)

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(w))
    } catch {
      // 隐私模式下写不进去，忽略
    }
  }, [w])

  const setWidth = useCallback((side: Side, v: number) => {
    setW((s) => (s[side] === clamp(side, v) ? s : { ...s, [side]: clamp(side, v) }))
  }, [])
  const reset = useCallback((side: Side) => setW((s) => ({ ...s, [side]: DEFAULT[side] })), [])

  return (
    <SplitCtx.Provider value={{ ...w, setWidth, reset, dragging, setDragging }}>
      {children}
    </SplitCtx.Provider>
  )
}

export function useSplit(): Ctx {
  const c = useContext(SplitCtx)
  if (!c) throw new Error('useSplit 必须放在 SplitProvider 内')
  return c
}

/** 分栏拖拽条。左侧贴左栏右边缘，右侧贴右栏左边缘；双击复位。 */
export function Splitter({ side }: { side: Side }) {
  const { left, right, setWidth, reset, dragging, setDragging } = useSplit()
  const width = side === 'left' ? left : right
  const start = useRef({ x: 0, w: 0 })
  // 用 ref 而不是 dragging state 判断拖拽中：setDragging 是异步的，
  // 快速单击时第一个 pointermove/up 会读到旧值，导致 dragging 残留、全局光标卡住
  const active = useRef(false)

  const onDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    // 只响应主键；阻止默认避免拖拽时选中文本
    if (e.button !== 0) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    start.current = { x: e.clientX, w: width }
    active.current = true
    setDragging(side)
  }

  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!active.current) return
    const dx = e.clientX - start.current.x
    setWidth(side, side === 'left' ? start.current.w + dx : start.current.w - dx)
  }

  const onUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!active.current) return
    active.current = false
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
    setDragging(null)
  }

  const onKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 48 : 16
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      setWidth(side, width - step)
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      setWidth(side, width + step)
    } else if (e.key === 'Home') {
      e.preventDefault()
      reset(side)
    }
  }

  return (
    <div
      className={`split split-${side} ${dragging === side ? 'on' : ''}`}
      role="separator"
      aria-orientation="vertical"
      aria-label={side === 'left' ? '调整左栏宽度' : '调整右栏宽度'}
      aria-valuenow={width}
      aria-valuemin={MIN[side]}
      aria-valuemax={MAX[side]}
      tabIndex={0}
      title="拖动调整宽度 · 双击恢复默认"
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onDoubleClick={() => reset(side)}
      onKeyDown={onKey}
    >
      <i />
    </div>
  )
}
