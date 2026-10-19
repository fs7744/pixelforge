import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

export type TaskProgress = {
  type: 'progress' | 'done' | 'error' | 'cancelled'
  step: number
  total: number
  message: string
  elapsed: number
  error?: string
}

/**
 * 任务进度订阅（SSE）+ 取消（移植自 FrameBaker 的任务队列 + WS 广播，
 * 换成浏览器原生 EventSource，无需额外依赖）。
 *
 * 用法：发起长任务前 `start(taskId)`（taskId 同时传给后端），
 * 组件里渲染 <TaskProgressBar progress={progress} onCancel={cancel} />。
 * 终态事件（done/error/cancelled）到达后自动断开。
 */
export function useTaskProgress() {
  const [taskId, setTaskId] = useState('')
  const [progress, setProgress] = useState<TaskProgress | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const stop = useCallback(() => {
    esRef.current?.close()
    esRef.current = null
  }, [])

  const start = useCallback((id: string) => {
    stop()
    setTaskId(id)
    setProgress(null)
    const es = new EventSource(`/api/tasks/${id}/stream`)
    es.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data) as TaskProgress
        setProgress(d)
        if (d.type !== 'progress') es.close()
      } catch {
        /* 心跳或坏行：忽略 */
      }
    }
    es.onerror = () => es.close() // 断线/终态后重连没有意义，任务状态由请求响应兜底
    esRef.current = es
  }, [stop])

  const cancel = useCallback(async () => {
    if (!taskId) return
    try {
      await api.cancelTask(taskId)
    } catch {
      /* 取消失败不挡 UI：请求侧出错时前端一样会收到错误响应 */
    }
  }, [taskId])

  useEffect(() => stop, [stop])
  return { taskId, progress, start, stop, cancel }
}

/** 生成一个任务 id（后端 register 幂等，前端随机即可） */
export function newTaskId(): string {
  return (crypto.randomUUID?.() ?? Math.random().toString(16)).replace(/-/g, '').slice(0, 12)
}

function fmtElapsed(s: number): string {
  const m = Math.floor(s / 60)
  return m > 0 ? `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}` : `${Math.floor(s)}s`
}

/** 进度条 + 取消按钮：message 由后端逐帧/逐步上报，直接展示 */
export function TaskProgressBar({
  progress,
  onCancel,
}: {
  progress: TaskProgress | null
  onCancel: () => void
}) {
  if (!progress || progress.type === 'done') return null
  const pct = progress.total > 0 ? Math.min(100, Math.round((progress.step / progress.total) * 100)) : null
  return (
    <div className="task-progress">
      <div className="row between">
        <span className="hint" style={{ margin: 0 }}>
          {progress.message || '处理中'}
          {pct != null ? ` · ${pct}%` : ''} · 已用 {fmtElapsed(progress.elapsed)}
        </span>
        <button className="btn btn-ghost" style={{ padding: '2px 10px', fontSize: 12 }} onClick={onCancel}>
          取消
        </button>
      </div>
      <div className="bar">
        <i style={{ width: `${pct ?? 8}%` }} className={pct == null ? 'indeterminate' : ''} />
      </div>
    </div>
  )
}
