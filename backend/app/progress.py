"""生成进度总线 + 取消标志（SSE 推送 + 任务取消）。

动机：AI 生成单次 40 步要 2-4 分钟，前端原来只有一个「生成中」转圈——
用户不知道卡在哪、要等多久、能不能停。FrameBaker 的做法是任务队列 +
WS 广播 progress；这里用更轻的进程内总线拿到同样的体验：

- 端点发起长任务前 ``register(task_id)``，任务结束 ``finish``；
- provider 的每步回调（diffusers ``callback_on_step_end``）把
  ``(step, total, message)`` 报进来——回调在工作线程里跑，总线内部加锁；
- 前端 ``EventSource`` 订阅 ``GET /api/tasks/{id}/stream`` 实时显示；
- 取消：``POST /api/tasks/{id}/cancel`` 置位，provider 下一步回调抛
  ``CancelledError``，diffusers 中止推理，不再白烧 GPU。

前提：单进程部署（start.sh 默认单 worker uvicorn）。多 worker 时各进程
总线不相通，届时把 ``_TASKS`` 换成 Redis 等共享存储即可，接口不用变。
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class CancelledError(Exception):
    """任务被用户取消。provider 的步进回调抛出它来中止推理。"""


@dataclass
class TaskState:
    task_id: str
    created_at: float = field(default_factory=time.time)
    step: int = 0
    total: int = 0
    message: str = ""
    done: bool = False
    error: str = ""
    cancelled: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)
    subscribers: list = field(default_factory=list)  # list[queue.Queue]

    def snapshot(self, ev_type: str = "progress") -> Dict[str, Any]:
        return {
            "type": ev_type,
            "taskId": self.task_id,
            "step": self.step,
            "total": self.total,
            "message": self.message,
            "elapsed": round(time.time() - self.created_at, 1),
        }


_TASKS: Dict[str, TaskState] = {}
_GUARD = threading.Lock()
# 终态任务保留时长：SSE 消费者可能还没读到最后一个事件就断线重连
_TASK_TTL = 300


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def register(task_id: Optional[str] = None) -> str:
    """注册任务（幂等：重复 register 同一 id 只重置计数不清订阅者）。"""
    tid = task_id or new_task_id()
    with _GUARD:
        st = _TASKS.get(tid)
        if st is None:
            st = TaskState(task_id=tid)
            _TASKS[tid] = st
        return tid


def get(task_id: str) -> Optional[TaskState]:
    with _GUARD:
        return _TASKS.get(task_id)


def report(task_id: Optional[str], step: int, total: int, message: str = "") -> None:
    """上报一步进度（工作线程调用）。任务未注册时静默忽略——
    不注册任务的任务（测试、内部调用）不该被遥测拖累。"""
    if not task_id:
        return
    with _GUARD:
        st = _TASKS.get(task_id)
        if st is None or st.done:
            return
        st.step, st.total, st.message = int(step), int(total), message
        ev = st.snapshot()
        for q in list(st.subscribers):
            q.put(ev)


def check_cancel(task_id: Optional[str]) -> None:
    """已取消则抛 CancelledError（provider 步进回调里调用）。"""
    if not task_id:
        return
    st = get(task_id)
    if st is not None and st.cancel_event.is_set():
        raise CancelledError(f"任务已取消：{task_id}")


def cancel(task_id: str) -> bool:
    """请求取消。返回 False 表示任务不存在或已结束。"""
    st = get(task_id)
    if st is None or st.done:
        return False
    st.cancel_event.set()
    st.cancelled = True
    with _GUARD:
        ev = {"type": "cancelled", "taskId": st.task_id, "message": "已取消",
              "elapsed": round(time.time() - st.created_at, 1)}
        for q in list(st.subscribers):
            q.put(ev)
    return True


def finish(task_id: Optional[str], ok: bool = True, error: str = "") -> None:
    """标记任务结束，向订阅者发终态事件。"""
    if not task_id:
        return
    st = get(task_id)
    if st is None:
        return
    st.done = True
    st.error = error
    ev_type = "done" if ok else "error"
    with _GUARD:
        ev = st.snapshot(ev_type)
        if error:
            ev["error"] = error
        for q in list(st.subscribers):
            q.put(ev)


def subscribe(task_id: str) -> Optional["queue.Queue"]:
    """订阅任务事件流。任务不存在返回 None；已终态则补发终态事件——
    否则迟到的订阅者（任务已结束才拉流）永远等不到断开信号，SSE 连接
    会一直挂着。"""
    st = get(task_id)
    if st is None:
        return None
    q: "queue.Queue" = queue.Queue()
    with _GUARD:
        st.subscribers.append(q)
        # 立即补发当前进度：订阅往往比第一个 report 晚，别让前端空窗
        q.put(st.snapshot())
        if st.done:
            ev = st.snapshot("done" if not st.error else "error")
            if st.error:
                ev["error"] = st.error
            q.put(ev)
    return q


def unsubscribe(task_id: str, q: "queue.Queue") -> None:
    st = get(task_id)
    if st is None:
        return
    with _GUARD:
        if q in st.subscribers:
            st.subscribers.remove(q)


def sweep(ttl: int = _TASK_TTL) -> int:
    """清理终态超过 TTL 的任务（启动时与 SSE 断开时调用，防字典无限涨）。"""
    now = time.time()
    removed = 0
    with _GUARD:
        for tid in [t for t, s in _TASKS.items()
                    if s.done and now - s.created_at > ttl]:
            _TASKS.pop(tid, None)
            removed += 1
    return removed


def make_step_callback(task_id: Optional[str], total: int, label: str = ""):
    """造一个 diffusers ``callback_on_step_end``：上报进度 + 检查取消。

    返回的函数签名 ``(pipe, i, t, kwargs) -> kwargs``；取消时抛
    CancelledError（diffusers 会当作推理失败向上传播，请求随之结束）。
    total = 本次推理步数；label 用于多帧编排，如「第 2/4 帧」。
    """
    def _cb(pipe, i, t, callback_kwargs):
        check_cancel(task_id)
        prefix = f"{label} · " if label else ""
        report(task_id, i + 1, int(total), f"{prefix}第 {i + 1}/{int(total)} 步")
        return callback_kwargs

    return _cb
