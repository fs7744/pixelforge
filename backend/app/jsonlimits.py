"""任意 JSON 入参的有界校验（移植自 FrameBaker 的 packages/shared/src/json.ts）。

为什么需要：FastAPI 的 ``Dict[str, Any]`` / pydantic 都不拦「形状合法但有害」的
JSON，而 Python 的 ``json.loads`` **默认接受** ``NaN`` / ``Infinity`` /
``-Infinity``。这些值会一路流进 run / settings 落盘：``json.dumps`` 默认会把
它们写成 ``NaN`` 字面量，产出严格 JSON 解析器（JS、Go、其它语言）拒绝的文件——
索引、run 状态都可能变成半个库读不动的雷。

FrameBaker 的做法是入口处统一校验：深度上限 + 节点数预算 + 数值必须有限。
这里保持同样的契约，错误信息带路径，方便定位是哪个字段。
"""
from __future__ import annotations

import math
from typing import Any

# 与 FrameBaker 的 DEFAULT_JSON_LIMITS 对齐：深度 32、节点 1600 万。
# pixelforge 的 params/settings 实际只有两三层，32 层对正常使用毫无影响，
# 但能把「十万层嵌套」这类形状炸弹挡在门外。
MAX_DEPTH = 32
MAX_NODES = 16_000_000


def validate_bounded(value: Any, *, max_depth: int = MAX_DEPTH,
                     max_nodes: int = MAX_NODES, path: str = "$") -> None:
    """校验任意 JSON 值的深度 / 节点数 / 数值有限性，不合法抛 ``ValueError``。

    字符串只查类型与代理项太贵（每个字符一遍），这里聚焦真正会伤人的两类：
    非有限数值与超限结构。循环引用不可能来自 HTTP 线格式（json.loads 的
    输入是字符串），但进程内调用方可能传自引用结构，用栈外集合兜住。
    """
    budget = [max_nodes]
    active: set = set()

    def visit(item: Any, p: str, depth: int) -> None:
        if budget[0] <= 0:
            raise ValueError(f"JSON 节点数超过上限 {max_nodes}")
        budget[0] -= 1
        if depth > max_depth:
            raise ValueError(f"JSON 嵌套超过 {max_depth} 层：{p}")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"JSON 数值必须有限（不允许 NaN/Infinity）：{p}")
            return
        if isinstance(item, str):
            return
        if isinstance(item, dict):
            marker = id(item)
            if marker in active:
                raise ValueError(f"JSON 包含循环引用：{p}")
            active.add(marker)
            for k, v in item.items():
                if not isinstance(k, str):
                    raise ValueError(f"JSON 对象键必须是字符串：{p}")
                visit(v, f"{p}.{k}", depth + 1)
            active.discard(marker)
            return
        if isinstance(item, (list, tuple)):
            marker = id(item)
            if marker in active:
                raise ValueError(f"JSON 包含循环引用：{p}")
            active.add(marker)
            for i, v in enumerate(item):
                visit(v, f"{p}[{i}]", depth + 1)
            active.discard(marker)
            return
        raise ValueError(f"不是可序列化的 JSON 值（{type(item).__name__}）：{p}")

    visit(value, path, 0)
