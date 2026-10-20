"""序列帧轨道的循环诊断：接缝跳不跳、段内稳不稳。

为什么在编辑器里做
------------------
工作流侧已经有等价能力（pipelines/metrics.py 的 seam_ratio /
pipelines/review.py 的接缝描边），但那是在**资产级**：跑完拼表才给结论。
编辑器里的序列帧轨道是拼好之后才导入的，用户看到的是「循环播放时抖一下」，
却不知道是第几格接回首格跳的、更不知道跳多少。

所以这里补一层**轨道级**诊断，直接对着轨道正在播的那几格算：
- 接缝差（末格 → 首格）：和普通相邻帧差比一比，就知道是不是「顿一下」
- 段内最大的帧间差 + 发生在哪一格：定位「姿势跑飞」的那一格

口径与 pipelines/metrics.py 完全一致（RGB 的平均绝对差），这样同一个素材
在拼表阶段和编辑器里算出来的数字能对上，不是两套标准。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _mad(x: np.ndarray, y: np.ndarray) -> float:
    """平均绝对差（与 pipelines/metrics._mad 同款）。"""
    return float(np.abs(x - y).mean())


def _rgb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGBA")).astype(np.float32)[:, :, :3]


def seam_report(frames: Sequence[Image.Image]) -> Dict[str, Any]:
    """一组帧的循环接缝诊断。

    frames 是**当前循环段**的格子（按播放顺序）。少于 2 格没有接缝可言。
    """
    if not frames:
        return {"frames": 0, "verdict": [], "note": "没有格子"}
    if len(frames) < 2:
        return {"frames": len(frames), "verdict": [],
                "note": "只有一格，没有循环接缝"}

    rgbs = [_rgb(f) for f in frames]
    steps = [_mad(rgbs[i], rgbs[i + 1]) for i in range(len(rgbs) - 1)]
    # 基准用**中位数**：编辑器里的循环段通常只有 3–8 格，
    # 一个极端值（一帧姿势跑飞）就能把 mean 拉高一倍，接缝反而显得正常。
    # 中位数不受离群值影响，「接缝比普通帧间差大多少」才问得准。
    step_med = float(np.median(steps))
    step_mean = float(np.mean(steps))
    step_max = float(np.max(steps))
    step_max_at = int(np.argmax(steps))
    # 接缝：末格 → 首格（循环播放时真正接上的一对）
    seam = _mad(rgbs[-1], rgbs[0])
    ratio = seam / max(step_med, 1e-6)

    res: Dict[str, Any] = {
        "frames": len(frames),
        "seam_first_last": round(seam, 3),
        "step_mean": round(step_mean, 3),
        "step_median": round(step_med, 3),
        "step_max": round(step_max, 3),
        "step_max_at": step_max_at,
        "seam_ratio": round(ratio, 3),
    }

    verdict: List[str] = []
    if ratio > 2.0:
        verdict.append(
            f"首尾衔接差（接缝是普通帧间差的 {ratio:.1f} 倍）：循环播放会顿一下，"
            "建议让首末格姿势一致，或去掉与首格重复的末格")
    elif ratio < 1.0:
        verdict.append(f"首尾衔接良好（接缝比普通相邻帧还紧，{ratio:.2f}）")
    # 「哪一格跑飞」：拿**除它自己以外**的帧间差做基准。
    # 直接用中位数不行 —— 4 格循环只有 3 个差值，中位数会被离群值本身
    # 拉到中间（[5,55,5] 的中位数是 55 而不是 5），跑飞的那格反倒不显眼。
    # 而且跑飞的一格会同时污染它**前后两个**差值，剔掉一个不够。
    # 所以这里换成「这一格与**所有其它格**的最小差异」：
    # 正常格至少和邻居挨得很近（差异小），跑飞的格离谁都远。
    # 基准取「所有格的这个值」的中位数。
    near: List[float] = []
    for i in range(len(rgbs)):
        d = min((_mad(rgbs[i], rgbs[j]) for j in range(len(rgbs)) if j != i),
                default=0.0)
        near.append(d)
    near_med = float(np.median(near)) if near else 0.0
    near_max = float(np.max(near)) if near else 0.0
    near_at = int(np.argmax(near)) if near else 0
    res["near_median"] = round(near_med, 3)
    res["near_max"] = round(near_max, 3)
    res["near_max_at"] = near_at
    if near_max > 3 * max(near_med, 1e-6):
        verdict.append(
            f"第 {near_at + 1} 格与其它格都差很多：那一格可能姿势跑飞")
    res["verdict"] = verdict
    return res


def cut_range(sheet: Image.Image, seq: Dict[str, Any]) -> Tuple[List[Image.Image], int, int]:
    """从精灵表里裁出当前循环段的格子。

    返回（格子列表, 起始格, 结束格）。段外格子不参与接缝计算 ——
    用户选了第 3–6 格当循环，接缝就该是「第 6 格 → 第 3 格」。
    """
    from . import sequence as seq_mod

    boxes = seq_mod.cell_boxes(sheet, seq)
    if not boxes:
        return ([], 0, 0)
    start, stop = seq_mod.range_of(seq, len(boxes))
    cells = [sheet.convert("RGBA").crop(boxes[i]) for i in range(start, stop + 1)]
    return (cells, start, stop)
