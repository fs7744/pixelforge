"""精灵表审查图：把「帧之间稳不稳」从数字变成一眼能看的图。

移植自 h3-character-studio 的 extraction.review：指标（seam_ratio /
step_max）回答「哪一帧有问题」，但用户最终是用眼睛验收的。这张图把
所有帧按网格排开，每格下方标注序号与到下一帧的平均差异，首尾帧
（循环接缝）描边标出——接缝跳不跳、哪帧姿势跑飞，放大看一眼就知道。

与 metrics 的关系：数字给 metrics_verdict，图给肉眼，两者互补。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 深色底 + 白字 + 琥珀色接缝标：与编辑器类工具的审查视图一致
_BG = (26, 28, 34)
_FG = (235, 238, 245)
_DIM = (140, 146, 160)
_SEAM = (255, 213, 0)
_STUCK = (255, 90, 90)      # 卡住/跑飞的帧
_LABEL_H = 26
_PAD = 6
_COLS_MAX = 8


def contact_sheet(
    frames: Sequence[Image.Image],
    *,
    cols: int = 0,
    label_h: int = _LABEL_H,
    pad: int = _PAD,
) -> Image.Image:
    """把一组帧排成审查网格。空序列抛 ValueError。"""
    if not frames:
        raise ValueError("没有可排的帧")
    n = len(frames)
    cols = max(1, min(int(cols) or _COLS_MAX, n))
    rows = (n + cols - 1) // cols
    cw = max(f.width for f in frames)
    ch = max(f.height for f in frames)

    sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + label_h + pad) + pad), _BG)
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    # 逐帧差异（与 metrics 同款 MAD，但只看 RGB，避免透明区干扰）
    arrs = [np.asarray(f.convert("RGBA")).astype(np.float32) for f in frames]
    rgbs = [a[:, :, :3] for a in arrs]
    diffs: List[Optional[float]] = [None] * n
    for i in range(n - 1):
        diffs[i] = float(np.abs(rgbs[i] - rgbs[i + 1]).mean())

    # 异常帧检测：某一帧与前后两帧的相似度明显高于其它帧 → 姿势卡住/跑飞。
    #
    # 注意：**不能只看「到下一帧的差异」**——跑飞的那帧往往与前后帧差得远，
    # 于是它的 Δ 反而最小（看起来"没动"）。所以这里比的是
    # 「本帧与前后帧的平均差异」是否**明显低于**整段的典型值：
    # 一帧静止在循环里，其它帧都在动 → 它就是卡住的那帧。
    #
    # 卡住帧的判据：**连续两个极小的 Δ 夹着的那一帧**。
    #
    # 为什么是这个形状：一帧卡住（复制了上一帧）时，
    # 「上一帧→它」和「它→下一帧」两个 Δ 都≈0，序列里出现 **0, 0**；
    # 卡住的那一帧就在这两个 0 中间。
    #
    # 试过其它判据都不对：
    # · 只看「Δ 小于中位数一半」→ 会标 3 帧（卡住帧 + 两个邻居）；
    # · 只看「局部极小」→ 一帧都标不出（卡住时是两个连续的 0，不构成极小）。
    med = float(np.median([d for d in diffs if d is not None])) if n > 1 else 0.0
    still: List[bool] = [False] * n
    if n > 2 and med > 0.5:
        thr = med * 0.5
        for i in range(n - 1):
            # diffs[i] 与 diffs[i+1] 都极小 → 第 i+1 帧（0 基）卡住
            if diffs[i] < thr and diffs[i + 1] < thr:
                still[i + 1] = True

    for i, f in enumerate(frames):
        c, r = i % cols, i // cols
        x = pad + c * (cw + pad)
        y = pad + r * (ch + label_h + pad)
        rgba = f if f.mode == "RGBA" else f.convert("RGBA")
        sheet.paste(rgba, (x + (cw - f.width) // 2, y + (ch - f.height) // 2), rgba)
        # 首尾帧（循环接缝的两端）描琥珀边
        if i == 0 or i == n - 1:
            d.rectangle([x - 1, y - 1, x + cw, y + ch], outline=_SEAM, width=2)
        # 卡住/跑飞的帧描红边——这是最需要人看一眼的地方
        if still[i]:
            d.rectangle([x - 2, y - 2, x + cw + 1, y + ch + 1], outline=_STUCK, width=2)
        # 标签：序号 + 到下一帧的差异 + 接缝标记
        parts = [f"{i + 1}"]
        if diffs[i] is not None:
            parts.append(f"Δ{diffs[i]:.1f}")
        if i == n - 1:
            parts.append("接缝→1")
        if still[i]:
            parts.append("⚠ 卡住")
        color = _STUCK if still[i] else (_FG if i < n - 1 else _SEAM)
        d.text((x + 2, y + ch + 5), "  ".join(parts), fill=color, font=font)

    return sheet
