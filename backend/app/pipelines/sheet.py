"""Sprite Sheet 拆分与合成 + 引擎索引 JSON。

透明像素的 RGB 恒为 0（alpha_composite 前统一清零）：透明区携带底色
（绿幕残色、抠图前的背景）时，引擎做双线性采样或预乘 alpha 会把它们
渲成一层色晕，图集单元格之间也会互相串色。做法移植自
h3-character-studio 的 core.process（a[~visible, :3] = 0）。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image


def zero_transparent_rgb(img: Image.Image) -> Image.Image:
    """把全透明像素的 RGB 置 0（alpha 不动）。没有透明像素时原样返回。"""
    rgba = img if img.mode == "RGBA" else img.convert("RGBA")
    arr = np.asarray(rgba)
    if not (arr[:, :, 3] == 0).any():
        return rgba
    arr = arr.copy()
    arr[arr[:, :, 3] == 0, :3] = 0
    return Image.fromarray(arr, "RGBA")


def split_sheet(img: Image.Image, cols: int, rows: int) -> Tuple[List[Image.Image], dict]:
    """按行列均匀拆分。列数为每行帧数，行数为动作组数。"""
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    fw = img.width // cols
    fh = img.height // rows
    frames: List[Image.Image] = []
    index = []
    for r in range(rows):
        for c in range(cols):
            box = (c * fw, r * fh, (c + 1) * fw, (r + 1) * fh)
            frames.append(img.crop(box))
            index.append({"i": r * cols + c, "x": c * fw, "y": r * fh, "w": fw, "h": fh,
                          "t": round((r * cols + c) * 0.1, 3)})
    meta = {
        "version": "1.0",
        "frame_size": {"w": fw, "h": fh},
        "sheet_size": {"w": img.width, "h": img.height},
        "grid": {"cols": cols, "rows": rows},
        "frames": index,
    }
    return frames, meta


def compose_sheet(frames: List[Image.Image], cols: int) -> Tuple[Image.Image, dict]:
    if not frames:
        raise ValueError("没有可合成的帧")
    cols = max(1, int(cols))
    fw = max(f.width for f in frames)
    fh = max(f.height for f in frames)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGBA", (fw * cols, fh * rows), (0, 0, 0, 0))
    index = []
    for i, f in enumerate(frames):
        c, r = i % cols, i // cols
        frame = zero_transparent_rgb(f)
        sheet.alpha_composite(frame, (c * fw, r * fh))
        index.append({"i": i, "x": c * fw, "y": r * fh, "w": fw, "h": fh, "t": round(i * 0.1, 3)})
    meta = {
        "version": "1.0",
        "frame_size": {"w": fw, "h": fh},
        "sheet_size": {"w": sheet.width, "h": sheet.height},
        "grid": {"cols": cols, "rows": rows},
        "frames": index,
    }
    return sheet, meta


def verify_composed(sheet: Image.Image, frames: List[Image.Image], cols: int) -> dict:
    """自检合成表：每个单元格必须与源帧一致，空白区必须全透明。

    移植自 h3-character-studio 的 check_bundle——它交付前逐帧验 sha256、
    验图集切片、验二值 alpha，「宁可在这里炸掉，也不把错表交出去」。
    alpha_composite 是确定性的，正常情况下不可能不一致；真出现不一致
    说明有 bug 或 PIL 版本问题。结果写进资产 meta，前端/导出都能看到。
    """
    cols = max(1, int(cols))
    if not frames:
        return {"cellsMatchFrames": False, "reason": "没有帧"}
    fw = max(f.width for f in frames)
    fh = max(f.height for f in frames)
    rows = (len(frames) + cols - 1) // cols
    if sheet.size != (fw * cols, fh * rows):
        return {"cellsMatchFrames": False, "reason": f"尺寸不符：{sheet.size}"}
    s = np.asarray(sheet.convert("RGBA"))
    for i, f in enumerate(frames):
        c, r = i % cols, i // cols
        cell = s[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw]
        # 与「清零后」的源帧比对：compose_sheet 贴的就是 zero_transparent_rgb(f)，
        # 直接拿原帧比会在透明区颜色上误报不匹配
        src = np.asarray(zero_transparent_rgb(f))
        # 帧比单元格小（compose_sheet 按最大帧定格）时，重叠区必须逐位一致，
        # 剩余区域必须全透明——alpha_composite 的画布初始就是透明的
        if not np.array_equal(cell[:src.shape[0], :src.shape[1]], src):
            return {"cellsMatchFrames": False, "mismatchAt": i}
        rest = cell[src.shape[0]:, :]
        if rest.size and (rest[:, :, 3] != 0).any():
            return {"cellsMatchFrames": False, "mismatchAt": i, "reason": "空白区不透明"}
    return {"cellsMatchFrames": True, "cells": len(frames)}
