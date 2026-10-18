"""Sprite Sheet 拆分与合成 + 引擎索引 JSON。"""
from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image


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
        frame = f if f.mode == "RGBA" else f.convert("RGBA")
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
