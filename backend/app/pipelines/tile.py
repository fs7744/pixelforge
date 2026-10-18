"""瓦片接缝修补：把任意贴图变成可平铺的无缝单元。

做法（偏移-修复法）：
1. 把图像平移半宽半高，使原来的四条边移到画面中央（接缝集中暴露）；
2. 在中央十字区域用轻微模糊 + 邻域插值抹平接缝；
3. 平移回来。

相比直接模糊边缘，这个方法能保留图案结构，只修复接缝处。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def make_seamless(img: Image.Image, band: int = 12, feather: int = 3) -> Tuple[Image.Image, dict]:
    src = img.convert("RGBA")
    w, h = src.size
    band = max(2, min(int(band), min(w, h) // 3))

    shifted = np.roll(np.array(src), (h // 2, w // 2), axis=(0, 1)).astype(np.float32)

    # 中央十字带（对应原图四边）做柔和过渡
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    half = band // 2
    mask[max(0, cy - half):cy + half, :] = 1.0
    mask[:, max(0, cx - half):cx + half] = 1.0

    blurred = np.array(
        Image.fromarray(shifted.astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(feather))
    ).astype(np.float32)

    m = mask[:, :, None]
    merged = shifted * (1 - m) + blurred * m
    out = np.roll(merged, (-(h // 2), -(w // 2)), axis=(0, 1))

    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")
    meta = {"method": "offset-heal", "band": band, "feather": feather, "size": [w, h], "tileable": True}
    return result, meta


def split_tiles(img: Image.Image, cols: int, rows: int) -> list:
    """按网格裁切瓦片集（供逐格检查接缝）。"""
    cols, rows = max(1, cols), max(1, rows)
    tw, th = img.width // cols, img.height // rows
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)))
    return out
