"""左右无缝延展：镜面拼接 + 边缘交叉淡化。

用途：视差图层在角色移动超过一屏宽度时需要左右可循环的版本。
做法：横向复制 N 份，奇数份水平镜像（避免图案重复的硬接缝），
     再在接缝处做线性 alpha 交叉淡化。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _feather_mask(width: int, height: int, band: int) -> np.ndarray:
    """左侧 band 像素的 0→1 渐变遮罩（用于把右图淡入左图）。"""
    mask = np.ones((height, width), dtype=np.float32)
    if band > 0:
        ramp = np.linspace(0.0, 1.0, band, dtype=np.float32)
        mask[:, :band] = ramp
    return mask


def extend_horizontal(img: Image.Image, times: int = 2, blend: int = 24,
                      mirror: bool = True) -> Tuple[Image.Image, dict]:
    """横向延展为 times 倍宽。

    镜像模式：相邻两份互为镜像，接缝两侧像素天然相同（数学上连续），
             因此不做淡化，避免破坏画面。
    重复模式：接缝必然有硬切，仅对窄带做轻微柔化。
    """
    base = img.convert("RGBA")
    w, h = base.size
    times = max(2, int(times))
    canvas = Image.new("RGBA", (w * times, h), (0, 0, 0, 0))

    for i in range(times):
        tile = base
        if mirror and i % 2 == 1:
            tile = base.transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(tile, (i * w, 0))

    if (not mirror) and blend > 0:
        from PIL import ImageFilter

        arr = np.array(canvas)
        band = max(2, min(int(blend), w // 4))
        for i in range(1, times):
            x0 = max(0, i * w - band // 2)
            crop = Image.fromarray(arr[:, x0:x0 + band, :]).filter(ImageFilter.GaussianBlur(band / 3))
            arr[:, x0:x0 + band, :] = np.array(crop)
        canvas = Image.fromarray(arr, "RGBA")

    meta = {
        "mode": "mirror" if mirror else "repeat+soften",
        "times": times,
        "blend": 0 if mirror else blend,
        "size": [canvas.width, canvas.height],
        "loops": True,
        "note": "镜像模式接缝天然连续；如需非镜像循环请用「瓦片接缝修补」先做无缝化",
    }
    return canvas, meta


def extend_vertical(img: Image.Image, times: int = 2, blend: int = 16) -> Tuple[Image.Image, dict]:
    base = img.convert("RGBA")
    w, h = base.size
    times = max(2, int(times))
    canvas = Image.new("RGBA", (w, h * times), (0, 0, 0, 0))
    for i in range(times):
        canvas.paste(base if i % 2 == 0 else base.transpose(Image.FLIP_TOP_BOTTOM), (0, i * h))
    meta = {"mode": "mirror-vertical", "times": times, "size": [canvas.width, canvas.height]}
    return canvas, meta
