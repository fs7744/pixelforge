"""连通域色度键抠图：只移除与画面边缘连通的同色区域，不误删内部同色像素。"""
from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def _bg_color(arr: np.ndarray, sample: str = "topleft") -> np.ndarray:
    h, w = arr.shape[:2]
    if sample == "topleft":
        return arr[0, 0, :3].astype(np.int32)
    # 四角取众数（更稳）
    corners = [arr[0, 0, :3], arr[0, w - 1, :3], arr[h - 1, 0, :3], arr[h - 1, w - 1, :3]]
    vals = np.array([c.astype(np.int32) for c in corners])
    uniq, counts = np.unique(vals, axis=0, return_counts=True)
    return uniq[int(np.argmax(counts))]


def connected_matte(
    img: Image.Image,
    tolerance: int = 80,
    feather: int = 5,
    sample: str = "topleft",
) -> Tuple[Image.Image, dict]:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    bg = _bg_color(arr, sample)

    rgb = arr[:, :, :3].astype(np.int32)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    match = dist <= float(tolerance)

    visited = np.zeros((h, w), dtype=bool)
    dq: deque = deque()

    # 从四角 + 四边匹配像素出发（保证大面积背景连通）
    seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for x in range(0, w, 2):
        seeds.append((0, x)); seeds.append((h - 1, x))
    for y in range(0, h, 2):
        seeds.append((y, 0)); seeds.append((y, w - 1))
    for (y, x) in seeds:
        if match[y, x] and not visited[y, x]:
            visited[y, x] = True
            dq.append((y, x))

    removed = 0
    while dq:
        y, x = dq.popleft()
        removed += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and match[ny, nx]:
                visited[ny, nx] = True
                dq.append((ny, nx))

    alpha = np.where(visited, 0, 255).astype(np.uint8)
    if feather > 0:
        softened = np.array(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(max(0.5, feather / 2.0))))
        # 内部保持不透明，只柔化边缘
        alpha = np.where(softened > 250, 255, softened).astype(np.uint8)
        alpha[~visited] = np.maximum(alpha[~visited], 128)

    arr[:, :, 3] = alpha
    out = Image.fromarray(arr, "RGBA")
    meta = {
        "background": "#%02X%02X%02X" % tuple(int(v) for v in bg),
        "tolerance": tolerance,
        "feather": feather,
        "removedRatio": round(removed / float(h * w), 4),
    }
    return out, meta
