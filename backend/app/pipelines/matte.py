"""连通域色度键抠图：只移除与画面边缘连通的同色区域，不误删内部同色像素。

AI 出图的背景常常不是单色（渐变 / 暗角 / 多块底色），只看左上角会漏抠。
这里改为「边缘多色板 + 连通域洪泛 + 有限梯度生长」三段式：
1. 统计画面四边出现频率高的颜色，得到 1–4 个背景色（覆盖渐变与分段底色）
2. 从四边匹配任一背景色的像素出发洪泛，只删与边缘连通的区域
3. 对被背景包围、颜色又与相邻已删像素接近的残边做有限生长（收掉羽化过渡）
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter


def _border_palette(arr: np.ndarray, bucket: int = 24, min_ratio: float = 0.03,
                    limit: int = 4) -> List[np.ndarray]:
    """四角取样得到背景色板。

    只取四角 16px 见方的小块：像素角色总是居中，边角必定是背景；
    而单看整条边会把画面里的角色 / 描边颜色也统计进来（会导致把角色抠掉）。
    一个颜色必须在至少两个角出现，才认定为背景。
    """
    h, w = arr.shape[:2]
    pad = max(4, min(16, h // 8, w // 8))
    patches = [
        arr[0:pad, 0:pad, :3],
        arr[0:pad, w - pad:w, :3],
        arr[h - pad:h, 0:pad, :3],
        arr[h - pad:h, w - pad:w, :3],
    ]
    votes: dict = {}
    sums: dict = {}
    for pi, patch in enumerate(patches):
        px = patch.reshape(-1, 3).astype(np.int32)
        q = (px // bucket) * bucket
        uniq, counts = np.unique(q, axis=0, return_counts=True)
        # 每个角只投自己最高频的一两种颜色
        for i in np.argsort(-counts)[:2]:
            key = tuple(int(v) for v in uniq[i])
            votes[key] = votes.get(key, 0) + 1
            sums.setdefault(key, []).append(uniq[i].astype(np.int32) + bucket // 2)

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    out: List[np.ndarray] = []
    for key, v in ranked:
        if v < 2 and out:
            continue
        out.append(np.mean(sums[key], axis=0).astype(np.int32))
        if len(out) >= limit:
            break
    if not out:
        out.append(arr[0, 0, :3].astype(np.int32))
    return out


def _neighbour_shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """把 mask 按 (dy,dx) 平移，越界填 False。"""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, ye = (dy, h) if dy >= 0 else (0, h + dy)
    xs, xe = (dx, w) if dx >= 0 else (0, w + dx)
    out[ys:ye, xs:xe] = mask[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def connected_matte(
    img: Image.Image,
    tolerance: int = 80,
    feather: int = 5,
    sample: str = "auto",
    edge_grow: int = 10,
    grow_passes: int = 6,
) -> Tuple[Image.Image, dict]:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]

    rgb = arr[:, :, :3].astype(np.int32)
    if sample == "topleft":
        palette = [rgb[0, 0]]
    else:
        palette = _border_palette(arr)

    dist = np.full((h, w), 1e9, dtype=np.float32)
    for c in palette:
        d = np.sqrt(((rgb - c) ** 2).sum(axis=2)).astype(np.float32)
        dist = np.minimum(dist, d)
    match = dist <= float(tolerance)
    # 已经是透明的像素直接算背景
    match |= arr[:, :, 3] < 8

    visited = np.zeros((h, w), dtype=bool)
    dq: deque = deque()
    for x in range(w):
        for y in (0, h - 1):
            if match[y, x] and not visited[y, x]:
                visited[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
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

    # 有限梯度生长：收掉抗锯齿 / 渐变残边，高对比的角色边缘不会被吃掉
    grown = 0
    if edge_grow > 0:
        for _ in range(max(0, grow_passes)):
            added = np.zeros((h, w), dtype=bool)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                near_removed = _neighbour_shift(visited, dy, dx)
                cand = near_removed & ~visited
                if not cand.any():
                    continue
                shifted_rgb = np.zeros_like(rgb)
                h2, w2 = rgb.shape[:2]
                ys, ye = (dy, h2) if dy >= 0 else (0, h2 + dy)
                xs, xe = (dx, w2) if dx >= 0 else (0, w2 + dx)
                shifted_rgb[ys:ye, xs:xe] = rgb[ys - dy:ye - dy, xs - dx:xe - dx]
                close = np.abs(rgb - shifted_rgb).sum(axis=2) <= edge_grow * 3
                added |= cand & close
            if not added.any():
                break
            visited |= added
            grown += int(added.sum())

    alpha = np.where(visited, 0, 255).astype(np.uint8)
    if feather > 0:
        softened = np.array(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(max(0.5, feather / 2.0))))
        alpha = np.where(softened > 250, 255, softened).astype(np.uint8)
        alpha[~visited] = np.maximum(alpha[~visited], 128)

    arr[:, :, 3] = alpha
    out = Image.fromarray(arr, "RGBA")
    meta = {
        "background": "#%02X%02X%02X" % tuple(int(v) for v in palette[0]),
        "palette": ["#%02X%02X%02X" % tuple(int(v) for v in c) for c in palette],
        "tolerance": tolerance,
        "feather": feather,
        "edgeGrowPixels": grown,
        "removedRatio": round((removed + grown) / float(h * w), 4),
    }
    return out, meta
