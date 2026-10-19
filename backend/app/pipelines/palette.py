"""全局调色板：从一组图里取共享色板，再把每张图映射回这块色板。

移植自 game-asset-video-sprites 的 sprite-postprocess（D:/code/gen）。

为什么需要：逐帧各自量化（MEDIANCUT）时，每帧都会为「自己的颜色分布」
重新分一份色板——同一个肤色在第 1 帧偏橙、第 3 帧偏红，动画一播就闪。
游戏素材的做法是**先取一块全局色板，再让每一帧都映射回它**，帧间颜色严格
一致，这也正是引擎里调色板动画（palette swap）能用的前提。

取色用 k-means（只用不透明像素，避免背景色混进色板）；映射用 RGB 欧氏
最近色，不抖动——抖动会让「每格单色」的像素语法失效。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

Color = Tuple[int, int, int]
# 每批处理的像素数：640² 一张约 41 万像素，全量和色板算距离会占几百 MB，
# 分批后峰值可控，结果逐位一致（只是分段 argmin）。
_CHUNK = 1 << 16


def _opaque_pixels(img: Image.Image, exclude: Optional[Color] = None,
                   exclude_dist: float = 90.0, max_px: int = 200_000) -> np.ndarray:
    """取不透明像素；可排除接近某个颜色的像素（背景色）。"""
    arr = np.asarray(img.convert("RGBA"))
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    px = rgb[alpha > 127].reshape(-1, 3).astype(np.float32)
    if exclude is not None:
        ex = np.asarray(exclude, dtype=np.float32)
        px = px[((px - ex) ** 2).sum(1) > exclude_dist ** 2]
    if len(px) > max_px:
        # 固定种子抽样：色板必须可复现，否则同一批帧每次生成的色板都不同
        rng = np.random.default_rng(0)
        px = px[rng.choice(len(px), max_px, replace=False)]
    return px


def extract_palette(
    images: Sequence[Image.Image],
    k: int = 24,
    exclude: Optional[Color] = None,
    exclude_dist: float = 90.0,
) -> List[Color]:
    """从多张图里取一块共享色板（k-means，按簇大小降序）。"""
    import cv2

    k = max(2, min(int(k), 64))
    chunks = [_opaque_pixels(im, exclude, exclude_dist) for im in images]
    px = np.concatenate([c for c in chunks if len(c)]) if any(len(c) for c in chunks) else None
    if px is None or len(px) < k:
        return []
    # k 不应超过真实颜色数：限量素材（像素画、LOGO）本身只有几种颜色，
    # k-means 硬凑 k 个簇会把同一种色拆成几个差几级的簇心，色板一半是废名额。
    # 先按 8 级一档统计唯一色（吸收压缩噪点），据此收敛 k。
    uniq = np.unique((px // 8).astype(np.uint8), axis=0)
    k = max(2, min(k, len(uniq)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(px, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    order = np.argsort(-counts)
    out: List[Color] = []
    for i in order:
        c = (int(np.clip(centers[i][0], 0, 255)),
             int(np.clip(centers[i][1], 0, 255)),
             int(np.clip(centers[i][2], 0, 255)))
        # k-means 簇数多于真实颜色数时会给出重复/近似重复的簇心（两色图 k=4
        # 会拿到四个色，其中三个一样）——留着只会白白占掉色板名额。
        # 注意 out 为空时 all() 恒为 True，第一个颜色会被误判成重复。
        # 阈值取平方距离 64（≈8 级）：k-means 对同一个色簇会给出几个差几级的
        # 簇心，肉眼完全一样，留着只是白占色板名额；游戏色板里真正相邻的两档
        # （皮肤暗部/亮部）通常差得比这远。
        if out and all(sum((c[j] - e[j]) ** 2 for j in range(3)) <= 64 for e in out):
            continue
        out.append(c)
    return out


def quantize_to_palette(rgb: np.ndarray, palette: Sequence[Color]) -> np.ndarray:
    """把 RGB 图映射到色板（最近色，无抖动），返回索引图 (H, W)。"""
    pal = np.asarray(palette, dtype=np.int32)
    if pal.ndim != 2 or len(pal) == 0:
        raise ValueError("色板为空")
    flat = rgb.reshape(-1, 3).astype(np.int32)
    idx = np.empty(len(flat), dtype=np.int32)
    for s in range(0, len(flat), _CHUNK):
        d = ((flat[s:s + _CHUNK, None, :] - pal[None, :, :]) ** 2).sum(-1)
        idx[s:s + _CHUNK] = d.argmin(1)
    return idx.reshape(rgb.shape[:2])


def palette_error(rgb: np.ndarray, palette: Sequence[Color]) -> float:
    """平均每像素到色板的欧氏距离——锁色损失有多大，能量化。"""
    pal = np.asarray(palette, dtype=np.int32)
    flat = rgb.reshape(-1, 3).astype(np.int32)
    total, n = 0.0, 0
    for s in range(0, len(flat), _CHUNK):
        d = ((flat[s:s + _CHUNK, None, :] - pal[None, :, :]) ** 2).sum(-1)
        total += float(d.min(1).sum())
        n += len(d)
    return round(total / max(1, n), 2)


def apply_palette(img: Image.Image, palette: Sequence[Color]) -> Image.Image:
    """整图锁色（保留 alpha），返回 RGB(A) 图。

    只量化可见像素；透明像素的 RGB 清零（h3-character-studio 的
    core.process 同款做法）。原先透明区也被映射进色板，绿幕残色会以
    「不可见但可采样」的形式留在图里，引擎一采样就是晕边。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).copy()
    visible = arr[:, :, 3] > 0
    idx = quantize_to_palette(arr[:, :, :3], palette)
    pal = np.asarray(palette, dtype=np.uint8)
    arr[:, :, :3] = pal[idx]
    arr[~visible, :3] = 0
    return Image.fromarray(arr, "RGBA")
