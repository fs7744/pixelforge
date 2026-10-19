"""逐帧几何法线与材质（金属）近似掩码。

h3-character-studio 的做法是：人工标注关键姿态的形体、双向传播解析曲面，
再算向量。那套要求逐帧标注，本项目没有标注界面，所以这里走**从像素反推**
的路线，说清楚它是什么、不是什么：

- 高度场 = 0.78 × 内缩距离（alpha 距离变换归一化，越靠内越高，形成圆润的
  凸起）+ 0.22 × 亮度（把明暗当作形状暗示）。
- 法线 = 高度场梯度（Sobel）转切空间向量，再按强度放大。
- 材质掩码 = 高光近似：亮且低饱和（金属高光）或高饱和（彩色金属漆），
  两种模式可选，阈值可调。

局限必须讲明白：它不知道「这是剑刃还是袖子」，遮挡边界和薄片会被压成
同一个鼓包；正对镜头的平面也会因为亮度渐变而带一点假斜率。要精确形体
法线，得上 h3 那种标注传播，这里给的是「够用的光照贴图」。

法线图约定：切线空间、OpenGL 风格（Y 向上为正，绿通道朝上），
Godot / Unity 直接用；DirectX 风格需要翻转绿通道（导出参数可选）。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "法线生成需要 opencv-python：pip install opencv-python（或跑 ./start.sh --gpu）")


def _normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def analyze(
    img: Image.Image,
    *,
    strength: float = 1.3,
    blur: float = 2.0,
    metal_mode: str = "highlight",
    metal_threshold: float = 0.6,
    flip_green: bool = False,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """返回（法线图 RGBA, 材质掩码 L, meta）。"""
    _require_cv2()
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    alpha = arr[..., 3]
    solid = alpha > 0.02

    # 距离变换：边缘 0、内部越深越大 → 圆柱/球面那样的圆润高度
    dist = cv2.distanceTransform((solid * 255).astype(np.uint8), cv2.DIST_L2, 5)
    dist_n = _normalize01(dist)

    rgb = arr[..., :3]
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    sigma = max(0.1, float(blur))
    lum_b = cv2.GaussianBlur(lum, (0, 0), sigma)

    height = 0.78 * dist_n + 0.22 * lum_b
    height = cv2.GaussianBlur(height, (0, 0), max(0.1, sigma * 0.6))

    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    k = float(max(0.05, strength))
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(gx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    if flip_green:
        ny = -ny

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., 0] = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    out[..., 3] = (solid * 255).astype(np.uint8)
    # 透明区域填成「正对镜头」的平法线，引擎里不会出现黑色凹坑
    flat = ~solid
    out[flat] = (128, 128, 255, 0)

    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    t = float(max(0.01, min(metal_threshold, 0.99)))
    if metal_mode == "chroma":
        metal = np.clip((sat - t) / max(1e-3, 1.0 - t), 0, 1)
    elif metal_mode == "none":
        metal = np.zeros_like(sat)
    else:
        bright = np.clip((val - t) / max(1e-3, 1.0 - t), 0, 1)
        lowsat = np.clip((0.62 - sat) / 0.62, 0, 1)
        metal = bright * lowsat
    metal = cv2.GaussianBlur(metal, (0, 0), max(0.5, sigma))
    mask = (np.clip(metal, 0, 1) * solid * 255).astype(np.uint8)

    meta = {
        "size": [w, h],
        "strength": k,
        "blur": sigma,
        "metalMode": metal_mode,
        "metalThreshold": t,
        "flipGreen": bool(flip_green),
        "solidRatio": round(float(solid.mean()), 4),
        "metalRatio": round(float((mask > 40).mean()), 4),
        "heightRange": [round(float(dist_n.min()), 3), round(float(dist_n.max()), 3)],
        "note": "几何法线由内缩距离+亮度反推，非 h3 的标注传播；薄片与遮挡边界会有偏差",
    }
    return Image.fromarray(out, "RGBA"), Image.fromarray(mask, "L"), meta


def compose_preview(color: Image.Image, normal: Image.Image, mask: Image.Image,
                    light=(0.4, -0.6, 0.7)) -> Image.Image:
    """用固定点光把法线「打」在原图上，肉眼快速判断法线是否合理。

    只做 N·L 漫反射 + 高光：不是渲染器，是检查工具——法线如果上下颠倒，
    这里会立刻显得「光从底下打上来」。
    """
    c = np.asarray(color.convert("RGBA")).astype(np.float32) / 255.0
    n = np.asarray(normal.convert("RGBA")).astype(np.float32) / 255.0
    nx, ny, nz = n[..., 0] * 2 - 1, n[..., 1] * 2 - 1, n[..., 2] * 2 - 1
    lx, ly, lz = light
    ln = (lx * lx + ly * ly + lz * lz) ** 0.5 or 1.0
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    ndl = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    shade = 0.35 + 0.65 * ndl
    out = c.copy()
    out[..., :3] = np.clip(c[..., :3] * shade[..., None], 0, 1)
    # 金属区加一点镜面高光，掩码是否正确一眼能看出来
    m = np.asarray(mask).astype(np.float32) / 255.0
    spec = np.clip(ndl ** 12, 0, 1) * m
    out[..., :3] = np.clip(out[..., :3] + spec[..., None] * 0.9, 0, 1)
    out[..., 3] = c[..., 3]
    return Image.fromarray((out * 255).astype(np.uint8), "RGBA")
