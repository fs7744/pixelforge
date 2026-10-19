"""参考图规范化（移植自 h3-character-studio 的 normalize-master 流程）。

为什么需要：参考图来自各处——截图、相机照片、别的工具导出——画幅、留白、
主体位置全不一样。直接送进编辑底座会有两个问题：

1. 模型按参考图构图：主体在画面角落，生成结果也偏向角落；主体顶天立地，
   生成时容易裁头裁脚。
2. 同一角色换参考图重跑，画幅变了，输出尺寸和构图跟着漂。

规范化做三件事：裁到主体包围盒 → 铺到方形画布（边长 16 对齐）→ 主体底部
对齐到统一基线。之后所有参考图共享同一套几何，重跑结果才可比。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def _subject_box(img: Image.Image, tol: int = 24) -> Tuple[int, int, int, int]:
    """主体包围盒：有 alpha 用 alpha；没有就按四角底色做颜色键。

    与 matte 的连通域洪泛不同，这里只需要包围盒，不需要精确掩码——
    便宜很多，也足够支撑裁切与对齐。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    alpha = arr[:, :, 3]
    if (alpha < 8).any():
        kept = alpha >= 8
        if kept.any():
            ys, xs = np.where(kept)
            return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    # 无透明：四角取底色，颜色距离超过容差的算主体
    h, w = arr.shape[:2]
    corners = np.array([arr[0, 0, :3], arr[0, w - 1, :3],
                        arr[h - 1, 0, :3], arr[h - 1, w - 1, :3]], dtype=np.int16)
    bg = np.median(corners, axis=0)
    dist = np.abs(arr[:, :, :3].astype(np.int16) - bg).sum(axis=2)
    mask = dist > tol
    if not mask.any():
        return 0, 0, w, h
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalize_ref(
    img: Image.Image,
    *,
    tolerance: int = 24,
    pad_ratio: float = 0.06,
    baseline: float = 0.94,
    size: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """裁主体 → 方形画布 → 底部对齐基线。

    pad_ratio：主体四周留白占比（相对画布边长）。
    baseline：主体底部落在画布的哪个高度比例（0.94 = 底部留 6%）。
    size：目标边长；0 = 按主体自动取（最长边 / (1-2*pad) 向上取整到 16）。
    背景：能读到统一底色就填底色（编辑底座转 RGB 时透明会变黑，不能留透明），
    否则填中性灰 #808080（与生成默认底色一致，见 backdrop.py）。
    """
    rgba = img.convert("RGBA")
    x0, y0, x1, y1 = _subject_box(rgba, tol=tolerance)
    subject = rgba.crop((x0, y0, x1, y1))
    sw, sh = subject.size
    if sw < 1 or sh < 1:
        raise ValueError("参考图里没有可识别的主体（整幅都被判为背景）")

    side = int(size) if size else int(np.ceil(max(sw, sh) / max(0.2, 1 - 2 * pad_ratio)))
    side = max(64, (side + 15) // 16 * 16)
    inner = side * (1 - 2 * pad_ratio)
    scale = min(inner / sw, inner / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    scaled = subject.resize((nw, nh), Image.LANCZOS)

    # 背景判定：原始四角是否一致（差值 <12 视为统一底色）
    src = np.asarray(rgba)
    h_, w_ = src.shape[:2]
    cs = np.array([src[0, 0, :3], src[0, w_ - 1, :3],
                   src[h_ - 1, 0, :3], src[h_ - 1, w_ - 1, :3]], dtype=np.int16)
    uniform = bool(np.abs(cs - np.median(cs, axis=0)).max() <= 12)
    bg = tuple(int(v) for v in np.median(cs, axis=0)) if uniform else (128, 128, 128)

    canvas = Image.new("RGBA", (side, side), (*bg, 255))
    # 水平居中；底部对齐到 baseline（主体底边落在 side*baseline）
    x = (side - nw) // 2
    y = max(0, int(round(side * baseline)) - nh)
    canvas.alpha_composite(scaled, (x, y))

    meta = {
        "pipeline": "refnorm",
        "sourceSize": [img.width, img.height],
        "subjectBox": [x0, y0, x1, y1],
        "subjectSize": [sw, sh],
        "canvas": [side, side],
        "scale": round(scale, 4),
        "background": "#%02X%02X%02X" % bg,
        "backgroundDetected": uniform,
        "baseline": baseline,
        "padRatio": pad_ratio,
    }
    return canvas.convert("RGB"), meta
