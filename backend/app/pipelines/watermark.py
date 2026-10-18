"""去水印管线：反向 Alpha 混合。

原理（对齐 allenk/GeminiWatermarkTool）：
    watermarked = a * logo + (1 - a) * original
    => original = (watermarked - a * logo) / (1 - a)

需要提供水印 logo 的 alpha 掩码（mask）。MVP 未内置官方掩码时，
接口返回原图并给出提示，不做破坏性处理。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image


def reverse_alpha(img: Image.Image, mask: Image.Image, logo_rgb: Tuple[int, int, int] = (255, 255, 255),
                  strength: float = 1.0) -> Image.Image:
    """按掩码做反向 Alpha 混合，恢复被水印覆盖的像素。"""
    base = np.array(img.convert("RGBA")).astype(np.float64)
    m = np.array(mask.convert("L").resize(img.size, Image.BILINEAR)).astype(np.float64) / 255.0
    m = np.clip(m * float(strength), 0.0, 0.95)

    logo = np.array(logo_rgb, dtype=np.float64)
    out = base.copy()
    for ch in range(3):
        out[:, :, ch] = (base[:, :, ch] - m * logo[ch]) / np.maximum(1e-6, (1.0 - m))
    out = np.clip(out, 0, 255)
    alpha = base[:, :, 3]
    out[:, :, 3] = alpha
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def remove_watermark(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    size: int = 48,
    strength: float = 1.0,
) -> Tuple[Image.Image, dict]:
    if mask is None:
        return img, {
            "applied": False,
            "reason": "未提供水印掩码。反向 Alpha 混合需要 logo alpha 掩码，"
                      "MVP 不做破坏性猜测；可在 V2 内置 Gemini 48/96 掩码后自动生效。",
        }
    out = reverse_alpha(img, mask, strength=strength)
    return out, {"applied": True, "size": size, "strength": strength, "method": "reverse-alpha"}
