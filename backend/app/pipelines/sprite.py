"""精灵帧规整：抠图 → 裁空白 → 最近邻缩放 → 底部居中对齐。

动画素材的三个硬伤都靠这一步解决：
1. 背景不透明 → 连通域抠图
2. 每帧角色大小不一 → 统一包围盒缩放
3. 每帧角色位置漂移 → 固定锚点（默认底部居中）重新贴图
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from PIL import Image

from . import matte as matte_pipe


def trim_alpha(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """裁掉全透明边缘，返回裁剪图与包围盒。"""
    rgba = img.convert("RGBA")
    bbox = rgba.split()[-1].getbbox()
    if not bbox:
        return rgba, (0, 0, rgba.width, rgba.height)
    return rgba.crop(bbox), bbox


def normalize_frame(
    img: Image.Image,
    size: int = 128,
    anchor: str = "bottom",
    pad: int = 2,
    matte: bool = True,
    tolerance: int = 80,
    feather: int = 4,
) -> Tuple[Image.Image, dict]:
    """把任意帧规整成 size×size 的透明底精灵帧。"""
    work = img.convert("RGBA")
    matte_meta = {}
    if matte:
        work, matte_meta = matte_pipe.connected_matte(work, tolerance=tolerance, feather=feather)

    trimmed, bbox = trim_alpha(work)
    if trimmed.width < 1 or trimmed.height < 1:
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        return out, {"size": [size, size], "empty": True}

    inner = max(1, size - pad * 2)
    scale = min(inner / trimmed.width, inner / trimmed.height)
    nw = max(1, int(round(trimmed.width * scale)))
    nh = max(1, int(round(trimmed.height * scale)))
    # 像素素材必须用最近邻，双线性会把像素糊掉
    scaled = trimmed.resize((nw, nh), Image.NEAREST)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - nw) // 2
    if anchor == "center":
        y = (size - nh) // 2
    elif anchor == "top":
        y = pad
    else:
        y = size - nh - pad
    canvas.alpha_composite(scaled, (x, y))

    meta = {
        "size": [size, size],
        "sourceBox": list(bbox),
        "scale": round(scale, 4),
        "anchor": anchor,
        "matte": matte_meta,
    }
    return canvas, meta


def normalize_frames(frames: List[Image.Image], **kwargs) -> Tuple[List[Image.Image], dict]:
    outs: List[Image.Image] = []
    scales: List[float] = []
    removed: List[float] = []
    for f in frames:
        out, m = normalize_frame(f, **kwargs)
        outs.append(out)
        scales.append(float(m.get("scale", 0)))
        if m.get("matte"):
            removed.append(float(m["matte"].get("removedRatio", 0)))
    size = int(kwargs.get("size", 128))
    meta = {
        "size": [size, size],
        "frames": len(outs),
        "scaleRange": [round(min(scales), 3), round(max(scales), 3)] if scales else [],
        "bgRemovedAvg": round(sum(removed) / len(removed), 3) if removed else 0,
        "anchor": kwargs.get("anchor", "bottom"),
    }
    return outs, meta
