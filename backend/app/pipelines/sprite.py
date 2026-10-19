"""精灵帧规整：抠图 → 裁空白 → 最近邻缩放 → 底部居中对齐。

动画素材的三个硬伤都靠这一步解决：
1. 背景不透明 → 连通域抠图
2. 每帧角色大小不一 → 统一包围盒缩放
3. 每帧角色位置漂移 → 固定锚点（默认底部居中）重新贴图

批量规整（normalize_frames）默认逐帧各自算缩放，这在「同尺寸关键帧」上够用，
但做动作循环时会露馅：下蹲的帧包围盒更矮，各自缩放到撑满画布 → 角色忽大忽小，
播起来一蹦一蹦。shared_scale=True 改成「先量所有帧的最大包围盒，再统一缩放」，
角色尺寸在整段动作里恒定。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import Image

from . import matte as matte_pipe


def trim_alpha(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """裁掉全透明边缘，返回裁剪图与包围盒。"""
    rgba = img.convert("RGBA")
    bbox = rgba.split()[-1].getbbox()
    if not bbox:
        return rgba, (0, 0, rgba.width, rgba.height)
    return rgba.crop(bbox), bbox


def _prepare(
    img: Image.Image,
    *,
    matte: bool = True,
    tolerance: int = 80,
    feather: int = 0,
    hard: bool = True,
    matte_method: str = "auto",
    inset: float = 0.0,
) -> Tuple[Image.Image, dict]:
    """抠图 + 边缘清洗 + 裁空白。返回裁剪后的 RGBA 与中间信息。"""
    work = img.convert("RGBA")
    if inset > 0:
        dx = int(work.width * inset)
        dy = int(work.height * inset)
        if dx * 2 < work.width and dy * 2 < work.height:
            work = work.crop((dx, dy, work.width - dx, work.height - dy))
    matte_meta: dict = {}
    if matte:
        work, matte_meta = matte_pipe.connected_matte(
            work, tolerance=tolerance, feather=feather, hard=hard, method=matte_method
        )
        # 硬 alpha 会保留「描边 + 键色」的抗锯齿混色：绿幕底实测边缘环 51% 偏绿，
        # 只看 alpha 是干净的，放大到游戏里就是一圈绿线。清洗只改 RGB 不动 alpha。
        bg_hex = matte_meta.get("background")
        if hard and bg_hex:
            bg = tuple(int(bg_hex[i:i + 2], 16) for i in (1, 3, 5))
            work, spilled = matte_pipe.clean_edge_spill(work, bg)
            matte_meta["edgePixelsRecolored"] = spilled

    trimmed, bbox = trim_alpha(work)
    return trimmed, {"matte": matte_meta, "sourceBox": list(bbox)}


def _place(
    trimmed: Image.Image,
    size: int,
    scale: float,
    *,
    anchor: str = "bottom",
    pad: int = 2,
) -> Image.Image:
    """按给定缩放把裁剪图贴进 size×size 画布（最近邻，像素素材不能插值）。"""
    inner = max(1, size - pad * 2)
    nw = max(1, int(round(trimmed.width * scale)))
    nh = max(1, int(round(trimmed.height * scale)))
    nw, nh = min(nw, inner), min(nh, inner)
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
    return canvas


def normalize_frame(
    img: Image.Image,
    size: int = 128,
    anchor: str = "bottom",
    pad: int = 2,
    matte: bool = True,
    tolerance: int = 80,
    feather: int = 0,
    hard: bool = True,
    matte_method: str = "auto",
    inset: float = 0.0,
) -> Tuple[Image.Image, dict]:
    """把任意帧规整成 size×size 的透明底精灵帧。

    inset：先向内裁掉一圈（按比例）。AI 出图常自带「画框 / 描边相框」，
    不先切掉会被当成角色的一部分留下。
    """
    trimmed, info = _prepare(
        img, matte=matte, tolerance=tolerance, feather=feather,
        hard=hard, matte_method=matte_method, inset=inset,
    )
    if trimmed.width < 1 or trimmed.height < 1:
        return Image.new("RGBA", (size, size), (0, 0, 0, 0)), {
            "size": [size, size], "empty": True, "matte": info["matte"],
        }

    inner = max(1, size - pad * 2)
    scale = min(inner / trimmed.width, inner / trimmed.height)
    canvas = _place(trimmed, size, scale, anchor=anchor, pad=pad)
    meta = {
        "size": [size, size],
        "sourceBox": info["sourceBox"],
        "scale": round(scale, 4),
        "anchor": anchor,
        "inset": inset,
        "matte": info["matte"],
    }
    return canvas, meta


def normalize_frames(
    frames: List[Image.Image],
    *,
    shared_scale: bool = False,
    **kwargs,
) -> Tuple[List[Image.Image], dict]:
    """批量规整。

    shared_scale=True：所有帧用同一个缩放系数（按最大的那个包围盒算），
    角色在整段动作里保持恒定大小——动作循环必须这样，否则下蹲帧会被放大。
    """
    size = int(kwargs.get("size", 128))
    anchor = kwargs.get("anchor", "bottom")
    pad = int(kwargs.get("pad", 2))
    prep_keys = ("matte", "tolerance", "feather", "hard", "matte_method", "inset")
    prep_kw = {k: kwargs[k] for k in prep_keys if k in kwargs}

    prepared = [_prepare(f, **prep_kw) for f in frames]
    outs: List[Image.Image] = []
    scales: List[float] = []
    removed: List[float] = []
    inner = max(1, size - pad * 2)

    if shared_scale:
        max_w = max((t.width for t, _ in prepared), default=1)
        max_h = max((t.height for t, _ in prepared), default=1)
        common = min(inner / max(1, max_w), inner / max(1, max_h))
    else:
        common = 0.0

    for trimmed, info in prepared:
        if trimmed.width < 1 or trimmed.height < 1:
            outs.append(Image.new("RGBA", (size, size), (0, 0, 0, 0)))
            scales.append(0.0)
            continue
        scale = common or min(inner / trimmed.width, inner / trimmed.height)
        outs.append(_place(trimmed, size, scale, anchor=anchor, pad=pad))
        scales.append(float(scale))
        if info["matte"]:
            removed.append(float(info["matte"].get("removedRatio", 0)))

    meta = {
        "size": [size, size],
        "frames": len(outs),
        "scaleRange": [round(min(scales), 3), round(max(scales), 3)] if scales else [],
        "sharedScale": bool(shared_scale),
        "bgRemovedAvg": round(sum(removed) / len(removed), 3) if removed else 0,
        "anchor": anchor,
    }
    return outs, meta
