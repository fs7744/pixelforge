"""尺寸规整管线：最近邻硬缩放 / 等比适配留白 / RPGMaker 一键。"""
from __future__ import annotations

from typing import Tuple

from PIL import Image


def hard_resize(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """最近邻硬缩放：像素素材不允许插值，否则会糊。"""
    return img.resize(size, Image.NEAREST)


def fit_pad(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """等比缩放后居中留白（透明），适合任意长宽比素材。"""
    w, h = size
    sw = w / img.width
    sh = h / img.height
    scale = min(sw, sh)
    new = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    resized = img.resize(new, Image.NEAREST)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.paste(resized, ((w - new[0]) // 2, (h - new[1]) // 2), resized)
    return canvas


def extend_bottom(img: Image.Image, px: int) -> Image.Image:
    canvas = Image.new("RGBA", (img.width, img.height + px), (0, 0, 0, 0))
    canvas.paste(img, (0, 0), img)
    return canvas


def crop_edges(img: Image.Image, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> Image.Image:
    w, h = img.size
    return img.crop((left, top, max(1, w - right), max(1, h - bottom)))


def rpgmaker(img: Image.Image, variant: str = "v1") -> Tuple[Image.Image, dict]:
    """RPGMaker 一键处理（对齐 FrameRonin 的确定性参数）。

    v1: 144×144 硬缩放 + 底部扩 48px 透明
    v2: 256×256 → 右裁 64 / 下扩 64 → 144×240
    v2_four_row: v2 再裁下方 48 → 144×192
    """
    if variant == "v2":
        base = hard_resize(img, (256, 256))
        base = crop_edges(base, right=64)
        base = extend_bottom(base, 64)
        base = hard_resize(base, (144, 240))
        return base, {"variant": "v2", "size": [144, 240], "note": "右裁64 / 下扩64 后硬缩放"}
    if variant == "v2_four_row":
        base = hard_resize(img, (256, 256))
        base = crop_edges(base, right=64)
        base = extend_bottom(base, 64)
        base = hard_resize(base, (144, 240))
        base = crop_edges(base, bottom=48)
        return base, {"variant": "v2_four_row", "size": [144, 192], "note": "v2 基础上裁下方48"}
    base = hard_resize(img, (144, 144))
    base = extend_bottom(base, 48)
    return base, {"variant": "v1", "size": [144, 192], "note": "144×144 硬缩放 + 底部扩 48px"}
