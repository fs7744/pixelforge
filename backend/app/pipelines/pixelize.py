"""像素化对齐：把 AI 插画转成规整的像素风素材。

流程：分块降采样 → 限定调色板量化 → 最近邻放大回原尺寸。
比单纯模糊/降采样更接近"手工像素画"，因为每一格只保留一个颜色。

注意：这是「视觉风格化」，不做像素网格检测；AI 图网格不齐时
可先用 perfectPixel 类算法对齐（规划中），本管线适合已大致规整的图。

两块来自 game-asset-video-sprites（D:/code/gen）的增强：
- ``downsample="box"``：分块取**区域平均**而不是最近邻单像素。AI 出图带
  渐变和噪点，最近邻等于随机挑一颗像素，量化后是一片脏斑；区域平均先求
  均值再量化，色块干净得多。已经是规整像素图的素材请继续用 nearest
  （box 会把相邻两颗不同色的像素混成中间色）。
- ``shared_palette``：传入一块全局色板，每格映射回色板而不是各自量化。
  逐帧各自量化时每帧色板都不同，同一个肤色会逐帧偏色，动画一播就闪。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PIL import Image

PALETTES = {
    "auto": None,          # 自动中位切分
    "16": 16,
    "32": 32,
    "8": 8,
    "gameboy": ["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"],
    "nes": ["#000000", "#FCFCFC", "#F8F8F8", "#BCBCBC", "#7C7C7C",
            "#A4E4FC", "#3CBCFC", "#0078F8", "#0000FC", "#B8B8F8",
            "#6888FC", "#0058F8", "#0000BC", "#F8B8F8", "#F878F8",
            "#F85898", "#D800CC", "#F8A4C0", "#F0D0B0", "#F87858",
            "#F83800", "#E45C10", "#FCA044", "#F8B800", "#FCE0A8",
            "#F8D878", "#B8F818", "#58D854", "#00A800", "#00E8D8"],
}


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _quantize_palette(rgb: Image.Image, palette_name: str) -> Image.Image:
    """用固定调色板量化（提供 gameboy / nes 等经典色板）。"""
    colors = PALETTES.get(palette_name)
    if not colors or not isinstance(colors, list):
        return rgb
    pal_img = Image.new("P", (1, 1))
    flat = []
    for c in colors:
        flat.extend(_hex_to_rgb(c))
    flat += [0, 0, 0] * (256 - len(colors))
    pal_img.putpalette(flat)
    return rgb.quantize(palette=pal_img, dither=Image.Dither.NONE).convert("RGB")


def pixelize(
    img: Image.Image,
    colors: int = 16,
    pixel_size: int = 4,
    dither: bool = False,
    palette: str = "auto",
    keep_size: bool = True,
    shared_palette: Optional[Sequence[Sequence[int]]] = None,
    downsample: str = "nearest",
) -> Tuple[Image.Image, dict]:
    rgba = img.convert("RGBA")
    w, h = rgba.size
    ps = max(1, min(int(pixel_size), max(1, min(w, h) // 2)))

    small_w, small_h = max(1, w // ps), max(1, h // ps)
    resample = Image.BOX if downsample == "box" else Image.NEAREST
    # alpha 永远走最近邻：区域平均会把硬边混成半透明，像素素材必须是二值 alpha
    small = rgba.resize((small_w, small_h), resample)
    alpha = small.getchannel("A")
    rgb = small.convert("RGB")

    shared = None
    if shared_palette:
        from . import palette as palette_pipe

        pal = [(int(c[0]), int(c[1]), int(c[2])) for c in shared_palette]
        shared = palette_pipe.apply_palette(rgb, pal)
        quant = shared.convert("RGB")
        used = len(pal)
        pal_name = "shared"
    elif palette != "auto" and isinstance(PALETTES.get(palette), list):
        quant = _quantize_palette(rgb, palette)
        used = len(quant.getcolors(maxcolors=256) or [])
        pal_name = palette
    else:
        n = max(2, min(int(colors), 256))
        method = Image.Quantize.MEDIANCUT if hasattr(Image, "Quantize") else 0
        # 先中位切分出调色板，再按需用调色板做一次有序抖动。
        # 注意：quantize(method=MEDIANCUT, dither=...) 里的 dither 会被忽略
        # （中位切分本身不做误差扩散），必须走 palette 模式才真正生效。
        pal = rgb.quantize(colors=n, method=method)
        if dither:
            quant = rgb.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")
        else:
            quant = pal.convert("RGB")
        used = len(quant.getcolors(maxcolors=256) or [])
        pal_name = "auto"

    merged = quant.convert("RGBA")
    merged.putalpha(alpha)

    out = merged.resize((w, h), Image.NEAREST) if keep_size else merged
    meta = {
        "pixelSize": ps,
        "colors": used,
        "palette": pal_name,
        "dither": dither,
        "grid": [small_w, small_h],
        "size": [out.width, out.height],
        "downsample": downsample,
        "note": "每格单色 + 限定调色板，可再走硬缩放导出目标尺寸",
    }
    if shared is not None:
        meta["sharedPalette"] = len(shared_palette or [])
        import numpy as np

        before = np.asarray(rgb, dtype=np.int32)
        after = np.asarray(shared.convert("RGB"), dtype=np.int32)
        meta["paletteError"] = round(float(np.abs(before - after).mean()), 2)
    return out, meta
