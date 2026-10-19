"""攻击特效笔刷：把矢量笔迹画成像素（移植自 FrameBaker 的 attackEffects）。

为什么用「压力点列 + 程序化笔触」而不是贴图：贴图缩放就糊，而且做不出
「一斩到底」的粗细变化。这里按折线生成等宽/变宽多边形，笔刷差异体现在
宽度包络、抖动、断笔与叠加方式上——同一份数据在任何画布尺寸下都清晰。

所有随机都走固定种子：同一份笔迹重渲染必须一模一样，否则导出前后对不上，
用户会以为是随机出错。
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter

BRUSHES = ("slash", "bristle", "dry", "spark", "echo")


def _hex_rgb(color: str) -> Tuple[int, int, int]:
    c = (color or "#ffffff").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except Exception:
        return (255, 255, 255)


def _normals(pts: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """每个点的法线（垂直于走向），端点沿用相邻段方向。"""
    out = []
    for i, p in enumerate(pts):
        if i == 0:
            dx, dy = pts[1][0] - p[0], pts[1][1] - p[1]
        elif i == len(pts) - 1:
            dx, dy = p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        ln = math.hypot(dx, dy) or 1.0
        out.append((-dy / ln, dx / ln))
    return out


def _strip(draw: ImageDraw.ImageDraw, pts, widths, rgb, alpha: int) -> None:
    """折线 → 变宽多边形（左岸正向 + 右岸回程），一次填充。"""
    if len(pts) < 2:
        return
    nor = _normals(pts)
    left, right = [], []
    for p, n, w in zip(pts, nor, widths):
        left.append((p[0] + n[0] * w / 2, p[1] + n[1] * w / 2))
        right.append((p[0] - n[0] * w / 2, p[1] - n[1] * w / 2))
    draw.polygon(left + right[::-1], fill=(rgb[0], rgb[1], rgb[2], alpha))
    # 圆头：折线拐角处单独补点，否则尖角内侧会缺一块
    for p, w in zip(pts[1:-1], widths[1:-1]):
        r = w / 2
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r),
                     fill=(rgb[0], rgb[1], rgb[2], alpha))


def _taper(n: int, strength: float = 1.0) -> List[float]:
    """两端细中间粗的宽度包络；strength 越大端部收得越尖。"""
    if n <= 1:
        return [1.0]
    out = []
    for i in range(n):
        t = i / (n - 1)
        env = math.sin(math.pi * t) ** (1.0 / max(0.2, strength))
        out.append(0.15 + 0.85 * env)
    return out


def _render_one(layer: Image.Image, stroke: Dict[str, Any], seed: int) -> None:
    rng = random.Random(seed)
    pts = [(float(p[0]), float(p[1])) for p in stroke["points"]]
    # 压力可省（前端画线时不一定带）：缺省按 1.0，别让一条缺字段的笔迹整帧崩掉
    pressure = [float(p[2]) if len(p) > 2 else 1.0 for p in stroke["points"]]
    size = float(stroke.get("size") or 8)
    rgb = _hex_rgb(stroke.get("color"))
    brush = stroke.get("brush") or "slash"
    draw = ImageDraw.Draw(layer)

    base = [size * pr for pr in pressure]

    if brush == "slash":
        env = _taper(len(pts), 1.4)
        _strip(draw, pts, [b * e for b, e in zip(base, env)], rgb, 255)
        # 刃口：沿中线再压一条更亮的细线，像素风里就是「高光刀锋」
        _strip(draw, pts, [max(1.0, b * e * 0.28) for b, e in zip(base, env)],
               (255, 255, 255), 150)
    elif brush == "bristle":
        for k in range(5):
            jitter = (rng.random() - 0.5) * size * 0.55
            off = [(x + jitter, y + jitter * 0.35) for x, y in pts]
            w = [max(0.8, b * 0.28 * (0.7 + rng.random() * 0.6)) for b in base]
            _strip(draw, off, w, rgb, 210)
    elif brush == "dry":
        # 断笔：按段随机丢弃，保留飞白（漫画里的干笔速度线）
        i = 0
        while i < len(pts) - 1:
            run = max(2, int(rng.uniform(2, 5)))
            seg = pts[i:i + run + 1]
            if len(seg) >= 2 and rng.random() > 0.32:
                w = [base[min(i + j, len(base) - 1)] * (0.6 + rng.random() * 0.4)
                     for j in range(len(seg))]
                _strip(draw, seg, w, rgb, 235)
            i += run
    elif brush == "spark":
        for (x, y), b in zip(pts, base):
            r = b * (0.55 + rng.random() * 0.6)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb + (235,))
            # 十字星芒：小尺寸下比圆点更像「火花」
            s = r * 2.2
            draw.line((x - s, y, x + s, y), fill=rgb + (160,), width=max(1, int(r * 0.3)))
            draw.line((x, y - s, x, y + s), fill=rgb + (160,), width=max(1, int(r * 0.3)))
    elif brush == "echo":
        # 残影：沿走向反向偏移，逐层变淡变细
        for k in range(3, 0, -1):
            dx = (pts[-1][0] - pts[0][0]) * 0.06 * k
            dy = (pts[-1][1] - pts[0][1]) * 0.06 * k
            off = [(x - dx, y - dy) for x, y in pts]
            w = [max(1.0, b * (1.0 - k * 0.18)) for b in base]
            _strip(draw, off, w, rgb, int(150 / k))
        _strip(draw, pts, base, rgb, 255)


def render(strokes: List[Dict[str, Any]], size: Tuple[int, int],
           frame: int = 0) -> Image.Image:
    """把笔迹画成一张透明图层（只画 from <= frame <= to 的笔迹）。"""
    w, h = size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gl = ImageDraw.Draw(glow)
    for i, s in enumerate(strokes):
        lo = int(s.get("from") or 0)
        hi = int(s.get("to", frame) or frame)
        if not (lo <= frame <= hi):
            continue
        _render_one(layer, s, seed=i * 7919 + 13)
        # 辉光：只在尺寸够大的笔迹上做，细线加辉光会糊成一团
        if float(s.get("size") or 0) >= 6 and s.get("brush") in ("slash", "spark", "echo"):
            _render_one(glow, s, seed=i * 7919 + 13)
    if layer.getbbox() is None:
        return layer
    out = glow.filter(ImageFilter.GaussianBlur(max(1.0, min(w, h) / 160))).point(
        lambda v: int(v * 0.55))
    return Image.alpha_composite(out, layer)


def palette_check(strokes: List[Dict[str, Any]]) -> List[str]:
    """笔迹里出现的颜色（前端色板与导出说明用）。"""
    seen: List[str] = []
    for s in strokes:
        c = str(s.get("color") or "").lower()
        if c and c not in seen:
            seen.append(c)
    return seen
