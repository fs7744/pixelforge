"""内置模拟引擎：无需任何模型即可跑通全链路（生成 → 抠图 → 规整 → 导出）。

产出一张 4×4 的像素角色精灵表，纯色背景 + 黑色描边，
用于验证色度键抠图、硬缩放、序列帧拆分等管线。
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from .base import BaseProvider, ProviderResult

BACKGROUNDS = ["#2A153F", "#1B4332", "#3A1F2B", "#12303A"]
PALETTES = [
    # (主色, 副色, 皮肤, 金属)
    ("#00FF88", "#0A0A0A", "#FFD500", "#FFFFFF"),
    ("#FF3366", "#0A0A0A", "#FFD500", "#FFFFFF"),
    ("#5B57EB", "#0A0A0A", "#FFB4A2", "#FFD500"),
    ("#FFD500", "#0A0A0A", "#FF3366", "#FFFFFF"),
]


def _hash_pick(prompt: str, arr):
    h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
    return arr[h % len(arr)]


def _draw_sprite(d: ImageDraw.ImageDraw, ox: int, oy: int, frame: int, pal) -> None:
    """在 32×32 单元格内绘制一个 4 帧行走循环的像素小人。"""
    main, outline, skin, metal = pal
    cell = 32
    cx = ox + cell // 2
    bob = 1 if frame % 2 else 0

    # 腿（左右交替）
    leg_shift = [-2, 0, 2, 0][frame % 4]
    d.rectangle([cx - 5, oy + 22, cx - 2 + leg_shift, oy + 29], fill=main, outline=outline)
    d.rectangle([cx + 2, oy + 22, cx + 5 - leg_shift, oy + 29], fill=main, outline=outline)

    # 躯干
    d.rectangle([cx - 6, oy + 12 + bob, cx + 6, oy + 23], fill=main, outline=outline)
    # 腰带
    d.rectangle([cx - 6, oy + 20, cx + 6, oy + 22], fill=metal, outline=outline)

    # 手臂
    arm_shift = [0, -2, 0, 2][frame % 4]
    d.rectangle([cx - 10, oy + 13 + bob + arm_shift, cx - 7, oy + 21 + bob], fill=main, outline=outline)
    d.rectangle([cx + 7, oy + 13 + bob - arm_shift, cx + 10, oy + 21 + bob], fill=main, outline=outline)

    # 头
    d.rectangle([cx - 6, oy + 3 + bob, cx + 6, oy + 12 + bob], fill=skin, outline=outline)
    # 头盔
    d.rectangle([cx - 7, oy + 2 + bob, cx + 7, oy + 6 + bob], fill=metal, outline=outline)
    # 眼睛
    d.point([(cx - 3, oy + 8 + bob), (cx + 3, oy + 8 + bob)], fill=outline)
    d.point([(cx - 3, oy + 9 + bob), (cx + 3, oy + 9 + bob)], fill=outline)


def render_sprite_sheet(prompt: str, width: int, height: int, seed: Optional[int] = None,
                        cols: int = 4, rows: int = 4) -> Image.Image:
    bg = _hash_pick(prompt + "|bg", BACKGROUNDS)
    pal = _hash_pick(prompt + "|pal", PALETTES)
    if seed is not None:
        rng = random.Random(seed)
        pal = PALETTES[rng.randrange(len(PALETTES))]

    cell = 32
    base = Image.new("RGB", (cols * cell, rows * cell), bg)
    d = ImageDraw.Draw(base)
    for r in range(rows):
        for c in range(cols):
            _draw_sprite(d, c * cell, r * cell, frame=c, pal=pal)

    # 最近邻硬放大，保持像素锐利
    scale = max(1, min(width // (cols * cell), height // (rows * cell)))
    up = base.resize((cols * cell * scale, rows * cell * scale), Image.NEAREST)
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(up, ((width - up.width) // 2, (height - up.height) // 2))
    return canvas


def render_single_sprite(prompt: str, width: int, height: int, seed: Optional[int] = None,
                         pose: int = 0) -> Image.Image:
    """单帧姿势：同角色同配色，仅四肢姿态不同（用于动作关键帧）。"""
    bg = _hash_pick(prompt + "|bg", BACKGROUNDS)
    pal = _hash_pick(prompt + "|pal", PALETTES)
    if seed is not None:
        rng = random.Random(seed)
        pal = PALETTES[rng.randrange(len(PALETTES))]

    base = Image.new("RGB", (32, 32), bg)
    d = ImageDraw.Draw(base)
    _draw_sprite(d, 0, 0, frame=pose % 4, pal=pal)

    scale = max(1, min(width // 32, height // 32))
    up = base.resize((32 * scale, 32 * scale), Image.NEAREST)
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(up, ((width - up.width) // 2, (height - up.height) // 2))
    return canvas


class MockProvider(BaseProvider):
    id = "mock"
    label = "内置模拟引擎"

    async def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 28,
        cfg: float = 7.5,
        seed: Optional[int] = None,
        n: int = 1,
        ref: Optional[Image.Image] = None,
    ) -> ProviderResult:
        n = max(1, min(n, 6))
        if n == 1:
            images = [render_sprite_sheet(prompt, width, height, seed)]
        else:
            # 多张：同角色同配色，不同四肢姿态，模拟动作关键帧
            images = [
                render_single_sprite(prompt, width, height, seed, pose=i)
                for i in range(n)
            ]
        return ProviderResult(
            images=images,
            meta={
                "provider": "mock",
                "model": "mock-pixel-v1",
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "note": "本地模拟产出，用于跑通管线。配置真实 Provider 后即可接入模型。",
            },
        )

    async def test(self) -> Dict[str, Any]:
        return {"ok": True, "message": "内置模拟引擎可用（无需模型）"}
