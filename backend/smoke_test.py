"""MVP 冒烟测试：不依赖前端与模型，直接验证确定性管线。

    python backend/smoke_test.py

测试素材由本脚本自己画（4×4 纯色块网格 + 纯色底），不经过任何
生成模型——模拟引擎已移除，管线测试也不该依赖 GPU。
"""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from backend.app.pipelines import matte, resize, sheet

OUT = Path(__file__).resolve().parents[1] / "data" / "tmp"
OUT.mkdir(parents=True, exist_ok=True)

CELL = 32
BG = (42, 21, 63)  # 暗紫底，与管线默认一致
BLOCKS = [
    (0, 255, 136), (255, 51, 102), (91, 87, 235), (255, 213, 0),
    (10, 10, 10), (255, 255, 255), (255, 180, 162), (0, 200, 200),
]


def make_fake_sheet() -> Image.Image:
    """画一张 4×4 的假精灵表：每格一个纯色方块 + 黑描边。"""
    img = Image.new("RGB", (4 * CELL, 4 * CELL), BG)
    d = ImageDraw.Draw(img)
    for r in range(4):
        for c in range(4):
            ox, oy = c * CELL, r * CELL
            d.rectangle([ox + 6, oy + 6, ox + 25, oy + 25],
                        fill=BLOCKS[(r * 4 + c) % len(BLOCKS)],
                        outline=(10, 10, 10))
    return img


async def main() -> None:
    sheet_img = make_fake_sheet()
    sheet_img.save(OUT / "smoke-sheet.png")
    print(f"[1] 素材     → {sheet_img.size} 4×4 合成网格")

    cut, meta = matte.connected_matte(sheet_img, tolerance=80, feather=5)
    cut.save(OUT / "smoke-matte.png")
    print(f"[2] matte    → bg={meta['background']} removed={meta['removedRatio']*100:.1f}%")

    small = resize.hard_resize(cut, (144, 144))
    rpg, rmeta = resize.rpgmaker(cut, "v1")
    rpg.save(OUT / "smoke-rpgmaker.png")
    print(f"[3] resize   → hard={small.size} rpgmaker={rmeta['size']}")

    frames, index = sheet.split_sheet(sheet_img, 4, 4)
    print(f"[4] split    → {len(frames)} frames, each {index['frame_size']['w']}×{index['frame_size']['h']}")

    merged, _ = sheet.compose_sheet(frames, 4)
    print(f"[5] compose  → {merged.size}")

    # 透明率校验：抠图后四角必须透明
    alpha = cut.getchannel("A")
    corners = [alpha.getpixel((0, 0)), alpha.getpixel((cut.width - 1, 0)),
               alpha.getpixel((0, cut.height - 1)), alpha.getpixel((cut.width - 1, cut.height - 1))]
    assert all(a == 0 for a in corners), f"四角未透明: {corners}"
    print("[6] alpha    → 四角透明 OK")

    png = io.BytesIO()
    rpg.save(png, "PNG")
    print(f"[7] export   → rpgmaker png {len(png.getvalue())} bytes")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
