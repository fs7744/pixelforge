"""MVP 冒烟测试：不依赖前端，直接验证管线与生成链路。

    python backend/smoke_test.py
"""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from backend.app.pipelines import matte, resize, sheet
from backend.app.providers.mock import MockProvider

OUT = Path(__file__).resolve().parents[1] / "data" / "tmp"
OUT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    provider = MockProvider({})
    res = await provider.generate("knight walk cycle pixel art", 512, 512, n=1, seed=42)
    sheet_img = res.images[0]
    sheet_img.save(OUT / "smoke-sheet.png")
    print(f"[1] generate  → {sheet_img.size}  mode={sheet_img.mode}")

    cut, meta = matte.connected_matte(sheet_img, tolerance=80, feather=5)
    cut.save(OUT / "smoke-matte.png")
    print(f"[2] matte     → bg={meta['background']} removed={meta['removedRatio']*100:.1f}%")

    small = resize.hard_resize(cut, (144, 144))
    rpg, rmeta = resize.rpgmaker(cut, "v1")
    rpg.save(OUT / "smoke-rpgmaker.png")
    print(f"[3] resize    → hard={small.size} rpgmaker={rmeta['size']}")

    frames, index = sheet.split_sheet(sheet_img, 4, 4)
    print(f"[4] split     → {len(frames)} frames, each {index['frame_size']['w']}×{index['frame_size']['h']}")

    merged, _ = sheet.compose_sheet(frames, 4)
    print(f"[5] compose   → {merged.size}")

    # 透明率校验：抠图后四角必须透明
    alpha = cut.getchannel("A")
    corners = [alpha.getpixel((0, 0)), alpha.getpixel((cut.width - 1, 0)),
               alpha.getpixel((0, cut.height - 1)), alpha.getpixel((cut.width - 1, cut.height - 1))]
    assert all(a == 0 for a in corners), f"四角未透明: {corners}"
    print("[6] alpha     → 四角透明 OK")

    png = io.BytesIO()
    rpg.save(png, "PNG")
    print(f"[7] export    → rpgmaker png {len(png.getvalue()) // 1024} KB")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
