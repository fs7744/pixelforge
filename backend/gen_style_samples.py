"""画风示例图生成：每种风格一张样本，供前端画风 chip 的悬浮 tooltip 展示。

设计要点：
- 用本地 Z-Image-Turbo 跑（中文提示词直喂，9 步蒸馏，~5s/张）；
- 主体按风格 tags[0] 选择（角色/场景/道具/UI），固定 seed=42，保证可复现；
- 像素类画风出图后按 suggestPost 参数接 pixelize 后处理——tooltip 里展示的
  就是「按建议走完後处理」的最终观感，而不是半成品；
- 输出 data/style_samples/<id>.png，另拼一张 _contact_sheet.png 便于人工检查。

用法：
    "C:/Users/Windows/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe" backend/gen_style_samples.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "z-image-turbo")
OUT_DIR = ROOT / "data" / "style_samples"
SIZE = 384  # 16 对齐；tooltip 展示约 224px，足够
SEED = 42

# 主体按 tags[0] 匹配；同一 tag 共用同一主体，跨风格可直接对比
SUBJECTS = {
    "角色": "一个可爱的猫咪骑士游戏角色，全身站立姿势，面向观众，纯色深灰背景",
    "场景": "一个宁静的奇幻森林村庄场景，小木屋与帐篷，溪流，白天",
    "道具": "一把发光的魔法剑，游戏物品道具，居中展示，纯色背景",
    "UI": "一个金色的爱心生命值图标，游戏UI元素，居中构图，简洁",
}


async def main() -> int:
    sys.path.insert(0, str(ROOT))

    if not Path(MODEL).is_dir():
        print(f"模型目录不存在：{MODEL}\n先跑 bash backend/fetch_zimage.sh")
        return 1

    from backend.app.providers.diffusers_local import DiffusersProvider
    from backend.app.pipelines.pixelize import pixelize
    from backend.app.styles import STYLES

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provider = DiffusersProvider({"model": MODEL})
    print(f"模型：{MODEL}（{len(STYLES)} 种画风，{SIZE}px，seed={SEED}）", flush=True)

    tiles: list[tuple[Image.Image, str]] = []
    for s in STYLES:
        tag = s["tags"][0] if s["tags"] else "角色"
        subject = SUBJECTS.get(tag, SUBJECTS["角色"])
        prompt = subject + "，" + s["prompt"]
        t0 = time.time()
        result = await provider.generate(prompt=prompt, width=SIZE, height=SIZE, seed=SEED)
        img = result.images[0]

        note = ""
        px = s.get("pixelize")
        if px:
            img, _meta = pixelize(
                img, colors=px["colors"], pixel_size=px["block"], dither=False, palette="auto"
            )
            note = f" +pixelize({px['block']}x/{px['colors']}色)"

        out = OUT_DIR / f"{s['id']}.png"
        img.save(out)
        tiles.append((img, s["name"]))
        print(f"  [{len(tiles):2d}/{len(STYLES)}] {s['id']:<14s} {time.time() - t0:5.1f}s{note}", flush=True)

    # 联系表：4 列网格 + 风格名标注，供人工检查
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    cell = 256
    sheet = Image.new("RGB", (cols * cell, rows * cell), (10, 10, 10))
    draw = ImageDraw.Draw(sheet)
    for i, (im, name) in enumerate(tiles):
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(im.resize((cell, cell), Image.NEAREST), (x, y))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(60, 60, 60))
        draw.text((x + 6, y + 4), name, fill=(0, 255, 136))
    sheet_path = OUT_DIR / "_contact_sheet.png"
    sheet.save(sheet_path)
    print(f"联系表：{sheet_path}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
