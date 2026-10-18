"""重跑 AI 延展步骤（接缝索引修复后），产出正确对比图。

背景：test_ai_pipelines.py 首次运行时的延展输出带有右接缝色偏 bug
（原图列号按接缝中心而非原图左边缘偏移），已在 ai_edit.py 修复，
本脚本用同一seed重出正确结果。
"""
import asyncio
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.pipelines import ai_edit  # noqa: E402

OUT = ROOT / "data" / "smoke"


async def main():
    scene = Image.open(ROOT / "data" / "style_samples" / "watercolor.png").convert("RGB")
    t0 = time.time()
    out, meta = await ai_edit.extend_horizontal_ai(scene, times=2, blend=6, seed=42)
    out.save(OUT / "aipipe-seamless.png")
    print(f"[seamless_ai rerun] {time.time() - t0:.0f}s | {meta}", flush=True)

    w = max(scene.width, out.width)
    h = max(scene.height, out.height)
    c = Image.new("RGB", (w * 2 + 12, h + 22), (24, 24, 24))
    c.paste(scene, (0, 22))
    c.paste(out, (w + 12, 22))
    d = ImageDraw.Draw(c)
    d.text((4, 4), "原场景 384px", fill=(230, 230, 230))
    d.text((w + 16, 4), "AI 延展 x2", fill=(230, 230, 230))
    c.save(OUT / "aipipe-seamless-compare.png")
    print("DONE", flush=True)


asyncio.run(main())
