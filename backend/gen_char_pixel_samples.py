"""人物像素化档位示例图生成：三档各一张，供档位卡片展示。

流程：
1. 用 Z-Image 合成一张写实人像当「用户照片」（不拿真人照片做示例素材）；
2. 三档各走一次完整管线端点（含 AI 重绘 40 步），背景分别覆盖
   keep / green / alpha 三条路径；
3. mid 档的绿幕产物再本地走一次 cut_out 得到透明版——这正是
   「绿幕 → 透明」的二次处理路径，示例图同时展示两种背景选项的效果；
4. 产出 data/char_pixel_samples/{fine,mid,chibi}.png + _source.png，
   另拼一张 4 宫格对比图到 data/smoke/ 供人工检查。

用法：PYTHONPATH=. python backend/gen_char_pixel_samples.py
（需要 GPU；三档共 3 次 40 步编辑，约 8-12 分钟）
"""
import asyncio
import io
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "data" / "char_pixel_samples"
SMOKE = ROOT / "data" / "smoke"
SEED = 7
STEPS = 40
# 三档固定用同一张脸、同一个种子：卡片并排时差异全部来自档位本身
PLANS = [("fine", "keep"), ("mid", "green"), ("chibi", "alpha")]
PORTRAIT_PROMPT = (
    "写实照片风格，一位年轻女性的半身肖像，长棕色头发，穿白色T恤，"
    "坐在公园长椅上，背景是绿树和草地，自然光，柔和"
)


async def main() -> int:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    SMOKE.mkdir(parents=True, exist_ok=True)

    from backend.app.providers.diffusers_local import DiffusersProvider

    p = DiffusersProvider({"model": str(ROOT / "models" / "z-image-turbo")})
    t0 = time.time()
    res = await p.generate(PORTRAIT_PROMPT, 640, 640, seed=SEED)
    src_path = SAMPLES / "_source.png"
    photo = res.images[0].convert("RGB")
    photo.save(src_path)
    print(f"[photo] {time.time() - t0:.0f}s -> {src_path}", flush=True)
    DiffusersProvider.unload()

    from fastapi.testclient import TestClient

    from backend.app import storage
    from backend.app.main import app
    from backend.app.pipelines import charpixel as cp

    client = TestClient(app)
    results: dict[str, Image.Image] = {}
    created: list[str] = []

    for level, bg in PLANS:
        buf = io.BytesIO()
        photo.save(buf, "PNG")
        buf.seek(0)
        t0 = time.time()
        r = client.post(
            "/api/pipeline/char-pixel",
            data={"level": level, "background": bg, "seed": str(SEED), "steps": str(STEPS)},
            files={"file": ("photo.png", buf, "image/png")})
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"[{level}] FAIL {r.status_code}: {r.text[:400]}", flush=True)
            return 1
        j = r.json()
        meta = j["meta"]
        created += [j["asset"]["id"], j["draft"]["id"]]
        print(f"[{level}/{bg}] {dt:.0f}s · {meta['grid']}格 / "
              f"{meta['pixelize']['colors']}色 · {j['asset']['width']}x{j['asset']['height']}",
              flush=True)

        out = Image.open(storage.asset_path(j["asset"])).convert("RGBA")
        if level == "mid" and bg == "green":
            out, minfo = cp.cut_out(out)
            print(f"[mid] 绿幕→透明：移除 {minfo['removedRatio'] * 100:.1f}% · "
                  f"洗边 {minfo.get('edgePixelsRecolored')}px", flush=True)
        results[level] = out
        out.save(SAMPLES / f"{level}.png")

    removed = 0
    for aid in created:
        if client.delete(f"/api/assets/{aid}").status_code == 200:
            removed += 1
    print(f"cleaned {removed}/{len(created)} test assets", flush=True)

    cell = 384
    grid_img = Image.new("RGB", (cell * 2 + 12, cell * 2 + 12), (32, 32, 36))

    def paste(im: Image.Image, ix: int, iy: int, label: str) -> None:
        x, y = ix * (cell + 12), iy * (cell + 12)
        base = Image.new("RGBA", (cell, cell), (24, 24, 28, 255))
        ratio = min(cell / im.width, cell / im.height)
        small = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))),
                          Image.LANCZOS)
        base.alpha_composite(small.convert("RGBA"),
                             ((cell - small.width) // 2, (cell - small.height) // 2))
        grid_img.paste(base.convert("RGB"), (x, y))
        ImageDraw.Draw(grid_img).text((x + 6, y + 6), label, fill=(0, 255, 136))

    paste(Image.open(src_path), 0, 0, "SOURCE")
    paste(results["fine"], 1, 0, "FINE (keep)")
    paste(results["mid"], 0, 1, "MID (green->alpha)")
    paste(results["chibi"], 1, 1, "CHIBI (alpha)")
    out_path = SMOKE / "char-pixel-compare.png"
    grid_img.save(out_path)
    print("compare ->", out_path, flush=True)
    return 0


sys.exit(asyncio.run(main()))
