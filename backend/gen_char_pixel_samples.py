"""人物像素化档位示例图生成：三档各一张，供档位卡片展示。

流程：
1. 用 Z-Image 合成一张写实人像当「用户照片」（不拿真人照片做示例素材）；
2. 三档各跑一次完整管线端点（含 AI 重绘 40 步）。生成时用保留背景——背景不写进
   生成指令（绿幕/透明都是本地后处理，原因见 charpixel.BACKGROUNDS 注释）；
3. mid / chibi 再用 /api/pipeline/char-pixel/repixel 换成透明背景（CPU，不占
   GPU），顺带验证「换背景不用重新生成」这条路；
4. 产出 data/char_pixel_samples/{fine,mid,chibi}.png + _source.png，
   另拼一张 4 宫格对比图到 data/smoke/ 供人工检查。

用法：PYTHONPATH=. python backend/gen_char_pixel_samples.py
（需要 GPU；三档共 3 次 40 步编辑，约 8-12 分钟）
"""
import asyncio
import io
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "data" / "char_pixel_samples"
SMOKE = ROOT / "data" / "smoke"
SEED = 7
STEPS = 40
# 三档固定用同一张脸、同一个种子：卡片并排时差异全部来自档位本身。
# 档位 → 示例图用的背景（生成全是 keep，mid/chibi 走 repixel 换成透明）。
PLANS = [("fine", "keep"), ("mid", "alpha"), ("chibi", "alpha")]
PORTRAIT_PROMPT = (
    "写实照片风格，一位年轻女性的半身肖像，长棕色头发，穿白色T恤，"
    "坐在公园长椅上，背景是绿树和草地，自然光，柔和"
)


def top_colors(im: Image.Image, n: int = 5) -> list[tuple]:
    """角色区域主色——用来客观发现「整片被绿幕染色」这类问题。"""
    px = [p for p in im.convert("RGBA").getdata() if p[3] > 200]
    if not px:
        return []
    c = Counter((p[0], p[1], p[2]) for p in px)
    return [(col, round(k / len(px) * 100, 1)) for col, k in c.most_common(n)]


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

    client = TestClient(app)
    results: dict[str, Image.Image] = {}
    created: list[str] = []

    for level, show_bg in PLANS:
        buf = io.BytesIO()
        photo.save(buf, "PNG")
        buf.seek(0)
        t0 = time.time()
        r = client.post(
            "/api/pipeline/char-pixel",
            data={"level": level, "background": "keep", "seed": str(SEED), "steps": str(STEPS)},
            files={"file": ("photo.png", buf, "image/png")})
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"[{level}] FAIL {r.status_code}: {r.text[:400]}", flush=True)
            return 1
        j = r.json()
        created += [j["asset"]["id"], j["draft"]["id"]]
        print(f"[{level}/keep] {dt:.0f}s · {j['meta']['grid']}格/"
              f"{j['meta']['pixelize']['colors']}色", flush=True)

        final_id = j["asset"]["id"]
        if show_bg != "keep":
            # 换背景走 repixel：毫秒级、不占 GPU
            rp = client.post("/api/pipeline/char-pixel/repixel", json={
                "draftAssetId": j["draft"]["id"],
                "grid": j["meta"]["grid"],
                "colors": j["meta"]["pixelize"]["colors"],
                "background": show_bg,
            })
            if rp.status_code != 200:
                print(f"[{level}] repixel FAIL: {rp.text[:300]}", flush=True)
                return 1
            rpj = rp.json()
            final_id = rpj["asset"]["id"]
            created.append(final_id)
            ratio = (rpj["meta"].get("matte") or {}).get("removedRatio", "?")
            print(f"[{level}] repixel → {show_bg} · 移除 {ratio}", flush=True)

        out = Image.open(storage.asset_path(storage.get_asset(final_id))).convert("RGBA")
        results[level] = out
        out.save(SAMPLES / f"{level}.png")
        for col, pct in top_colors(out):
            print(f"    {col} {pct}%", flush=True)

    for aid in created:
        client.delete(f"/api/assets/{aid}")
    print(f"cleaned {len(created)} test assets", flush=True)

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
    paste(results["mid"], 0, 1, "MID (alpha)")
    paste(results["chibi"], 1, 1, "CHIBI (alpha)")
    out_path = SMOKE / "char-pixel-compare.png"
    grid_img.save(out_path)
    print("compare ->", out_path, flush=True)
    return 0


sys.exit(asyncio.run(main()))
