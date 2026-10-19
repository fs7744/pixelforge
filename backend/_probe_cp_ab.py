"""一次性探针（跑完即删）：人物像素化 A/B——旧提示词 vs 新提示词。

同一张合成人像（data/char_pixel_samples/_source.png）、同一种子，三档各跑一次
新提示词，输出到 data/smoke/cp2/，再拼一张「上=旧 / 下=新」的 4 列对比图。
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
OUT = ROOT / "data" / "smoke" / "cp2"
PLANS = [("fine", "keep"), ("mid", "green"), ("chibi", "alpha")]
SEED = 7


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src_path = SAMPLES / "_source.png"
    photo = Image.open(src_path).convert("RGB")

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
            data={"level": level, "background": bg, "seed": str(SEED), "steps": "40"},
            files={"file": ("photo.png", buf, "image/png")})
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"[{level}] FAIL {r.status_code}: {r.text[:300]}", flush=True)
            return 1
        j = r.json()
        created += [j["asset"]["id"], j["draft"]["id"]]
        out = Image.open(storage.asset_path(j["asset"])).convert("RGBA")
        if level == "mid" and bg == "green":
            out, _ = cp.cut_out(out)
        results[level] = out
        out.save(OUT / f"{level}.png")
        print(f"[{level}/{bg}] {dt:.0f}s · {j['meta']['grid']}格/"
              f"{j['meta']['pixelize']['colors']}色", flush=True)

    for aid in created:
        client.delete(f"/api/assets/{aid}")

    # 4 列 × 2 行：列 = 原图 / 三档，行 = 旧 / 新
    cols = ["src", "fine", "mid", "chibi"]
    labels_top = {"src": "SOURCE", "fine": "OLD FINE", "mid": "OLD MID", "chibi": "OLD CHIBI"}
    labels_bot = {"src": "", "fine": "NEW FINE", "mid": "NEW MID", "chibi": "NEW CHIBI"}
    cell = 360
    gap = 8
    canvas = Image.new("RGB", (cell * 4 + gap * 3, (cell + 18) * 2 + gap), (28, 28, 32))

    def paste(im: Image.Image, col: int, row: int, label: str) -> None:
        x = col * (cell + gap)
        y = row * (cell + 18 + gap)
        base = Image.new("RGBA", (cell, cell), (22, 22, 26, 255))
        ratio = min(cell / im.width, cell / im.height)
        small = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))),
                          Image.LANCZOS)
        base.alpha_composite(small.convert("RGBA"),
                             ((cell - small.width) // 2, (cell - small.height) // 2))
        canvas.paste(base.convert("RGB"), (x, y + 18))
        if label:
            ImageDraw.Draw(canvas).text((x + 4, y + 4), label, fill=(0, 255, 136))

    for i, key in enumerate(cols):
        src_old = photo if key == "src" else Image.open(SAMPLES / f"{key}.png")
        paste(src_old, i, 0, labels_top[key])
        if key == "src":
            paste(photo, i, 1, "")
        else:
            paste(Image.open(OUT / f"{key}.png"), i, 1, labels_bot[key])

    p = ROOT / "data" / "smoke" / "char-pixel-ab.png"
    canvas.save(p)
    print("ab ->", p, flush=True)
    return 0


sys.exit(asyncio.run(main()))
