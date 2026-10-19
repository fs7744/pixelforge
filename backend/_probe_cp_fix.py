"""一次性探针（跑完即删）：修绿幕染色后再验 mid/chibi。

对比上一版（cp2）与新一版（cp3），同时打印角色区域的主色——
「整片橄榄绿」这种问题肉眼看图能发现，但主色统计能给出客观阈值。
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
OUT = ROOT / "data" / "smoke" / "cp3"
PREV = ROOT / "data" / "smoke" / "cp2"
PLANS = [("mid", "green"), ("chibi", "alpha")]
SEED = 7


def top_colors(im: Image.Image, n: int = 6) -> list[tuple]:
    px = [p for p in im.convert("RGBA").getdata() if p[3] > 200]
    if not px:
        return []
    c = Counter((p[0], p[1], p[2]) for p in px)
    return [(col, round(k / len(px) * 100, 1)) for col, k in c.most_common(n)]


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SAMPLES / "_source.png").convert("RGB")

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
        out.save(OUT / f"{level}.png")
        results[level] = out
        print(f"[{level}/{bg}] {dt:.0f}s · {j['meta']['grid']}格/"
              f"{j['meta']['pixelize']['colors']}色", flush=True)
        for col, pct in top_colors(out):
            print(f"    {col} {pct}%", flush=True)

    for aid in created:
        client.delete(f"/api/assets/{aid}")

    # 2 行对比：上=上一版，下=本版
    cell, gap = 420, 8
    canvas = Image.new("RGB", (cell * 2 + gap, (cell + 18) * 2 + gap), (28, 28, 32))

    def paste(im, col, row, label):
        x, y = col * (cell + gap), row * (cell + 18 + gap)
        base = Image.new("RGBA", (cell, cell), (22, 22, 26, 255))
        ratio = min(cell / im.width, cell / im.height)
        small = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))),
                          Image.LANCZOS)
        base.alpha_composite(small.convert("RGBA"),
                             ((cell - small.width) // 2, (cell - small.height) // 2))
        canvas.paste(base.convert("RGB"), (x, y + 18))
        ImageDraw.Draw(canvas).text((x + 4, y + 4), label, fill=(0, 255, 136))

    for i, key in enumerate(("mid", "chibi")):
        paste(Image.open(PREV / f"{key}.png"), i, 0, f"PREV {key.upper()}")
        paste(results[key], i, 1, f"NEW {key.upper()}")

    p = ROOT / "data" / "smoke" / "char-pixel-fix.png"
    canvas.save(p)
    print("compare ->", p, flush=True)
    return 0


sys.exit(asyncio.run(main()))
