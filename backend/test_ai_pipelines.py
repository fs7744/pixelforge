"""端到端测试：三个 AI 编辑管线 + 大族显存换出。

覆盖：
0. 先用 Z-Image 生成一张（占住显存），随后 Edit 管线应打印「显存换出」并不 OOM；
1. AI 去水印：合成假水印 → 移除，对比；
2. AI 左右延展：水彩村庄场景 ×2；
3. 画风重统一：忍者猫 → 复古16-bit像素（自动 pixelize 归一化）/ 水彩。

用法：python backend/test_ai_pipelines.py（后台跑，约 15 分钟）
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


def fake_watermark(img: Image.Image) -> Image.Image:
    """合成一个「AI 图自带」式水印：右下角白底红字 logo 条。"""
    wm = img.copy()
    d = ImageDraw.Draw(wm)
    x0, y0 = wm.width - 360, wm.height - 140
    d.rectangle([x0, y0, x0 + 320, y0 + 100], fill=(255, 255, 255), outline=(200, 30, 30), width=4)
    d.rectangle([x0 + 20, y0 + 24, x0 + 180, y0 + 44], fill=(200, 30, 30))
    d.rectangle([x0 + 20, y0 + 60, x0 + 240, y0 + 76], fill=(120, 120, 120))
    return wm


def side_by_side(pairs, path, labels=None):
    imgs = [p for pair in pairs for p in pair]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)
    c = Image.new("RGB", (w * len(imgs) + 12 * (len(imgs) - 1), h + 22), (24, 24, 24))
    x = 0
    for i, im in enumerate(imgs):
        c.paste(im, (x, 22))
        if labels:
            d = ImageDraw.Draw(c)
            d.text((x + 4, 4), labels[i], fill=(230, 230, 230))
        x += w + 12
    c.save(path)


async def main():
    cat = Image.open(OUT / "zimage-smoke-0.png").convert("RGB")

    # 0. 先让 Z-Image 占住显存，验证换出
    from backend.app.providers.diffusers_local import DiffusersProvider

    z = DiffusersProvider({"model": "models/z-image-turbo"})
    res = await z.generate("一只戴着头带的忍者猫，站立持剑，纯色背景", 512, 512, seed=7)
    zimg = res.images[0].convert("RGB")
    zimg.save(OUT / "aipipe-zimage-frame1.png")
    print("[step0] Z-Image 帧1 已生成（占住显存），接下来 Edit 管线应触发换出", flush=True)

    # 1. AI 去水印
    wm = fake_watermark(cat)
    wm.save(OUT / "aipipe-wm-input.png")
    t0 = time.time()
    out, meta = await ai_edit.remove_watermark_ai(wm, seed=42, max_side=768)
    out.save(OUT / "aipipe-wm-output.png")
    print(f"[watermark_ai] {time.time() - t0:.0f}s | {meta}", flush=True)

    # 2. AI 左右延展
    scene = Image.open(ROOT / "data" / "style_samples" / "watercolor.png").convert("RGB")
    t0 = time.time()
    out, meta = await ai_edit.extend_horizontal_ai(scene, times=2, seed=42)
    out.save(OUT / "aipipe-seamless.png")
    print(f"[seamless_ai] {time.time() - t0:.0f}s | {meta}", flush=True)

    # 3. 画风重统一 ×2
    t0 = time.time()
    out, meta = await ai_edit.restyle(cat, "pixel_16bit", seed=42, max_side=768)
    out.save(OUT / "aipipe-restyle-pixel.png")
    print(f"[restyle pixel_16bit] {time.time() - t0:.0f}s | {meta}", flush=True)

    t0 = time.time()
    out, meta = await ai_edit.restyle(cat, "watercolor", seed=42, max_side=768)
    out.save(OUT / "aipipe-restyle-watercolor.png")
    print(f"[restyle watercolor] {time.time() - t0:.0f}s | {meta}", flush=True)

    # 对比图
    side_by_side([(wm, Image.open(OUT / "aipipe-wm-output.png"))],
                 OUT / "aipipe-watermark-compare.png",
                 ["原图+合成水印", "AI 去水印"])
    side_by_side([(scene, Image.open(OUT / "aipipe-seamless.png"))],
                 OUT / "aipipe-seamless-compare.png",
                 ["原场景 384px", "AI 延展 ×2"])
    side_by_side([(cat, Image.open(OUT / "aipipe-restyle-pixel.png")),
                  (cat, Image.open(OUT / "aipipe-restyle-watercolor.png"))],
                 OUT / "aipipe-restyle-compare.png",
                 ["原图", "→ 复古16-bit像素", "原图", "→ 水彩手绘"])
    print("ALL DONE", flush=True)


asyncio.run(main())
