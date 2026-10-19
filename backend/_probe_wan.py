"""一次性探针（跑完即删）：Wan2.2 TI2V-5B 真机验证「视频 → 精灵」全链路。

链路：Z-Image 出锚点图（480×832，纯绿底）→ Wan 生成 33 帧行走（4k+1）
→ 逐帧抠图 + 共享缩放/底部对齐 → 全局锁色 → 拼精灵表 + 一致性指标
→ 存一张 N 宫格联系图供肉眼复核。

为什么先跑 33 帧而不是 81 帧：首次验证要确认显存与耗时，33 帧（1.4s @24fps）
足够看动作连贯性，跑通再放开。
"""
import asyncio
import io
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOKE = ROOT / "data" / "smoke"
MODEL_WAN = str(ROOT / "models" / "wan2.2-ti2v-5b")
MODEL_TXT = str(ROOT / "models" / "z-image-turbo")
CHARACTER = "灰白英短猫武士，红披风，腰挎武士刀"
ACTION = ("画面中的角色在原地行走，双脚交替蹬地，手臂与腿反向摆动，披风随步伐轻微晃动；"
          "角色位置、大小、朝向始终不变，没有位移、没有镜头运动、没有起停；背景保持纯绿色。")
W, H = 480, 832
FRAMES = 33
STEPS = 30


async def main() -> int:
    SMOKE.mkdir(parents=True, exist_ok=True)
    from fastapi.testclient import TestClient

    from backend.app import storage
    from backend.app.main import app
    from backend.app.providers.diffusers_local import DiffusersProvider

    # 1. 锚点图（Z-Image 文生图，绿底）
    txt = DiffusersProvider({"model": MODEL_TXT})
    t0 = time.time()
    res = await txt.generate(
        f"{CHARACTER}。游戏角色正面立绘，全身，双脚踩在同一水平线上，"
        "纯绿色背景 #00FF00，无渐变、无纹理、无文字、无水印、无边框。",
        W, H, seed=11)
    anchor = res.images[0].convert("RGB")
    print(f"[anchor] {time.time() - t0:.0f}s {anchor.size}", flush=True)
    DiffusersProvider.unload()

    # 2. Wan 图生视频
    wan = DiffusersProvider({"model": MODEL_WAN})
    t0 = time.time()
    vres = await wan.generate_video(ACTION, anchor, width=W, height=H,
                                    steps=STEPS, seed=11, num_frames=FRAMES, fps=24,
                                    max_side=832)
    frames = [f.convert("RGB") for f in vres.images]
    print(f"[wan] {time.time() - t0:.0f}s · {len(frames)} 帧 · {vres.meta.get('seconds')}s · "
          f"steps {vres.meta.get('steps')}", flush=True)

    # 3. 本地后处理：抠图 + 规整 + 全局锁色 + 拼表（复用 compose，含一致性指标）
    from backend.app.pipelines import sprite_loop as sl

    sheet, norm, index, meta = sl.compose(frames, cols=6, size=128, anchor="bottom",
                                          matte=True, tolerance=72, colors=24)
    print(f"[compose] 色板 {meta.get('palette', {}).get('colors')} 色 · "
          f"指标 seam_ratio={meta['metrics'].get('seam_ratio')} "
          f"step_mean={meta['metrics'].get('step_mean')} "
          f"off_palette={meta['metrics'].get('off_palette_colors_max')}", flush=True)
    for v in meta.get("metricsVerdict", []):
        print("   ·", v, flush=True)

    client = TestClient(app)
    created = [storage.save_asset(f, kind="workflow", name=f"wan-frame-{i:03d}",
                                  meta={"pipeline": "wan2.2-ti2v", "index": i})["id"]
               for i, f in enumerate(frames)]
    sheet_item = storage.save_asset(sheet, kind="sheet", name="wan-sprite-sheet",
                                    meta={"pipeline": "wan2.2-ti2v", **meta})
    created.append(sheet_item["id"])

    # 4. 联系图：锚点 + 精灵表 + 每 3 帧取样
    picks = [frames[i] for i in range(0, len(frames), max(1, len(frames) // 8))]
    cell = 160
    cols = min(6, len(picks) + 1)
    rows = (len(picks) + 1 + cols - 1) // cols
    canvas = Image.new("RGB", (cell * cols + 8 * (cols - 1), cell * rows + 8 * (rows - 1) + 22),
                       (28, 28, 32))

    def put(im, i, label):
        x, y = (i % cols) * (cell + 8), (i // cols) * (cell + 8) + 22
        base = Image.new("RGBA", (cell, cell), (22, 22, 26, 255))
        r = min(cell / im.width, cell / im.height)
        small = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
        base.alpha_composite(small.convert("RGBA"), ((cell - small.width) // 2,
                                                     (cell - small.height) // 2))
        canvas.paste(base.convert("RGB"), (x, y))
        ImageDraw.Draw(canvas).text((x + 4, y + 4), label, fill=(0, 255, 136))

    put(anchor, 0, "ANCHOR")
    for i, f in enumerate(picks):
        put(f, i + 1, f"f{i * max(1, len(frames) // 8)}")
    put(sheet.convert("RGB"), 1 + len(picks), "SHEET") if (1 + len(picks)) < rows * cols else None
    p = SMOKE / "wan-sprite.png"
    canvas.save(p)
    print("contact ->", p, flush=True)
    print(f"sheet {sheet_item['width']}x{sheet_item['height']} · {len(norm)} 帧 · "
          f"asset {sheet_item['id']}", flush=True)

    for aid in created:
        client.delete(f"/api/assets/{aid}")
    print("cleaned", len(created), "assets", flush=True)
    return 0


sys.exit(asyncio.run(main()))
