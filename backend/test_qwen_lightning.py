"""Qwen-Image-Edit-2511 Lightning LoRA 对照测试。

对照三档（同一参考图、同一指令、同种子 42）：
  A. 40 步原版（fp8，已有结果 data/smoke/qwen-edit-smoke-0.png 直接复用）
  B. 8 步 Lightning LoRA（fp8 + LoRA，true_cfg=1.0 跳过第二次 CFG 前向）
  C. 4 步 Lightning LoRA（同上，steps=4）

通过标准：B/C 的外观保持度（头带/面罩/服装配色）与 A 相当，出图时间显著下降。
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = str(ROOT / "models" / "qwen-image-edit-2511")
LORA_DIR = ROOT / "models" / "qwen-image-edit-2511-lightning"
OUT_DIR = ROOT / "data" / "smoke"
REF_IMAGE = OUT_DIR / "zimage-smoke-0.png"
BASELINE_40 = OUT_DIR / "qwen-edit-smoke-0.png"  # A：40 步原版（上次冒烟产物）

INSTRUCTION = "保持角色外观、服装、配色完全不变，将姿势改为：蹲下戒备姿态，纯色背景"

LORA_8 = LORA_DIR / "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
LORA_4 = LORA_DIR / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def run_edit(pipe, ref, steps: int, true_cfg: float):
    """跑一次编辑，返回 (PIL.Image, 秒数)。只计推理时间。"""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(
        image=ref,
        prompt=INSTRUCTION,
        negative_prompt=" ",
        num_inference_steps=steps,
        true_cfg_scale=true_cfg,
        guidance_scale=1.0,
        generator=gen,
    )
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return out.images[0], dt, peak


def main() -> int:
    from PIL import Image as _Image

    from backend.app.providers.diffusers_local import DiffusersProvider, detect_family

    for p in (LORA_8, LORA_4):
        if not p.is_file():
            print(f"LoRA 不存在：{p}\n先跑 bash backend/fetch_qwen_lightning.sh")
            return 1
    ref = _Image.open(REF_IMAGE).convert("RGB")
    print(f"模型：{MODEL}（族：{detect_family(MODEL)}）")
    print(f"参考图：{REF_IMAGE}（{ref.size[0]}x{ref.size[1]}）")

    provider = DiffusersProvider({"model": MODEL})

    async def _go():
        # 只加载一次 fp8 底座（含量化），后续 LoRA 换入换出
        pipe = await asyncio.to_thread(provider._load, MODEL)
        results = {}
        for tag, lora, steps, cfg in (
            ("8steps", LORA_8, 8, 1.0),
            ("4steps", LORA_4, 4, 1.0),
        ):
            await asyncio.to_thread(pipe.load_lora_weights, str(lora))
            img, dt, peak = await asyncio.to_thread(run_edit, pipe, ref, steps, cfg)
            p = OUT_DIR / f"lightning-{tag}.png"
            img.save(p)
            print(f"[{tag}] steps={steps} true_cfg={cfg} 推理 {dt:.1f}s · 显存峰值 {peak:.1f}GB -> {p}")
            results[tag] = (img, dt)
            await asyncio.to_thread(pipe.unload_lora_weights)
        return results

    results = asyncio.run(_go())

    # 拼三联对比图：40 步原版 | 8 步 Lightning | 4 步 Lightning
    tiles = []
    labels = []
    if BASELINE_40.is_file():
        tiles.append(_Image.open(BASELINE_40).convert("RGB"))
        labels.append("40 steps (fp8, %.0fs)" % 250)
    for tag in ("8steps", "4steps"):
        if tag in results:
            tiles.append(results[tag][0].convert("RGB"))
            labels.append(f"{tag} Lightning ({results[tag][1]:.0f}s)")
    if len(tiles) >= 2:
        w = max(t.size[0] for t in tiles)
        h = max(t.size[1] for t in tiles) + 24
        canvas = _Image.new("RGB", (w * len(tiles) + 8 * (len(tiles) - 1), h), (24, 24, 24))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        for i, (t, lab) in enumerate(zip(tiles, labels)):
            x = i * (w + 8)
            canvas.paste(t, (x, 24))
            draw.text((x + 4, 6), lab, fill=(230, 230, 230))
        cp = OUT_DIR / "lightning-compare.png"
        canvas.save(cp)
        print(f"对比图：{cp}")

    print("LIGHTNING TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
