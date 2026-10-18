"""定位 Lightning LoRA 的开销来源：同 8 步，对比「无 LoRA / 有 LoRA」的推理耗时。

若两者差距大 → PEFT 包装是瓶颈，解决方案：把 LoRA 融进 fp8 权重（一次性合并，
推理时零包装开销）。若差距小 → 瓶颈在别处（offload/调度），换方向排查。
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = str(ROOT / "models" / "qwen-image-edit-2511")
LORA_8 = ROOT / "models" / "qwen-image-edit-2511-lightning" / "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
OUT_DIR = ROOT / "data" / "smoke"
REF_IMAGE = OUT_DIR / "zimage-smoke-0.png"
INSTRUCTION = "保持角色外观、服装、配色完全不变，将姿势改为：蹲下戒备姿态，纯色背景"


def timed_edit(pipe, ref, steps, true_cfg, tag):
    import torch

    gen = torch.Generator(device="cuda").manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = pipe(
        image=ref, prompt=INSTRUCTION, negative_prompt=" ",
        num_inference_steps=steps, true_cfg_scale=true_cfg,
        guidance_scale=1.0, generator=gen,
    )
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{tag}] steps={steps} lora={'Y' if getattr(pipe, '_active_lora', False) else 'N'} "
          f"推理 {dt:.1f}s · 峰值 {peak:.1f}GB", flush=True)
    return out.images[0], dt


def main() -> int:
    from PIL import Image as _Image

    from backend.app.providers.diffusers_local import DiffusersProvider

    ref = _Image.open(REF_IMAGE).convert("RGB")
    provider = DiffusersProvider({"model": MODEL})

    async def _go():
        pipe = await asyncio.to_thread(provider._load, MODEL)
        # (a) 无 LoRA，8 步（质量无所谓，只测速）
        await asyncio.to_thread(timed_edit, pipe, ref, 8, 1.0, "a-no-lora")
        # (b) 挂 LoRA，8 步
        await asyncio.to_thread(pipe.load_lora_weights, str(LORA_8))
        await asyncio.to_thread(timed_edit, pipe, ref, 8, 1.0, "b-with-lora")
        # (c) 再跑一次无 LoRA（确认不是首次预热）
        await asyncio.to_thread(pipe.unload_lora_weights)
        await asyncio.to_thread(timed_edit, pipe, ref, 8, 1.0, "c-no-lora-again")

    asyncio.run(_go())
    print("TIMING TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
