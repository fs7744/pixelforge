"""Z-Image 直连管线诊断：绕开 provider，带每步回调，定位慢点。

用法：
    "C:/Users/Windows/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe" backend/diag_zimage.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "z-image-turbo")
PROMPT = "一个Q版忍者猫忍者，像素风格，粗黑描边，站立姿势，挥动手里的小太刀，纯色背景，游戏角色素材"


def main() -> int:
    import torch

    if "--bench" in sys.argv:
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark = True", flush=True)

    from diffusers import ZImagePipeline

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] 加载管线 ...", flush=True)
    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[{time.time()-t0:6.1f}s] 管线就绪，显存 {torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

    step_times = []

    def cb(pipe_, step, timestep, cbk):
        now = time.time()
        if step_times:
            print(f"[{now-t0:6.1f}s] step {step} 上一步耗时 {now-step_times[-1]:.2f}s", flush=True)
        else:
            print(f"[{now-t0:6.1f}s] step {step} 开始", flush=True)
        step_times.append(now)
        return cbk

    gen = torch.Generator(device="cuda").manual_seed(42)
    print(f"[{time.time()-t0:6.1f}s] 开始推理（1024x1024, 9 步）...", flush=True)
    out = pipe(
        prompt=PROMPT,
        num_inference_steps=9,
        guidance_scale=0.0,
        height=1024,
        width=1024,
        generator=gen,
        callback_on_step_end=cb,
    )
    print(f"[{time.time()-t0:6.1f}s] 推理完成，峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.1f} GB", flush=True)

    if len(step_times) >= 2:
        diffs = [step_times[i+1] - step_times[i] for i in range(len(step_times)-1)]
        print(f"各步耗时：{[f'{d:.2f}s' for d in diffs]}", flush=True)

    out_p = ROOT / "data" / "smoke" / "zimage-diag.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.images[0].save(out_p)
    print(f"已保存：{out_p}，总耗时 {time.time()-t0:.1f}s", flush=True)
    print("DIAG DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
