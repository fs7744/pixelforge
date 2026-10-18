"""最小复现：定位「导入 backend 包 → 推理变慢」的因果链。

三段，每段加载一次管线并出一张图：
  1. 纯净：只 import torch/diffusers
  2. 先 import backend.app.providers.diffusers_local，再走直连管线
  3. 完全走 provider.generate()

用法：python backend/diag_import.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "z-image-turbo")
PROMPT = "一个Q版忍者猫忍者，像素风格，粗黑描边，站立姿势，挥动手里的小太刀，纯色背景，游戏角色素材"


def run_once(tag: str) -> None:
    import torch
    from diffusers import ZImagePipeline

    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    t_load = time.time() - t0
    gen = torch.Generator(device="cuda").manual_seed(42)
    t0 = time.time()
    pipe(prompt=PROMPT, num_inference_steps=9, guidance_scale=0.0,
         height=1024, width=1024, generator=gen)
    t_inf = time.time() - t0
    print(f"[{tag}] 加载 {t_load:.1f}s · 推理 {t_inf:.1f}s", flush=True)
    del pipe
    torch.cuda.empty_cache()


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "1"):
        run_once("1-纯净")

    if which in ("all", "2"):
        sys.path.insert(0, str(ROOT))
        t0 = time.time()
        import backend.app.providers.diffusers_local  # noqa: F401
        print(f"（导入 backend 包耗时 {time.time()-t0:.1f}s）", flush=True)
        run_once("2-导入backend后")

    if which in ("all", "3"):
        import asyncio
        from backend.app.providers.diffusers_local import DiffusersProvider
        provider = DiffusersProvider({"model": MODEL})
        t0 = time.time()
        asyncio.run(provider.generate(prompt=PROMPT, width=1024, height=1024, seed=42))
        print(f"[3-provider.generate] 总耗时 {time.time()-t0:.1f}s", flush=True)

    print("DIAG DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
