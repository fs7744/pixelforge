"""Provider 路径对照实验：分离「加载参数」与「线程上下文」两个变量。

三段测试，各出一张图并计时：
  A. 直连管线（默认加载参数，主线程）——已知 ~30s 基线
  B. provider._load + 主线程直接 pipe()——隔离加载参数影响
  C. provider.generate()（to_thread 线程池）——复现冒烟/bench 路径

用法：
    python backend/diag_provider_path.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "z-image-turbo")
PROMPT = "一个Q版忍者猫忍者，像素风格，粗黑描边，站立姿势，挥动手里的小太刀，纯色背景，游戏角色素材"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import torch

    from backend.app.providers.diffusers_local import DiffusersProvider

    provider = DiffusersProvider({"model": MODEL})
    gen = torch.Generator(device="cuda").manual_seed(42)

    print("=== B. provider._load + 主线程 pipe() ===", flush=True)
    t0 = time.time()
    pipe = provider._load(MODEL)
    print(f"  _load 耗时 {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    out = pipe(prompt=PROMPT, num_inference_steps=9, guidance_scale=0.0,
               height=1024, width=1024, generator=gen)
    print(f"  主线程推理耗时 {time.time()-t0:.1f}s", flush=True)
    out.images[0].save(ROOT / "data" / "smoke" / "zimage-pathB.png")

    gen2 = torch.Generator(device="cuda").manual_seed(43)
    print("=== C. provider.generate()（to_thread 路径）===", flush=True)
    import asyncio
    t0 = time.time()
    result = asyncio.run(provider.generate(prompt=PROMPT, width=1024, height=1024, seed=43))
    print(f"  generate 耗时 {time.time()-t0:.1f}s", flush=True)
    result.images[0].save(ROOT / "data" / "smoke" / "zimage-pathC.png")

    print("DIAG DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
