"""Z-Image-Turbo 稳态测速：加载一次，连出 3 张，报告每张纯推理耗时。

用法：
    "C:/Users/Windows/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe" backend/bench_zimage.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "z-image-turbo")
OUT_DIR = ROOT / "data" / "smoke"

PROMPT = "一个Q版忍者猫忍者，像素风格，粗黑描边，站立姿势，挥动手里的小太刀，纯色背景，游戏角色素材"


async def main() -> int:
    sys.path.insert(0, str(ROOT))
    from backend.app.providers.diffusers_local import DiffusersProvider

    provider = DiffusersProvider({"model": MODEL})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        t0 = time.time()
        result = await provider.generate(
            prompt=PROMPT, width=1024, height=1024, seed=42 + i,
        )
        dt = time.time() - t0
        tag = "含加载" if i == 0 else "稳态"
        print(f"[{i+1}/3] {tag} 耗时：{dt:.1f}s  steps={result.meta.get('steps')} cfg={result.meta.get('cfg')}")
        result.images[0].save(OUT_DIR / f"zimage-bench-{i}.png")

    try:
        import torch
        print(f"显存峰值：{torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    except Exception:
        pass
    print("BENCH DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
