"""Z-Image-Turbo 冒烟测试：中文提示词直喂，验证 4090 + Python 3.13 全链路。

用法（依赖装完后）：
    "C:/Users/Windows/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe" backend/smoke_zimage.py

通过标准：出一张图 + 打印耗时/显存峰值，无异常即视为环境就绪。
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


def main() -> int:
    sys.path.insert(0, str(ROOT))
    if not Path(MODEL).is_dir():
        print(f"模型目录不存在：{MODEL}\n先跑 bash backend/fetch_zimage.sh")
        return 1

    from backend.app.providers.diffusers_local import DiffusersProvider, detect_family

    print(f"模型：{MODEL}")
    print(f"模型族：{detect_family(MODEL)}")

    provider = DiffusersProvider({"model": MODEL})

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async def _go():
        t0 = time.time()
        result = await provider.generate(
            prompt=PROMPT, width=1024, height=1024, seed=42,
        )
        dt = time.time() - t0
        return result, dt

    result, dt = asyncio.run(_go())
    print(f"元信息：{result.meta}")
    print(f"耗时：{dt:.1f}s")

    try:
        import torch
        print(f"显存峰值：{torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    except Exception:
        pass

    for i, img in enumerate(result.images):
        p = OUT_DIR / f"zimage-smoke-{i}.png"
        img.save(p)
        print(f"已保存：{p}（{img.size[0]}x{img.size[1]}）")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
