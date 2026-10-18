"""fp8 落盘缓存端到端测试。

同一进程内跑两次生成不具代表性（provider 有 _CACHE），所以：
  第一次运行（MODE=miss）：bf16 加载 + GPU 量化 + 落盘缓存 + 生成
  第二次运行（MODE=hit）：缓存命中加载 + 生成
两次用同种子同指令，输出图应几乎一致（量化确定性，推理内核允许微小差异）。
用法：
  MODE=miss python backend/test_fp8_cache.py
  MODE=hit  python backend/test_fp8_cache.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = str(ROOT / "models" / "qwen-image-edit-2511")
OUT_DIR = ROOT / "data" / "smoke"
REF_IMAGE = OUT_DIR / "zimage-smoke-0.png"
INSTRUCTION = "保持角色外观、服装、配色完全不变，将姿势改为：蹲下戒备姿态，纯色背景"
MODE = os.environ.get("MODE", "miss")
# 测试用 20 步（provider 允许 20-50），两次一致即可比；生产默认仍是 40
STEPS = 20


def main() -> int:
    from PIL import Image as _Image

    from backend.app.providers.diffusers_local import DiffusersProvider

    ref = _Image.open(REF_IMAGE).convert("RGB")
    provider = DiffusersProvider({"model": MODEL})

    async def _go():
        t0 = time.time()
        result = await provider.generate(
            prompt=INSTRUCTION, width=ref.size[0], height=ref.size[1],
            steps=STEPS, seed=42, ref=ref,
        )
        return result, time.time() - t0

    result, dt = asyncio.run(_go())
    print(f"[{MODE}] meta: {result.meta}")
    print(f"[{MODE}] 总耗时（含加载）：{dt:.1f}s")

    import torch
    print(f"[{MODE}] 显存峰值：{torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

    p = OUT_DIR / f"fp8-cache-{MODE}.png"
    result.images[0].save(p)
    print(f"[{MODE}] 已保存：{p}")
    print(f"FP8-CACHE-{MODE.upper()} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
