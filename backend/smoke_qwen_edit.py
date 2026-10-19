"""Qwen-Image-Edit-2511 冒烟测试：参考图 + 中文编辑指令，验证角色一致性链路。

用法（模型下载完后）：
    "C:/Users/Windows/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe" backend/smoke_qwen_edit.py

通过标准：出一张编辑图 + 打印耗时/显存峰值；并拼一张「原图 | 编辑图」对比图。
默认参考图取 data/smoke/zimage-smoke-0.png（Z-Image 首图），构成双底座接力：
Z-Image 生成帧 1 → Edit-2511 编辑帧 2。
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "qwen-image-edit-2511")
OUT_DIR = ROOT / "data" / "smoke"
REF_IMAGE = OUT_DIR / "zimage-smoke-0.png"

# 编辑指令（不是画面描述）：锁外观、改姿势——2511 版专项优化此类指令
INSTRUCTION = "保持角色外观、服装、配色完全不变，将姿势改为：蹲下戒备姿态，纯色背景"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    if not Path(MODEL).is_dir():
        print(f"模型目录不存在：{MODEL}\n先跑 bash backend/fetch_qwen_edit.sh")
        return 1
    if not REF_IMAGE.is_file():
        print(f"参考图不存在：{REF_IMAGE}\n先跑 backend/smoke_zimage.py 生成帧 1")
        return 1

    from PIL import Image as _Image

    from backend.app.providers.diffusers_local import DiffusersProvider, detect_family

    print(f"模型：{MODEL}")
    print(f"模型族：{detect_family(MODEL)}")

    provider = DiffusersProvider({"model": MODEL})

    ref = _Image.open(REF_IMAGE).convert("RGB")
    print(f"参考图：{REF_IMAGE}（{ref.size[0]}x{ref.size[1]}）")
    print(f"编辑指令：{INSTRUCTION}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async def _go():
        t0 = time.time()
        result = await provider.generate(
            prompt=INSTRUCTION, width=ref.size[0], height=ref.size[1],
            seed=42, ref=ref,
        )
        return result, time.time() - t0

    result, dt = asyncio.run(_go())
    print(f"元信息：{result.meta}")
    print(f"耗时：{dt:.1f}s")

    try:
        import torch
        print(f"显存峰值：{torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    except Exception:
        pass

    for i, img in enumerate(result.images):
        p = OUT_DIR / f"qwen-edit-smoke-{i}.png"
        img.save(p)
        print(f"已保存：{p}（{img.size[0]}x{img.size[1]}）")

    # 拼对比图：左原图（帧 1）| 右编辑图（帧 2）
    if result.images:
        h = max(ref.size[1], result.images[0].size[1])
        canvas = _Image.new("RGB", (ref.size[0] + result.images[0].size[0] + 16, h), (32, 32, 32))
        canvas.paste(ref, (0, 0))
        canvas.paste(result.images[0].convert("RGB"), (ref.size[0] + 16, 0))
        cp = OUT_DIR / "qwen-edit-compare.png"
        canvas.save(cp)
        print(f"对比图：{cp}")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
