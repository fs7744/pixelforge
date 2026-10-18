"""下载本地模型到 models/ 目录（默认走 hf-mirror 镜像，支持断点续传与自动重试）。

用法：
    python backend/fetch_model.py                      # 默认 segmind/tiny-sd（~1.1GB，CPU 可跑）
    python backend/fetch_model.py stabilityai/sd-turbo # ~2.5GB，1-4 步，6GB 显存友好
    python backend/fetch_model.py <本地已有目录>        # 已是本地目录则跳过

下载完成后把「设置 → 内置推理服务 → 模型」填成本地路径（如 models/tiny-sd）即可离线运行。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# xet 存储不走镜像代理，会 401；强制走普通 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 镜像偶发挂起：调低读超时，让 huggingface_hub 主动重连而不是无限等待
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

ALLOW = [
    "model_index.json",
    "*.txt",
    "scheduler/*",
    "tokenizer/*",
    # SD1.5 / SDXL 布局：只取 fp32 组件；CPU 推理需要 fp32；GPU 用户可另加 *.fp16.safetensors
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/pytorch_model.bin",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.bin",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.bin",
    # Z-Image 布局（S3-DiT 单流架构）：transformer 分片 + Qwen 文本编码器分片 + VAE
    "transformer/config.json",
    "transformer/diffusion_pytorch_model*.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/model-*.safetensors",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/generation_config.json",
]

MAX_RETRIES = int(os.environ.get("PF_FETCH_RETRIES", "8"))


def human(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1024 / 1024:.0f} MB"


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else "segmind/tiny-sd"
    local = Path(repo)
    if local.exists():
        print(f"本地目录已存在，跳过下载：{local}")
        return

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"缺少 huggingface_hub：{e}\n请先 pip install huggingface_hub")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / repo.split("/")[-1]
    print(f"下载 {repo} → {target}")
    print(f"镜像：{os.environ['HF_ENDPOINT']} · 超时 {os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s · 最多重试 {MAX_RETRIES} 次")

    last = -1
    stall = 0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(target),
                allow_patterns=ALLOW,
                max_workers=4,
                resume_download=True,
            )
            print(f"\n完成（{human(target)}）。把「模型」设置为：{target}")
            print("建议 sd-turbo（1-4 步，6GB 显存友好），或 sdxl-turbo（需 8GB+）")
            return
        except Exception as e:
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) if target.exists() else 0
            print(f"\n[重试 {attempt}/{MAX_RETRIES}] 断点处 {size / 1024 / 1024:.0f} MB · {type(e).__name__}: {str(e)[:160]}")
            if size == last:
                stall += 1
            else:
                stall = 0
            last = size
            if stall >= 3:
                print("连续 3 次无进展，疑似镜像不可用。稍后重试或换源：")
                print("  set HF_ENDPOINT=https://hf-mirror.com   （或自备代理直连 huggingface.co）")
                sys.exit(2)
            time.sleep(min(5 * attempt, 30))


if __name__ == "__main__":
    main()
