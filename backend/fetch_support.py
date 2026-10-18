"""下载 IP-Adapter 与像素 LoRA 支持文件（国内走 hf-mirror，禁用 xet）。

用法：python backend/fetch_support.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

MODELS = Path(__file__).resolve().parent.parent / "models"

JOBS = [
    (
        "h94/IP-Adapter",
        # 只取 safetensors 版图像编码器，避开同仓库的 pytorch_model.bin（同样 2.4GB，白下）
        [
            "models/ip-adapter_sd15.safetensors",
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        ],
        MODELS / "ip-adapter-sd15",
    ),
    (
        "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        ["*.safetensors"],
        MODELS / "lora-pixel-redmond",
    ),
    # SD1.5 底座：sd-turbo 是 SD2.1 架构，跨注意力维度 1024，
    # 装不下 SD1.5 生态里 768 维的像素 LoRA 与 IP-Adapter，所以必须换底座。
    (
        "Lykon/dreamshaper-8",
        [
            "model_index.json",
            "scheduler/*",
            "feature_extractor/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "tokenizer/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "safety_checker/config.json",
        ],
        MODELS / "dreamshaper-8",
    ),
    # LCM 蒸馏：把 SD1.5 的 25 步压到 4 步，CPU 才跑得动
    (
        "latent-consistency/lcm-lora-sdv1-5",
        ["*.safetensors"],
        MODELS / "lcm-lora-sdv1-5",
    ),
]


def main() -> None:
    from huggingface_hub import snapshot_download

    for repo, patterns, local in JOBS:
        print(f"\n== {repo} -> {local}", flush=True)
        try:
            snapshot_download(
                repo_id=repo,
                allow_patterns=patterns,
                local_dir=str(local),
                max_workers=2,
            )
            got = [f for f in local.rglob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in got) / 1024**3
            print(f"   ok {len(got)} files, {size:.2f} GB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   fail: {e}", flush=True)

    total = sum(f.stat().st_size for f in MODELS.rglob("*") if f.is_file())
    print(f"\nmodels/ 合计 {total / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
