"""下载本地模型到 models/ 目录（默认走 hf-mirror 镜像）。

用法：
    python backend/fetch_model.py                      # 默认 stabilityai/sd-turbo
    python backend/fetch_model.py stabilityai/sdxl-turbo
    python backend/fetch_model.py <本地已有目录>        # 已是本地目录则跳过

下载完成后把「设置 → 内置推理服务 → 模型」填成本地路径（如 models/sd-turbo）即可离线运行。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# xet 存储不走镜像代理，会 401；强制走普通 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

ALLOW = [
    "*.json", "*.txt", "*.safetensors", "*.bin", "*.model",
    "tokenizer/*", "scheduler/*", "text_encoder/*", "unet/*", "vae/*",
]


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else "stabilityai/sd-turbo"
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
    print(f"镜像：{os.environ['HF_ENDPOINT']}")
    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        allow_patterns=ALLOW,
        max_workers=4,
    )
    print("\n完成。把「模型」设置为：", target)
    print("建议 sd-turbo（1-4 步，6GB 显存友好），或 sdxl-turbo（需 8GB+）")


if __name__ == "__main__":
    main()
