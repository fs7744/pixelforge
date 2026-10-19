"""从 ModelScope 下载 Wan2.2 TI2V-5B（diffusers 版）到 models/wan2.2-ti2v-5b/。

为什么是 TI2V-5B：Wan2.2 里最小的视频底座（5B），24GB 单卡在 transformer /
text_encoder 都做 fp8 后能整模驻留；14B I2V 要 block swap，先用 5B 验证
「参考图 → 视频 → 抽帧 → 抠图 → 锁色 → 拼表」整条链路。

为什么走 ModelScope：huggingface.co 在本机代理下 502，而 Z-Image / Qwen-Edit
都是从 ModelScope 拉的（见 fetch_zimage.sh / fetch_qwen_edit.sh）。

合计约 31.9GB（transformer 18.6 + umT5-xxl 10.6 + VAE 2.6 + tokenizer）。
text_encoder 必须下：TI2V-5B 的文本条件走 umT5-xxl，没有替代品；fp8 化放在
provider 首次加载时做（本项目既有的 _FP8Linear + 落盘缓存）。

用法：PYTHONPATH=. python backend/fetch_wan22.py
"""
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REV = "master"
BASE = f"https://modelscope.cn/models/{REPO}/resolve/{REV}"
DEST = ROOT / "models" / "wan2.2-ti2v-5b"

FILES = [
    "configuration.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def fetch(rel: str, retries: int = 5) -> None:
    out = DEST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{rel}"
    have = out.stat().st_size if out.is_file() else 0
    for attempt in range(retries):
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:      # 已下完整
                    return
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                with open(out, mode) as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        print(f"\r  {rel}  {have / 2**30:.2f}/{total / 2**30:.2f} GB",
                              end="", flush=True)
                print(flush=True)
                return
        except Exception as e:
            print(f"\n  重试 {attempt + 1}/{retries}（{type(e).__name__}: {e}）", flush=True)
            have = out.stat().st_size if out.is_file() else 0
            time.sleep(3)
    raise RuntimeError(f"下载失败：{rel}")


def main() -> int:
    if DEST.is_dir() and any(DEST.iterdir()):
        print(f"目标目录已存在且非空：{DEST}\n先删掉或改名再下，避免半套模型被当成可用。")
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"{REPO} → {DEST}（{len(FILES)} 个文件，约 32GB）", flush=True)
    for i, rel in enumerate(FILES, 1):
        print(f"[{i}/{len(FILES)}] {rel}", flush=True)
        fetch(rel)
    print("完成。provider 首次加载时会把 transformer / text_encoder 量化成 fp8 并落盘缓存。",
          flush=True)
    return 0


sys.exit(main())
