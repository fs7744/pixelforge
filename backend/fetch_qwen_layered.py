"""从 ModelScope 下载 Qwen-Image-Layered（diffusers 版）到 models/qwen-image-layered/。

为什么需要它：视差工作流的「图层提取」原本是把同一个场景**生成四遍**
（每遍只留一层、其余抹成纯色键）。四次独立推理的透视和轮廓对不上，
叠图只能靠色键硬抠，边缘一圈杂色。语义分层是**一次推理拆同一张图**，
天然坐标对齐、无接缝、带 alpha。

合计约 57.7GB（transformer 40.9 + text_encoder 16.6 + VAE 0.25 + 词表）。
transformer 是 20B bf16，24GB 卡必须量化——provider 首次加载时会做 fp8
并落盘缓存（复用 Qwen-Edit 那条 _load_qwen_transformer_fp8 的路径）。

为什么走 ModelScope：huggingface.co 在本机代理下 502，其余底座也都是
从 ModelScope 拉的（见 fetch_zimage.sh / fetch_qwen_edit.sh）。

用法：PYTHONPATH=. python backend/fetch_qwen_layered.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO = "Qwen/Qwen-Image-Layered"
REV = "master"
BASE = f"https://modelscope.cn/models/{REPO}/resolve/{REV}"
DEST = ROOT / "models" / "qwen-image-layered"

FILES = [
    "configuration.json",
    "model_index.json",
    # 处理器与词表：Qwen2.5-VL 的 processor 同时供文本与多分辨率图像编码用
    "processor/added_tokens.json",
    "processor/chat_template.json",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    # 文本编码器 Qwen2.5-VL 7B，bf16 约 16.6GB
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    # 主 DiT：20B，bf16 分 5 片共 40.9GB
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
                    print(f"  跳过（已完整）：{rel}", flush=True)
                    return
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                mark = 0
                with open(out, mode) as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        # 每 1GB 报一次：多文件并行时逐块刷屏会把日志冲垮
                        if have - mark >= 1 << 30:
                            mark = have
                            print(f"  {rel}  {have / 2**30:.1f}/{total / 2**30:.1f} GB", flush=True)
                if total and have < total:
                    raise IOError(f"只下到 {have}/{total} 字节")
                print(f"  完成 {rel}（{have / 2**30:.2f} GB）", flush=True)
                return
        except Exception as e:
            print(f"  重试 {attempt + 1}/{retries} {rel}（{type(e).__name__}: {e}）", flush=True)
            have = out.stat().st_size if out.is_file() else 0
            # 小文件（配置/词表/索引）走 Range 反而容易被服务端掐断
            # （实测 IncompleteRead: 0 bytes read）。整份重下更快也更干净。
            if have and have < (1 << 20):
                out.unlink(missing_ok=True)
                have = 0
            time.sleep(3)
    raise RuntimeError(f"下载失败：{rel}")


def main() -> int:
    jobs = max(1, int(os.environ.get("PF_FETCH_JOBS", "5")))
    DEST.mkdir(parents=True, exist_ok=True)
    # 不因为目录非空就退出：并行/中断后要能续传，fetch() 自己按 Range 接着下
    print(f"{REPO} → {DEST}（{len(FILES)} 个文件，约 57.7GB，{jobs} 路并行）", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(fetch, rel): rel for rel in FILES}
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                fut.result()
            except Exception as e:
                # 单个文件失败不整体中断：已下完的部分保留，重跑即可续传
                print(f"失败：{rel}（{e}）", flush=True)
                failed.append(rel)
    if failed:
        print(f"\n{len(failed)} 个文件未完成，重跑本脚本会续传：\n  "
              + "\n  ".join(failed), flush=True)
        return 1
    print("完成。provider 首次加载时会把 transformer 量化成 fp8 并落盘缓存。", flush=True)
    return 0


sys.exit(main())
