#!/usr/bin/env bash
# PixelForge GPU 依赖一键安装（RTX 4090 / CUDA 12.4）
set -e
PY="C:/Users/Windows/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe"
MIRROR="https://mirrors.aliyun.com/pypi/simple"
RETRY="--retries 5 --timeout 60"

echo "== [1/4] pip =="
"$PY" -m pip install -U pip -i "$MIRROR" $RETRY

echo "== [2/4] torch + torchvision (cu124, torchvision 必须与 torch 同源同版本，~2.9GB，已装则跳过) =="
# torchvision 0.21.0 对应 torch 2.6.0；Qwen-Image-Edit-2511 的 processor 带
# video_preprocessor_config.json，transformers 加载 Qwen2VLVideoProcessor 时强制要求 torchvision
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 $RETRY
"$PY" -m pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 $RETRY

echo "== [3/4] web 依赖 =="
"$PY" -m pip install -r D:/code/pixelforge/backend/requirements.txt -i "$MIRROR" $RETRY

echo "== [4/4] AI 依赖 =="
# torchao 版本窗口（2026-10 实测，torch 2.6.0+cu124 固定）：
# - >=0.18 用 torch.utils._pytree.register_constant（torch 2.7+ 才有）→ ImportError
# - <0.15 缺 FqnToConfig，diffusers 0.41 的 quantizers/auto.py 导入即崩
# 0.15.x 是两者交集；cpp 扩展会跳过（走 Python 回退内核，速度实测可接受）
"$PY" -m pip install diffusers transformers accelerate safetensors huggingface_hub -i "$MIRROR" $RETRY
"$PY" -m pip install "torchao==0.15.0" -i "$MIRROR" $RETRY

echo "== 验证 =="
"$PY" -c "import torch; print('TORCH OK', torch.__version__, 'cuda:', torch.cuda.is_available())"
"$PY" -c "import torchvision; print('TORCHVISION OK', torchvision.__version__)"
"$PY" -c "import torchao; print('TORCHAO OK', torchao.__version__)"
"$PY" -c "from diffusers import ZImagePipeline; print('ZImagePipeline OK')"
"$PY" -c "from diffusers import QwenImageEditPlusPipeline; print('QwenImageEditPlusPipeline OK')"
echo "ALL DONE"
