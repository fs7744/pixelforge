#!/usr/bin/env bash
# PixelForge GPU 依赖一键安装（RTX 4090 / CUDA 12.4）
set -e
PY="C:/Users/Windows/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe"
MIRROR="https://mirrors.aliyun.com/pypi/simple"
RETRY="--retries 5 --timeout 60"

echo "== [1/4] pip =="
"$PY" -m pip install -U pip -i "$MIRROR" $RETRY

echo "== [2/4] torch (cu124, ~2.8GB，已装则跳过) =="
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 $RETRY

echo "== [3/4] web 依赖 =="
"$PY" -m pip install -r D:/code/pixelforge/backend/requirements.txt -i "$MIRROR" $RETRY

echo "== [4/4] AI 依赖 =="
"$PY" -m pip install diffusers transformers accelerate safetensors huggingface_hub -i "$MIRROR" $RETRY

echo "== 验证 =="
"$PY" -c "import torch; print('TORCH OK', torch.__version__, 'cuda:', torch.cuda.is_available())"
"$PY" -c "from diffusers import ZImagePipeline; print('ZImagePipeline OK')"
"$PY" -c "from diffusers import QwenImageEditPlusPipeline; print('QwenImageEditPlusPipeline OK')"
echo "ALL DONE"
