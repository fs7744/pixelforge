#!/usr/bin/env bash
# PixelForge GPU 推理依赖一键安装（CUDA 12.4，适配 RTX 30/40 系）
#
# 前置：先跑一次 ./start.sh --setup 建好 .venv
# 用法：bash backend/install_gpu_deps.sh
#
# 可用环境变量覆盖：
#   PF_VENV        虚拟环境目录（默认 <项目>/.venv）
#   PF_PYTHON      指定解释器（跳过 .venv 探测）
#   PF_PIP_INDEX   pip 源（默认阿里云）
#   TORCH_INDEX    torch wheel 源（默认 cu124；无显卡可用 .../whl/cpu）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PF_VENV:-$ROOT/.venv}"
MIRROR="${PF_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
RETRY=(--retries 5 --timeout 60)

PY="${PF_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$VENV/bin/python" "$VENV/Scripts/python.exe"; do
    [ -x "$c" ] && { PY="$c"; break; }
  done
fi
if [ -z "$PY" ]; then
  echo "[错误] 没找到虚拟环境，请先运行：./start.sh --setup"
  exit 1
fi
echo "解释器：$PY"
echo "pip 源：$MIRROR"
echo "torch 源：$TORCH_INDEX"

echo "== [1/4] pip =="
"$PY" -m pip install -U pip -i "$MIRROR" "${RETRY[@]}"

echo "== [2/4] torch + torchvision (约 2.9GB，已装则跳过) =="
# torchvision 0.21.0 对应 torch 2.6.0；Qwen-Image-Edit-2511 的 processor 带
# video_preprocessor_config.json，transformers 加载 Qwen2VLVideoProcessor 时强制要求 torchvision
"$PY" -m pip install torch --index-url "$TORCH_INDEX" "${RETRY[@]}"
"$PY" -m pip install torchvision==0.21.0 --index-url "$TORCH_INDEX" "${RETRY[@]}"

echo "== [3/4] web 依赖 =="
"$PY" -m pip install -r "$ROOT/backend/requirements.txt" -i "$MIRROR" "${RETRY[@]}"

echo "== [4/4] AI 依赖 =="
# torchao 版本窗口（2026-10 实测，torch 2.6.0+cu124 固定）：
# - >=0.18 用 torch.utils._pytree.register_constant（torch 2.7+ 才有）→ ImportError
# - <0.16 缺 FqnToConfig 崩 diffusers 0.41 导入链；且 peft>=0.21 的 is_torchao_available() 也要求 >=0.16
# 0.16.x 是唯一交集；cpp 扩展会跳过（走 Python 回退内核，速度实测可接受）
"$PY" -m pip install diffusers transformers accelerate safetensors huggingface_hub -i "$MIRROR" "${RETRY[@]}"
"$PY" -m pip install peft -i "$MIRROR" "${RETRY[@]}"
"$PY" -m pip install "torchao==0.16.0" -i "$MIRROR" "${RETRY[@]}"

echo "== 可选：深度抠图（rembg，装不上只降级不影响其它功能） =="
"$PY" -m pip install -r "$ROOT/backend/requirements-optional.txt" -i "$MIRROR" "${RETRY[@]}" \
  || echo "  [!] rembg 安装失败，抠图会退到 GrabCut"

echo "== 验证 =="
"$PY" -c "import torch; print('TORCH OK', torch.__version__, 'cuda:', torch.cuda.is_available())"
"$PY" -c "import torchvision; print('TORCHVISION OK', torchvision.__version__)"
"$PY" -c "import torchao; print('TORCHAO OK', torchao.__version__)"
"$PY" -c "from diffusers import ZImagePipeline; print('ZImagePipeline OK')"
"$PY" -c "from diffusers import QwenImageEditPlusPipeline; print('QwenImageEditPlusPipeline OK')"
echo "ALL DONE"
echo
echo "下一步：下载模型（约 20GB / 54GB）"
echo "  bash backend/fetch_zimage.sh        # Z-Image-Turbo（生成底座，必装）"
echo "  bash backend/fetch_qwen_edit.sh     # Qwen-Image-Edit-2511（编辑底座，可选）"
