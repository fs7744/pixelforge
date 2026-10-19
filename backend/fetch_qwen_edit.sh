#!/usr/bin/env bash
# 从 ModelScope 下载 Qwen-Image-Edit-2511 到 models/qwen-image-edit-2511
# （断点续传 + 重试）。bf16 共约 53.8GB：transformer 5 片 + text_encoder 4 片。
# 用途：角色一致性第二底座（指令式编辑，中文直喂）。
set -u
MODEL="Qwen/Qwen-Image-Edit-2511"
REV="master"
BASE="https://modelscope.cn/models/$MODEL/resolve/$REV"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/qwen-image-edit-2511"

FILES=(
  "model_index.json"
  "configuration.json"
  "scheduler/scheduler_config.json"
  # processor（Qwen2-VL 风格，管线主用）
  "processor/added_tokens.json"
  "processor/chat_template.jinja"
  "processor/merges.txt"
  "processor/preprocessor_config.json"
  "processor/special_tokens_map.json"
  "processor/tokenizer.json"
  "processor/tokenizer_config.json"
  "processor/video_preprocessor_config.json"
  "processor/vocab.json"
  # text_encoder（Qwen2.5-VL，4 片）
  "text_encoder/config.json"
  "text_encoder/generation_config.json"
  "text_encoder/model.safetensors.index.json"
  "text_encoder/model-00001-of-00004.safetensors"
  "text_encoder/model-00002-of-00004.safetensors"
  "text_encoder/model-00003-of-00004.safetensors"
  "text_encoder/model-00004-of-00004.safetensors"
  # tokenizer（兼容目录，与 processor 内容重复但体积小）
  "tokenizer/added_tokens.json"
  "tokenizer/chat_template.jinja"
  "tokenizer/merges.txt"
  "tokenizer/special_tokens_map.json"
  "tokenizer/tokenizer_config.json"
  "tokenizer/vocab.json"
  # transformer（20B MMDiT，5 片）
  "transformer/config.json"
  "transformer/diffusion_pytorch_model.safetensors.index.json"
  "transformer/diffusion_pytorch_model-00001-of-00005.safetensors"
  "transformer/diffusion_pytorch_model-00002-of-00005.safetensors"
  "transformer/diffusion_pytorch_model-00003-of-00005.safetensors"
  "transformer/diffusion_pytorch_model-00004-of-00005.safetensors"
  "transformer/diffusion_pytorch_model-00005-of-00005.safetensors"
  # VAE
  "vae/config.json"
  "vae/diffusion_pytorch_model.safetensors"
)

mkdir -p "$DEST"

TOTAL=${#FILES[@]}
IDX=0
for f in "${FILES[@]}"; do
  IDX=$((IDX+1))
  OUT="$DEST/$f"
  mkdir -p "$(dirname "$OUT")"
  if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT" 2>/dev/null || echo 0)" -gt 0 ]; then
    echo "[$IDX/$TOTAL] 已存在，跳过：$f"
    continue
  fi
  echo "[$IDX/$TOTAL] 下载：$f"
  START=$(date +%s)
  curl -L -C - --retry 8 --retry-delay 5 --retry-all-errors \
       --connect-timeout 20 --speed-limit 1024 --speed-time 60 \
       -o "$OUT" "$BASE/$f" || echo "!! 失败：$f（可重跑本脚本续传）"
  END=$(date +%s)
  SZ=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  DT=$((END-START)); [ $DT -eq 0 ] && DT=1
  echo "    -> $((SZ/1024/1024)) MB · $((SZ/1024/1024/DT)) MB/s · ${DT}s"
done

echo "==== 完成，总大小：$(du -sh "$DEST" | cut -f1) ===="
