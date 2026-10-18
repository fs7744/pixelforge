#!/usr/bin/env bash
# 从 ModelScope 下载 Z-Image-Turbo 到 models/z-image-turbo（断点续传 + 重试）
set -u
MODEL="Tongyi-MAI/Z-Image-Turbo"
REV="master"
BASE="https://modelscope.cn/models/$MODEL/resolve/$REV"
DEST="D:/code/pixelforge/models/z-image-turbo"

FILES=(
  "model_index.json"
  "configuration.json"
  "scheduler/scheduler_config.json"
  "text_encoder/config.json"
  "text_encoder/generation_config.json"
  "text_encoder/model.safetensors.index.json"
  "text_encoder/model-00001-of-00003.safetensors"
  "text_encoder/model-00002-of-00003.safetensors"
  "text_encoder/model-00003-of-00003.safetensors"
  "tokenizer/tokenizer.json"
  "tokenizer/tokenizer_config.json"
  "tokenizer/vocab.json"
  "tokenizer/merges.txt"
  "transformer/config.json"
  "transformer/diffusion_pytorch_model.safetensors.index.json"
  "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
  "transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
  "transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
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
