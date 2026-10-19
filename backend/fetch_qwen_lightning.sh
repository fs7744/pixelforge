#!/usr/bin/env bash
# 从 ModelScope 下载 Qwen-Image-Edit-2511-Lightning 蒸馏 LoRA（lightx2v 官方）
# 到 models/qwen-image-edit-2511-lightning/。bf16 LoRA 各约 810MB。
#
# 【实验结论 2026-10-18，暂不作默认】4/8 步蒸馏 + true_cfg=1.0：
# - 速度：8 步 123s / 4 步 82s（fp8 底座 40 步约 250s），只有 2-3 倍而非
#   宣传的 10 倍——PEFT 包装让每次 transformer 前向从 2.8s 涨到 12s；
# - 质量：角色一致性完美，但姿势忠实度明显不如 40 步（8 步版把「蹲下戒备」
#   演成了更偏站立），用户裁定质量优先，provider 保持 40 步默认。
# - 留档原因：LoRA 融合进 fp8 权重（一次性合并、零包装开销）是潜在的
#   4 步 50s 方案，值得后续验证；文件仅 1.6GB，留着不占地方。
set -u
MODEL="lightx2v/Qwen-Image-Edit-2511-Lightning"
REV="master"
BASE="https://modelscope.cn/models/$MODEL/resolve/$REV"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/qwen-image-edit-2511-lightning"

FILES=(
  "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
  "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
)

mkdir -p "$DEST"

TOTAL=${#FILES[@]}
IDX=0
for f in "${FILES[@]}"; do
  IDX=$((IDX+1))
  OUT="$DEST/$f"
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
