#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 模型尺寸表的效率欄：InternVL3.5-4B / 14B × 主表的 5 個測試集 × b1536 的 TTFT / 記憶體 / FLOPs。
# SparseVLM 的 λ 用 run_model_scale.sh 在該模型、該資料集上校準出的值（從 logs/model_scale.log 讀）。
# 結果：results/efficiency/InternVL3_5-<size>/metrics_<ds>.csv（build_paper_results.py 會讀）
# 用法：WAIT_PID=<run_model_scale.sh 的 pid> nohup ./run_efficiency_scale.sh > logs/efficiency/scale.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
read -r -a SIZES <<< "${SIZES:-4B 14B}"
DATASETS=${DATASETS:-docvqa infovqa mme_rw_lite hrbench_4k vstar}

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date '+%m-%d %H:%M')] 等 PID $WAIT_PID 結束"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
fi
for size in "${SIZES[@]}"; do
    # logs/model_scale.log 裡「InternVL3.5-<size>」區段內的「<ds> sparsevlm b1536：校準 λ=<x>」
    lams=$(awk -v s="InternVL3.5-$size" '/\] InternVL3.5-/{on = ($NF == s)} on && /sparsevlm b1536：校準 λ=/{split($0, a, "λ="); print $1 "=1536:" a[2]}' logs/model_scale.log | paste -sd ';')
    echo "[$(date '+%m-%d %H:%M')] InternVL3.5-$size  λ: $lams"
    MODEL="OpenGVLab/InternVL3_5-$size" OUT="results/efficiency/InternVL3_5-$size" LOG_DIR="logs/efficiency/InternVL3_5-$size" \
        BUDGETS=1536 DATASETS="$DATASETS" LAMS_BY_DS="$lams" ./run_efficiency.sh
done
echo "[$(date '+%m-%d %H:%M')] all done"
