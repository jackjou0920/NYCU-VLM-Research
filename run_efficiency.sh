#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 效率總表（docs/03_experiment_plan.md P0 #1）：每個測試集取 NUM_IMAGES 張圖（前 1 張 warmup 不計），
# 用 bench_metrics.py 量 uncompressed / SVMA / FastV / SparseVLM / VisionZip / IF-Prune × BUDGETS 的
# TTFT（中位數）、記憶體增量（扣權重）、KV cache、FLOPs，以及 SVMA 的 TTFT 分段。
#   - SparseVLM 用 accuracy 實驗校準出的 λ（每個 budget 一個值），原圖 token <= budget 時不剪枝
#   - 每張 GPU 同時只跑一個程序，數字才乾淨；資料集輪流分到 GPUS
# 結果：results/efficiency/metrics_<dataset>.csv；log：logs/efficiency/<dataset>.log
# 用法：nohup ./run_efficiency.sh > logs/efficiency/run.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
read -r -a DATASETS <<< "${DATASETS:-docvqa infovqa hrbench_4k mme_rw_lite vstar}"
read -r -a GPUS <<< "${GPUS:-0 1}"
BUDGETS=${BUDGETS:-1024,2048,3072,4096,5120}
NUM_IMAGES=${NUM_IMAGES:-51}
LAMS=${LAMS:-1024:0.5666,2048:0.3794,3072:0.1922,4096:0.1922,5120:0.1922}
OUT=${OUT:-results/efficiency}
LOG=${LOG_DIR:-logs/efficiency}
mkdir -p "$OUT" "$LOG"

run_ds() {   # $1 dataset
    echo "[$(date '+%m-%d %H:%M')] GPU$CUDA_VISIBLE_DEVICES $1"
    $PY bench_metrics.py --dataset "$1" --num_images "$NUM_IMAGES" --warmup_images 1 --budgets "$BUDGETS" \
        --run_fastv --run_sparsevlm --sparsevlm_lams "$LAMS" --run_visionzip --run_ifprune \
        --out_csv "$OUT/metrics_$1.csv" > "$LOG/$1.log" 2>&1 || echo "  [error] $1，看 $LOG/$1.log"
}
run_shard() {   # $1 shard index $2 shard 數
    local k
    for ((k = $1; k < ${#DATASETS[@]}; k += $2)); do run_ds "${DATASETS[$k]}"; done
}
for ((g = 0; g < ${#GPUS[@]}; g++)); do CUDA_VISIBLE_DEVICES=${GPUS[$g]} run_shard "$g" "${#GPUS[@]}" & done
wait
echo "[$(date '+%m-%d %H:%M')] done"
