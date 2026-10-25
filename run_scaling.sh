#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 解析度縮放曲線（docs/03_experiment_plan.md P0 #2）：同一批大圖（預設 HR-Bench 8K），把 dynamic tiling 的
# tile 上限 MAX_NUM 從 7 逐步放大到 201（網格 ≤ MAX_NUM-1 塊 + 縮圖），在固定 budget（預設 b1536）下量每個方法的
# TTFT（中位數）、記憶體增量（扣權重）、FLOPs。SVMA 的細節網格上限也是 MAX_NUM-1，所以兩邊看到的是同一張大圖、
# 同樣多的原始 tile。
#   - 每個 (MAX_NUM, 方法) 是獨立程序：某個方法 OOM 只記錄在 log，不影響其他組合
#   - uncompressed 的程序同時量 SVMA（bench_metrics 每次都會跑 stream），SVMA 的數字以那份為準
#   - SparseVLM 在 b1536 沒有 accuracy 實驗校準過的 λ，這裡逐張 auto_lam（校準時間不算進 TTFT）
# 結果：results/scaling/metrics_n<MAX_NUM>_<method>.csv；log：logs/scaling/n<MAX_NUM>_<method>.log
# 用法：nohup ./run_scaling.sh > logs/scaling/run.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
DATASET=${DATASET:-hrbench_8k}
read -r -a MAX_NUMS <<< "${MAX_NUMS:-7 13 25 49 97 145 201}"
read -r -a METHODS <<< "${METHODS:-uncompressed fastv sparsevlm visionzip ifprune}"
read -r -a GPUS <<< "${GPUS:-0 1}"
BUDGET=${BUDGET:-1536}
NUM_IMAGES=${NUM_IMAGES:-11}
OUT=${OUT:-results/scaling}
LOG=${LOG_DIR:-logs/scaling}
mkdir -p "$OUT" "$LOG"

run_job() {   # $1 max_num $2 method
    local n=$1 m=$2 flags
    case $m in
        uncompressed) flags="" ;;
        *)            flags="--skip_uncompressed --run_$m" ;;
    esac
    echo "[$(date '+%m-%d %H:%M')] GPU$CUDA_VISIBLE_DEVICES max_num=$n $m"
    $PY bench_metrics.py --dataset "$DATASET" --num_images "$NUM_IMAGES" --warmup_images 1 --max_num "$n" \
        --budgets "$BUDGET" $flags --out_csv "$OUT/metrics_n${n}_$m.csv" > "$LOG/n${n}_$m.log" 2>&1 \
        || { grep -q "OutOfMemoryError" "$LOG/n${n}_$m.log" && echo "  [OOM] max_num=$n $m" || echo "  [error] max_num=$n $m"; }
}
jobs=()
for n in "${MAX_NUMS[@]}"; do for m in "${METHODS[@]}"; do jobs+=("$n $m"); done; done
echo "== scaling：${#jobs[@]} 個工作，dataset=$DATASET budget=$BUDGET =="
run_shard() {
    local k
    for ((k = $1; k < ${#jobs[@]}; k += $2)); do run_job ${jobs[$k]}; done
}
for ((g = 0; g < ${#GPUS[@]}; g++)); do CUDA_VISIBLE_DEVICES=${GPUS[$g]} run_shard "$g" "${#GPUS[@]}" & done
wait
echo "[$(date '+%m-%d %H:%M')] done"
