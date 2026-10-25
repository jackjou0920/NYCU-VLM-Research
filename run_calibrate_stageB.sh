#!/usr/bin/env bash
# 階段 B（S）一次跑完：accuracy（兩個校準集 × 5 個 budget × 4 個新的 S，兩張卡平行）→ 每張卡各量一個校準集的 TTFT。
# 結果：results/calib/<dataset>/foveatile_b<B>_m<m>_S<S>_*.json、results/calib/metrics/metrics_S_<dataset>.csv
# 用法：nohup ./run_calibrate_stageB.sh > logs/calib/stageB.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
BUDGETS_CSV=${BUDGETS_CSV:-1024,1536,2048,2560,3072}

echo "[$(date '+%m-%d %H:%M')] accuracy"
STAGE=S CALIB_SETS="infovqa_train_calib mme_rw_calib" BUDGETS="${BUDGETS_CSV//,/ }" GPUS="0 1" ./run_calibrate.sh

echo "[$(date '+%m-%d %H:%M')] TTFT"
bench() {   # $1 GPU, $2 dataset
    CUDA_VISIBLE_DEVICES=$1 $PY bench_metrics.py --dataset "$2" --num_images 51 --warmup_images 1 \
        --budgets "$BUDGETS_CSV" --foveatile_s_mults 0,0.5,1,1.5,2 --skip_uncompressed \
        --out_csv "results/calib/metrics/metrics_S_$2.csv" > "logs/calib/bench_S_$2.log" 2>&1 \
        || echo "  [error] bench $2"
}
bench 0 infovqa_train_calib &
bench 1 mme_rw_calib &
wait
echo "[$(date '+%m-%d %H:%M')] done"
