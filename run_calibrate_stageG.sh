#!/usr/bin/env bash
# 3b（打分窗口隨細節網格放大，--qattn_grid_frac）一次跑完：accuracy（兩個校準集 × 5 個 budget × GRID_FRACS，
# 兩張卡平行）→ 每張卡各量一個校準集的 TTFT（含 f = 0 = 目前預設）。
# 結果：results/calib/<dataset>/svma_b<B>_m<m>_gf<f>_*.json、results/calib/metrics/metrics_G_<dataset>.csv
# 用法：nohup ./run_calibrate_stageG.sh > logs/calib/stageG.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
GRID_FRACS=${GRID_FRACS:-0.2 0.3 0.4 0.5}
BUDGETS_CSV=${BUDGETS_CSV:-1024,1536,2048,2560,3072}

echo "[$(date '+%m-%d %H:%M')] accuracy"
STAGE=G GRID_FRACS="$GRID_FRACS" CALIB_SETS="infovqa_train_calib mme_rw_calib" BUDGETS="${BUDGETS_CSV//,/ }" \
    GPUS="0 1" ./run_calibrate.sh

echo "[$(date '+%m-%d %H:%M')] TTFT"
bench() {   # $1 GPU, $2 dataset
    CUDA_VISIBLE_DEVICES=$1 $PY bench_metrics.py --dataset "$2" --num_images 51 --warmup_images 1 \
        --budgets "$BUDGETS_CSV" --svma_grid_fracs "0,${GRID_FRACS// /,}" --skip_uncompressed \
        --out_csv "results/calib/metrics/metrics_G_$2.csv" > "logs/calib/bench_G_$2.log" 2>&1 \
        || echo "  [error] bench $2"
}
bench 0 infovqa_train_calib &
bench 1 mme_rw_calib &
wait
echo "[$(date '+%m-%d %H:%M')] done"
