#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Ablation：random prior（細節 tile 的 prior 換成隨機分數，其餘同預設 m = K/2、S = 預設表），只跑 held-out 校準集。
# 結果：results/calib/<dataset>/foveatile_b<B>_m<m>_S<S>_randprior_InternVL3_5-8B.json
#       （results/paper/build_paper_results.py 的 Table「Ablations」會自動讀進去）
# 用法：nohup ./run_ablation_randprior.sh > logs/calib/randprior.log 2>&1 &
#       WAIT_PID=<pid> ./run_ablation_randprior.sh   # 先等某個程序結束
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
read -r -a CALIB_SETS <<< "${CALIB_SETS:-infovqa_train_calib mme_rw_calib}"
read -r -a BUDGETS <<< "${BUDGETS:-1024 1536 2048 2560 3072}"
read -r -a GPUS <<< "${GPUS:-0 1}"
MIN_FREE_GB=${MIN_FREE_GB:-20}

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date '+%m-%d %H:%M')] 等 PID $WAIT_PID 結束"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
fi

wait_my_gpu() {   # GPU 跟別人共用：等這張卡空出 MIN_FREE_GB（不動別人的程序）
    while [ "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)" -lt $((MIN_FREE_GB * 1024)) ]; do
        sleep 120
    done
}

jobs=()
for ds in "${CALIB_SETS[@]}"; do for b in "${BUDGETS[@]}"; do jobs+=("$ds $b"); done; done
run_shard() {   # $1 shard index $2 shard 數
    local k ds b m S
    for ((k = $1; k < ${#jobs[@]}; k += $2)); do
        read -r ds b <<< "${jobs[$k]}"
        read -r m S <<< "$($PY -c "from foveatile_config import default_coverage_tiles as c, default_qattn_tiles as q; m=c($b); print(m, q($b, m))")"
        wait_my_gpu
        echo "[$(date '+%m-%d %H:%M')] GPU$CUDA_VISIBLE_DEVICES $ds b$b m$m S$S random prior"
        mkdir -p "logs/calib/$ds"
        $PY foveatile_internvl.py --dataset "$ds" --budget "$b" --coverage_tiles "$m" --qattn_tiles "$S" --random_prior \
            --batch_size 4 --output_json "results/calib/$ds/foveatile_b${b}_m${m}_S${S}_randprior_InternVL3_5-8B.json" \
            >> "logs/calib/$ds/foveatile_b${b}_m${m}_S${S}_randprior.log" 2>&1 || echo "  [error] $ds b$b"
    done
}
for ((g = 0; g < ${#GPUS[@]}; g++)); do CUDA_VISIBLE_DEVICES=${GPUS[$g]} run_shard "$g" "${#GPUS[@]}" & done
wait
echo "[$(date '+%m-%d %H:%M')] done"
