#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 同預算的「直接降解析度」對照（docs/06 §9 #2，2026-11-07）：
#   lowres  ：官方動態切 tile，上限 K 塊（含縮圖）→ foveatile_internvl.py --mode uncompressed --max_num K（batch 1）
#   coverage：FoveaTile 覆蓋 tile 拿全部預算（m = K，無細節 tile）→ --mode foveatile --coverage_tiles K（batch 4）
# 結果 results/runs_lowres/<ds>/{lowres,coverage}_b<B>_<model>.json；可中斷續跑（腳本會從結果檔斷點接著跑）。
# 用法：nohup ./run_lowres_baselines.sh > logs/runs_lowres/suite.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
MTAG=$(basename "$MODEL")
read -r -a DATASETS <<< "${DATASETS:-vstar hrbench_4k hrbench_8k mme_rw_lite infovqa docvqa}"
read -r -a BUDGETS <<< "${BUDGETS:-1536 2560 3072}"
read -r -a VARIANTS <<< "${VARIANTS:-coverage lowres}"
read -r -a GPUS <<< "${GPUS:-0 1}"
NUM_IMAGES=${NUM_IMAGES:--1}
OUT=${OUT:-results/runs_lowres}
LOG=${LOG_DIR:-logs/runs_lowres}
MIN_FREE_GB=${MIN_FREE_GB:-30}
MAX_WAIT_H=${MAX_WAIT_H:-24}

wait_my_gpu() {   # GPU 跟別人共用：等這張卡空出 MIN_FREE_GB，最多 MAX_WAIT_H 小時，不動別人的程序
    [ "$MIN_FREE_GB" -gt 0 ] || return 0
    local start=$(date +%s) free
    while :; do
        free=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)
        [ "$free" -ge $((MIN_FREE_GB * 1024)) ] && return 0
        if [ $(( $(date +%s) - start )) -ge $(( MAX_WAIT_H * 3600 )) ]; then
            echo "  [TIMEOUT] GPU$CUDA_VISIBLE_DEVICES 等了 ${MAX_WAIT_H} 小時，跳過"; return 1
        fi
        sleep 120
    done
}

run_job() {   # $1 dataset $2 variant $3 budget
    local ds=$1 v=$2 b=$3 K=$(( $3 / 256 ))
    mkdir -p "$OUT/$ds" "$LOG/$ds"
    local json="$OUT/$ds/${v}_b${b}_$MTAG.json" log="$LOG/$ds/${v}_b${b}.log"
    wait_my_gpu || { echo "  [skip] $ds $v b$b"; return; }
    echo "[$(date '+%m-%d %H:%M')] GPU$CUDA_VISIBLE_DEVICES $ds $v b$b"
    case $v in
        lowres)   $PY foveatile_internvl.py --dataset "$ds" --mode uncompressed --batch_size 1 --max_num "$K" \
                      --num_images "$NUM_IMAGES" --model_name "$MODEL" --output_json "$json" >> "$log" 2>&1 ;;
        coverage) $PY foveatile_internvl.py --dataset "$ds" --mode foveatile --budget "$b" --coverage_tiles "$K" --batch_size 4 \
                      --max_num 48 --num_images "$NUM_IMAGES" --model_name "$MODEL" --output_json "$json" >> "$log" 2>&1 ;;
    esac
    [ $? -eq 0 ] || echo "  [error] $ds $v b$b，看 $log"
}

jobs=()
for b in "${BUDGETS[@]}"; do for ds in "${DATASETS[@]}"; do for v in "${VARIANTS[@]}"; do jobs+=("$ds $v $b"); done; done; done
echo "== lowres baselines：${#jobs[@]} 個工作，GPUS=${GPUS[*]} =="
run_shard() { local k; for ((k = $1; k < ${#jobs[@]}; k += $2)); do run_job ${jobs[$k]}; done; }
for ((g = 0; g < ${#GPUS[@]}; g++)); do CUDA_VISIBLE_DEVICES=${GPUS[$g]} run_shard "$g" "${#GPUS[@]}" & done
wait
echo "[$(date '+%m-%d %H:%M')] done"
