#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# v1.1 候選（docs/04_calibration_protocol.md §7b）：tile 打分的 query × 聚合，只在 held-out 校準集上跑。
#   query ∈ QUERY_MODES、agg ∈ AGGS（扣掉 all × mean = v1.0），m、S 用 v1.0 預設且明確寫進 tag，
#   跟既有的 v1.0 對照檔 results/calib/<ds>/svma_b<B>_m<m>_S<S>_*.json 同名前綴，可以直接配對比較。
#
# 每張卡同時跑 PROCS_PER_GPU 個程序（A6000 48GB，SVMA 一個程序約 18GB）。可中斷續跑。
#   nohup ./run_calibrate_v11.sh > logs/calib/v11.log 2>&1 &
#   QUERY_MODES="stem" AGGS="mean" BUDGETS="1536" ./run_calibrate_v11.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
read -r -a CALIB_SETS <<< "${CALIB_SETS:-mme_rw_calib infovqa_train_calib}"
read -r -a BUDGETS <<< "${BUDGETS:-1024 1536 2560}"
read -r -a QUERY_MODES <<< "${QUERY_MODES:-all stem stemopt}"
read -r -a AGGS <<< "${AGGS:-mean top0.1 top0.25}"
read -r -a GPUS <<< "${GPUS:-0 1}"
PROCS_PER_GPU=${PROCS_PER_GPU:-2}
BATCH=${BATCH:-4}
OUT_ROOT=results/calib
LOG_ROOT=logs/calib
MTAG=$(basename "$MODEL")

run_one() {   # $1 dataset $2 budget $3 query $4 agg
    local ds=$1 b=$2 q=$3 agg=$4 m S tag json log
    read -r m S <<< "$($PY -c "
from svma_config import default_coverage_tiles, default_qattn_tiles
m = default_coverage_tiles($b); print(m, default_qattn_tiles($b, m))")"
    tag="svma_b${b}_m${m}_S${S}"
    [ "$q" != all ] && tag="${tag}_q${q}"
    [ "$agg" != mean ] && tag="${tag}_agg${agg}"
    json="$OUT_ROOT/$ds/${tag}_${MTAG}.json"; log="$LOG_ROOT/$ds/${tag}.log"
    mkdir -p "$OUT_ROOT/$ds" "$LOG_ROOT/$ds"
    echo "[$(date '+%m-%d %H:%M')] GPU$CUDA_VISIBLE_DEVICES $ds $tag"
    $PY internvl_stream.py --dataset "$ds" --mode svma --budget "$b" --coverage_tiles "$m" --qattn_tiles "$S" \
        --query_mode "$q" --tile_agg "$agg" --batch_size "$BATCH" --model_name "$MODEL" \
        --output_json "$json" >> "$log" 2>&1 || echo "  [error] $ds $tag，看 $log"
}

jobs=()
for ds in "${CALIB_SETS[@]}"; do
    for b in "${BUDGETS[@]}"; do
        for q in "${QUERY_MODES[@]}"; do
            for agg in "${AGGS[@]}"; do
                [ "$q" = all ] && [ "$agg" = mean ] && continue      # v1.0 對照已經有了
                jobs+=("$ds $b $q $agg")
            done
        done
    done
done
slots=()
for g in "${GPUS[@]}"; do for ((p = 0; p < PROCS_PER_GPU; p++)); do slots+=("$g"); done; done
echo "== v1.1 calibration：${#jobs[@]} 個工作，${#slots[@]} 個並行槽（GPUS=${GPUS[*]} × $PROCS_PER_GPU）=="

run_shard() {   # $1 slot index
    local k
    for ((k = $1; k < ${#jobs[@]}; k += ${#slots[@]})); do run_one ${jobs[$k]}; done
}
for ((s = 0; s < ${#slots[@]}; s++)); do
    CUDA_VISIBLE_DEVICES=${slots[$s]} run_shard "$s" &
    sleep 20      # 錯開模型載入
done
wait
echo "[$(date '+%m-%d %H:%M')] done"
