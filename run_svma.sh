#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SVMA / baseline 批次評測：DATASETS × MODES × BUDGETS，每個組合一個 JSON + 評分 + log，最後彙整成表。
#
#   結果 ：results/runs/<dataset>/<tag>_<model>.json（tag = svma_b<B> / lowres_fill_b<B> / uncompressed）
#          評分輸出放旁邊：<同名>.summary.json（ANLS）或 <同名>.circ.json（HR-Bench CircularEval）
#   log  ：logs/runs/<dataset>/<tag>.log
#   彙整 ：results/runs/summary.csv（summarize_results.py）
#   中斷後直接重跑同一個指令即可：已完成的題目會跳過（internvl_stream.py 會從斷點接著跑）。
#
# 用法：
#   ./run_svma.sh                                                    # 預設 4 個 dataset、SVMA、b1024 b2048
#   DATASETS="docvqa infovqa" BUDGETS="1024 2048 4096" ./run_svma.sh
#   MODES="svma uncompressed lowres_fill" NUM_IMAGES=500 ./run_svma.sh
#   GPUS="0 1" ./run_svma.sh                                         # dataset 輪流分到兩張卡平行跑
#   METRICS=1 ./run_svma.sh                                          # 另外用 bench_metrics.py 量 TTFT / peak memory
#
# 新增 dataset：在 internvl_stream.py 的 DATASETS 加一筆即可。名稱以 hrbench 開頭的用 CircularEval，
# 其他用 ANLS（要新的評分方式時，改下面的 evaluate()，以及 summarize_results.py 的 METRIC_OF）。
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
read -r -a DATASETS <<< "${DATASETS:-docvqa infovqa hrbench_4k hrbench_8k}"
read -r -a MODES <<< "${MODES:-svma}"                        # svma | uncompressed | lowres_fill
read -r -a BUDGETS <<< "${BUDGETS:-1024 2048}"
read -r -a GPUS <<< "${GPUS:-}"                              # 空 = 用目前看得到的卡（單一序列）
NUM_IMAGES=${NUM_IMAGES:--1}
BATCH=${BATCH:-4}
METRICS=${METRICS:-0}
METRIC_IMAGES=${METRIC_IMAGES:-24}
OUT_ROOT=results/runs
LOG_ROOT=logs/runs
MTAG=$(basename "$MODEL")

evaluate() {   # $1 = dataset, $2 = json
    local stem=${2%.json}
    case "$1" in
        mme_rw*)  echo "  ($1 用選擇題 accuracy，由 summarize_results.py 直接評分)" ;;
        hrbench*) $PY evaluate_hrbench_circular.py --input_json "$2" --group_by id \
                      --dump_csv "${stem}.circ.csv" --summary_json "${stem}.circ.json" ;;
        *)        $PY evaluate_anls.py --input_json "$2" \
                      --dump_csv "${stem}.anls.csv" --summary_json "${stem}.summary.json" ;;
    esac
}

run_dataset() {   # $1 = dataset：跑完這個 dataset 的所有 mode × budget
    local ds=$1 mode b tag json log
    mkdir -p "$OUT_ROOT/$ds" "$LOG_ROOT/$ds"
    for mode in "${MODES[@]}"; do
        for b in "${BUDGETS[@]}"; do
            if [ "$mode" = uncompressed ]; then tag=uncompressed; else tag="${mode}_b${b}"; fi
            json="$OUT_ROOT/$ds/${tag}_${MTAG}.json"
            log="$LOG_ROOT/$ds/${tag}.log"
            echo "[$(date '+%m-%d %H:%M')] $ds $tag  (log: $log)"
            if $PY internvl_stream.py --dataset "$ds" --mode "$mode" --budget "$b" \
                   --num_images "$NUM_IMAGES" --batch_size "$BATCH" --model_name "$MODEL" \
                   --output_json "$json" >> "$log" 2>&1; then
                evaluate "$ds" "$json" >> "$log" 2>&1 || echo "  [warn] $ds $tag 評分失敗，看 $log"
            else
                echo "  [error] $ds $tag 失敗，看 $log"
            fi
            [ "$mode" = uncompressed ] && break        # uncompressed 跟 budget 無關，跑一次就好
        done
    done
    if [ "$METRICS" = 1 ]; then
        echo "[$(date '+%m-%d %H:%M')] $ds metrics (TTFT / peak memory, n=$METRIC_IMAGES)"
        $PY bench_metrics.py --model_name "$MODEL" --dataset "$ds" --num_images "$METRIC_IMAGES" \
            --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
            --out_csv "$OUT_ROOT/$ds/metrics_${MTAG}.csv" >> "$LOG_ROOT/$ds/metrics.log" 2>&1 \
            || echo "  [warn] $ds metrics 失敗，看 $LOG_ROOT/$ds/metrics.log"
    fi
}

echo "== SVMA runs :: model=$MTAG  datasets=${DATASETS[*]}  modes=${MODES[*]}  budgets=${BUDGETS[*]}  n=$NUM_IMAGES  gpus=${GPUS[*]:-default} =="
if [ "${#GPUS[@]}" -gt 1 ]; then
    # dataset 輪流分配到各張卡；每張卡上依序跑自己的 dataset
    for g in "${!GPUS[@]}"; do
        (
            export CUDA_VISIBLE_DEVICES=${GPUS[$g]}
            for k in "${!DATASETS[@]}"; do
                [ $((k % ${#GPUS[@]})) -eq "$g" ] && run_dataset "${DATASETS[$k]}"
            done
        ) &
    done
    wait
else
    [ "${#GPUS[@]}" -eq 1 ] && export CUDA_VISIBLE_DEVICES=${GPUS[0]}
    for ds in "${DATASETS[@]}"; do run_dataset "$ds"; done
fi

$PY summarize_results.py --root "$OUT_ROOT" --datasets "${DATASETS[@]}" --num_images "$NUM_IMAGES"
