#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 測試集全套：DATASETS × {uncompressed, SVMA, FastV, VisionZip, IF-Prune, SparseVLM} × BUDGETS。
# 所有結果放 results/runs/<dataset>/<method>_b<B>_<model>.json（uncompressed 沒有 budget），
# 之後用 summarize_results.py 一次彙整（選擇題 accuracy / ANLS / CircularEval 依 dataset 自動選）。
#
#   uncompressed / SVMA ：internvl_stream.py（SVMA 用 svma_config 的預設 m、S）
#   FastV / VisionZip / IF-Prune：各自的 *_internvl.py，照論文 / 官方預設超參數
#   SparseVLM ：先用 CALIB_IMAGES 張圖校準 λ（auto_lam 取中位數，只看 token 數、不看答案），再跑全量；
#               原圖 token <= budget 的圖不剪枝（sparsevlm_internvl.budget_lam）
#
# 所有 (dataset, method, budget) 組合排成工作清單，GPUS 的每張卡輪流分到一份、各自只跑一個程序
# （TTFT 不在這裡量，之後用 bench_metrics.py 另外量）。可中斷續跑：每支腳本都會從結果檔斷點接著跑。
#
# 用法：
#   nohup ./run_test_suite.sh > logs/test_suite.log 2>&1 &
#   DATASETS="vstar" METHODS="svma fastv" BUDGETS="1024 2048" GPUS="0" ./run_test_suite.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
read -r -a DATASETS <<< "${DATASETS:-vstar mme_rw_lite}"
read -r -a METHODS <<< "${METHODS:-uncompressed svma fastv visionzip ifprune sparsevlm}"
read -r -a BUDGETS <<< "${BUDGETS:-1024 2048 3072 4096 5120}"
read -r -a GPUS <<< "${GPUS:-0 1}"
NUM_IMAGES=${NUM_IMAGES:--1}
CALIB_IMAGES=${CALIB_IMAGES:-32}
MTAG=$(basename "$MODEL")
MIN_FREE_GB=${MIN_FREE_GB:-30}     # 每個工作開始前，這張卡至少要有這麼多空間（GPU 跟別人共用；0 = 不檢查）
MAX_WAIT_H=${MAX_WAIT_H:-24}       # 每次最多等幾小時，超過就跳過這個工作（不動別人的程序）

wait_my_gpu() {   # 等 CUDA_VISIBLE_DEVICES 這張卡空出 MIN_FREE_GB；超時回傳 1
    [ "$MIN_FREE_GB" -gt 0 ] || return 0
    local start=$(date +%s) free
    while :; do
        free=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)
        [ "$free" -ge $((MIN_FREE_GB * 1024)) ] && return 0
        if [ $(( $(date +%s) - start )) -ge $(( MAX_WAIT_H * 3600 )) ]; then
            echo "  [TIMEOUT] GPU$CUDA_VISIBLE_DEVICES 等了 ${MAX_WAIT_H} 小時還沒有 ${MIN_FREE_GB} GB，跳過"
            return 1
        fi
        sleep 120
    done
}
OUT=${OUT:-results/runs}
LOG=${LOG_DIR:-logs/runs}

run_job() {   # $1 dataset $2 method $3 budget（uncompressed 給 0）
    local ds=$1 m=$2 b=$3 json log common lam
    mkdir -p "$OUT/$ds" "$LOG/$ds"
    if [ "$m" = uncompressed ]; then json="$OUT/$ds/uncompressed_$MTAG.json"; log="$LOG/$ds/uncompressed.log"
    else json="$OUT/$ds/${m}_b${b}_$MTAG.json"; log="$LOG/$ds/${m}_b${b}.log"; fi
    wait_my_gpu || { echo "  [skip] $ds $m b$b（等 GPU 超時）"; return; }
    echo "[$(date '+%m-%d %H:%M')] GPU$CUDA_VISIBLE_DEVICES $ds $m ${b/#0/}"
    # baseline 腳本的 --dataset 只認 hrbench，4K / 8K 用 HRBENCH_SPLIT 環境變數指定（internvl_stream.py 直接認 hrbench_4k）
    local bds=$ds
    if [[ $ds == hrbench_* ]]; then bds=hrbench; export HRBENCH_SPLIT=$ds; fi
    common="--use_ds --dataset $bds --num_images $NUM_IMAGES --model_name $MODEL --save --output_json $json"
    case $m in
        uncompressed) $PY internvl_stream.py --dataset "$ds" --mode uncompressed --batch_size 1 \
                          --num_images "$NUM_IMAGES" --model_name "$MODEL" --output_json "$json" >> "$log" 2>&1 ;;
        svma)         $PY internvl_stream.py --dataset "$ds" --mode svma --budget "$b" --batch_size 4 \
                          --num_images "$NUM_IMAGES" --model_name "$MODEL" --output_json "$json" >> "$log" 2>&1 ;;
        fastv)        $PY fastv_internvl.py $common --budget "$b" --run_fastv >> "$log" 2>&1 ;;
        visionzip)    $PY visionzip_internvl.py $common --budget "$b" --run_visionzip >> "$log" 2>&1 ;;
        ifprune)      $PY ifprune_internvl.py $common --budget "$b" --run_ifprune >> "$log" 2>&1 ;;
        sparsevlm)
            lam=$($PY sparsevlm_internvl.py --use_ds --dataset "$bds" --num_images "$CALIB_IMAGES" --model_name "$MODEL" \
                      --budget "$b" --max_new_tokens 1 --auto_lam --run_sparsevlm 2>&1 | tee -a "$log.calib" \
                  | grep -oP '\[auto_lam\] chosen lam=\K[0-9.eE+-]+' \
                  | $PY -c "import sys,statistics as s; v=[float(x) for x in sys.stdin.read().split()]; print(round(s.median(v),4) if v else 0.1)")
            echo "  $ds sparsevlm b$b：校準 λ=$lam" | tee -a "$log"
            $PY sparsevlm_internvl.py $common --budget "$b" --lam "$lam" --run_sparsevlm >> "$log" 2>&1 ;;
    esac
    [ $? -eq 0 ] || echo "  [error] $ds $m b$b，看 $log"
}

jobs=()
for ds in "${DATASETS[@]}"; do
    for m in "${METHODS[@]}"; do
        if [ "$m" = uncompressed ]; then jobs+=("$ds $m 0"); continue; fi
        for b in "${BUDGETS[@]}"; do jobs+=("$ds $m $b"); done
    done
done
echo "== test suite：${#jobs[@]} 個工作，GPUS=${GPUS[*]} =="

run_shard() {   # $1 shard index $2 shard 數
    local k
    for ((k = $1; k < ${#jobs[@]}; k += $2)); do run_job ${jobs[$k]}; done
}
for ((g = 0; g < ${#GPUS[@]}; g++)); do
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} run_shard "$g" "${#GPUS[@]}" &
done
wait

for ds in "${DATASETS[@]}"; do $PY summarize_results.py --root "$OUT" --datasets "$ds" --ref uncompressed || true; done
echo "[$(date '+%m-%d %H:%M')] done"
