#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SparseVLM 對照組：跟 run_internvl.sh（你的 streaming 方法）同一組 budget，
# 跑 accuracy + 主張指標（peak memory / KV cache），方便放進同一張表比較。
#
# 跟 run_internvl.sh 的差異只有一個地方，但很關鍵：SparseVLM 的 λ（Eq.7）
# 沒有官方數值，需要「先校準、再大量跑」，不能每張圖都即時 --auto_lam
# （見 sparsevlm_internvl.py 的 auto_tune_lam() docstring：每張圖要多跑
# 最多 --auto_lam_max_evals 次 prefill-only 探測，1000 張圖 x 5 個 budget
# 乘起來的額外算力不便宜）。所以這支腳本分兩階段：
#
#   Phase 1（校準，小 n = CALIB_IMAGES）：對每個 budget 跑一次
#     --auto_lam --run_sparsevlm，只看 stdout 印出的「chosen lam」，取中位數
#     當這個 budget 的代表 λ——這個 λ 公平代表 SparseVLM 的漸進機制（Eq.7
#     自然壓到接近 budget，不靠保底硬裁切），只是不逐張圖重新校準。
#   Phase 2（accuracy，大 n = NUM_IMAGES）：用 Phase 1 校準出的固定 λ 正式跑，
#     速度回到跟 streaming 差不多的量級。
#
# 產出檔名：
#   accuracy JSON   : out_sparsevlm_<model>_<dataset>_b<budget>.json
#   circular summary: out_sparsevlm_<model>_<dataset>_circ_b<budget>.json
#   metrics CSV     : metrics_sparsevlm_<model>_<dataset>.csv（跟 run_internvl.sh
#                     的 metrics_<model>_<dataset>.csv 並排看，兩份 CSV 的
#                     method 欄位命名不同，可以直接合併/對照）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}                   # hrbench | docvqa | infovqa，跟 run_internvl.sh 保持一致才能對照
read -r -a BUDGETS <<< "${BUDGETS:-1024 2048 3072 4096 5120}"
NUM_IMAGES=${NUM_IMAGES:--1}                 # accuracy 用（投稿前拉全量，跟 run_internvl.sh 的 NUM_IMAGES 對齊）
CALIB_IMAGES=${CALIB_IMAGES:-8}              # λ 校準用，小 n 就夠（校準成本是 accuracy pass 的好幾倍/張）
METRIC_IMAGES=${METRIC_IMAGES:-24}           # peak/KV cache 用，小 n 即可，跟 run_internvl.sh 一致
MAX_NUM=${MAX_NUM:-48}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-500}
FALLBACK_LAM=${FALLBACK_LAM:-0.1}            # 校準失敗（stdout 沒抓到任何 chosen lam）時的備援值

# --auto_tune_lam 的搜尋範圍/收斂條件，預設跟 sparsevlm_internvl.py 的 CLI 預設一致
AUTO_LAM_LO=${AUTO_LAM_LO:-0.005}
AUTO_LAM_HI=${AUTO_LAM_HI:-3.0}
AUTO_LAM_MAX_EVALS=${AUTO_LAM_MAX_EVALS:-6}
AUTO_LAM_OVERSHOOT_TOL=${AUTO_LAM_OVERSHOOT_TOL:-0.5}

DS_ARGS="--use_ds --dataset $DATASET"
MTAG=$(basename "$MODEL")

# 跟 run_internvl.sh 用同一招從 internvl_stream.py 讀 HR-Bench/DocVQA 的
# split/prompt 常數——sparsevlm_internvl.py 也是 import 同一份常數，語意一致。
PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo letter)
SPLIT_TAG=${HRBENCH_SPLIT:-hrbench_4k}   # 跟 internvl_stream.py 同一個環境變數
SPLIT_TAG=${SPLIT_TAG#hrbench_}
DOCVQA_PROMPT_MODE=$(grep -oP 'DOCVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
INFOVQA_PROMPT_MODE=$(grep -oP 'INFOVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
# docvqa / infovqa 共用的 prompt 模式，後面的 OUTBASE / eval / similarity 都吃這個。
if [ "$DATASET" = infovqa ]; then QA_PROMPT_MODE=$INFOVQA_PROMPT_MODE; else QA_PROMPT_MODE=$DOCVQA_PROMPT_MODE; fi

# 輸出放 results/baselines/internvl_sparsevlm_<dataset>/（HR-Bench 8K 是 hrbench8k），OUTDIR 可覆寫
if [ "$DATASET" = hrbench ] && [ "${SPLIT_TAG#hrbench_}" = 8k ]; then DSDIR=hrbench8k; else DSDIR=$DATASET; fi
OUTDIR=${OUTDIR:-results/baselines/internvl_sparsevlm_${DSDIR}}
mkdir -p "$OUTDIR"

if [ "$DATASET" = hrbench ]; then
    OUTBASE="$OUTDIR/out_sparsevlm_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    OUTBASE="$OUTDIR/out_sparsevlm_${QA_PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
else
    OUTBASE="$OUTDIR/out_sparsevlm_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi

echo "== SparseVLM :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  hrbench_split=$SPLIT_TAG  hrbench_prompt=$PROMPT_MODE  docvqa_prompt=$DOCVQA_PROMPT_MODE  infovqa_prompt=$INFOVQA_PROMPT_MODE =="

# 每個 budget 校準出來的 λ，供本次 shell 執行期間重複使用（例如最後 bench_metrics
# 那段如果你想手動固定 λ 也可以參考這裡印出來的值，目前 bench_metrics.py 本身仍是
# 逐張圖 auto_lam，見腳本最後一段的說明）。
declare -A LAM_FOR_BUDGET

calibrate_lam() {
    # Phase 1：小 n 跑 --auto_lam，只取 stdout 印出的 chosen lam 取中位數。
    # --run_standard 不用開（校準不需要 baseline 答案），省一點時間。
    local budget=$1
    local lams
    lams=$($PY sparsevlm_internvl.py --model_name "$MODEL" $DS_ARGS --num_images "$CALIB_IMAGES" \
                --max_num "$MAX_NUM" --budget "$budget" --max_new_tokens 1 \
                --auto_lam --auto_lam_lo "$AUTO_LAM_LO" --auto_lam_hi "$AUTO_LAM_HI" \
                --auto_lam_max_evals "$AUTO_LAM_MAX_EVALS" --auto_lam_overshoot_tol "$AUTO_LAM_OVERSHOOT_TOL" \
                --run_sparsevlm \
        2>&1 | tee /dev/stderr | grep -oP '\[auto_lam\] chosen lam=\K[0-9.eE+-]+') || true

    if [ -z "$lams" ]; then
        echo "$FALLBACK_LAM"
    else
        $PY -c "
import statistics, sys
vals = [float(x) for x in sys.argv[1:]]
print(round(statistics.median(vals), 4))
" $lams
    fi
}

for BUDGET in "${BUDGETS[@]}"; do
    echo
    echo "==== $MTAG $DATASET b$BUDGET :: Phase 1 calibrate lam (n=$CALIB_IMAGES) ===="
    LAM=$(calibrate_lam "$BUDGET")
    LAM_FOR_BUDGET[$BUDGET]=$LAM
    echo "   -> budget=$BUDGET  calibrated lam=$LAM"

    J="${OUTBASE}_b${BUDGET}.json"
    echo
    echo "==== $MTAG $DATASET b$BUDGET :: Phase 2 accuracy run (n=$NUM_IMAGES, lam=$LAM) ===="
    $PY sparsevlm_internvl.py --model_name "$MODEL" $DS_ARGS --num_images "$NUM_IMAGES" \
        --max_num "$MAX_NUM" --budget "$BUDGET" --lam "$LAM" --max_new_tokens "$MAX_NEW_TOKENS" \
        --run_standard --run_sparsevlm --save --output_json "$J"

    # candidates 只有一個 key（這個 budget/lam 的 tag），直接讀出來當 --baseline_tag，
    # 避免自己組字串跟 sparsevlm_internvl.py 的 f"sparsevlm_budget={budget}_lam={lam}"
    # 因為浮點數格式化不一致而對不上。
    TAG=$($PY -c "import json; print(list(json.load(open('$J'))['candidates'].keys())[0])")

    echo; echo "==== $MTAG $DATASET b$BUDGET :: eval (tag=$TAG) ===="
    case "$DATASET" in
        hrbench)
            $PY evaluate_hrbench_circular.py --input_json "$J" --baseline_tag "$TAG" \
                --group_by id \
                --dump_csv "${OUTBASE}_circ_b${BUDGET}.csv" \
                --summary_json "${OUTBASE}_circ_b${BUDGET}.json" || true
            $PY evaluate_hrbench.py --input_json "$J" --baseline_tag "$TAG" \
                --summary_json "${OUTBASE}_vanilla_b${BUDGET}.json" || true ;;
        docvqa|infovqa)
            $PY evaluate_anls.py --input_json "$J" --baseline_tag "$TAG" \
                --dump_csv "${OUTBASE}_anls_b${BUDGET}.csv" \
                --summary_json "${OUTBASE}_b${BUDGET}.summary.json" || true ;;
    esac
done

# ---- 主張指標：peak memory / KV cache（跑一次涵蓋所有 budget，跟 stream_b<K>
#      並排比較）。bench_metrics.py 內部逐張圖對 SparseVLM 自己做 auto_lam
#      校準（小 n=METRIC_IMAGES 才跑得動），跟 Phase 1/2 的固定 λ 是兩回事，
#      不需要共用 LAM_FOR_BUDGET——記憶體指標本來就要看「校準到位」的 SparseVLM
#      表現，跟 accuracy pass 用固定 λ 圖省時間的考量不衝突。----
echo; echo "==== $MTAG $DATASET :: metrics (peak memory / KV cache, streaming vs SparseVLM) ===="
$PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
    --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" \
    --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
    --run_sparsevlm \
    --out_csv "$OUTDIR/metrics_sparsevlm_${METRIC_TAG}_${DATASET}.csv" || true

echo
echo "===== SparseVLM done  ($MTAG / $DATASET) ====="
echo "calibrated lam per budget:"
for BUDGET in "${BUDGETS[@]}"; do
    echo "  budget=$BUDGET  lam=${LAM_FOR_BUDGET[$BUDGET]}"
done
if [ "$DATASET" = hrbench ]; then
    echo "circular summary : ${OUTBASE}_circ_b*.json"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    echo "ANLS summary     : ${OUTBASE}_b*.summary.json  (per-sample: ${OUTBASE}_anls_b*.csv)"
fi
echo "metrics          : $OUTDIR/metrics_sparsevlm_${METRIC_TAG}_${DATASET}.csv"
echo "                    (跟 run_internvl.sh 產的 metrics_${METRIC_TAG}_${DATASET}.csv 並排比較)"
