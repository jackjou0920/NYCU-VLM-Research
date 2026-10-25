#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# VisionZip 對照組：跟 run_internvl.sh（你的 streaming 方法）/ run_sparsevlm.sh /
# run_fastv.sh 同一組 budget，跑 accuracy + 主張指標（peak memory / KV cache /
# TTFT），方便放進同一張表比較。
#
# 跟 run_fastv.sh 的差異：
#   1. VisionZip 沒有需要校準的超參數（budget 平均分給每個 tile，contextual 佔
#      5/32，見 visionzip_internvl.py 檔頭），所以跟 FastV 一樣只有一個 phase。
#   2. 指標段的 stream_b<K> 用 bench_metrics.py 預設的 SVMA 設定（--qattn_tiles auto），
#      跟 run_internvl.sh 的 accuracy 設定一致。
#   3. accuracy 段不是註解掉的；只想重跑其中一段用 RUN_ACCURACY=0 / RUN_METRICS=0。
#
# 產出檔名：
#   accuracy JSON   : out_visionzip_<promptmode>_<model>_<dataset>_b<budget>.json
#   ANLS / circular : 同一個前綴 + _anls_b<K>.csv / _b<K>.summary.json / _circ_b<K>.*
#   metrics CSV     : metrics_visionzip_<model>_<dataset>.csv（一次跑 uncompressed /
#                     stream_b<K> / visionzip_b<K>，同一批圖並排）
#
# 斷點續跑：visionzip_internvl.py 會讀既有的 output_json 逐張跳過已完成的，
# 中斷後直接重跑同一個指令即可。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}                   # hrbench | docvqa | infovqa，跟 run_internvl.sh 保持一致才能對照
BUDGET_LIST=${BUDGET_LIST:-"1024 2048 3072 4096 5120"}
BUDGETS=($BUDGET_LIST)
NUM_IMAGES=${NUM_IMAGES:--1}                 # accuracy 用（投稿前拉全量，跟 run_internvl.sh 的 NUM_IMAGES 對齊）
MAX_NUM=${MAX_NUM:-48}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-500}
METRIC_IMAGES=${METRIC_IMAGES:-24}           # peak/KV cache 用，小 n 即可，跟 run_sparsevlm.sh / run_fastv.sh 一致
CTX_RATIO=${CTX_RATIO:-0.15625}              # contextual 佔比 = 5/32，論文 Table 8/9 每一組數字都是這個比例
RUN_ACCURACY=${RUN_ACCURACY:-1}
RUN_METRICS=${RUN_METRICS:-1}

DS_ARGS="--use_ds --dataset $DATASET"
MTAG=$(basename "$MODEL")

# 跟 run_fastv.sh 用同一招從 internvl_stream.py 讀 HR-Bench/DocVQA 的
# split/prompt 常數——visionzip_internvl.py 也是 import 同一份常數，語意一致。
PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo letter)
SPLIT_TAG=${HRBENCH_SPLIT:-hrbench_4k}   # 跟 internvl_stream.py 同一個環境變數
SPLIT_TAG=${SPLIT_TAG#hrbench_}
DOCVQA_PROMPT_MODE=$(grep -oP 'DOCVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
INFOVQA_PROMPT_MODE=$(grep -oP 'INFOVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
# docvqa / infovqa 共用的 prompt 模式，後面的 OUTBASE / eval / similarity 都吃這個。
if [ "$DATASET" = infovqa ]; then QA_PROMPT_MODE=$INFOVQA_PROMPT_MODE; else QA_PROMPT_MODE=$DOCVQA_PROMPT_MODE; fi

# 輸出放 results/baselines/internvl_visionzip_<dataset>/（HR-Bench 8K 是 hrbench8k），OUTDIR 可覆寫
if [ "$DATASET" = hrbench ] && [ "${SPLIT_TAG#hrbench_}" = 8k ]; then DSDIR=hrbench8k; else DSDIR=$DATASET; fi
OUTDIR=${OUTDIR:-results/baselines/internvl_visionzip_${DSDIR}}
mkdir -p "$OUTDIR"

if [ "$DATASET" = hrbench ]; then
    OUTBASE="$OUTDIR/out_visionzip_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    OUTBASE="$OUTDIR/out_visionzip_${QA_PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
else
    OUTBASE="$OUTDIR/out_visionzip_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi


echo "== VisionZip :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  ctx_ratio=$CTX_RATIO  hrbench_split=$SPLIT_TAG  hrbench_prompt=$PROMPT_MODE  docvqa_prompt=$DOCVQA_PROMPT_MODE  infovqa_prompt=$INFOVQA_PROMPT_MODE =="

if [ "$RUN_ACCURACY" = 1 ]; then
    for BUDGET in "${BUDGETS[@]}"; do
        J="${OUTBASE}_b${BUDGET}.json"
        echo
        echo "==== $MTAG $DATASET b$BUDGET :: accuracy run (n=$NUM_IMAGES, ctx_ratio=$CTX_RATIO) ===="
        $PY visionzip_internvl.py --model_name "$MODEL" $DS_ARGS --num_images "$NUM_IMAGES" \
            --max_num "$MAX_NUM" --budget "$BUDGET" --contextual_ratio "$CTX_RATIO" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --run_standard --run_visionzip --save --output_json "$J"

        # candidates 只有一個 key（這個 budget 的 tag），直接讀出來當 --baseline_tag，
        # 避免自己組字串跟 visionzip_internvl.py 的 f"visionzip_budget={budget}" 對不上。
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
fi

# ---- 主張指標：peak memory / KV cache / TTFT（跑一次涵蓋所有 budget，跟
#      stream_b<K>(SVMA) 同一批圖並排比較）。----
if [ "$RUN_METRICS" = 1 ]; then
    echo; echo "==== $MTAG $DATASET :: metrics (peak memory / KV cache / TTFT, SVMA vs VisionZip) ===="
    $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
        --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" \
        --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
        --run_visionzip --visionzip_contextual_ratio "$CTX_RATIO" \
        --out_csv "$OUTDIR/metrics_visionzip_${METRIC_TAG}_${DATASET}.csv" || true
fi

echo
echo "===== VisionZip done  ($MTAG / $DATASET) ====="
echo "contextual_ratio=$CTX_RATIO (contextual = round(keep_per_tile * ratio))"
if [ "$DATASET" = hrbench ]; then
    echo "circular summary : ${OUTBASE}_circ_b*.json"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    echo "ANLS summary     : ${OUTBASE}_b*.summary.json  (per-sample: ${OUTBASE}_anls_b*.csv)"
fi
echo "metrics          : $OUTDIR/metrics_visionzip_${METRIC_TAG}_${DATASET}.csv"
echo "                    (uncompressed / stream_b<K>(SVMA) / visionzip_b<K> 同一批圖並排)"
