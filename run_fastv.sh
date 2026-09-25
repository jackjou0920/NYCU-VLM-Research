#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# FastV 對照組：跟 run_internvl.sh（你的 streaming 方法）/ run_sparsevlm.sh
# 同一組 budget，跑 accuracy，方便放進同一張表比較。
#
# 跟 run_sparsevlm.sh 最大的差異：FastV 沒有 SparseVLM 那種需要另外校準的
# λ——budget 直接對應 fastv_internvl.py 的 ATTENTION_RANK（保留幾個 image
# token），final_vis_tokens 恆等於 min(budget, raw_vis_tokens)，所以這裡只有
# 一個 phase，不用像 run_sparsevlm.sh 先跑一輪校準。
#
# 產出檔名：
#   accuracy JSON   : out_fastv_<model>_<dataset>_b<budget>.json
#   circular summary: out_fastv_<model>_<dataset>_circ_b<budget>.json
#
#   metrics CSV     : metrics_fastv_<model>_<dataset>.csv（bench_metrics.py 一次
#                     跑 uncompressed / stream_b<K>(SVMA) / fastv_b<K>，同一批圖
#                     並排比較 peak memory / KV cache MB）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}                   # hrbench | docvqa | infovqa，跟 run_internvl.sh 保持一致才能對照
BUDGETS=(1024 2048 3072 4096 5120)
NUM_IMAGES=${NUM_IMAGES:--1}                 # accuracy 用（投稿前拉全量，跟 run_internvl.sh 的 NUM_IMAGES 對齊）
MAX_NUM=${MAX_NUM:-48}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-500}
METRIC_IMAGES=${METRIC_IMAGES:-24}           # peak/KV cache 用，小 n 即可，跟 run_sparsevlm.sh 一致
AGG_LAYER=${AGG_LAYER:-2}                    # 官方預設值，所有官方 eval 腳本都固定用這個

DS_ARGS="--use_ds --dataset $DATASET"
MTAG=$(basename "$MODEL")

# 跟 run_sparsevlm.sh 用同一招從 internvl_stream.py 讀 HR-Bench/DocVQA 的
# split/prompt 常數——fastv_internvl.py 也是 import 同一份常數，語意一致。
PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo letter)
SPLIT_TAG=$(grep -oP 'HRBENCH_SPLIT\s*=\s*"\K[^"]+' internvl_stream.py || echo hrbench_4k)
SPLIT_TAG=${SPLIT_TAG#hrbench_}
DOCVQA_PROMPT_MODE=$(grep -oP 'DOCVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
INFOVQA_PROMPT_MODE=$(grep -oP 'INFOVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
# docvqa / infovqa 共用的 prompt 模式，後面的 OUTBASE / eval / similarity 都吃這個。
if [ "$DATASET" = infovqa ]; then QA_PROMPT_MODE=$INFOVQA_PROMPT_MODE; else QA_PROMPT_MODE=$DOCVQA_PROMPT_MODE; fi

if [ "$DATASET" = hrbench ]; then
    OUTBASE="out_fastv_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    OUTBASE="out_fastv_${QA_PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
else
    OUTBASE="out_fastv_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi

echo "== FastV :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  agg_layer=$AGG_LAYER  hrbench_split=$SPLIT_TAG  hrbench_prompt=$PROMPT_MODE  docvqa_prompt=$DOCVQA_PROMPT_MODE  infovqa_prompt=$INFOVQA_PROMPT_MODE =="

for BUDGET in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${BUDGET}.json"
    echo
    echo "==== $MTAG $DATASET b$BUDGET :: accuracy run (n=$NUM_IMAGES, agg_layer=$AGG_LAYER) ===="
    $PY fastv_internvl.py --model_name "$MODEL" $DS_ARGS --num_images "$NUM_IMAGES" \
        --max_num "$MAX_NUM" --budget "$BUDGET" --agg_layer "$AGG_LAYER" --max_new_tokens "$MAX_NEW_TOKENS" \
        --run_standard --run_fastv --save --output_json "$J"

    # candidates 只有一個 key（這個 budget 的 tag），直接讀出來當 --baseline_tag，
    # 避免自己組字串跟 fastv_internvl.py 的 f"fastv_budget={budget}_k={agg_layer}" 對不上。
    TAG=$($PY -c "import json; print(list(json.load(open('$J'))['candidates'].keys())[0])")

    echo; echo "==== $MTAG $DATASET b$BUDGET :: eval (tag=$TAG) ===="
    case "$DATASET" in
        hrbench)
            if [ "$PROMPT_MODE" = open ]; then
                $PY evaluate_hrbench.py --input_json "$J" --prompt_mode open \
                    --baseline_tag "$TAG" \
                    --summary_json "${OUTBASE}_open_b${BUDGET}.json" || true
            else
                $PY evaluate_hrbench_circular.py --input_json "$J" --baseline_tag "$TAG" \
                    --group_by id \
                    --dump_csv "${OUTBASE}_circ_b${BUDGET}.csv" \
                    --summary_json "${OUTBASE}_circ_b${BUDGET}.json" || true
                $PY evaluate_hrbench.py --input_json "$J" --baseline_tag "$TAG" \
                    --summary_json "${OUTBASE}_vanilla_b${BUDGET}.json" || true
            fi ;;
        docvqa|infovqa)
            if [ "$QA_PROMPT_MODE" = open ]; then
                : # open：沒有 GT，見下面跨 budget 的語意相似度
            else
                $PY evaluate_anls.py --input_json "$J" --baseline_tag "$TAG" \
                    --dump_csv "${OUTBASE}_anls_b${BUDGET}.csv" \
                    --summary_json "${OUTBASE}_b${BUDGET}.summary.json" || true
            fi ;;
    esac
done

# ---- open 模式：沒有 GT，改成跨 budget 的語意相似度（輸出 vs uncompressed
#      reference），跟 run_sparsevlm.sh 同一段邏輯，套用在 FastV 的 json 上 ----
IS_OPEN_RUN=0
[ "$DATASET" = hrbench ] && [ "$PROMPT_MODE" = open ] && IS_OPEN_RUN=1
{ [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; } && [ "$QA_PROMPT_MODE" = open ] && IS_OPEN_RUN=1
if [ "$IS_OPEN_RUN" = 1 ]; then
    echo; echo "==== $MTAG $DATASET :: semantic similarity vs reference (all budgets) ===="
    SIM_JSONS=(); for B in "${BUDGETS[@]}"; do SIM_JSONS+=("${OUTBASE}_b${B}.json"); done
    MERGED="${OUTBASE}_sim_merged.json"
    $PY merge_sim_inputs.py --out "$MERGED" "${SIM_JSONS[@]}" \
        && $PY semantic_similarity_evaluator.py --input_json "$MERGED" \
               --output_csv "${OUTBASE}_sim_persample.csv" \
               --summary_csv "${OUTBASE}_sim_summary.csv" || true
fi

# ---- 主張指標：peak memory / KV cache（跑一次涵蓋所有 budget，跟 stream_b<K>
#      (SVMA) 同一批圖並排比較）。FastV 沒有 λ 要校準，直接 --run_fastv。----
echo; echo "==== $MTAG $DATASET :: metrics (peak memory / KV cache, SVMA vs FastV) ===="
$PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
    --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" \
    --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
    --run_fastv --fastv_agg_layer "$AGG_LAYER" \
    --out_csv "metrics_fastv_${METRIC_TAG}_${DATASET}.csv" || true

echo
echo "===== FastV done  ($MTAG / $DATASET) ====="
echo "agg_layer=$AGG_LAYER for all budgets"
if [ "$IS_OPEN_RUN" = 1 ]; then
    [ "$DATASET" = hrbench ] && echo "open matchRef    : ${OUTBASE}_open_b*.json"
    echo "similarity table : ${OUTBASE}_sim_summary.csv  (per-sample: ${OUTBASE}_sim_persample.csv)"
elif [ "$DATASET" = hrbench ]; then
    echo "circular summary : ${OUTBASE}_circ_b*.json"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    echo "ANLS summary     : ${OUTBASE}_b*.summary.json  (per-sample: ${OUTBASE}_anls_b*.csv)"
fi
echo "metrics          : metrics_fastv_${METRIC_TAG}_${DATASET}.csv"
echo "                    (uncompressed / stream_b<K>(SVMA) / fastv_b<K> 同一批圖並排)"
