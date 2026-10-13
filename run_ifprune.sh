#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# IF-Prune 對照組：跟 run_internvl.sh（你的 streaming/SVMA 方法）/ run_fastv.sh /
# run_sparsevlm.sh 同一組 budget，跑 accuracy + 主張指標（peak memory / KV cache），
# 方便放進同一張表比較。
#
# 跟 run_sparsevlm.sh 的差異：IF-Prune 沒有需要校準的超參數。budget 直接是「剪枝後保留的視覺
# token 數」（= 官方 keep_ratio * vis_len，見 ifprune_internvl.py 檔頭第 2 點），
# final_vis_tokens 恆等於 min(budget, raw_vis_tokens)，所以只有一個 phase。
#
# 需要的外部權重（第一次跑會自動從 HF 下載，也可以用環境變數指到本地路徑）：
#   SMALL_CKPT : 官方 IF-Prune 小 VLM checkpoint（ZachSun/internvl2_5_1b_pruning_kl_lora32）
#   SMALL_BASE : 小 VLM 骨架（OpenGVLab/InternVL2_5-1B）
#
# 產出檔名：
#   accuracy JSON   : out_ifprune_<model>_<dataset>_b<budget>.json
#   circular summary: out_ifprune_<model>_<dataset>_circ_b<budget>.json
#   metrics CSV     : metrics_ifprune_<model>_<dataset>.csv（bench_metrics.py 一次
#                     跑 uncompressed / stream_b<K>(SVMA) / ifprune_b<K>，同一批圖
#                     並排比較 peak memory / KV cache MB）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}                   # hrbench | docvqa | infovqa，跟 run_internvl.sh 保持一致才能對照
read -r -a BUDGETS <<< "${BUDGETS:-1024}"
NUM_IMAGES=${NUM_IMAGES:--1}                 # accuracy 用（投稿前拉全量，跟 run_internvl.sh 的 NUM_IMAGES 對齊）
MAX_NUM=${MAX_NUM:-48}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-500}
METRIC_IMAGES=${METRIC_IMAGES:-24}           # peak/KV cache 用，小 n 即可，跟 run_sparsevlm.sh 一致
PRUNE_LAYER=${PRUNE_LAYER:-2}                # 剪枝層 index（0-indexed）；論文主表 L=0 與 L=2 都有報，Table 3 用 2
HEAD_DTYPE=${HEAD_DTYPE:-bf16}               # bf16 = 官方數值；fp32 = 偏離官方的 ablation（見 ifprune_internvl.py 檔頭第 4 點）
SMALL_CKPT=${SMALL_CKPT:-ZachSun/internvl2_5_1b_pruning_kl_lora32}
SMALL_BASE=${SMALL_BASE:-OpenGVLab/InternVL2_5-1B}

DS_ARGS="--use_ds --dataset $DATASET"
MTAG=$(basename "$MODEL")

# 跟 run_internvl.sh 用同一招從 internvl_stream.py 讀 HR-Bench/DocVQA 的
# split/prompt 常數——ifprune_internvl.py 也是 import 同一份常數，語意一致。
PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo letter)
SPLIT_TAG=${HRBENCH_SPLIT:-hrbench_4k}   # 跟 internvl_stream.py 同一個環境變數
SPLIT_TAG=${SPLIT_TAG#hrbench_}
DOCVQA_PROMPT_MODE=$(grep -oP 'DOCVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
INFOVQA_PROMPT_MODE=$(grep -oP 'INFOVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
# docvqa / infovqa 共用的 prompt 模式，後面的 OUTBASE / eval / similarity 都吃這個。
if [ "$DATASET" = infovqa ]; then QA_PROMPT_MODE=$INFOVQA_PROMPT_MODE; else QA_PROMPT_MODE=$DOCVQA_PROMPT_MODE; fi

# 輸出放 results/baselines/internvl_ifprune_<dataset>/（HR-Bench 8K 是 hrbench8k），OUTDIR 可覆寫
if [ "$DATASET" = hrbench ] && [ "${SPLIT_TAG#hrbench_}" = 8k ]; then DSDIR=hrbench8k; else DSDIR=$DATASET; fi
OUTDIR=${OUTDIR:-results/baselines/internvl_ifprune_${DSDIR}}
mkdir -p "$OUTDIR"

if [ "$DATASET" = hrbench ]; then
    OUTBASE="$OUTDIR/out_ifprune_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    OUTBASE="$OUTDIR/out_ifprune_${QA_PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
else
    OUTBASE="$OUTDIR/out_ifprune_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi

echo "== IF-Prune :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  prune_layer=$PRUNE_LAYER  head=$HEAD_DTYPE  hrbench_split=$SPLIT_TAG  hrbench_prompt=$PROMPT_MODE  docvqa_prompt=$DOCVQA_PROMPT_MODE  infovqa_prompt=$INFOVQA_PROMPT_MODE =="

for BUDGET in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${BUDGET}.json"
    echo
    echo "==== $MTAG $DATASET b$BUDGET :: accuracy run (n=$NUM_IMAGES, prune_layer=$PRUNE_LAYER) ===="
    $PY ifprune_internvl.py --model_name "$MODEL" $DS_ARGS --num_images "$NUM_IMAGES" \
        --small_checkpoint "$SMALL_CKPT" --small_base "$SMALL_BASE" --head_dtype "$HEAD_DTYPE" \
        --max_num "$MAX_NUM" --budget "$BUDGET" --prune_layer "$PRUNE_LAYER" --max_new_tokens "$MAX_NEW_TOKENS" \
        --run_standard --run_ifprune --save --output_json "$J"

    # candidates 只有一個 key（這個 budget 的 tag），直接讀出來當 --baseline_tag，
    # 避免自己組字串跟 ifprune_internvl.py 的 f"ifprune_budget={budget}_L={prune_layer}" 對不上。
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
#      (SVMA) 同一批圖並排比較）。IF-Prune 沒有 λ 要校準，直接 --run_ifprune。
#      小 VLM 只在 IF-Prune 區段上 GPU：ifprune_b<K> 的 peak_alloc 含小模型權重，
#      kv_cache_mb 只算大模型（小模型的 KV 是暫存、不留存）。----
echo; echo "==== $MTAG $DATASET :: metrics (peak memory / KV cache, SVMA vs IF-Prune) ===="
$PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
    --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" \
    --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
    --run_ifprune --ifprune_prune_layer "$PRUNE_LAYER" --ifprune_head_dtype "$HEAD_DTYPE" \
    --ifprune_small_checkpoint "$SMALL_CKPT" --ifprune_small_base "$SMALL_BASE" \
    --out_csv "$OUTDIR/metrics_ifprune_${METRIC_TAG}_${DATASET}.csv" || true

echo
echo "===== IF-Prune done  ($MTAG / $DATASET) ====="
echo "prune_layer=$PRUNE_LAYER  head_dtype=$HEAD_DTYPE for all budgets"
if [ "$DATASET" = hrbench ]; then
    echo "circular summary : ${OUTBASE}_circ_b*.json"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    echo "ANLS summary     : ${OUTBASE}_b*.summary.json  (per-sample: ${OUTBASE}_anls_b*.csv)"
fi
echo "metrics          : $OUTDIR/metrics_ifprune_${METRIC_TAG}_${DATASET}.csv"
echo "                    (uncompressed / stream_b<K>(SVMA) / ifprune_b<K> 同一批圖並排)"
