#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Block B：往「能投 paper」的方向補三個軸 —— 評測廣度 / 不同模型 / 主張指標
#
# 只 for-loop budget（跟 Block A 一樣）。要換模型 / 資料集就改下面的 MODEL / DATASET
# / HRBENCH_SPLIT 變數（或用 env 覆寫），手動一輪一輪跑。
#
# 方法已收斂（Block A 的結論）：
#   Ours = streaming（tile 逐個 encode → bounded pool，delay_tiles=0）
#          + info_density（Tier-0 清理全內建，無旗標）
#          + thumbnail-attention prior（--thumb_attn，w = THUMB_ATTN_W = 0.9）
#   拿掉的：delay window（D=0≈D=2≈D=48，無效）、spatial_bin、tile_agg 掃描、
#           rescore/novelty 旗標（併進 info_density 預設）。
#
#   評測廣度：改 DATASET / HRBENCH_SPLIT 手動跑 hrbench_4k / hrbench_8k / mmmu / docvqa
#             （V*Bench / MME-RealWorld 需要新 loader，未含）
#   不同模型：改 MODEL 手動跑（pipeline 綁 InternVL tile 結構，換 family 要另外接）
#   主張指標：budget 迴圈跑完後，跑一次 bench_metrics.py —— 未壓縮 vs streaming 的
#             peak memory / TTFT / vision time，小 n（METRIC_IMAGES）即可，輸出 CSV
#
#   HR-Bench 用官方 CircularEval 計分（evaluate_hrbench_circular.py，組為單位做
#   bootstrap），另存 vanilla per-sample 供對照。mmmu→evaluate_mmmu，docvqa→evaluate_anls。
#
# 產出檔名：
#   accuracy JSON   : out_block_b_<model>_<dataset>_b<budget>.json
#   circular summary: out_block_b_<model>_<dataset>_circ_b<budget>.json
#   metrics CSV     : metrics_<model>_<dataset>.csv
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}     # 換模型改這裡
DATASET=${DATASET:-hrbench}                  # hrbench | mmmu | docvqa
BUDGETS=(1024 2048 3072 4096 5120)
# NUM_IMAGES=${NUM_IMAGES:-100}                # accuracy 用（投稿前拉全量）
NUM_IMAGES=${NUM_IMAGES:--1}
METRIC_IMAGES=${METRIC_IMAGES:-24}           # peak/TTFT 用，小 n 即可
MAX_NUM=${MAX_NUM:-48}
BATCH=${BATCH:-4}
STD_BATCH=${STD_BATCH:-2}
CHUNK=1024
TAW=${TAW:-0.9}                              # thumb-attn 混合權重（收斂預設；OURS_TAG 會自動跟）

DS_ARGS="--use_ds --dataset $DATASET"
MTAG=$(basename "$MODEL")
WSUF=$($PY -c "print('' if abs($TAW-0.9)<1e-9 else '_w'+format($TAW,'g'))")

# HR-Bench 的 split / prompt 模式（internvl_stream_v2.py 裡的 HRBENCH_SPLIT /
# HRBENCH_PROMPT 常數）。split 決定 hrbench_4k | hrbench_8k；prompt 決定 accuracy
# 段怎麼評：letter -> 官方 CircularEval；open -> prompt 不帶選項，四選項輪換的
# CircularEval 不適用，改比「每個 budget 的自由生成輸出 vs uncompressed reference」
# 的語意相似度（見迴圈後的 similarity 段）。兩者都進 output naming。
PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream_v2.py || echo letter)
SPLIT_TAG=$(grep -oP 'HRBENCH_SPLIT\s*=\s*"\K[^"]+' internvl_stream_v2.py || echo hrbench_4k)
SPLIT_TAG=${SPLIT_TAG#hrbench_}              # hrbench_4k -> 4k

if [ "$DATASET" = hrbench ]; then
    OUTBASE="out_block_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
else
    OUTBASE="out_block_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi

echo "== Block B :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  taw=$TAW  hrbench_split=$SPLIT_TAG  hrbench_prompt=$PROMPT_MODE =="

for BUDGET in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${BUDGET}.json"
    OURS_TAG="budget=${BUDGET}_info_density_delay0_thumbattn${WSUF}"
    COMMON="--model_name $MODEL $DS_ARGS --num_images $NUM_IMAGES --max_num $MAX_NUM \
            --budget $BUDGET --chunk_size $CHUNK --save --output_json $J"
    run() { echo; echo "==== $MTAG $DATASET b$BUDGET :: $* ===="; $PY internvl_stream_v2.py $COMMON "$@"; }

    # 0) uncompressed 上界（references，accuracy 的比較基準）
    run --batch_size "$STD_BATCH" --run_standard

    # 1) Ours：收斂方法 = streaming + info_density + thumb-attn(w=TAW)
    run --batch_size "$BATCH" --select topk --score_fn info_density --delay_tiles 0 \
        --thumb_attn --thumb_attn_w "$TAW" --run_stream

    # --- ablation（預設關；要 Tier-0 對照 / random / oracle 再 uncomment）---
    # run --batch_size "$BATCH" --select topk --score_fn info_density --delay_tiles 0 --run_stream
    # run --batch_size "$BATCH" --select topk --score_fn random       --delay_tiles 0 --run_stream
    # [ "$DATASET" != docvqa ] && run --batch_size 1 --select oracle --score_fn info_density --run_stream

    # ---- accuracy 評測 ----
    echo; echo "==== $MTAG $DATASET b$BUDGET :: eval ===="
    case "$DATASET" in
        hrbench)
            if [ "$PROMPT_MODE" = open ]; then
                # open：只算 answer_text 命中 / 與 reference 是否一致（matchRef%）。
                # 逐 budget 的語意相似度在迴圈跑完後一次算（見下方 similarity 段）。
                $PY evaluate_hrbench.py --input_json "$J" --prompt_mode open \
                    --baseline_tag "$OURS_TAG" \
                    --summary_json "${OUTBASE}_open_b${BUDGET}.json" || true
            else
                $PY evaluate_hrbench_circular.py --input_json "$J" --baseline_tag "$OURS_TAG" \
                    --group_by id \
                    --dump_csv "${OUTBASE}_circ_b${BUDGET}.csv" \
                    --summary_json "${OUTBASE}_circ_b${BUDGET}.json" || true
                $PY evaluate_hrbench.py --input_json "$J" --baseline_tag "$OURS_TAG" \
                    --summary_json "${OUTBASE}_vanilla_b${BUDGET}.json" || true
            fi ;;
        mmmu)
            $PY evaluate_mmmu.py --input_json "$J" --baseline_tag "$OURS_TAG" \
                --summary_json "${OUTBASE}_b${BUDGET}.summary.json" || true ;;
        docvqa)
            $PY evaluate_anls.py --input_json "$J" --use_hf_gt --baseline_tag "$OURS_TAG" \
                --summary_json "${OUTBASE}_b${BUDGET}.summary.json" || true ;;
    esac
done

# ---- HR-Bench open：跨 budget 的語意相似度（每個 budget 輸出 vs uncompressed reference）----
if [ "$DATASET" = hrbench ] && [ "$PROMPT_MODE" = open ]; then
    echo; echo "==== $MTAG hrbench :: semantic similarity vs reference (all budgets) ===="
    SIM_JSONS=(); for B in "${BUDGETS[@]}"; do SIM_JSONS+=("${OUTBASE}_b${B}.json"); done
    MERGED="${OUTBASE}_sim_merged.json"
    $PY merge_sim_inputs.py --out "$MERGED" "${SIM_JSONS[@]}" \
        && $PY semantic_similarity_evaluator.py --input_json "$MERGED" \
               --output_csv "${OUTBASE}_sim_persample.csv" \
               --summary_csv "${OUTBASE}_sim_summary.csv" || true
fi

# ---- 主張指標：peak memory / TTFT（跑一次涵蓋所有 budget）----
echo; echo "==== $MTAG $DATASET :: metrics (peak memory / TTFT) ===="
$PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
    --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" \
    --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
    --thumb_attn --thumb_attn_w "$TAW" \
    --out_csv "metrics_${METRIC_TAG}_${DATASET}.csv" || true

echo
echo "===== Block B done  ($MTAG / $DATASET) ====="
if [ "$DATASET" = hrbench ] && [ "$PROMPT_MODE" = open ]; then
    echo "open matchRef    : ${OUTBASE}_open_b*.json"
    echo "similarity table : ${OUTBASE}_sim_summary.csv  (per-sample: ${OUTBASE}_sim_persample.csv)"
else
    echo "circular summary : ${OUTBASE}_circ_b*.json"
fi
echo "metrics          : metrics_${METRIC_TAG}_${DATASET}.csv"
