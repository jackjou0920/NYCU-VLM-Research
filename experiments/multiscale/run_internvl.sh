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
#          + thumbnail-attention prior（永遠開，w = THUMB_ATTN_W = 0.9，層 L21-28:q）
#          + 邊界打分：prior 粗排後只對 S 塊邊界 tile 跑 q_attn（L20 + thumbnail context）
#            S = QT=auto（預設）依 budget 自動決定（svma_config.py：<=1024→15、<=3072→12、更大→0）
#   拿掉的：delay window（D=0≈D=2≈D=48，無效）、spatial_bin、tile_agg 掃描、
#           rescore/novelty 旗標（併進 info_density 預設）。這些都已改成 internvl_stream.py
#           的常數，不再是 CLI 旗標。
#
#   評測廣度：改 DATASET / HRBENCH_SPLIT 手動跑 hrbench_4k / hrbench_8k / docvqa / infovqa
#             （V*Bench / MME-RealWorld 需要新 loader，未含）
#   不同模型：改 MODEL 手動跑（pipeline 綁 InternVL tile 結構，換 family 要另外接）
#   主張指標：budget 迴圈跑完後，跑一次 bench_metrics.py —— 未壓縮 vs streaming 的
#             peak memory / TTFT / vision time，小 n（METRIC_IMAGES）即可，輸出 CSV
#
#   HR-Bench 用官方 CircularEval 計分（evaluate_hrbench_circular.py，組為單位做
#   bootstrap），另存 vanilla per-sample 供對照。docvqa / infovqa→evaluate_anls。
#
# 產出檔名：
#   accuracy JSON   : out_block_b_<model>_<dataset>_b<budget>.json
#   circular summary: out_block_b_<model>_<dataset>_circ_b<budget>.json
#   metrics CSV     : metrics_<model>_<dataset>.csv
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}     # 換模型改這裡
DATASET=${DATASET:-infovqa}                  # hrbench | docvqa | infovqa
read -r -a BUDGETS <<< "${BUDGETS:-1024 2048 3072 4096 5120}"
# NUM_IMAGES=${NUM_IMAGES:-1000}                # accuracy 用（投稿前拉全量）
NUM_IMAGES=${NUM_IMAGES:--1}
METRIC_IMAGES=${METRIC_IMAGES:-24}           # peak/TTFT 用，小 n 即可
MAX_NUM=${MAX_NUM:-48}
BATCH=${BATCH:-4}
STD_BATCH=${STD_BATCH:-2}
QT=${QT:-auto}                               # 邊界打分的 S（--qattn_tiles）；auto = 依 budget（svma_config.py）

DS_ARGS="--dataset $DATASET"
MTAG=$(basename "$MODEL")
# OURS_TAG 要跟 internvl_stream.py main() 產生的 tag 一致：budget=<B>_svma_S<S>[_qa<L>ctx]
# q_attn 固定第 QATTN_LAYER=20 層、含 thumbnail context（internvl_stream.py 的常數）
QA_SUF=$(grep -oP '^QATTN_LAYER = \K\d+' internvl_stream.py)ctx
s_for() { if [ "$QT" = auto ]; then $PY -c "from svma_config import default_qattn_tiles as f; print(f($1))"; else echo "$QT"; fi; }
tag_for() { if [ "$2" -gt 0 ]; then echo "budget=$1_svma_S$2_qa$QA_SUF"; else echo "budget=$1_svma_S0"; fi; }

# HR-Bench 的 split / prompt 模式（internvl_stream.py 裡的 HRBENCH_SPLIT /
# HRBENCH_PROMPT 常數）。split 決定 hrbench_4k | hrbench_8k；prompt 決定 accuracy
# 段怎麼評：letter -> 官方 CircularEval；open -> prompt 不帶選項，四選項輪換的
# CircularEval 不適用，改比「每個 budget 的自由生成輸出 vs uncompressed reference」
# 的語意相似度（見迴圈後的 similarity 段）。兩者都進 output naming。
PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo letter)
SPLIT_TAG=${HRBENCH_SPLIT:-hrbench_4k}   # 跟 internvl_stream.py 同一個環境變數
SPLIT_TAG=${SPLIT_TAG#hrbench_}              # hrbench_4k -> 4k

# DocVQA 的 prompt 模式（internvl_stream.py 裡的 DOCVQA_PROMPT 常數），跟 HR-Bench
# 的 letter/open 同一個精神：short -> 官方 ANLS 協定（evaluate_anls.py）；
# open -> 不加作答指示、自由生成，沒有 GT 可比，改用語意相似度（同一個 similarity 段）。
DOCVQA_PROMPT_MODE=$(grep -oP 'DOCVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)

# InfoVQA（validation split）同 DocVQA：INFOVQA_PROMPT 常數決定 short(ANLS) / open(語意相似度)。
INFOVQA_PROMPT_MODE=$(grep -oP 'INFOVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
# docvqa / infovqa 共用的 prompt 模式，後面的 OUTBASE / eval / similarity 都吃這個。
if [ "$DATASET" = infovqa ]; then QA_PROMPT_MODE=$INFOVQA_PROMPT_MODE; else QA_PROMPT_MODE=$DOCVQA_PROMPT_MODE; fi

if [ "$DATASET" = hrbench ]; then
    OUTBASE="out_block_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    OUTBASE="out_block_${QA_PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
else
    OUTBASE="out_block_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi

echo "== Block B :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  qattn_tiles=$QT  hrbench_split=$SPLIT_TAG  hrbench_prompt=$PROMPT_MODE  docvqa_prompt=$DOCVQA_PROMPT_MODE  infovqa_prompt=$INFOVQA_PROMPT_MODE =="

for BUDGET in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${BUDGET}.json"
    S_B=$(s_for "$BUDGET")
    OURS_TAG=$(tag_for "$BUDGET" "$S_B")
    COMMON="--model_name $MODEL $DS_ARGS --num_images $NUM_IMAGES --max_num $MAX_NUM \
            --budget $BUDGET --save --output_json $J"
    run() { echo; echo "==== $MTAG $DATASET b$BUDGET :: $* ===="; $PY internvl_stream.py $COMMON "$@"; }

    # 0) uncompressed 上界（references，accuracy 的比較基準）
    run --batch_size "$STD_BATCH" --run_standard

    # 1) Ours：SVMA（thumbnail prior 粗排 → 邊界打分 S 塊）
    run --batch_size "$BATCH" --qattn_tiles "$S_B" --run_stream

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
        docvqa|infovqa)
            if [ "$QA_PROMPT_MODE" = open ]; then
                # open：沒有 GT 可比對，逐 budget 的語意相似度在迴圈跑完後一次算
                # （見下方 similarity 段），這裡只跳過。
                :
            else
                # short：ANLS，GT 優先讀 meta.answers（load_docvqa / load_infovqa 存的，天生對齊），
                # 不用 --use_hf_gt 重載 HF dataset。
                $PY evaluate_anls.py --input_json "$J" --baseline_tag "$OURS_TAG" \
                    --dump_csv "${OUTBASE}_anls_b${BUDGET}.csv" \
                    --summary_json "${OUTBASE}_b${BUDGET}.summary.json" || true
            fi ;;
    esac
done

# ---- open 模式（HR-Bench / DocVQA / InfoVQA 的 open）：沒有 GT 可比對，改成跨 budget 的
#      語意相似度（每個 budget 輸出 vs uncompressed reference）----
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

# ---- 主張指標：peak memory / TTFT（跑一次涵蓋所有 budget）----
echo; echo "==== $MTAG $DATASET :: metrics (peak memory / TTFT) ===="
$PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
    --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" \
    --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
    --qattn_tiles "$QT" \
    --out_csv "metrics_${METRIC_TAG}_${DATASET}.csv" || true

echo
echo "===== Block B done  ($MTAG / $DATASET) ====="
if [ "$IS_OPEN_RUN" = 1 ]; then
    [ "$DATASET" = hrbench ] && echo "open matchRef    : ${OUTBASE}_open_b*.json"
    echo "similarity table : ${OUTBASE}_sim_summary.csv  (per-sample: ${OUTBASE}_sim_persample.csv)"
elif [ "$DATASET" = hrbench ]; then
    echo "circular summary : ${OUTBASE}_circ_b*.json"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    echo "ANLS summary     : ${OUTBASE}_b*.summary.json  (per-sample: ${OUTBASE}_anls_b*.csv)"
fi
echo "metrics          : metrics_${METRIC_TAG}_${DATASET}.csv"
