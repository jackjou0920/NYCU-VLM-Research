#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Low-Res（Resize-to-Budget）對照組：不做任何 token 壓縮，直接把 dynamic tile
# 上限降到剛好放得下 budget —— max_num = budget / 256（含縮圖），模型用較低解析度
# 看整張圖。回答「既然只有 budget 個 token，為什麼不直接少切幾塊？」這個質疑。
#
# 跟 run_internvl.sh / run_fastv.sh 同一組 budget，放進同一張表比較：
#   b1024 → 4 tiles   b2048 → 8   b3072 → 12   b4096 → 16   b5120 → 20
# visual token 數 ≤ budget，跟 SVMA 同一個上限。
#
# FILL=1：Low-Res（fill）變體。官方選法挑「比例最接近、塊數最少」的網格，常常用不滿 budget
#   （HR-Bench 4:3 在 b1024 只有 1 塊 = 256 token），FILL=1 改成網格 <= budget/256 塊、不加縮圖、
#   拉伸 <= 1.5 倍裡塊數最多的（b1024 → 2x2），輸出檔名 / tag 用 lowresfill 區分。
#
# 不用新 code：直接跑 internvl_stream.py --run_standard（官方 batch_chat 未壓縮路徑），
# 只換 --max_num。JSON 的 references 就是 Low-Res 的答案，跑完後複製一份成
# candidates[lowres_max<N>] 讓 evaluator 當一列計分
# （evaluator 只有 references 時會出錯）。--budget 對 --run_standard
# 沒作用，只是讓檔名 / 設定對得上。
#
# 產出檔名：
#   accuracy JSON   : out_lowres_<...>_<model>_<dataset>_b<budget>.json
#   circular summary: out_lowres_<...>_circ_b<budget>.json
#   metrics CSV     : metrics_lowres_<model>_<dataset>_b<budget>.csv
#                     （max_num 隨 budget 變，所以每個 budget 各跑一次 bench_metrics.py；
#                      看 uncompressed 那列 = Low-Res 的 TTFT / peak memory，
#                      stream_b<K> 那列在只有 budget/256 塊 tile 時沒有比較意義）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}                   # hrbench | docvqa | infovqa，跟 run_internvl.sh 保持一致才能對照
read -r -a BUDGETS <<< "${BUDGETS:-1024 2048 3072 4096 5120}"
NUM_IMAGES=${NUM_IMAGES:--1}                 # accuracy 用（投稿前拉全量，跟 run_internvl.sh 的 NUM_IMAGES 對齊）
STD_BATCH=${STD_BATCH:-2}
METRIC_IMAGES=${METRIC_IMAGES:-24}           # peak/TTFT 用，小 n 即可
NUM_IMAGE_TOKEN=256                          # InternVL 每塊 tile 的 token 數
FILL=${FILL:-0}                              # 1 = Low-Res（fill）：網格 <= budget/256 塊、不加縮圖、盡量用滿 budget
                                             #     （internvl_preprocess.fill_grid）；0 = 官方 dynamic_preprocess 選法
if [ "$FILL" = 1 ]; then NAME=lowresfill; FILL_ARGS="--lowres_fill"; else NAME=lowres; FILL_ARGS=""; fi

DS_ARGS="--dataset $DATASET"
MTAG=$(basename "$MODEL")

# 跟 run_internvl.sh 用同一招從 internvl_stream.py 讀 HR-Bench/DocVQA/InfoVQA 的 split/prompt 常數
PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo letter)
SPLIT_TAG=${HRBENCH_SPLIT:-hrbench_4k}   # 跟 internvl_stream.py 同一個環境變數
SPLIT_TAG=${SPLIT_TAG#hrbench_}
DOCVQA_PROMPT_MODE=$(grep -oP 'DOCVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
INFOVQA_PROMPT_MODE=$(grep -oP 'INFOVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
if [ "$DATASET" = infovqa ]; then QA_PROMPT_MODE=$INFOVQA_PROMPT_MODE; else QA_PROMPT_MODE=$DOCVQA_PROMPT_MODE; fi

if [ "$DATASET" = hrbench ]; then
    OUTBASE="out_${NAME}_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    OUTBASE="out_${NAME}_${QA_PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
else
    OUTBASE="out_${NAME}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi

echo "== Low-Res ($NAME) :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  hrbench_split=$SPLIT_TAG  hrbench_prompt=$PROMPT_MODE  docvqa_prompt=$DOCVQA_PROMPT_MODE  infovqa_prompt=$INFOVQA_PROMPT_MODE =="

for BUDGET in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${BUDGET}.json"
    MN=$((BUDGET / NUM_IMAGE_TOKEN))          # 總 tile 上限（含縮圖），internvl_stream.py 內部會再扣 1 給網格
    echo
    echo "==== $MTAG $DATASET b$BUDGET :: accuracy run (n=$NUM_IMAGES, max_num=$MN) ===="
    $PY internvl_stream.py --model_name "$MODEL" $DS_ARGS --num_images "$NUM_IMAGES" \
        --max_num "$MN" --budget "$BUDGET" --batch_size "$STD_BATCH" \
        --run_standard $FILL_ARGS --save --output_json "$J"

    # evaluator 需要至少一個 candidates tag（只有 references 時 pick_baseline_tag 會 IndexError），
    # 把 references 複製一份成 candidates[lowres_max<MN>]，Low-Res 就是有名字的一列。
    TAG="${NAME}_max${MN}"
    $PY -c "import json,sys; p,t=sys.argv[1:]; d=json.load(open(p)); d.setdefault('candidates',{})[t]=list(d['references']); json.dump(d,open(p,'w'),ensure_ascii=False,indent=2)" "$J" "$TAG"

    echo; echo "==== $MTAG $DATASET b$BUDGET :: eval (tag=$TAG) ===="
    case "$DATASET" in
        hrbench)
            if [ "$PROMPT_MODE" = open ]; then
                : # open：沒有 GT，見下面跨 budget 的語意相似度
            else
                $PY evaluate_hrbench_circular.py --input_json "$J" --baseline_tag "$TAG" --group_by id \
                    --dump_csv "${OUTBASE}_circ_b${BUDGET}.csv" \
                    --summary_json "${OUTBASE}_circ_b${BUDGET}.json" || true
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

    # ---- 主張指標：Low-Res 的 TTFT / peak memory = 這次 CSV 的 uncompressed 列 ----
    echo; echo "==== $MTAG $DATASET b$BUDGET :: metrics (max_num=$MN) ===="
    $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
        --num_images "$METRIC_IMAGES" --max_num "$MN" --budgets "$BUDGET" $FILL_ARGS \
        --out_csv "metrics_${NAME}_${METRIC_TAG}_${DATASET}_b${BUDGET}.csv" || true
done

# ---- open 模式：Low-Res 的 references 本身就是被比較的對象，沒有 full-res reference
#      可比；要比的話把 run_internvl.sh 的 uncompressed 輸出跟這裡的 JSON 合併再算。----
IS_OPEN_RUN=0
[ "$DATASET" = hrbench ] && [ "$PROMPT_MODE" = open ] && IS_OPEN_RUN=1
{ [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; } && [ "$QA_PROMPT_MODE" = open ] && IS_OPEN_RUN=1
if [ "$IS_OPEN_RUN" = 1 ]; then
    echo; echo "[warn] open prompt 模式：Low-Res 沒有自動評分，需要另外對 full-res reference 算語意相似度"
fi

echo
echo "===== Low-Res ($NAME) done  ($MTAG / $DATASET) ====="
if [ "$DATASET" = hrbench ]; then
    echo "circular summary : ${OUTBASE}_circ_b*.json   （看 ${NAME}_max<N> 那列）"
elif [ "$DATASET" = docvqa ] || [ "$DATASET" = infovqa ]; then
    echo "ANLS summary     : ${OUTBASE}_b*.summary.json  （看 ${NAME}_max<N> 那列）"
fi
echo "metrics          : metrics_${NAME}_${METRIC_TAG}_${DATASET}_b*.csv  （看 uncompressed 那列）"
