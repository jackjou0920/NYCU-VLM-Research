#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 多尺度 SVMA：掃覆蓋 tile 數 m（--coverage_tiles）
#
# v1（2026-10-11）相對 v0 的差別：
#   - S 跟細節名額等比例縮小：S(K, m) = round(S(K) * (K - m) / (K - 1))（svma_config.default_qattn_tiles），
#     v0 是所有 m 都用 S(K)，m 大時對很多塊打分卻只剩 1~2 個名額，TTFT 浪費
#   - 覆蓋網格拉伸上限 1.5 → 1.75（internvl_preprocess.COVER_MAX_STRETCH），名目 m 跟實際塊數比較一致
#   - JSON 多存 tile_stats（實際網格、每塊細節 tile 的 prior），給「自動決定 m」的離線分析
#   輸出前綴 out_ms1_（v0 是 out_ms_，OUT_PREFIX 可改），m=1 跟 v0 / 原本的 SVMA 逐字相同
#
# budget K 塊 tile = m 塊低解析度全圖覆蓋（fill_grid，一定寫進 KV）+ K - m 塊原解析度細節 tile
# （prior 粗排 + 邊界打分選）。兩個端點是已經有結果的方法，可以當正確性檢查：
#   m = 1 ：覆蓋 = 官方縮圖，跟原本的 SVMA 逐字相同（tag 也相同，沒有 _m 後綴）
#   m = K ：只有覆蓋 ≈ Low-Res-Fill（差別只在 streaming chunked prefill vs batch_chat）
#
# 每個 budget 掃 m ∈ {1, K/4, K/2, 3K/4, K}（去重、>= 1）：b1024 → 1 2 3 4；b2048 → 1 2 4 6 8。
# 同一個 budget 的所有 m 寫進同一個 JSON（tag 不同），最後一次評分；baseline_tag = m=1（原本的 SVMA），
# 所以 Δ 欄就是「相對原本 SVMA」的差。
#
# 用法：
#   DATASET=docvqa NUM_IMAGES=500 ./run_multiscale.sh
#   DATASET=hrbench ./run_multiscale.sh
#   M_LIST="1 2" BUDGETS="1024" DATASET=infovqa NUM_IMAGES=50 ./run_multiscale.sh   # 指定 m（正確性檢查用）
#   METRICS=1 ...                                                                  # 另外量 TTFT / peak memory
#
# 產出檔名：
#   accuracy JSON   : ${OUT_PREFIX}_<...>_<model>_<dataset>_b<budget>.json（所有 m 的 tag 在同一檔）
#   ANLS summary    : ${OUT_PREFIX}_<...>_b<budget>.summary.json；HR-Bench：${OUT_PREFIX}_<...>_circ_b<budget>.json
#   metrics CSV     : metrics_${OUT_PREFIX#out_}_<model>_<dataset>_m<m>.csv（METRICS=1 才跑）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}                   # hrbench | docvqa | infovqa
read -r -a BUDGETS <<< "${BUDGETS:-1024 2048}"
NUM_IMAGES=${NUM_IMAGES:--1}
MAX_NUM=${MAX_NUM:-48}
BATCH=${BATCH:-4}
M_LIST=${M_LIST:-}                            # 空 = 依 budget 自動 {1, K/4, K/2, 3K/4, K}；可寫整數或 K 的比例，例 "1 3K/4"
DETAIL_SCALE=${DETAIL_SCALE:-0}               # 細節網格：0 = 官方網格（上限 MAX_NUM）；> 0 = 原生解析度網格（tag 後綴 _ds<s>）
METRICS=${METRICS:-0}
METRIC_IMAGES=${METRIC_IMAGES:-24}
OUT_PREFIX=${OUT_PREFIX:-out_ms1}
NUM_IMAGE_TOKEN=256

MTAG=$(basename "$MODEL")
QA_SUF=$(grep -oP '^QATTN_LAYER = \K\d+' internvl_stream.py)ctx

PROMPT_MODE=$(grep -oP 'HRBENCH_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo letter)
SPLIT_TAG=${HRBENCH_SPLIT:-hrbench_4k}   # 跟 internvl_stream.py 同一個環境變數
SPLIT_TAG=${SPLIT_TAG#hrbench_}
DOCVQA_PROMPT_MODE=$(grep -oP 'DOCVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
INFOVQA_PROMPT_MODE=$(grep -oP 'INFOVQA_PROMPT\s*=\s*"\K[^"]+' internvl_stream.py || echo short)
if [ "$DATASET" = infovqa ]; then QA_PROMPT_MODE=$INFOVQA_PROMPT_MODE; else QA_PROMPT_MODE=$DOCVQA_PROMPT_MODE; fi

if [ "$DATASET" = hrbench ]; then
    OUTBASE="${OUT_PREFIX}_${SPLIT_TAG}_${PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="${SPLIT_TAG}_${MTAG}"
else
    OUTBASE="${OUT_PREFIX}_${QA_PROMPT_MODE}_${MTAG}_${DATASET}"
    METRIC_TAG="$MTAG"
fi
if { [ "$DATASET" = hrbench ] && [ "$PROMPT_MODE" = open ]; } || \
   { [ "$DATASET" != hrbench ] && [ "$QA_PROMPT_MODE" = open ]; }; then
    echo "[error] open prompt 模式沒有自動評分，這支腳本只支援 letter / short" >&2; exit 1
fi

m_list_for() {   # $1 = budget → 這個 budget 要掃的 m（M_LIST 的 "aK/b" 依 K 換算成整數）
    local K=$(( $1 / NUM_IMAGE_TOKEN )) tok num den out=()
    if [ -z "$M_LIST" ]; then
        out=(1 $((K / 4)) $((K / 2)) $((K * 3 / 4)) "$K")
    else
        for tok in $M_LIST; do
            if [[ "$tok" == *K* ]]; then
                num=${tok%%K*}; num=${num:-1}
                den=${tok#*K}; den=${den#/}; den=${den:-1}
                out+=($(( K * num / den )))
            else
                out+=("$tok")
            fi
        done
    fi
    printf "%s\n" "${out[@]}" | awk '$1 >= 1' | sort -n -u | tr '\n' ' '
}
s_for() { $PY -c "from svma_config import default_qattn_tiles as f; print(f($1, $2))"; }   # $1 = budget, $2 = m
tag_for() {      # $1 = budget, $2 = S, $3 = m（跟 internvl_stream.py main() 的 tag 一致）
    local t
    if [ "$2" -gt 0 ]; then t="budget=$1_svma_S$2_qa$QA_SUF"; else t="budget=$1_svma_S0"; fi
    if [ "$3" -ne 1 ]; then t="${t}_m$3"; fi
    if [ "$DETAIL_SCALE" != 0 ]; then t="${t}_ds$($PY -c "print(f'{float($DETAIL_SCALE):g}')")"; fi
    echo "$t"
}

echo "== multiscale SVMA ($OUT_PREFIX) :: model=$MTAG  dataset=$DATASET  budgets=${BUDGETS[*]}  n=$NUM_IMAGES  detail_scale=$DETAIL_SCALE =="

for BUDGET in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${BUDGET}.json"
    MS=$(m_list_for "$BUDGET")
    echo; echo "==== $MTAG $DATASET b$BUDGET :: m ∈ { $MS} ===="
    for M in $MS; do
        echo; echo "---- b$BUDGET m=$M S=$(s_for "$BUDGET" "$M") ----"
        $PY internvl_stream.py --model_name "$MODEL" --dataset "$DATASET" --num_images "$NUM_IMAGES" \
            --max_num "$MAX_NUM" --budget "$BUDGET" --batch_size "$BATCH" \
            --coverage_tiles "$M" --detail_scale "$DETAIL_SCALE" --run_stream --save --output_json "$J"
    done

    # baseline = 原本的 SVMA（m=1、官方細節網格，tag 沒有 _m / _ds）；同一個 JSON 裡沒有就改用這次的 m=1
    BASE_TAG=$(DETAIL_SCALE=0 tag_for "$BUDGET" "$(s_for "$BUDGET" 1)" 1)
    if ! $PY -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1]))['candidates'] else 1)" "$J" "$BASE_TAG"; then
        BASE_TAG=$(tag_for "$BUDGET" "$(s_for "$BUDGET" 1)" 1)
    fi
    echo; echo "==== $MTAG $DATASET b$BUDGET :: eval (baseline = m=1 = $BASE_TAG) ===="
    if [ "$DATASET" = hrbench ]; then
        $PY evaluate_hrbench_circular.py --input_json "$J" --baseline_tag "$BASE_TAG" --group_by id \
            --dump_csv "${OUTBASE}_circ_b${BUDGET}.csv" \
            --summary_json "${OUTBASE}_circ_b${BUDGET}.json" || true
    else
        $PY evaluate_anls.py --input_json "$J" --baseline_tag "$BASE_TAG" \
            --dump_csv "${OUTBASE}_anls_b${BUDGET}.csv" \
            --summary_json "${OUTBASE}_b${BUDGET}.summary.json" || true
    fi
done

# ---- 主張指標：TTFT / peak memory（每個 m 一次，涵蓋所有 budget；m > K 的 budget 自動跳過）----
if [ "$METRICS" = 1 ]; then
    ALL_M=$(for B in "${BUDGETS[@]}"; do m_list_for "$B"; done | tr ' ' '\n' | awk 'NF' | sort -n -u | tr '\n' ' ')
    for M in $ALL_M; do
        echo; echo "==== $MTAG $DATASET :: metrics m=$M ===="
        $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
            --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" \
            --budgets "$(IFS=,; echo "${BUDGETS[*]}")" --coverage_tiles "$M" --detail_scale "$DETAIL_SCALE" \
            --skip_uncompressed \
            --out_csv "metrics_${OUT_PREFIX#out_}_${METRIC_TAG}_${DATASET}_m${M}_ds${DETAIL_SCALE}.csv" || true
    done
fi

echo
echo "===== multiscale ($OUT_PREFIX) done  ($MTAG / $DATASET) ====="
if [ "$DATASET" = hrbench ]; then
    echo "circular summary : ${OUTBASE}_circ_b*.json"
else
    echo "ANLS summary     : ${OUTBASE}_b*.summary.json"
fi
[ "$METRICS" = 1 ] && echo "metrics          : metrics_${OUT_PREFIX#out_}_${METRIC_TAG}_${DATASET}_m*.csv"
