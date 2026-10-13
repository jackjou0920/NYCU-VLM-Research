#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SVMA V2 實驗：thumbnail prior 粗排 → 邊界打分（--qattn_tiles S）
#
# 兩種用途（同一支腳本）：
#   (1) 校準：在隨機抽樣的校準集上掃 (budget, S) 格點，依規則 A 選出每個 budget 的 S，
#       寫 svma_calibration.json（明細），並把選定的 S 寫進 svma_config.py 的 QATTN_TILES_BY_BUDGET
#       （auto 的預設表；其他 budget 依 K 內插）
#         SUBSET_SEED=0 NUM_IMAGES=600 bash run_svma_v2.sh
#   (2) 報告：用選定的 S 在測試集上跑
#         S_LIST=auto bash run_svma_v2.sh                                         # 全量
#         S_LIST=auto SUBSET_SEED=0 NUM_IMAGES=600 SUBSET_COMPLEMENT=1 bash run_svma_v2.sh  # 校準集以外的部分
#
# 規則 A（calibrate_qattn_tiles.py）：在 TTFT <= IF-Prune - MARGIN 的 S 裡選 ANLS 最高的。
#   IF-Prune 的 TTFT：RUN_IFPRUNE=1 時在同一個 session 實測；否則用 calibrate_qattn_tiles.REF_TTFT（線性內插）。
#
# S 格點：S_GRID（每個 budget 各自的 S，例 "1024:0,4,8,12,15 2048:0,4,8,12,14"）優先；
#   沒給就所有 budget 共用 S_LIST（"auto" = svma_config 的預設）。
# 沿用結果（reuse_results.py）：全量 json 裡已有的同一個 S，或等價的舊 cascade
#   （S//2 >= K-1 且 S 是 K-1 的倍數 → 舊 cascade m = S/(K-1)），依題目切片沿用、不重跑。
# resume：internvl_stream.py 會從每個 tag「已完成的筆數」接著跑，已跑滿的 tag 直接跳過。
#
# 其他用法：
#   SKIP_ACC=1 bash run_svma_v2.sh                  # 只量 TTFT / peak memory
#   SKIP_METRICS=1 bash run_svma_v2.sh              # 只跑 accuracy
#   SKIP_ACC=1 SKIP_METRICS=1 bash run_svma_v2.sh   # 只重印彙整表（校準模式會順便重選 S）
#   SKIP_CHECK=1 bash run_svma_v2.sh                # 跳過開頭的 q_attn 正確性檢查
#
# 產出：
#   accuracy JSON : 全量 out_v2_<model>_<dataset>_b<B>.json；
#                   抽樣 out_v2_<model>_<dataset>_s<seed>n<N>[c]_b<B>.json（c = complement）
#                   tag：budget=<B>_svma_S<S>[_qa<L>ctx]
#   metrics CSV   : metrics_v2_<model>_<dataset>_S<S>[_qa<L>ctx].csv；IF-Prune：..._ifprune.csv
#   校準檔        : svma_calibration.json + 更新 svma_config.py 的 QATTN_TILES_BY_BUDGET
#                   （只在校準模式、CALIBRATE=1 時寫；UPDATE_CONFIG=0 可只寫 json、不動 svma_config.py）
#   彙整表        : table_v2_<model>_<dataset>[_s<seed>n<N>[c]].csv
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}                   # infovqa | docvqa | hrbench
read -r -a BUDGETS <<< "${BUDGETS:-1024 1536 2048 2560 3072 4096 5120}"
read -r -a S_LIST <<< "${S_LIST:-0 4 8 12 16 24}"
S_GRID=${S_GRID:-}                            # 例 "1024:0,4,8,12,15 2048:0,4,8,12,14,21"；給了就覆蓋 S_LIST
NUM_IMAGES=${NUM_IMAGES:--1}                  # 全量 = -1；有 SUBSET_SEED 時 = 抽幾筆
SUBSET_SEED=${SUBSET_SEED:-}                  # 給了就隨機抽樣（固定 seed）
SUBSET_COMPLEMENT=${SUBSET_COMPLEMENT:-0}     # 1 = 取抽樣以外的部分（測試集）
METRIC_IMAGES=${METRIC_IMAGES:-24}            # TTFT / peak 用，小 n 即可
MAX_NUM=${MAX_NUM:-48}
BATCH=${BATCH:-1}                             # batch=1 時 json 裡的 ttft 才是逐張圖
RUN_IFPRUNE=${RUN_IFPRUNE:-0}                 # 1 = 同一個 session 實測 IF-Prune 的 TTFT 當規則 A 的目標
MARGIN=${MARGIN:-0.03}                        # TTFT 至少比 IF-Prune 快幾秒
CALIBRATE=${CALIBRATE:-}                      # 預設：校準模式（有 SUBSET_SEED 且非 complement）才寫校準檔
UPDATE_CONFIG=${UPDATE_CONFIG:-1}             # 校準後把選定的 S 寫進 svma_config.py（0 = 只寫 json）
SKIP_ACC=${SKIP_ACC:-0}
SKIP_METRICS=${SKIP_METRICS:-0}
SKIP_CHECK=${SKIP_CHECK:-0}
METRICS_FORCE=${METRICS_FORCE:-0}             # 1 = 已有的 metrics csv 也重量（預設：已涵蓋這些 budget 就略過）

# q_attn 設定（tag 後綴用）：環境變數 SVMA_QATTN_LAYER 優先，否則讀 internvl_stream.py 的預設
QA_SUF=${SVMA_QATTN_LAYER:-$(grep -oP '^QATTN_LAYER = int\(os.environ.get\("SVMA_QATTN_LAYER", "\K\d+' internvl_stream.py)}$(grep -qP '^QATTN_CTX = True' internvl_stream.py && echo ctx || true)
MTAG=$(basename "$MODEL")
FULLBASE="out_v2_${MTAG}_${DATASET}"
SUBARGS=()
SUFFIX=""
if [ -n "$SUBSET_SEED" ]; then
    [ "$NUM_IMAGES" -ge 1 ] || { echo "SUBSET_SEED 需要 NUM_IMAGES=N（要抽幾筆）"; exit 1; }
    SUBARGS=(--subset_seed "$SUBSET_SEED")
    SUFFIX="_s${SUBSET_SEED}n${NUM_IMAGES}"
    if [ "$SUBSET_COMPLEMENT" = 1 ]; then SUBARGS+=(--subset_complement); SUFFIX="${SUFFIX}c"; fi
fi
OUTBASE="${FULLBASE}${SUFFIX}"
[ -z "$CALIBRATE" ] && { [ -n "$SUBSET_SEED" ] && [ "$SUBSET_COMPLEMENT" != 1 ] && CALIBRATE=1 || CALIBRATE=0; }
DS_ARGS="--use_ds --dataset $DATASET"

# 這個 budget 實際要跑的 S（S_GRID 優先；auto 換成 svma_config 的預設；去重、由小到大）
s_values() {
    local B=$1 out=() S item list=""
    for item in $S_GRID; do [ "${item%%:*}" = "$B" ] && list=${item#*:}; done
    if [ -n "$list" ]; then IFS=, read -r -a out <<< "$list"; else out=("${S_LIST[@]}"); fi
    local res=()
    for S in "${out[@]}"; do
        [ "$S" = auto ] && S=$($PY -c "from svma_config import default_qattn_tiles as f; print(f($B))")
        res+=("$S")
    done
    printf '%s\n' "${res[@]}" | sort -n -u | tr '\n' ' '
}
mname_for() { if [ "$1" -gt 0 ]; then echo "S$1_qa$QA_SUF"; else echo "S0"; fi; }

echo "== SVMA V2 :: model=$MTAG dataset=$DATASET budgets=${BUDGETS[*]} qattn=$QA_SUF n=$NUM_IMAGES" \
     "subset=${SUBSET_SEED:-none}${SUFFIX:+ ($SUFFIX)} calibrate=$CALIBRATE =="
for B in "${BUDGETS[@]}"; do echo "   b$B: S = $(s_values "$B")"; done

# ---------------------------------------------------------------------------
# 正確性檢查：q_attn 重用前綴 KV 的分數要跟整段 forward 一致（5 張圖，幾分鐘），不一致就中止
# ---------------------------------------------------------------------------
NEED_QATTN=0
for B in "${BUDGETS[@]}"; do for S in $(s_values "$B"); do [ "$S" -gt 0 ] && NEED_QATTN=1; done; done
if [ "$SKIP_CHECK" != 1 ] && [ "$SKIP_ACC" != 1 ] && [ "$NEED_QATTN" = 1 ] && [[ "$QA_SUF" == *ctx ]]; then
    echo; echo "==== check: q_attn prefix-KV reuse ===="
    $PY check_qattn_prefix.py --model_name "$MODEL" --num_images 5 --max_num "$MAX_NUM" \
        || { echo "[abort] check_qattn_prefix.py 沒通過，正式實驗不跑"; exit 1; }
fi

# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
if [ "$SKIP_ACC" != 1 ]; then
for BUDGET in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${BUDGET}.json"
    COMMON="--model_name $MODEL $DS_ARGS --num_images $NUM_IMAGES --max_num $MAX_NUM \
            --budget $BUDGET --batch_size $BATCH --run_stream --save --output_json $J"
    run() { echo; echo "==== $MTAG $DATASET b$BUDGET :: $* ===="; $PY internvl_stream.py $COMMON "${SUBARGS[@]}" "$@"; }

    for S in $(s_values "$BUDGET"); do
        # 全量 json 裡已有同一個 S 或等價的舊 cascade → 依題目切片沿用（之後的 run 會直接跳過）
        RARGS=()
        [ -n "$SUBSET_SEED" ] && RARGS=(--subset_seed "$SUBSET_SEED" --subset_n "$NUM_IMAGES")
        [ "$SUBSET_COMPLEMENT" = 1 ] && RARGS+=(--subset_complement)
        $PY reuse_results.py --full_json "${FULLBASE}_b${BUDGET}.json" --target_json "$J" \
            --budget "$BUDGET" --S "$S" --qa "$QA_SUF" --dataset "$DATASET" "${RARGS[@]}" || true
        run --qattn_tiles "$S"
    done

    BASE_TAG="budget=${BUDGET}_svma_S0"
    echo; echo "==== $MTAG $DATASET b$BUDGET :: eval (baseline = $BASE_TAG) ===="
    case "$DATASET" in
        docvqa|infovqa)
            $PY evaluate_anls.py --input_json "$J" --baseline_tag "$BASE_TAG" \
                --dump_csv "${OUTBASE}_anls_b${BUDGET}.csv" \
                --summary_json "${OUTBASE}_b${BUDGET}.summary.json" || true ;;
        hrbench)
            $PY evaluate_hrbench_circular.py --input_json "$J" --baseline_tag "$BASE_TAG" \
                --group_by id \
                --dump_csv "${OUTBASE}_circ_b${BUDGET}.csv" \
                --summary_json "${OUTBASE}_circ_b${BUDGET}.json" || true ;;
    esac
done
fi

# ---------------------------------------------------------------------------
# TTFT / peak memory / KV（bench_metrics.py；每個 S 跑一次，涵蓋有這個 S 的所有 budget）
# TTFT 跟抽樣無關（bench 自己取前 METRIC_IMAGES 張），校準與報告共用同一組 csv。
# ---------------------------------------------------------------------------
MPREFIX="metrics_v2_${MTAG}_${DATASET}"
if [ "$SKIP_METRICS" != 1 ]; then
    ALL_S=$(for B in "${BUDGETS[@]}"; do s_values "$B"; done | tr ' ' '\n' | grep -v '^$' | sort -n -u)
    for S in $ALL_S; do
        # 注意 set -e + pipefail：迴圈最後一個 budget 沒有這個 S 時 grep 會回傳 1，要 || true，否則整支腳本無聲結束
        BL=$(for B in "${BUDGETS[@]}"; do if s_values "$B" | tr ' ' '\n' | grep -qx "$S"; then echo "$B"; fi; done | paste -sd,)
        CSV="${MPREFIX}_$(mname_for "$S").csv"
        if [ "$METRICS_FORCE" != 1 ] && [ -f "$CSV" ] && \
           $PY -c "import csv,sys; have={r['method'] for r in csv.DictReader(open('$CSV'))}; \
sys.exit(0 if all(f'stream_b{b}' in have for b in '$BL'.split(',')) else 1)"; then
            echo; echo "==== metrics [S=$S]：$CSV 已涵蓋 $BL，略過 ===="; continue
        fi
        echo; echo "==== $MTAG $DATASET :: metrics [S=$S] budgets=$BL ===="
        $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
            --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" --budgets "$BL" \
            --skip_uncompressed --thumb_attn --qattn_tiles "$S" \
            --out_csv "$CSV" || true
    done
    if [ "$RUN_IFPRUNE" = 1 ] && { [ "$METRICS_FORCE" = 1 ] || [ ! -f "${MPREFIX}_ifprune.csv" ]; }; then
        echo; echo "==== $MTAG $DATASET :: metrics [IF-Prune] budgets=${BUDGETS[*]} ===="
        $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
            --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
            --skip_uncompressed --run_ifprune --out_csv "${MPREFIX}_ifprune.csv" || true
    fi
fi

# ---------------------------------------------------------------------------
# 校準（規則 A）：選每個 budget 的 S，寫 svma_calibration.json
# ---------------------------------------------------------------------------
if [ "$CALIBRATE" = 1 ] && [ "$DATASET" != hrbench ]; then
    echo; echo "==== calibrate (rule A, margin=${MARGIN}s) ===="
    IFP_ARGS=(); [ -f "${MPREFIX}_ifprune.csv" ] && IFP_ARGS=(--ifprune_csv "${MPREFIX}_ifprune.csv")
    UPD_ARGS=(); [ "$UPDATE_CONFIG" = 1 ] && UPD_ARGS=(--update_config)
    $PY calibrate_qattn_tiles.py --outbase "$OUTBASE" --budgets "$(IFS=,; echo "${BUDGETS[*]}")" \
        --metrics_prefix "$MPREFIX" --qa "$QA_SUF" --margin "$MARGIN" "${IFP_ARGS[@]}" "${UPD_ARGS[@]}" || true
fi

# ---------------------------------------------------------------------------
# 彙整表（投影片格式）
# ---------------------------------------------------------------------------
SL=$(for B in "${BUDGETS[@]}"; do s_values "$B"; done | tr ' ' '\n' | grep -v '^$' | sort -n -u | paste -sd,)
$PY summarize_svma_v2.py --model_tag "$MTAG" --dataset "$DATASET" --outbase "$OUTBASE" \
    --budgets "$(IFS=,; echo "${BUDGETS[*]}")" --s_list "$SL" --qa "$QA_SUF" \
    --out_csv "table_v2_${MTAG}_${DATASET}${SUFFIX}.csv" || true

echo
echo "===== SVMA V2 done ($MTAG / $DATASET${SUFFIX:+ / $SUFFIX}) ====="
[ "$SKIP_ACC" != 1 ] && echo "accuracy : ${OUTBASE}_b*.json"
[ "$SKIP_METRICS" != 1 ] && echo "metrics  : ${MPREFIX}_S*.csv"
[ "$CALIBRATE" = 1 ] && echo "calib    : svma_calibration.json（S 已寫進 svma_config.py：python svma_config.py 可查看）"
echo "table    : table_v2_${MTAG}_${DATASET}${SUFFIX}.csv"
