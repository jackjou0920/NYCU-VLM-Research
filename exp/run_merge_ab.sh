#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 實驗 1 + 3：DocVQA, budget=4096（有復原空間的操作點）, n 拉大
#
#   1  三方比較        evict / merge / merge_spatial   （皆 select=topk）
#   3  空間分箱選擇      merge_spatial 換成 select=spatial_bin，
#                        讓存活 tile 均勻鋪滿全圖 → 每個被淘汰 tile 才真的有近鄰可折
#      （另加 evict + spatial_bin，隔離「均勻鋪滿」這個選擇策略本身的效果）
#
# 五個 stream cell + 一次未壓縮 baseline，全部寫進同一個 JSON，最後一次 eval。
# 已完成的 cell 會被 internvl_svm.py 的 resume 機制自動跳過，可安全重跑。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}                 # 需要時： PY=/home/jackjou/.virtualenvs/internvl/bin/python ./run_merge_ab.sh
BUDGET=1024
NUM_IMAGES=10                  # 設成 "" 跑全量 5349（CI 最緊但最久）
BATCH=4                          # 五個 stream cell 都用這個，確保可比
STD_BATCH=2                      # 未壓縮 baseline 較吃顯存（~24GB@bs4），保守一點
CHUNK=1024
SCORE=info_density
AGG=mean

OUT=out_merge_ab_mmmu_${BUDGET}.json
CSV=anls_merge_ab_mmmu_${BUDGET}_per_sample.csv

NI_ARG=""
[ -n "$NUM_IMAGES" ] && NI_ARG="--num_images $NUM_IMAGES"

COMMON="--use_ds $NI_ARG --budget $BUDGET --chunk_size $CHUNK \
        --score_fn $SCORE --tile_agg $AGG --save --output_json $OUT"

run() { echo; echo "======== $* ========"; $PY internvl_svm.py $COMMON "$@"; }

# 1a  未壓縮 references（只跑 standard，小 batch）
run --batch_size $STD_BATCH --merge_mode evict        --select topk        --run_standard

# 1b  evict / topk           ← baseline cell
run --batch_size $BATCH      --merge_mode evict        --select topk        --run_stream

# 2   merge / topk           （全域 cosine 折合）
run --batch_size $BATCH      --merge_mode merge        --select topk        --merge_weight score --run_stream

# 3   merge_spatial / topk   （空間約束折合，存活 tile 仍按分數選）
run --batch_size $BATCH      --merge_mode merge_spatial --select topk       --merge_weight score --run_stream

# 4   evict / spatial_bin    （隔離：只換選擇策略，不折合）
run --batch_size $BATCH      --merge_mode evict        --select spatial_bin --run_stream

# 5   merge_spatial / spatial_bin  ← 實驗 3 主角：空間折合 + 均勻存活 tile
run --batch_size $BATCH      --merge_mode merge_spatial --select spatial_bin --merge_weight score --run_stream

# # ── 評測：五個 cell 一次比，baseline = evict/topk ──
# echo; echo "======== eval ========"
# $PY evaluate_anls.py --input_json "$OUT" --use_hf_gt \
#     --baseline_tag "budget=${BUDGET}_evict_${SCORE}" \
#     --dump_csv "$CSV"

# 產生的 tag：
#   budget=4096_evict_info_density                             (baseline)
#   budget=4096_merge_info_density_wscore
#   budget=4096_merge_spatial_info_density_wscore
#   budget=4096_evict_info_density_spatial_bin
#   budget=4096_merge_spatial_info_density_wscore_spatial_bin
#
# 看：ΔANLS vs baseline 的 95% CI（不含 0 才算數）、zero% / exact%、
#     以及每張圖 stream log 印的 mean_merge_sim（低=折合在補真東西，高=冗餘）。
