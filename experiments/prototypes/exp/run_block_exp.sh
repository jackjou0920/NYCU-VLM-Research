#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Block A：accuracy vs budget
#
# 每個 budget 跑以下幾格做「accuracy / peak memory / TTFT vs budget」的跨方法比較，
# 全部落在同一份 per-budget JSON（fixed_tiles 另存再 merge 回來）：
#
#   0  Uncompressed          --run_standard                       壓縮的對象；accuracy 上界、peak/TTFT 的「省多少」基準
#   1  Random tiles          select=topk, score=random, delay0    dumb-selection 對照：scorer 有沒有贏過亂選
#   2  Ours                  select=topk, score=info_density, delay0   ← headline。info_density 已內建 Tier-0 清理：
#                                                                   * novelty（多樣性）項永遠參與
#                                                                   * pool 每個 tile 到貨都用最新全域統計量重算
#                                                                   * relevance 只 pool「問句本身」（strip 掉選項區塊與作答指示）
#                                                                   * signal + 全域 norm 統計量先 winsorize 掉 tile 內 norm 最高 SIGNAL_TRIM(=2%) 的 token
#                                                                   * flush 的 top-K 保留 COVERAGE_FLOOR(=1) 個 farthest-point 名額
#                                                                   * tile_agg 預設 max（per-token 分數取 max 聚合成 tile 分數，不稀釋答案 token）
#                                                                  全部沒有旗標，見 internvl_stream_v2.py 頂端常數 / argparse 預設。
#   2ta Ours + thumb-attn        select=topk, score=info_density, delay0, --thumb_attn   ← 本輪 headline，重點看 b1024。
#                                                                  thumbnail 跑一次 LLM forward → 取「答案位置對 256 個
#                                                                  thumbnail patch 的 attention」→ 16x16 粗 saliency
#                                                                  → 每個 grid tile 一個 prior，跟 info_density 分數
#                                                                  線性混合：final = (1-w)*info_density + w*prior，
#                                                                  w = THUMB_ATTN_W（見檔案頂端常數，--thumb_attn_w 可覆寫）。
#                                                                  peak memory 不動（thumbnail 本來就是一塊 tile）；
#                                                                  TTFT 多一個「與 tile 數無關」的短 prefill。
#                                                                  診斷:Tier-0 在 b1024 完全沒吃到 oracle 的 35 點
#                                                                  headroom（l2_norm ≈ info_density ≈ random ≈ 50），
#                                                                  thumb-attn 用模型自己的 attention 定位，目標把 b1024
#                                                                  從 ~50 拉向 oracle 的 85。
#   3  Ours + spatial_bin    select=spatial_bin, score=info_density, delay0   hard 覆蓋率（每個空間 bin 取 1 塊）。
#                                                                  跟 2 對照：比 topk 內建的 COVERAGE_FLOOR=1 再多拿分嗎？
#   4  Answer-aware oracle   select=oracle（偷看 gold answer，診斷用天花板，不可部署；hrbench/mmmu 才有 answer）
#   8  Fixed-few-tiles       --max_num=K 直接粗切、不經壓縮        均勻網格對照上界；Ours 的目標就是在相同 token 預算下贏過它。
#
#   ablation / 掃參（預設註解掉，要隔離單一因素或調參再 uncomment）：
#     - thumb-attn 混合權重掃描 --thumb_attn_w 0.4 / 0.8 / 0.9 / 1.0（tag _thumbattn_w<x>；w=1.0 = 純 prior）
#     - thumb-attn 疊 spatial_bin
#     - l2_norm only / tile_agg=topk_mean（都跑過了：K=4 bundle 沒用 K>=8 才有用；max > topk_mean）
#
# 為什麼只剩這些：Tier-0 把 rescore / novelty 併進 info_density 預設；delay-window 先前
# 實驗證實 D=0 ≈ D=2 ≈ D=48（無貢獻），故所有 streaming cell 固定 --delay_tiles 0。
#
# 每格的 peak memory 看該次 run 印的 [Peak Memory Breakdown] / [internvl_online_kv_memory_bank]，
# TTFT 看 run_online_kv 回傳的 timing["TTFT"]。每張圖 stream log 也會印 selected_tile_indices
# 跟 score spread（std 相對 mean 太小 = scorer 沒在分辨 tile 好壞）。
#
# 已完成的 cell 會被 internvl_stream_v2.py 的 resume 機制自動跳過，可安全重跑。
# ⚠ Tier-0 之後 info_density 的 tag 語意變了；跟舊 JSON 比較請用新的 out_*.json，或先清掉舊 candidates。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}                        # 需要時： PY=/home/jackjou/.virtualenvs/vlm/bin/python ./run_block_a.sh
DATASET=${DATASET:-hrbench}              # docvqa | mmmu | hrbench
NUM_IMAGES=${NUM_IMAGES:-200}           # 先用子集跑通；投稿前拉到全量（docvqa=5349 / mmmu=900 / hrbench_4k=800 / hrbench_8k=800）
MAX_NUM=${MAX_NUM:-48}                  # dynamic tiles 上限（含縮圖）
NUM_IMAGE_TOKEN=256                     # InternVL3.5-8B 每 tile 的 token 數
BUDGETS=(1024 2048 4096)
BATCH=${BATCH:-4}
STD_BATCH=${STD_BATCH:-2}               # 未壓縮 baseline 較吃顯存，保守一點
CHUNK=1024
SCORE=info_density

# HR-Bench 的 split / prompt 模式（hrbench_4k、letter）直接吃 load_hrbench 與
# internvl_stream_v2.py 的預設值，這裡不再另外傳旗標。

for BUDGET in "${BUDGETS[@]}"; do
    OUT="out_block_a_${DATASET}_b${BUDGET}.json"
    CSV="eval_block_a_${DATASET}_b${BUDGET}.csv"
    SUMMARY="summary_block_a_${DATASET}_b${BUDGET}.json"
    K=$(( BUDGET / NUM_IMAGE_TOKEN ))
    OURS_TAG="budget=${BUDGET}_${SCORE}_delay0"   # eval 的 --baseline_tag，要跟 cell 2 的 tag 完全一致

    COMMON="--use_ds --dataset $DATASET --num_images $NUM_IMAGES --max_num $MAX_NUM \
            --budget $BUDGET --chunk_size $CHUNK --save --output_json $OUT"

    run() { echo; echo "======== budget=$BUDGET :: $* ========"; $PY internvl_stream_v2.py $COMMON "$@"; }

    # # 0) Uncompressed 上界（full context，不壓縮）
    # run --batch_size "$STD_BATCH" --run_standard

    # # 1) Random tiles（dumb-selection 對照）
    # run --batch_size "$BATCH" --select topk --score_fn random --run_stream

    # # 2) Ours (Tier-0)（info_density，topk，delay0）  tag = budget=${BUDGET}_info_density_delay0
    # run --batch_size "$BATCH" --select topk --score_fn "$SCORE" --run_stream

    # 2ta) Ours + thumb-attn（thumbnail 一次 forward 取 attention → 16x16 粗 saliency → 每個 grid tile 一個
    #      prior，跟 info_density 線性混合 w=THUMB_ATTN_W）。 ← 本輪 headline，重點看 b1024。
    #      peak memory 不動；TTFT 多一個「與 tile 數無關」的短 prefill。tag = ..._delay0_thumbattn。
    # run --batch_size "$BATCH" --select topk --score_fn "$SCORE" --thumb_attn --run_stream

    # # 3) Ours + spatial_bin（hard 覆蓋率：每個空間 bin 取 1 塊；跟 2 對照）
    # run --batch_size "$BATCH" --select spatial_bin --score_fn "$SCORE" --run_stream

    # # 4) Answer-aware oracle（偷看 gold answer 的診斷天花板，不可部署）
    # run --batch_size 1 --select oracle --score_fn "$SCORE" --run_stream

    # --- ablation / 掃參：預設註解掉，要隔離單一因素或調參再 uncomment ---
    #   thumb-attn 混合權重掃描（tag = ..._thumbattn_w<x>）。w=1.0 = 純 prior，完全丟掉 info_density
    #   → 若 w=1.0 不輸 w=0.8，代表 K=4 的 info_density 是死重，可以簡化。
    # run --batch_size "$BATCH" --select topk --score_fn "$SCORE" --thumb_attn --thumb_attn_w 0.4 --run_stream
    # run --batch_size "$BATCH" --select topk --score_fn "$SCORE" --thumb_attn --thumb_attn_w 0.8 --run_stream
    run --batch_size "$BATCH" --select topk --score_fn "$SCORE" --thumb_attn --thumb_attn_w 0.9 --run_stream
    run --batch_size "$BATCH" --select topk --score_fn "$SCORE" --thumb_attn --thumb_attn_w 1.0 --run_stream
    # #   thumb-attn 疊 spatial_bin（prior 也會混進每個 bin 的分數）
    # run --batch_size "$BATCH" --select spatial_bin --score_fn "$SCORE" --thumb_attn --run_stream
    
    # #   l2_norm only：拿掉 relevance 項（已跑過，結論：K=4 這 bundle 完全沒用、K>=8 才有用）
    # run --batch_size "$BATCH" --select topk --score_fn l2_norm --run_stream
    # #   tile_agg=topk_mean（已跑過，結論：max 比較好）
    # run --batch_size "$BATCH" --select topk --score_fn "$SCORE" --tile_agg topk_mean --run_stream

    # # 8) Fixed-few-tiles baseline：獨立跑進另一份 JSON，再併回主 JSON 的 candidates
    # #    K 不能超過 MAX_NUM：K > MAX_NUM 時（例如診斷用的 budget=20000）代表 selection
    # #    早就不會丟任何 tile 了，這條 baseline 沒有意義，也沒必要為了它去搜尋比
    # #    reference 更細的網格（多吃顯存），封頂在 MAX_NUM 即可。
    # K_FIXED=$K
    # if [ "$K_FIXED" -gt "$MAX_NUM" ]; then K_FIXED=$MAX_NUM; fi
    # FIXED_OUT="out_block_a_${DATASET}_fixed_b${BUDGET}.json"
    # echo; echo "======== budget=$BUDGET :: fixed_tiles=$K_FIXED (max_num=$K_FIXED, uncompressed) ========"
    # $PY internvl_stream_v2.py --use_ds --dataset "$DATASET" --num_images "$NUM_IMAGES" \
    #     --max_num "$K_FIXED" --batch_size "$STD_BATCH" --run_standard \
    #     --save --output_json "$FIXED_OUT"
    # $PY merge_baseline.py --main_json "$OUT" --baseline_json "$FIXED_OUT" \
    #     --tag "fixed_tiles=${K_FIXED}"

    # ---- 評測：各 dataset 用各自的評分腳本，統一輸出 per-sample CSV + summary JSON ----
    echo; echo "======== budget=$BUDGET :: eval ========"
    case "$DATASET" in
        docvqa)
            $PY evaluate_anls.py --input_json "$OUT" --use_hf_gt \
                --baseline_tag "$OURS_TAG" --dump_csv "$CSV" --summary_json "$SUMMARY" || true ;;
        mmmu)
            $PY evaluate_mmmu.py --input_json "$OUT" \
                --baseline_tag "$OURS_TAG" --dump_csv "$CSV" --summary_json "$SUMMARY" || true ;;
        hrbench)
            $PY evaluate_hrbench.py --input_json "$OUT" \
                --baseline_tag "$OURS_TAG" --dump_csv "$CSV" --summary_json "$SUMMARY" || true ;;
    esac
done

echo
echo "===== Block A done ====="
echo "各 budget 的 summary：summary_block_a_${DATASET}_b*.json"
