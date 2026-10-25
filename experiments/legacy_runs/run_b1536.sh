#!/usr/bin/env bash
# 補預設操作點 b1536（docs/03_experiment_plan.md：所有測試集的 accuracy 與效率原本只有 b1024~b5120）。
#   1. accuracy：5 個測試集 × {SVMA, FastV, VisionZip, IF-Prune, SparseVLM} × b1536（run_test_suite.sh）
#   2. 效率：同樣的方法 × b1536（run_efficiency.sh），SparseVLM 用第 1 步各資料集校準出的 λ
# 等 run_scaling.sh 結束（logs/scaling/run.log 出現 done）才開始，避免干擾縮放曲線的 TTFT。
# 用法：nohup ./run_b1536.sh > logs/b1536.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
DATASETS="docvqa infovqa hrbench_4k mme_rw_lite vstar"

until grep -q "done" logs/scaling/run.log 2>/dev/null; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] 縮放曲線結束，開始 b1536 accuracy"
DATASETS="$DATASETS" METHODS="svma fastv visionzip ifprune sparsevlm" BUDGETS="1536" GPUS="0 1" \
    ./run_test_suite.sh > logs/test_suite_b1536.log 2>&1

echo "[$(date '+%m-%d %H:%M')] 開始 b1536 效率"
gpu=0
for ds in $DATASETS; do
    lam=$(grep -oP "  $ds sparsevlm b1536：校準 λ=\K[0-9.]+" logs/test_suite_b1536.log | tail -n 1)
    echo "  $ds：SparseVLM λ=${lam:-（沒找到，退回 auto_lam）}"
    DATASETS="$ds" GPUS="$gpu" BUDGETS=1536 LAMS="${lam:+1536:$lam}" OUT=results/efficiency/b1536 \
        LOG_DIR=logs/efficiency/b1536 ./run_efficiency.sh &
    gpu=$((1 - gpu))
    [ $gpu -eq 0 ] && wait          # 兩張卡各一個程序，跑完一對再下一對
done
wait
echo "[$(date '+%m-%d %H:%M')] done"
