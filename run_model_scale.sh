#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 模型尺寸：InternVL3.5-4B / 14B × 6 個測試集 × {uncompressed, FoveaTile, FastV, VisionZip, IF-Prune, SparseVLM}
# × BUDGETS（預設 b1536）。FoveaTile 的 m、S 沿用 8B 的設定；attention 層號依層數換算（foveatile_internvl.attn_layers）。
# 結果：results/runs/<dataset>/<method>_b<B>_InternVL3_5-<size>.json（檔名含模型，不會跟 8B 撞）
# 用法：nohup ./run_model_scale.sh > logs/model_scale.log 2>&1 &
#       WAIT_PID=<pid> ./run_model_scale.sh      # 先等某個程序（例如還在跑的實驗）結束再開始
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
read -r -a SIZES <<< "${SIZES:-4B 14B}"
DATASETS=${DATASETS:-docvqa infovqa hrbench_4k hrbench_8k mme_rw_lite vstar}
BUDGETS=${BUDGETS:-1536}

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date '+%m-%d %H:%M')] 等 PID $WAIT_PID 結束"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
fi
for size in "${SIZES[@]}"; do
    # 14B 權重約 30 GB，每個工作開始前要求較多的空閒顯存
    min_free=30; [ "$size" = 14B ] && min_free=40
    echo "[$(date '+%m-%d %H:%M')] InternVL3.5-$size"
    MODEL="OpenGVLab/InternVL3_5-$size" MIN_FREE_GB=$min_free DATASETS="$DATASETS" BUDGETS="$BUDGETS" ./run_test_suite.sh
done
echo "[$(date '+%m-%d %H:%M')] all done"
