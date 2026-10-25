#!/usr/bin/env bash
# 補一個 budget 的完整測試結果（最早用來補預設點 b1536，之後補 b2560；方案 A：正文 budget 網格 = 1024~3072 每 512 一點）。
#   1. accuracy：5 個測試集 × {FoveaTile, FastV, VisionZip, IF-Prune, SparseVLM} × BUDGET（run_test_suite.sh）
#   2. 效率：同樣的方法 × BUDGET（run_efficiency.sh），SparseVLM 用第 1 步各資料集校準出的 λ
# WAIT_FOR=<log 檔> 時，等那個檔案出現 done 才開始（避免干擾其他實驗的 TTFT）。
# 用法：BUDGET=2560 nohup ./run_budget_fill.sh > logs/b2560.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
BUDGET=${BUDGET:-1536}
DATASETS=${DATASETS:-docvqa infovqa hrbench_4k mme_rw_lite vstar}
WAIT_FOR=${WAIT_FOR:-}
ACC_FREE_GB=${ACC_FREE_GB:-30}      # accuracy：每張卡至少要有這麼多空間才開始（跟別人共用 GPU 時）
EFF_FREE_GB=${EFF_FREE_GB:-44}      # 效率量測：每張卡要接近全空，TTFT 才乾淨（GPU0 有別人的常駐小服務約 2.5 GB）

MAX_WAIT_H=${MAX_WAIT_H:-24}        # 每次等 GPU 最多等幾小時，超過就放棄（不動別人的程序）

wait_gpus() {   # $1 = 每張卡最少空閒 GB；等到所有卡都滿足為止，超過 MAX_WAIT_H 小時就結束整支腳本
    local need=$1 start=$(date +%s)
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
              | awk -v n=$((need * 1024)) '$1 < n {bad=1} END {print bad ? 0 : 1}')" = 1 ]; do
        if [ $(( $(date +%s) - start )) -ge $(( MAX_WAIT_H * 3600 )) ]; then
            echo "[$(date '+%m-%d %H:%M')] TIMEOUT：等 GPU 超過 ${MAX_WAIT_H} 小時（每張卡要 >= $need GB），放棄，沒有啟動任何工作"
            nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
            exit 1
        fi
        sleep 120
    done
    echo "[$(date '+%m-%d %H:%M')] GPU 空閒（每張卡 >= $need GB）"
}

if [ -n "$WAIT_FOR" ]; then
    until grep -q "done" "$WAIT_FOR" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%m-%d %H:%M')] $WAIT_FOR 已結束"
fi
wait_gpus "$ACC_FREE_GB"
echo "[$(date '+%m-%d %H:%M')] 開始 b$BUDGET accuracy"
DATASETS="$DATASETS" METHODS="foveatile fastv visionzip ifprune sparsevlm" BUDGETS="$BUDGET" GPUS="0 1" \
    ./run_test_suite.sh > "logs/test_suite_b$BUDGET.log" 2>&1

wait_gpus "$EFF_FREE_GB"
echo "[$(date '+%m-%d %H:%M')] 開始 b$BUDGET 效率"
gpu=0
for ds in $DATASETS; do
    lam=$(grep -oP "  $ds sparsevlm b$BUDGET：校準 λ=\K[0-9.]+" "logs/test_suite_b$BUDGET.log" | tail -n 1)
    echo "  $ds：SparseVLM λ=${lam:-（沒找到，退回 auto_lam）}"
    DATASETS="$ds" GPUS="$gpu" BUDGETS="$BUDGET" LAMS="${lam:+$BUDGET:$lam}" OUT="results/efficiency/b$BUDGET" \
        LOG_DIR="logs/efficiency/b$BUDGET" ./run_efficiency.sh &
    gpu=$((1 - gpu))
    [ $gpu -eq 0 ] && wait          # 兩張卡各一個程序，跑完一對再下一對
done
wait
echo "[$(date '+%m-%d %H:%M')] done"
