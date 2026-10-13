#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 一次跑完「校準 → 報告」（兩步都呼叫 run_svma_v2.sh）：
#   第 1 步 校準：隨機抽 CALIB_N 張（seed = SEED），掃 S_GRID，依規則 A 選每個 budget 的 S，
#                 寫 svma_calibration.json（明細），並把選定的 S 寫進 svma_config.py（auto 的預設表）
#   第 2 步 報告：用選出的 S（S_LIST=auto）跑校準集以外的部分（EVAL=complement，預設）或全量（EVAL=full）
#                 TTFT 直接沿用第 1 步量好的 csv（同一組 S），不重量
#
# 用法：
#   nohup bash run_svma_calib_eval.sh > output_calib_eval.log 2>&1 &
#   EVAL=full bash run_svma_calib_eval.sh             # 第 2 步改跑全量 2801 題
#   SKIP_CALIB=1 bash run_svma_calib_eval.sh          # 已校準過，只跑第 2 步
#   CALIB_SKIP_ACC=1 bash run_svma_calib_eval.sh      # 第 1 步的 accuracy 已跑完（中斷後接續）：只補 TTFT + 校準
# 其他 run_svma_v2.sh 的環境變數（MODEL、DATASET、MARGIN、METRIC_IMAGES …）都會往下傳。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SEED=${SEED:-0}
CALIB_N=${CALIB_N:-600}
EVAL=${EVAL:-complement}                     # complement | full
SKIP_CALIB=${SKIP_CALIB:-0}
CALIB_SKIP_ACC=${CALIB_SKIP_ACC:-1}
BUDGETS=${BUDGETS:-1024 1536 2048 2560 3072 4096 5120}
S_GRID=${S_GRID:-1024:0,6,9,12,15 1536:0,4,8,10,15 2048:0,4,8,12,14,21 2560:0,4,8,12,16 3072:0,4,8,12,16,22 4096:0,4,8,12,16 5120:0,4,8,12,16}
CAL=svma_calibration.json
export BUDGETS

echo "== calib + eval :: seed=$SEED calib_n=$CALIB_N eval=$EVAL budgets=$BUDGETS =="
echo "   S_GRID=$S_GRID"

# ---- 第 1 步：校準 ----
if [ "$SKIP_CALIB" != 1 ]; then
    START=$(date +%s)
    echo; echo "########## step 1/2: calibration ##########"
    SUBSET_SEED=$SEED NUM_IMAGES=$CALIB_N S_GRID="$S_GRID" RUN_IFPRUNE=${RUN_IFPRUNE:-1} CALIBRATE=1 \
        SKIP_ACC=$CALIB_SKIP_ACC SKIP_CHECK=$CALIB_SKIP_ACC bash run_svma_v2.sh
    # 一定要是這次產生的校準檔，否則第 2 步會用到舊的 S
    if [ ! -f "$CAL" ] || [ "$(stat -c %Y "$CAL")" -lt "$START" ]; then
        echo "[abort] 第 1 步沒有產生新的 $CAL（看上面 calibrate 的輸出），第 2 步不跑"; exit 1
    fi
else
    [ -f "$CAL" ] || { echo "[abort] SKIP_CALIB=1 但沒有 $CAL"; exit 1; }
fi

echo; echo "########## 選定的 S（svma_config.py 的預設表；其他 budget 依 K 內插）##########"
python svma_config.py

# ---- 第 2 步：報告 ----
echo; echo "########## step 2/2: evaluation ($EVAL) ##########"
if [ "$EVAL" = full ]; then
    S_LIST=auto NUM_IMAGES=-1 SKIP_ACC=0 SKIP_METRICS=1 SKIP_CHECK=1 CALIBRATE=0 bash run_svma_v2.sh
else
    S_LIST=auto SUBSET_SEED=$SEED NUM_IMAGES=$CALIB_N SUBSET_COMPLEMENT=1 \
        SKIP_ACC=0 SKIP_METRICS=1 SKIP_CHECK=1 CALIBRATE=0 bash run_svma_v2.sh
fi

echo; echo "===== calib + eval done ====="
echo "校準檔 : $CAL"
echo "報告表 : table_v2_*$( [ "$EVAL" = full ] && echo "" || echo "_s${SEED}n${CALIB_N}c" ).csv"
