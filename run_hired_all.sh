#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HiRED 的全部實驗（docs/06 §9 #1，2026-11-08）：論文正文與補充材料中，每個有 baseline 的表 / 圖都補上 HiRED，
# 設定與現有 baseline（IF-Prune 等）完全相同。結果先放獨立資料夾，是否放進論文由使用者看完結果決定。
#
#  A 準確率（run_test_suite.sh，全量資料集）                                  用在
#    A1  5 個主表資料集 + HR-8K × B ∈ {1024,1536,2048,2560,3072}，8B        tab_main、tab_memcap、fig_pareto、tab_analysis
#    A2  HR-8K，tile 上限 201（188 塊）× B=1536                              tab_hr8k
#    A3  InternVL3.5-4B / 14B × 6 資料集 × B=1536                            tab_scale（supp）
#  B 效率（bench_metrics.py，每個資料集 51 張、第 1 張 warmup；每張 GPU 同時只跑一個程序）
#    B1  5 資料集 × B ∈ {1024,2048,3072}       → results/efficiency_hired/metrics_<ds>.csv
#    B2  5 資料集 × B=1536 / 2560               → results/efficiency_hired/b1536、b2560/
#    B3  HR-8K × 5 個 budget                    → results/efficiency_hired/hrbench_8k/
#    B4  4B / 14B × 5 資料集 × B=1536           → results/efficiency_hired/InternVL3_5-<size>/
#    B5  縮放：HR-8K tile 上限 {7..201} × B=1536（11 張）→ results/scaling_hired/
#    B6  同時常駐請求數：tile 上限 {25,201,7,97,145}  → results/concurrency_hired/（GPU 必須全空）
#
# A0（B=1536 / 2560 / 3072 的主表資料集）已由先前的佇列在跑，這裡會先等它結束。
# 全部可中斷續跑（準確率從結果檔斷點接著跑；效率檔已存在就跳過）。
# 用法：nohup ./run_hired_all.sh > logs/runs_hired/all.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
read -r -a GPUS <<< "${GPUS:-0 1}"
MAIN5="docvqa infovqa mme_rw_lite hrbench_4k vstar"
LOG=logs/runs_hired
mkdir -p $LOG results/efficiency_hired results/scaling_hired results/concurrency_hired
ts() { date '+%m-%d %H:%M'; }

wait_gpu() {   # $1 GPU $2 最少空間 GB（0 = 只能有自己的程序以外 <1 GB，給 B6 用）；最多等 24 小時，不動別人的程序
    local start=$(date +%s) free used
    while :; do
        free=$(nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits)
        used=$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$2" -gt 0 ]; then [ "$free" -ge $(( $2 * 1024 )) ] && return 0
        else [ "$used" -lt 1024 ] && return 0; fi
        [ $(( $(date +%s) - start )) -ge 86400 ] && { echo "  [TIMEOUT] GPU$1 等了 24 小時"; return 1; }
        sleep 120
    done
}

# 依序在多張 GPU 上跑一串指令（每張卡同時一個），$1 = 檔名含指令清單（每行一個；行首 "GB|" 指定所需空間）
run_list() {
    local list=$1 g
    mapfile -t cmds < "$list"
    shard() {
        local k need cmd
        for ((k = $1; k < ${#cmds[@]}; k += ${#GPUS[@]})); do
            need=${cmds[$k]%%|*}; cmd=${cmds[$k]#*|}
            wait_gpu "$2" "$need" || continue
            echo "[$(ts)] GPU$2 $cmd" | cut -c1-220
            CUDA_VISIBLE_DEVICES=$2 bash -c "$cmd" || echo "  [error] $cmd" | cut -c1-220
        done
    }
    for ((g = 0; g < ${#GPUS[@]}; g++)); do shard "$g" "${GPUS[$g]}" & done
    wait
}

# ── 先等先前佇列（A0：B=1536 → 2560 / 3072）結束 ──
echo "[$(ts)] 等 A0 結束"
until grep -q 'done$' $LOG/suite_b2560_b3072.log 2>/dev/null; do sleep 120; done

# ── A 準確率 ──
echo "[$(ts)] A1"
METHODS=hired BUDGETS="1024 1536 2048 2560 3072" DATASETS="vstar hrbench_4k hrbench_8k mme_rw_lite infovqa docvqa" \
    OUT=results/runs_hired LOG_DIR=$LOG ./run_test_suite.sh
echo "[$(ts)] A2"
MAX_NUM=201 METHODS=hired BUDGETS=1536 DATASETS=hrbench_8k OUT=results/runs_hired_maxnum201 LOG_DIR=$LOG/maxnum201 \
    ./run_test_suite.sh
for size in 4B 14B; do
    echo "[$(ts)] A3 $size"
    mf=30; [ $size = 14B ] && mf=40
    MODEL=OpenGVLab/InternVL3_5-$size MIN_FREE_GB=$mf METHODS=hired BUDGETS=1536 \
        DATASETS="docvqa infovqa hrbench_4k hrbench_8k mme_rw_lite vstar" OUT=results/runs_hired LOG_DIR=$LOG ./run_test_suite.sh
done

# ── B 效率 ──
E=results/efficiency_hired
BM="$PY bench_metrics.py --num_images 51 --warmup_images 1 --skip_uncompressed --run_hired"
L=$LOG/eff_jobs.txt; : > $L
add() { [ -f "$2" ] || echo "$1|$3 > $LOG/eff_$(basename "$(dirname "$2")")_$(basename "$2" .csv).log 2>&1" >> $L; }
for ds in $MAIN5; do
    add 30 $E/metrics_$ds.csv            "$BM --dataset $ds --budgets 1024,2048,3072 --out_csv $E/metrics_$ds.csv"
    mkdir -p $E/b1536 $E/b2560
    add 30 $E/b1536/metrics_$ds.csv      "$BM --dataset $ds --budgets 1536 --out_csv $E/b1536/metrics_$ds.csv"
    add 30 $E/b2560/metrics_$ds.csv      "$BM --dataset $ds --budgets 2560 --out_csv $E/b2560/metrics_$ds.csv"
done
mkdir -p $E/hrbench_8k
add 30 $E/hrbench_8k/metrics_hrbench_8k.csv "$BM --dataset hrbench_8k --budgets 1024,1536,2048,2560,3072 --out_csv $E/hrbench_8k/metrics_hrbench_8k.csv"
for size in 4B 14B; do
    mf=30; [ $size = 14B ] && mf=40
    mkdir -p $E/InternVL3_5-$size
    for ds in $MAIN5; do
        add $mf $E/InternVL3_5-$size/metrics_$ds.csv \
            "$BM --model_name OpenGVLab/InternVL3_5-$size --dataset $ds --budgets 1536 --out_csv $E/InternVL3_5-$size/metrics_$ds.csv"
    done
done
for n in 7 13 25 49 97 145 201; do
    add 30 results/scaling_hired/metrics_n${n}_hired.csv \
        "$PY bench_metrics.py --dataset hrbench_8k --num_images 11 --warmup_images 1 --max_num $n --budgets 1536 --skip_uncompressed --run_hired --out_csv results/scaling_hired/metrics_n${n}_hired.csv"
done
echo "[$(ts)] B1–B5：$(wc -l < $L) 個工作"
run_list $L

L=$LOG/conc_jobs.txt; : > $L
for n in 25 201 7 97 145; do
    f=results/concurrency_hired/n${n}_hired.csv
    [ -f $f ] || echo "0|$PY bench_concurrency.py --method hired --max_num $n --out_csv $f > $LOG/conc_n${n}.log 2>&1" >> $L
done
echo "[$(ts)] B6：$(wc -l < $L) 個工作"
run_list $L
echo "[$(ts)] done"
