#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 實測同時請求數（bench_concurrency.py）：HR-Bench 8K、b1536，tile 上限 MAX_NUM 同 fig_scaling（7 ~ 201），
# 6 個方法。每組一個獨立程序；全部排在同一張 GPU（預設 GPU1，GPU0 有別人常駐的 2.5 GB 會讓數字偏小），
# 量測前 bench_concurrency.py 會檢查 GPU 上沒有別人的程序，有就略過該組（log 記 [skip]，之後再補跑）。
# 結果：results/concurrency/n<MAX_NUM>_<method>.csv；log：logs/concurrency/n<MAX_NUM>_<method>.log
# 用法：nohup ./run_concurrency.sh > logs/concurrency/run.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
read -r -a MAX_NUMS <<< "${MAX_NUMS:-25 201 7 97 145}"
read -r -a METHODS <<< "${METHODS:-foveatile uncompressed fastv ifprune visionzip sparsevlm}"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
OUT=${OUT:-results/concurrency}
LOG=${LOG_DIR:-logs/concurrency}
mkdir -p "$OUT" "$LOG"

for n in "${MAX_NUMS[@]}"; do
    for m in "${METHODS[@]}"; do
        [ -f "$OUT/n${n}_$m.csv" ] && continue
        echo "[$(date '+%m-%d %H:%M')] GPU$CUDA_VISIBLE_DEVICES max_num=$n $m"
        $PY bench_concurrency.py --method "$m" --max_num "$n" --out_csv "$OUT/n${n}_$m.csv" > "$LOG/n${n}_$m.log" 2>&1 \
            || { grep -q "被佔用" "$LOG/n${n}_$m.log" && echo "  [skip] GPU 上有別人的程序" || echo "  [error] 看 $LOG/n${n}_$m.log"; }
        grep -h "n_requests=" "$LOG/n${n}_$m.log" | sed 's/^/  /'
    done
done
echo "[$(date '+%m-%d %H:%M')] done"
