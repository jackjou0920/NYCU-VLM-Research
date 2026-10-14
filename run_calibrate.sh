#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SVMA 超參數的 held-out 校準（docs/03_experiment_plan.md 階段 1.5 / 1.6）。
# 只在校準集上跑，測試集（DocVQA val / InfoVQA val / HR-Bench）一律不參與 m、S、預設 budget 的選擇。
#
#   階段 A（STAGE=m）：每個 budget 掃 m（S 用 svma_config 的 S(K, m) 規則）→ 選 m / K 比例
#   階段 B（STAGE=S）：固定階段 A 選出的 m，掃 S → 更新 svma_config.QATTN_TILES_BY_BUDGET
#   之後用 summarize_results.py --root results/calib 看 accuracy，bench_metrics.py 看 TTFT。
#
# 用法：
#   ./run_calibrate.sh                                   # 階段 A，預設校準集與 budget
#   STAGE=S ./run_calibrate.sh                          # 階段 B，m = floor(M_FRAC * K)，M_FRAC 預設 0.5
#   CALIB_SETS="infovqa_train_calib" BUDGETS="1024 2048" GPUS="0 1" ./run_calibrate.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
STAGE=${STAGE:-m}
read -r -a CALIB_SETS <<< "${CALIB_SETS:-infovqa_train_calib}"
read -r -a BUDGETS <<< "${BUDGETS:-1024 1536 2048 2560 3072}"
read -r -a GPUS <<< "${GPUS:-}"
M_FRAC=${M_FRAC:-0.5}
BATCH=${BATCH:-4}
OUT_ROOT=results/calib
LOG_ROOT=logs/calib
MTAG=$(basename "$MODEL")

# 每個 budget 要掃的 m：1（原本的 SVMA）、K/4、K/2、3K/4、K（純覆蓋），去重
m_grid() { $PY -c "k=$1//256; print(' '.join(map(str, sorted({1, max(1,k//4), max(1,k//2), max(1,3*k//4), k}))))"; }
# 階段 B 的 S：預設 S(K, m) 的 0、0.5、1、1.5、2 倍
s_grid() { $PY -c "
from svma_config import default_qattn_tiles
s = default_qattn_tiles($1, $2)
# 預設 S 本身不再跑：階段 A 的 svma_b<B>_m<m>（m = floor(M_FRAC*K)）就是同一組設定
print(' '.join(map(str, sorted({0, s // 2, s, (3 * s) // 2, 2 * s} - {s}))))"; }

run_one() {   # $1 dataset $2 budget $3 m $4 S(可空)
    local ds=$1 b=$2 m=$3 S=${4:-} tag extra json log
    tag="svma_b${b}_m${m}"; extra="--coverage_tiles $m"
    if [ -n "$S" ]; then tag="${tag}_S${S}"; extra="$extra --qattn_tiles $S"; fi
    json="$OUT_ROOT/$ds/${tag}_${MTAG}.json"; log="$LOG_ROOT/$ds/${tag}.log"
    mkdir -p "$OUT_ROOT/$ds" "$LOG_ROOT/$ds"
    echo "[$(date '+%m-%d %H:%M')] $ds $tag"
    $PY internvl_stream.py --dataset "$ds" --mode svma --budget "$b" $extra \
        --batch_size "$BATCH" --model_name "$MODEL" --output_json "$json" >> "$log" 2>&1 \
        || echo "  [error] $ds $tag 失敗，看 $log"
}

# 所有 (dataset, budget, m, S) 組合排成一個工作清單，GPUS 有給時輪流分到各張卡平行跑
jobs=()
for ds in "${CALIB_SETS[@]}"; do
    for b in "${BUDGETS[@]}"; do
        if [ "$STAGE" = m ]; then
            for m in $(m_grid "$b"); do jobs+=("$ds $b $m"); done
        else
            m=$($PY -c "print(max(1, int($b // 256 * $M_FRAC)))")
            for S in $(s_grid "$b" "$m"); do jobs+=("$ds $b $m $S"); done
        fi
    done
done
echo "== calibrate STAGE=$STAGE: ${#jobs[@]} 個組合 =="

run_shard() {   # $1 = shard index, $2 = shard 數
    local k
    for ((k = $1; k < ${#jobs[@]}; k += $2)); do run_one ${jobs[$k]}; done
}

if [ ${#GPUS[@]} -gt 0 ]; then
    for ((g = 0; g < ${#GPUS[@]}; g++)); do
        CUDA_VISIBLE_DEVICES=${GPUS[$g]} run_shard "$g" "${#GPUS[@]}" &
    done
    wait
else
    run_shard 0 1
fi

$PY summarize_results.py --root "$OUT_ROOT" --ref "svma_b1024_m1" || true
