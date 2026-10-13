#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 效能表（TTFT / peak memory）補齊：六種方法在同一個 process、同一批圖上量。
#
# 之前的 metrics CSV 是各 run_*.sh 分開跑的（n=24），有幾個問題：
#   - FastV HR-Bench 沒跑、FastV DocVQA / SparseVLM DocVQA,HR-Bench 沒有 TTFT
#     （舊版 bench_metrics.py 還沒量）、IF-Prune DocVQA 缺 b1024
#   - 每份 CSV 各自量一次 uncompressed / stream，數字互相對不齊
#   - 部分 stream 列是 thumb_attn=off，不是最終 SVMA 設定
# 這支每個 dataset 只跑一次 bench_metrics.py，--run_* 全開，uncompressed /
# stream_b<K>(SVMA) / sparsevlm / fastv / ifprune / visionzip
# 全部並排，一份 CSV 就是一張完整的表。
#
# TTFT 是牆鐘時間，同一張 GPU 上兩個 job 會互搶、數字失真，所以一張卡一次只跑
# 一個 dataset：GPU_A 依序跑 DATASETS_A，GPU_B 依序跑 DATASETS_B，兩條佇列並行。
#
# HR-Bench split 讀 internvl_stream.py 的 HRBENCH_SPLIT（目前 hrbench_4k）。
#
# 用法：
#   bash run_metrics_all.sh                         # 預設 n=100，兩張卡
#   METRIC_IMAGES=24 bash run_metrics_all.sh        # 改 n
#   GPU_A=0 GPU_B=0 bash run_metrics_all.sh         # 只有一張卡 → 兩條佇列會搶同一張，不建議
#   只重畫表：bash run_metrics_all.sh table
#
# 產出（OUTDIR）：
#   metrics_all_<split>_<model>_<dataset>_n<N>.csv
#   log_<dataset>.log
#   summary_tables.md    —— TTFT / peak memory markdown 表
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
BUDGETS=${BUDGETS:-1024,2048,3072,4096,5120}
METRIC_IMAGES=${METRIC_IMAGES:-100}
MAX_NUM=${MAX_NUM:-48}
AGG_LAYER=${AGG_LAYER:-2}                    # FastV，跟 run_fastv.sh 一致
CTX_RATIO=${CTX_RATIO:-0.15625}              # VisionZip，跟 run_visionzip.sh 一致
PRUNE_LAYER=${PRUNE_LAYER:-2}                # IF-Prune，跟 run_ifprune.sh 一致
HEAD_DTYPE=${HEAD_DTYPE:-bf16}
SMALL_CKPT=${SMALL_CKPT:-ZachSun/internvl2_5_1b_pruning_kl_lora32}
SMALL_BASE=${SMALL_BASE:-OpenGVLab/InternVL2_5-1B}

GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
DATASETS_A=${DATASETS_A:-"hrbench infovqa"}
DATASETS_B=${DATASETS_B:-"docvqa"}

MTAG=$(basename "$MODEL")
SPLIT_TAG=${HRBENCH_SPLIT:-hrbench_4k}   # 跟 internvl_stream.py 同一個環境變數
SPLIT_TAG=${SPLIT_TAG#hrbench_}
OUTDIR=${OUTDIR:-internvl_metrics_n${METRIC_IMAGES}}
mkdir -p "$OUTDIR"

csv_for() {
    local ds=$1
    if [ "$ds" = hrbench ]; then
        echo "$OUTDIR/metrics_all_${SPLIT_TAG}_${MTAG}_${ds}_n${METRIC_IMAGES}.csv"
    else
        echo "$OUTDIR/metrics_all_${MTAG}_${ds}_n${METRIC_IMAGES}.csv"
    fi
}

run_one() {
    local gpu=$1 ds=$2
    local out; out=$(csv_for "$ds")
    echo "==== [GPU $gpu] $MTAG $ds :: n=$METRIC_IMAGES budgets=$BUDGETS -> $out ===="
    CUDA_VISIBLE_DEVICES=$gpu $PY bench_metrics.py --model_name "$MODEL" --dataset "$ds" \
        --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" --budgets "$BUDGETS" \
        --run_sparsevlm \
        --run_fastv --fastv_agg_layer "$AGG_LAYER" \
        --run_visionzip --visionzip_contextual_ratio "$CTX_RATIO" \
        --run_ifprune --ifprune_prune_layer "$PRUNE_LAYER" --ifprune_head_dtype "$HEAD_DTYPE" \
        --ifprune_small_checkpoint "$SMALL_CKPT" --ifprune_small_base "$SMALL_BASE" \
        --out_csv "$out" > "$OUTDIR/log_${ds}.log" 2>&1 \
        && echo "==== [GPU $gpu] $ds done ====" \
        || echo "==== [GPU $gpu] $ds FAILED (see $OUTDIR/log_${ds}.log) ===="
}

run_queue() {
    local gpu=$1; shift
    for ds in "$@"; do run_one "$gpu" "$ds"; done
}

make_tables() {
    OUTDIR="$OUTDIR" MTAG="$MTAG" SPLIT_TAG="$SPLIT_TAG" N="$METRIC_IMAGES" BUDGETS="$BUDGETS" \
    $PY - <<'EOF' | tee "$OUTDIR/summary_tables.md"
import csv, os
outdir, mtag, split, n = os.environ["OUTDIR"], os.environ["MTAG"], os.environ["SPLIT_TAG"], os.environ["N"]
budgets = [int(b) for b in os.environ["BUDGETS"].split(",")]
files = {
    f"HR-Bench {split.upper()}": f"{outdir}/metrics_all_{split}_{mtag}_hrbench_n{n}.csv",
    "DocVQA": f"{outdir}/metrics_all_{mtag}_docvqa_n{n}.csv",
    "InfoVQA": f"{outdir}/metrics_all_{mtag}_infovqa_n{n}.csv",
}
methods = [("Uncompressed", "uncompressed"), ("SparseVLM", "sparsevlm"), ("FastV", "fastv"),
           ("VisionZip", "visionzip"), ("IF-Prune", "ifprune"), ("SVMA (Ours)", "stream")]

def cell(r):
    if not r:
        return "—"
    t, m = r.get("ttft_s_mean") or "", r.get("peak_alloc_gb_mean") or ""
    return f"{float(t):.2f} / {float(m):.2f}" if t and m else \
           f"{float(t):.2f} / —" if t else f"— / {float(m):.2f}" if m else "—"

print(f"# TTFT (s) / Peak Memory (GB, peak_alloc) — {mtag}, n={n}\n")
for ds, f in files.items():
    print(f"## {ds}\n")
    if not os.path.exists(f):
        print(f"(missing: {f})\n")
        continue
    rows = {r["method"]: r for r in csv.DictReader(open(f))}
    print("| Method | " + " | ".join(f"b{b}" for b in budgets) + " |")
    print("|---|" + "---|" * len(budgets))
    for label, key in methods:
        if key == "uncompressed":
            cells = [cell(rows.get("uncompressed"))] + [""] * (len(budgets) - 1)
        else:
            cells = [cell(rows.get(f"{key}_b{b}")) for b in budgets]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()
EOF
}

if [ "${1:-}" = table ]; then make_tables; exit 0; fi

echo "== metrics :: model=$MTAG n=$METRIC_IMAGES budgets=$BUDGETS hrbench_split=$SPLIT_TAG outdir=$OUTDIR =="
echo "   GPU $GPU_A: $DATASETS_A"
echo "   GPU $GPU_B: $DATASETS_B"

# shellcheck disable=SC2086
run_queue "$GPU_A" $DATASETS_A &
PID_A=$!
# shellcheck disable=SC2086
run_queue "$GPU_B" $DATASETS_B &
PID_B=$!
wait $PID_A $PID_B

echo; make_tables
echo "===== metrics done → $OUTDIR/summary_tables.md ====="
