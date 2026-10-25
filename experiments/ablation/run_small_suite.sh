#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 少量樣本對照實驗：不同 S（--qattn_tiles）/ q_attn 層數，在同一批 N_ACC 張圖（accuracy）與
# 同一個 session（TTFT / peak）比較。全量實驗請用 run_svma_v2.sh。
#
# CONFIGS 每一項 = "budget:S:layer"
#   S      : 邊界打分要跑 q_attn 的 tile 數（0 = 只用 prior）
#   layer  : q_attn 用第幾層（環境變數 SVMA_QATTN_LAYER；S=0 時不起作用）
# Δ 都是對同 budget 的 S=0。所有輸出放 suite_small/；已跑完的設定會略過（中斷後重跑同一指令即可）。
#
# 用法：
#   bash run_small_suite.sh
#   N_ACC=300 CONFIGS="2048:0:20 2048:8:20 2048:16:20" bash run_small_suite.sh
#   SKIP_ACC=1 bash run_small_suite.sh                  # 只量 TTFT / peak
#   SKIP_ACC=1 SKIP_METRIC=1 bash run_small_suite.sh    # 只重印彙整表
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}
read -r -a CONFIGS <<< "${CONFIGS:-\
1024:0:20 1024:8:20 1024:12:20 1024:15:20 \
2048:0:20 2048:8:20 2048:12:20 2048:16:20 \
3072:0:20 3072:8:20 3072:12:20 3072:16:20}"
N_ACC=${N_ACC:-200}
N_METRIC=${N_METRIC:-24}
MAX_NUM=${MAX_NUM:-48}
SKIP_ACC=${SKIP_ACC:-0}
SKIP_METRIC=${SKIP_METRIC:-0}
OUT=${OUT:-suite_small}
mkdir -p "$OUT"

echo "== small suite :: n_acc=$N_ACC n_metric=$N_METRIC configs=${#CONFIGS[@]} =="
name_of() { echo "b$1_S$2_L$3"; }

for C in "${CONFIGS[@]}"; do
    IFS=: read -r B S L <<< "$C"
    NAME=$(name_of "$B" "$S" "$L")
    J="$OUT/acc_${NAME}.json"
    if [ "$SKIP_ACC" != 1 ]; then
        echo; echo "==== accuracy $NAME (n=$N_ACC) ===="
        SVMA_QATTN_LAYER=$L $PY internvl_stream.py --model_name "$MODEL" \
            --use_ds --dataset "$DATASET" --num_images "$N_ACC" --max_num "$MAX_NUM" \
            --budget "$B" --qattn_tiles "$S" --batch_size 1 --run_stream --save --output_json "$J"
    fi
    if [ "$SKIP_METRIC" != 1 ] && [ ! -f "$OUT/metric_${NAME}.csv" ]; then
        echo; echo "==== metrics $NAME (n=$N_METRIC) ===="
        SVMA_QATTN_LAYER=$L $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
            --num_images "$N_METRIC" --max_num "$MAX_NUM" --budgets "$B" \
            --thumb_attn --qattn_tiles "$S" --skip_uncompressed \
            --out_csv "$OUT/metric_${NAME}.csv" || true
    fi
done

echo; echo "==== summary ===="
$PY - "$OUT" "${CONFIGS[*]}" <<'EOF'
import csv, json, os, sys
from evaluate_anls import calculate_single_anls, paired_bootstrap_delta
out, configs = sys.argv[1], sys.argv[2].split()
IFP = {1024: (70.74, 1.02), 2048: (73.43, 1.18), 3072: (75.08, 1.32)}   # 投影片上的 IF-Prune

def acc(f):
    if not os.path.exists(f): return None
    d = json.load(open(f, encoding="utf-8"))
    tag, p = next(iter(d["candidates"].items()))
    return [calculate_single_anls(x, m.get("answers") or [m["answer"]]) for x, m in zip(p, d["meta"])]
def metric(f, B):
    if not os.path.exists(f): return None
    for r in csv.DictReader(open(f)):
        if r["method"] == f"stream_b{B}":
            return float(r["ttft_s_mean"]), float(r["peak_alloc_gb_mean"]), float(r["peak_resv_gb_mean"])

fmt = lambda x, nd: "—" if x is None else f"{x:.{nd}f}"
for B in sorted({int(c.split(":")[0]) for c in configs}):
    rows = [c.split(":") for c in configs if int(c.split(":")[0]) == B]
    base = next((acc(f"{out}/acc_b{B}_S0_L{L}.json") for _, S, L in rows if S == "0"), None)
    ifp = IFP.get(B)
    print(f"\n== b{B}" + (f"   IF-Prune {ifp[0]} / {ifp[1]}s" if ifp else ""))
    print(f"  {'設定':12s} {'ANLS':>6s} {'vs S=0（95% CI）':>24s} {'TTFT':>6s} {'vs IF-Prune':>11s} "
          f"{'peak alloc':>10s} {'peak resv':>9s}")
    for _, S, L in rows:
        nm = f"b{B}_S{S}_L{L}"
        a, mt = acc(f"{out}/acc_{nm}.json"), metric(f"{out}/metric_{nm}.csv", B)
        d = "—"
        if a and base and S != "0":
            n = min(len(a), len(base)); r = paired_bootstrap_delta(a[:n], base[:n])
            d = f"{100*r[0]:+.2f} [{100*r[1]:+.2f},{100*r[2]:+.2f}]{'*' if r[1] > 0 or r[2] < 0 else ''}"
        rel = f"{100 * (mt[0] / ifp[1] - 1):+.0f}%" if (mt and ifp) else "—"
        lab = "S=0（prior）" if S == "0" else f"S={S} L{L}"
        print(f"  {lab:12s} {fmt(a and 100*sum(a)/len(a), 2):>6s} {d:>24s} "
              f"{fmt(mt and mt[0], 3):>6s} {rel:>11s} {fmt(mt and mt[1], 2):>10s} {fmt(mt and mt[2], 2):>9s}")
print("\nTTFT / peak = bench_metrics（同一個 session）；* = 95% CI 不含 0")
EOF
