#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# q_attn 打分 torch.compile 的少量樣本實驗：memory / accuracy / TTFT
#
# 三種模式（環境變數 SVMA_QATTN_COMPILE）：none（現行）/ default（kernel fusion）/
# reduce-overhead（+ CUDA Graph）。每種模式各自存檔，不動正式實驗的 out_v2_*.json。
#
#   步驟 1  打分層級：bench_qattn_compile.py（速度、暫時 / 常駐記憶體、選到的 tile 是否一致）
#   步驟 2  accuracy：internvl_stream.py 跑前 N_ACC 張，ANLS 跟 none 比（同一批圖）
#   步驟 3  TTFT / peak：bench_metrics.py（N_METRIC 張；每種模式、每個 budget 各量一次）
#   最後    彙整成一張表
#
# 用法：
#   bash run_compile_small.sh
#   N_ACC=100 MODES="none default" PLAN="1024:15 2048:12" bash run_compile_small.sh
#   SKIP_STEP1=1 bash run_compile_small.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}
read -r -a MODES <<< "${MODES:-none default reduce-overhead}"
# budget:S（--qattn_tiles；要有跑 q_attn 的設定 compile 才有意義，S=0 不打分）
read -r -a PLAN <<< "${PLAN:-1024:15 2048:12 3072:12}"
N_ACC=${N_ACC:-200}           # accuracy 用的張數
N_METRIC=${N_METRIC:-24}      # TTFT / peak 用的張數
N_STEP1=${N_STEP1:-20}        # 打分層級比較用的張數
MAX_NUM=${MAX_NUM:-48}
SKIP_STEP1=${SKIP_STEP1:-0}
SKIP_ACC=${SKIP_ACC:-0}
SKIP_METRIC=${SKIP_METRIC:-0}
MTAG=$(basename "$MODEL")
OUT=${OUT:-compile_small}     # 所有輸出放這個資料夾
mkdir -p "$OUT"

echo "== q_attn compile small run :: modes=${MODES[*]} plan=${PLAN[*]} n_acc=$N_ACC n_metric=$N_METRIC =="

# ---- 步驟 1：打分層級 ----
if [ "$SKIP_STEP1" != 1 ]; then
    echo; echo "==== step 1: scoring-level speed / memory / consistency ===="
    $PY bench_qattn_compile.py --model_name "$MODEL" --num_images "$N_STEP1" --max_num "$MAX_NUM" \
        --modes "$(IFS=,; echo "${MODES[*]}")" | tee "$OUT/step1_scoring.log"
fi

# ---- 步驟 2：accuracy（前 N_ACC 張）----
if [ "$SKIP_ACC" != 1 ]; then
    for MODE in "${MODES[@]}"; do
        for P in "${PLAN[@]}"; do
            B=${P%%:*}; M=${P##*:}
            J="$OUT/acc_${MODE}_b${B}.json"
            echo; echo "==== step 2: accuracy mode=$MODE b$B S=$M (n=$N_ACC) ===="
            SVMA_QATTN_COMPILE=$MODE $PY internvl_stream.py --model_name "$MODEL" \
                --use_ds --dataset "$DATASET" --num_images "$N_ACC" --max_num "$MAX_NUM" \
                --budget "$B" --qattn_tiles "$M" --batch_size 1 --run_stream --save --output_json "$J"
        done
    done
fi

# ---- 步驟 3：TTFT / peak memory ----
if [ "$SKIP_METRIC" != 1 ]; then
    for MODE in "${MODES[@]}"; do
        for P in "${PLAN[@]}"; do
            B=${P%%:*}; M=${P##*:}
            echo; echo "==== step 3: metrics mode=$MODE b$B S=$M (n=$N_METRIC) ===="
            SVMA_QATTN_COMPILE=$MODE $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
                --num_images "$N_METRIC" --max_num "$MAX_NUM" --budgets "$B" \
                --thumb_attn --qattn_tiles "$M" --skip_uncompressed \
                --out_csv "$OUT/metric_${MODE}_b${B}.csv" || true
        done
    done
fi

# ---- 彙整 ----
echo; echo "==== summary ===="
$PY - "$OUT" "$(IFS=,; echo "${MODES[*]}")" "${PLAN[*]}" <<'EOF'
import csv, json, os, sys
from evaluate_anls import calculate_single_anls
out, modes, plan = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split()
sc = lambda p, meta: [calculate_single_anls(x, m.get("answers") or [m["answer"]]) for x, m in zip(p, meta)]

def acc(mode, B):
    f = f"{out}/acc_{mode}_b{B}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f, encoding="utf-8"))
    tag, preds = next(iter(d["candidates"].items()))
    meta = d["meta"][: len(preds)]
    s = sc(preds, meta)
    t = [x for x in d.get("ttft", {}).get(tag, [])[1:] if x]
    sel = d.get("tile_selection", {}).get(tag, [])
    return {"anls": 100 * sum(s) / len(s), "n": len(s), "ttft_full": sum(t) / len(t) if t else None, "sel": sel}

def metric(mode, B):
    f = f"{out}/metric_{mode}_b{B}.csv"
    if not os.path.exists(f):
        return None
    for r in csv.DictReader(open(f)):
        if r["method"] == f"stream_b{B}":
            return float(r["ttft_s_mean"]), float(r["peak_alloc_gb_mean"]), float(r["peak_resv_gb_mean"])
    return None

fmt = lambda x, nd: "—" if x is None else f"{x:.{nd}f}"
print(f"{'budget:S':9s} {'mode':16s} {'ANLS':>6s} {'n':>4s} {'同選tile%':>9s} {'TTFT(full)':>10s} "
      f"{'TTFT(bench)':>11s} {'peak alloc GB':>13s} {'peak resv GB':>12s}")
for p in plan:
    B, M = p.split(":")
    base = acc("none", B)
    for mode in modes:
        a, mt = acc(mode, B), metric(mode, B)
        same = None
        if a and base and a["sel"] and base["sel"]:
            k = min(len(a["sel"]), len(base["sel"]))
            same = 100 * sum(set(a["sel"][i]) == set(base["sel"][i]) for i in range(k)) / k
        print(f"{p:9s} {mode:16s} {fmt(a and a['anls'], 2):>6s} {str(a['n']) if a else '—':>4s} "
              f"{fmt(same, 1):>9s} {fmt(a and a['ttft_full'], 3):>10s} {fmt(mt and mt[0], 3):>11s} "
              f"{fmt(mt and mt[1], 3):>13s} {fmt(mt and mt[2], 3):>12s}")
print("\n同選tile% = 跟 none 選到完全相同 tile 的圖比例；TTFT(full) = 步驟 2 逐張平均（略過第 1 張）；"
      "TTFT(bench)/peak = 步驟 3 bench_metrics")
EOF
