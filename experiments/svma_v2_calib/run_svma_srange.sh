#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SVMA：每個 budget 用預設 S（svma_config.default_qattn_tiles）跑整個資料集；DELTA > 0 時改成掃預設 S 的 ±DELTA。
# 資料集：DATASETS="docvqa hrbench"（預設，依序跑），可選 infovqa | docvqa | hrbench
#   infovqa / docvqa：ANLS（evaluate_anls.py）
#   hrbench        ：官方 CircularEval（同一題 4 種選項輪換全對才算對；bootstrap 以「組」為單位），
#                    split / prompt 由 internvl_stream.py 的 HRBENCH_SPLIT / HRBENCH_PROMPT 決定
#   預設 S（DELTA=0）：b1024 → 15、b2048 → 14、b3072 → 12、b4096 → 8、b5120 → 4
#   DELTA=2 時：b1024 → S = 13..17、b2048 → 12..16 …，Δ 是對該 budget 預設 S 的 paired bootstrap
# 每個 (budget, S)：accuracy（全量）+ TTFT / peak memory（bench_metrics，小 n），每個資料集各一張彙整表。
#
# 用法：
#   nohup bash run_svma_srange.sh > output_srange.log 2>&1 &                 # DocVQA + HR-Bench，五個 budget
#   兩張 GPU 平行（資料集分開跑，輸出檔各自獨立不會衝突）：
#     GPU=0 DATASETS=docvqa  nohup bash run_svma_srange.sh > output_srange_docvqa.log 2>&1 &
#     GPU=1 DATASETS=hrbench nohup bash run_svma_srange.sh > output_srange_hrbench.log 2>&1 &
#   DELTA=2 DATASETS=infovqa bash run_svma_srange.sh     # 掃預設 S 的 ±2
#   SKIP_ACC=1 bash run_svma_srange.sh                  # 只量 TTFT / peak
#   SKIP_ACC=1 SKIP_METRICS=1 bash run_svma_srange.sh   # 只重印彙整表
#
# resume：internvl_stream.py 從每個 tag「已完成的筆數」接著跑；已量過的 TTFT csv 會略過；
#   全量 json 裡已有的同一個 S（或等價的舊 cascade 結果）會直接沿用（reuse_results.py）。
#
# 產出：
#   accuracy : out_v2_<model>_<ds>_b<B>.json（tag：budget=<B>_svma_S<S>[_qa<L>ctx]）
#   TTFT     : metrics_srange_<model>_<ds>_b<B>_S<S>.csv
#   彙整表   : table_srange_<model>_<ds>.csv
#   <ds> = infovqa / docvqa / hrbench_4k（或 hrbench_8k）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 多個資料集：每個資料集用子程序各跑一次（DATASET 由這裡指定）
DATASETS=${DATASETS:-${DATASET:-docvqa hrbench}}
if [ -z "${_SRANGE_CHILD:-}" ]; then
    for D in $DATASETS; do
        echo; echo "################ dataset = $D ################"
        _SRANGE_CHILD=1 DATASET=$D bash "$0"
    done
    exit 0
fi

PY=${PY:-python}
MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
DATASET=${DATASET:-infovqa}
read -r -a BUDGETS <<< "${BUDGETS:-1024 2048 3072 4096 5120}"
DELTA=${DELTA:-0}                             # 0 = 只跑預設 S；> 0 = 掃預設 S 的 ±DELTA
NUM_IMAGES=${NUM_IMAGES:--1}                  # -1 = 全量
METRIC_IMAGES=${METRIC_IMAGES:-24}            # TTFT / peak 用，小 n 即可
MAX_NUM=${MAX_NUM:-48}
SKIP_ACC=${SKIP_ACC:-0}
SKIP_METRICS=${SKIP_METRICS:-0}
METRICS_FORCE=${METRICS_FORCE:-0}             # 1 = 已有的 TTFT csv 也重量
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU

QA_SUF=${SVMA_QATTN_LAYER:-$(grep -oP '^QATTN_LAYER = int\(os.environ.get\("SVMA_QATTN_LAYER", "\K\d+' internvl_stream.py)}$(grep -qP '^QATTN_CTX = True' internvl_stream.py && echo ctx || true)
MTAG=$(basename "$MODEL")
DSNAME=$DATASET
HB_PROMPT=letter
if [ "$DATASET" = hrbench ]; then
    DSNAME=$(grep -oP '^HRBENCH_SPLIT = "\K[^"]+' internvl_stream.py)          # hrbench_4k | hrbench_8k
    HB_PROMPT=$(grep -oP '^HRBENCH_PROMPT = "\K[^"]+' internvl_stream.py)      # letter | open
    [ "$HB_PROMPT" = letter ] || { echo "[abort] HRBENCH_PROMPT=$HB_PROMPT：CircularEval 只適用 letter"; exit 1; }
fi
OUTBASE="out_v2_${MTAG}_${DSNAME}"
MPREFIX="metrics_srange_${MTAG}_${DSNAME}"

default_s() { $PY -c "from svma_config import default_qattn_tiles as f; print(f($1))"; }
s_range() {   # 預設 S 的 ±DELTA（不小於 0）
    local d0; d0=$(default_s "$1")
    $PY -c "print(' '.join(str(s) for s in sorted({max(0, $d0 + d) for d in range(-$DELTA, $DELTA + 1)})))"
}
tag_for() { if [ "$2" -gt 0 ]; then echo "budget=$1_svma_S$2_qa$QA_SUF"; else echo "budget=$1_svma_S0"; fi; }

echo "== SVMA :: model=$MTAG dataset=$DSNAME n=$NUM_IMAGES delta=±$DELTA qattn=$QA_SUF gpu=${GPU:-auto} =="
for B in "${BUDGETS[@]}"; do echo "   b$B: 預設 S=$(default_s "$B")  跑 S = $(s_range "$B")"; done

for B in "${BUDGETS[@]}"; do
    J="${OUTBASE}_b${B}.json"
    DEF=$(default_s "$B")

    # ---- accuracy（全量）----
    if [ "$SKIP_ACC" != 1 ]; then
        for S in $(s_range "$B"); do
            $PY reuse_results.py --full_json "$J" --target_json "$J" --budget "$B" --S "$S" \
                --qa "$QA_SUF" --dataset "$DATASET" || true
            echo; echo "==== $MTAG $DATASET b$B :: accuracy S=$S ===="
            $PY internvl_stream.py --model_name "$MODEL" --use_ds --dataset "$DATASET" \
                --num_images "$NUM_IMAGES" --max_num "$MAX_NUM" --budget "$B" --qattn_tiles "$S" \
                --batch_size 1 --run_stream --save --output_json "$J"
        done
        echo; echo "==== $MTAG $DSNAME b$B :: eval (baseline = 預設 S=$DEF) ===="
        if [ "$DATASET" = hrbench ]; then
            $PY evaluate_hrbench_circular.py --input_json "$J" --baseline_tag "$(tag_for "$B" "$DEF")" \
                --group_by id --prompt_mode "$HB_PROMPT" \
                --summary_json "${OUTBASE}_srange_circ_b${B}.json" || true
        else
            $PY evaluate_anls.py --input_json "$J" --baseline_tag "$(tag_for "$B" "$DEF")" \
                --summary_json "${OUTBASE}_srange_b${B}.summary.json" || true
        fi
    fi

    # ---- TTFT / peak memory ----
    if [ "$SKIP_METRICS" != 1 ]; then
        for S in $(s_range "$B"); do
            CSV="${MPREFIX}_b${B}_S${S}.csv"
            if [ "$METRICS_FORCE" != 1 ] && [ -f "$CSV" ]; then
                echo "==== metrics b$B S=$S：$CSV 已存在，略過 ===="; continue
            fi
            echo; echo "==== $MTAG $DATASET b$B :: metrics S=$S (n=$METRIC_IMAGES) ===="
            $PY bench_metrics.py --model_name "$MODEL" --dataset "$DATASET" \
                --num_images "$METRIC_IMAGES" --max_num "$MAX_NUM" --budgets "$B" \
                --skip_uncompressed --thumb_attn --qattn_tiles "$S" --out_csv "$CSV" || true
        done
    fi
done

# ---- 彙整 ----
echo; echo "==== summary ===="
$PY - "$OUTBASE" "$MPREFIX" "$QA_SUF" "$DELTA" "$MTAG" "$DSNAME" "$DATASET" "${BUDGETS[@]}" <<'EOF'
import csv, json, os, sys
from evaluate_anls import calculate_single_anls, paired_bootstrap_delta
from svma_config import default_qattn_tiles
outbase, mprefix, qa, delta, mtag, dsname, dataset, *budgets = sys.argv[1:]
delta = int(delta)

def unit_scores(preds, meta):
    """比較用的單位分數：ANLS 每題一個；HR-Bench 每「組」（4 種選項輪換）一個 0/1（全對才算對）。"""
    if dataset != "hrbench":
        return [calculate_single_anls(x, m.get("answers") or [m.get("answer", "")]) for x, m in zip(preds, meta)]
    from evaluate_hrbench import canon_pred, gold_letter
    from evaluate_hrbench_circular import build_groups
    ok = [1 if (canon_pred(str(x), m, "letter") or None) == gold_letter(m) and gold_letter(m) else 0
          for x, m in zip(preds, meta)]
    return [1.0 if all(ok[i] for i in g) else 0.0 for g in build_groups(meta, "id")]
METRIC = "CircAcc" if dataset == "hrbench" else "ANLS"

def bench(path, B):
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("method") == f"stream_b{B}":
            return float(r["ttft_s_mean"]), float(r["peak_alloc_gb_mean"]), float(r.get("kv_cache_mb_mean") or "nan")

fmt = lambda x, nd: "—" if x is None else f"{x:.{nd}f}"
rows_out = [["budget", "S", "預設", "ANLS" if dataset != "hrbench" else "CircAcc", "vs 預設 S (95% CI)",
             "TTFT(s)", "peak GB", "KV MB", "n"]]
for B in map(int, budgets):
    d0 = default_qattn_tiles(B)
    path = f"{outbase}_b{B}.json"
    d = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    meta = d.get("meta", [])
    scores = {}
    for S in sorted({max(0, d0 + k) for k in range(-delta, delta + 1)}):
        tag = f"budget={B}_svma_S{S}" + (f"_qa{qa}" if S > 0 else "")
        p = d.get("candidates", {}).get(tag, [])
        if meta and len(p) == len(meta):
            scores[S] = unit_scores(p, meta)
        else:
            scores[S] = None if not p else f"{len(p)}/{len(meta)}"
    print(f"\n== b{B}（預設 S={d0}，K={B // 256}）")
    print(f"  {'S':>3s} {METRIC:>7s} {'vs 預設 S (95% CI)':>24s} {'TTFT':>7s} {'peak GB':>8s} {'KV MB':>7s}")
    base = scores.get(d0) if isinstance(scores.get(d0), list) else None
    for S, sc in scores.items():
        m = bench(f"{mprefix}_b{B}_S{S}.csv", B)
        if isinstance(sc, list):
            anls = 100 * sum(sc) / len(sc)
            dl = "—"
            if base and S != d0:
                r = paired_bootstrap_delta(sc, base)
                dl = f"{100*r[0]:+.2f} [{100*r[1]:+.2f},{100*r[2]:+.2f}]{'*' if r[1] > 0 or r[2] < 0 else ''}"
            a_s, n = f"{anls:.2f}", len(sc)
        else:
            a_s, dl, n = ("—" if sc is None else f"({sc})"), "—", 0
        mark = "  ← 預設" if S == d0 else ""
        print(f"  {S:3d} {a_s:>7s} {dl:>24s} {fmt(m and m[0], 3):>7s} {fmt(m and m[1], 2):>8s} {fmt(m and m[2], 1):>7s}{mark}")
        rows_out.append([B, S, "*" if S == d0 else "", a_s, dl, fmt(m and m[0], 3), fmt(m and m[1], 2),
                         fmt(m and m[2], 1), n])
out = f"table_srange_{mtag}_{dsname}.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows_out)
unit = "組（4 種選項輪換）" if dataset == "hrbench" else "題"
print(f"\n{METRIC}；n = {unit}數；* = 95% CI 不含 0；TTFT / peak = bench_metrics（{os.path.basename(mprefix)}_b<B>_S<S>.csv）\n[saved] {out}")
EOF
