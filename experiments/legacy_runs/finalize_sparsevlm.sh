#!/usr/bin/env bash
# SparseVLM 重跑的收尾：等所有 sparsevlm 程序結束 → 套 budget 規則（sparsevlm_small_images.py patch）→ 重新評分。
# 用法：nohup ./finalize_sparsevlm.sh > logs/baselines_v2/finalize.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/vlm/bin/python}
R=results/baselines_v2

while pgrep -f "sparsevlm_internvl.py|sparsevlm_small_images.py run|run_sparsevlm.sh" > /dev/null; do sleep 120; done
echo "[$(date '+%m-%d %H:%M')] 所有 SparseVLM 程序已結束，開始 patch"

for ds in docvqa infovqa hrbench; do
    D=$R/internvl_sparsevlm_$ds
    if [ $ds = hrbench ]; then pat="$D/out_sparsevlm_4k_letter_InternVL3_5-8B_hrbench_b*.json"
    else pat="$D/out_sparsevlm_short_InternVL3_5-8B_${ds}_b*.json"; fi
    jsons=$(ls $pat 2>/dev/null | grep -v "summary\|pre_budget_rule\|misaligned\|_circ_\|_vanilla_")
    [ -z "$jsons" ] && { echo "  $ds：沒有結果檔"; continue; }
    $PY sparsevlm_small_images.py patch --small $D/lam0_small.json --jsons $jsons || { echo "  [error] patch $ds"; continue; }
    for J in $jsons; do
        b=$(echo $J | grep -oP '_b\K[0-9]+(?=\.json$)'); TAG=$($PY -c "import json;print(list(json.load(open('$J'))['candidates'])[0])")
        stem=${J%_b$b.json}
        if [ $ds = hrbench ]; then
            $PY evaluate_hrbench_circular.py --input_json $J --baseline_tag "$TAG" --group_by id \
                --dump_csv ${stem}_circ_b$b.csv --summary_json ${stem}_circ_b$b.json > /dev/null 2>&1 || echo "  [error] eval $J"
        else
            $PY evaluate_anls.py --input_json $J --baseline_tag "$TAG" \
                --dump_csv ${stem}_anls_b$b.csv --summary_json ${stem}_b$b.summary.json > /dev/null 2>&1 || echo "  [error] eval $J"
        fi
    done
done

$PY - <<'PY'
import json, glob, re, statistics as st, sys
sys.path.insert(0, ".")
import summarize_results as R
for ds, metric, pat in [("docvqa", "anls", "short_InternVL3_5-8B_docvqa"), ("infovqa", "anls", "short_InternVL3_5-8B_infovqa"),
                        ("hrbench_4k", "hrbench_circular", "4k_letter_InternVL3_5-8B_hrbench")]:
    row = []
    for b in (1024, 2048, 3072, 4096, 5120):
        fs = glob.glob(f"results/baselines_v2/internvl_sparsevlm_*/out_sparsevlm_{pat}_b{b}.json")
        if not fs:
            row.append("   -  "); continue
        d = json.load(open(fs[0])); t = list(d["candidates"])[0]
        sc = R.unit_scores(d, t, metric, None)
        row.append(f"{100 * st.mean(sc.values()):6.1f}")
    print(f"{ds:12s} " + " ".join(row))
PY
echo "[$(date '+%m-%d %H:%M')] done"
