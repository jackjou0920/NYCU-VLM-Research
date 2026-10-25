"""彙整 InternVL3.5-8B 主結果：5 個測試集 × 6 個方法 × b1024~b3072（每 512 一點）的 accuracy 與效率 → results/paper/main_internvl.csv。
來源：b1536 / b2560 在 results/runs/；其餘 budget 的 DocVQA / InfoVQA / HR-4K baseline 在 results/baselines(_v2)/，
SVMA 與 MME-RW Lite / V* 在 results/runs/；效率在 results/efficiency/（b1536、b2560 各自子資料夾）。"""
import csv, glob, json, os, statistics as st, sys
sys.path.insert(0, ".")
import summarize_results as R

DS = ["docvqa", "infovqa", "hrbench_4k", "mme_rw_lite", "vstar"]
B = [1024, 1536, 2048, 2560, 3072]
M = ["svma", "ifprune", "fastv", "sparsevlm", "visionzip"]
OLD = {"docvqa": ("docvqa", "short_InternVL3_5-8B_docvqa"), "infovqa": ("infovqa", "short_InternVL3_5-8B_infovqa"),
       "hrbench_4k": ("hrbench", "4k_letter_InternVL3_5-8B_hrbench")}

def scores(ds, m, b):
    f = f"results/runs/{ds}/{m}_b{b}_InternVL3_5-8B.json"
    if not os.path.exists(f) and m != "svma" and ds in OLD:
        d_, pat = OLD[ds]
        root = "baselines_v2" if m == "sparsevlm" else "baselines"
        hit = glob.glob(f"results/{root}/internvl_{m}_{d_}/out_{m}_{pat}_b{b}.json")
        f = hit[0] if hit else None
    if not f or not os.path.exists(f):
        return None
    d = json.load(open(f)); t = list(d["candidates"])[0]
    assert len(d["candidates"][t]) == len(d["meta"]), f
    return R.unit_scores(d, t, R.METRIC_OF(ds), None)

def uncompressed(ds):
    if ds in OLD:
        d_, pat = OLD[ds]
        d = json.load(open(glob.glob(f"results/baselines/internvl_fastv_{d_}/out_fastv_{pat}_b1024.json")[0]))
        return R.unit_scores({"candidates": {"u": d["references"]}, "meta": d["meta"]}, "u", R.METRIC_OF(ds), None)
    d = json.load(open(f"results/runs/{ds}/uncompressed_InternVL3_5-8B.json"))
    return R.unit_scores(d, list(d["candidates"])[0], R.METRIC_OF(ds), None)

def eff(ds, b):
    f = {1536: f"results/efficiency/b1536/metrics_{ds}.csv", 2560: f"results/efficiency/b2560/metrics_{ds}.csv"}.get(
        b, f"results/efficiency/metrics_{ds}.csv")
    return {r["method"]: r for r in csv.DictReader(open(f))}

rows = []
for ds in DS:
    unc = uncompressed(ds); e_unc = eff(ds, 1024)["uncompressed"]
    rows.append(dict(dataset=ds, method="uncompressed", budget="", score=round(100 * st.mean(unc.values()), 2),
                     d_vs_svma="", ci_lo="", ci_hi="", ttft_s=e_unc["ttft_s_median"], mem_delta_gb=e_unc["mem_delta_gb_mean"],
                     total_tflops=e_unc["total_tflops_mean"]))
    for b in B:
        s = scores(ds, "svma", b); E = eff(ds, b)
        for m in M:
            sc = scores(ds, m, b)
            er = E.get(("stream" if m == "svma" else m) + f"_b{b}")
            d = lo = hi = ""
            if m != "svma" and sc:
                d, lo, hi, _ = R.paired_delta(s, sc)          # SVMA − 該方法
                d, lo, hi = round(d, 2), round(lo, 2), round(hi, 2)
            rows.append(dict(dataset=ds, method=m, budget=b, score=round(100 * st.mean(sc.values()), 2) if sc else "",
                             d_vs_svma=d, ci_lo=lo, ci_hi=hi, ttft_s=er["ttft_s_median"] if er else "",
                             mem_delta_gb=er["mem_delta_gb_mean"] if er else "", total_tflops=er["total_tflops_mean"] if er else ""))
with open("results/paper/main_internvl.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("saved results/paper/main_internvl.csv", len(rows), "rows")
