"""彙整 InternVL3.5-8B 主結果：5 個測試集 × 6 個方法 × b1024~b3072（每 512 一點）的 accuracy 與效率 → results/paper/main_internvl.csv。
來源：accuracy 全部在 results/runs/<dataset>/<method>_b<B>_InternVL3_5-8B.json（DocVQA / InfoVQA / HR-4K 的
uncompressed = 舊 baseline 檔裡 batch_chat 的 references）；效率在 results/efficiency/（b1536、b2560 各自子資料夾）。"""
import csv, json, os, statistics as st, sys
sys.path.insert(0, ".")
import summarize_results as R

DS = ["docvqa", "infovqa", "hrbench_4k", "mme_rw_lite", "vstar"]
B = [1024, 1536, 2048, 2560, 3072]
M = ["foveatile", "ifprune", "fastv", "sparsevlm", "visionzip"]

def scores(ds, m, b):
    f = f"results/runs/{ds}/{m}_b{b}_InternVL3_5-8B.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); t = list(d["candidates"])[0]
    assert len(d["candidates"][t]) == len(d["meta"]), f
    return R.unit_scores(d, t, R.METRIC_OF(ds), None)

def uncompressed(ds):
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
                     d_vs_foveatile="", ci_lo="", ci_hi="", ttft_s=e_unc["ttft_s_median"], mem_delta_gb=e_unc["mem_delta_gb_mean"],
                     total_tflops=e_unc["total_tflops_mean"]))
    for b in B:
        s = scores(ds, "foveatile", b); E = eff(ds, b)
        for m in M:
            sc = scores(ds, m, b)
            er = E.get(f"{m}_b{b}")
            d = lo = hi = ""
            if m != "foveatile" and sc:
                d, lo, hi, _ = R.paired_delta(s, sc)          # FoveaTile − 該方法
                d, lo, hi = round(d, 2), round(lo, 2), round(hi, 2)
            rows.append(dict(dataset=ds, method=m, budget=b, score=round(100 * st.mean(sc.values()), 2) if sc else "",
                             d_vs_foveatile=d, ci_lo=lo, ci_hi=hi, ttft_s=er["ttft_s_median"] if er else "",
                             mem_delta_gb=er["mem_delta_gb_mean"] if er else "", total_tflops=er["total_tflops_mean"] if er else ""))
with open("results/paper/main_internvl.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("saved results/paper/main_internvl.csv", len(rows), "rows")
