# 06 CVPR 2027 投稿總規劃與待辦（2026-10-26 建立，之後的 session 從這份接續）

## 0. 一句話

**FoveaTile**（原名 SVMA）：*Look Before You Encode: Foveated Tile Streaming for Resolution-Independent VLM Inference*。
在 ViT 之前用 VLM 自己對低解析度覆蓋 tile 的 attention 決定要編碼哪些高解析度細節 tile，邊界 tile 再用 q_attn 重排，
逐 tile 串流寫 KV；ViT、KV、記憶體都是 O(K)，跟解析度 N 無關。訓練免、不用額外模型。凍結版本 v1.0（2026-10-17）。

- 投稿：CVPR 2027，截稿約 **2026-11-08**（確切時間以官網為準）。用 Overleaf 寫。
- 實驗必須在 **約 11/01 前**跑完，留一週寫作。

## 1. 論文各章狀態

| 章節 | 檔案 | 狀態 |
|---|---|---|
| Abstract | `paper/sec/0_abstract.tex` | **未寫**（最後寫） |
| 1 Introduction | `paper/sec/1_intro.tex` | **未寫**。Fig. 1 teaser 建議：成本 vs tile 數 + 平均保留率 vs 記憶體；"foveated" 引 RAM（`refs_method.bib` 的 `ram`） |
| 2 Related Work | `paper/sec/2_related.tex` | **未寫**。三段：token pruning（FastV、SparseVLM、VisionZip、IF-Prune）、高解析度壓縮（HiRED、GlobalCom2）、zoom-in / search（ViCrop、V*、ZoomEye、DyFo、RAP）。要寫清楚跟 ViCrop、GlobalCom2 的差別（02 文件） |
| 3 Method | `paper/sec/3_method.tex` | **完成**（已核對程式碼；Fig. 2 overview、Fig. 3 boundary scoring、Alg. 1、Tab. 1 成本） |
| 4 Experiments | `paper/sec/4_experiments.tex` | **草稿完成**（4.1 Setup、4.2 主結果、4.3 解析度、4.4 模型尺寸、4.5 Ablation）；`\TODO` 待補：模型尺寸、同時請求數（估計值→實測）、random prior |
| 5 Conclusion + Limitations | `paper/sec/5_conclusion.tex` | **未寫**。限制見第 6 節 |
| Supplementary | — | **未寫**：校準流程（`docs/04_calibration_protocol.md` 改寫）、baseline 移植細節、所有 budget 的完整表 + CI、負面結果（3b、v1.1、A1/A2）、layer 在 InfoVQA val 上選的揭露 |

Overleaf 上傳清單：
- `preamble.tex` 要有：`algorithm`、`algpseudocode`（`[noend]`）、`xspace`、`\newcommand{\method}{FoveaTile\xspace}`（已加）；
  **還要加** `\usepackage{multirow}`、`\usepackage{colortbl}`、`\definecolor{oursbg}{HTML}{E8F0FC}`
- `main.bib` = `paper/main.bib`（20 筆）；Intro / Related 要用的額外引用在 `paper/refs_method.bib`（`streamingllm`、`ram`）
- `main.tex` 依序 `\input` sec/0~5
- 上傳 `paper/figures/*.pdf`、`paper/tables/*.tex`
- 篇幅：第 3、4 章含圖表約 5.5~6 頁，加上 Intro / Related 會逼近 8 頁 → 可能要把 Tab.（模型尺寸）或 Alg. 1 移到 supplementary

## 2. 圖表怎麼產生

- 第 4 章所有表格與圖：`python results/paper/build_paper_results.py` → `paper/tables/tab_{memcap,main,scale,hr8k,analysis,ablation}.tex`、
  `paper/figures/fig_{pareto,scaling}.pdf`、`results/paper/main_6ds.csv`。**實驗補完後重跑一次即可**，缺的格子會印紅色 `?`（`\pend`）
- 第 3 章的兩張圖：`python paper/figures/make_method_figures.py`（範例 = V* Bench direct_attributes 第 51 題，資料在
  `paper/figures/method_examples_vstar_b1536.json`，由 `paper/figures/extract_method_example.py` 產生）
- 呈現規劃：`docs/05_results_presentation_plan.md`

## 3. 實驗狀態

### 已完成（InternVL3.5-8B）
- 6 個測試集（DocVQA val、InfoVQA val、MME-RW Lite、HR-Bench 4K、HR-Bench 8K、V* Bench）× 6 個方法 × b1024~b3072（4K/8K 外加 b4096/b5120 部分）：`results/runs/<ds>/`
- 效率：`results/efficiency/`（b1536、b2560 在子資料夾；HR-8K 在 `results/efficiency/hrbench_8k/`）
- 解析度縮放（HR-8K 10 張圖，7~188 塊）：`results/scaling/summary.json`
- held-out 校準（m、S 掃描）：`results/calib/`

### 背景執行中（2026-10-26 13:40 的狀態）
| 工作 | 腳本 / log | 結果 | 預計 |
|---|---|---|---|
| HR-8K tile 上限 201（188 塊）× 所有方法 × b1024~b3072（b1536、b2560 已完成；第二輪 10/26 17:51 起，使用者決定繼續跑） | PID 3466489；`logs/hrbench8k_eff_and_maxnum201.log` | `results/runs_maxnum201/hrbench_8k/` | 10/27 凌晨 |
| random prior ablation（held-out × 5 budget） | `run_ablation_randprior.sh`；`logs/calib/randprior.log` | `results/calib/<ds>/*_randprior_*.json` | 上一項結束後約 1.5 小時 |
| InternVL3.5-4B → 14B × 6 測試集 × 6 方法 × b1536 | `run_model_scale.sh`；`logs/model_scale.log` | `results/runs/<ds>/*_InternVL3_5-{4B,14B}.json` | 約 10/28 晚上 |

檢查進度：`tail logs/model_scale.log logs/calib/randprior.log logs/hrbench8k_eff_and_maxnum201.log`；GPU 跟別人共用，不要砍別人的程序（等，最多 24 小時）。

### 還沒排的
1. **4B / 14B 的效率量測**（Tab. 模型尺寸的 TTFT / 記憶體欄）：`run_efficiency.sh` 目前沒有 `MODEL` 參數 → 要加（`bench_metrics.py --model_name`），
   輸出到 `results/efficiency/InternVL3_5-<size>/metrics_<ds>.csv`（`build_paper_results.py` 已經照這個路徑讀）。
   SparseVLM 的 λ 要用該模型 accuracy 實驗校準出的值：`grep 校準 logs/model_scale.log`（注意 `logs/runs/<ds>/sparsevlm_b1536.log` 8B/4B/14B 共用同一個檔名）
2. （有時間才做）同時請求數實測、decode / total latency
3. 全部跑完後：重跑 `build_paper_results.py`，補 `4_experiments.tex` 的 `\TODO`

## 4. 已定的決策（不要再改）

- 方法凍結 v1.0：m = ⌊K/2⌋、S = {1024:5, 1536:4, 2048:4, 2560:3, 3072:3}、預設 b1536；v1.1 候選、3b、A1/A2 都依事先規則否決（`docs/04_calibration_protocol.md`）
- 換模型不重調 m、S；attention 層號依相對深度換算（`foveatile_internvl.attn_layers`：36 層 = (21–28, 20)，40 層 = (23–31, 22)），已驗證 8B 輸出逐題不變
- 資料集名稱寫 **V\* Bench**；HR-Bench 主設定 tile 上限 47（8K 約切 25 塊），188 塊是另外的縮放實驗
- 主表區塊放 b1536 / b2560 / b3072；b1024、b2048 放 Pareto 圖與附錄（所有方法同一組 budget）
- 主表 / 平均保留率 / 記憶體上限表 / Pareto / 模型尺寸表都用 **5 個測試集**（DocVQA、InfoVQA、MME-RW Lite、HR-Bench 4K、V* Bench）；HR-Bench 8K 另外放在 `tab_hr8k`（25 vs 188 塊）與分析表（使用者 2026-10-26 決定）
- 主表第二行是保留率（加 %）
- 正文不放「Where FoveaTile helps and where it fails」分析節（使用者 2026-10-26 決定）；內容移到 `paper/sec/supp_analysis.tex` 給 supplementary。Conclusion 仍要用一兩句交代限制（極小物件的 prior 召回率），建議正面寫成未來方向
- 主表的 ↑/↓ = FoveaTile 對該欄最強 baseline 的配對 bootstrap（雙向都標）
- 主結果先放 **Table「記憶體上限下的 accuracy」**（`tab_memcap.tex`）：≤0.5 / ≤1 / ≤2 GB（= uncompressed）/ 不限，FoveaTile 每一欄都第一；FastV、SparseVLM、IF-Prune 在 ≤1 GB 跑不了，IF-Prune 比 uncompressed 還吃記憶體（2.9 vs 2.0 GB）
- 主表成本欄改成相對 uncompressed 的節省倍率（TTFT / 記憶體 / FLOPs，越大越好；絕對值放附錄），最佳粗體、第二名底線
- Ablation 用 5 個 budget 的平均（單看 b1536 時「只有覆蓋」平均略高，跨 budget 才完整）
- 呈現原則：圖表用最能凸顯 FoveaTile 的方式（成本與 accuracy 並列、Pareto、縮放），但不刪資料集、不只挑對自己有利的 budget；弱點寫在文字
- Low-Res-Fill 不當 baseline（使用者決定）；「只有覆蓋」放在 ablation
- 第二個模型家族不做，用 InternVL3.5 4B / 8B / 14B 代替

## 5. 目前主要數字（8B；平均 = 主表的 5 個測試集，不含 HR-8K）

- 平均保留率 FoveaTile / IF-Prune / FastV：b1024 91.5 / 94.5 / 82.1；b1536 **95.0** / 94.9 / 90.0；b2560 **97.6** / 97.1 / 95.5；b3072 **99.3** / 97.4 / 97.2
- b1536 記憶體節省（相對 uncompressed）：FoveaTile 5.0×、FastV 1.5×、IF-Prune 0.7×（比 uncompressed 還多）；TTFT 比 IF-Prune 快 1.6×
- 記憶體上限表：≤0.5 / ≤1 / ≤2 GB / 不限，FoveaTile 95.0 / 99.3 / 99.3 / 99.3，全部第一
- InfoVQA、MME-RW 每個 budget 都顯著贏所有 baseline；HR-Bench、V* Bench 跟 baseline 的差距全部不顯著；DocVQA 只有 b1536 顯著輸 IF-Prune（−1.2）
- 188 塊：TTFT 24×、記憶體 31× 低於 uncompressed；所有方法（含 uncompressed 56.5→49.0）accuracy 都下降，IF-Prune 52.0 > FoveaTile 47.0 > FastV 42.5，但 IF-Prune 要 7.0 s / 8.3 GB
- V* Bench：選中目標時 88.2%（高於 uncompressed 86.8%），沒選中 50.9%；失敗來自 prior 沒排進窗口

## 6. 要誠實寫的限制

- b1024 平均保留率低於 IF-Prune（91.7 vs 94.3）
- 單一小物件題落後（HR single、V* attribute）：低解析度 prior 漏掉小於一個 patch 的物件
- tile 數少（< 約 35 塊）時 TTFT 不比 FastV / VisionZip 快
- 4B / 14B 沿用 8B 超參數；prior / q_attn 的層號是在建立 held-out 流程前（InfoVQA val）選的

## 7. 時程

| 日期 | 工作 |
|---|---|
| 10/26–10/28 | GPU：HR-8K 188 塊 → random prior → 4B → 14B。寫作：Intro、Related Work |
| 10/29–10/30 | 4B / 14B 效率量測；重跑 `build_paper_results.py`；補第 4 章 `\TODO`；Conclusion |
| 10/31–11/04 | Abstract、Fig. 1 teaser、Supplementary、整篇順稿、頁數調整 |
| 11/05–11/08 | 校對、匿名化檢查（不出現 NYCU、GitHub、致謝）、提交 |

## 8. 尚未 commit 的改動（2026-10-26）

`foveatile_internvl.py`（層號換算、`--random_prior`）、`bench_metrics.py`、`run_test_suite.sh`（`MAX_NUM`）、`plot_budget_curves.py`、
`run_model_scale.sh`、`run_ablation_randprior.sh`、`paper/`、`docs/05`、`docs/06`、`results/paper/build_paper_results.py`、
`results/runs/hrbench_8k/`（800 題完整結果）。使用者要求時再 commit。
