# 06 CVPR 2027 投稿總規劃與待辦（2026-10-26 建立，最後更新 2026-11-09，之後的 session 從這份接續）

## 0. 一句話

**FoveaTile**（原名 SVMA）：*Look Before You Encode: Foveated Tile Streaming for Resolution-Independent VLM Inference*。
在 ViT 之前用 VLM 自己對低解析度覆蓋 tile 的 attention 決定要編碼哪些高解析度細節 tile，邊界 tile 再用 q_attn 重排，
逐 tile 串流寫 KV；ViT、KV、記憶體都是 O(K)，跟解析度 N 無關。訓練免、不用額外模型。凍結版本 v1.0（2026-10-17）。

- 投稿：CVPR 2027（官網 2026-11-02 查證，皆 AoE）：**摘要／論文註冊 11/10**（作者名單此後鎖定，OpenReview profile 須完整）、**全文 11/16**、**Supplementary 11/23**。正文限 8 頁含圖表（超過可不審退稿）；references 與必填的 LLM Usage Statement 可放額外頁。用 Overleaf 寫。
- 所有實驗（含 HiRED、同預算降解析度）已於 2026-11-09 跑完；待辦見 §7。

## 1. 論文各章狀態

| 章節 | 檔案 | 狀態 |
|---|---|---|
| Abstract | `paper/sec/0_abstract.tex` | 初稿完成 10/30（10/30 濃縮為 ≈170 字；數字同 intro contributions） |
| 1 Introduction | `paper/sec/1_intro.tex` | **完成（10/29）**：加了 V* 稀疏性證據（目標中位數佔 0.08% 面積、24 塊 tile 中只落在 1 塊）、問題重新定義句；Fig. 1 teaser 做過但 10/30 拿掉（和 fig_method、fig_scaling 重複；`make_teaser.py`、`fig_teaser.pdf` 保留備用），改引用 fig:method 與 fig:scaling |
| 2 Related Work | `paper/sec/2_related.tex` | **完成**：token pruning（FastV、SparseVLM、VisionZip、IF-Prune）、高解析度壓縮（HiRED、GlobalCom2）、zoom-in / search（ViCrop、V*、ZoomEye、DyFo、RAP）。HiRED 已是實驗 baseline；和 ViCrop、GlobalCom2 的性質比較見 §9 #9 |
| 3 Background and Motivation | `paper/sec/3_motivation.tex`（2026-11-05 新增） | **初稿完成**：3.1 成本（由 Method 移來，引用 `tab_hr8k` 數字）、3.2 How reliable is low-resolution attention?（凍結 v1.0 後的事後分析，圖 `fig_recall`、表 `tab_recall`、`tab_window`，由 `paper/figures/make_recall_figure.py` 產生）、3.3 Motivation + Problem setting。`CVPR27.tex` 已加 `\input` |
| 4 Method（舊第 3 章） | `paper/sec/4_method.tex` | **完成**（已核對程式碼；Fig. 2 overview、Fig. 3 boundary scoring、Alg. 1、Tab. 1 成本） |
| 5 Experiments（舊第 4 章） | `paper/sec/5_experiments.tex` | **完成**：5.1 Setup、5.2 記憶體預算下的準確率（含同預算降解析度、Model scale 段）、5.3 解析度縮放、5.4 Ablation；baseline 含 HiRED；同時請求數為實測（`bench_concurrency.py` → `results/concurrency/`、`results/concurrency_hired/`） |
| 6 Conclusion + Limitations | `paper/sec/6_conclusion.tex` | **完成**：Limitations 只寫 tile 獨立架構（LLaVA-OneVision 可套用、未評估；原生解析度編碼器不適用；使用者 2026-11-08 決定）；LLM Usage Statement 已寫 |
| Supplementary | `paper/sec/supp_{analysis,reliability,scale,downscale,ttft,calibration,baselines}.tex` | **初稿完成**：題型分析、Sec. 3.2 細節（`tab_recall_full`）、模型尺寸、同預算降解析度、TTFT 拆解、校準流程（含負面結果與偏離）、baseline 移植細節（含 HiRED）。截止 11/23 |

Overleaf 上傳清單：
- `preamble.tex` 要有：`algorithm`、`algpseudocode`（`[noend]`）、`xspace`、`\newcommand{\method}{FoveaTile\xspace}`（已加）；
  **還要加** `\usepackage{multirow}`、`\usepackage{colortbl}`、`\definecolor{oursbg}{HTML}{E8F0FC}`
- `main.bib` = `paper/main.bib`；Intro / Related 要用的額外引用在 `paper/refs_method.bib`（`streamingllm`、`ram`）
- `CVPR27.tex` 依序 `\input` sec/0~6，references 之後接 Supp（`supp_*`）
- 上傳 `paper/figures/*.pdf`、`paper/tables/*.tex`
- 篇幅：**2026-11-09 Overleaf 實測正文約 9.65 頁，超過 8 頁上限約 1.65 頁**（References 與 LLM Usage Statement 不計）；11/16 前必須刪減
- **2026-11-10 刪減（第一組 + 第二組 + A，使用者決定）**：刪減前完整版備份在 `paper_full_2026-11-10/`。Algorithm 1、Fig. boundary → 新的 Supp `sec/supp_method.tex`（S1）；tab_recall、tab_window → `supp_reliability.tex`；tab_downscale → `supp_downscale.tex`；主表拿掉第二行；§1–§6 正文共少 650 字。估計省約 1.65 頁，待 Overleaf 實測。CVPR27.tex 的 Supp 部分加了 figure / algorithm 的 S 編號。
- LLM Usage Statement：已寫在 `sec/6_conclusion.tex`（正文後、references 前，匿名）

## 2. 圖表怎麼產生

- 第 5 章與 Supp 的主要表格與圖：`python results/paper/build_paper_results.py` → `paper/tables/tab_{memcap,main,scale,hr8k,analysis,ablation}.tex`、
  `paper/figures/fig_{pareto,scaling}.pdf`、`results/paper/main_6ds.csv`。**實驗補完後重跑一次即可**，缺的格子會印紅色 `?`（`\pend`）
- 第 4 章（Method）的兩張圖：`python paper/figures/make_method_figures.py`（範例 = V* Bench direct_attributes 第 51 題，資料在
  `paper/figures/method_examples_vstar_b1536.json`，由 `paper/figures/extract_method_example.py` 產生）
- 第 3 章的圖表：`python paper/figures/make_recall_figure.py`（不需 GPU；原始資料 `results/diagnostics/vstar_selection/b1536_observation.json` 由 `experiments/diagnostics/vstar_observation_run.py` 產生）
- 同預算降解析度：`python results/paper/build_downscale_tables.py` → `tab_downscale{,_full}.tex`
- TTFT 拆解：`python results/paper/build_ttft_breakdown.py` → `tab_ttft.tex`
- HiRED 對照：`python results/paper/analyze_hired.py <B>`
- 呈現規劃：`docs/05_results_presentation_plan.md`

## 3. 實驗狀態

### 已完成（InternVL3.5-8B）
- 6 個測試集（DocVQA val、InfoVQA val、MME-RW Lite、HR-Bench 4K、HR-Bench 8K、V* Bench）× 6 個方法 × b1024~b3072（4K/8K 外加 b4096/b5120 部分）：`results/runs/<ds>/`
- 效率：`results/efficiency/`（b1536、b2560 在子資料夾；HR-8K 在 `results/efficiency/hrbench_8k/`）
- 解析度縮放（HR-8K 10 張圖，7~188 塊）：`results/scaling/summary.json`
- held-out 校準（m、S 掃描）：`results/calib/`

- HR-8K 188 塊（tile 上限 201）× 所有方法 × b1024~b3072：`results/runs_maxnum201/hrbench_8k/`
- random prior ablation：`results/calib/<ds>/*_randprior_*.json`
- InternVL3.5-4B / 14B × 6 測試集 × 6 方法 × b1536：`results/runs/<ds>/*_InternVL3_5-{4B,14B}.json`；效率 `results/efficiency/InternVL3_5-{4B,14B}/`
- 同時請求數實測：`results/concurrency/n<max_num>_<method>.csv`
- 同預算降解析度（Low-Res、Low-Res-Fill）：`results/runs_lowres/`
- HiRED（全部實驗）：`results/runs_hired/`、`results/runs_hired_maxnum201/`（b1536、b2560）、`results/efficiency_hired/`、`results/scaling_hired/`、`results/concurrency_hired/`；`build_paper_results.py` 會自動換到這些路徑，不要併進主結果資料夾

目前沒有背景工作在跑。GPU 跟別人共用，不要砍別人的程序（等，最多 24 小時）。GPU0 上有別人常駐的服務（約 2.5 GB），需要整張 GPU 全空的量測（`bench_concurrency.py`）要用 GPU1。

## 4. 已定的決策（不要再改）

- 方法凍結 v1.0：m = ⌊K/2⌋、S = {1024:5, 1536:4, 2048:4, 2560:3, 3072:3}、預設 b1536；v1.1 候選、3b、A1/A2 都依事先規則否決（`docs/04_calibration_protocol.md`）
- 換模型不重調 m、S；attention 層號依相對深度換算（`foveatile_internvl.attn_layers`：36 層 = (21–28, 20)，40 層 = (23–31, 22)），已驗證 8B 輸出逐題不變
- 資料集名稱寫 **V\* Bench**；HR-Bench 主設定 tile 上限 47（8K 約切 25 塊），188 塊是另外的縮放實驗
- 主表區塊放 b1536 / b2560 / b3072；b1024、b2048 放 Pareto 圖與附錄（所有方法同一組 budget）
- 主表 / 平均保留率 / 記憶體上限表 / Pareto / 模型尺寸表都用 **5 個測試集**（DocVQA、InfoVQA、MME-RW Lite、HR-Bench 4K、V* Bench）；HR-Bench 8K 另外放在 `tab_hr8k`（25 vs 188 塊）與分析表（使用者 2026-10-26 決定）
- 主表每個方法一行（2026-11-10 為頁數拿掉第二行的各資料集保留率，Avg. 仍是保留率；使用者決定）
- 第 5 章主軸 = 記憶體 / 延遲不隨解析度增長：章首總述；5.3 的縮放圖是單欄 2x2（TTFT、記憶體 + 每 tile 增量、每張 GPU 同時請求數、圖例）。同時請求數為實測：188 塊時 FoveaTile 124、FastV 37、IF-Prune 38、SparseVLM 20、uncompressed 3。VisionZip、HiRED 記憶體低於 1 GB，但 TTFT 隨 N 增長、accuracy 低 23.5 / 17.0 分
- 正文不放「Where FoveaTile helps and where it fails」分析節（使用者 2026-10-26 決定）；內容在 `paper/sec/supp_analysis.tex`。Limitations 只寫 tile 獨立架構（使用者 2026-11-08 決定）
- 主表的 ↑/↓ = FoveaTile 對該欄最強 baseline 的配對 bootstrap（雙向都標）
- 主結果先放 **Table「記憶體上限下的 accuracy」**（`tab_memcap.tex`）：≤0.5 / ≤1 / ≤2 GB（= uncompressed）/ 不限，FoveaTile 每一欄都第一；FastV、SparseVLM、IF-Prune 在 ≤1 GB 跑不了，IF-Prune 比 uncompressed 還吃記憶體（3.0 vs 2.0 GB）
- 主表成本欄改成相對 uncompressed 的節省倍率（TTFT / 記憶體 / FLOPs，越大越好；絕對值放附錄），最佳粗體、第二名底線
- Ablation 用 5 個 budget 的平均（單看 b1536 時「只有覆蓋」平均略高，跨 budget 才完整）
- 呈現原則：圖表用最能凸顯 FoveaTile 的方式（成本與 accuracy 並列、Pareto、縮放），但不刪資料集、不只挑對自己有利的 budget；正文只寫 FoveaTile 贏的地方，表格保持完整（使用者 2026-10-28 決定）
- 同預算降解析度（Low-Res、Low-Res-Fill）放 §5.2 當對照（使用者 2026-11-08 決定）；「只有覆蓋」同時留在 ablation；合併檢定不放
- 第二個模型家族不做，用 InternVL3.5 4B / 8B / 14B 代替

## 5. 目前主要數字（8B；平均 = 主表的 5 個測試集，不含 HR-8K）

- 平均保留率 FoveaTile / IF-Prune / FastV / SparseVLM / HiRED / VisionZip：b1536 **95.0** / 94.9 / 90.0 / 85.5 / 78.0 / 71.5；b2560 **97.6** / 97.1 / 95.5 / 91.2 / 85.8 / 83.0；b3072 **99.3** / 97.4 / 97.2 / 91.8 / 89.2 / 86.1
- b1536 記憶體節省（相對 uncompressed）：FoveaTile 5.0×、FastV 1.5×、IF-Prune 0.7×（比 uncompressed 還多）；TTFT 比 IF-Prune 快 1.6×
- 記憶體上限表：≤0.5 / ≤1 / ≤2 GB / 不限，FoveaTile 95.0 / 99.3 / 99.3 / 99.3，全部第一
- InfoVQA 從 b1536 起每個 budget 都顯著贏所有 baseline（對最強者 +2.1～+4.3）；MME-RW 在 b1536、b3072 顯著贏所有 baseline，且高於 uncompressed；對 HiRED，DocVQA、InfoVQA、MME-RW 五個 budget 全部顯著贏
- 188 塊：TTFT 比 uncompressed 低 24×、記憶體低 31×；FoveaTile 0.71 s / 0.41 GB 兩者都最低；accuracy 47.0 高於 FastV 42.5、HiRED 41.0、VisionZip 27.0；b2560 達 52.5，與最佳壓縮方法同分，高於 uncompressed 49.0；HiRED TTFT 2.35 s、記憶體 0.88 GB，FoveaTile 分別低 3.3×、2.1×
- V* Bench：選中目標時 88.2%，高於 uncompressed 86.8%

## 6. 已知限制（內部紀錄；論文正文只寫 Limitations 的架構段）

- b1024 平均保留率低於 IF-Prune（91.5 vs 94.5）
- 單一小物件題落後（HR single、V* attribute）：低解析度 prior 漏掉小於一個 patch 的物件
- tile 數少（< 約 40 塊）時 TTFT 不比 FastV / VisionZip / HiRED 快
- 4B / 14B 沿用 8B 超參數；prior / q_attn 的層號是在建立 held-out 流程前（InfoVQA val）選的

## 7. 時程（皆 AoE）

| 日期 | 工作 |
|---|---|
| 11/10（台灣 11/11 19:59） | OpenReview 論文註冊：作者名單定案、所有作者 profile 完整、標題與摘要 |
| 11/16 | 全文：正文刪減約 1.65 頁、PDF 改名（目前檔名含 JackJou）並檢查 metadata、匿名檢查、Overleaf 同步 |
| 11/23 | Supplementary：#5 層號掃描、#10 質性範例（是否做由使用者決定）；日期建議登入 OpenReview 再確認一次 |

## 8. Git 狀態（2026-11-09）

全部改動已由使用者 commit（`47359c7`）。之後使用者要求時再 commit。

## 9. 投稿前審查缺口（2026-11-07，以 CVPR 審查者角度檢查 PDF；**尚未決定要做哪些**，使用者看時間再決定）

時限：全文 11/16、Supplementary 11/23。正文超過 8 頁約 1.65 頁（2026-11-09 實測），正文要先刪才能加。GPU 共用，跑前先確認沒人在用。

### 第一級：很可能被審查者點名
| # | 缺失 | 審查者會問 | 建議做法 | 工作量 | 放哪 | 狀態 |
|---|---|---|---|---|---|---|
| 1 | 缺最相近 baseline：HiRED、GlobalCom2（`docs/03` 標為「必要」，checklist 2.1–2.3 未做） | 同樣用全圖資訊分配每塊 tile 的預算，為何不比？ | 至少補 HiRED；時間夠再補 GlobalCom2 | HiRED 約 1 天 + GPU；GlobalCom2 2–4 天 | Supp（11/23）或正文主表 | **HiRED 已完成並放進論文（2026-11-09）**，見下方「#1 HiRED」；GlobalCom2 未做（未決定） |
| 2 | 缺「直接降解析度到同預算」對照（**2026-11-07 開跑**，設計見下） | Ablation 顯示只用覆蓋 tile（m=K）平均打平且更快，選 tile 機制有何用？ | 測試集補 m=K，至少 HR-Bench 4K、V* Bench；論點聚焦「細節 tile 防止小物件題失敗」 | 低（現有程式 `--coverage_tiles`） | 正文 §5.2 + Supp | **已完成並放進論文（2026-11-08）**，結果見下方 #2 實驗設計之後的「結果」 |
| 3 | 只有一個模型家族 | 是否只適用 InternVL 的切 tile 架構？ | 不補實驗；在 Limitations 說明適用 tile 式 VLM，並討論原生解析度編碼器（Qwen2-VL）怎麼套用 | 只需寫作 | 正文 | **已處理（2026-11-08）**：§3.1 加前提句、結論 Limitations 寫 LLaVA-OV 可套用（未評估）、原生解析度不適用 |
| 4 | 缺 Limitations 段落（結論只有一句） | CVPR 審查表會看限制 | 獨立一段：多輪對話（KV 依問題選 tile，換問題要重選、無法共用 prefix cache）、極小物件 prior 漏掉、約 40 塊以下 TTFT 不比 FastV 快、attention 層號在測試集上選定 | 只需寫作 | 正文 | **已補（2026-11-08）**：結論加 \paragraph{Limitations}，只寫 tile 獨立架構（LLaVA-OV 可套用、未評估；Qwen2-VL 不適用）；多輪、極小物件、延遲三點使用者決定不寫 |

### #2 實驗設計（2026-11-07 開跑前寫下，結果出來前不改）
- **問題**：同樣的 visual token 預算下，「直接降解析度」能不能取代 FoveaTile 的細節 tile 選擇？
- **兩個對照（KV 都是 B 個 visual token，跟 FoveaTile 同預算）**：
  - `lowres`：官方動態切 tile，上限改成 K 塊（含縮圖）＝ InternVL 原生的低解析度用法（`--mode uncompressed --max_num K`，batch 1，同 uncompressed）
  - `coverage`：FoveaTile 的覆蓋 tile 拿全部預算（m = K，沒有細節 tile、沒有 prior / q_attn；＝ Low-Res-Fill，`--coverage_tiles K`，batch 4，同 FoveaTile）
- **資料集**：主表 5 個（DocVQA、InfoVQA、MME-RW Lite、HR-Bench 4K、V* Bench）＋ HR-Bench 8K（tile 上限 47）；全量。
- **Budget**：先 1536 / 2560 / 3072（主表），GPU 有空再 1024 / 2048。
- **主要檢定（事先指定）**：小物件資料集（HR-Bench 4K、HR-Bench 8K、V* Bench）上 FoveaTile vs 兩個對照，每個 budget 配對 bootstrap 95%（HR-Bench 用 CircularEval 組）。
- **次要**：5 個主表資料集的平均保留率；文件類（DocVQA、InfoVQA）預期對照可能打平或贏（SVMA 時期的觀察），照實報告。
- **呈現原則**：所有資料集與 budget 都列在表裡，不挑；正文文字只寫顯著的部分（同 paper-text-positive-only 規則）。是否放進論文由使用者看完結果決定。
- 程式：`run_lowres_baselines.sh`；結果 `results/runs_lowres/<ds>/{lowres,coverage}_b<B>_InternVL3_5-8B.json`；log `logs/runs_lowres/`。

### #2 結果（2026-11-08 跑完，36 個工作全數完成）
- 平均保留率（5 個主表資料集）Low-Res / Low-Res-Fill / FoveaTile：b1536 84.2 / 91.7 / **95.0**；b2560 92.9 / 94.1 / **97.6**；b3072 93.3 / 96.2 / **99.3**。小物件（HR-4K、HR-8K、V*）：73.5 / 82.6 / **91.9**；85.4 / 86.6 / **93.8**；85.9 / 90.3 / **95.5**。
- 主要檢定 vs Low-Res-Fill：小物件 9 格全為正（+2.5～+8.0），HR-8K b1536 顯著（+8.0 [+0.5, +15.0]）；MME-RW 三個 budget 都顯著贏（+2.2～+2.5）。
- vs Low-Res：b1536 六個資料集全部顯著贏；InfoVQA / MME-RW / V* 每個 budget 都顯著。Low-Res 用不滿預算（平均 1,149 / 1,931 / 1,972 個 token）；b2560 有 41/11,860 題多一個 tile。
- 論文：主文 §5.2 新段落 `sec:downscale` + `tab:downscale`；Supp `sec/supp_downscale.tex`（`tab:downscale_full`、`tab:downscale_delta`）；§2 第 32 行、§5.1 Baselines、§5.4 消融各加一句引用。表格由 `results/paper/build_downscale_tables.py` 產生。合併檢定（三個小物件資料集合併）使用者決定不放（事後分析）。

### #1 HiRED（2026-11-08）
- 移植：`hired_internvl.py`，照官方碼（github.com/hasanar1f/HiRED commit c5978a5）：alpha=0.5、第 0 層 CLS attention 的 top-100 patch 決定每塊 sub-image 的 budget、特徵輸出層 CLS attention 選 token。InternVL 逼出來的差異列在檔頭 (a)–(g)；論文 Eq.(1)（連續值總和）與官方碼（top-100 個數）不一致，照官方碼。
- `bench_metrics.py --run_hired`、`bench_concurrency.py --method hired`、`run_test_suite.sh` 的 `hired` 都已加上。
- 全部實驗由 `run_hired_all.sh` 依序跑（log `logs/runs_hired/all.log`）：A0 主表 5 資料集 × {1536, 2560, 3072} → A1 補 {1024, 2048} 與 HR-8K × 5 budget → A2 HR-8K 188 塊 → A3 4B / 14B → B1–B5 效率與縮放 → B6 同時常駐請求數。
- 結果放 `results/runs_hired/`、`results/runs_hired_maxnum201/`、`results/efficiency_hired/`、`results/scaling_hired/`、`results/concurrency_hired/`。比較：`python results/paper/analyze_hired.py <B>`。
- **已放進論文（2026-11-09，使用者決定）**：`build_paper_results.py` 的 METHODS 加 hired（結果仍在獨立資料夾，loader 自動換路徑）；tab_main / memcap / hr8k / scale / analysis、fig_pareto / fig_scaling 重新產生；S7 加 HiRED 段；§1 §3 §4 §5 supp_scale 文字已更新。
- 全量結果：平均保留率 B=1024…3072，FoveaTile 91.5 / 95.0 / 95.8 / 97.6 / 99.3，HiRED 72.3 / 78.0 / 82.5 / 85.8 / 89.2；FoveaTile 在 DocVQA、InfoVQA、MME-RW 五個 budget 都顯著贏；HR-4K、HR-8K 在 b1024 顯著贏；超過約 40 塊 tile 時 TTFT 也比較快。

### 第二級：補上會明顯加分
| # | 缺失 | 建議做法 | 工作量 | 放哪 | 狀態 |
|---|---|---|---|---|---|
| 5 | 層號在測試集（InfoVQA val）上選定 | 在校準集掃幾組層號，證明不敏感 | 約 1 天 GPU | Supp | 未決定 |
| 6 | 缺額外 forward 的成本拆解 | TTFT 拆成 ViT / prior / q_attn / prefill（`stats` 已有 `_t_detail_vit`、`_t_qattn`、`_t_commit`） | 低 | Supp | **已完成（2026-11-08）**：`results/paper/build_ttft_breakdown.py` → `paper/tables/tab_ttft.tex`，Supp `sec/supp_ttft.tex`（CVPR27.tex 接在 supp_downscale 後）；用既有效率量測，不另跑 GPU。§4 Cost 已引用 |
| 7 | 服務吞吐量只用估算（保留 KV、無 batching） | 誠實說明，或補簡單批次實驗（`docs/03` checklist 4.2） | 寫作或中等 | 正文文字 | 未決定 |
| 8 | 第 1 頁沒有 teaser | `paper/figures/make_teaser.py` 已有，但占篇幅 | 低（與頁數衝突） | 正文 | 未決定 |
| 9 | 和 ViCrop、GlobalCom2 的差異不夠清楚 | Related Work 加性質比較表：免訓練 / ViT 前 / 看問題 / KV O(K) / 需額外模型 | 低 | 正文 | 未決定 |
| 10 | 缺質性範例 | Supp 放成功與失敗的選擇視覺化 | 低 | Supp | 未決定 |

### 第三級：寫作風險
- L453 "highest average accuracy at every budget in Tab. 4"：Tab. 4 沒放 b1024，Fig. 4 可見 b1024 輸 IF-Prune（91.5 vs 94.5）→ 改成明列 b1536、b2560、b3072（主表的三個 budget；b2048 IF-Prune 96.0 略高於 95.8，所以不能寫「從 b1536 起」）。**已修（2026-11-09）**
- Abstract 說 six benchmarks，主表平均只用五個（HR-8K 另放）→ 第一次出現時說明。**已修（2026-11-08）**：摘要與 intro 改 five benchmarks、memory 句註明 HR-Bench 8K；§5.1 加一句平均只含前五個。
- 加一句會公開程式碼（審查期用匿名連結）。**使用者決定不加（2026-11-08）**

### 剩下的事
- 11/10：OpenReview 註冊。
- 11/16：正文刪減約 1.65 頁、PDF 改名、匿名檢查。
- 11/23（Supp）：#5 層號掃描、#10 質性範例。
- 未決定：#7、#8、#9、GlobalCom2。
