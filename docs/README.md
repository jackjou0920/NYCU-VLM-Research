# SVMA 投稿 ICML 準備文件

整理自 2026-10-13 的討論：SVMA 目前離 ICML 還差什麼、跟最接近的工作有什麼差異，以及接下來要依序完成的實驗。

| 檔案 | 內容 |
|---|---|
| [01_paper_readiness.md](01_paper_readiness.md) | 現況評估：目前結果、預設 budget 的建議、動機故事的缺口 |
| [02_related_work_comparison.md](02_related_work_comparison.md) | 跟 HiRED、GlobalCom2、RAP 的比較與 SVMA 的新穎性 |
| [03_experiment_plan.md](03_experiment_plan.md) | 實驗規劃：baseline 組合、公平比較的規則、論文圖表，以及**逐步執行的 checklist** |
| [04_calibration_protocol.md](04_calibration_protocol.md) | SVMA 超參數的 held-out 校準流程（凍結版本 v1.0 的來源）：校準集建立、階段 A / B、預設 budget、3a 診斷、3b 未採用、偏離事先規則的紀錄、重現指令；寫論文 Method / Appendix 用 |

使用方式：照 [03_experiment_plan.md](03_experiment_plan.md) 第 5 節的 checklist 由上往下做，每完成一項就打勾，並把結果檔路徑填在該項後面。

注意：01 裡的數字大多是 DocVQA / InfoVQA 前 500 題、HR-Bench 200 組的子集結果，論文要用的最終數字必須依 03 的規則重跑完整 val 集。
