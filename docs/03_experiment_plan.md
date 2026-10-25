# 03 實驗規劃與執行 checklist

## 1. Baseline 組合

現有的 FastV、SparseVLM、VisionZip、IF-Prune 都是通用的 token pruning，沒有一個針對「高解析度 + tile 切割」設計。FoveaTile 主打高解析度，必須跟同類方法比。

| 等級 | 方法 | 備註 |
|---|---|---|
| 必要 | Uncompressed | 上限參考 |
| 必要 | Low-Res-Fill | 已證明是最強的簡單 baseline，不放一定會被問 |
| 必要 | FastV、SparseVLM、VisionZip、IF-Prune | 已有；SparseVLM、VisionZip 要先修（見 01 第 2 節） |
| 必要 | **GlobalCom2** | 移植到 InternVL3.5，優先度最高 |
| 必要 | **HiRED** | 移植到 InternVL3.5，跟 GlobalCom2 共用 CLS attention 程式碼 |
| 建議 | GlobalCom2+FastV | GlobalCom2 論文自己提出的看問題版本 |
| 建議 | RAP-fixed-K | VisRAG 挑 crop + Spatial Layout，拿掉 RE-Search，K 跟 FoveaTile 一樣 |
| 選做 | 完整 RAP | 只跑 HR-Bench 4K 和 V\*，當 accuracy 上限參考並列出 latency |
| 選做 | VisPruner 或 DART | 代表較新的 token pruning 方法 |

### 移植注意事項

- **GlobalCom2**（預估 2 到 4 天）：InternVL 的動態 tile + thumbnail 跟 LLaVA-NeXT 的 anyres 幾乎同構。InternViT 有 CLS token（InternVL 在 ViT 輸出後丟掉），可以取最後一層 CLS 對 patch 的 attention。pixel shuffle 會把 1024 個 patch 併成 256 個 token，CLS attention 要按 2×2 加總。原論文設定：crop 豐富度 = thumbnail 對應區域的 CLS attention 加總；保留比例 r_j = R × (1 + σ_j − 1/n)，σ 為 softmax((s − max)/τ)，τ=10；crop 內 token 分數 = 0.5 × 內插後的全域分數 + 0.5 × 局部 CLS 分數。
- **HiRED**（約 1 天）：依 CLS attention 分配每個 partition 的 budget，再於 partition 內取 top token。
- **RAP-fixed-K**：用 VisRAG 算問題與每個 tile 的相似度，取 top-K，用 Spatial-Awareness Layout 重拼（刪掉全空的行列）後送進 InternVL。
- 介面沿用 `fastv_internvl.py` / `ifprune_internvl.py` 的輸入輸出格式，新增 `run_globalcom2.sh`、`run_hired.sh`。

## 2. 公平比較的規則（先定好再跑）

**不要只讓 FoveaTile 在每個 dataset 挑最好的 budget。** baseline 固定 budget、FoveaTile 卻挑最好的，等於在測試集上調參。

- **budget 的定義**：decode 時留在 LLM KV cache 裡的 vision token 數。FastV、IF-Prune 在前幾層保留全部 token，要另外列 LLM FLOPs 或「實際處理的 token 數」。
- **同一套環境**：同一張 GPU、同一個 attention 實作。FastV、SparseVLM 需要 attention weight，若被迫用 eager attention 要註明，或改成只算部分 attention，否則 TTFT 會被高估。
- **量測方式**：先 warmup，同一批圖取中位數；記憶體列「扣掉模型權重後的增量」。
- **評測流程**：所有方法走同一條評測流程（建議 VLMEvalKit 或 lmms-eval 的 protocol），全部跑完整 val 集，附 bootstrap CI 和配對檢定。
- **超參數只在 held-out 資料上決定**：S、m、預設 budget 用 DocVQA / InfoVQA 的 train split 或 ChartQA val 校準，定好後所有測試集共用，不再改。

## 2.5 資料切分（2026-10-15 確認）

| 用途 | 資料集 | 題數 | 評分 |
|---|---|---|---|
| 校準（只用來決定 FoveaTile 超參數） | `infovqa_train_calib`：InfoVQA train，seed 0 抽樣 | 600 | ANLS |
| 校準 | `mme_rw_calib`：MME-RealWorld 完整版扣掉 Lite 的題目與圖片，5 個 subtask 各 120 題 | 600 | 選擇題 accuracy |
| 測試 | DocVQA val | 5,349 | ANLS |
| 測試 | InfoVQA val | 2,801 | ANLS |
| 測試 | HR-Bench 4K | 800 題（200 組） | CircularEval |
| 測試 | `mme_rw_lite`：MME-RealWorld Lite | 1,919 | 選擇題 accuracy |
| 測試 | `vstar`：V* Bench（direct_attributes 115 + relative_position 76） | 191 | 選擇題 accuracy |
| 測試（待定） | HR-Bench 8K | 800 題（200 組） | CircularEval |

- 規則：測試集只報告結果，不參與任何選擇；看測試結果只能做診斷，要改 FoveaTile 預設必須回到校準集驗證，通過後再把所有測試集重跑一次
- DocVQA 沒有帶答案的 train split，文件類由 InfoVQA train 代表（論文要註明）

## 2.6 FoveaTile 凍結版本 v1.0（2026-10-17，使用者確認）

之後所有主實驗（測試集、效率量測、ablation 的對照組）都用這一版。設定集中在 `foveatile_config.py`、`foveatile_internvl.py` 檔頭常數、`internvl_preprocess.py`。

| 項目 | 設定 | 位置 | 依據 |
|---|---|---|---|
| 覆蓋 tile 數 m | ⌊K/2⌋（K = budget // 256） | `foveatile_config.COVERAGE_FRAC = 0.5` | 階段 A（held-out）：m/K 在 0.5~1.0 平均打平，0.5 最差任務最穩；m=1 平均少約 5 分 |
| 打分窗口 S | {1024: 5, 1536: 4, 2048: 4, 2560: 3, 3072: 3}；b3072 以上外推為 3 | `foveatile_config.QATTN_TILES_BY_BUDGET` | 階段 B（held-out）：S 超過 S0/2 沒有可靠增益，一律取 S0/2（偏離事先規則，使用者同意，附錄註明） |
| 邊界分配 | a = min(K−m, ⌊S/2⌋)；prior 前 (K−m−a) 塊直接保留，接下來 S 塊跑 q_attn 取 a 塊，其餘不跑 ViT | `foveatile_config.split_slots` | 沿用 |
| 預設 budget | 1536 | `foveatile_config.DEFAULT_BUDGET` | held-out 保留率轉折點（b1024 90% → b1536 96%，b2048 沒有再進步） |
| prior | 覆蓋 tile 跑一次 LLM 短 forward，第 21~28 層 text_after 對覆蓋 patch 的 attention 平均 | `THUMB_ATTN_LAYERS = (21, 28)` | 沿用（InfoVQA 全量驗證過） |
| q_attn | 第 20 層；context = [text_before][覆蓋 tile][候選 tile][text_after]，前綴 KV 重用 prior 那次 forward | `QATTN_LAYER = 20` | 沿用；held-out S=0 對照：Remote Sensing −10.8（顯著） |
| 覆蓋網格 | `fill_grid`，用滿 m 塊、長寬比拉伸 ≤ 1.75 | `COVER_MAX_STRETCH = 1.75` | 沿用（1.5 時直式文件切不滿） |
| 細節網格 | 官方 dynamic tiling，上限 47 塊（`--max_num 48` 含縮圖） | `internvl_preprocess.dynamic_grid` | 官方預設；native grid 在 8K 變差，已移除 |
| 寫 KV 順序 | 選中的細節 tile 依空間順序在前，m 塊覆蓋 tile 在最後 | `stream_image_to_kv_cascade` | 同訓練時「網格 tile → 縮圖」 |
| 系統參數 | `VIT_BATCH = 4`、`CHUNK_SIZE = 1024` | `foveatile_internvl.py` | 只影響 TTFT / 記憶體，不影響輸出 |
| 掃描用選項（不屬於預設） | `--coverage_tiles` / `--qattn_tiles`（m、S 掃描用）；沒通過的候選（3b、v1.1、A1/A2）已移除，含候選的版本在 `experiments/candidates_rejected/` | `foveatile_internvl.py` | — |

**修改規則**：
- 只有兩種情況考慮改方法：一般場景資料集暴露明顯錯誤，或第二個模型（LLaVA-OV）上行為異常
- 任何改動都必須先在 held-out 校準集（`infovqa_train_calib`、`mme_rw_calib`）上用事先定好的標準驗證，通過後版本號 +0.1，並重跑所有測試集；論文只報同一版本的數字
- 純工程加速（不改變輸出，例如 KV 寫入、ViT batch）不受此限，但要驗證輸出逐題相同
- 換模型時方法與流程不變，m / S 用同一套 held-out 流程重新校準，並在論文說明

**已知限制（論文要寫）**：
- b1024 在 DocVQA（−2.81）、InfoVQA（−2.98）、V*（−6.81）顯著輸 IF-Prune；3a 診斷：主因是 prior 沒把目標 tile 排進打分窗口（V* b1024 26%），q_attn 本身精準（92~94%）
- 擴大打分窗口（3b）在 V* 有幫助、在 held-out 沒有穩定增益 → 沒採用，列為未來方向（改進 prior 的召回率）
- 純覆蓋（m=K）在 held-out 平均打平、TTFT 約一半；細節選擇的價值在局部細節任務與最差任務的穩定度
- b4096 / b5120 的 S 是外推值
- HR-Bench 4K（200 組）、V*（191 題）太小，方法間差距大多不顯著

**v1.0 的測試集結果**：見本檔階段 1.2（baseline）、1.8（FoveaTile 與 HR 類測試集），結果檔在 `results/runs/<dataset>/foveatile_b<B>_InternVL3_5-8B.json`

## 3. 三種比較方式

| 方式 | 回答的問題 | 做法 |
|---|---|---|
| **A. 固定 budget**（主表） | 一樣多的 token，誰比較準？ | 所有方法用同樣的 budget（1024、2048、3072），比 accuracy 並附 CI |
| **B. Pareto 曲線**（主圖） | 整體 accuracy 和成本取捨誰比較好？ | 每個方法掃多個 budget，畫 accuracy 對 TTFT、accuracy 對峰值記憶體 |
| **C. 固定 accuracy**（效率表） | 達到同樣 accuracy，誰比較省？ | 例如「達到 uncompressed 95% 時，各方法需要多少 TTFT 和記憶體」 |

**FoveaTile 預設 budget 的放法：**

- 預設值（目前候選 b2048，K=8、m=6、S=4）在 held-out 校準資料上決定，所有測試集共用，論文寫成「FoveaTile（default）」，在 Pareto 圖上標成一個點。
- 若用逐圖規則 `budget = min(2048, 原生 token 數)`，baseline 也要用同樣的平均 token 數比，並列出每個 dataset 實際用了多少 token。
- 不要宣稱 FoveaTile 會「自動選 budget」：已驗證 answer confidence 和 prior 都無法逐題選出最好的 m。只說「有一個在 held-out 資料上校準的固定預設值」。

## 4. 論文圖表規劃

| 編號 | 內容 |
|---|---|
| 圖 1（teaser） | 多個 benchmark 平均的 accuracy 對 TTFT Pareto 圖，所有方法畫在一起 |
| 表 1 | 主表：2 個模型（InternVL3.5-8B + LLaVA-OV）× 約 8 個 benchmark × b1024 / b2048 的 accuracy |
| 表 2 | 效率：預設 budget 下的 TTFT（拆成 ViT、prior、打分、prefill）、峰值記憶體增量、vision FLOPs、LLM FLOPs、throughput |
| 圖 2 | 解析度縮放：tile 數 12 到 200 時各方法的峰值記憶體和 TTFT（核心主張靠這張） |
| 圖 3 | 固定 GPU 記憶體（例如 24 GB 或 48 GB）下的最大 batch 和 throughput |
| 表 3 | HR 類專表：HR-Bench 4K/8K、V\*、MME-RealWorld，加入 RAP 與 RAP-fixed-K，附 latency |
| 表 4 | Ablation：prior 來源（CLS 式、thumbnail 式、q_attn）、整塊 tile vs. 挑 token、m、S、串流 vs. 一次 prefill、只有覆蓋 / 只有細節 / 兩者都有 |
| Appendix | 負面結果：逐題選 m 做不到、confidence cascade 沒用、native grid 在 8K 反而變差；prior 與 q_attn 排名一致性分析 |

Benchmark 清單：

- 高解析度：HR-Bench 4K/8K（完整）、V\* Bench、MME-RealWorld(-lite)
- 文字密集：DocVQA、InfoVQA、ChartQA、TextVQA、OCRBench
- 一般場景：MMBench、MME、GQA、POPE（或 MMStar）
- 加分項：Video-MME、MLVU

## 5. 執行 checklist

完成一項就打勾，並在後面填上結果檔路徑。

### 階段 1：修好基礎（約 1 到 2 週）

- [x] 1.1 統一評測流程（2026-10-13）
  - 91.4 vs 92.0 的原因：FastV / SparseVLM / VisionZip 的舊 summary 開了答案抽取，IF-Prune / FoveaTile 沒開。統一用 `evaluate_anls.py`（不抽取）重新評分後 uncompressed DocVQA = 91.4（前 500 題 91.9），其他方法差 ±0.6，只需重新評分、不需重跑
  - 手動 prefill 路徑會省略 `<img>` / `</img>`：用 FastV 路徑不壓縮跟 batch_chat 比（各 100 題），DocVQA 95.8 vs 95.8、InfoVQA 79.4 vs 79.4，95~99% 答案逐字相同，影響可忽略，不改 prompt 組法
  - 原則：baseline 照論文與官方原始碼實作，不替其他方法加改良；只修「非官方、且偏離官方行為」的部分，並在論文附錄註明
- [x] 1.2 SparseVLM（2026-10-13，使用者同意保留；結果待重跑）
  - bug：repo 自己加的最後一步「裁切到 budget」照位置取前 N 個 token（等於丟掉圖片後段），官方 score.py 一律 `topk` 依分數挑、沒有照位置截斷。改成依分數裁切（`sparsevlm_core.py`）。InfoVQA b4096 λ=0.005 前 100 題：62.0 → 75.8（uncompressed 79.4）
  - λ 校準：原圖 token 不超過門檻的圖不算進中位數；校準張數 8 → 32（`sparsevlm_internvl.py`、`experiments/legacy_runs/run_sparsevlm.sh`）
  - 附錄寫法：官方固定 token 表只支援 LLaVA 576 token，我們用 Eq.7 決定每層剔除數，最後依分數裁切到 budget
  - [x] 重跑（2026-10-14 起）：DocVQA / InfoVQA / HR-4K × b1024~b5120，`RUN_STANDARD=0`（不重跑 batch_chat，評分用 meta 的 GT）、`RUN_METRICS=0`（效率之後用統一的 bench_metrics 量）；輸出 `results/archive/baselines_v2/internvl_sparsevlm_<ds>/`，log `logs/baselines_v2/`（舊結果保留在 `results/archive/baselines/`）
    - GPU0：DocVQA 拆兩個程序（b1024/2048/3072、b4096/5120）；GPU1：InfoVQA → HR-4K。DocVQA 單程序每分鐘約 20 題，預估全部 15~20 小時
    - 中途重啟造成 DocVQA b1024 答案錯位（多 112 筆）：已修正檔案（原檔 `.misaligned_bak`）並重新評分；`sparsevlm / fastv / ifprune_internvl.py` 的迴圈加上逐題續跑（只動續跑邏輯）；掃過所有舊結果，沒有其他錯位
    - **budget 規則（2026-10-14，使用者同意，論文附錄要註明）**：原圖 token ≤ budget 時 SparseVLM 不剪枝（`sparsevlm_internvl.budget_lam`，λ = 0），跟其他方法的 min(budget, raw) 一致。原因：Eq.7 的剔除數跟 budget 無關，固定 λ 會把小圖剪到遠低於 budget（DocVQA b4096：不需壓縮的 2,287 題 91.0 → 75.9；需要壓縮的 3,062 題 76.1 vs 74.4 差不多）
      - 規則加入前跑的結果：`experiments/legacy_runs/sparsevlm_small_images.py run` 對 token ≤ 5120 的圖跑一次 λ=0，再 `patch` 換掉每個 budget 裡 raw ≤ budget 的題目（原檔備份 `.pre_budget_rule.json`）
      - `experiments/legacy_runs/finalize_sparsevlm.sh` 在背景等所有 SparseVLM 程序結束後自動 patch + 重新評分，最後印出總表（log：`logs/baselines_v2/finalize.log`）
    - 2026-10-15 最終結果（修正 + budget 規則，完整資料集；b1024 / 2048 / 3072 / 4096 / 5120）：
      - InfoVQA（ANLS）：**59.1 / 69.1 / 72.5 / 74.9 / 75.5**（舊 50.3 / 56.6 / 65.1 / 59.5 / 66.8；uncompressed 78.7），單調；b1024 / b2048 贏 FastV（51.7 / 66.2），低於 IF-Prune（70.7 / 73.4）
      - HR-Bench 4K（CircularEval，200 組）：**60.0 / 58.0 / 57.0 / 62.0 / 62.0**（舊 60.0 / 55.0 / 57.5 / 58.5 / 62.0；uncompressed 62.0）；仍不單調但都在 200 組的雜訊（約 ±7）內
      - DocVQA：b1024 55.2（λ=0.567）、b4096 / b5120 待 patch；程序 A 在 b2048 第 5,189 題被系統 OOM 砍掉（同時 3 個 DocVQA 程序各自把全部圖片載進 RAM），b3072 沒跑 → 2026-10-15 09:30 重跑（GPU0 b2048 續跑、GPU1 b3072），結束後 `experiments/legacy_runs/finalize_sparsevlm.sh` 自動重新 patch + 評分（log：`logs/baselines_v2/finalize2.log`）
      - 注意：同一台機器不要同時開 3 個以上載入 DocVQA 全量圖片的程序（251 GB RAM）
    - [x] 2026-10-15 22:50 全部完成（patch + 重新評分）。DocVQA（ANLS）：**55.18 / 69.46 / 69.77 / 81.99 / 82.13**（uncompressed 91.41）；InfoVQA / HR-4K 同上
- 測試集上的 FoveaTile 新預設（m = K/2、S 表 {5,4,4,3,3}；`run_foveatile.sh`，結果 `results/runs/<ds>/foveatile_b<B>_*.json`，2026-10-15）
  - InfoVQA：67.75 / 77.18 / 79.36 / 79.60 / 80.17（舊 m=1：61.28 / 65.41 / 69.75 / 72.72 / 73.89；uncompressed 78.73）。b2048 起顯著贏 IF-Prune（+3.0~+4.3）、FastV、SparseVLM；b3072 起跟 uncompressed 打平或略高（b5120 +1.44 顯著）。b1024 輸 IF-Prune −2.98（顯著）
  - DocVQA：b1024 84.06、b2048 89.01（舊 74.01 / 82.24）；b1024 輸 IF-Prune −2.81（顯著）、贏 FastV +10.59；b2048 跟 IF-Prune 打平（−0.28 n.s.）、贏 FastV +1.04（顯著）。b3072~5120 執行中
  - HR-4K：58.00 / 57.00 / 59.50 / 59.50 / 57.50（舊 56.50 / 55.50 / 55.50 / 59.50 / 60.50）；跟所有方法的差距都不顯著（200 組，CI 約 ±5~6）
- [x] 1.3 VisionZip（2026-10-13，演算法未改；只加診斷選項 `--score_mode`、`--score_layer`，預設輸出跟舊結果 200/200 相同）
  - b1024 前 200 題對照組（ANLS）：DocVQA 官方 CLS 30.5 / random 25.5 / uniform 19.2 / CLS 第 17 層 39.5 / 第 11 層 37.3；InfoVQA 官方 40.7 / random 39.5 / uniform 35.9 / 第 17 層 39.6 / 第 11 層 39.4
  - 結論：實作行為合理（贏過 random、uniform），分數低是方法本身（不看問題、每塊 tile 平均分到約 39 個 token）加上 InternViT 最後一層 CLS 打分弱。主實驗維持官方最後一層，這組數字放附錄當「實作正確」的證據
  - 結果檔：`results/diagnostics/visionzip/`
- [~] 1.4 held-out 校準集：已加 `infovqa_train_calib`（InfoVQA train，seed 0 抽 600 題，`foveatile_internvl.py --dataset infovqa_train_calib`）。DocVQA 沒有帶答案的 train split
  - HR 類校準集（2026-10-13，使用者同意）：`mme_rw_calib` = MME-RealWorld 完整版（英文 23,609 題）扣掉 Lite 的題目、以及圖片出現在 Lite 的題目，5 個 Subtask 各抽 120 題（seed 0）共 600 題。測試集改用 `mme_rw_lite`（1,919 題），校準 / 測試連圖片都不重疊
  - 準備：`python prepare_mme_realworld.py`（只從 tar 分割檔抽出校準集用到的圖 + 解壓 Lite，放 `~/datasets/MME-RealWorld`，可用 `MME_RW_ROOT` 改）；loader `process_common.load_mme_realworld`（圖片延遲開檔），prompt 用官方作答指示，評分 = 選擇題 accuracy（`summarize_results.py` 的 `mc_accuracy`）
- [x] 1.5 校準腳本 `run_calibrate.sh`（階段 A 掃 m、階段 B 掃 S）；`foveatile_internvl.py` 新增 `--coverage_tiles`、`--qattn_tiles`。2026-10-13 22:29 起跑階段 A（InfoVQA train，b1024~b3072，23 組，log：`logs/calib/stageA_m.log`）；23:20 起跑 MME-RealWorld 校準集的階段 A（23 組，log：`logs/calib/stageA_m_mme_rw.log`）。結果都在 `results/calib/<dataset>/`，看表：`python summarize_results.py --root results/calib`
  - [ ] 待做：baseline 腳本（fastv / sparsevlm / visionzip / ifprune 的 `--dataset`）還不支援 `mme_rw_lite`，測試階段要補（沿用 `process_common.load_mme_realworld` + `get_image`）
  - 原項目：在 held-out 上重新校準 `foveatile_config.py` 的 S 表與 `COVERAGE_FRAC`，並補掃 multi-scale 的 b1536、b2560、b3072
  - 階段 A 結果（2026-10-14，兩個校準集各 600 題，分數 = 兩者平均；InfoVQA 用 ANLS、MME-RW 用 accuracy）：

    | m 規則 | b1024 | b1536 | b2048 | b2560 | b3072 | 平均 |
    |---|---|---|---|---|---|---|
    | m=1（v0，只有官方縮圖） | 57.9 | 61.2 | 63.9 | 65.2 | 68.4 | 63.3 |
    | m/K=0.5 | 63.9 | 68.4 | 68.8 | 70.7 | 71.0 | 68.5 |
    | m/K=0.75（目前預設） | 65.2 | 67.6 | 68.3 | 70.0 | 70.9 | 68.4 |
    | m=K（純覆蓋，沒有細節 tile） | 63.5 | 68.8 | 69.1 | 70.1 | 71.8 | 68.6 |

    - m=1 在每個 budget 都差約 5 分，multi-scale 在 held-out 上成立
    - m/K 在 0.5~1.0 是平台，配對 CI 幾乎都包含 0；唯一顯著的是 InfoVQA b2048 純覆蓋 +3.0
    - **風險**：純覆蓋（≈ Low-Res-Fill）平均打平最佳，代表細節 tile 選擇在平均上沒有加分；加分只出現在特定任務：MME-RW 的 Remote Sensing / Monitoring / OCR 偏好小 m（b3072 Remote Sensing：m=3 57.5 vs m=12 42.5），Diagram and Table、InfoVQA 偏好大 m
    - 結果檔：`results/calib/`；彙整：scratchpad 的 calibA.json（可用 `summarize_results.py --root results/calib` 重算）
  - 保留率與 TTFT（2026-10-14；accuracy 600 題，TTFT 中位數 / 記憶體 50 張圖、每張卡只跑一個程序、第 1 張 warmup 不計）
    - uncompressed：InfoVQA train 83.9（TTFT 1.21 s，記憶體增量 1.60 GB）；MME-RW calib 59.5（TTFT 2.81 s，3.20 GB）
    - 兩個校準集的平均保留率 / 平均 TTFT（m/K = 0.5 | 0.75 | 1.0）：

      | budget | m/K=0.5 | m/K=0.75 | m/K=1.0（純覆蓋） |
      |---|---|---|---|
      | b1024 | 89.7% / 0.90 s | 91.2% / 0.69 s | 88.5% / 0.40 s |
      | b1536 | 95.7% / 0.99 s | 94.3% / 0.89 s | 96.2% / 0.45 s |
      | b2048 | 95.9% / 1.08 s | 95.5% / 1.00 s | 96.0% / 0.56 s |
      | b2560 | 98.9% / 1.19 s | 97.9% / 1.15 s | 97.7% / 0.69 s |
      | b3072 | 99.3% / 1.41 s | 99.1% / 1.32 s | 100.1% / 0.80 s |

    - 記憶體增量（扣掉 17.06 GB 權重）全部在 0.28 ~ 0.75 GB
    - 最差任務（6 組：InfoVQA + MME-RW 5 個 subtask，每組 120~600 題）的保留率，m/K=0.5 最穩：b2048 85% vs 純覆蓋 74%；b2560 94% vs 69%。最差的都是 Remote Sensing
    - **關鍵發現**：純覆蓋的平均保留率跟有細節選擇的一樣，TTFT 只有一半左右；細節選擇（prior + q_attn）約多 0.5 s，換到的是最差任務的穩定度，不是平均分數
    - b2048 相對 b1536 沒有進步（95.9% vs 95.7%）；轉折點在 b1536
    - 結果檔：`results/calib/metrics/`、`results/calib/*/uncompressed_InternVL3_5-8B.json`；`bench_metrics.py` 新增 `--foveatile_m_fracs`、`--warmup_images`、`ttft_s_median`
  - [x] 預設 m/K 與 budget 已決定（見 1.6）
  - [x] 階段 B（S）：2026-10-14 08:54~15:15，`run_calibrate_stageB.sh`（m = K/2；S ∈ {0, S0/2, S0, 1.5 S0, 2 S0}，S0 = 舊規則的值；兩個校準集 × 5 個 budget，accuracy 600 題、TTFT 50 張圖）
    - 事先定的規則：平均保留率與最差任務保留率都在最佳值 1 個百分點內的 S 中取最小 → 選出 {1024: 5, 1536: 13, 2048: 4, 2560: 3, 3072: 14}
    - 問題：最差任務每組只有 120 題（1 題 ≈ 1.5 個百分點），最差的是哪一組也會隨 S 改變，規則被雜訊主導；配對 bootstrap 顯示 b1536 S=13、b3072 S=14 跟 S0/2 的差距都不顯著
    - **決定（使用者同意，偏離事先規則，論文附錄要註明）：一律 S = S0/2 → `QATTN_TILES_BY_BUDGET = {1024: 5, 1536: 4, 2048: 4, 2560: 3, 3072: 3}`**（`foveatile_config.py`；舊表留為 `LEGACY_QATTN_TILES_M1`）
    - 預設點 b1536（m=3、S=4）：平均保留率 95.6%、TTFT 0.77 s（S0=9 時 0.99 s）
    - **論文可用的 ablation（held-out，顯著）**：S=0（只靠 prior、不做 q_attn）比 S0/2 差：Remote Sensing b1024 −10.8 [−18, −4]、b1536 −10.8 [−18, −4]；Monitoring b2560 −7.5 [−12, −3] → 依問題打分在局部細節任務上有實質貢獻
    - S 超過 S0/2 沒有可靠增益（CI 大多包含 0），只多花 TTFT
    - 驗證：新預設跑 8 題，跟階段 B 的 S=4 結果逐題相同
    - 結果檔：`results/calib/*/foveatile_b*_m*_S*.json`、`results/calib/metrics/metrics_S_*.csv`
- [x] 1.6 預設操作點（2026-10-14，使用者選方案 A）：**b1536、m/K = 0.5、S = 4**（`foveatile_config.COVERAGE_FRAC = 0.5`、`DEFAULT_BUDGET = 1536`，`foveatile_internvl.py --budget` 預設跟著改）
  - 依據：held-out 平均保留率 96%（b1024 90%、b2048 96% 沒有再進步），最差任務 85%，TTFT 0.99 s（uncompressed 1.21 / 2.81 s），記憶體增量 0.43 GB（uncompressed 1.6 / 3.2 GB）。第二個操作點 b2560（平均 99%、最差 94%、1.19 s）
  - 論文定位（方案 A）：主打「最差任務也穩、成本有上限」，純覆蓋（m=K）放 ablation 誠實呈現：平均保留率相同、TTFT 約一半，但最差任務（Remote Sensing）差很多
  - 主表的固定 budget（1024 / 2048 / 3072）照舊，b1536 是在 Pareto 圖上標出的預設點
- [x] 1.7 統一量測腳本（2026-10-14，`bench_metrics.py` + 新增 `flops.py`；baseline 檔案未改）
  - `--warmup_images`（預設 1，前幾張不計）、`ttft_s_median`、`mem_delta_gb` = peak_alloc − 模型權重（IF-Prune 含小 VLM 權重，論文要註明）
  - FoveaTile TTFT 分段（`foveatile_internvl.py` 的 stats["ttft_breakdown"]）：text_prefill / cover_vit / prior / detail_vit / qattn / kv_commit / first_token → CSV 的 `t_*` 欄
  - 解析 FLOPs（TTFT 階段 = ViT + 到第一個 token 的 prefill）：每層 2·n·P_lin + 4·n·c·H·d，P_lin 直接讀模型參數；每個方法依實際處理的 token 數（FastV / IF-Prune 剪枝前後、SparseVLM 依 stage_logs、FoveaTile 含 prior / q_attn / 寫 KV）→ `vit_tflops` / `llm_tflops` / `aux_tflops`（IF-Prune 小 VLM）/ `total_tflops`
  - `--foveatile_m_fracs`、`--foveatile_s_mults`：一次量多組 m / S
  - smoke test（MME-RW 3 張、b1536、GPU 有其他程序，TTFT 不可信）：total TFLOPs uncompressed 242 / FoveaTile 53 / VisionZip 59 / FastV 69 / IF-Prune 117 / SparseVLM 135；FoveaTile 的 ViT 5.9 vs 其他方法 34 TFLOPs（被丟掉的 tile 不跑 ViT）

### 階段 1.8：補 HR 類測試集（2026-10-15 規劃，使用者同意）

- [x] 1.8.1 FoveaTile 新預設跑完 DocVQA（b3072 / b4096 在 GPU0，b5120 在 GPU1）
- [x] 1.8.2 MME-RW Lite + V* × {uncompressed, FoveaTile, FastV, VisionZip, IF-Prune, SparseVLM} × b1024~b5120：`run_test_suite.sh`（結果 `results/runs/<ds>/`，log `logs/test_suite.log`）；DocVQA 跑完後由監看程序自動啟動（`logs/step2_launcher.log`）
  - 為此：`process_common.load_vstar`（資料在 `~/datasets/vstar_bench`）、4 支 baseline 腳本的 `--dataset` 加上 `mme_rw_lite` / `vstar` 並改用 `get_image` 延遲載圖（只動資料載入，不動方法）
  - smoke test：兩個資料集各 2 題、6 個方法都跑通
  - [x] 2026-10-16 16:24 完成（26 組 × 2 個資料集，沒有失敗）。結果（b1024 / 2048 / 3072 / 4096 / 5120）：
    - MME-RW Lite（1,919 題，uncompressed 51.59）：FoveaTile 48.78 / 51.22 / 53.47 / 52.42 / 53.41；IF-Prune 48.93 / 49.45 / 49.77 / 50.44 / 51.22；FastV 45.54 / 48.20 / 49.66 / 51.17 / 52.01；SparseVLM 48.31 / 50.55 / 50.96 / 51.43 / 51.38；VisionZip 36.58 / 44.14 / 47.42 / 49.77 / 49.61。FoveaTile b3072 起顯著贏 IF-Prune（+2.0~+3.7）、b2048 起贏 FastV；b1024 跟 IF-Prune 打平
    - V*（191 題，uncompressed 83.77）：FoveaTile 76.44 / 77.49 / 81.68 / 84.82 / 83.25；IF-Prune 83.25 / 82.72 / 82.20 / 85.34 / 85.34；FastV 75.39 / 80.63 / 82.72 / 83.77 / 84.29。b1024 顯著輸 IF-Prune −6.81，其餘不顯著（191 題，CI 約 ±5）
    - DocVQA 補完：FoveaTile 84.06 / 89.01 / 90.29 / 90.79 / 90.88；b3072~5120 跟 IF-Prune 打平，比 FastV 低 0.5~0.8（b4096 / b5120 顯著）
    - 3a 的線索：FoveaTile 低 budget 弱在局部細節題（MME-RW Remote Sensing b1024 34.00 vs IF-Prune 46.00；V* direct_attributes b2048 78.26 vs 87.83），強在需要全域版面的題（Diagram and Table b1024 62.00 vs IF-Prune 51.50、FastV 39.50）
- [x] 1.8.3 診斷（3a，2026-10-16~17，V*，用 bbox 判斷目標物件所在的細節 tile；`foveatile_internvl.py` 的 stats 新增 `prior_ranked`，只記錄不影響選擇；結果 `results/diagnostics/vstar_selection/`）
  - 命中目標 tile 時 FoveaTile 正確率約 90%（高於 uncompressed 83.77），沒命中時約 44%（接近亂猜）；命中率 b1024 69.1% / b2048 73.3% / b3072 80.1%
  - 沒命中的主因是 **prior 沒把目標排進打分窗口**（b1024 26.2%、b2048 22.5%、b3072 14.7%）；q_attn 很準：目標進窗口後被選中 94% / 92%（隨機 40% / 50%）
  - prior 本身不差（目標排名中位數在前 10~12%），但窗口太小：b1024 只打分前 5 塊（細節網格中位數 24 塊）；目標落在 prior 前 N 名的比例 N=5 73.8%、N=8 85.3%、N=10 89.5%、N=12 95.3%、N=15 97.4%
  - 64% 的 V* 目標物件在 b1024 覆蓋圖裡短邊 < 14 px（不到一個 ViT patch），覆蓋 tile 救不回來
  - 跟 held-out 階段 B 的矛盾：階段 B 平均下來 S 超過 S0/2 沒增益（InfoVQA 不需要細節；MME-RW 每個 subtask 只有 120 題，雜訊大），但 V* 這類局部小物件題，窗口大小直接決定命中率
  - （補充，以物件為單位統計）prior 排進窗口的比例跟物件大小有關：b1024 覆蓋圖裡短邊 < 7 px 66.7%、7~14 px 73.3%、≥ 14 px 約 86%；relative_position（要找到兩個物件）71.5% < direct_attributes 83.5%
  - （補充）`THUMB_ATTN_LAYERS = (21, 28)` 與 `QATTN_LAYER = 20` 當初是在 InfoVQA val（測試集）上選的，跟舊 S 表一樣有 leakage，3b 要在校準集上重新驗證
- [x] 1.8.4 調整（3b，只能用校準集）：候選規則「打分窗口隨細節網格放大」（2026-10-17，使用者同意）
  - 規則：直接保留 + 打分的塊數 >= ceil(f × 細節網格塊數)（`experiments/candidates_rejected/svma_config.grid_qattn_tiles`；`experiments/candidates_rejected/internvl_stream.py --qattn_grid_frac f`，預設關閉，現行預設完全不變）
  - 掃描 f ∈ {0.2, 0.3, 0.4, 0.5}（使用者同意的是 0.3 / 0.4 / 0.5；smoke test 發現 MME-RW 的圖幾乎都切滿 45 塊、f=0.4 要打分 18 塊，所以多加較便宜的 0.2），m、S 用現行預設；兩個校準集 × b1024~b3072
  - **事先定的判斷標準（三項都要通過才改預設，之後重跑全部測試集）**：
    1. 兩個校準集的平均保留率不比現行預設低超過 0.5 個百分點
    2. MME-RW 校準集的局部細節題（Remote Sensing + Monitoring + Autonomous Driving 合併，360 題）相對現行預設顯著進步（配對 bootstrap 95% CI 下界 > 0）
    3. b1536 的 TTFT（兩個校準集平均）增加 <= 0.3 秒
  - 風險（事先說明）：階段 B 裡 b1024 S=10 對 S=5，Remote Sensing 沒有進步（−0.8），V* 的改善不一定能在校準集重現；沒通過就維持現行預設，診斷結果寫進論文
  - 執行：`experiments/candidates_rejected/run_calibrate_stageG.sh`（2026-10-16 23:27 起；accuracy 40 組 → TTFT），log `logs/calib/stageG.log`
  - **結果（2026-10-17）：四個 f 都沒有同時通過三條標準 → 不採用，維持現行預設（v1.0）**
    - 標準 1：每個 f 都有某個 budget 的平均保留率掉超過 0.5 個百分點（f=0.2 b3072 −1.01；f=0.3 b2560 −0.91；f=0.4 b1536 −1.16；f=0.5 b1536 −1.07）
    - 標準 2：局部細節題（360 題）只有 b1024 f=0.2 顯著（+2.22 [+0.28, +4.44]），20 次檢定裡 1 次，5 個 budget 合併後都不顯著（f=0.2 +0.67 [−0.44, +1.78]）
    - 標準 3：只有 f=0.2 通過（b1536 +0.15 s）；f 每多 0.1，TTFT 約 +0.15 s
    - 可能原因：MME-RW 的局部細節題多半要看整個場景（計數、交通），不像 V* 集中在單一小物件；MME-RW 的圖幾乎都切滿 45 塊，窗口變大後 q_attn 選錯的機會也變多；MME-RW 沒有 bbox，無法直接驗證命中率
- [ ] 1.8.5 （1.8.1~1.8.4 完成後再規劃）補圖表 / 場景文字（ChartQA、TextVQA、OCRBench）與一般場景（MMBench、MME、POPE 或 GQA）測試集；HR-Bench 8K 不加（使用者決定，4K + MME-RW Lite + V* 已涵蓋 HR 類）

### 階段 2：新 baseline（約 1 週）

- [ ] 2.1 實作 InternVL 版 CLS attention 擷取（含 pixel shuffle 2×2 加總），供 HiRED 與 GlobalCom2 共用
- [ ] 2.2 實作 GlobalCom2（`globalcom2_internvl.py` + `run_globalcom2.sh`），先在小樣本上 sanity check
- [ ] 2.3 實作 HiRED（`hired_internvl.py` + `run_hired.sh`）
- [ ] 2.4 （建議）GlobalCom2+FastV 組合
- [ ] 2.5 （建議）RAP-fixed-K：接 VisRAG retriever + Spatial-Awareness Layout

### 階段 3：InternVL3.5-8B 主結果（約 2 週）

- [ ] 3.1 新增 dataset 載入與評分：V\*、MME-RealWorld(-lite)、ChartQA、TextVQA、OCRBench、MMBench、MME、GQA、POPE
- [ ] 3.2 所有方法 × budget {1024, 2048, 3072}（Pareto 再加 1536、2560、4096）× 全部 benchmark，跑完整 val
- [ ] 3.3 產出表 1（InternVL 部分）、表 2、圖 1（Pareto），附 CI
- [ ] 3.4 產出「固定 accuracy」效率表（達到 uncompressed 95% 所需的 TTFT 與記憶體）

### 階段 4：效率與縮放（約 1 週）

- [x] 4.0 效率總表（P0 #1，2026-10-17 19:33 起）：`run_efficiency.sh`，5 個測試集 × 6 個方法 × b1024~b5120，每個資料集 51 張圖（第 1 張 warmup），SparseVLM 用 accuracy 實驗的 λ；結果 `results/efficiency/metrics_<ds>.csv`，log `logs/efficiency/`
- [x] 4.1 解析度縮放（P0 #2）：`run_scaling.sh`，HR-Bench 8K 的 11 張圖（第 1 張 warmup），`--max_num` ∈ {7, 13, 25, 49, 97, 145, 201}，b1536，每個 (max_num, 方法) 獨立程序（OOM 只記錄）；效率總表跑完後自動接著跑；結果 `results/scaling/`，log `logs/scaling/`
  - smoke test（2 張圖）：188 塊原始 tile 時 uncompressed 17.09 s / 12.90 GB / 1,495 TFLOPs、FastV 3.15 s / 8.34 GB / 237、FoveaTile 0.72 s / 0.41 GB / 46；FoveaTile 從 7 塊到 188 塊 TTFT 與記憶體幾乎不變
  - 要誠實呈現：7 塊 tile 時 FoveaTile（0.69 s）比 uncompressed（0.41 s）慢，prior + q_attn 有固定成本，要找出交叉點
- [x] 4.0b 補預設點 b1536（2026-10-17 發現：所有測試集的 accuracy 與效率都只有 b1024~b5120，缺預設點）：`experiments/legacy_runs/run_b1536.sh`，等縮放曲線結束後自動執行 → ① 5 個測試集 × {FoveaTile, FastV, VisionZip, IF-Prune, SparseVLM} 的 accuracy（`run_test_suite.sh`，結果 `results/runs/<ds>/<method>_b1536_*.json`）② 效率（`results/efficiency/b1536/`，SparseVLM 用 ① 校準出的 λ）；log `logs/b1536.log`
  - 為此修正：`run_test_suite.sh` 對 baseline 腳本把 `hrbench_4k` 轉成 `--dataset hrbench` + `HRBENCH_SPLIT`；`summarize_results.py` 處理沒有分數的情況
- [x] 4.0c 正文 budget 網格改成 1024~3072 每 512 一點（方案 A，2026-10-19 使用者決定）：1024 / 1536 / 2048 / **2560** / 3072 進正文（Pareto 曲線、主表），4096 / 5120 放附錄；不補 3584 / 4608（已飽和、S 是外推值）
  - 補 b2560：`BUDGET=2560 ./run_budget_fill.sh`（通用版，取代 `experiments/legacy_runs/run_b1536.sh`），5 個測試集 × 5 個方法的 accuracy + 效率；log `logs/b2560.log`
  - 2026-10-19 00:22 第一次啟動時兩張卡被另一位使用者（harrykuo1 的 vLLM，各約 45 GB）佔用、FastV OOM → 已停掉並刪除不完整結果；腳本改成先等 GPU 空出來（accuracy 每張卡 ≥ 30 GB、效率每張卡 ≥ 44 GB）再開始。之前的效率量測都在對方啟動前完成，不受影響
  - 使用者指示：不動別人的程序；每次等 GPU 最多 24 小時（`MAX_WAIT_H=24`），超過就放棄並在 log 留 `TIMEOUT`，等使用者回來再通知。2026-10-19 00:26 重新啟動，accuracy 階段最晚等到 10-20 00:26
  - 第二次（10-19 00:32 開始）：開始時 GPU 空閒，但中途另一位使用者在 GPU0 啟動約 43.7 GB 的程序（PID 934101），GPU0 上的 13 個工作全部 OOM（GPU1 的 12 個正常完成）；SparseVLM 在 DocVQA / HR-4K / V* 的 λ 校準也 OOM、退回預設 0.1，所以這 3 個資料集的 SparseVLM 效率數字無效。log 留在 `logs/b2560_try1.log`、`logs/test_suite_b2560_try1.log`
  - 修正：`run_test_suite.sh` 改成**每個工作開始前**檢查自己那張卡的空間（`MIN_FREE_GB=30`、`MAX_WAIT_H=24`，超時就跳過該工作並留 `[TIMEOUT]`）
  - 第三次（10-19 23:01 開始）：已完成的 12 個工作自動跳過，補 13 個失敗的；效率 5 個資料集全部重量
  - 已完成的部分（FoveaTile b2560）：InfoVQA 78.60（b1024~b3072：67.75 / 74.82 / 77.18 / 78.60 / 79.36，單調）；MME-RW Lite 52.16（48.78 / 52.06 / 51.22 / 52.16 / 53.47）
  - 2026-10-20 全部完成（accuracy 10:42、效率 11:22，25 個結果檔都完整，SparseVLM λ 都是校準值）
  - **彙整**：`results/paper/build_main_table.py` → `results/paper/main_internvl.csv`（5 個測試集 × 6 個方法 × b1024~b3072 的 accuracy、FoveaTile 對各方法的配對差與 CI、TTFT、記憶體增量、TFLOPs）
  - 5 個資料集平均（保留率 % / TTFT s / 記憶體增量 GB），b1024 → b3072：
    - FoveaTile：91.5 / 0.63 / 0.33 → 95.0 / 0.72 / 0.41 → 95.8 / 0.91 / 0.53 → 97.6 / 1.01 / 0.56 → 99.3 / 1.20 / 0.77
    - IF-Prune：94.5 / 1.11 / 2.94 → 94.9 / 1.18 / 2.95 → 96.0 / 1.28 / 2.96 → 97.1 / 1.34 / 2.98 → 97.4 / 1.45 / 3.03
    - FastV：82.1 / 0.62 / 1.33 → 90.0 / 0.69 / 1.33 → 92.7 / 0.78 / 1.34 → 95.5 / 0.85 / 1.35 → 97.2 / 0.95 / 1.36
    - SparseVLM：83.7 → 85.5 → 90.1 → 91.2 → 91.8；VisionZip：61.4 → 71.5 → 78.5 → 83.0 → 86.1；uncompressed TTFT 1.74 s、1.74、記憶體 2.05 GB
  - 結論：b1536 起 FoveaTile 平均保留率跟 IF-Prune 打平或更高（b3072 99.3 vs 97.4），TTFT 約 60%、記憶體約 1/7~1/4；每個 budget 都比 FastV 高（b1024 +9.4、b1536 +5.0），TTFT 差不多、記憶體約 1/3；只有 b1024 輸 IF-Prune（91.5 vs 94.5）
  - 固定 accuracy（約 95% 保留率）時：FoveaTile b1536 0.72 s / 0.41 GB；IF-Prune b1536 1.18 s / 2.95 GB；FastV 要到 b2560 才達到，0.85 s / 1.35 GB
- 4.1 縮放曲線結果（2026-10-17 22:53 完成，沒有 OOM；HR-Bench 8K 10 張圖，b1536；8K 圖長寬比固定，max_num 7/13 都切 7 塊、25/49 都切 25 塊，實際 5 個點：7 / 25 / 78 / 127 / 188 塊；`results/scaling/summary.json`）
  - FoveaTile：TTFT 0.69~0.71 s、記憶體增量 0.41 GB、45.6 TFLOPs，**從 7 到 188 塊完全持平**
  - 188 塊時：uncompressed 17.22 s / 12.90 GB / 1,495 TF；FastV 3.19 / 8.34 / 236；SparseVLM 4.09 / 8.44 / 307；VisionZip 2.45 / 0.49 / 162；IF-Prune 7.00 / 8.28 / 509 → FoveaTile 的 TTFT 是 uncompressed 的 1/24、FastV 的 1/4.5、VisionZip 的 1/3.4
  - VisionZip 記憶體也持平（約 0.49 GB），但 TTFT 跟著 ViT 線性增加
  - 交叉點：7 塊時 FoveaTile 最慢（0.69 s vs uncompressed 0.41、FastV / VisionZip 0.37）；25 塊時已比 uncompressed 快（0.70 vs 1.43）但仍慢於 FastV（0.62）、VisionZip（0.58）；線性內插約 **30~35 塊**後比所有方法都快
  - 同時請求數換算（A6000 50.9 GB − 權重 17.06 GB ≈ 33.8 GB ÷ 每請求記憶體增量，188 塊）：FoveaTile 約 82、VisionZip 約 69、FastV / IF-Prune 約 4、uncompressed 約 2
- 4.0b b1536 結果（2026-10-18；accuracy 完整資料集、效率 50 張圖）
  - accuracy（FoveaTile − 對手，* 顯著）：DocVQA 87.02（IF-Prune 88.17 −1.15*、FastV 84.01 +3.01*）；InfoVQA 74.82（IF-Prune 72.68 +2.14*、FastV 60.96 +13.86*）；HR-4K 56.50（全部不顯著，VisionZip 除外）；MME-RW Lite 52.06（高於 uncompressed 51.59；IF-Prune 49.30 +2.76*、FastV 46.90 +5.16*、SparseVLM 50.23 +1.82*）；V* 77.49（IF-Prune 81.68 −4.19 n.s.）
  - 效率（TTFT / 記憶體增量 / TFLOPs）：FoveaTile 0.68~0.76 s / 0.38~0.42 GB / 43~49；FastV 0.63~0.80 / 1.09~1.66 / 45~58；VisionZip 0.59~0.73 / 0.47~0.50 / 41~51；IF-Prune 1.04~1.41 / 2.77~3.19 / 69~96
  - FoveaTile TTFT 分段（DocVQA b1536）：寫 KV 0.292 s 最多，其次 q_attn 0.132、prior 0.101
  - SparseVLM b1536 校準出的 λ：DocVQA / HR-4K 0.5666，InfoVQA / MME-RW Lite / V* 0.3794
- 4.0 效率總表的初步觀察（4 個資料集，2026-10-17）：
  - 記憶體增量 FoveaTile 最低或跟 VisionZip 並列（b2048 約 0.53 GB；FastV / SparseVLM 約 1.3 GB、IF-Prune 約 3.0 GB）
  - 一般測試集解析度（28~37 塊 tile）下，FoveaTile 的 TTFT 跟 FastV 差不多或慢約 0.2 s、比 VisionZip 慢 0.1~0.2 s，只比 uncompressed 與 IF-Prune 快 → TTFT 的優勢要靠縮放曲線（高解析度）證明
  - FoveaTile 的 TTFT 隨 budget 增加較快（b1024→b3072 +0.57 s，VisionZip +0.36 s），主要在寫 KV（`kv_commit`）→ 可做不改變輸出的工程優化（待辦）
- [ ] 4.2 固定 GPU 記憶體下的最大 batch 與 throughput（圖 3）：FoveaTile 是逐張串流、沒有 batched 實作，改用「每個請求的記憶體增量」換算同一張 GPU 最多能同時服務的請求數（論文註明為換算值）
- [ ] 4.3 （選做）修 native grid 在 8K 變差的問題：detail_scale 0.5 到 0.7，或 S 隨 grid 大小縮放；修不好就把主張收斂成「有上限的成本」

### 階段 5：第二個模型（約 2 週）

- [ ] 5.1 完成 LLaVA-OneVision 版 FoveaTile（從 `exp/llava_ov_svm.py` 出發）
- [ ] 5.2 在 LLaVA-OV 上跑主要 baseline 與主要 benchmark（表 1 的第二個模型）
- [ ] 5.3 在 LLaVA-OV 上跑 HiRED、GlobalCom2 官方程式碼，對照原論文數字，驗證移植正確
- [ ] 5.4 （選做）InternVL3.5-2B 或 38B，看模型大小的趨勢

### 階段 6：HR 專表與 ablation（約 1 到 2 週）

- [ ] 6.1 HR 專表：HR-Bench 4K/8K、V\*、MME-RealWorld，含 RAP-fixed-K；完整 RAP 只跑 HR-Bench 4K 與 V\*，記錄每題 latency（表 3）
- [ ] 6.2 Prior ablation：CLS attention / thumbnail 引導 / q_attn，其他固定
- [ ] 6.3 粒度 ablation：整塊 tile vs. 挑分散 token（可沿用已有的空間局部性實驗）
- [ ] 6.4 覆蓋 / 細節 ablation：只有覆蓋（≈ Low-Res-Fill 同路徑）/ 只有細節（m=1）/ 兩者都有
- [ ] 6.5 m、S 敏感度；串流 vs. 一次 prefill
- [ ] 6.6 prior 排名與 q_attn 排名的一致性分析

### 階段 7：寫作

- [ ] 7.1 依 02 第 5 節的貢獻寫法重寫 introduction 與 contribution 條列
- [ ] 7.2 Related work：token pruning（FastV、SparseVLM、VisionZip、IF-Prune、PyramidDrop、VisPruner、DART）、高解析度壓縮（HiRED、GlobalCom2）、zoom-in / crop（RAP、ViCrop / "MLLMs know where to look"、ZoomEye、DyFo）
- [ ] 7.3 方法章節加入 O(K) vs. O(N) 的記憶體與 FLOPs 複雜度分析
- [ ] 7.4 Limitations：任務類型影響最佳 m；HR 類在預設 m 下保留率約 89%
- [ ] 7.5 Appendix：負面結果與完整超參數、校準流程

## 6. 決策點

- **階段 4.1 之後**：若縮放曲線顯示其他方法線性增加、FoveaTile 持平，照原計畫衝 ICML；若差距不明顯，考慮改投 workshop 或次一級會議，或把重點轉到串流的影片 / 多圖場景。
- **階段 5 之後**：若 LLaVA-OV 上的趨勢跟 InternVL 不一致，要先查原因再寫作。
