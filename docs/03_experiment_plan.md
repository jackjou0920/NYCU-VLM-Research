# 03 實驗規劃與執行 checklist

## 1. Baseline 組合

現有的 FastV、SparseVLM、VisionZip、IF-Prune 都是通用的 token pruning，沒有一個針對「高解析度 + tile 切割」設計。SVMA 主打高解析度，必須跟同類方法比。

| 等級 | 方法 | 備註 |
|---|---|---|
| 必要 | Uncompressed | 上限參考 |
| 必要 | Low-Res-Fill | 已證明是最強的簡單 baseline，不放一定會被問 |
| 必要 | FastV、SparseVLM、VisionZip、IF-Prune | 已有；SparseVLM、VisionZip 要先修（見 01 第 2 節） |
| 必要 | **GlobalCom2** | 移植到 InternVL3.5，優先度最高 |
| 必要 | **HiRED** | 移植到 InternVL3.5，跟 GlobalCom2 共用 CLS attention 程式碼 |
| 建議 | GlobalCom2+FastV | GlobalCom2 論文自己提出的看問題版本 |
| 建議 | RAP-fixed-K | VisRAG 挑 crop + Spatial Layout，拿掉 RE-Search，K 跟 SVMA 一樣 |
| 選做 | 完整 RAP | 只跑 HR-Bench 4K 和 V\*，當 accuracy 上限參考並列出 latency |
| 選做 | VisPruner 或 DART | 代表較新的 token pruning 方法 |

### 移植注意事項

- **GlobalCom2**（預估 2 到 4 天）：InternVL 的動態 tile + thumbnail 跟 LLaVA-NeXT 的 anyres 幾乎同構。InternViT 有 CLS token（InternVL 在 ViT 輸出後丟掉），可以取最後一層 CLS 對 patch 的 attention。pixel shuffle 會把 1024 個 patch 併成 256 個 token，CLS attention 要按 2×2 加總。原論文設定：crop 豐富度 = thumbnail 對應區域的 CLS attention 加總；保留比例 r_j = R × (1 + σ_j − 1/n)，σ 為 softmax((s − max)/τ)，τ=10；crop 內 token 分數 = 0.5 × 內插後的全域分數 + 0.5 × 局部 CLS 分數。
- **HiRED**（約 1 天）：依 CLS attention 分配每個 partition 的 budget，再於 partition 內取 top token。
- **RAP-fixed-K**：用 VisRAG 算問題與每個 tile 的相似度，取 top-K，用 Spatial-Awareness Layout 重拼（刪掉全空的行列）後送進 InternVL。
- 介面沿用 `fastv_internvl.py` / `ifprune_internvl.py` 的輸入輸出格式，新增 `run_globalcom2.sh`、`run_hired.sh`。

## 2. 公平比較的規則（先定好再跑）

**不要只讓 SVMA 在每個 dataset 挑最好的 budget。** baseline 固定 budget、SVMA 卻挑最好的，等於在測試集上調參。

- **budget 的定義**：decode 時留在 LLM KV cache 裡的 vision token 數。FastV、IF-Prune 在前幾層保留全部 token，要另外列 LLM FLOPs 或「實際處理的 token 數」。
- **同一套環境**：同一張 GPU、同一個 attention 實作。FastV、SparseVLM 需要 attention weight，若被迫用 eager attention 要註明，或改成只算部分 attention，否則 TTFT 會被高估。
- **量測方式**：先 warmup，同一批圖取中位數；記憶體列「扣掉模型權重後的增量」。
- **評測流程**：所有方法走同一條評測流程（建議 VLMEvalKit 或 lmms-eval 的 protocol），全部跑完整 val 集，附 bootstrap CI 和配對檢定。
- **超參數只在 held-out 資料上決定**：S、m、預設 budget 用 DocVQA / InfoVQA 的 train split 或 ChartQA val 校準，定好後所有測試集共用，不再改。

## 3. 三種比較方式

| 方式 | 回答的問題 | 做法 |
|---|---|---|
| **A. 固定 budget**（主表） | 一樣多的 token，誰比較準？ | 所有方法用同樣的 budget（1024、2048、3072），比 accuracy 並附 CI |
| **B. Pareto 曲線**（主圖） | 整體 accuracy 和成本取捨誰比較好？ | 每個方法掃多個 budget，畫 accuracy 對 TTFT、accuracy 對峰值記憶體 |
| **C. 固定 accuracy**（效率表） | 達到同樣 accuracy，誰比較省？ | 例如「達到 uncompressed 95% 時，各方法需要多少 TTFT 和記憶體」 |

**SVMA 預設 budget 的放法：**

- 預設值（目前候選 b2048，K=8、m=6、S=4）在 held-out 校準資料上決定，所有測試集共用，論文寫成「SVMA（default）」，在 Pareto 圖上標成一個點。
- 若用逐圖規則 `budget = min(2048, 原生 token 數)`，baseline 也要用同樣的平均 token 數比，並列出每個 dataset 實際用了多少 token。
- 不要宣稱 SVMA 會「自動選 budget」：已驗證 answer confidence 和 prior 都無法逐題選出最好的 m。只說「有一個在 held-out 資料上校準的固定預設值」。

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
  - 91.4 vs 92.0 的原因：FastV / SparseVLM / VisionZip 的舊 summary 開了答案抽取，IF-Prune / SVMA 沒開。統一用 `evaluate_anls.py`（不抽取）重新評分後 uncompressed DocVQA = 91.4（前 500 題 91.9），其他方法差 ±0.6，只需重新評分、不需重跑
  - 手動 prefill 路徑會省略 `<img>` / `</img>`：用 FastV 路徑不壓縮跟 batch_chat 比（各 100 題），DocVQA 95.8 vs 95.8、InfoVQA 79.4 vs 79.4，95~99% 答案逐字相同，影響可忽略，不改 prompt 組法
  - 原則：baseline 照論文與官方原始碼實作，不替其他方法加改良；只修「非官方、且偏離官方行為」的部分，並在論文附錄註明
- [x] 1.2 SparseVLM（2026-10-13，使用者同意保留；結果待重跑）
  - bug：repo 自己加的最後一步「裁切到 budget」照位置取前 N 個 token（等於丟掉圖片後段），官方 score.py 一律 `topk` 依分數挑、沒有照位置截斷。改成依分數裁切（`sparsevlm_core.py`）。InfoVQA b4096 λ=0.005 前 100 題：62.0 → 75.8（uncompressed 79.4）
  - λ 校準：原圖 token 不超過門檻的圖不算進中位數；校準張數 8 → 32（`sparsevlm_internvl.py`、`run_sparsevlm.sh`）
  - 附錄寫法：官方固定 token 表只支援 LLaVA 576 token，我們用 Eq.7 決定每層剔除數，最後依分數裁切到 budget
  - [~] 重跑（2026-10-14 起）：DocVQA / InfoVQA / HR-4K × b1024~b5120，`RUN_STANDARD=0`（不重跑 batch_chat，評分用 meta 的 GT）、`RUN_METRICS=0`（效率之後用統一的 bench_metrics 量）；輸出 `results/baselines_v2/internvl_sparsevlm_<ds>/`，log `logs/baselines_v2/`（舊結果保留在 `results/baselines/`）
    - GPU0：DocVQA 拆兩個程序（b1024/2048/3072、b4096/5120）；GPU1：InfoVQA → HR-4K。DocVQA 單程序每分鐘約 20 題，預估全部 15~20 小時
- [x] 1.3 VisionZip（2026-10-13，演算法未改；只加診斷選項 `--score_mode`、`--score_layer`，預設輸出跟舊結果 200/200 相同）
  - b1024 前 200 題對照組（ANLS）：DocVQA 官方 CLS 30.5 / random 25.5 / uniform 19.2 / CLS 第 17 層 39.5 / 第 11 層 37.3；InfoVQA 官方 40.7 / random 39.5 / uniform 35.9 / 第 17 層 39.6 / 第 11 層 39.4
  - 結論：實作行為合理（贏過 random、uniform），分數低是方法本身（不看問題、每塊 tile 平均分到約 39 個 token）加上 InternViT 最後一層 CLS 打分弱。主實驗維持官方最後一層，這組數字放附錄當「實作正確」的證據
  - 結果檔：`results/diagnostics/visionzip/`
- [~] 1.4 held-out 校準集：已加 `infovqa_train_calib`（InfoVQA train，seed 0 抽 600 題，`internvl_stream.py --dataset infovqa_train_calib`）。DocVQA 沒有帶答案的 train split
  - HR 類校準集（2026-10-13，使用者同意）：`mme_rw_calib` = MME-RealWorld 完整版（英文 23,609 題）扣掉 Lite 的題目、以及圖片出現在 Lite 的題目，5 個 Subtask 各抽 120 題（seed 0）共 600 題。測試集改用 `mme_rw_lite`（1,919 題），校準 / 測試連圖片都不重疊
  - 準備：`python prepare_mme_realworld.py`（只從 tar 分割檔抽出校準集用到的圖 + 解壓 Lite，放 `~/datasets/MME-RealWorld`，可用 `MME_RW_ROOT` 改）；loader `process_common.load_mme_realworld`（圖片延遲開檔），prompt 用官方作答指示，評分 = 選擇題 accuracy（`summarize_results.py` 的 `mc_accuracy`）
- [x] 1.5 校準腳本 `run_calibrate.sh`（階段 A 掃 m、階段 B 掃 S）；`internvl_stream.py` 新增 `--coverage_tiles`、`--qattn_tiles`。2026-10-13 22:29 起跑階段 A（InfoVQA train，b1024~b3072，23 組，log：`logs/calib/stageA_m.log`）；23:20 起跑 MME-RealWorld 校準集的階段 A（23 組，log：`logs/calib/stageA_m_mme_rw.log`）。結果都在 `results/calib/<dataset>/`，看表：`python summarize_results.py --root results/calib`
  - [ ] 待做：baseline 腳本（fastv / sparsevlm / visionzip / ifprune 的 `--dataset`）還不支援 `mme_rw_lite`，測試階段要補（沿用 `process_common.load_mme_realworld` + `get_image`）
  - 原項目：在 held-out 上重新校準 `svma_config.py` 的 S 表與 `COVERAGE_FRAC`，並補掃 multi-scale 的 b1536、b2560、b3072
  - 階段 A 結果（2026-10-14，兩個校準集各 600 題，分數 = 兩者平均；InfoVQA 用 ANLS、MME-RW 用 accuracy）：

    | m 規則 | b1024 | b1536 | b2048 | b2560 | b3072 | 平均 |
    |---|---|---|---|---|---|---|
    | m=1（舊 SVMA） | 57.9 | 61.2 | 63.9 | 65.2 | 68.4 | 63.3 |
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
    - 結果檔：`results/calib/metrics/`、`results/calib/*/uncompressed_InternVL3_5-8B.json`；`bench_metrics.py` 新增 `--svma_m_fracs`、`--warmup_images`、`ttft_s_median`
  - [x] 預設 m/K 與 budget 已決定（見 1.6）
  - [x] 階段 B（S）：2026-10-14 08:54~15:15，`run_calibrate_stageB.sh`（m = K/2；S ∈ {0, S0/2, S0, 1.5 S0, 2 S0}，S0 = 舊規則的值；兩個校準集 × 5 個 budget，accuracy 600 題、TTFT 50 張圖）
    - 事先定的規則：平均保留率與最差任務保留率都在最佳值 1 個百分點內的 S 中取最小 → 選出 {1024: 5, 1536: 13, 2048: 4, 2560: 3, 3072: 14}
    - 問題：最差任務每組只有 120 題（1 題 ≈ 1.5 個百分點），最差的是哪一組也會隨 S 改變，規則被雜訊主導；配對 bootstrap 顯示 b1536 S=13、b3072 S=14 跟 S0/2 的差距都不顯著
    - **決定（使用者同意，偏離事先規則，論文附錄要註明）：一律 S = S0/2 → `QATTN_TILES_BY_BUDGET = {1024: 5, 1536: 4, 2048: 4, 2560: 3, 3072: 3}`**（`svma_config.py`；舊表留為 `LEGACY_QATTN_TILES_M1`）
    - 預設點 b1536（m=3、S=4）：平均保留率 95.6%、TTFT 0.77 s（S0=9 時 0.99 s）
    - **論文可用的 ablation（held-out，顯著）**：S=0（只靠 prior、不做 q_attn）比 S0/2 差：Remote Sensing b1024 −10.8 [−18, −4]、b1536 −10.8 [−18, −4]；Monitoring b2560 −7.5 [−12, −3] → 依問題打分在局部細節任務上有實質貢獻
    - S 超過 S0/2 沒有可靠增益（CI 大多包含 0），只多花 TTFT
    - 驗證：新預設跑 8 題，跟階段 B 的 S=4 結果逐題相同
    - 結果檔：`results/calib/*/svma_b*_m*_S*.json`、`results/calib/metrics/metrics_S_*.csv`
- [x] 1.6 預設操作點（2026-10-14，使用者選方案 A）：**b1536、m/K = 0.5、S = 4**（`svma_config.COVERAGE_FRAC = 0.5`、`DEFAULT_BUDGET = 1536`，`internvl_stream.py --budget` 預設跟著改）
  - 依據：held-out 平均保留率 96%（b1024 90%、b2048 96% 沒有再進步），最差任務 85%，TTFT 0.99 s（uncompressed 1.21 / 2.81 s），記憶體增量 0.43 GB（uncompressed 1.6 / 3.2 GB）。第二個操作點 b2560（平均 99%、最差 94%、1.19 s）
  - 論文定位（方案 A）：主打「最差任務也穩、成本有上限」，純覆蓋（m=K）放 ablation 誠實呈現：平均保留率相同、TTFT 約一半，但最差任務（Remote Sensing）差很多
  - 主表的固定 budget（1024 / 2048 / 3072）照舊，b1536 是在 Pareto 圖上標出的預設點
- [x] 1.7 統一量測腳本（2026-10-14，`bench_metrics.py` + 新增 `flops.py`；baseline 檔案未改）
  - `--warmup_images`（預設 1，前幾張不計）、`ttft_s_median`、`mem_delta_gb` = peak_alloc − 模型權重（IF-Prune 含小 VLM 權重，論文要註明）
  - SVMA TTFT 分段（`internvl_stream.py` 的 stats["ttft_breakdown"]）：text_prefill / cover_vit / prior / detail_vit / qattn / kv_commit / first_token → CSV 的 `t_*` 欄
  - 解析 FLOPs（TTFT 階段 = ViT + 到第一個 token 的 prefill）：每層 2·n·P_lin + 4·n·c·H·d，P_lin 直接讀模型參數；每個方法依實際處理的 token 數（FastV / IF-Prune 剪枝前後、SparseVLM 依 stage_logs、SVMA 含 prior / q_attn / 寫 KV）→ `vit_tflops` / `llm_tflops` / `aux_tflops`（IF-Prune 小 VLM）/ `total_tflops`
  - `--svma_m_fracs`、`--svma_s_mults`：一次量多組 m / S
  - smoke test（MME-RW 3 張、b1536、GPU 有其他程序，TTFT 不可信）：total TFLOPs uncompressed 242 / SVMA 53 / VisionZip 59 / FastV 69 / IF-Prune 117 / SparseVLM 135；SVMA 的 ViT 5.9 vs 其他方法 34 TFLOPs（被丟掉的 tile 不跑 ViT）

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

- [ ] 4.1 解析度縮放：`--max_num` 或輸入解析度從 12 到 200 塊 tile，量所有方法的峰值記憶體與 TTFT（圖 2）
- [ ] 4.2 固定 GPU 記憶體下的最大 batch 與 throughput（圖 3）
- [ ] 4.3 （選做）修 native grid 在 8K 變差的問題：detail_scale 0.5 到 0.7，或 S 隨 grid 大小縮放；修不好就把主張收斂成「有上限的成本」

### 階段 5：第二個模型（約 2 週）

- [ ] 5.1 完成 LLaVA-OneVision 版 SVMA（從 `exp/llava_ov_svm.py` 出發）
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

- **階段 4.1 之後**：若縮放曲線顯示其他方法線性增加、SVMA 持平，照原計畫衝 ICML；若差距不明顯，考慮改投 workshop 或次一級會議，或把重點轉到串流的影片 / 多圖場景。
- **階段 5 之後**：若 LLaVA-OV 上的趨勢跟 InternVL 不一致，要先查原因再寫作。
