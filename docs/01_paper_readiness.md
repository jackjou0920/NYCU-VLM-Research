# 01 現況評估：FoveaTile 夠不夠投 ICML

## 結論

**還不夠。** 系統設計和負面分析都很紮實，但現在的結果撐不起「FoveaTile 比現有方法好」這個主張。比較撐得起的定位是：

> 在 accuracy 和效率的取捨上，FoveaTile 是幾乎不用付代價的那個點：accuracy 接近 IF-Prune，但只用約 1/5 的 activation 記憶體、約 60% 的 TTFT，而且不需要訓練、不需要額外模型。

這要畫成 accuracy 對 TTFT / 記憶體的 Pareto 圖，不要寫成在每個 benchmark 都贏。

## 1. 目前結果（InternVL3.5-8B）

DocVQA / InfoVQA 為前 500 題（ANLS），HR-Bench 為 200 組（CircularEval）。FoveaTile 用目前的預設 m=3K/4。

| 方法 | Doc b1k | Doc b2k | Info b1k | Info b2k | HR4K b1k | HR4K b2k | HR8K b1k | HR8K b2k | 平均 b1k / b2k |
|---|---|---|---|---|---|---|---|---|---|
| Uncompressed | 92.5 | | 78.9 | | 62.0 | | 56.5 | | |
| **FoveaTile（預設）** | 82.6 | 90.0 | 66.7 | **76.7** | 55.5 | 55.0 | 51.0 | 50.5 | 64.0 / 68.1 |
| IF-Prune | **87.6** | 89.9 | **70.7** | 72.9 | 58.0 | **58.5** | **52.5** | **54.5** | **67.2 / 69.0** |
| FastV | 71.9 | 89.0 | 49.8 | 63.0 | 53.5 | 58.0 | 47.5 | 53.5 | 55.7 / 65.9 |
| SparseVLM | 61.7 | 66.8 | 52.2 | 56.0 | 60.0 | 55.0 | – | – | |
| VisionZip | 33.7 | 53.8 | 33.9 | 51.6 | 43.0 | 52.5 | – | – | |
| Low-Res-Fill | 85.1 | **91.1** | 63.6 | 72.7 | 44.0 | 53.0 | 34.0 | 49.0 | 56.7 / 66.5 |

結果檔：`results/archive/baselines/`、`results/archive/multiscale/v1/`。

- FoveaTile 明顯贏的只有 InfoVQA b2048。平均分數在兩個 budget 都輸 IF-Prune 1 到 3 分。
- FoveaTile 真正的優勢在成本。以 HR-4K 為例，扣掉 17.06 GB 的模型權重後：
  - 峰值記憶體增量：FoveaTile 0.3 到 0.6 GB，IF-Prune 約 3.2 GB，Uncompressed 2.5 GB。
  - TTFT：FoveaTile 0.60 / 0.97 s，IF-Prune 1.35 / 1.52 s，Uncompressed 2.20 s。
- `results/archive/metrics_n100/summary_tables.md` 的 TTFT 與記憶體是舊版 m=1 的數字，要用新預設重新量。

## 2. 實驗本身要先修的問題

這些不修，reviewer 會直接質疑實驗：

- **SparseVLM 的曲線不是單調的**：InfoVQA b3072 是 65.1，b4096 反而掉到 59.5；HR-4K b1024 是 60.0，b2048 掉到 55.0。lam=0.005 那幾組很可能有 bug。
- **VisionZip 低得不合理**（DocVQA b1024 只有 33.7）。要先在它官方支援的模型上重現原論文數字，或寫清楚移植到 InternVL 時做了哪些改動。
- **評測流程不一致**：IF-Prune 那組的 uncompressed DocVQA 是 91.4，其他組是 92.0；FoveaTile 的串流路徑本身也比 batch_chat 高約 1.3 ANLS。全部方法要改用同一條流程。
- **S 和 m 是在測試集上調的**：S 的表（`foveatile_config.py` 的 `QATTN_TILES_BY_BUDGET`）是用 InfoVQA val 抽 600 題校準的，m=0.75 也是在同一批前 500 題上選的，這算 test leakage。要改在 held-out 資料上校準，定好之後就不再動。
- **只跑了前 500 題**。最終結果要跑完整 val（DocVQA 5349 題、InfoVQA 2801 題），並附上 CI。HR-Bench 只有 200 組，CI 約 ±6 到 7 分，目前大部分的 HR 差距都不顯著。
- **只有一個模型**（InternVL3.5-8B），這是最大的硬傷之一。至少再加一個模型家族（建議 LLaVA-OneVision，`exp/llava_ov_svm.py` 已有雛形），最好再加一個不同大小（2B 或 38B）。

## 3. 要補的 dataset

| 類型 | 建議 | 理由 |
|---|---|---|
| 高解析度細節 | V\* Bench、MME-RealWorld(-lite) | FoveaTile 的主戰場；HR-Bench 樣本太少，需要更多高解析度 benchmark 才有統計力；也跟 RAP 對齊 |
| 文字密集 | ChartQA、TextVQA、OCRBench | 標準配備，也能驗證「覆蓋 + 細節」的設計 |
| 一般場景 | MMBench、MME、GQA、POPE（或 MMStar） | VisionZip、SparseVLM、GlobalCom2 都有報，reviewer 一定會問。這類圖 tile 數少，budget 建議改用相對比例來報 |
| 加分項 | 影片或多圖（Video-MME、MLVU） | 最能展現「串流」優勢，但工作量大 |

建議用 VLMEvalKit 或 lmms-eval 的標準 protocol 跑，避免自己寫的評分方式被挑。

## 4. 預設 budget：建議 b2048（K=8、m=6、S=4）

| budget | Doc 保留率 | Info 保留率 | HR4K 保留率 | HR8K 保留率 | HR4K TTFT | 記憶體增量 |
|---|---|---|---|---|---|---|
| b1024（m=3） | 89% | 85% | 90% | 90% | 0.60 s（27%） | 0.31 GB |
| **b2048（m=6）** | **97%** | **97%** | 89% | 89% | 0.97 s（44%） | 0.62 GB |
| Uncompressed | 100% | 100% | 100% | 100% | 2.20 s | 2.50 GB |

（保留率 = FoveaTile 分數 ÷ uncompressed 分數；TTFT 括號內是佔 uncompressed 的比例。）

- **b2048 是文件類任務的轉折點**：從 b1024 到 b2048，保留率從約 87% 跳到 97%，TTFT 只多 0.37 s，記憶體增量仍只有 uncompressed 的 1/4。
- **HR 類任務在 m=6 時卡在約 89%**，因為覆蓋 tile 佔了太多名額。HR 類用 m=1 搭配較大 budget 較好：b4096 時 59.5（96%），b2048 加 native grid 61.0。論文裡要把「任務類型會影響最佳 m」寫成限制或使用建議。
- **「自動」的定義**建議是 `budget = min(2048, 這張圖原生的 token 數)`，讓小圖不被強制壓縮。
- **還沒驗證**：multi-scale 版本只跑過 b1024 和 b2048，要再補 b1536、b2560、b3072，畫出完整 Pareto 曲線，才能說 b2048 是轉折點。
- 以上數字也是在測試子集上看的；正式的預設值要依 [03_experiment_plan.md](03_experiment_plan.md) 在 held-out 資料上決定。

## 5. 動機故事的缺口

**目前的故事：** 高解析度切 tile 導致 token 爆量 → 現有 pruning 方法仍要讓所有 tile 都過 ViT、所有 token 都 prefill（至少前幾層），成本隨解析度增加 → FoveaTile 用串流方式選 tile、KV 有上限、被丟掉的 tile 連 ViT 都不跑、保留空間局部性 → 成本跟解析度無關。

**缺口：**

1. **核心主張「成本跟解析度無關」，是在看不出差別的情況下驗證的。** 現在約 20 到 37 塊 tile、batch=1、8B 模型，各方法記憶體差異只有 1 到 3 GB，而且 VisionZip 的記憶體跟 FoveaTile 差不多、TTFT 還更快。要補**縮放實驗圖**：tile 數從 12 增加到 200（或解析度 2K、8K、16K），畫各方法的峰值記憶體和 TTFT，外加固定 GPU 記憶體下的最大 batch / throughput。已有「35 到 169 塊 tile 時 TTFT 和記憶體不變」的證據，應該當成第一張主圖。
2. **「多給解析度但不加成本」只證明了一半。** HR-4K 用 native grid 時 +5.5，HR-8K 反而 −5 到 −7.5。要嘛修好（例如 detail_scale 約 0.5 到 0.7，或讓 S 隨 grid 大小縮放），要嘛把主張收斂成「有上限的成本」。
3. **Reviewer 可能說「multi-scale 主要就是降解析度」。** m=3K/4 時 75% 的 tile 都是覆蓋 tile，而且 DocVQA 還輸 Low-Res-Fill。要反過來寫成研究發現：文件類需要全域覆蓋、HR 類需要局部細節，兩者在每題上互補（oracle 上限高出 4 到 13 分）。再加上「只有覆蓋 / 只有細節 / 兩者都有」的 ablation。
4. **ICML 偏好方法層面的 insight。** 要補：
   - 記憶體和 FLOPs 的複雜度分析（FoveaTile 是 O(K)，其他方法是 O(N)）。
   - 為什麼只在邊界 tile 跑中層 q_attn 就夠了，例如 prior 排名和 q_attn 排名的一致性分析。
   - 「prior 和 confidence 都選不出好的 m」這類負面結果放 appendix，回答「為什麼固定 m=0.75」。

## 6. 結論與時程

ICML 截稿通常在一月底，約 3.5 個月。如果縮放曲線做得出來（其他方法線性增加、FoveaTile 持平），再加上第二個模型，這篇才有機會；否則以目前的結果，比較適合先投 workshop 或次一級的會議。執行順序見 [03_experiment_plan.md](03_experiment_plan.md)。
