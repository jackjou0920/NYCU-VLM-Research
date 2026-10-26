# 05 第 4 章（Experiments）呈現規劃（2026-10-26）

原則：圖表用最能呈現 FoveaTile 貢獻的方式（同成本比較、成本欄位跟 accuracy 並列、Pareto），但**所有方法、資料集、budget 一視同仁**：
不刪資料集、不只挑對自己有利的 budget 給 baseline；正文沒放的 budget 進附錄與 Pareto 圖。限制與弱點寫在文字（4.4、Conclusion）。

## 設定

- 模型：InternVL3.5-4B / 8B / 14B（m、S 全部沿用 8B 的 held-out 校準；attention 層號依層數換算，`foveatile_internvl.attn_layers`）
- 方法：Uncompressed、FastV、SparseVLM、VisionZip、IF-Prune、FoveaTile
- 測試集（6 個）：DocVQA val、InfoVQA val、MME-RW Lite、HR-Bench 4K、HR-Bench 8K、V* Bench
  - 欄位順序：文件類（DocVQA、InfoVQA）→ 真實場景（MME-RW Lite）→ 高解析度（HR-4K、HR-8K、V* Bench）
- 指標：各資料集原始分數 + 相對 uncompressed 的保留率；Avg 保留率；TTFT（s）；記憶體增量（GB）；TFLOPs

## 圖表

| 編號 | 內容 | 資料狀態 |
|---|---|---|
| Fig. 1（Intro teaser） | 左：TTFT / 記憶體 vs tile 數（7→188，FoveaTile 持平）；右：Avg 保留率 vs 記憶體 | 有 |
| **Table 2（主表，figure\* 全寬，SparseVLM 風格）** | 8B；區塊 = budget（b1536 預設、b2560、b3072）；每區塊 5 個方法，FoveaTile 最後一列、灰底；每格兩行（原始分數 / 保留率）；右側 Avg %（↓ 相對 100%）、TTFT、Mem、TFLOPs；每區塊每欄最佳者粗體 | 有（HR-8K 已補） |
| **Table 3（模型尺寸）** | 4B / 8B / 14B，b1536；區塊 = 模型；欄位同 Table 2（保留率為主，原始分數可放附錄） | 4B / 14B 執行中 |
| Fig. 4 | 8B：Avg 保留率 vs TTFT、vs 記憶體（b1024~b3072 全部點，含 b1024） | 有 |
| Fig. 5 + Table 4 | 解析度縮放：成本曲線；HR-8K 上限 48（25 塊）vs 上限 201（188 塊）的 accuracy 與成本 | 201 執行中 |
| Table 5 | 分析：按題型（HR cross / single、V* Bench position / attribute）＋ V* Bench 命中 / 沒命中 | 有 |
| Table 6 | Ablation（held-out）：只有覆蓋 / 只有細節 / 兩者；S=0 / 預設 / 加大；random prior；附 TTFT | random prior 待跑 |
| 附錄 | 所有 budget（b1024~b5120）× 所有資料集的完整表、CI、配對檢定；4B/14B 原始分數 | 有 / 執行中 |

## 已知要在文字裡誠實寫的

- b1024 時平均保留率低於 IF-Prune（8B：91.5 vs 94.5）；Fig. 4 會顯示這一點
- 單一小物件題（HR single、V* Bench attribute）落後；原因是低解析度 prior 漏掉小於一個 patch 的物件（命中時不輸）
- tile 數少（< 約 30 塊）時 TTFT 不比 FastV / VisionZip 快；優勢在記憶體與高解析度
- 4B / 14B 沿用 8B 的超參數，沒有重新校準

## GPU 排程

1. HR-8K 上限 201（2026-10-26 晚上完成）
2. `run_model_scale.sh`：4B → 14B × 6 測試集 × 6 方法 × b1536（約 1.5~2 天）
3. 4B / 14B 的效率量測（b1536，SparseVLM λ 用 2. 校準出的值）
4. random prior ablation（held-out，8B）
5. （有時間才做）decode / total latency、同時請求數
