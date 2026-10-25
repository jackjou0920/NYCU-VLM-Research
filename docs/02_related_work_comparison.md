# 02 相關工作比較：HiRED、GlobalCom2、RAP 與 FoveaTile

## 結論

FoveaTile 有創新點，但**不在「用全圖資訊決定每塊 tile 留多少」**，因為那正是 GlobalCom2 的核心。FoveaTile 能站得住的新穎性在三個地方：

1. **選擇時看問題**，而且分數來自 VLM 本身，不需要額外模型。
2. **在進 ViT 前就丟 tile**，被丟掉的區域不用跑 vision encoder。
3. **KV 串流建構且有上限**，O(K)，跟原圖 tile 數 N 無關。

## 1. 四個方法的機制比較

| 面向 | HiRED（AAAI'25） | GlobalCom2（AAAI'26） | RAP（ICML'25 oral） | **FoveaTile** |
|---|---|---|---|---|
| 解決的問題 | 高解析度帶來太多 token，要降低推論成本 | 同左 | 高解析度看不清細節，目標是提高 accuracy | 降低推論成本，並讓成本有上限 |
| 打分依據 | ViT 的 CLS attention | thumbnail 的 CLS attention，加上每個 crop 自己的 CLS attention（α=0.5 混合） | 外部 retriever（VisRAG）算問題和 crop 的相似度 | VLM 本身：coverage prior，再對邊界 tile 算中層 q_attn |
| 看不看問題 | ✗ | ✗（論文說可以再接上 FastV 或 SparseVLM） | ✓ | ✓ |
| 需不需要額外模型 | ✗ | ✗ | ✓（VisRAG retriever） | ✗ |
| 保留的單位 | 先分配每個 partition 的 budget，再挑 token | 先依 thumbnail 分配每個 crop 的保留比例，再挑 token | 整塊 crop | 整塊 tile |
| 空間結構 | token 是分散的 | token 是分散的 | Spatial-Awareness Layout：刪掉全空的行列後重新拼成小圖 | tile 依空間順序寫入 KV，加上 coverage 全圖 |
| 怎麼決定保留多少 | 固定 budget | 固定比例 R | RE-Search：類似 A\* 的樹搜尋，由「Yes」的機率決定 K | 固定 budget；m、S 由 budget 推出 |
| 被丟掉的區域還跑不跑 ViT | 全部跑 | 全部跑 | 每塊 crop 都要過 retriever，留下的再過 MLLM 的 ViT | **不跑** |
| 每題要跑幾次 LLM forward | 1 | 1 | 很多次（每個搜尋節點至少一次，上限 200 步） | 1 次主流程，加上 1 次短的 prior forward 和 S 塊的部分層 q_attn |
| KV 建構方式 | 一次 prefill | 一次 prefill | 一次 prefill（對重拼後的小圖） | **逐 tile 串流寫入，上限 O(K)** |
| 原本實驗用的模型 | LLaVA-NeXT | LLaVA-NeXT、LLaVA-OV；影片用 Qwen2-VL | LLaVA-v1.5、LLaVA-OV-0.5B | InternVL3.5-8B |
| Benchmark | 一般與文字類 | 一般 12 項，加上影片 4 項 | V\*、HR-Bench 4K/8K、MME-RealWorld | DocVQA、InfoVQA、HR-Bench 4K/8K |
| 報告的成本 | budget 20% 時 throughput 4.7×、latency −78% | R=10% 時 FLOPs 剩 9.1%、峰值記憶體 23.0→13.9 GB | 只報 throughput：每分鐘 4.2 題（13B） | TTFT 和 peak memory（扣掉權重後 0.3 到 0.6 GB） |

## 2. 設計空間中的位置

```
                    看問題
                      ↑
           RAP ●      │      ● FoveaTile
   （要額外模型、      │   （VLM 自己打分、
    多次 forward）     │     單次串流）
  慢 ←────────────────┼────────────────→ 快
                      │
                      │   ● GlobalCom2
                      │   ● HiRED
                      │   （只看 ViT、不看問題）
```

FoveaTile 的位置：「看問題」的能力向 RAP 看齊，成本向 HiRED / GlobalCom2 看齊，而且是四個裡唯一不用把整張圖都送進 ViT、KV 也有上限的方法。

## 3. 新穎性強弱

**較強，可以當主要貢獻：**

1. **看問題的 tile 選擇，分數來自 VLM 本身。** HiRED 和 GlobalCom2 只用 ViT 的視覺顯著性；在 InfoVQA、HR-Bench 這種答案只在一小塊區域、而且取決於問題的任務上，這是本質上的差別。要用 ablation 證明（把 FoveaTile 的 prior 換成 CLS attention，其他固定）。
2. **串流建構、成本有上限。** 另外兩篇都要先把 N 塊 crop 全部編碼完才開始丟。FoveaTile 只為留下的 tile 和邊界 tile 跑 ViT。要靠**縮放實驗**才能變成實際優勢。
3. **Cascade 式的邊界打分。** 只對排名卡在臨界的 S 塊付出 q_attn 的成本，是一個 accuracy 和 TTFT 可以調整的旋鈕。

**較弱，容易被 reviewer 打：**

- **「全域 + 細節」以及「全域資訊引導 budget 分配」**：GlobalCom2 已經做了。m 塊 coverage 加細節只能算 budget 分配的變體，不要寫成主要貢獻；改當分析：文件類需要覆蓋、HR 類需要細節。
- **以整塊 tile 為單位**：可能被認為比 token 級選擇更粗。要用已有的「空間局部性」實驗回應：整塊保留勝過挑分散的 token，而 merge 模式在 DocVQA 少了約 2.7 ANLS。

## 4. 跟每個方法比的優勢與風險

| 對手 | FoveaTile 的主要優勢 | FoveaTile 的風險 | 需要的證據 |
|---|---|---|---|
| HiRED | 看問題；丟掉的 tile 不跑 ViT；整塊 tile 保留空間結構 | 一般 benchmark 上，挑 token 可能比整塊 tile 精細 | prior ablation：打分換成 CLS attention，其他固定 |
| GlobalCom2 | 看問題；成本有上限；在 ViT 之前就丟 | 「全域資訊引導 budget 分配」它已經做了；它的記憶體降幅數字很漂亮 | 跟 GlobalCom2+FastV 比；在相同設定下比 vision FLOPs 和峰值記憶體 |
| RAP | 不用額外模型；單次 forward；TTFT 可能差一個數量級；KV 有上限 | RAP 在 HR-Bench、V\* 上的 accuracy 可能比 FoveaTile 高很多 | accuracy 對 TTFT 的 Pareto 圖；RAP-fixed-K 變體分開比較打分方式和搜尋的貢獻 |

**可以寫進論文的觀察：** RAP 的 RE-Search 用 MLLM 回答「Yes」的機率來決定 K。FoveaTile 的 confidence-gated cascade 實驗已經發現，answer confidence 沒辦法可靠地逐題選出最好的 m，在文件類任務上幾乎沒用。這可以當作對「用 confidence 自動決定保留多少」的實證討論，也回答為什麼 FoveaTile 用固定的 m。

## 5. 建議的貢獻寫法

> 第一個在 VLM **內部**、以**問題**為條件、**在 ViT 之前**就決定要保留哪些 tile 的串流框架。KV 和 vision 計算都有上限 O(K)，跟輸入解析度無關。

撐起這句話需要的實驗（細節見 [03_experiment_plan.md](03_experiment_plan.md)）：

1. Prior ablation：CLS attention（HiRED 式）、thumbnail 引導（GlobalCom2 式）、FoveaTile 的 q_attn。
2. 縮放曲線：tile 數 12 到 200 時，HiRED、GlobalCom2、FoveaTile 的 vision FLOPs、峰值記憶體、TTFT。
3. 直接比較：HiRED、GlobalCom2 移植到 InternVL；並在 LLaVA-OV 上跑 FoveaTile 對照原論文數字。

## Sources

- HiRED：[arXiv 2408.10945](https://arxiv.org/html/2408.10945v3)、[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32171)、code: github.com/hasanar1f/HiRED
- GlobalCom2：[arXiv 2501.05179 v5](https://arxiv.org/html/2501.05179v5)、[AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/37673/41635)、code: github.com/xuyang-liu16/GlobalCom2
- RAP：[arXiv 2503.01222](https://arxiv.org/html/2503.01222v2)、[ICML 2025 (PMLR v267)](https://proceedings.mlr.press/v267/wang25at.html)、code: github.com/DreamMr/RAP
