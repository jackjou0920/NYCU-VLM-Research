"""
FoveaTile 的超參數：全部由 budget 決定，使用者只需要給 budget。
（方法原名 SVMA；檔名與 import 名稱沿用 svma_config，跟既有腳本相容。）

  K = budget // 256        能放進 KV 的 tile 數
  m = default_coverage_tiles(budget) = max(1, floor(0.5 K))
                           低解析度全圖覆蓋 tile 數。2026-10-14 在 held-out 校準集（InfoVQA train 600 題 +
                           MME-RealWorld 完整版扣掉 Lite 600 題）上決定：m/K 在 0.5~1.0 平均分數打平，
                           0.5 的最差任務保留率最高（b2048 85% vs 純覆蓋 74%），見 docs/03_experiment_plan.md 1.5
                           （b1024 → 2、b1536 → 3、b2048 → 4；舊值 0.75 是在測試集上選的，已棄用）
  S = default_qattn_tiles(budget, m)
                           第二階段要跑 q_attn 的細節 tile 數（accuracy / TTFT 的取捨）。
                           2026-10-14 在同一組 held-out 校準集上決定（見 QATTN_TILES_BY_BUDGET 的說明）

邊界打分（boundary scoring）：細節名額 slots = K - m：
    a = min(slots, S // 2)，b = S - a
    1. prior 排名前 (slots-a) 塊：直接保留（只跑 ViT，不打分）
    2. 接下來 S 塊：跑 ViT + q_attn，取分數最高的 a 塊
    3. 其餘 tile：直接丟，連 ViT 都不跑
S = 0 時不打分，只用 prior 取前 slots 塊。

用法：
    python svma_config.py                    # 印出常用 budget 的預設 S
    python svma_config.py 1000 1792 6144     # 印出指定 budget 的預設 S
"""
from __future__ import annotations

COVERAGE_FRAC = 0.5       # m / K（held-out 校準決定，見檔頭）
DEFAULT_BUDGET = 1536     # 論文的預設操作點（held-out 平均保留率 96%，b2048 沒有再進步），見 docs/03_experiment_plan.md 1.6

# ── tile 打分的 query 與聚合（v1.1 候選，2026-10-25；docs/04_calibration_protocol.md §7b）──
#   QUERY_MODE：prior / q_attn 用 text_after 的哪些位置當 query
#     "all"     ：text_after 全部（問題 + 選項 + 作答指示 + template）＝ v1.0
#     "stem"    ：只用問題主幹（causal，主幹看不到後面的選項 → 跟選項順序無關）
#     "stemopt" ：另組 [主幹 + 依字母排序的選項文字] 當打分 query（同樣跟選項順序無關）
#     "sortopt" ：選項文字排序後重組整個 prompt，query 用全部列（同 v1.0；沒有選項的題目 = "all"）
#   TILE_AGG：一塊細節 tile 內 patch（prior）/ token（q_attn）分數怎麼併成 tile 分數
#     "mean"（v1.0）、"max"、"top<f>"（前 f 比例取平均，例如 "top0.1"）
QUERY_MODE = "all"
TILE_AGG = "mean"

# ── prior 的 head 選擇（A1）與問題無關基準校正（A2）（候選，2026-10-25；docs/04_calibration_protocol.md §7c）──
#   PRIOR_HEADS：None = v1.0（第 21~28 層全部 head 平均）；"a1" = PRIOR_HEADS_A1 這些 (layer, head)
#   PRIOR_BASE：None = v1.0；"nl" = text_after 第一列（不多算）；"gen" = 通用問題再跑一次 forward
#   PRIOR_BASE_MODE："ratio"（p_q / (p_base + ε)）或 "sub"（p_q - p_base）
PRIOR_HEADS = None
PRIOR_BASE = None
PRIOR_BASE_MODE = "ratio"
# 2026-10-25 在合成 bbox 校準集 fit 上選出（單 head 召回率前 2；val：v1.0 72.7 → A1-top2+A2-nl-ratio 76.9）
PRIOR_HEADS_A1: tuple = ((18, 26), (19, 18))


def default_coverage_tiles(budget: int, num_image_token: int = 256) -> int:
    """最終設定的覆蓋 tile 數 m = max(1, floor(COVERAGE_FRAC * K))。"""
    return max(1, int(budget // num_image_token * COVERAGE_FRAC))

# ── 每個 budget 的預設 S（在預設 m = floor(K/2) 下）──
# 來源：2026-10-14 階段 B 的 held-out 校準（InfoVQA train 600 題 + MME-RealWorld 完整版扣掉 Lite 600 題，
# 每個 budget 掃 S ∈ {0, S0/2, S0, 1.5 S0, 2 S0}，S0 = 舊規則的值），見 docs/03_experiment_plan.md 1.5。
#   - S = 0（只靠 prior）在局部細節任務顯著較差（Remote Sensing b1024 / b1536 −10.8，Monitoring b2560 −7.5）
#   - S 超過 S0/2 之後沒有可靠的增益（配對 CI 大多包含 0），只多花 TTFT → 一律取 S0/2
#   （原本事先定的「最差任務在最佳 1 個百分點內取最小 S」規則被 120 題 subtask 的雜訊主導，
#     b1536 / b3072 會選到 13 / 14，跟 S0/2 的差距不顯著；經使用者同意改用 S0/2，論文附錄要註明）
QATTN_TILES_BY_BUDGET = {
    1024: 5,
    1536: 4,
    2048: 4,
    2560: 3,
    3072: 3,
}

# 舊表（m = 1 下、在 InfoVQA val = 測試集上校準，已棄用；只留著對照舊結果檔用，程式不再讀它）
LEGACY_QATTN_TILES_M1 = {1024: 15, 1536: 15, 2048: 14, 2560: 12, 3072: 12, 4096: 8, 5120: 4}


def default_qattn_tiles(budget: int, coverage_tiles: int | None = None) -> int:
    """budget（與覆蓋 tile 數 m）的預設 S。

    以 K = budget // 256（實際能放進 KV 的 tile 數）為座標查 QATTN_TILES_BY_BUDGET：
      - K 跟表上某個 budget 相同：直接用表上的 S
      - 落在兩個表上 budget 之間：線性內插，無條件捨去（S 越大越慢，取小的一側）
      - 比表上最小的還小 / 比最大的還大：用最近兩點的斜率外插，再限制在 [0, 表上最大的 S]
    表是在預設 m（default_coverage_tiles）下校準的。coverage_tiles 給別的 m（ablation / 掃描用）時，
    S 跟著細節名額 K - m 等比例縮放：S = round(S_table * (K - m) / (K - m_default))；None = 預設 m。
    """
    s = _table_qattn_tiles(budget)
    k = budget // 256
    m_def = default_coverage_tiles(budget)
    if coverage_tiles is None or coverage_tiles == m_def:
        return s
    if k - m_def <= 0:
        return 0
    return int(round(s * max(0, k - coverage_tiles) / (k - m_def)))


def grid_qattn_tiles(slots: int, n_detail: int, s_base: int, grid_frac: float | None) -> int:
    """候選規則（3b，2026-10-17，尚未採用為預設）：打分窗口隨細節網格大小放大。
    邊界分配下，prior 前 (slots - a) 塊直接保留、接下來 S 塊打分（a = min(slots, S // 2)），
    兩者合計 = prior 前 (slots - a + S) 名。這裡從 s_base 往上加 S，直到合計 >= ceil(grid_frac * n_detail)
    （最多到全部細節 tile）。grid_frac 為 None / 0 時回傳 s_base（= 目前的預設）。
    依據：V* 診斷顯示 FoveaTile 沒選中目標 tile 的主因是 prior 沒把它排進窗口，而目標的 prior 排名
    跟網格大小成比例（中位數在前 10~12%），固定的 S 對大網格太小。"""
    if not grid_frac or slots <= 0 or n_detail <= 0:
        return s_base
    need = min(n_detail, -(-int(grid_frac * n_detail * 1000) // 1000))   # ceil，避免浮點誤差
    s = max(0, int(s_base))
    while s < 2 * n_detail:
        a, _ = split_slots(slots, s)
        if (slots - a) + s >= need:
            break
        s += 1
    return s


def _table_qattn_tiles(budget: int) -> int:
    """QATTN_TILES_BY_BUDGET 查表 / 內插 / 外插（預設 m 下）。"""
    pts = sorted((b // 256, s) for b, s in QATTN_TILES_BY_BUDGET.items())
    ks = [k for k, _ in pts]
    ss = [s for _, s in pts]
    k = budget // 256
    if k <= 1:                                          # 只放得下 thumbnail，沒有 grid tile 可選
        return 0
    if k in ks:
        return ss[ks.index(k)]
    if k < ks[0]:
        i = 0
    elif k > ks[-1]:
        i = len(ks) - 2
    else:
        i = max(j for j in range(len(ks) - 1) if ks[j] < k)
    k0, k1, s0, s1 = ks[i], ks[i + 1], ss[i], ss[i + 1]
    s = int(s0 + (s1 - s0) * (k - k0) / (k1 - k0))      # int() 對正數 = 無條件捨去
    return max(0, min(max(ss), s))


def boundary_split(budget: int, qattn_tiles: int, num_image_token: int = 256) -> tuple[int, int, int]:
    """回傳 (slots = K-1, a, b)：a = min(K-1, S // 2)，b = S - a。
    a = 0 時（S <= 1，或 K = 1 沒有 grid tile 可選）邊界的 tile 一塊都留不下，打分沒有意義，b 也設成 0。"""
    slots = max(0, budget // num_image_token - 1)
    return (slots, *split_slots(slots, qattn_tiles))


def split_slots(slots: int, qattn_tiles: int) -> tuple[int, int]:
    """slots 塊名額的邊界分配：回傳 (a, b)，a = min(slots, S // 2)，b = S - a。
    FoveaTile 的細節名額是 K - m（m = 覆蓋 tile 數），v0（只有官方縮圖）是 K - 1（m = 1）。
    a = 0 時（S <= 1 或沒有名額）不打分，b 也設成 0。"""
    s = max(0, int(qattn_tiles))
    a = min(max(0, int(slots)), s // 2)
    return (a, s - a) if a > 0 else (0, 0)


if __name__ == "__main__":
    import sys
    budgets = [int(x) for x in sys.argv[1:]] or [1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 5120]
    table_k = {b // 256 for b in QATTN_TILES_BY_BUDGET}
    for b in budgets:
        k, m = b // 256, default_coverage_tiles(b)
        s = default_qattn_tiles(b)
        slots = k - m
        a, bb = split_slots(slots, s)
        src = "表" if k in table_k else "內插 / 外插"
        print(f"budget={b:5d} (K={k:2d}) -> m={m:2d}、細節名額 {slots:2d}、S={a + bb:2d}（{src}）  "
              f"直接保留 {slots - a:2d} 塊、打分 {a + bb:2d} 塊取 {a:2d} 塊")
