"""
FoveaTile 的超參數（凍結版本 v1.0，2026-10-17）：全部由 budget 決定，使用者只需要給 budget。
校準流程與依據見 docs/04_calibration_protocol.md。

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
    python foveatile_config.py                    # 印出常用 budget 的預設 m、S
    python foveatile_config.py 1000 1792 6144     # 印出指定 budget 的預設 m、S
"""
from __future__ import annotations

COVERAGE_FRAC = 0.5       # m / K（held-out 校準決定，見檔頭）
DEFAULT_BUDGET = 1536     # 論文的預設操作點（held-out 平均保留率 96%，b2048 沒有再進步），見 docs/03_experiment_plan.md 1.6


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


def split_slots(slots: int, qattn_tiles: int) -> tuple[int, int]:
    """slots 塊名額的邊界分配：回傳 (a, b)，a = min(slots, S // 2)，b = S - a。
    FoveaTile 的細節名額是 K - m（m = 覆蓋 tile 數）。
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
