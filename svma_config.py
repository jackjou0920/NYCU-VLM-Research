"""
SVMA 的超參數：全部由 budget 決定，使用者只需要給 budget。

  K = budget // 256        能放進 KV 的 tile 數
  m = default_coverage_tiles(budget) = max(1, floor(0.75 K))
                           低解析度全圖覆蓋 tile 數（多尺度掃描裡三類任務平均最好的比例，
                           b1024 → 3、b2048 → 6；b3072 以上沒有另外驗證過）
  S = default_qattn_tiles(budget, m)
                           第二階段要跑 q_attn 的細節 tile 數（accuracy / TTFT 的取捨）

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

COVERAGE_FRAC = 0.75      # m / K


def default_coverage_tiles(budget: int, num_image_token: int = 256) -> int:
    """最終設定的覆蓋 tile 數 m = max(1, floor(COVERAGE_FRAC * K))。"""
    return max(1, int(budget // num_image_token * COVERAGE_FRAC))

# ── 每個 budget 的預設 S ──
# 來源：InfoVQA 隨機抽樣 600 張（seed 0）的校準實驗，每個 budget 掃過多個 S，
# 在 TTFT 限制內選 ANLS 最高的 S（out_v2_InternVL3_5-8B_infovqa_s0n600_b<budget>.json）。
QATTN_TILES_BY_BUDGET = {
    1024: 15,
    1536: 15,
    2048: 14,
    2560: 12,
    3072: 12,
    4096: 8,
    5120: 4,
}


def default_qattn_tiles(budget: int, coverage_tiles: int = 1) -> int:
    """budget（與覆蓋 tile 數 m）的預設 S。

    以 K = budget // 256（實際能放進 KV 的 tile 數）為座標：
      - K 跟表上某個 budget 相同：直接用表上的 S
      - 落在兩個表上 budget 之間：線性內插，無條件捨去（S 越大越慢，取小的一側）
      - 比表上最小的還小 / 比最大的還大：用最近兩點的斜率外插，再限制在 [0, 表上最大的 S]
    表是在 m = 1（細節名額 K - 1）下校準的。多尺度 m > 1 時細節名額只剩 K - m，S 跟著名額等比例縮小：
      S(K, m) = round(S(K) * (K - m) / (K - 1))，維持「每個名額打分幾塊」跟校準時相同；m = 1 時不變。
    （還沒有在 m > 1 下重新校準，是暫定規則。）
    """
    s = _table_qattn_tiles(budget)
    k = budget // 256
    if coverage_tiles <= 1 or k <= 1:
        return s
    return int(round(s * max(0, k - coverage_tiles) / (k - 1)))


def _table_qattn_tiles(budget: int) -> int:
    """QATTN_TILES_BY_BUDGET 查表 / 內插 / 外插（m = 1）。"""
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
    多尺度 SVMA 的細節名額是 K - m（m = 覆蓋 tile 數），原本的 SVMA 是 K - 1（m = 1）。
    a = 0 時（S <= 1 或沒有名額）不打分，b 也設成 0。"""
    s = max(0, int(qattn_tiles))
    a = min(max(0, int(slots)), s // 2)
    return (a, s - a) if a > 0 else (0, 0)


if __name__ == "__main__":
    import sys
    budgets = [int(x) for x in sys.argv[1:]] or [512, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
                                                  2816, 3072, 3584, 4096, 4608, 5120, 6144]
    if not sys.argv[1:]:
        print("多尺度 S(K, m)：")
        for b in (1024, 2048):
            k = b // 256
            print(f"  budget={b} (K={k}): " + "  ".join(
                f"m={m}->S={default_qattn_tiles(b, m)}" for m in sorted({1, 2, k // 4, k // 2, 3 * k // 4, k} - {0})))
    table_k = {b // 256 for b in QATTN_TILES_BY_BUDGET}
    for b in budgets:
        s = default_qattn_tiles(b)
        slots, a, bb = boundary_split(b, s)
        s = a + bb                                      # 實際打分的塊數（a = 0 時不打分）
        src = "表" if b // 256 in table_k else "內插"
        print(f"budget={b:5d} (K={b // 256:2d}) -> S={s:2d}（{src}）  "
              f"直接保留 {slots - a:2d} 塊、打分 {s:2d} 塊取 {a:2d} 塊")
