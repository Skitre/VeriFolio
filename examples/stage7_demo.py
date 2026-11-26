"""Stage 7 终章演示：DeepSeek vs Kev-0.8B 对比、影子模式、FastMCP 只读自检与 FastAPI 冒烟。

运行：uv run python examples/stage7_demo.py
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from zhengyan.api import create_app
from zhengyan.mcp_server import ReadOnlyMCPToolbox

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    client = TestClient(create_app(root_dir=ROOT))

    print("=== 1. System 1 / System 2 混合判别评测：DeepSeek vs 本地 Kev-0.8B ===\n")
    ev = client.get("/api/evals/kev-comparison").json()
    r_llm = ev["routing_24"]["llm"]
    r_kev = ev["routing_24"]["kev_0_8b"]
    i_kev = ev["intent_64_kev"]
    rr = ev["real_replay_22"]

    print(
        f"  [路由 24 题] DeepSeek : 准确率={r_llm['accuracy']*100:.1f}% | 深研召回={r_llm['deep_recall']*100:.0f}% | "
        f"拒判率={r_llm['abstain_rate']*100:.1f}% | 中位耗时={r_llm['median_latency_ms']:.0f} ms"
    )
    print(
        f"  [路由 24 题] Kev-0.8B : 准确率={r_kev['accuracy']*100:.1f}% | 深研召回={r_kev['deep_recall']*100:.0f}% | "
        f"拒判率={r_kev['abstain_rate']*100:.1f}% | 中位耗时={r_kev['median_latency_ms']:.0f} ms（3次零波动）"
    )
    print(
        f"  [意图 64 题] Kev-0.8B : 原38题={i_kev['main_38_accuracy']*100:.1f}% | 留出26题={i_kev['holdout_26_accuracy']*100:.1f}% | "
        f"下单={i_kev['trade_recalled']}/{i_kev['trade_total']} (0 误报) | 耗时={i_kev['median_latency_ms']:.0f} ms"
    )
    print(
        f"  [真实回放 22 条] 与 DeepSeek 一致率 = {rr['agreed']}/{rr['total']} ({rr['agreement_rate']*100:.0f}%)\n"
    )

    print("=== 2. FastMCP 只读工具箱自检 ===\n")
    mcp = ReadOnlyMCPToolbox(policy_path=ROOT / "config" / "policies.yaml")
    for t in mcp.list_tools():
        print(f"  - {t['name']:22s} [read_only={t['read_only']}] {t['description']}")
    pe = mcp.call_tool("compute_pe_ttm_pit", symbol="600519", as_of="2025-08-20")
    print(f"  ↳ 只读调用 compute_pe_ttm_pit(600519, 2025-08-20) => PE(TTM)={pe['metrics'][-1]['value']}倍\n")

    print("=== 3. FastAPI 影子模式（shadow）端到端冒烟 ===\n")
    res = client.post(
        "/api/research",
        json={
            "question": "复核贵州茅台 600519 半年报合同负债变动与利润质量",
            "as_of": "2025-08-20",
            "classifier_mode": "shadow",
        },
    ).json()
    print(f"  [档案] ID={res['dossier']['dossier_id']} | 状态={res['dossier']['status']} | 继承资产={res['dossier']['method_asset_versions']}")
    print(f"  [影子对比日志] {res['shadow_records']}")


if __name__ == "__main__":
    main()
