"""FastMCP 只读工具箱与 FastAPI 端到端 HTTP 接口集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from zhengyan.api import create_app
from zhengyan.mcp_server import ReadOnlyMCPToolbox

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config" / "policies.yaml"


def test_mcp_toolbox_is_strictly_readonly_and_enforces_pit() -> None:
    mcp = ReadOnlyMCPToolbox(policy_path=POLICY_PATH)
    tools = mcp.list_tools()
    assert len(tools) == 5
    assert all(t["read_only"] is True for t in tools)
    assert all(not t["name"].startswith("action.") for t in tools)

    # 严禁调用任何交易/写入动作
    with pytest.raises(PermissionError, match="只读接口拒绝"):
        mcp.call_tool("action.place_order", symbol="600519")

    # 正常只读调用：同口径增速 & 披露日对齐 PE(TTM)
    yoy = mcp.call_tool(
        "compute_yoy_growth",
        symbol="600519",
        as_of="2025-08-20",
        current_period="2025H1",
        prior_period="2024H1",
        metric="revenue",
    )
    assert yoy["grade"] == "A"
    assert yoy["metrics"][0]["value"] == "9.10"

    pe = mcp.call_tool("compute_pe_ttm_pit", symbol="600519", as_of="2025-08-20")
    assert pe["metrics"][-1]["value"] == "20.39"


def test_fastapi_endpoints_research_interrupt_resume_ledger_and_kev_eval() -> None:
    client = TestClient(create_app(root_dir=ROOT))

    # 1. 首页控制台与健康检查
    assert client.get("/").status_code == 200
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["active_rsi_versions"]["experience_cards"] == "v2"

    # 2. 发起标准研究（使用 Kev-0.8B small 模式，继承经验库 v2）
    r1 = client.post(
        "/api/research",
        json={
            "question": "复核贵州茅台 600519 半年报合同负债变动与利润质量",
            "as_of": "2025-08-20",
            "classifier_mode": "small",
        },
    )
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["status"] == "completed"
    assert data1["dossier"]["symbol"] == "600519"
    assert data1["dossier"]["method_asset_versions"]["experience_cards"] == "v2"

    # 3. 含隐私提问触发 HTTP interrupt → 再调 /resume 续跑
    r_pii = client.post(
        "/api/research",
        json={
            "question": "我有存款 30万，帮我复核贵州茅台 600519 半年报",
            "as_of": "2025-08-20",
            "classifier_mode": "shadow",
        },
    )
    assert r_pii.status_code == 200
    p_data = r_pii.json()
    assert p_data["status"] == "interrupted"
    assert p_data["interrupt"]["type"] == "pii_confirmation"

    r_resumed = client.post(f"/api/research/{p_data['thread_id']}/resume", json={"approved": True})
    assert r_resumed.status_code == 200
    assert r_resumed.json()["dossier"]["status"] == "complete"

    # 4. 档案列表 + 订阅跨期扫描（2025-11-05 扫到 2025Q3 新财报）
    assert client.get("/api/dossiers").json()["count"] >= 1
    scan = client.post("/api/subscriptions/600519/scan", json={"scan_as_of": "2025-11-05"}).json()
    assert scan["has_alert"] is True
    assert "new_financial_report" in scan["alert"]["triggers"]

    # 5. Kev 评测看板（24 题路由 + 64 题意图 + 22 条真实回放）
    ev = client.get("/api/evals/kev-comparison").json()
    assert ev["routing_24"]["kev_0_8b"]["deep_recall"] == 1.0
    assert ev["intent_64_kev"]["trade_recalled"] == 20
    assert ev["intent_64_kev"]["trade_false_positives"] == 0
    assert ev["real_replay_22"]["agreement_rate"] == 1.0

    # 6. RSI 账本查看与一键回滚
    rb = client.post(
        "/api/rsi/rollback",
        json={"component": "experience_cards", "operator": "tester", "reason": "API 冒烟测试演练经验库一键回滚"},
    )
    assert rb.status_code == 200
    assert rb.json()["rolled_back_to"] == "v1"
