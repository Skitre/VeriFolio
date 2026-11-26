"""LangGraph 10 节点固定状态图端到端测试。

覆盖核心工程与治理场景：
1. 标准研究全链路（10 节点全走通 + 只追加版本档案 + 审计 Trace）
2. 快查概念解释（不取财报、不搜网页、直接起草核验提交）
3. 含个人敏感信息 → authorize 节点原生 interrupt() 暂停 → Command(resume) 断点续跑（不重跑 intake；分别测试同意与拒绝）
4. 下单意图 → 附加边界声明 + 至少走标准研究
5. 模型幻觉注入 → verify 规则核验器精准拦下，幻觉主张绝不进入 verified_claims
6. 首轮证据缺口 → quality_gate 自动退回 plan 补证 1 轮后通过
7. 预算耗尽 → 绝不跳过 verify，按已有证据如实提交 PARTIAL 部分结论
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.core.enums import (
    DossierStatus,
    QualityGateAction,
    ResearchTier,
    UserIntent,
    VerificationStatus,
)
from zhengyan.graph import ResearchRunner
from zhengyan.llm import MockLLMClient

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config" / "policies.yaml"
SEARCH_PATH = ROOT / "config" / "search_sources.yaml"


def _make_runner(llm: MockLLMClient | None = None) -> ResearchRunner:
    return ResearchRunner(
        as_of=date(2025, 8, 20),
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        llm_client=llm or MockLLMClient(),
    )


def test_standard_research_full_10_node_flow() -> None:
    runner = _make_runner()
    out = runner.start("复核贵州茅台 600519 最新一期半年报，利润质量如何", thread_id="th-std-1")

    assert runner.runtime.node_calls == [
        "intake", "authorize", "route", "plan", "collect", "search", "draft", "verify", "quality_gate", "commit"
    ]
    dossier = out["dossier"]
    assert dossier is not None
    assert dossier.symbol == "600519"
    assert dossier.tier == ResearchTier.STANDARD
    assert dossier.status == DossierStatus.COMPLETE
    assert dossier.version == 1
    assert len(dossier.verified_claims) >= 3
    assert len(dossier.excluded_claims) == 0
    assert out["quality_gate"].action == QualityGateAction.PASS


def test_quick_concept_skips_collect_and_search() -> None:
    runner = _make_runner()
    out = runner.start("什么是市盈率", thread_id="th-concept-1")

    assert "collect" not in runner.runtime.node_calls
    assert "search" not in runner.runtime.node_calls
    assert runner.runtime.node_calls == ["intake", "authorize", "route", "draft", "verify", "quality_gate", "commit"]
    assert out["dossier"].tier == ResearchTier.QUICK
    assert out["dossier"].intent == UserIntent.CONCEPT
    assert out["dossier"].status == DossierStatus.COMPLETE


def test_pii_interrupt_and_resume_approved_without_rerunning_intake() -> None:
    llm = MockLLMClient()
    runner = _make_runner(llm=llm)
    tid = "th-pii-approve"

    # 第一次调用：authorize 检测到个人敏感信息，原地 interrupt 暂停
    runner.start("我有存款 30万，房贷每月 8000元，帮我复核贵州茅台半年报利润质量", thread_id=tid)
    pending = runner.get_pending_interrupts(thread_id=tid)
    assert len(pending) == 1
    assert pending[0]["type"] == "pii_confirmation"
    assert runner.runtime.node_calls == ["intake", "authorize"]

    # 用户确认同意后，通过 Command(resume=...) 从检查点续跑
    resumed = runner.resume({"approved": True}, thread_id=tid)
    # intake 只执行了 1 次，证明是从检查点续跑而非从头重跑！
    assert runner.runtime.node_calls.count("intake") == 1
    assert resumed["dossier"].status == DossierStatus.COMPLETE
    # 发往 LLM 的所有 prompt 均已经过脱敏，不含 30万 / 8000元
    assert all("30万" not in p and "8000元" not in p for p in llm.received_prompts)


def test_pii_interrupt_and_resume_rejected_aborts_safely() -> None:
    runner = _make_runner()
    tid = "th-pii-reject"

    runner.start("我的持仓 50万，复核贵州茅台半年报", thread_id=tid)
    assert len(runner.get_pending_interrupts(thread_id=tid)) == 1

    resumed = runner.resume({"approved": False}, thread_id=tid)
    assert resumed["dossier"].status == DossierStatus.ABORTED
    assert "collect" not in runner.runtime.node_calls
    assert "search" not in runner.runtime.node_calls


def test_trade_order_intent_adds_boundary_notice_and_minimum_standard_tier() -> None:
    runner = _make_runner()
    out = runner.start("帮我买入 100 股贵州茅台", thread_id="th-trade-1")

    dossier = out["dossier"]
    assert dossier.intent == UserIntent.TRADE_ORDER
    assert dossier.tier == ResearchTier.STANDARD
    assert dossier.boundary_notice is not None
    assert "不给出买卖指令" in dossier.boundary_notice


def test_hallucinated_claim_blocked_by_verify_never_enters_verified_claims() -> None:
    runner = _make_runner(llm=MockLLMClient(inject_hallucination=True))
    out = runner.start("复核贵州茅台半年报利润质量", thread_id="th-hallucination")

    dossier = out["dossier"]
    # 幻觉主张“毛利率同比提升 3 个百分点”被核验器判为 DOUBTFUL，绝不能进入 verified_claims
    assert all(c.claim_id != "CLM-HALLUCINATED" for c in dossier.verified_claims)
    assert any(c.claim_id == "CLM-HALLUCINATED" for c in dossier.excluded_claims)
    ver_hal = next(r for r in dossier.verification_results if r.claim_id == "CLM-HALLUCINATED")
    assert ver_hal.status == VerificationStatus.DOUBTFUL
    assert dossier.status == DossierStatus.PARTIAL


def test_quality_gate_retries_evidence_once_then_passes() -> None:
    runner = _make_runner(llm=MockLLMClient(simulate_gap_on_round_0=True))
    out = runner.start("复核贵州茅台半年报利润质量", thread_id="th-retry-1")

    # plan → collect → search → draft → verify → quality_gate 走了两轮（首轮 + 补证 1 轮）
    assert runner.runtime.node_calls.count("plan") == 2
    assert runner.runtime.node_calls.count("verify") == 2
    assert out["supplement_rounds"] == 1
    assert out["dossier"].status == DossierStatus.COMPLETE


def test_high_disagreement_triggers_deep_escalation_flag() -> None:
    runner = _make_runner(llm=MockLLMClient(simulate_high_disagreement=True))
    out = runner.start("复核贵州茅台半年报利润质量", thread_id="th-escalate")

    assert out["quality_gate"].action == QualityGateAction.ESCALATE_DEEP
    assert out["dossier"].needs_deep_escalation is True
    # Stage 4 接入深研子图后，高分歧标准研究会自动升级进入深研并完成裁决
    assert "deep_research" in runner.runtime.node_calls
    assert out["dossier"].tier == ResearchTier.DEEP


def test_budget_exhaustion_still_runs_verify_and_commits_partial() -> None:
    runner = _make_runner()
    # 预先把搜索预算消耗完，模拟预算耗尽场景
    runner.harness.budget.searches_used = runner.harness.budget.max_searches
    out = runner.start("复核贵州茅台半年报利润质量", thread_id="th-budget-out")

    # 即使搜索预算耗尽，依然完整经过 verify 与 quality_gate，绝不跳过核验
    assert "verify" in runner.runtime.node_calls
    assert "quality_gate" in runner.runtime.node_calls
    assert out["dossier"] is not None
    assert any("预算耗尽" in g for g in out["dossier"].evidence_gaps)
