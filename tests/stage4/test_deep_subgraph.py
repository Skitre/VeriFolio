"""多专家争议深研子图核心机制单测（Send 并行、首轮隔离、反证官、非投票裁决、高危争议 interrupt）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from langgraph.types import Command

from zhengyan.core.enums import DossierStatus, ResearchTier, VerificationStatus
from zhengyan.data import MockMarketDataProvider, compute_pe_ttm_pit, compute_ratio, compute_yoy_growth
from zhengyan.core.enums import MetricUnit
from zhengyan.deep import (
    DeepSubgraphRuntime,
    MockDeepResearchEngine,
    compile_deep_research_subgraph,
    count_independent_sources,
)
from zhengyan.graph import ResearchRunner
from zhengyan.graph.workflow import create_default_checkpointer
from zhengyan.harness import HarnessExecutor, HarnessPolicyConfig
from zhengyan.search import SourcePolicy

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config" / "policies.yaml"
SEARCH_PATH = ROOT / "config" / "search_sources.yaml"


def _build_maotai_shared_a_cards(as_of: date) -> dict:
    provider = MockMarketDataProvider()
    cards = {c.evidence_id: c for c in provider.to_evidence_cards("600519", as_of)}
    cl_yoy = compute_yoy_growth(
        cards["EV-FIN-600519-2025H1"], cards["EV-FIN-600519-2024H1"], "contract_liabilities", as_of=as_of
    )
    ocf_r = compute_ratio(
        cards["EV-FIN-600519-2025H1"],
        "ocf",
        "net_profit",
        output_name="ocf_to_np",
        output_display_name="净现比",
        output_unit=MetricUnit.MULTIPLE,
    )
    price = provider.get_price_on_or_before("600519", as_of)
    assert price is not None
    pe = compute_pe_ttm_pit(provider.get_financial_statements("600519", as_of), price, as_of)
    for c in (cl_yoy, ocf_r, pe):
        cards[c.evidence_id] = c
    return cards


def test_send_parallel_convene_max_5_and_maotai_contract_liabilities_challenge_upheld() -> None:
    """还原文档贵州茅台真实深研：
    - 主席最多召集 5 位专家，Send 并行下发，每人独立 2 搜 2 抓子预算
    - 基本面专家“合同负债下降 42% 说明需求走弱”数字对得上、核验器放行，但被反证官质询成立 → 主席拦截不入结论！
    - 估值专家未经财报证实的派生估算数字被规则核验器标为 DOUBTFUL → 不入结论！
    - 多空分歧作为保留意见写入裁决，不强行抹平。
    """
    as_of = date(2025, 8, 20)
    harness = HarnessExecutor(
        policy=HarnessPolicyConfig.from_yaml(POLICY_PATH),
        tier=ResearchTier.DEEP,
        as_of=as_of,
    )
    rt = DeepSubgraphRuntime(
        harness=harness,
        search_policy=SourcePolicy.from_yaml(SEARCH_PATH),
        engine=MockDeepResearchEngine(),
    )
    sub = compile_deep_research_subgraph(rt, checkpointer=create_default_checkpointer())
    out = sub.invoke(
        {
            "question": "贵州茅台合同负债下降 42%，估值是否合理、还要不要继续持有？",
            "symbol": "600519",
            "as_of": as_of.isoformat(),
            "shared_evidence_store": _build_maotai_shared_a_cards(as_of),
            "expert_submissions": [],
        },
        config={"configurable": {"thread_id": "sub-mt-1"}},
    )

    assert len(out["selected_experts"]) <= 5
    assert len(out["expert_submissions"]) == len(out["selected_experts"])
    assert all(
        s.sub_budget.searches_used <= 2 and s.sub_budget.fetches_used <= 2
        for s in out["expert_submissions"]
    )
    assert len(out["challenges"]) <= 8

    # 关键验证 1：合同负债因果主张“数字正确”通过了规则核验，但被反证官质询成立拦在最终结论外！
    vr_cl = next(r for r in out["verification_results"] if r.claim_id == "EXP-FUN-CL-CAUSAL")
    assert vr_cl.status == VerificationStatus.VERIFIED
    adj = out["adjudication"]
    assert "CHG-01" in adj.upheld_challenge_ids
    assert "EXP-FUN-CL-CAUSAL" in adj.blocked_claim_ids
    assert all(c.claim_id != "EXP-FUN-CL-CAUSAL" for c in out["final_claims"])

    # 关键验证 2：估值专家推断派生数字被规则核验器标为 DOUBTFUL，拦在结论外
    vr_est = next(r for r in out["verification_results"] if r.claim_id == "EXP-VAL-DERIVED-EST")
    assert vr_est.status == VerificationStatus.DOUBTFUL
    assert all(c.claim_id != "EXP-VAL-DERIVED-EST" for c in out["final_claims"])

    # 关键验证 3：多空分歧保留为保留意见
    assert len(adj.dissenting_opinions) >= 1


def test_round1_isolation_violation_rejected_by_blackboard_verifier() -> None:
    """专家首轮违规引用其他专家的私有证据 ID → 核验器直接 REJECTED。"""
    as_of = date(2025, 8, 20)
    harness = HarnessExecutor(
        policy=HarnessPolicyConfig.from_yaml(POLICY_PATH),
        tier=ResearchTier.DEEP,
        as_of=as_of,
    )
    rt = DeepSubgraphRuntime(
        harness=harness,
        search_policy=SourcePolicy.from_yaml(SEARCH_PATH),
        engine=MockDeepResearchEngine(inject_isolation_violation=True),
    )
    sub = compile_deep_research_subgraph(rt, checkpointer=create_default_checkpointer())
    out = sub.invoke(
        {
            "question": "贵州茅台估值合理吗",
            "symbol": "600519",
            "as_of": as_of.isoformat(),
            "shared_evidence_store": _build_maotai_shared_a_cards(as_of),
            "expert_submissions": [],
        },
        config={"configurable": {"thread_id": "sub-iso-1"}},
    )

    assert len(out["isolation_violations"]) == 1
    vr_leak = next(r for r in out["verification_results"] if r.claim_id == "EXP-IND-LEAK")
    assert vr_leak.status == VerificationStatus.REJECTED
    assert any("首轮隔离" in r for r in vr_leak.reasons)
    assert all(c.claim_id != "EXP-IND-LEAK" for c in out["final_claims"])


def test_non_voting_reprint_dedup_counts_three_experts_same_article_as_one_source() -> None:
    """3 位专家各自从证券时报、新浪财经、东方财富抓到同一篇转载稿，独立来源计数必须合并为 1（打破多数投票）。"""
    as_of = date(2025, 8, 20)
    harness = HarnessExecutor(
        policy=HarnessPolicyConfig.from_yaml(POLICY_PATH),
        tier=ResearchTier.DEEP,
        as_of=as_of,
    )
    rt = DeepSubgraphRuntime(
        harness=harness,
        search_policy=SourcePolicy.from_yaml(SEARCH_PATH),
        engine=MockDeepResearchEngine(reprint_echo_mode=True),
    )
    sub = compile_deep_research_subgraph(rt, checkpointer=create_default_checkpointer())
    out = sub.invoke(
        {
            "question": "贵州茅台渠道打款节奏变化分析",
            "symbol": "600519",
            "as_of": as_of.isoformat(),
            "shared_evidence_store": _build_maotai_shared_a_cards(as_of),
            "expert_submissions": [],
        },
        config={"configurable": {"thread_id": "sub-echo-1"}},
    )

    bb = out["blackboard_evidence"]
    web_eids = [eid for eid in bb if eid.startswith("EV-EXP-")]
    assert len(web_eids) >= 3
    # 所有跨专家抓到的同篇转载稿被统一归入同一事件簇，合并后独立来源数 == 1！
    assert count_independent_sources(web_eids, bb) == 1


def test_unresolved_high_severity_dispute_interrupts_for_human_review() -> None:
    """高严重度且未解决争议 → 提交前触发 interrupt 暂停等人复核：
    - 确认批准 → 以 PARTIAL 部分结论提交
    - 拒绝提交 → ABORTED 终止不提交结论
    """
    as_of = date(2025, 8, 20)

    # Case A: 人工确认批准 → PARTIAL 部分结论提交
    runner_ok = ResearchRunner(
        as_of=as_of,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        deep_engine=MockDeepResearchEngine(inject_unresolved_high_dispute=True),
    )
    tid_ok = "th-dispute-approve"
    runner_ok.start("贵州茅台现在估值合理吗，还要不要继续持有", thread_id=tid_ok)
    pending = runner_ok.get_pending_interrupts(thread_id=tid_ok)
    assert len(pending) == 1
    assert pending[0]["type"] == "unresolved_high_severity_dispute"

    res_ok = runner_ok.resume({"approved": True}, thread_id=tid_ok)
    assert res_ok["dossier"].status == DossierStatus.PARTIAL
    assert len(res_ok["dossier"].dissenting_opinions) >= 1

    # Case B: 人工拒绝 → ABORTED
    runner_no = ResearchRunner(
        as_of=as_of,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        deep_engine=MockDeepResearchEngine(inject_unresolved_high_dispute=True),
    )
    tid_no = "th-dispute-reject"
    runner_no.start("贵州茅台现在估值合理吗，还要不要继续持有", thread_id=tid_no)
    assert len(runner_no.get_pending_interrupts(thread_id=tid_no)) == 1

    res_no = runner_no.resume({"approved": False}, thread_id=tid_no)
    assert res_no["dossier"].status == DossierStatus.ABORTED
