"""综合质量分公式校验、3 例单 Agent vs 多专家对照评测、以及主图直达/升级单次约束测试。"""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal
from pathlib import Path

from zhengyan.core.enums import EvidenceGrade, MetricUnit, ResearchTier, VerificationStatus
from zhengyan.core.models import Claim, NumberCheckDetail, VerificationResult
from zhengyan.deep import (
    compute_composite_quality_score,
    evaluate_ab_comparison,
    load_ab_eval_cases,
)
from zhengyan.graph import ResearchRunner
from zhengyan.llm import MockLLMClient

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config" / "policies.yaml"
SEARCH_PATH = ROOT / "config" / "search_sources.yaml"
AB_CASES_PATH = ROOT / "evals" / "golden" / "ab_eval_cases.json"


def test_composite_quality_score_exact_weights_formula() -> None:
    """精确检验文档第 16 题权重：30%核验通过 + 30%关键项覆盖 + 25%数字复算 + 5%保留分歧(封顶3) + 10%质量门控。"""
    claims = [
        Claim(claim_id="C1", text="营业收入 893.89 亿元，现金流稳健", evidence_ids=["E1"]),
        Claim(claim_id="C2", text="当前市盈率 PE 约 20.4 倍", evidence_ids=["E2"]),
        Claim(claim_id="C3", text="估算未来增速 15%", evidence_ids=["E1"]),
    ]
    vrs = [
        VerificationResult(
            claim_id="C1",
            status=VerificationStatus.VERIFIED,
            number_checks=[
                NumberCheckDetail(
                    raw_text="893.89亿元", parsed_value=Decimal("893.89"), unit=MetricUnit.HUNDRED_MILLION_CNY,
                    tolerance=Decimal("0.005"), matched=True, matched_grade=EvidenceGrade.A,
                )
            ],
        ),
        VerificationResult(
            claim_id="C2",
            status=VerificationStatus.VERIFIED,
            number_checks=[
                NumberCheckDetail(
                    raw_text="20.4倍", parsed_value=Decimal("20.4"), unit=MetricUnit.MULTIPLE,
                    tolerance=Decimal("0.05"), matched=True, matched_grade=EvidenceGrade.A,
                )
            ],
        ),
        VerificationResult(
            claim_id="C3",
            status=VerificationStatus.DOUBTFUL,
            number_checks=[
                NumberCheckDetail(
                    raw_text="15%", parsed_value=Decimal("15"), unit=MetricUnit.PERCENT,
                    tolerance=Decimal("0.5"), matched=False,
                )
            ],
        ),
    ]
    # 核验通过 2/3, 覆盖 profit_quality+cashflow+valuation = 3/5, 数字复算 2/3, 保留分歧 4(封顶3->3/3=1.0), 门控通过=1
    bd = compute_composite_quality_score(
        claims=claims,
        verification_results=vrs,
        required_dimensions=["profit_quality", "cashflow", "valuation", "channel_industry", "risk_counterpoint"],
        preserved_dissents_count=4,
        quality_gate_passed=True,
    )
    expected = 0.30 * (2 / 3) + 0.30 * (3 / 5) + 0.25 * (2 / 3) + 0.05 * 1.0 + 0.10 * 1.0
    assert math.isclose(bd.composite_score, expected, abs_tol=1e-5)


def test_ab_comparison_3_cases_isolated_namespaces_and_admission_criteria() -> None:
    """3 例对照评测：验证隔离命名空间、多专家综合质量显著提升、数字复算率低于单Agent（符合文档观察）、Token 倍数 <= 5x 准入通过。"""
    cases = load_ab_eval_cases(AB_CASES_PATH)
    assert len(cases) == 3

    report = evaluate_ab_comparison(
        cases,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
    )
    assert report.case_count == 3
    assert len(report.runs) == 6
    assert {r.namespace for r in report.runs} == {"single_agent_ns", "multi_expert_ns"}
    assert report.admitted is True
    assert report.quality_delta > 0.0
    assert 1.5 <= report.token_ratio <= 5.0
    # 复现文档第 15 题核心观察：多专家因专家写出派生估算数字，数字可复算率略低于单 Agent，但核验器全拦下且综合质量更高
    assert report.multi_avg_recompute_rate < report.single_avg_recompute_rate


def test_main_graph_deep_direct_and_standard_escalate_once_reusing_evidence() -> None:
    as_of = date(2025, 8, 20)

    # 1. DEEP 档：collect 后直接进入 deep_research，不走单 Agent 的 search/draft/verify/quality_gate
    r_deep = ResearchRunner(as_of=as_of, policy_path=POLICY_PATH, search_sources_path=SEARCH_PATH)
    out_deep = r_deep.start("贵州茅台现在估值合理吗，还要不要继续持有", thread_id="th-direct-deep")
    assert r_deep.runtime.node_calls == ["intake", "authorize", "route", "plan", "collect", "deep_research", "commit"]
    assert out_deep["dossier"].tier == ResearchTier.DEEP
    assert "CHG-01" in out_deep["dossier"].upheld_challenges

    # 2. STANDARD 档在 quality_gate 高分歧升级 → 复用已取证据进 deep_research 且只升级一次
    r_esc = ResearchRunner(
        as_of=as_of,
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        llm_client=MockLLMClient(simulate_high_disagreement=True),
    )
    out_esc = r_esc.start("复核贵州茅台半年报利润质量", thread_id="th-std-esc-once")
    assert r_esc.runtime.node_calls.count("deep_research") == 1
    assert r_esc.runtime.node_calls.count("collect") == 1  # 已取结构化证据直接复用，没有重跑 collect！
    assert out_esc["escalated_to_deep"] is True
    assert out_esc["dossier"].tier == ResearchTier.DEEP
