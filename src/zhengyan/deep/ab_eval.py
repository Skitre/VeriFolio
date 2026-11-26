"""单 Agent vs 多专家对照评测与代码综合质量分计算（对应文档第 15、16、28 题）。

严格按文档第 16 题精确实现综合质量分（不用 LLM-as-a-Judge）：
  CompositeScore =
      0.30 * 主张核验通过率
    + 0.30 * 关键项覆盖率
    + 0.25 * 数字可复算率
    + 0.05 * min(保留分歧数, 3) / 3
    + 0.10 * (1.0 if 质量门控通过 else 0.0)

对照评测准入判据（文档第 15 题，跑前定好）：
  1. 质量差 delta = multi_avg_quality - single_avg_quality >= -0.02
  2. Token 倍数 ratio = multi_avg_tokens / single_avg_tokens <= 5.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from zhengyan.core.enums import DossierStatus, ResearchTier, VerificationStatus
from zhengyan.core.models import (
    ABComparisonReport,
    ABRunMetrics,
    Claim,
    CompositeQualityBreakdown,
    VerificationResult,
)

# 维度关键词映射，用于纯代码评估关键项覆盖率（不依赖 LLM 打分）
DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "profit_quality": ("营业收入", "归母净利润", "毛利率", "净利率", "利润"),
    "cashflow": ("现金流", "净现比", "回款"),
    "valuation": ("市盈率", "PE", "估值", "安全边际", "倍数"),
    "channel_industry": ("渠道", "行业", "库存", "批价", "打款", "产能", "竞争"),
    "risk_counterpoint": ("风险", "分歧", "质疑", "下行", "波动", "承压", "反证"),
}


@dataclass(frozen=True)
class ABEvalCase:
    case_id: str
    symbol: str
    as_of: date
    question: str
    required_dimensions: tuple[str, ...]


def load_ab_eval_cases(path: Path) -> list[ABEvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        ABEvalCase(
            case_id=item["case_id"],
            symbol=item["symbol"],
            as_of=date.fromisoformat(item["as_of"]),
            question=item["question"],
            required_dimensions=tuple(item["required_dimensions"]),
        )
        for item in raw
    ]


def compute_composite_quality_score(
    *,
    claims: Sequence[Claim],
    verification_results: Sequence[VerificationResult],
    required_dimensions: Sequence[str],
    preserved_dissents_count: int,
    quality_gate_passed: bool,
    extra_texts: Sequence[str] = (),
) -> CompositeQualityBreakdown:
    """严格按文档第 16 题权重公式计算综合质量分（纯确定性代码）。"""
    # 1. 主张核验通过率（30%）
    total_claims = len(verification_results)
    verified_cnt = sum(1 for r in verification_results if r.status == VerificationStatus.VERIFIED)
    claim_verify_rate = verified_cnt / total_claims if total_claims > 0 else 0.0

    # 2. 关键项覆盖率（30%）：仅统计已核验通过的主张与保留意见覆盖到的维度
    verified_ids = {r.claim_id for r in verification_results if r.status == VerificationStatus.VERIFIED}
    corpus = "\n".join([c.text for c in claims if c.claim_id in verified_ids] + list(extra_texts))
    covered = 0
    for dim in required_dimensions:
        kws = DIMENSION_KEYWORDS.get(dim, (dim,))
        if any(kw in corpus for kw in kws):
            covered += 1
    key_item_coverage = covered / len(required_dimensions) if required_dimensions else 1.0

    # 3. 数字可复算率（25%）：所有被抽取金融数字中，在 A 级证据复算通过的比例
    total_nums = sum(len(r.number_checks) for r in verification_results)
    matched_nums = sum(
        sum(1 for nc in r.number_checks if nc.matched)
        for r in verification_results
    )
    number_recompute_rate = matched_nums / total_nums if total_nums > 0 else 1.0

    # 4. 保留分歧数（最多计 3 条，5%）
    dissent_ratio = min(max(preserved_dissents_count, 0), 3) / 3.0

    # 5. 质量门控是否通过（10%）
    gate_score = 1.0 if quality_gate_passed else 0.0

    score = (
        0.30 * claim_verify_rate
        + 0.30 * key_item_coverage
        + 0.25 * number_recompute_rate
        + 0.05 * dissent_ratio
        + 0.10 * gate_score
    )

    return CompositeQualityBreakdown(
        claim_verify_rate=round(claim_verify_rate, 6),
        key_item_coverage=round(key_item_coverage, 6),
        number_recompute_rate=round(number_recompute_rate, 6),
        preserved_dissents_count=preserved_dissents_count,
        quality_gate_passed=quality_gate_passed,
        composite_score=round(score, 6),
    )


def evaluate_ab_comparison(
    cases: Sequence[ABEvalCase],
    *,
    policy_path: Path,
    search_sources_path: Path,
    min_quality_delta: float = -0.02,
    max_token_ratio: float = 5.0,
) -> ABComparisonReport:
    """在相互隔离的档案命名空间下，对同一问题同一时点分别跑单 Agent 和多专家并判定准入。"""
    from zhengyan.graph import InMemoryDossierStore, ResearchRunner

    single_ns_store = InMemoryDossierStore()
    multi_ns_store = InMemoryDossierStore()
    runs: list[ABRunMetrics] = []

    for case in cases:
        # 1. 单 Agent 基线（强制走 STANDARD，写入 single_agent 隔离命名空间）
        single_runner = ResearchRunner(
            as_of=case.as_of,
            policy_path=policy_path,
            search_sources_path=search_sources_path,
            dossier_store=single_ns_store,
        )
        s_out = single_runner.start(
            case.question,
            thread_id=f"ab-single-{case.case_id}",
            user_selected_tier=ResearchTier.STANDARD,
        )
        s_dossier = s_out["dossier"]
        s_quality = compute_composite_quality_score(
            claims=s_out.get("claims") or [],
            verification_results=s_dossier.verification_results,
            required_dimensions=case.required_dimensions,
            preserved_dissents_count=len(s_dossier.dissenting_opinions),
            quality_gate_passed=(s_dossier.status == DossierStatus.COMPLETE),
            extra_texts=s_dossier.dissenting_opinions,
        )
        runs.append(
            ABRunMetrics(
                case_id=case.case_id,
                mode="single_agent",
                namespace="single_agent_ns",
                tokens_used=s_dossier.budget_snapshot.tokens_used if s_dossier.budget_snapshot else 0,
                quality=s_quality,
                dossier_status=s_dossier.status,
            )
        )

        # 2. 多专家争议深研（强制走 DEEP，写入 multi_expert 隔离命名空间）
        multi_runner = ResearchRunner(
            as_of=case.as_of,
            policy_path=policy_path,
            search_sources_path=search_sources_path,
            dossier_store=multi_ns_store,
        )
        m_out = multi_runner.start(
            case.question,
            thread_id=f"ab-multi-{case.case_id}",
            user_selected_tier=ResearchTier.DEEP,
        )
        m_dossier = m_out["dossier"]
        m_quality = compute_composite_quality_score(
            claims=m_out.get("claims") or [],
            verification_results=m_dossier.verification_results,
            required_dimensions=case.required_dimensions,
            preserved_dissents_count=len(m_dossier.dissenting_opinions),
            quality_gate_passed=(m_dossier.status == DossierStatus.COMPLETE),
            extra_texts=m_dossier.dissenting_opinions,
        )
        runs.append(
            ABRunMetrics(
                case_id=case.case_id,
                mode="multi_expert",
                namespace="multi_expert_ns",
                tokens_used=m_dossier.budget_snapshot.tokens_used if m_dossier.budget_snapshot else 0,
                quality=m_quality,
                dossier_status=m_dossier.status,
            )
        )

    s_runs = [r for r in runs if r.mode == "single_agent"]
    m_runs = [r for r in runs if r.mode == "multi_expert"]
    n = len(cases)

    s_avg_q = sum(r.quality.composite_score for r in s_runs) / n
    m_avg_q = sum(r.quality.composite_score for r in m_runs) / n
    q_delta = m_avg_q - s_avg_q

    s_avg_tok = sum(r.tokens_used for r in s_runs) / n
    m_avg_tok = sum(r.tokens_used for r in m_runs) / n
    tok_ratio = m_avg_tok / s_avg_tok if s_avg_tok > 0 else 0.0

    s_avg_recomp = sum(r.quality.number_recompute_rate for r in s_runs) / n
    m_avg_recomp = sum(r.quality.number_recompute_rate for r in m_runs) / n

    reasons: list[str] = []
    if q_delta < min_quality_delta:
        reasons.append(f"质量差 {q_delta:.4f} 低于准入下限 {min_quality_delta}")
    if tok_ratio > max_token_ratio:
        reasons.append(f"Token 倍数 {tok_ratio:.2f}x 超过准入上限 {max_token_ratio}x")

    return ABComparisonReport(
        case_count=n,
        single_avg_quality=round(s_avg_q, 6),
        multi_avg_quality=round(m_avg_q, 6),
        quality_delta=round(q_delta, 6),
        single_avg_tokens=round(s_avg_tok, 2),
        multi_avg_tokens=round(m_avg_tok, 2),
        token_ratio=round(tok_ratio, 4),
        single_avg_recompute_rate=round(s_avg_recomp, 6),
        multi_avg_recompute_rate=round(m_avg_recomp, 6),
        admitted=(len(reasons) == 0),
        rejection_reasons=reasons,
        runs=runs,
    )
