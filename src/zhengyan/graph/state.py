"""LangGraph 显式状态定义（仅包含可检查点序列化的纯状态字段）。"""

from __future__ import annotations

from typing import Any, TypedDict

from zhengyan.core.enums import DossierStatus, QualityGateAction, ResearchTier, UserIntent
from zhengyan.core.models import (
    AdjudicationDecision,
    ChallengeItem,
    ChallengeResponse,
    Claim,
    DossierRecord,
    EvidenceCard,
    QualityGateReport,
    ResearchPlan,
    VerificationResult,
)


class ResearchState(TypedDict, total=False):
    # 输入参数
    question: str
    as_of: str  # ISO 日期字符串（YYYY-MM-DD），便于检查点跨进程持久化
    user_selected_tier: ResearchTier | None

    # intake 节点产出
    symbol: str | None
    company_name: str | None
    intent: UserIntent
    focus_points: list[str]
    pii_hits: list[dict[str, str]]

    # authorize 节点产出
    pii_approved: bool | None
    boundary_notice: str | None
    aborted: bool
    abort_reason: str | None

    # route 节点产出（Stage 3 扩展双轴路由、概率校准与拒判审计字段）
    tier: ResearchTier
    route_reason: str
    route_raw_probs: dict[str, float] | None
    route_calibrated_probs: dict[str, float] | None
    route_abstained: bool
    model_bindings: dict[str, str]

    # plan / collect / search 产出
    supplement_rounds: int
    plan: ResearchPlan | None
    evidence_store: dict[str, EvidenceCard]
    budget_exhausted: bool

    # draft / verify / quality_gate 产出
    summary: str
    claims: list[Claim]
    evidence_gaps: list[str]
    counterpoints: list[str]
    verification_results: list[VerificationResult]
    quality_gate: QualityGateReport | None
    needs_deep_escalation: bool
    escalated_to_deep: bool

    # Stage 4 深研子图产出
    deep_challenges: list[ChallengeItem]
    deep_challenge_responses: list[ChallengeResponse]
    deep_adjudication: AdjudicationDecision | None
    deep_isolation_violations: list[str]
    deep_final_claims: list[Claim]
    deep_excluded_claims: list[Claim]

    # commit 节点产出
    dossier: DossierRecord | None
    final_status: DossierStatus
    trace_summary: dict[str, Any]
