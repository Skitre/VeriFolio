"""多专家争议深研子图状态定义（利用 operator.add reducer 归约 LangGraph Send 并行分支）。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from zhengyan.core.enums import ExpertRole
from zhengyan.core.models import (
    AdjudicationDecision,
    ChallengeItem,
    ChallengeResponse,
    Claim,
    EvidenceCard,
    ExpertSubmission,
    VerificationResult,
)


class ExpertWorkerInput(TypedDict):
    """LangGraph Send 下发给单个专家的严格隔离输入（严禁包含其他专家结论或私有材料）。"""

    question: str
    symbol: str | None
    as_of: str
    expert_role: ExpertRole
    model_role: str
    # 仅包含共享 A 级结构化快照，绝不传入其他专家产出
    shared_structured_evidence: dict[str, EvidenceCard]
    max_searches: int
    max_fetches: int


class DeepResearchState(TypedDict, total=False):
    # 输入（来自主图 collect 后或 standard 质量门控升级时复用的证据）
    question: str
    symbol: str | None
    as_of: str
    shared_evidence_store: dict[str, EvidenceCard]

    # chair_convene 产出
    selected_experts: list[ExpertRole]
    convene_rationale: str

    # Send 并行 expert_worker 归约结果（Annotated + operator.add）
    expert_submissions: Annotated[list[ExpertSubmission], operator.add]

    # blackboard_verify 产出（汇总黑板 + 首轮隔离检查 + 规则核验）
    blackboard_evidence: dict[str, EvidenceCard]
    blackboard_claims: list[Claim]
    verification_results: list[VerificationResult]
    isolation_violations: list[str]

    # adversary_challenge 与 expert_respond 产出
    challenges: list[ChallengeItem]
    challenge_responses: list[ChallengeResponse]

    # chair_adjudicate 产出
    adjudication: AdjudicationDecision | None
    final_claims: list[Claim]
    excluded_claims: list[Claim]
    aborted_by_human_review: bool
    deep_tokens_used: int
