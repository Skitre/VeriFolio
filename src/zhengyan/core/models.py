"""核心数据契约（Pydantic v2）。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zhengyan.core.enums import (
    AccountingBasis,
    ChallengeResponseAction,
    ChallengeSeverity,
    ClaimType,
    DossierStatus,
    EvidenceGrade,
    EvidenceSourceType,
    ExpertRole,
    MetricUnit,
    PITGuarantee,
    QualityGateAction,
    ResearchTier,
    UserIntent,
    VerificationStatus,
)


class MetricEntry(BaseModel):
    """A 级结构化证据中的单条指标记录。"""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="标准化指标名，如 revenue / net_profit / gross_margin")
    display_name: str = Field(default="", description="中文展示名，如 营业收入")
    value: Decimal = Field(description="精确十进制数值")
    unit: MetricUnit = Field(description="量纲单位")
    period: str | None = Field(default=None, description="所属报告期，如 2025H1 / 2025Q3")
    period_end: date | None = Field(default=None, description="报告期截止日")
    basis: AccountingBasis = Field(default=AccountingBasis.GAAP, description="会计口径")


class FinancialStatementRecord(BaseModel):
    """结构化财报记录（含严格区分的 period_end 与 disclosed_at）。"""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="股票代码，如 600519")
    company_name: str = Field(description="公司简称")
    period: str = Field(description="报告期标识，如 2025H1 / 2025Q3 / 2024FY")
    period_end: date = Field(description="报告期截止日（如 2025-06-30）")
    disclosed_at: date = Field(description="实际公告披露日（时点过滤唯一依据）")
    basis: AccountingBasis = Field(default=AccountingBasis.CUMULATIVE)
    pit_guarantee: PITGuarantee = Field(default=PITGuarantee.STRICT)
    source_name: str = Field(default="cninfo_structured")
    metrics: dict[str, MetricEntry] = Field(default_factory=dict)


class PricePoint(BaseModel):
    """日度行情与股本快照（用于 PE(TTM) 按披露日逐日对齐）。"""

    model_config = ConfigDict(frozen=True)

    symbol: str
    trade_date: date
    close_price: Decimal = Field(description="收盘价（元）")
    total_shares: Decimal = Field(description="当日总股本（股）")
    market_cap_yi: Decimal = Field(description="当日总市值（亿元）")
    source_name: str = Field(default="exchange_quote")


class AnnouncementRecord(BaseModel):
    """巨潮公告元数据记录。"""

    model_config = ConfigDict(frozen=True)

    announcement_id: str
    symbol: str
    title: str
    summary: str
    disclosed_at: date
    url: str = ""
    source_name: str = Field(default="cninfo")


class RawWebResult(BaseModel):
    """搜索接口返回的原始网页条目（尚未经过清洗、去重和分级）。"""

    url: str
    title: str
    snippet: str = ""
    raw_html: str | None = None
    published_at: date | None = None
    fetched: bool = False


class EvidenceCard(BaseModel):
    """系统统一证据卡（A / B / C 级）。"""

    evidence_id: str = Field(description="全局唯一证据 ID，如 EV-FIN-600519-2025H1")
    grade: EvidenceGrade
    source_type: EvidenceSourceType
    source_name: str = Field(description="来源名称或站点域名")
    title: str
    symbol: str | None = None
    published_at: date | None = Field(
        default=None,
        description="披露日（财报/公告/计算卡取依赖最晚披露日）或网页发布日",
    )
    period: str | None = Field(default=None, description="所属报告期，便于后续版本依赖追踪")
    pit_guarantee: PITGuarantee = Field(default=PITGuarantee.STRICT)

    metrics: list[MetricEntry] = Field(default_factory=list)
    derived_from: list[str] = Field(default_factory=list)

    url: str | None = None
    authority: int = Field(default=0, ge=0, le=5)
    has_full_snapshot: bool = False
    excerpt: str = ""
    simhash: int | None = None
    event_cluster_id: str | None = None
    original_source: str | None = Field(default=None, description="正文抽取的“来源：XX”")
    suspected_injection: bool = False
    stripped_injection_lines: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    """研究草稿中的单条主张（对应文档第 3 题与第 7 题）。"""

    claim_id: str
    text: str = Field(description="主张陈述文本")
    claim_type: ClaimType = Field(default=ClaimType.FACT)
    evidence_ids: list[str] = Field(default_factory=list)
    author: str = Field(default="standard_researcher")


class NumberCheckDetail(BaseModel):
    """核验器对主张中单个数字的逐项核验结果。"""

    raw_text: str
    parsed_value: Decimal
    unit: MetricUnit
    tolerance: Decimal
    matched: bool
    matched_grade: EvidenceGrade | None = None
    matched_evidence_id: str | None = None
    matched_detail: str = ""


class VerificationResult(BaseModel):
    """规则核验器对单条主张的确定性判定结果。"""

    claim_id: str
    status: VerificationStatus
    is_fact: bool = True
    valid_evidence_ids: list[str] = Field(default_factory=list)
    ignored_evidence_ids: list[str] = Field(default_factory=list)
    number_checks: list[NumberCheckDetail] = Field(default_factory=list)
    single_source_merged: bool = False
    reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def can_enter_conclusion(self) -> bool:
        """只有已核验（非存疑、非驳回）的主张才能进入最终结论。"""
        return self.status == VerificationStatus.VERIFIED


class GoldenComputeCase(BaseModel):
    """计算金标单条测试用例格式（对应 evals/golden/compute_cases.json）。"""

    case_id: str
    category: Literal["yoy_growth", "single_quarter", "ratio", "pe_ttm", "basis_mismatch"]
    description: str
    params: dict[str, str]
    expected_value: str | None = Field(
        default=None,
        description="期望字符串（用于同时确定精确值和书写精度半单位），口径报错用例为 None",
    )
    expected_unit: MetricUnit | None = None
    expect_error: bool = False


# ==================== Stage 2 新增编排与治理数据契约 ====================


class ResearchPlan(BaseModel):
    """plan 节点的强类型输出契约（强制含风险/反证方向查询）。"""

    queries: list[str] = Field(default_factory=list, description="常规信息检索词")
    risk_queries: list[str] = Field(
        min_length=1,
        description="强制至少包含一条风险或反证方向的查询词（对应文档第 3 题）",
    )
    target_metrics: list[str] = Field(default_factory=list, description="计划计算的同口径派生指标")
    rationale: str = Field(default="", description="规划说明")


class DraftReport(BaseModel):
    """draft 节点的强类型输出契约。"""

    summary: str = Field(description="研究摘要")
    claims: list[Claim] = Field(default_factory=list, description="结构化主张列表")
    evidence_gaps: list[str] = Field(default_factory=list, description="当前尚未补齐的证据缺口")
    counterpoints: list[str] = Field(default_factory=list, description="识别到的反方观点/风险点")


class QualityGateReport(BaseModel):
    """quality_gate 节点的判定报告。"""

    action: QualityGateAction
    verified_fact_count: int
    doubtful_count: int
    rejected_count: int
    has_derived_metric: bool
    counterpoint_count: int
    reasons: list[str] = Field(default_factory=list)


class BudgetSnapshot(BaseModel):
    """单次研究任务的实时预算消耗快照。"""

    tier: ResearchTier
    tokens_used: int = 0
    max_tokens: int
    searches_used: int = 0
    max_searches: int
    fetches_used: int = 0
    max_fetches: int
    elapsed_seconds: float = 0.0
    max_duration_seconds: int


class AuditSpan(BaseModel):
    """OpenTelemetry 风格的链路审计 Span（节点 / Skill / LLM 调用各占一条）。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    node_name: str
    caller: str
    action_type: Literal["node", "skill", "llm", "decision", "approval"]
    target: str
    status: Literal["ok", "denied", "error", "cached", "interrupted"]
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    tokens_used: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class DossierRecord(BaseModel):
    """只追加提交的研究决策档案（Stage 5 将进一步扩展版本依赖图）。"""

    dossier_id: str
    symbol: str | None
    version: int = 1
    as_of: date
    question: str
    intent: UserIntent
    tier: ResearchTier
    status: DossierStatus
    boundary_notice: str | None = None
    summary: str = ""
    verified_claims: list[Claim] = Field(default_factory=list)
    excluded_claims: list[Claim] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    dissenting_opinions: list[str] = Field(default_factory=list)
    upheld_challenges: list[str] = Field(default_factory=list)
    method_asset_versions: dict[str, str] = Field(default_factory=dict)
    needs_deep_escalation: bool = False
    budget_snapshot: BudgetSnapshot | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==================== Stage 4 新增多专家深研与对照评测契约 ====================


class ExpertSubBudget(BaseModel):
    """深研专家独立子预算（对应文档第 13 题：各 2 次搜索、2 次抓取）。"""

    max_searches: int = 2
    max_fetches: int = 2
    searches_used: int = 0
    fetches_used: int = 0


class ExpertSubmission(BaseModel):
    """单名专家在首轮隔离取证后的独立提交。"""

    expert_role: ExpertRole
    model_role: str = "draft"
    visible_evidence_ids: list[str] = Field(
        default_factory=list,
        description="该专家首轮合法可见的证据 ID（共享 A 级快照 + 自己搜到的私有材料）",
    )
    private_evidence: list[EvidenceCard] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    key_dimensions_covered: list[str] = Field(default_factory=list)
    sub_budget: ExpertSubBudget = Field(default_factory=ExpertSubBudget)
    tokens_used: int = 0


class ChallengeItem(BaseModel):
    """反证官对黑板主张发起的单条质询（最多 8 条）。"""

    challenge_id: str
    target_claim_id: str
    target_expert: ExpertRole
    severity: ChallengeSeverity
    critique: str = Field(description="质询理由与反例说明")
    counter_evidence_ids: list[str] = Field(default_factory=list)
    is_causal_fallacy: bool = False


class ChallengeResponse(BaseModel):
    """被质询专家的 1 轮回应。"""

    challenge_id: str
    expert_role: ExpertRole
    action: ChallengeResponseAction
    response_text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class DissentingOpinion(BaseModel):
    """有独立证据支撑但存在多空分歧的保留意见（不通过投票强行抹平）。"""

    topic: str
    bull_or_main_view: str
    bear_or_counter_view: str
    independent_evidence_ids: list[str] = Field(default_factory=list)


class AdjudicationDecision(BaseModel):
    """主席基于独立证据与质询结果的最终裁决（严禁多数投票制）。"""

    summary: str
    adopted_claim_ids: list[str] = Field(default_factory=list)
    blocked_claim_ids: list[str] = Field(default_factory=list)
    upheld_challenge_ids: list[str] = Field(default_factory=list)
    unresolved_high_severity_ids: list[str] = Field(default_factory=list)
    dissenting_opinions: list[DissentingOpinion] = Field(default_factory=list)
    independent_source_counts: dict[str, int] = Field(
        default_factory=dict,
        description="每条主张去重转载事件簇后的独立来源计数（反多数投票刷票）",
    )
    human_review_required: bool = False
    human_review_approved: bool | None = None


class CompositeQualityBreakdown(BaseModel):
    """代码计算的综合质量分明细（严格对应文档第 16 题权重）。"""

    claim_verify_rate: float        # 权重 30%
    key_item_coverage: float        # 权重 30%
    number_recompute_rate: float    # 权重 25%
    preserved_dissents_count: int   # 最多计 3 条，权重 5%
    quality_gate_passed: bool       # 权重 10%
    composite_score: float


class ABRunMetrics(BaseModel):
    """单次对照运行（单 Agent 或 多专家）的隔离评测指标。"""

    case_id: str
    mode: Literal["single_agent", "multi_expert"]
    namespace: str
    tokens_used: int
    quality: CompositeQualityBreakdown
    dossier_status: DossierStatus


class ABComparisonReport(BaseModel):
    """单 Agent vs 多专家对照评测汇总与准入判据报告（文档第 15 题）。"""

    case_count: int
    single_avg_quality: float
    multi_avg_quality: float
    quality_delta: float            # multi - single（判据 >= -0.02）
    single_avg_tokens: float
    multi_avg_tokens: float
    token_ratio: float              # multi / single（判据 <= 5.0）
    single_avg_recompute_rate: float
    multi_avg_recompute_rate: float
    admitted: bool                  # 两条准入判据同时满足才准入
    rejection_reasons: list[str] = Field(default_factory=list)
    runs: list[ABRunMetrics] = Field(default_factory=list)

