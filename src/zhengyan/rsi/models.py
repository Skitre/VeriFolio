"""RSI 受控自进化数据契约（对应文档第 19–25 题）。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RSIComponent(StrEnum):
    """允许受控自进化的三类模型外围资产。"""

    S1_EXPERIENCE_CARDS = "experience_cards"
    S2_ROUTING_PARAMS = "routing_params"
    S3_SEARCH_TEMPLATES = "search_templates"


# 永远不碰的治理红线（文档第 25 题：权限、隐私、核验规则、交易边界、用户约束、买卖推荐）
FORBIDDEN_RSI_TARGETS: frozenset[str] = frozenset({
    "permissions",
    "privacy_rules",
    "verifier_rules",
    "trading_boundary",
    "user_constraints",
    "buy_sell_recommendations",
    "model_weights",
})


class BadCaseSource(StrEnum):
    UPHELD_CHALLENGE = "upheld_challenge"      # 深研中被裁定成立的质询
    DOUBTFUL_CLAIM = "doubtful_claim"          # 核验器判为存疑的主张
    ROUTING_ERROR = "routing_error"            # 路由评测里的错判
    LOW_QUALITY_SEARCH = "low_quality_search"  # 搜索 C 级占比高 / 转载多 / 抓取未引用


class BadCaseRecord(BaseModel):
    """坏案例强制四要素（文档第 20 题：案例、类型、证据、假设，缺假设禁止登记）。"""

    model_config = ConfigDict(frozen=True)

    case_id: str
    source_case_ref: str = Field(min_length=1, description="案例来源（如档案 600519 v5 / 路由题 RT-15）")
    failure_type: BadCaseSource = Field(description="坏案例类型")
    evidence_summary: str = Field(min_length=5, description="支撑该坏案例的客观证据摘要")
    hypothesis: str = Field(description="根因假设（必须明确写清“我猜是因为……”，无假设不允许登记）")
    symbol: str | None = None
    period: str | None = None
    target_component: RSIComponent = RSIComponent.S1_EXPERIENCE_CARDS
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("hypothesis")
    @classmethod
    def _hypothesis_must_be_meaningful(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 6:
            raise ValueError("坏案例登记必须提供明确的根因假设（至少 6 个字），禁止为了改而改")
        return stripped


class ExperienceCard(BaseModel):
    """S1 经验卡结构（对应文档第 22 题 L-007 真实格式）。"""

    model_config = ConfigDict(frozen=True)

    card_id: str
    version: str = "v1"
    tags: list[str] = Field(min_length=1, description="召回匹配标签")
    applicable_scope: str = Field(description="适用范围（若含行业专用词须明确标出行业）")
    lesson: str = Field(description="方法论教训（禁止出现具体股票名、数字与买卖结论）")
    must_check: str = Field(description="研究时必查项")
    invalidation_condition: str = Field(description="该经验卡的失效条件")
    source_ref: str = Field(description="沉淀来源（如深研中被裁定成立的质询：档案 600519 v5）")


class RoutingParamsAsset(BaseModel):
    """S2 路由可进化参数资产。"""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(gt=0.0, le=3.0)
    abstain_threshold: float = Field(ge=0.30, le=0.90)
    escalation_doubtful_threshold: int = Field(default=2, ge=1, le=5)


class SearchTemplateAsset(BaseModel):
    """S3 检索策略可进化资产。"""

    model_config = ConfigDict(frozen=True)

    expert_focus_prompts: dict[str, str] = Field(default_factory=dict)
    relevance_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    whitelist_domains: list[str] = Field(default_factory=list)


class EvalProtocolFingerprint(BaseModel):
    """评测协议指纹：确保候选与基线使用完全同一套评测集、评分器和审稿提示词。"""

    model_config = ConfigDict(frozen=True)

    eval_set_hash: str
    scorer_version_hash: str
    reviewer_prompt_hash: str


class SliceEvalStats(BaseModel):
    """单个数据切片（新失败集 / 固定回放集 / 时间外测试集）的多次重复评测统计。"""

    slice_name: Literal["new_failures", "frozen_replay", "out_of_time"]
    run_scores: list[float] = Field(min_length=1)
    mean_score: float
    spread: float  # max - min 波动范围
    mean_tokens: float = 0.0
    off_mean_score: float | None = Field(
        default=None,
        description="开/关对照：关掉该资产时的基线均分，用于证明增益来自被继承的改进",
    )


class CandidateProposal(BaseModel):
    """优化器提交的单点改动候选（一次只针对一个问题、只改一个组件）。"""

    proposal_id: str
    component: RSIComponent
    round_index: int = Field(ge=0)
    parent_version: str
    candidate_version: str
    direction_key: str = Field(description="改动方向指纹，用于账本记录失败方向避免重复试错")
    change_reason: str = Field(description="改动理由（账本硬要求不少于 10 个字）")
    source_bad_cases: list[BadCaseRecord] = Field(min_length=1)
    baseline_payload: dict[str, Any]
    candidate_payload: dict[str, Any]


class GateCheckResult(BaseModel):
    gate_name: str
    passed: bool
    detail: str


class SixGatesReport(BaseModel):
    """六道门控独立代码裁决报告。"""

    proposal_id: str
    component: RSIComponent
    passed: bool
    checks: list[GateCheckResult] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)


class LedgerAssetRecord(BaseModel):
    """账本表 1：只追加不可变资产版本表。"""

    model_config = ConfigDict(frozen=True)

    component: RSIComponent
    version: str
    parent_version: str | None
    change_reason: str
    content_hash: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LedgerRunRecord(BaseModel):
    """账本表 2：只追加评测与门控运行表（含被拒绝的候选及原因，防止重复试错）。"""

    model_config = ConfigDict(frozen=True)

    run_id: str
    proposal_id: str
    component: RSIComponent
    candidate_version: str
    direction_key: str
    protocol: EvalProtocolFingerprint
    baseline_slices: dict[str, SliceEvalStats]
    candidate_slices: dict[str, SliceEvalStats]
    gate_report: SixGatesReport
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LedgerEventRecord(BaseModel):
    """账本表 3：只追加发布/回滚/拒绝事件表（当前在用版本由事件流推导，非可变字段）。"""

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_type: Literal["init", "publish", "rollback", "reject"]
    component: RSIComponent
    version: str
    actor: str
    approval_mode: Literal["independent_approved", "self_approved", "system"] = "system"
    reason: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
