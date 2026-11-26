"""五类版本化记忆治理仓储与精确依赖链（对应文档第 17 & 17.1 题）。

为什么不用“聊天历史 + 向量数据库”？
投研的核心查询是精确依赖定位：“哪些历史结论依赖了 2024FY 财报或 2025-04-01 行情？”
字段关联一查即准，向量相似度反而会召回语义相似但无关的噪声。

五类记忆与硬写入权限隔离：
1. 已核验事实（VerifiedFact）      -> 仅允许系统在核验状态 == VERIFIED 后写入
2. 用户确认约束（UserConstraints） -> 仅允许用户本人显式勾选确认写入（研究流程无权写入！）
3. 投资假设（InvestmentHypothesis）-> 研究产生，带显式失效条件，条件命中时自动标失效
4. 决策历程（VersionedDossier）    -> 只追加版本链（parent_dossier_id），永不覆盖旧版，记录 Claim→Evidence→Period 依赖
5. 方法经验（MethodAssetRef）      -> 仅允许通过 RSI 门控发布（Stage 6 接入），每份档案记录所用资产版本号
+ 工作记忆（WorkingMemoryEntry）   -> 存研究中间结果，30 天后自动清理
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from zhengyan.core.enums import EvidenceSourceType, VerificationStatus
from zhengyan.core.models import Claim, DossierRecord, EvidenceCard, VerificationResult


class MemoryPermissionError(PermissionError):
    """违反五类记忆写入权限隔离时抛出。"""


class ClaimLifecycleStatus(StrEnum):
    """主张在版本演进与增量复核中的状态。"""

    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    RETAINED = "retained"
    WEAKENED = "weakened"
    INVALIDATED = "invalidated"
    UNDECIDABLE = "undecidable"


class VerifiedFact(BaseModel):
    """记忆类 1：已核验事实（带来源与时点，仅核验通过后写入）。"""

    model_config = ConfigDict(frozen=True)

    fact_id: str
    symbol: str
    statement: str
    evidence_ids: list[str]
    as_of: date
    period: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserConfirmedConstraints(BaseModel):
    """记忆类 2：用户本人确认的个人投资约束（研究流程绝对无权写入）。"""

    user_id: str
    buy_thesis: str = Field(description="用户买入/关注的核心理由")
    horizon_months: int = Field(gt=0, description="计划持有资金期限（月）")
    max_drawdown_pct: Decimal = Field(gt=0, le=100, description="可承受最大回撤（%）")
    max_pe_tolerance: Decimal | None = Field(default=None, description="可接受估值 PE 上限")
    confirmed_by_user: bool = Field(default=True)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1


class InvestmentHypothesis(BaseModel):
    """记忆类 3：结论所依赖的投资假设及显式失效条件。"""

    hypothesis_id: str
    symbol: str
    dossier_id: str
    statement: str
    invalidation_metric: str = Field(description="触发复核/失效的监控指标，如 net_profit_yoy / pe_ttm / price_change_pct")
    invalidation_op: Literal["<", "<=", ">", ">="]
    invalidation_threshold: Decimal
    active: bool = True
    invalidated_reason: str | None = None


class ClaimDependency(BaseModel):
    """主张 → 证据 → 报告期/时点 精确依赖记录。"""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    evidence_id: str
    source_type: EvidenceSourceType
    period: str | None = None
    published_at: date | None = None


class ClaimUpdateVerdict(BaseModel):
    """增量复核时，新版本对上一版旧结论的逐条对照裁定。"""

    old_claim_id: str
    old_text: str
    old_as_of: date
    verdict: ClaimLifecycleStatus
    reason: str
    new_evidence_ids: list[str] = Field(default_factory=list)


class VersionedDossier(BaseModel):
    """记忆类 4：只追加决策档案版本链节点（永不改写历史版本）。"""

    dossier: DossierRecord
    parent_dossier_id: str | None = None
    evidence_cards: dict[str, EvidenceCard] = Field(
        default_factory=dict,
        description="当时所用证据卡不可变快照，供后续增量复核追溯与保留旧主张校验",
    )
    dependencies: list[ClaimDependency] = Field(default_factory=list)
    claim_statuses: dict[str, ClaimLifecycleStatus] = Field(default_factory=dict)
    review_verdicts: list[ClaimUpdateVerdict] = Field(default_factory=list)
    abandoned_alternatives: list[str] = Field(default_factory=list)
    # 档案级客观指标快照（供个人情境官与策略卡读取）
    metrics_snapshot: dict[str, Decimal] = Field(default_factory=dict)
    # 记忆类 5 关联：记录本研究用了哪版经验卡/路由参数/检索模板（Stage 6 证明继承）
    method_asset_versions: dict[str, str] = Field(
        default_factory=lambda: {
            "experience_cards": "v1",
            "routing_params": "v1",
            "search_templates": "v1",
        }
    )


class WorkingMemoryEntry(BaseModel):
    """工作记忆：研究过程临时产物，30 天后自动清理。"""

    entry_id: str
    task_id: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_days: int = 30


class FiveLayerMemoryStore:
    """五类记忆统一治理仓储（内存实现，接口与 MongoDB 集合设计一一对应）。"""

    def __init__(self) -> None:
        self._verified_facts: list[VerifiedFact] = []
        self._user_constraints: dict[str, list[UserConfirmedConstraints]] = {}
        self._hypotheses: dict[str, InvestmentHypothesis] = {}
        self._dossiers: dict[str, VersionedDossier] = {}
        self._symbol_chains: dict[str, list[str]] = {}
        self._method_assets: dict[str, list[dict[str, Any]]] = {}
        self._working_memory: dict[str, WorkingMemoryEntry] = {}

    # ---------- 记忆类 1：已核验事实（仅允许核验通过后写入） ----------
    def record_verified_fact(
        self,
        *,
        symbol: str,
        claim: Claim,
        verification: VerificationResult,
        as_of: date,
        period: str | None = None,
    ) -> VerifiedFact:
        if verification.status != VerificationStatus.VERIFIED or not verification.is_fact:
            raise MemoryPermissionError(
                f"拒绝写入已核验事实库：主张 {claim.claim_id} 状态为 {verification.status.value}（is_fact={verification.is_fact}）"
            )
        fact = VerifiedFact(
            fact_id=f"FACT-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            statement=claim.text,
            evidence_ids=list(verification.valid_evidence_ids),
            as_of=as_of,
            period=period,
        )
        self._verified_facts.append(fact)
        return fact

    def list_verified_facts(self, symbol: str | None = None) -> list[VerifiedFact]:
        if symbol is None:
            return list(self._verified_facts)
        return [f for f in self._verified_facts if f.symbol == symbol]

    # ---------- 记忆类 2：用户确认约束（仅允许用户本人勾选确认写入） ----------
    def save_user_constraints(
        self,
        constraints: UserConfirmedConstraints,
        *,
        caller: str,
        explicit_user_confirmation: bool,
    ) -> UserConfirmedConstraints:
        if caller != "user" or not explicit_user_confirmation or not constraints.confirmed_by_user:
            raise MemoryPermissionError(
                f"越权写入用户约束：仅允许用户本人显式勾选确认写入（拒绝 caller={caller!r}）"
            )
        history = self._user_constraints.setdefault(constraints.user_id, [])
        versioned = constraints.model_copy(update={"version": len(history) + 1})
        history.append(versioned)
        return versioned

    def get_latest_user_constraints(self, user_id: str) -> UserConfirmedConstraints | None:
        history = self._user_constraints.get(user_id)
        return history[-1] if history else None

    # ---------- 记忆类 3：投资假设 ----------
    def add_hypothesis(self, hyp: InvestmentHypothesis) -> InvestmentHypothesis:
        self._hypotheses[hyp.hypothesis_id] = hyp
        return hyp

    def list_hypotheses(self, symbol: str, *, active_only: bool = False) -> list[InvestmentHypothesis]:
        return [
            h for h in self._hypotheses.values()
            if h.symbol == symbol and (not active_only or h.active)
        ]

    def mark_hypothesis_invalidated(self, hypothesis_id: str, reason: str) -> None:
        hyp = self._hypotheses[hypothesis_id]
        self._hypotheses[hypothesis_id] = hyp.model_copy(
            update={"active": False, "invalidated_reason": reason}
        )

    # ---------- 记忆类 4：只追加决策档案版本链 + 精确依赖图 ----------
    def append_dossier(
        self,
        dossier: DossierRecord,
        *,
        evidence_store: dict[str, EvidenceCard],
        parent_dossier_id: str | None = None,
        review_verdicts: list[ClaimUpdateVerdict] | None = None,
        abandoned_alternatives: list[str] | None = None,
        metrics_snapshot: dict[str, Decimal] | None = None,
        method_asset_versions: dict[str, str] | None = None,
    ) -> VersionedDossier:
        """只追加写入新版本档案，自动展开 Claim → Evidence → ReportPeriod 依赖链，永不覆盖旧版。"""
        if dossier.dossier_id in self._dossiers:
            raise MemoryPermissionError(f"决策档案 {dossier.dossier_id} 已存在，版本链只允许追加，禁止覆盖！")

        symbol = dossier.symbol or "UNKNOWN"
        chain = self._symbol_chains.setdefault(symbol, [])
        next_ver = len(chain) + 1
        actual_parent = parent_dossier_id if parent_dossier_id is not None else (chain[-1] if chain else None)

        versioned_record = dossier.model_copy(update={"version": next_ver})
        deps = self._extract_dependencies(versioned_record.verified_claims, evidence_store)
        statuses = {c.claim_id: ClaimLifecycleStatus.ACTIVE for c in versioned_record.verified_claims}

        node = VersionedDossier(
            dossier=versioned_record,
            parent_dossier_id=actual_parent,
            evidence_cards=dict(evidence_store),
            dependencies=deps,
            claim_statuses=statuses,
            review_verdicts=list(review_verdicts or []),
            abandoned_alternatives=list(abandoned_alternatives or []),
            metrics_snapshot=dict(metrics_snapshot or {}),
            method_asset_versions=dict(
                method_asset_versions
                or {"experience_cards": "v1", "routing_params": "v1", "search_templates": "v1"}
            ),
        )
        self._dossiers[versioned_record.dossier_id] = node
        chain.append(versioned_record.dossier_id)
        return node

    def get_dossier(self, dossier_id: str) -> VersionedDossier | None:
        return self._dossiers.get(dossier_id)

    def get_latest_dossier(self, symbol: str) -> VersionedDossier | None:
        chain = self._symbol_chains.get(symbol)
        if not chain:
            return None
        return self._dossiers[chain[-1]]

    def get_version_chain(self, symbol: str) -> list[VersionedDossier]:
        return [self._dossiers[did] for did in self._symbol_chains.get(symbol, [])]

    def list_latest_dossiers_all_symbols(self) -> list[VersionedDossier]:
        out: list[VersionedDossier] = []
        for chain in self._symbol_chains.values():
            if chain:
                out.append(self._dossiers[chain[-1]])
        return out

    def mark_claims_pending_review(self, dossier_id: str, claim_ids: list[str]) -> None:
        """将受新事件影响的旧结论标记为 PENDING_REVIEW（不篡改原结论内容）。"""
        node = self._dossiers[dossier_id]
        updated = dict(node.claim_statuses)
        for cid in claim_ids:
            if cid in updated:
                updated[cid] = ClaimLifecycleStatus.PENDING_REVIEW
        self._dossiers[dossier_id] = node.model_copy(update={"claim_statuses": updated})

    def find_claims_depending_on_period(self, dossier_id: str, period: str) -> list[str]:
        """精确依赖查询：返回该档案中依赖指定报告期（如 2024FY）的全部主张 ID。"""
        node = self._dossiers[dossier_id]
        return sorted({d.claim_id for d in node.dependencies if d.period == period})

    def find_claims_depending_on_source_types(
        self,
        dossier_id: str,
        source_types: set[EvidenceSourceType],
    ) -> list[str]:
        """按底层证据类型精确反查受影响主张 ID（支持计算卡递归溯源后的底层类型）。"""
        node = self._dossiers[dossier_id]
        return sorted({d.claim_id for d in node.dependencies if d.source_type in source_types})

    # ---------- 记忆类 5：方法经验（仅 RSI 门控发布可写） ----------
    def publish_method_asset(
        self,
        *,
        asset_type: str,
        version: str,
        payload: dict[str, Any],
        caller: str,
        gate_passed: bool,
        human_approved: bool,
    ) -> None:
        if caller != "rsi_gate" or not gate_passed or not human_approved:
            raise MemoryPermissionError("方法经验资产仅允许通过 RSI 六道门控并经人工批准后写入")
        self._method_assets.setdefault(asset_type, []).append(
            {"version": version, "payload": payload}
        )

    # ---------- 工作记忆（30 天 TTL 自动清理） ----------
    def put_working_memory(
        self,
        task_id: str,
        payload: dict[str, Any],
        *,
        created_at: datetime | None = None,
        ttl_days: int = 30,
    ) -> WorkingMemoryEntry:
        entry = WorkingMemoryEntry(
            entry_id=f"WM-{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            payload=payload,
            created_at=created_at or datetime.now(timezone.utc),
            ttl_days=ttl_days,
        )
        self._working_memory[entry.entry_id] = entry
        return entry

    def purge_expired_working_memory(self, now: datetime | None = None) -> int:
        ts = now or datetime.now(timezone.utc)
        expired_ids = [
            eid
            for eid, entry in self._working_memory.items()
            if ts - entry.created_at > timedelta(days=entry.ttl_days)
        ]
        for eid in expired_ids:
            del self._working_memory[eid]
        return len(expired_ids)

    def working_memory_count(self) -> int:
        return len(self._working_memory)

    @staticmethod
    def _extract_dependencies(
        claims: list[Claim],
        evidence_store: dict[str, EvidenceCard],
    ) -> list[ClaimDependency]:
        """递归展开计算卡 derived_from，生成精确的 Claim → Evidence → Period/SourceType 依赖边。"""
        deps: list[ClaimDependency] = []
        for claim in claims:
            visited: set[str] = set()
            stack = list(claim.evidence_ids)
            while stack:
                eid = stack.pop()
                if eid in visited:
                    continue
                visited.add(eid)
                card = evidence_store.get(eid)
                if card is None:
                    continue
                deps.append(
                    ClaimDependency(
                        claim_id=claim.claim_id,
                        evidence_id=card.evidence_id,
                        source_type=card.source_type,
                        period=card.period,
                        published_at=card.published_at,
                    )
                )
                for parent_id in card.derived_from:
                    if parent_id not in visited:
                        stack.append(parent_id)
        return deps
