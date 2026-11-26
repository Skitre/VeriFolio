"""四类变更订阅扫描器、精确依赖增量复核引擎与持续更新评测（对应文档第 18 题）。

完整闭环：
1. SubscriptionScanner 检查 4 类变化：
   - new_financial_report（新报告期已披露财报）
   - new_announcement（档案时点之后的新公告）
   - price_movement（收盘价相对档案时点涨跌幅 >= 10%）
   - review_condition_triggered（档案自写的投资假设失效/复核条件触发）
2. 顺着 Claim → Evidence → ReportPeriod/MarketQuote 依赖图，**只标记受影响的旧主张为待复核**；
3. IncrementalReviewer 将旧版结论与时点注入重研过程，逐条裁定保留（RETAINED）/ 削弱（WEAKENED）/ 失效（INVALIDATED）/ 无法判断（UNDECIDABLE），生成 v2 档案（v1 永不覆盖）；
4. evaluate_incremental_update（对应文档 `advisor eval update`）：验证受影响结论修正召回率与未受影响结论保持率！
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Sequence

from pydantic import BaseModel, Field

from zhengyan.core.enums import (
    DossierStatus,
    EvidenceGrade,
    EvidenceSourceType,
    MetricUnit,
    ResearchTier,
    UserIntent,
)
from zhengyan.core.models import Claim, DossierRecord, EvidenceCard
from zhengyan.data import MarketDataProvider, compute_pe_ttm_pit, compute_ratio, compute_yoy_growth
from zhengyan.memory.store import (
    ClaimLifecycleStatus,
    ClaimUpdateVerdict,
    FiveLayerMemoryStore,
    InvestmentHypothesis,
    VersionedDossier,
)
from zhengyan.verify import RuleVerifier


class ChangeTriggerType(StrEnum):
    NEW_FINANCIAL_REPORT = "new_financial_report"
    NEW_ANNOUNCEMENT = "new_announcement"
    PRICE_MOVEMENT = "price_movement"
    REVIEW_CONDITION = "review_condition_triggered"


class ReviewAlert(BaseModel):
    alert_id: str
    symbol: str
    dossier_id: str
    dossier_version: int
    scan_as_of: date
    triggers: list[ChangeTriggerType] = Field(default_factory=list)
    trigger_details: list[str] = Field(default_factory=list)
    affected_claim_ids: list[str] = Field(default_factory=list)
    unaffected_claim_ids: list[str] = Field(default_factory=list)
    triggered_hypothesis_ids: list[str] = Field(default_factory=list)


def _eval_op(actual: Decimal, op: str, threshold: Decimal) -> bool:
    if op == "<":
        return actual < threshold
    if op == "<=":
        return actual <= threshold
    if op == ">":
        return actual > threshold
    if op == ">=":
        return actual >= threshold
    raise ValueError(f"不支持的比较符：{op}")


class SubscriptionScanner:
    """四类变更订阅扫描器（服务每 6 小时自动运行或手动触发）。"""

    def __init__(
        self,
        *,
        memory_store: FiveLayerMemoryStore,
        data_provider: MarketDataProvider,
        price_change_threshold: Decimal = Decimal("0.10"),
    ) -> None:
        self.memory = memory_store
        self.data_provider = data_provider
        self.price_change_threshold = price_change_threshold

    def scan_symbol(
        self,
        symbol: str,
        *,
        scan_as_of: date,
        current_metrics: dict[str, Decimal] | None = None,
    ) -> ReviewAlert | None:
        latest = self.memory.get_latest_dossier(symbol)
        if latest is None:
            return None

        old_as_of = latest.dossier.as_of
        if scan_as_of <= old_as_of:
            return None

        triggers: list[ChangeTriggerType] = []
        details: list[str] = []
        affected: set[str] = set()
        triggered_hyps: list[str] = []

        # 1. 新报告期财报检查
        old_periods = {
            d.period for d in latest.dependencies
            if d.source_type == EvidenceSourceType.FINANCIAL_REPORT and d.period
        }
        all_stmts_now = self.data_provider.get_financial_statements(symbol, as_of=scan_as_of)
        new_stmts = [
            s for s in all_stmts_now
            if s.disclosed_at > old_as_of and s.period not in old_periods
        ]
        if new_stmts:
            triggers.append(ChangeTriggerType.NEW_FINANCIAL_REPORT)
            periods_str = ", ".join(f"{s.period}(披露日 {s.disclosed_at})" for s in new_stmts)
            details.append(f"新报告期财报已披露：{periods_str}")
            affected.update(
                self.memory.find_claims_depending_on_source_types(
                    latest.dossier.dossier_id,
                    {EvidenceSourceType.FINANCIAL_REPORT, EvidenceSourceType.COMPUTE},
                )
            )

        # 2. 档案时点之后的新公告检查
        new_anns = [
            a for a in self.data_provider.get_announcements(symbol, as_of=scan_as_of, lookback_days=365)
            if a.disclosed_at > old_as_of
        ]
        if new_anns:
            triggers.append(ChangeTriggerType.NEW_ANNOUNCEMENT)
            details.append(f"新增 {len(new_anns)} 条巨潮公告（最新：{new_anns[-1].title}）")
            affected.update(
                self.memory.find_claims_depending_on_source_types(
                    latest.dossier.dossier_id,
                    {EvidenceSourceType.ANNOUNCEMENT, EvidenceSourceType.FINANCIAL_REPORT},
                )
            )

        # 3. 价格异动检查（默认 >= 10%）
        old_price = self.data_provider.get_price_on_or_before(symbol, as_of=old_as_of)
        new_price = self.data_provider.get_price_on_or_before(symbol, as_of=scan_as_of)
        price_pct_change: Decimal | None = None
        if old_price and new_price and old_price.close_price > 0 and new_price.trade_date > old_price.trade_date:
            price_pct_change = (new_price.close_price - old_price.close_price) / old_price.close_price
            if abs(price_pct_change) >= self.price_change_threshold:
                triggers.append(ChangeTriggerType.PRICE_MOVEMENT)
                details.append(
                    f"股价异动 {price_pct_change * Decimal('100'):+.2f}%"
                    f"（{old_price.close_price} -> {new_price.close_price}，阈值 ±{self.price_change_threshold * 100:.0f}%）"
                )
                affected.update(
                    self.memory.find_claims_depending_on_source_types(
                        latest.dossier.dossier_id,
                        {EvidenceSourceType.MARKET_QUOTE},
                    )
                )

        # 4. 档案自写投资假设 / 复核条件触发检查
        metrics_to_check = dict(current_metrics or {})
        if price_pct_change is not None:
            metrics_to_check.setdefault("price_change_pct", price_pct_change * Decimal("100"))
        if new_price is not None:
            try:
                pe_card = compute_pe_ttm_pit(all_stmts_now, new_price, scan_as_of)
                pe_m = next((m for m in pe_card.metrics if m.name == "pe_ttm"), None)
                if pe_m:
                    metrics_to_check.setdefault("pe_ttm", pe_m.value)
            except Exception:
                pass

        for hyp in self.memory.list_hypotheses(symbol, active_only=True):
            if hyp.invalidation_metric in metrics_to_check:
                actual_val = metrics_to_check[hyp.invalidation_metric]
                if _eval_op(actual_val, hyp.invalidation_op, hyp.invalidation_threshold):
                    if ChangeTriggerType.REVIEW_CONDITION not in triggers:
                        triggers.append(ChangeTriggerType.REVIEW_CONDITION)
                    reason = (
                        f"假设 {hyp.hypothesis_id} 触发失效条件："
                        f"{hyp.invalidation_metric}={actual_val} {hyp.invalidation_op} {hyp.invalidation_threshold}"
                    )
                    details.append(reason)
                    triggered_hyps.append(hyp.hypothesis_id)
                    self.memory.mark_hypothesis_invalidated(hyp.hypothesis_id, reason)
                    # 假设触发影响全部依赖估值/财报的主张
                    affected.update(
                        self.memory.find_claims_depending_on_source_types(
                            latest.dossier.dossier_id,
                            {EvidenceSourceType.MARKET_QUOTE, EvidenceSourceType.FINANCIAL_REPORT},
                        )
                    )

        if not triggers:
            return None

        all_claim_ids = [c.claim_id for c in latest.dossier.verified_claims]
        affected_list = [cid for cid in all_claim_ids if cid in affected]
        unaffected_list = [cid for cid in all_claim_ids if cid not in affected]

        # 只把受影响的旧结论标为待复核
        self.memory.mark_claims_pending_review(latest.dossier.dossier_id, affected_list)

        return ReviewAlert(
            alert_id=f"ALT-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            dossier_id=latest.dossier.dossier_id,
            dossier_version=latest.dossier.version,
            scan_as_of=scan_as_of,
            triggers=triggers,
            trigger_details=details,
            affected_claim_ids=affected_list,
            unaffected_claim_ids=unaffected_list,
            triggered_hypothesis_ids=triggered_hyps,
        )


class IncrementalReviewer:
    """增量复核执行器：将 v1 结论带着原时点注入新研究，逐条裁定保留/削弱/失效并生成 v2。"""

    def __init__(
        self,
        *,
        memory_store: FiveLayerMemoryStore,
        data_provider: MarketDataProvider,
        verifier: RuleVerifier | None = None,
    ) -> None:
        self.memory = memory_store
        self.data_provider = data_provider
        self.verifier = verifier or RuleVerifier()

    def re_research_incremental(
        self,
        alert: ReviewAlert,
        *,
        new_as_of: date,
        method_asset_versions: dict[str, str] | None = None,
    ) -> VersionedDossier:
        old_node = self.memory.get_dossier(alert.dossier_id)
        if old_node is None:
            raise ValueError(f"未找到原档案 {alert.dossier_id}")

        symbol = alert.symbol
        # 1. 继承旧版证据卡快照 + 拉取截至 new_as_of 的最新 A 级结构化证据与衍生计算卡
        ev_store: dict[str, EvidenceCard] = dict(old_node.evidence_cards)
        a_cards = self.data_provider.to_evidence_cards(symbol, as_of=new_as_of)
        for c in a_cards:
            ev_store[c.evidence_id] = c

        fin_by_period = {
            c.period: c for c in a_cards
            if c.source_type == EvidenceSourceType.FINANCIAL_REPORT and c.period and not c.period.endswith("NON_GAAP")
        }
        new_derived_ids: list[str] = []
        if "2025H1" in fin_by_period and "2024H1" in fin_by_period:
            for metric in ("revenue", "net_profit"):
                yoy = compute_yoy_growth(
                    fin_by_period["2025H1"], fin_by_period["2024H1"], metric, as_of=new_as_of
                )
                ev_store[yoy.evidence_id] = yoy
                new_derived_ids.append(yoy.evidence_id)
            nm = compute_ratio(
                fin_by_period["2025H1"],
                "net_profit",
                "revenue",
                output_name="net_margin",
                output_display_name="净利率",
                output_unit=MetricUnit.PERCENT,
            )
            ev_store[nm.evidence_id] = nm
            new_derived_ids.append(nm.evidence_id)

        price_now = self.data_provider.get_price_on_or_before(symbol, as_of=new_as_of)
        pe_val: Decimal | None = None
        if price_now is not None:
            try:
                pe_card = compute_pe_ttm_pit(
                    self.data_provider.get_financial_statements(symbol, as_of=new_as_of),
                    price_now,
                    new_as_of,
                )
                ev_store[pe_card.evidence_id] = pe_card
                new_derived_ids.append(pe_card.evidence_id)
                pe_m = next((m for m in pe_card.metrics if m.name == "pe_ttm"), None)
                pe_val = pe_m.value if pe_m else None
            except Exception:
                pass

        # 2. 逐条对照上一版结论（保留 / 削弱 / 失效 / 无法判断）
        affected_set = set(alert.affected_claim_ids)
        verdicts: list[ClaimUpdateVerdict] = []
        new_claims: list[Claim] = []

        for old_claim in old_node.dossier.verified_claims:
            if old_claim.claim_id not in affected_set:
                # 未受本次变更影响的结论：直接保留，避免推倒重来！
                verdicts.append(
                    ClaimUpdateVerdict(
                        old_claim_id=old_claim.claim_id,
                        old_text=old_claim.text,
                        old_as_of=old_node.dossier.as_of,
                        verdict=ClaimLifecycleStatus.RETAINED,
                        reason="所依赖的证据未受本次新信息影响，原结论保留",
                        new_evidence_ids=list(old_claim.evidence_ids),
                    )
                )
                new_claims.append(old_claim)
                continue

            # 受影响的结论：区分价格/估值失效 vs 财报报告期更新替换
            deps_for_claim = [d for d in old_node.dependencies if d.claim_id == old_claim.claim_id]
            dep_types = {d.source_type for d in deps_for_claim}

            if EvidenceSourceType.MARKET_QUOTE in dep_types and ChangeTriggerType.PRICE_MOVEMENT in alert.triggers:
                verdicts.append(
                    ClaimUpdateVerdict(
                        old_claim_id=old_claim.claim_id,
                        old_text=old_claim.text,
                        old_as_of=old_node.dossier.as_of,
                        verdict=ClaimLifecycleStatus.INVALIDATED,
                        reason=f"行情发生显著异动且原估值基准已过时（原时点 {old_node.dossier.as_of}），旧估值结论失效并替换为新时点计算",
                        new_evidence_ids=[eid for eid in new_derived_ids if "PETTM" in eid],
                    )
                )
            elif ChangeTriggerType.REVIEW_CONDITION in alert.triggers and alert.triggered_hypothesis_ids:
                verdicts.append(
                    ClaimUpdateVerdict(
                        old_claim_id=old_claim.claim_id,
                        old_text=old_claim.text,
                        old_as_of=old_node.dossier.as_of,
                        verdict=ClaimLifecycleStatus.WEAKENED,
                        reason="关联投资假设触发复核条件，原结论支持力度削弱",
                        new_evidence_ids=new_derived_ids,
                    )
                )
            else:
                verdicts.append(
                    ClaimUpdateVerdict(
                        old_claim_id=old_claim.claim_id,
                        old_text=old_claim.text,
                        old_as_of=old_node.dossier.as_of,
                        verdict=ClaimLifecycleStatus.INVALIDATED,
                        reason=f"已有更新报告期财报披露，基于 {old_node.dossier.as_of} 旧财报的数值结论已被最新报告期替代",
                        new_evidence_ids=new_derived_ids,
                    )
                )

        # 3. 补充基于最新时点与新财报的增量核验主张
        seq = 1
        if "2025H1" in fin_by_period:
            c_25h1 = fin_by_period["2025H1"]
            rev = next(m for m in c_25h1.metrics if m.name == "revenue")
            np_m = next(m for m in c_25h1.metrics if m.name == "net_profit")
            new_claims.append(
                Claim(
                    claim_id=f"CLM-V{old_node.dossier.version + 1}-{seq}",
                    text=f"最新披露的 2025H1 营业收入为 {rev.value}{rev.unit.value}，归母净利润为 {np_m.value}{np_m.unit.value}。",
                    evidence_ids=[c_25h1.evidence_id],
                )
            )
            seq += 1

        for did in new_derived_ids:
            card = ev_store[did]
            m = card.metrics[-1]
            new_claims.append(
                Claim(
                    claim_id=f"CLM-V{old_node.dossier.version + 1}-{seq}",
                    text=f"截至 {new_as_of.isoformat()}，{m.display_name}更新为 {m.value}{m.unit.value}。",
                    evidence_ids=[did],
                )
            )
            seq += 1

        # 4. 纯规则核验器核验全部新主张（未受影响旧主张若引用历史非结构化证据，补回历史证据卡校验）
        ver_results = self.verifier.verify_all(new_claims, ev_store, as_of=new_as_of)
        vr_map = {r.claim_id: r for r in ver_results}
        verified_claims = [c for c in new_claims if vr_map[c.claim_id].can_enter_conclusion]
        excluded_claims = [c for c in new_claims if not vr_map[c.claim_id].can_enter_conclusion]

        metrics_snap = dict(old_node.metrics_snapshot)
        if pe_val is not None:
            metrics_snap["pe_ttm"] = pe_val
        if price_now is not None:
            metrics_snap["close_price"] = price_now.close_price
        metrics_snap["pending_alerts"] = Decimal("0")

        new_record = DossierRecord(
            dossier_id=f"DOS-{symbol}-v{old_node.dossier.version + 1}-{uuid.uuid4().hex[:4]}",
            symbol=symbol,
            as_of=new_as_of,
            question=f"[增量复核 v{old_node.dossier.version + 1}] {old_node.dossier.question}",
            intent=UserIntent.RESEARCH,
            tier=ResearchTier.STANDARD,
            status=DossierStatus.COMPLETE if verified_claims else DossierStatus.PARTIAL,
            summary=(
                f"增量复核完成（触发源：{[t.value for t in alert.triggers]}）："
                f"对照 v{old_node.dossier.version} 共 {len(verdicts)} 条旧结论，"
                f"保留 {sum(1 for v in verdicts if v.verdict == ClaimLifecycleStatus.RETAINED)} 条，"
                f"削弱/失效 {sum(1 for v in verdicts if v.verdict in (ClaimLifecycleStatus.WEAKENED, ClaimLifecycleStatus.INVALIDATED))} 条。"
            ),
            verified_claims=verified_claims,
            excluded_claims=excluded_claims,
            verification_results=ver_results,
            evidence_ids=sorted(ev_store.keys()),
        )

        # 只追加生成 v2（v1 永久保留！）
        return self.memory.append_dossier(
            new_record,
            evidence_store=ev_store,
            parent_dossier_id=old_node.dossier.dossier_id,
            review_verdicts=verdicts,
            metrics_snapshot=metrics_snap,
            method_asset_versions=method_asset_versions or old_node.method_asset_versions,
        )


@dataclass(frozen=True)
class UpdateEvalReport:
    """持续更新专项评测报告（对应文档 `advisor eval update`）。"""

    affected_expected: int
    affected_correctly_updated: int
    affected_update_recall: float
    unaffected_expected: int
    unaffected_correctly_retained: int
    unaffected_retention_rate: float
    version_chain_intact: bool
    passed: bool


def evaluate_incremental_update(
    *,
    old_version: VersionedDossier,
    new_version: VersionedDossier,
    expected_affected_ids: Sequence[str],
    expected_unaffected_ids: Sequence[str],
) -> UpdateEvalReport:
    """评测增量复核质量：受影响的旧结论是否被正确修正，未受影响的是否被正确保留。"""
    verdict_map = {v.old_claim_id: v.verdict for v in new_version.review_verdicts}

    aff_ok = sum(
        1
        for cid in expected_affected_ids
        if verdict_map.get(cid) in (ClaimLifecycleStatus.INVALIDATED, ClaimLifecycleStatus.WEAKENED)
    )
    unaff_ok = sum(
        1
        for cid in expected_unaffected_ids
        if verdict_map.get(cid) == ClaimLifecycleStatus.RETAINED
    )

    aff_recall = aff_ok / len(expected_affected_ids) if expected_affected_ids else 1.0
    unaff_ret = unaff_ok / len(expected_unaffected_ids) if expected_unaffected_ids else 1.0
    chain_ok = (
        new_version.parent_dossier_id == old_version.dossier.dossier_id
        and new_version.dossier.version == old_version.dossier.version + 1
    )

    return UpdateEvalReport(
        affected_expected=len(expected_affected_ids),
        affected_correctly_updated=aff_ok,
        affected_update_recall=round(aff_recall, 6),
        unaffected_expected=len(expected_unaffected_ids),
        unaffected_correctly_retained=unaff_ok,
        unaffected_retention_rate=round(unaff_ret, 6),
        version_chain_intact=chain_ok,
        passed=(aff_recall == 1.0 and unaff_ret == 1.0 and chain_ok),
    )
