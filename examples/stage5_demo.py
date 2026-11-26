"""Stage 5 端到端演示：宁德时代 300750 增量复核(v1→提醒→v2) + 约束变更联动 + 前向模拟盘晋级门控。

运行：uv run python examples/stage5_demo.py
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from zhengyan.core.enums import (
    DossierStatus,
    EvidenceGrade,
    EvidenceSourceType,
    ResearchTier,
    UserIntent,
)
from zhengyan.core.models import Claim, DossierRecord, EvidenceCard
from zhengyan.data import MockMarketDataProvider, compute_pe_ttm_pit
from zhengyan.memory import (
    DailyUniverseSnapshot,
    FiveLayerMemoryStore,
    ForwardPaperSimulator,
    IncrementalReviewer,
    InvestmentHypothesis,
    PersonalContextAdvisor,
    StrategyCard,
    StrategyRegistry,
    SubscriptionScanner,
    UserConfirmedConstraints,
    evaluate_incremental_update,
    evaluate_strategy_promotion_gate,
)


def main() -> None:
    provider = MockMarketDataProvider()
    mem = FiveLayerMemoryStore()

    print("=== 1. 宁德时代 300750：v1 建档 → 四类变更订阅扫描 → 增量重研 v2 + eval_update ===\n")
    t1 = date(2025, 4, 1)
    v1_ev = {c.evidence_id: c for c in provider.to_evidence_cards("300750", as_of=t1)}
    price_t1 = provider.get_price_on_or_before("300750", t1)
    assert price_t1 is not None
    pe_t1 = compute_pe_ttm_pit(provider.get_financial_statements("300750", t1), price_t1, t1)
    v1_ev[pe_t1.evidence_id] = pe_t1
    web_eu = EvidenceCard(
        evidence_id="EV-WEB-300750-EU",
        grade=EvidenceGrade.B,
        source_type=EvidenceSourceType.WEB,
        source_name="证券时报网",
        title="宁德时代欧洲匈牙利基地稳步建设",
        symbol="300750",
        published_at=date(2025, 3, 20),
        authority=4,
        has_full_snapshot=True,
        excerpt="海外本地化产能持续推进。",
    )
    v1_ev[web_eu.evidence_id] = web_eu

    v1 = mem.append_dossier(
        DossierRecord(
            dossier_id="DOS-300750-V1",
            symbol="300750",
            as_of=t1,
            question="复核宁德时代 2024FY 与估值",
            intent=UserIntent.RESEARCH,
            tier=ResearchTier.STANDARD,
            status=DossierStatus.COMPLETE,
            verified_claims=[
                Claim(claim_id="CLM-V1-FIN", text="2024FY 营收 3620.13 亿元，归母净利润 507.45 亿元。", evidence_ids=["EV-FIN-300750-2024FY"]),
                Claim(claim_id="CLM-V1-VAL", text=f"2025-04-01 PE(TTM) 为 {pe_t1.metrics[-1].value} 倍。", evidence_ids=[pe_t1.evidence_id]),
                Claim(claim_id="CLM-V1-EU", text="欧洲本地化建厂稳步推进，贴近海外主机厂客户。", evidence_ids=[web_eu.evidence_id]),
            ],
        ),
        evidence_store=v1_ev,
        metrics_snapshot={"pe_ttm": pe_t1.metrics[-1].value, "historical_drawdown_pct": Decimal("16")},
    )
    mem.add_hypothesis(
        InvestmentHypothesis(
            hypothesis_id="HYP-CATL-1",
            symbol="300750",
            dossier_id=v1.dossier.dossier_id,
            statement="股价变动超 15% 触发估值安全边际复核",
            invalidation_metric="price_change_pct",
            invalidation_op=">=",
            invalidation_threshold=Decimal("15"),
        )
    )

    scanner = SubscriptionScanner(memory_store=mem, data_provider=provider)
    alert = scanner.scan_symbol("300750", scan_as_of=date(2025, 8, 20))
    assert alert is not None
    print(f"  [扫描提醒] 触发类型={[t.value for t in alert.triggers]}")
    for d in alert.trigger_details:
        print(f"    ↳ {d}")
    print(f"  [精确依赖定位] 受影响主张={alert.affected_claim_ids} | 未受影响主张={alert.unaffected_claim_ids}")

    reviewer = IncrementalReviewer(memory_store=mem, data_provider=provider)
    v2 = reviewer.re_research_incremental(alert, new_as_of=date(2025, 8, 20))
    print(f"  [生成新版本] {v2.dossier.dossier_id} v{v2.dossier.version} (parent={v2.parent_dossier_id})")
    for rv in v2.review_verdicts:
        print(f"    ↳ 对照旧主张 {rv.old_claim_id}: {rv.verdict.value.upper()} — {rv.reason}")
    upd_eval = evaluate_incremental_update(
        old_version=v1,
        new_version=v2,
        expected_affected_ids=["CLM-V1-FIN", "CLM-V1-VAL"],
        expected_unaffected_ids=["CLM-V1-EU"],
    )
    print(f"  [eval_update] 修正召回={upd_eval.affected_update_recall*100:.0f}% | 保留率={upd_eval.unaffected_retention_rate*100:.0f}% | 通过={upd_eval.passed}\n")

    print("=== 2. 个人情境官（纯本地规则）与可承受回撤 20% → 10% 联动 ===\n")
    c_old = mem.save_user_constraints(
        UserConfirmedConstraints(user_id="demo_u", buy_thesis="核心成长", horizon_months=24, max_drawdown_pct=Decimal("20")),
        caller="user",
        explicit_user_confirmation=True,
    )
    c_new = mem.save_user_constraints(
        UserConfirmedConstraints(user_id="demo_u", buy_thesis="风控收紧", horizon_months=24, max_drawdown_pct=Decimal("10")),
        caller="user",
        explicit_user_confirmation=True,
    )
    impacts = PersonalContextAdvisor().detect_constraint_change_impacts(
        memory_store=mem, old_constraints=c_old, new_constraints=c_new
    )
    for imp in impacts:
        print(f"  ↳ 标的 {imp.symbol} ({imp.dossier_id}): {imp.old_suitability.value} -> {imp.new_suitability.value} | {imp.new_triggers}\n")

    print("=== 3. 策略卡 + 10 交易日前向模拟盘（等权/整手/禁回填）+ 晋级门控与回滚 ===\n")
    start = date(2025, 8, 21)
    s_v1 = StrategyCard(strategy_id="STR-CORE", version="v1", max_pe_percentile=Decimal("40"))
    s_v2 = StrategyCard(strategy_id="STR-CORE", version="v2", max_pe_percentile=Decimal("55"))
    sim_v1 = ForwardPaperSimulator(strategy=s_v1, start_date=start)
    sim_v2 = ForwardPaperSimulator(strategy=s_v2, start_date=start)
    for i in range(10):
        td = start + timedelta(days=i)
        snaps = [
            DailyUniverseSnapshot(
                symbol="600519", trade_date=td, close_price=Decimal("1460") + Decimal(i * 8),
                dossier_id="DOS-600519-V1", dossier_version=1, dossier_as_of=date(2025, 8, 20),
                pe_percentile=Decimal("35"), volatility_pct=Decimal("20"), verified_claims_count=5, pending_alerts_count=0,
            ),
            DailyUniverseSnapshot(
                symbol="300750", trade_date=td, close_price=Decimal("260") + Decimal(i * 4),
                dossier_id=v2.dossier.dossier_id, dossier_version=2, dossier_as_of=date(2025, 8, 20),
                pe_percentile=Decimal("48"), volatility_pct=Decimal("26"), verified_claims_count=4, pending_alerts_count=0,
            ),
        ]
        sim_v1.step_day(td, snaps)
        sim_v2.step_day(td, snaps)

    gate = evaluate_strategy_promotion_gate(candidate_sim=sim_v2, active_sim=sim_v1)
    print(
        f"  [晋级门控] 交易日={gate.trading_days} | 未来泄漏={gate.has_future_leak} | "
        f"最大回撤={gate.max_drawdown*100:.2f}% | 收益差={gate.return_diff*100:+.2f}% | 通过={gate.passed}"
    )
    reg = StrategyRegistry(active_strategy=s_v1)
    reg.promote(candidate=s_v2, gate_report=gate, approved_by="researcher", human_approved=True)
    print(f"  [人工晋级] 当前在用策略版本={reg.active_strategy.version}")
    reg.rollback(operator="researcher", reason="演练一键回滚")
    print(f"  [一键回滚] 当前在用策略版本={reg.active_strategy.version} | 账本事件数={len(reg.events)}")


if __name__ == "__main__":
    main()
