"""个人情境官本地规则（含回撤 20%→10% 约束变更联动）+ 前向模拟盘禁回填与 5 条件晋级门控/回滚测试。"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from zhengyan.core.enums import DossierStatus, ResearchTier, UserIntent
from zhengyan.core.models import DossierRecord
from zhengyan.memory import (
    BackfillForbiddenError,
    DailyUniverseSnapshot,
    FiveLayerMemoryStore,
    ForwardPaperSimulator,
    PersonalContextAdvisor,
    PromotionDeniedError,
    StrategyCard,
    StrategyRegistry,
    SuitabilityLevel,
    UserConfirmedConstraints,
    evaluate_strategy_promotion_gate,
)


def test_personal_advisor_local_rules_and_drawdown_constraint_change_impact() -> None:
    """文档第 18 题：用户将可承受回撤从 20% 改成 10%，系统精准标出个人情境结论因此改变的档案。"""
    mem = FiveLayerMemoryStore()
    mem.append_dossier(
        DossierRecord(
            dossier_id="DOS-MT-1",
            symbol="600519",
            as_of=date(2025, 8, 20),
            question="茅台研究",
            intent=UserIntent.RESEARCH,
            tier=ResearchTier.STANDARD,
            status=DossierStatus.COMPLETE,
        ),
        evidence_store={},
        metrics_snapshot={
            "historical_drawdown_pct": Decimal("15"),
            "pe_ttm": Decimal("20.39"),
        },
    )
    mem.append_dossier(
        DossierRecord(
            dossier_id="DOS-BOND-LIKE-1",
            symbol="600900",
            as_of=date(2025, 8, 20),
            question="低波公用事业研究",
            intent=UserIntent.RESEARCH,
            tier=ResearchTier.STANDARD,
            status=DossierStatus.COMPLETE,
        ),
        evidence_store={},
        metrics_snapshot={
            "historical_drawdown_pct": Decimal("7"),
            "pe_ttm": Decimal("16.0"),
        },
    )

    c_old = mem.save_user_constraints(
        UserConfirmedConstraints(
            user_id="u1",
            buy_thesis="长期配置高现金流龙头",
            horizon_months=36,
            max_drawdown_pct=Decimal("20"),
            max_pe_tolerance=Decimal("25"),
        ),
        caller="user",
        explicit_user_confirmation=True,
    )
    c_new = mem.save_user_constraints(
        UserConfirmedConstraints(
            user_id="u1",
            buy_thesis="降低波动暴露",
            horizon_months=36,
            max_drawdown_pct=Decimal("10"),  # 从 20% 收紧到 10%
            max_pe_tolerance=Decimal("25"),
        ),
        caller="user",
        explicit_user_confirmation=True,
    )

    advisor = PersonalContextAdvisor()
    impacts = advisor.detect_constraint_change_impacts(
        memory_store=mem,
        old_constraints=c_old,
        new_constraints=c_new,
    )
    # 600519（历史回撤 15%）从 ALIGNED 变为 MISALIGNED；600900（历史回撤 7%）不受影响
    assert len(impacts) == 1
    assert impacts[0].symbol == "600519"
    assert impacts[0].old_suitability == SuitabilityLevel.ALIGNED
    assert impacts[0].new_suitability == SuitabilityLevel.MISALIGNED


def _run_sim_days(
    strategy: StrategyCard,
    start: date,
    num_days: int,
    price_series: list[tuple[Decimal, Decimal]],
    *,
    inject_future_dossier: bool = False,
) -> ForwardPaperSimulator:
    sim = ForwardPaperSimulator(strategy=strategy, start_date=start, initial_cash=Decimal("1000000.00"))
    for i in range(num_days):
        td = start + timedelta(days=i)
        p_mt, p_catl = price_series[i]
        snaps = [
            DailyUniverseSnapshot(
                symbol="600519",
                trade_date=td,
                close_price=p_mt,
                dossier_id="DOS-600519-V1",
                dossier_version=1,
                dossier_as_of=(td + timedelta(days=5)) if inject_future_dossier else date(2025, 8, 20),
                pe_percentile=Decimal("35"),
                volatility_pct=Decimal("22"),
                verified_claims_count=5,
                pending_alerts_count=0,
            ),
            DailyUniverseSnapshot(
                symbol="300750",
                trade_date=td,
                close_price=p_catl,
                dossier_id="DOS-300750-V2",
                dossier_version=2,
                dossier_as_of=date(2025, 8, 20),
                pe_percentile=Decimal("42"),
                volatility_pct=Decimal("28"),
                verified_claims_count=4,
                pending_alerts_count=0,
            ),
        ]
        sim.step_day(td, snaps)
    return sim


def test_forward_paper_sim_forbids_backfill_enforces_lots_and_promotion_rollback() -> None:
    start = date(2025, 8, 21)
    s_v1 = StrategyCard(strategy_id="STR-CORE", version="v1", max_pe_percentile=Decimal("40"))
    s_v2 = StrategyCard(strategy_id="STR-CORE", version="v2", max_pe_percentile=Decimal("50"))

    # 1. 严禁历史回填（早于起始日或重复/倒退日期直接抛 BackfillForbiddenError）
    sim_guard = ForwardPaperSimulator(strategy=s_v2, start_date=start)
    with pytest.raises(BackfillForbiddenError):
        sim_guard.step_day(date(2025, 8, 20), [])
    sim_guard.step_day(start, [])
    with pytest.raises(BackfillForbiddenError):
        sim_guard.step_day(start, [])

    # 2. 10 个交易日稳健上涨路径：v2 纳入 600519 + 300750，等权整手，5 条晋级门控全过
    prices = [
        (Decimal("1460") + Decimal(i * 10), Decimal("260") + Decimal(i * 4))
        for i in range(10)
    ]
    sim_active = _run_sim_days(s_v1, start, 10, prices)
    sim_cand = _run_sim_days(s_v2, start, 10, prices)

    assert all(t.shares % 100 == 0 for t in sim_cand.trades)
    gate = evaluate_strategy_promotion_gate(candidate_sim=sim_cand, active_sim=sim_active)
    assert gate.passed is True
    assert gate.trading_days == 10
    assert gate.has_future_leak is False
    assert gate.all_trades_traceable is True

    # 3. 人工批准晋级 + 一键回滚
    reg = StrategyRegistry(active_strategy=s_v1)
    with pytest.raises(PromotionDeniedError, match="人工显式批准"):
        reg.promote(candidate=s_v2, gate_report=gate, approved_by="alice", human_approved=False)

    reg.promote(candidate=s_v2, gate_report=gate, approved_by="alice", human_approved=True)
    assert reg.active_strategy.version == "v2"

    rolled = reg.rollback(operator="alice", reason="观察期实盘波动扩大，回滚至 v1")
    assert rolled.version == "v1"
    assert reg.active_strategy.version == "v1"
    assert reg.events[-1]["event"] == "rollback"


def test_promotion_gate_blocks_short_days_future_leak_and_underperformance() -> None:
    start = date(2025, 8, 21)
    s_v1 = StrategyCard(strategy_id="STR-CORE", version="v1", max_pe_percentile=Decimal("40"))
    s_v2 = StrategyCard(strategy_id="STR-CORE", version="v2", max_pe_percentile=Decimal("50"))

    # Case A: 交易日不足 10 天（仅 5 天）→ 拦截
    prices_5 = [(Decimal("1460"), Decimal("260"))] * 5
    g_short = evaluate_strategy_promotion_gate(
        candidate_sim=_run_sim_days(s_v2, start, 5, prices_5),
        active_sim=_run_sim_days(s_v1, start, 5, prices_5),
    )
    assert g_short.passed is False
    assert any("不足 10 天" in c for c in g_short.failed_checks)

    # Case B: 存在前视偏差（引用未来档案）→ 拦截
    prices_10 = [(Decimal("1460"), Decimal("260"))] * 10
    g_leak = evaluate_strategy_promotion_gate(
        candidate_sim=_run_sim_days(s_v2, start, 10, prices_10, inject_future_dossier=True),
        active_sim=_run_sim_days(s_v1, start, 10, prices_10),
    )
    assert g_leak.passed is False
    assert g_leak.has_future_leak is True

    # Case C: 候选组合（多买了暴跌的 300750）相对在用版本落后超过 2% 且超回撤 → 拦截
    prices_crash = [
        (Decimal("1460"), Decimal("260") - Decimal(i * 12))
        for i in range(10)
    ]
    g_under = evaluate_strategy_promotion_gate(
        candidate_sim=_run_sim_days(s_v2, start, 10, prices_crash),
        active_sim=_run_sim_days(s_v1, start, 10, prices_crash),
    )
    assert g_under.passed is False
    assert any("同期收益落后" in c for c in g_under.failed_checks)
