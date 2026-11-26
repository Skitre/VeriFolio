"""四类变更订阅扫描 + 宁德时代 300750 “提醒 → 重新研究 → v2” 增量复核闭环与 eval_update 评测测试。"""

from __future__ import annotations

from datetime import date
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
    ChangeTriggerType,
    ClaimLifecycleStatus,
    FiveLayerMemoryStore,
    IncrementalReviewer,
    InvestmentHypothesis,
    SubscriptionScanner,
    evaluate_incremental_update,
)


def test_catl_300750_alert_re_research_v2_and_eval_update() -> None:
    """完整走通文档第 18 题宁德时代真实闭环：
    1. 2025-04-01 建立 300750 v1 档案（依赖 2024FY 财报、2025-04-01 PE 估值、以及 1 条纯行业产能 B 级网页结论）
    2. 2025-08-20 订阅扫描发现：
       - 新财报 2025H1（披露日 2025-07-31）
       - 新公告 ANN-300750-20250731
       - 股价大涨 +18.18%（220 → 260，超 10% 阈值）
       - 投资假设复核条件命中（price_change_pct >= 15%）
    3. 精确依赖定位：仅依赖财报与行情的旧主张标为 PENDING_REVIEW，纯行业海外建厂主张不受影响！
    4. 增量重新研究生成 v2（v1 完整保留），逐条裁定保留/失效，eval_update 100% 通过！
    """
    provider = MockMarketDataProvider()
    mem = FiveLayerMemoryStore()

    # Step 1: 构造宁德时代 2025-04-01 的 v1 档案
    t1 = date(2025, 4, 1)
    v1_ev = {c.evidence_id: c for c in provider.to_evidence_cards("300750", as_of=t1)}
    price_t1 = provider.get_price_on_or_before("300750", t1)
    assert price_t1 is not None
    pe_t1 = compute_pe_ttm_pit(provider.get_financial_statements("300750", t1), price_t1, t1)
    v1_ev[pe_t1.evidence_id] = pe_t1

    # 独立非财报/非行情结论（如欧洲匈牙利工厂长期战略布局，来自 B 级网页）
    web_moat = EvidenceCard(
        evidence_id="EV-WEB-300750-EU-PLANT",
        grade=EvidenceGrade.B,
        source_type=EvidenceSourceType.WEB,
        source_name="证券时报网",
        title="宁德时代欧洲匈牙利工厂推进建设",
        symbol="300750",
        published_at=date(2025, 3, 20),
        authority=4,
        has_full_snapshot=True,
        excerpt="宁德时代欧洲本地化产能持续推进，有助于服务当地主机厂客户。",
    )
    v1_ev[web_moat.evidence_id] = web_moat

    c_fin = Claim(
        claim_id="CLM-V1-FIN",
        text="宁德时代 2024FY 实现营业收入 3620.13 亿元，归母净利润 507.45 亿元。",
        evidence_ids=["EV-FIN-300750-2024FY"],
    )
    c_val = Claim(
        claim_id="CLM-V1-VAL",
        text=f"截至 2025-04-01，按披露日对齐的市盈率 PE(TTM) 为 {pe_t1.metrics[-1].value} 倍。",
        evidence_ids=[pe_t1.evidence_id],
    )
    c_eu = Claim(
        claim_id="CLM-V1-EU-MOAT",
        text="公司持续推进欧洲本地化建厂以贴近海外主机厂客户。",
        evidence_ids=[web_moat.evidence_id],
    )

    d1 = DossierRecord(
        dossier_id="DOS-300750-V1",
        symbol="300750",
        as_of=t1,
        question="复核宁德时代 2024 年报与估值水平",
        intent=UserIntent.RESEARCH,
        tier=ResearchTier.STANDARD,
        status=DossierStatus.COMPLETE,
        verified_claims=[c_fin, c_val, c_eu],
        evidence_ids=sorted(v1_ev.keys()),
    )
    v1 = mem.append_dossier(
        d1,
        evidence_store=v1_ev,
        metrics_snapshot={
            "pe_ttm": pe_t1.metrics[-1].value,
            "close_price": price_t1.close_price,
            "historical_drawdown_pct": Decimal("16"),
        },
    )
    # 登记档案自写复核条件（投资假设）
    mem.add_hypothesis(
        InvestmentHypothesis(
            hypothesis_id="HYP-CATL-PRICE",
            symbol="300750",
            dossier_id=v1.dossier.dossier_id,
            statement="股价相对研究时点波动超过 15% 时须重新复核估值安全边际",
            invalidation_metric="price_change_pct",
            invalidation_op=">=",
            invalidation_threshold=Decimal("15"),
        )
    )

    # Step 2: 2025-08-20 订阅扫描触发 4 类变化
    scanner = SubscriptionScanner(memory_store=mem, data_provider=provider)
    t2 = date(2025, 8, 20)
    alert = scanner.scan_symbol("300750", scan_as_of=t2)
    assert alert is not None
    assert ChangeTriggerType.NEW_FINANCIAL_REPORT in alert.triggers
    assert ChangeTriggerType.NEW_ANNOUNCEMENT in alert.triggers
    assert ChangeTriggerType.PRICE_MOVEMENT in alert.triggers
    assert ChangeTriggerType.REVIEW_CONDITION in alert.triggers

    # 精确受影响集合：财报与估值主张受影响，欧洲建厂 B 级网页主张不受影响！
    assert set(alert.affected_claim_ids) == {"CLM-V1-FIN", "CLM-V1-VAL"}
    assert alert.unaffected_claim_ids == ["CLM-V1-EU-MOAT"]

    # v1 档案中受影响主张已标为 PENDING_REVIEW，未受影响仍为 ACTIVE
    v1_refreshed = mem.get_dossier("DOS-300750-V1")
    assert v1_refreshed is not None
    assert v1_refreshed.claim_statuses["CLM-V1-FIN"] == ClaimLifecycleStatus.PENDING_REVIEW
    assert v1_refreshed.claim_statuses["CLM-V1-VAL"] == ClaimLifecycleStatus.PENDING_REVIEW
    assert v1_refreshed.claim_statuses["CLM-V1-EU-MOAT"] == ClaimLifecycleStatus.ACTIVE

    # Step 3: 增量重新研究生成 v2（v1 永不覆盖）
    reviewer = IncrementalReviewer(memory_store=mem, data_provider=provider)
    v2 = reviewer.re_research_incremental(alert, new_as_of=t2)
    assert v2.dossier.version == 2
    assert v2.parent_dossier_id == "DOS-300750-V1"
    assert len(mem.get_version_chain("300750")) == 2
    assert any(c.claim_id == "CLM-V1-EU-MOAT" for c in v2.dossier.verified_claims)
    assert any("2025H1" in c.text for c in v2.dossier.verified_claims)

    # Step 4: 持续更新专项评测（advisor eval update）
    eval_rep = evaluate_incremental_update(
        old_version=v1,
        new_version=v2,
        expected_affected_ids=["CLM-V1-FIN", "CLM-V1-VAL"],
        expected_unaffected_ids=["CLM-V1-EU-MOAT"],
    )
    assert eval_rep.passed is True
    assert eval_rep.affected_update_recall == 1.0
    assert eval_rep.unaffected_retention_rate == 1.0
    assert eval_rep.version_chain_intact is True
