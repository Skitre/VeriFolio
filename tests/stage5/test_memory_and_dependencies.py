"""五类记忆写入权限隔离、只追加版本链、精确依赖查询与 30 天工作记忆清理测试。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from zhengyan.core.enums import (
    DossierStatus,
    EvidenceGrade,
    EvidenceSourceType,
    ResearchTier,
    UserIntent,
    VerificationStatus,
)
from zhengyan.core.models import Claim, DossierRecord, EvidenceCard, VerificationResult
from zhengyan.data import MockMarketDataProvider, compute_yoy_growth
from zhengyan.memory import (
    FiveLayerMemoryStore,
    MemoryPermissionError,
    UserConfirmedConstraints,
)


def test_five_memory_write_permission_boundaries() -> None:
    store = FiveLayerMemoryStore()

    # 1. 已核验事实：仅 VERIFIED 且 is_fact=True 可写入；存疑/驳回/非事实一律拒绝
    claim = Claim(claim_id="C1", text="营收 893.89 亿元", evidence_ids=["E1"])
    with pytest.raises(MemoryPermissionError):
        store.record_verified_fact(
            symbol="600519",
            claim=claim,
            verification=VerificationResult(claim_id="C1", status=VerificationStatus.DOUBTFUL),
            as_of=date(2025, 8, 20),
        )
    fact = store.record_verified_fact(
        symbol="600519",
        claim=claim,
        verification=VerificationResult(claim_id="C1", status=VerificationStatus.VERIFIED, valid_evidence_ids=["E1"]),
        as_of=date(2025, 8, 20),
        period="2025H1",
    )
    assert fact.period == "2025H1"

    # 2. 用户约束：研究流程（caller="research_graph"）绝对无权写入，必须用户本人显式勾选确认
    c = UserConfirmedConstraints(
        user_id="u1",
        buy_thesis="长期持有高现金流消费龙头",
        horizon_months=36,
        max_drawdown_pct=Decimal("20"),
    )
    with pytest.raises(MemoryPermissionError):
        store.save_user_constraints(c, caller="research_graph", explicit_user_confirmation=True)
    with pytest.raises(MemoryPermissionError):
        store.save_user_constraints(c, caller="user", explicit_user_confirmation=False)

    saved_c = store.save_user_constraints(c, caller="user", explicit_user_confirmation=True)
    assert saved_c.version == 1

    # 3. 方法经验：必须经 RSI 门控通过 + 人工批准才可写入
    with pytest.raises(MemoryPermissionError):
        store.publish_method_asset(
            asset_type="experience_cards",
            version="v2",
            payload={"id": "L-007"},
            caller="optimizer",
            gate_passed=True,
            human_approved=True,
        )
    with pytest.raises(MemoryPermissionError):
        store.publish_method_asset(
            asset_type="experience_cards",
            version="v2",
            payload={"id": "L-007"},
            caller="rsi_gate",
            gate_passed=True,
            human_approved=False,
        )
    store.publish_method_asset(
        asset_type="experience_cards",
        version="v2",
        payload={"id": "L-007"},
        caller="rsi_gate",
        gate_passed=True,
        human_approved=True,
    )


def test_append_only_dossier_chain_and_exact_period_dependency_query() -> None:
    """精确回答“哪些结论依赖了 2025H1 财报？”——递归展开计算卡 derived_from，且禁止覆盖旧版本。"""
    as_of = date(2025, 8, 20)
    provider = MockMarketDataProvider()
    ev_store = {c.evidence_id: c for c in provider.to_evidence_cards("600519", as_of)}
    yoy = compute_yoy_growth(
        ev_store["EV-FIN-600519-2025H1"],
        ev_store["EV-FIN-600519-2024H1"],
        "revenue",
        as_of=as_of,
    )
    ev_store[yoy.evidence_id] = yoy
    web_card = EvidenceCard(
        evidence_id="EV-WEB-MOAT",
        grade=EvidenceGrade.B,
        source_type=EvidenceSourceType.WEB,
        source_name="财新网",
        title="品牌护城河分析",
        published_at=date(2025, 8, 10),
        authority=4,
        has_full_snapshot=True,
    )
    ev_store[web_card.evidence_id] = web_card

    d1 = DossierRecord(
        dossier_id="DOS-MT-V1",
        symbol="600519",
        as_of=as_of,
        question="复核茅台 2025H1",
        intent=UserIntent.RESEARCH,
        tier=ResearchTier.STANDARD,
        status=DossierStatus.COMPLETE,
        verified_claims=[
            Claim(claim_id="CLM-REV-YOY", text="2025H1 营收同比增长 9.10%", evidence_ids=[yoy.evidence_id]),
            Claim(claim_id="CLM-MOAT", text="高端品牌壁垒长期稳固", evidence_ids=[web_card.evidence_id]),
        ],
    )

    mem = FiveLayerMemoryStore()
    v1 = mem.append_dossier(d1, evidence_store=ev_store)
    assert v1.dossier.version == 1

    # 禁止覆盖已有 dossier_id
    with pytest.raises(MemoryPermissionError, match="禁止覆盖"):
        mem.append_dossier(d1, evidence_store=ev_store)

    # 精确依赖查询：CLM-REV-YOY 通过计算卡递归依赖 2025H1 和 2024H1，而 CLM-MOAT 不依赖财报！
    assert mem.find_claims_depending_on_period("DOS-MT-V1", "2025H1") == ["CLM-REV-YOY"]
    assert mem.find_claims_depending_on_period("DOS-MT-V1", "2024H1") == ["CLM-REV-YOY"]


def test_working_memory_30_day_ttl_auto_purge() -> None:
    mem = FiveLayerMemoryStore()
    t0 = datetime(2025, 8, 1, 12, 0, tzinfo=timezone.utc)
    mem.put_working_memory("task-old", {"draft": "old"}, created_at=t0 - timedelta(days=31))
    mem.put_working_memory("task-fresh", {"draft": "fresh"}, created_at=t0 - timedelta(days=5))

    removed = mem.purge_expired_working_memory(now=t0)
    assert removed == 1
    assert mem.working_memory_count() == 1
