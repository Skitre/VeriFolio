"""纯规则核验器 8 步全路径测试（对应文档第 7 题：C 级线索进入结论 0 次）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.core.enums import ClaimType, EvidenceGrade, VerificationStatus
from zhengyan.core.models import Claim, EvidenceCard, RawWebResult
from zhengyan.data import MockMarketDataProvider, compute_yoy_growth
from zhengyan.search import SourcePolicy, WebSearchPipeline
from zhengyan.verify import RuleVerifier

POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "search_sources.yaml"


def _build_fixture_store(as_of: date) -> dict[str, EvidenceCard]:
    provider = MockMarketDataProvider()
    a_cards = provider.to_evidence_cards("600519", as_of=as_of)
    store = {c.evidence_id: c for c in a_cards}

    # 派生计算卡：营收同比(+9.10%) 与 合同负债同比(-42.00%)
    rev_yoy = compute_yoy_growth(
        store["EV-FIN-600519-2025H1"],
        store["EV-FIN-600519-2024H1"],
        "revenue",
        as_of=as_of,
    )
    cl_yoy = compute_yoy_growth(
        store["EV-FIN-600519-2025H1"],
        store["EV-FIN-600519-2024H1"],
        "contract_liabilities",
        as_of=as_of,
    )
    store[rev_yoy.evidence_id] = rev_yoy
    store[cl_yoy.evidence_id] = cl_yoy

    # 网页证据卡（2 条同源 B 级 + 1 条 C 级）
    pipeline = WebSearchPipeline(SourcePolicy.from_yaml(POLICY_PATH))
    web_cards = pipeline.process(
        [
            RawWebResult(
                url="https://www.stcn.com/a/1.html",
                title="茅台渠道调研显示打款节奏调整 | 证券时报网",
                published_at=date(2025, 8, 14),
                fetched=True,
                raw_html="<p>来源：证券时报。渠道调研称公司主动调整打款节奏，媒体估算渠道库存同比上升 15%。</p>",
            ),
            RawWebResult(
                url="https://finance.sina.com.cn/a/2.shtml",
                title="茅台渠道调研显示打款节奏调整_新浪财经",
                published_at=date(2025, 8, 14),
                fetched=True,
                raw_html="<p>来源：证券时报。渠道调研称公司主动调整打款节奏，媒体估算渠道库存同比上升 15%。</p>",
            ),
            RawWebResult(
                url="https://www.yicai.com/c-only.html",
                title="市场传闻某白酒提价",
                snippet="传闻今年出厂价可能上调 10%",
                published_at=date(2025, 8, 16),
                fetched=False,  # 无全文快照 -> C 级
            ),
        ],
        as_of=as_of,
        symbol="600519",
    )
    for wc in web_cards:
        store[wc.evidence_id] = wc

    # 额外放一张未来时点财报卡（2025Q3，披露日 2025-10-29），测试核验器第 4 步
    future_cards = provider.to_evidence_cards("600519", as_of=date(2025, 11, 5))
    q3_card = next(c for c in future_cards if c.evidence_id == "EV-FIN-600519-2025Q3")
    store[q3_card.evidence_id] = q3_card

    return store


def test_step1_and_step3_nonexistent_ids_rejected() -> None:
    as_of = date(2025, 8, 20)
    store = _build_fixture_store(as_of)
    verifier = RuleVerifier()

    claim = Claim(claim_id="C1", text="贵州茅台业绩稳健", evidence_ids=["EV-FAKE-999"])
    res = verifier.verify_claim(claim, store, as_of=as_of)
    assert res.status == VerificationStatus.REJECTED
    assert res.can_enter_conclusion is False
    assert res.ignored_evidence_ids == ["EV-FAKE-999"]


def test_step2_hypothesis_and_scenario_pass_as_non_fact() -> None:
    as_of = date(2025, 8, 20)
    store = _build_fixture_store(as_of)
    verifier = RuleVerifier()

    for ctype in (ClaimType.HYPOTHESIS, ClaimType.SCENARIO):
        claim = Claim(
            claim_id=f"C-{ctype}",
            text="假设明年营收增速回升至 15%",
            claim_type=ctype,
            evidence_ids=[],
        )
        res = verifier.verify_claim(claim, store, as_of=as_of)
        assert res.status == VerificationStatus.VERIFIED
        assert res.is_fact is False


def test_step4_future_evidence_rejected() -> None:
    as_of = date(2025, 8, 20)
    store = _build_fixture_store(as_of)
    verifier = RuleVerifier()

    claim = Claim(
        claim_id="C-FUTURE",
        text="2025年前三季度营收达到 1284.54 亿元",
        evidence_ids=["EV-FIN-600519-2025Q3"],
    )
    res = verifier.verify_claim(claim, store, as_of=as_of)
    assert res.status == VerificationStatus.REJECTED
    assert any("晚于研究时点" in r for r in res.reasons)


def test_step5_c_grade_only_never_enters_conclusion() -> None:
    """核心指标：C 级线索进入结论 0 次。"""
    as_of = date(2025, 8, 20)
    store = _build_fixture_store(as_of)
    c_card = next(c for c in store.values() if c.grade == EvidenceGrade.C)
    verifier = RuleVerifier()

    claim = Claim(
        claim_id="C-ONLY-C",
        text="市场传闻公司年内可能上调出厂价",
        evidence_ids=[c_card.evidence_id],
    )
    res = verifier.verify_claim(claim, store, as_of=as_of)
    assert res.status == VerificationStatus.REJECTED
    assert res.can_enter_conclusion is False


def test_step6_a_grade_numbers_half_ulp_derived_from_and_unit_conversion() -> None:
    as_of = date(2025, 8, 20)
    store = _build_fixture_store(as_of)
    verifier = RuleVerifier()

    # 1. 只引用计算卡，核验器会递归纳入 derived_from 的财报卡；
    #    893.89亿元、9.1%（半单位容差±0.05 匹配 9.10%）、下降 42%（匹配 -42.00%）均应通过
    claim_ok = Claim(
        claim_id="C-OK",
        text="贵州茅台 600519 在 2025H1 实现营业收入 893.89 亿元，同比增长 9.1%，合同负债同比下降 42%。",
        evidence_ids=[
            "EV-CMP-YOY-600519-revenue-2025H1",
            "EV-CMP-YOY-600519-contract_liabilities-2025H1",
        ],
    )
    res_ok = verifier.verify_claim(claim_ok, store, as_of=as_of)
    assert res_ok.status == VerificationStatus.VERIFIED
    assert len(res_ok.number_checks) == 3
    assert all(nc.matched and nc.matched_grade == EvidenceGrade.A for nc in res_ok.number_checks)

    # 2. 万元 ↔ 亿元自动精确换算（8938900 万元 == 893.89 亿元）
    claim_unit = Claim(
        claim_id="C-UNIT",
        text="2025H1 营业收入为 8938900 万元。",
        evidence_ids=["EV-FIN-600519-2025H1"],
    )
    assert verifier.verify_claim(claim_unit, store, as_of=as_of).status == VerificationStatus.VERIFIED

    # 3. 符号反转幻觉拦截：把合同负债“下降 42%”误写成“增长 42%”必须存疑！
    claim_sign_flip = Claim(
        claim_id="C-SIGN-FLIP",
        text="2025H1 合同负债同比增长 42%。",
        evidence_ids=["EV-CMP-YOY-600519-contract_liabilities-2025H1"],
    )
    res_flip = verifier.verify_claim(claim_sign_flip, store, as_of=as_of)
    assert res_flip.status == VerificationStatus.DOUBTFUL

    # 4. 超出书写精度半单位容差（写 9.2%，容差 ±0.05，区间 [9.15, 9.25] 不含 9.10）
    claim_precision = Claim(
        claim_id="C-PREC",
        text="2025H1 营收同比增长 9.2%。",
        evidence_ids=["EV-CMP-YOY-600519-revenue-2025H1"],
    )
    assert verifier.verify_claim(claim_precision, store, as_of=as_of).status == VerificationStatus.DOUBTFUL


def test_step6_percentage_point_diff_and_hallucination_intercept() -> None:
    """文档经典案例：毛利率 2024H1=91.80%，2025H1=91.30%，同比下降 0.5 个百分点。
    模型幻觉编造“毛利率同比提升 3 个百分点”必须被拦下标存疑！
    """
    as_of = date(2025, 8, 20)
    store = _build_fixture_store(as_of)
    verifier = RuleVerifier()

    claim_true = Claim(
        claim_id="C-PP-OK",
        text="2025H1 毛利率为 91.3%，同比下降 0.5 个百分点。",
        evidence_ids=["EV-FIN-600519-2025H1", "EV-FIN-600519-2024H1"],
    )
    res_true = verifier.verify_claim(claim_true, store, as_of=as_of)
    assert res_true.status == VerificationStatus.VERIFIED

    claim_hallucinated = Claim(
        claim_id="C-PP-FAKE",
        text="2025H1 毛利率同比提升 3 个百分点。",
        evidence_ids=["EV-FIN-600519-2025H1", "EV-FIN-600519-2024H1"],
    )
    res_fake = verifier.verify_claim(claim_hallucinated, store, as_of=as_of)
    assert res_fake.status == VerificationStatus.DOUBTFUL
    assert any("无法由引用证据复算" in r for r in res_fake.reasons)


def test_step7_b_grade_number_doubtful_and_step8_single_source_merge() -> None:
    """数字只出现在 B 级网页 → 存疑（须回到财报复核）；
    引用同簇转载的多条 B 级网页 → 标注按单一来源计。
    """
    as_of = date(2025, 8, 20)
    store = _build_fixture_store(as_of)
    b_ids = [c.evidence_id for c in store.values() if c.grade == EvidenceGrade.B]
    assert len(b_ids) == 2
    verifier = RuleVerifier()

    # 含数字且仅在 B 级网页出现 → DOUBTFUL + 同源合并标注
    claim_b_num = Claim(
        claim_id="C-B-NUM",
        text="渠道库存同比上升 15%。",
        evidence_ids=b_ids,
    )
    res_b_num = verifier.verify_claim(claim_b_num, store, as_of=as_of)
    assert res_b_num.status == VerificationStatus.DOUBTFUL
    assert any("须回到财报复核" in r for r in res_b_num.reasons)
    assert res_b_num.single_source_merged is True

    # 不含数字、仅用 B 级说明“发生了什么” → VERIFIED + 同源合并标注
    claim_b_event = Claim(
        claim_id="C-B-EVENT",
        text="多家渠道调研反映公司近期主动调整经销商打款节奏。",
        evidence_ids=b_ids,
    )
    res_b_event = verifier.verify_claim(claim_b_event, store, as_of=as_of)
    assert res_b_event.status == VerificationStatus.VERIFIED
    assert res_b_event.single_source_merged is True
    assert any("按单一来源计" in n for n in res_b_event.notes)
