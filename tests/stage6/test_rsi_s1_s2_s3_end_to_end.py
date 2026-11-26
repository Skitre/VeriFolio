"""S1 经验卡 / S2 路由自校准 / S3 检索进化完整端到端闭环 + 档案继承证明测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.core.enums import ResearchTier
from zhengyan.graph import ResearchRunner
from zhengyan.llm import MockLLMClient
from zhengyan.rsi import (
    AppendOnlyRSILedger,
    BadCaseRecord,
    BadCaseSource,
    CachedSearchDoc,
    CandidateProposal,
    ClaimReviewSample,
    ExperienceCard,
    HistoricalRoutingSnapshot,
    IndustryTermsConfig,
    RSIComponent,
    RSIController,
    RSISixGates,
    RoutingParamsAsset,
    SearchTemplateAsset,
    build_protocol_fingerprint,
    compute_balanced_accuracy,
    evaluate_s1_slice,
    evaluate_s2_replay,
    evaluate_s3_search_replay,
    recall_experience_cards,
    sample_1_bad_to_2_good_stable,
)

ROOT = Path(__file__).resolve().parents[2]
INDUSTRY_PATH = ROOT / "config" / "industry_terms.yaml"
POLICY_PATH = ROOT / "config" / "policies.yaml"
SEARCH_PATH = ROOT / "config" / "search_sources.yaml"


def test_balanced_accuracy_and_1_bad_2_good_stable_sampling() -> None:
    """文档第 22.1 题：1:2 不均衡样本中，全放过普通准确率 66.7%，平衡准确率只有 50%。"""
    bad = [ClaimReviewSample(f"B{i}", "600519", "2025H1", "坏主张", True) for i in range(4)]
    good = [ClaimReviewSample(f"G{i}", "600519", "2025H1", "好主张", False) for i in range(20)]

    s1 = sample_1_bad_to_2_good_stable(bad, good)
    s2 = sample_1_bad_to_2_good_stable(bad, good)
    assert len(s1) == 12  # 4 bad + 8 good
    assert [x.sample_id for x in s1] == [x.sample_id for x in s2]  # 哈希抽样 100% 确定

    # 全部放过：平衡准确率必为 0.50
    y_true = [x.is_bad_claim for x in s1]
    assert compute_balanced_accuracy(y_true, [False] * len(s1)) == 0.5


def test_s1_experience_card_v2_publish_and_inherited_by_600519_v6() -> None:
    """完整复现文档核心故事：
    1. 经验库 v1（空）→ 坏案例（600519 v5 合同负债质询成立）→ 候选 v2（加入 L-007）
    2. 三切片（新失败集 / 固定回放集 / 时间外测试集）各跑 3 次 + 开/关对照，新失败集平衡准确率提升显著且超波动
    3. 通过六道门控 + 人工批准上线 v2
    4. 后续研究 600519 自动召回 L-007 注入 <method_hints>，并在新档案中记录 method_asset_versions['experience_cards'] == 'v2'！
    """
    ledger = AppendOnlyRSILedger()
    ledger.init_component(RSIComponent.S1_EXPERIENCE_CARDS, version="v1", payload={"cards": []})
    gates = RSISixGates(industry_config=IndustryTermsConfig.from_yaml(INDUSTRY_PATH))
    ctrl = RSIController(ledger=ledger, gates=gates)

    card_l007 = ExperienceCard(
        card_id="L-007",
        version="v1",
        tags=["合同负债", "预收款", "打款政策", "消费品"],
        applicable_scope="消费品公司，合同负债或预收款大幅变动时",
        lesson="先排除打款政策、渠道结构调整，再推断需求变化",
        must_check="公告或业绩说明会中关于渠道、打款政策的表述",
        invalidation_condition="公司已明确说明打款政策未变",
        source_ref="深研中被裁定成立的质询（档案 600519 v5）",
    )

    new_fail_samples = [
        ClaimReviewSample("NF-B1", "600519", "2025H1", "合同负债同比下降，直接说明消费品终端需求走弱", True, ("合同负债",)),
        ClaimReviewSample("NF-B2", "000858", "2025H1", "预收款减少直接证明高端白酒消费崩塌", True, ("预收款",)),
        ClaimReviewSample("NF-B3", "600519", "2025H1", "渠道编造无出处毛利率提升", True, ()),
        ClaimReviewSample("NF-G1", "600519", "2025H1", "2025H1 营业收入同比稳步增长，现金回款正常", False),
        ClaimReviewSample("NF-G2", "600519", "2025H1", "本期净现比保持高位，利润含金量充足", False),
        ClaimReviewSample("NF-G3", "000858", "2025H1", "公告披露经销商打款节奏调整影响预收款余额", False),
        ClaimReviewSample("NF-G4", "000858", "2025H1", "毛利率水平在行业内维持领先", False),
        ClaimReviewSample("NF-G5", "600519", "2025H1", "披露日对齐市盈率处于历史中低分位", False),
        ClaimReviewSample("NF-G6", "600519", "2025H1", "经营现金流净额覆盖归母净利润", False),
    ]
    frozen_replay = [
        ClaimReviewSample("FR-B1", "300750", "2024FY", "无财报支撑的虚构产能数字", True, ()),
        ClaimReviewSample("FR-G1", "300750", "2024FY", "归母净利润与营收同口径增长", False),
        ClaimReviewSample("FR-G2", "300750", "2024FY", "海外基地按公告推进建设", False),
    ]
    oot_samples = [
        ClaimReviewSample("OOT-B1", "600809", "2025Q3", "消费品公司预收款骤降直接说明终端需求断崖", True, ("预收款",)),
        ClaimReviewSample("OOT-G1", "600809", "2025Q3", "营业收入与净利润按披露日口径复核无误", False),
        ClaimReviewSample("OOT-G2", "600809", "2025Q3", "净现比表现平稳", False),
    ]
    oot_pairs = {(s.symbol, s.period) for s in oot_samples}

    base_slices = {
        "new_failures": evaluate_s1_slice(slice_name="new_failures", samples=new_fail_samples, cards=[], runs=3),
        "frozen_replay": evaluate_s1_slice(slice_name="frozen_replay", samples=frozen_replay, cards=[], runs=3),
        "out_of_time": evaluate_s1_slice(slice_name="out_of_time", samples=oot_samples, cards=[], runs=3),
    }
    cand_slices = {
        "new_failures": evaluate_s1_slice(slice_name="new_failures", samples=new_fail_samples, cards=[card_l007], runs=3),
        "frozen_replay": evaluate_s1_slice(slice_name="frozen_replay", samples=frozen_replay, cards=[card_l007], runs=3),
        "out_of_time": evaluate_s1_slice(slice_name="out_of_time", samples=oot_samples, cards=[card_l007], runs=3),
    }

    # 开/关对照增益显著
    assert cand_slices["new_failures"].mean_score - base_slices["new_failures"].mean_score >= 0.18
    assert cand_slices["new_failures"].off_mean_score == base_slices["new_failures"].mean_score

    proto = build_protocol_fingerprint(eval_set_id="s1-eval-v1", scorer_version="bal-acc-1.0", reviewer_prompt="fixed-v1")
    proposal = CandidateProposal(
        proposal_id="PROP-S1-V2",
        component=RSIComponent.S1_EXPERIENCE_CARDS,
        round_index=0,
        parent_version="v1",
        candidate_version="v2",
        direction_key="s1:add_contract_liabilities_card",
        change_reason="从茅台深研成立质询沉淀消费品合同负债归因核查经验卡",
        source_bad_cases=[
            BadCaseRecord(
                case_id="BC-MT-CL",
                source_case_ref="600519 v5",
                failure_type=BadCaseSource.UPHELD_CHALLENGE,
                evidence_summary="合同负债下降42%推出需求走弱被反证官推翻",
                hypothesis="我猜是因为起草节点缺少消费品预收款/合同负债变动排查方法提示",
                symbol="600519",
                period="2025H1",
            )
        ],
        baseline_payload={"cards": []},
        candidate_payload={"cards": [card_l007.model_dump()]},
    )

    gate_report = ctrl.submit_and_evaluate(
        proposal=proposal,
        baseline_protocol=proto,
        candidate_protocol=proto,
        oot_pairs=oot_pairs,
        baseline_slices=base_slices,
        candidate_slices=cand_slices,
    )
    assert gate_report.passed is True

    ledger.publish_candidate(
        component=RSIComponent.S1_EXPERIENCE_CARDS,
        candidate_version="v2",
        parent_version="v1",
        change_reason=proposal.change_reason,
        payload=proposal.candidate_payload,
        gate_report=gate_report,
        approver="solo_dev",
        human_approved=True,
        approval_mode="self_approved",
    )
    assert ledger.get_active_version(RSIComponent.S1_EXPERIENCE_CARDS) == "v2"

    # 验证标签召回上限（<=3张）与下一轮真实研究继承 v2！
    assert len(recall_experience_cards("茅台合同负债变动分析", [card_l007] * 5, max_cards=3)) == 3
    llm = MockLLMClient()
    runner = ResearchRunner(
        as_of=date(2025, 8, 20),
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
        llm_client=llm,
        rsi_ledger=ledger,
    )
    out = runner.start("复核贵州茅台 600519 半年报合同负债变动与利润质量", thread_id="th-600519-v6")
    assert out["dossier"].method_asset_versions.get("experience_cards") == "v2"
    assert any("<method_hints" in p and "L-007" in p for p in llm.received_prompts)


def test_s2_routing_replay_calibration_and_s3_search_policy_evolution() -> None:
    ledger = AppendOnlyRSILedger()
    ledger.init_component(
        RSIComponent.S2_ROUTING_PARAMS,
        version="v1",
        payload={"temperature": 1.0, "abstain_threshold": 0.55},
    )
    ledger.init_component(
        RSIComponent.S3_SEARCH_TEMPLATES,
        version="v1",
        payload={"whitelist_domains": ["stcn.com", "caixin.com"], "relevance_threshold": 0.50},
    )
    gates = RSISixGates(industry_config=IndustryTermsConfig.from_yaml(INDUSTRY_PATH))
    ctrl = RSIController(ledger=ledger, gates=gates)

    # ===== S2 路由自校准：在 3 次历史原始概率快照上回放 T=1.0 vs T=0.75 =====
    # 偏保守概率：standard=0.52, deep=0.33, quick=0.15 -> T=1.0 时 <0.55 被误拒判升成 deep；
    # T=0.75 拉尖后 standard=0.63 >= 0.55，准确命中 standard！
    snap_probs = [
        {"quick": 0.15, "standard": 0.52, "deep": 0.33},
        {"quick": 0.14, "standard": 0.53, "deep": 0.33},
        {"quick": 0.08, "standard": 0.22, "deep": 0.70},
        {"quick": 0.72, "standard": 0.18, "deep": 0.10},
    ]
    golds = [ResearchTier.STANDARD, ResearchTier.STANDARD, ResearchTier.DEEP, ResearchTier.QUICK]
    snaps = [HistoricalRoutingSnapshot(f"run-{i}", snap_probs, golds) for i in range(3)]

    p_base = RoutingParamsAsset(temperature=1.0, abstain_threshold=0.55)
    p_cand = RoutingParamsAsset(temperature=0.75, abstain_threshold=0.55)
    s2_base = {
        "new_failures": evaluate_s2_replay(slice_name="new_failures", snapshots=snaps, params=p_base),
        "frozen_replay": evaluate_s2_replay(slice_name="frozen_replay", snapshots=snaps, params=p_base),
        "out_of_time": evaluate_s2_replay(slice_name="out_of_time", snapshots=snaps, params=p_base),
    }
    s2_cand = {
        "new_failures": evaluate_s2_replay(slice_name="new_failures", snapshots=snaps, params=p_cand),
        "frozen_replay": evaluate_s2_replay(slice_name="frozen_replay", snapshots=snaps, params=p_cand),
        "out_of_time": evaluate_s2_replay(slice_name="out_of_time", snapshots=snaps, params=p_cand),
    }
    assert s2_cand["new_failures"].mean_score == 1.0
    assert s2_base["new_failures"].mean_score == 0.5

    proto_s2 = build_protocol_fingerprint(eval_set_id="s2-24", scorer_version="acc-1", reviewer_prompt="none")
    prop_s2 = CandidateProposal(
        proposal_id="PROP-S2-V2",
        component=RSIComponent.S2_ROUTING_PARAMS,
        round_index=0,
        parent_version="v1",
        candidate_version="v2",
        direction_key="s2:temp_1.0_to_0.75",
        change_reason="通过离线历史原始概率回放将温度从 1.0 校准至 0.75 降低误拒判",
        source_bad_cases=[
            BadCaseRecord(
                case_id="BC-RT-15",
                source_case_ref="路由金标 RT-15",
                failure_type=BadCaseSource.ROUTING_ERROR,
                evidence_summary="偏保守判别器将两道标准研判题误拒判升级到深研",
                hypothesis="我猜是因为判别器概率分布偏平导致最高置信度跌破 0.55 拒判阈值",
                target_component=RSIComponent.S2_ROUTING_PARAMS,
            )
        ],
        baseline_payload=p_base.model_dump(),
        candidate_payload=p_cand.model_dump(),
    )
    rep_s2 = ctrl.submit_and_evaluate(
        proposal=prop_s2, baseline_protocol=proto_s2, candidate_protocol=proto_s2,
        oot_pairs=set(), baseline_slices=s2_base, candidate_slices=s2_cand,
    )
    assert rep_s2.passed is True

    # ===== S3 检索策略进化：Precision × Recall 确定性回放 + 白名单安全新增 1 站 =====
    docs = [
        CachedSearchDoc("D1", "stcn.com", 0.85, True, True),
        CachedSearchDoc("D2", "cnstock.com", 0.82, True, True),  # 权威财经站，原白名单漏掉导致召回不足
        CachedSearchDoc("D3", "spam.example.com", 0.45, False, False),
    ]
    t_base = SearchTemplateAsset(whitelist_domains=["stcn.com", "caixin.com"], relevance_threshold=0.50)
    t_cand = SearchTemplateAsset(whitelist_domains=["stcn.com", "caixin.com", "cnstock.com"], relevance_threshold=0.50)
    s3_base = {
        "new_failures": evaluate_s3_search_replay(slice_name="new_failures", cached_docs=docs, template=t_base),
        "frozen_replay": evaluate_s3_search_replay(slice_name="frozen_replay", cached_docs=docs, template=t_base),
        "out_of_time": evaluate_s3_search_replay(slice_name="out_of_time", cached_docs=docs, template=t_base),
    }
    s3_cand = {
        "new_failures": evaluate_s3_search_replay(slice_name="new_failures", cached_docs=docs, template=t_cand),
        "frozen_replay": evaluate_s3_search_replay(slice_name="frozen_replay", cached_docs=docs, template=t_cand),
        "out_of_time": evaluate_s3_search_replay(slice_name="out_of_time", cached_docs=docs, template=t_cand),
    }
    # 加入上海证券报 cnstock.com 后 Precision×Recall 从 0.25 飙升至 1.00，且 3 次确定性一致
    assert s3_cand["new_failures"].mean_score > s3_base["new_failures"].mean_score
    assert s3_cand["new_failures"].spread == 0.0

    proto_s3 = build_protocol_fingerprint(eval_set_id="s3-cache-v1", scorer_version="pr-1", reviewer_prompt="none")
    prop_s3 = CandidateProposal(
        proposal_id="PROP-S3-V2",
        component=RSIComponent.S3_SEARCH_TEMPLATES,
        round_index=0,
        parent_version="v1",
        candidate_version="v2",
        direction_key="s3:add_cnstock",
        change_reason="在检索抓取白名单中新增权威财经媒体 cnstock.com 提升 B 级证据召回",
        source_bad_cases=[
            BadCaseRecord(
                case_id="BC-SRCH-1",
                source_case_ref="搜索日志 SRCH-20250814",
                failure_type=BadCaseSource.LOW_QUALITY_SEARCH,
                evidence_summary="上海证券报深度报道因未进白名单降为 C 级，导致可用证据覆盖偏低",
                hypothesis="我猜是因为抓取白名单遗漏了上证报 cnstock.com 权威财经站点",
                target_component=RSIComponent.S3_SEARCH_TEMPLATES,
            )
        ],
        baseline_payload=t_base.model_dump(),
        candidate_payload=t_cand.model_dump(),
    )
    rep_s3 = ctrl.submit_and_evaluate(
        proposal=prop_s3, baseline_protocol=proto_s3, candidate_protocol=proto_s3,
        oot_pairs=set(), baseline_slices=s3_base, candidate_slices=s3_cand,
    )
    assert rep_s3.passed is True
    ledger.publish_candidate(
        component=RSIComponent.S3_SEARCH_TEMPLATES,
        candidate_version="v2",
        parent_version="v1",
        change_reason=prop_s3.change_reason,
        payload=prop_s3.candidate_payload,
        gate_report=rep_s3,
        approver="solo_dev",
        human_approved=True,
        approval_mode="self_approved",
        approval_note="核验上海证券报官方域名资质无误，批准加入白名单 v2",
    )
    assert ledger.get_active_version(RSIComponent.S3_SEARCH_TEMPLATES) == "v2"

    # 连续失败停滞后自动推荐切换组件
    assert ctrl.recommend_next_component(RSIComponent.S1_EXPERIENCE_CARDS) == RSIComponent.S1_EXPERIENCE_CARDS
