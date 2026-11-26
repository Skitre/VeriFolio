"""六道门控逐一攻防单测（协议一致、防泄漏、防过拟合+行业词表、噪声门控、成本门控、回归门控）。"""

from __future__ import annotations

from pathlib import Path

from zhengyan.rsi import (
    BadCaseRecord,
    BadCaseSource,
    CandidateProposal,
    ExperienceCard,
    IndustryTermsConfig,
    RSIComponent,
    RSISixGates,
    SliceEvalStats,
    build_protocol_fingerprint,
)

INDUSTRY_TERMS_PATH = Path(__file__).resolve().parents[2] / "config" / "industry_terms.yaml"


def _make_gates() -> RSISixGates:
    return RSISixGates(industry_config=IndustryTermsConfig.from_yaml(INDUSTRY_TERMS_PATH))


def _good_bad_case(symbol: str = "600519", period: str = "2025H1") -> BadCaseRecord:
    return BadCaseRecord(
        case_id="BC-01",
        source_case_ref=f"{symbol} v5",
        failure_type=BadCaseSource.UPHELD_CHALLENGE,
        evidence_summary="合同负债下降被直接归因为需求走弱，被反证官以打款政策推翻",
        hypothesis="我猜是因为起草前缺少消费品合同负债变动排查方法提示",
        symbol=symbol,
        period=period,
    )


def _good_card() -> dict:
    return ExperienceCard(
        card_id="L-007",
        version="v1",
        tags=["合同负债", "预收款", "需求"],
        applicable_scope="消费品公司，合同负债或预收款大幅变动时",
        lesson="先排除打款政策、渠道结构调整，再推断需求变化",
        must_check="公告或业绩说明会中关于渠道、打款政策的表述",
        invalidation_condition="公司已明确说明打款政策未变",
        source_ref="深研中被裁定成立的质询（档案 600519 v5）",
    ).model_dump()


def _clean_slices(
    *,
    cand_new_scores: list[float] | None = None,
    base_new_scores: list[float] | None = None,
    base_tokens: float = 2400.0,
    cand_tokens: float = 2450.0,
    cand_oot_scores: list[float] | None = None,
) -> tuple[dict[str, SliceEvalStats], dict[str, SliceEvalStats]]:
    b_new = base_new_scores or [0.60, 0.62, 0.61]
    c_new = cand_new_scores or [0.80, 0.81, 0.79]
    c_oot = cand_oot_scores or [0.85, 0.85, 0.85]
    base = {
        "new_failures": SliceEvalStats(
            slice_name="new_failures", run_scores=b_new,
            mean_score=sum(b_new) / len(b_new), spread=max(b_new) - min(b_new), mean_tokens=base_tokens,
        ),
        "frozen_replay": SliceEvalStats(
            slice_name="frozen_replay", run_scores=[0.85, 0.85, 0.85],
            mean_score=0.85, spread=0.0, mean_tokens=base_tokens,
        ),
        "out_of_time": SliceEvalStats(
            slice_name="out_of_time", run_scores=[0.85, 0.85, 0.85],
            mean_score=0.85, spread=0.0, mean_tokens=base_tokens,
        ),
    }
    cand = {
        "new_failures": SliceEvalStats(
            slice_name="new_failures", run_scores=c_new,
            mean_score=sum(c_new) / len(c_new), spread=max(c_new) - min(c_new), mean_tokens=cand_tokens,
        ),
        "frozen_replay": SliceEvalStats(
            slice_name="frozen_replay", run_scores=[0.86, 0.86, 0.86],
            mean_score=0.86, spread=0.0, mean_tokens=cand_tokens,
        ),
        "out_of_time": SliceEvalStats(
            slice_name="out_of_time", run_scores=c_oot,
            mean_score=sum(c_oot) / len(c_oot), spread=max(c_oot) - min(c_oot), mean_tokens=cand_tokens,
        ),
    }
    return base, cand


def _base_proposal(cand_payload: dict | None = None, bad_case: BadCaseRecord | None = None) -> CandidateProposal:
    return CandidateProposal(
        proposal_id="PROP-01",
        component=RSIComponent.S1_EXPERIENCE_CARDS,
        round_index=0,
        parent_version="v1",
        candidate_version="v2",
        direction_key="s1:add_L007",
        change_reason="从成立质询中沉淀消费品合同负债变动归因排查经验卡",
        source_bad_cases=[bad_case or _good_bad_case()],
        baseline_payload={"cards": []},
        candidate_payload=cand_payload if cand_payload is not None else {"cards": [_good_card()]},
    )


def test_all_six_gates_pass_on_valid_candidate() -> None:
    gates = _make_gates()
    proto = build_protocol_fingerprint(eval_set_id="s1-v1", scorer_version="bal-acc-1", reviewer_prompt="p-v1")
    base_s, cand_s = _clean_slices()
    rep = gates.evaluate(
        proposal=_base_proposal(),
        baseline_protocol=proto,
        candidate_protocol=proto,
        oot_pairs={("300750", "2025Q3")},
        baseline_slices=base_s,
        candidate_slices=cand_s,
    )
    assert rep.passed is True
    assert len(rep.checks) == 6
    assert rep.failed_gates == []


def test_gate1_blocks_protocol_hash_mismatch() -> None:
    gates = _make_gates()
    p_base = build_protocol_fingerprint(eval_set_id="s1-v1", scorer_version="bal-acc-1", reviewer_prompt="prompt-v1")
    p_changed = build_protocol_fingerprint(eval_set_id="s1-v1", scorer_version="bal-acc-1", reviewer_prompt="prompt-v2-偷换提示词")
    base_s, cand_s = _clean_slices()
    rep = gates.evaluate(
        proposal=_base_proposal(),
        baseline_protocol=p_base,
        candidate_protocol=p_changed,
        oot_pairs=set(),
        baseline_slices=base_s,
        candidate_slices=cand_s,
    )
    assert rep.passed is False
    assert "protocol_consistency" in rep.failed_gates


def test_gate2_blocks_oot_data_leakage() -> None:
    gates = _make_gates()
    proto = build_protocol_fingerprint(eval_set_id="s1-v1", scorer_version="bal-acc-1", reviewer_prompt="p-v1")
    base_s, cand_s = _clean_slices()
    # 坏案例来自 600519/2025H1，恰好混入了时间外测试集 OOT → 防泄漏门控拦截！
    rep = gates.evaluate(
        proposal=_base_proposal(bad_case=_good_bad_case("600519", "2025H1")),
        baseline_protocol=proto,
        candidate_protocol=proto,
        oot_pairs={("600519", "2025H1")},
        baseline_slices=base_s,
        candidate_slices=cand_s,
    )
    assert rep.passed is False
    assert "anti_leakage" in rep.failed_gates


def test_gate3_blocks_s1_overfitting_company_number_advice_and_industry_term() -> None:
    """文档第 21 & 27 题：背具体股票名/数字/买卖结论，或使用“基酒”等行业词却不标明行业 → 全部拦截！"""
    gates = _make_gates()
    proto = build_protocol_fingerprint(eval_set_id="s1-v1", scorer_version="bal-acc-1", reviewer_prompt="p-v1")
    base_s, cand_s = _clean_slices()

    # Case A: 照抄文档第 27 题反面教材“茅台合同负债下降 42% 不代表需求走弱”
    overfit_card = ExperienceCard(
        card_id="L-BAD-1",
        tags=["合同负债"],
        applicable_scope="消费品公司",
        lesson="茅台合同负债下降 42% 不代表需求走弱，建议买入",
        must_check="查公告",
        invalidation_condition="无",
        source_ref="600519 v5",
    ).model_dump()
    rep_a = gates.evaluate(
        proposal=_base_proposal(cand_payload={"cards": [overfit_card]}),
        baseline_protocol=proto, candidate_protocol=proto, oot_pairs=set(),
        baseline_slices=base_s, candidate_slices=cand_s,
    )
    assert "anti_overfitting" in rep_a.failed_gates
    detail_a = next(c.detail for c in rep_a.checks if c.gate_name == "anti_overfitting")
    assert "茅台" in detail_a and "42%" in detail_a and "建议买入" in detail_a

    # Case B: 出现行业专用词“基酒”但适用范围写“全市场通用”而未标明白酒/消费品 → 拦截
    ind_leak_card = ExperienceCard(
        card_id="L-BAD-2",
        tags=["存货"],
        applicable_scope="全市场上市公司通用",
        lesson="评估存货周转时须关注基酒产能与陈化周期",
        must_check="查产能公告",
        invalidation_condition="非存货密集型业务",
        source_ref="600519 v5",
    ).model_dump()
    rep_b = gates.evaluate(
        proposal=_base_proposal(cand_payload={"cards": [ind_leak_card]}),
        baseline_protocol=proto, candidate_protocol=proto, oot_pairs=set(),
        baseline_slices=base_s, candidate_slices=cand_s,
    )
    assert "anti_overfitting" in rep_b.failed_gates
    assert "基酒" in next(c.detail for c in rep_b.checks if c.gate_name == "anti_overfitting")


def test_gate3_blocks_s2_step_overflow_and_s3_unsafe_whitelist_edits() -> None:
    gates = _make_gates()
    proto = build_protocol_fingerprint(eval_set_id="x", scorer_version="1", reviewer_prompt="1")
    base_s, cand_s = _clean_slices()

    # S2 第 3 轮步长上限为 0.30 * 0.7^3 = 0.1029，改动温度 0.40 超步长 → 拦截
    p_s2 = CandidateProposal(
        proposal_id="P-S2",
        component=RSIComponent.S2_ROUTING_PARAMS,
        round_index=3,
        parent_version="v1",
        candidate_version="v2",
        direction_key="s2:temp_jump",
        change_reason="一次性大幅拉低温度参数以压缩拒判率",
        source_bad_cases=[_good_bad_case()],
        baseline_payload={"temperature": 1.0, "abstain_threshold": 0.55},
        candidate_payload={"temperature": 0.60, "abstain_threshold": 0.55},
    )
    rep_s2 = gates.evaluate(
        proposal=p_s2, baseline_protocol=proto, candidate_protocol=proto,
        oot_pairs=set(), baseline_slices=base_s, candidate_slices=cand_s,
    )
    assert "anti_overfitting" in rep_s2.failed_gates

    # S3 试图把雪球自媒体加进白名单、或一次加 2 个站点、或删除已有白名单站点 → 全拦截
    for bad_wl in (
        ["stcn.com", "xueqiu.com"],              # 禁自媒体/黑名单
        ["stcn.com", "caixin.com", "yicai.com"], # 一次加 2 个
        [],                                      # 禁删站
    ):
        p_s3 = CandidateProposal(
            proposal_id="P-S3",
            component=RSIComponent.S3_SEARCH_TEMPLATES,
            round_index=0,
            parent_version="v1",
            candidate_version="v2",
            direction_key="s3:wl_test",
            change_reason="调整权威财经站点白名单以提升可用覆盖",
            source_bad_cases=[_good_bad_case()],
            baseline_payload={"whitelist_domains": ["stcn.com"], "relevance_threshold": 0.5},
            candidate_payload={"whitelist_domains": bad_wl, "relevance_threshold": 0.5},
        )
        rep_s3 = gates.evaluate(
            proposal=p_s3, baseline_protocol=proto, candidate_protocol=proto,
            oot_pairs=set(), baseline_slices=base_s, candidate_slices=cand_s,
        )
        assert "anti_overfitting" in rep_s3.failed_gates


def test_gate4_noise_gate_gate5_cost_gate_and_gate6_regression_gate() -> None:
    gates = _make_gates()
    proto = build_protocol_fingerprint(eval_set_id="s1-v1", scorer_version="bal-acc-1", reviewer_prompt="p-v1")

    # 1. 噪声门控：均值只涨了 +0.02，但基线自身波动 spread = 0.083（模拟 92%→83% 波动）→ 拦截！
    b_noisy, c_noisy = _clean_slices(base_new_scores=[0.917, 0.833, 0.875], cand_new_scores=[0.90, 0.89, 0.90])
    rep_noise = gates.evaluate(
        proposal=_base_proposal(), baseline_protocol=proto, candidate_protocol=proto,
        oot_pairs=set(), baseline_slices=b_noisy, candidate_slices=c_noisy,
    )
    assert "noise_gate" in rep_noise.failed_gates

    # 2. 成本门控：Token 暴涨 +80%，质量只相对提升 +30% → 拦截！
    b_cost, c_cost = _clean_slices(base_tokens=2000.0, cand_tokens=3600.0)
    rep_cost = gates.evaluate(
        proposal=_base_proposal(), baseline_protocol=proto, candidate_protocol=proto,
        oot_pairs=set(), baseline_slices=b_cost, candidate_slices=c_cost,
    )
    assert "cost_gate" in rep_cost.failed_gates

    # 3. 回归门控：新失败集大涨，但时间外测试集（OOT）从 0.85 退化到 0.78 → 拦截！
    b_reg, c_reg = _clean_slices(cand_oot_scores=[0.78, 0.78, 0.78])
    rep_reg = gates.evaluate(
        proposal=_base_proposal(), baseline_protocol=proto, candidate_protocol=proto,
        oot_pairs=set(), baseline_slices=b_reg, candidate_slices=c_reg,
    )
    assert "regression_gate" in rep_reg.failed_gates
