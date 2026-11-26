"""Stage 6 压轴演示：过拟合卡片被拦 → 泛化 L-007 过六道门控发布 v2 → 600519 v6 继承 → S3 进化 → 一键回滚。

运行：uv run python examples/stage6_demo.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.graph import ResearchRunner
from zhengyan.llm import MockLLMClient
from zhengyan.rsi import (
    AppendOnlyRSILedger,
    BadCaseRecord,
    BadCaseSource,
    CandidateProposal,
    ClaimReviewSample,
    ExperienceCard,
    IndustryTermsConfig,
    RSIComponent,
    RSIController,
    RSISixGates,
    build_protocol_fingerprint,
    evaluate_s1_slice,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ledger = AppendOnlyRSILedger()
    ledger.init_component(RSIComponent.S1_EXPERIENCE_CARDS, version="v1", payload={"cards": []})
    ledger.init_component(RSIComponent.S2_ROUTING_PARAMS, version="v1", payload={"temperature": 0.75, "abstain_threshold": 0.55})
    ledger.init_component(RSIComponent.S3_SEARCH_TEMPLATES, version="v1", payload={"whitelist_domains": ["stcn.com"], "relevance_threshold": 0.50})

    gates = RSISixGates(industry_config=IndustryTermsConfig.from_yaml(ROOT / "config" / "industry_terms.yaml"))
    ctrl = RSIController(ledger=ledger, gates=gates)
    proto = build_protocol_fingerprint(eval_set_id="s1-v1", scorer_version="bal-acc-1", reviewer_prompt="fixed-v1")

    bc = BadCaseRecord(
        case_id="BC-600519-V5",
        source_case_ref="档案 600519 v5 深研质询 CHG-01",
        failure_type=BadCaseSource.UPHELD_CHALLENGE,
        evidence_summary="基本面研究员凭合同负债下降 42% 推出需求走弱，被反证官以打款政策调整推翻",
        hypothesis="我猜是因为起草前缺少消费品合同负债归因核查的方法提示",
        symbol="600519",
        period="2025H1",
    )

    new_fail = [
        ClaimReviewSample("NF-B1", "600519", "2025H1", "合同负债同比下降直接说明需求走弱", True, ("合同负债",)),
        ClaimReviewSample("NF-B2", "000858", "2025H1", "预收款减少直接证明高端白酒消费崩塌", True, ("预收款",)),
        ClaimReviewSample("NF-B3", "600519", "2025H1", "编造无出处毛利率提升", True, ()),
        ClaimReviewSample("NF-G1", "600519", "2025H1", "2025H1 营收与现金流匹配良好", False),
        ClaimReviewSample("NF-G2", "600519", "2025H1", "公告显示公司调整经销商打款节奏", False),
        ClaimReviewSample("NF-G3", "000858", "2025H1", "毛利率维持行业高位", False),
        ClaimReviewSample("NF-G4", "000858", "2025H1", "披露日对齐估值处于历史低分位", False),
        ClaimReviewSample("NF-G5", "600519", "2025H1", "归母净利润同口径正增长", False),
        ClaimReviewSample("NF-G6", "600519", "2025H1", "经营现金流净额稳健", False),
    ]
    frozen = [
        ClaimReviewSample("FR-B1", "300750", "2024FY", "无财报支撑的虚构数字", True, ()),
        ClaimReviewSample("FR-G1", "300750", "2024FY", "财报营收同口径增长", False),
        ClaimReviewSample("FR-G2", "300750", "2024FY", "海外基地稳步建设", False),
    ]
    oot = [
        ClaimReviewSample("OOT-B1", "600809", "2025Q3", "预收款大幅下滑直接说明消费品需求断崖", True, ("预收款",)),
        ClaimReviewSample("OOT-G1", "600809", "2025Q3", "单季营收按披露日核验无误", False),
        ClaimReviewSample("OOT-G2", "600809", "2025Q3", "净现比平稳", False),
    ]
    oot_pairs = {(s.symbol, s.period) for s in oot}

    print("=== 1. 反面演示：优化器提交“背答案过拟合经验卡”被防过拟合门控拦截并记账 ===\n")
    bad_card = ExperienceCard(
        card_id="L-OVERFIT",
        tags=["合同负债"],
        applicable_scope="全市场通用",
        lesson="茅台合同负债下降 42% 是基酒排期与打款政策所致，不代表需求走弱",
        must_check="查公告",
        invalidation_condition="无",
        source_ref="600519 v5",
    )
    b_s = {k: evaluate_s1_slice(slice_name=k, samples=ds, cards=[], runs=3) for k, ds in (("new_failures", new_fail), ("frozen_replay", frozen), ("out_of_time", oot))}
    c_s_bad = {k: evaluate_s1_slice(slice_name=k, samples=ds, cards=[bad_card], runs=3) for k, ds in (("new_failures", new_fail), ("frozen_replay", frozen), ("out_of_time", oot))}
    rep_bad = ctrl.submit_and_evaluate(
        proposal=CandidateProposal(
            proposal_id="P-OVERFIT", component=RSIComponent.S1_EXPERIENCE_CARDS, round_index=0,
            parent_version="v1", candidate_version="v2-bad", direction_key="s1:overfit_maotai",
            change_reason="直接把茅台合同负债案例写成经验卡防止再错",
            source_bad_cases=[bc], baseline_payload={"cards": []}, candidate_payload={"cards": [bad_card.model_dump()]},
        ),
        baseline_protocol=proto, candidate_protocol=proto, oot_pairs=oot_pairs,
        baseline_slices=b_s, candidate_slices=c_s_bad,
    )
    print(f"  [门控结果] 通过={rep_bad.passed} | 未通过门控={rep_bad.failed_gates}")
    for ck in rep_bad.checks:
        if not ck.passed:
            print(f"    ↳ {ck.gate_name}: {ck.detail}")

    print("\n=== 2. 正面闭环：泛化版经验卡 L-007 通过六道门控 + 人工批准发布 v2 ===\n")
    good_card = ExperienceCard(
        card_id="L-007",
        version="v1",
        tags=["合同负债", "预收款", "消费品"],
        applicable_scope="消费品公司，合同负债或预收款大幅变动时",
        lesson="先排除打款政策、渠道结构调整，再推断需求变化",
        must_check="公告或业绩说明会中关于渠道、打款政策的表述",
        invalidation_condition="公司已明确说明打款政策未变",
        source_ref="深研中被裁定成立的质询（档案 600519 v5）",
    )
    c_s = {k: evaluate_s1_slice(slice_name=k, samples=ds, cards=[good_card], runs=3) for k, ds in (("new_failures", new_fail), ("frozen_replay", frozen), ("out_of_time", oot))}
    prop_ok = CandidateProposal(
        proposal_id="P-L007", component=RSIComponent.S1_EXPERIENCE_CARDS, round_index=0,
        parent_version="v1", candidate_version="v2", direction_key="s1:generalized_L007",
        change_reason="沉淀消费品合同负债变动归因排查泛化经验卡 L-007",
        source_bad_cases=[bc], baseline_payload={"cards": []}, candidate_payload={"cards": [good_card.model_dump()]},
    )
    rep_ok = ctrl.submit_and_evaluate(
        proposal=prop_ok, baseline_protocol=proto, candidate_protocol=proto, oot_pairs=oot_pairs,
        baseline_slices=b_s, candidate_slices=c_s,
    )
    print(
        f"  [新失败集平衡准确率] 基线(v1)={b_s['new_failures'].mean_score:.3f} | "
        f"关掉经验库(OFF)={c_s['new_failures'].off_mean_score:.3f} | "
        f"候选(v2)={c_s['new_failures'].mean_score:.3f} "
        f"(净提升 {c_s['new_failures'].mean_score - b_s['new_failures'].mean_score:+.3f})"
    )
    for ck in rep_ok.checks:
        print(f"    [{'PASS' if ck.passed else 'FAIL'}] {ck.gate_name}: {ck.detail}")

    ledger.publish_candidate(
        component=RSIComponent.S1_EXPERIENCE_CARDS,
        candidate_version="v2",
        parent_version="v1",
        change_reason=prop_ok.change_reason,
        payload=prop_ok.candidate_payload,
        gate_report=rep_ok,
        approver="solo_dev",
        human_approved=True,
        approval_mode="self_approved",
    )

    print("\n=== 3. 递归继承证明：下一轮便宜的单 Agent 研究 600519 自动继承经验库 v2 ===\n")
    llm = MockLLMClient()
    runner = ResearchRunner(
        as_of=date(2025, 8, 20),
        policy_path=ROOT / "config" / "policies.yaml",
        search_sources_path=ROOT / "config" / "search_sources.yaml",
        llm_client=llm,
        rsi_ledger=ledger,
    )
    out = runner.start("复核贵州茅台 600519 半年报合同负债变动与利润质量", thread_id="demo-600519-v6")
    print(f"  [新档案记录版本] {out['dossier'].method_asset_versions}")
    hint_line = next(line for p in llm.received_prompts for line in p.splitlines() if "L-007" in line)
    print(f"  [注入单Agent的提示] {hint_line}")

    print("\n=== 4. 只追加账本一键回滚演练 ===\n")
    rolled = ledger.rollback(
        RSIComponent.S1_EXPERIENCE_CARDS,
        operator="solo_dev",
        reason="演示生产发现异常时一条命令回滚至上一个已验证版本",
    )
    print(f"  [回滚后在用版本] experience_cards={rolled} | 账本事件流={[ (e.event_type, e.version) for e in ledger.events if e.component == RSIComponent.S1_EXPERIENCE_CARDS ]}")


if __name__ == "__main__":
    main()
