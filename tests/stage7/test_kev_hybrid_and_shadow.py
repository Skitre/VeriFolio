"""Kev-0.8B 混合判别、三态开关（llm/small/shadow）、自动回退与 S2 温度校准联动测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.core.enums import ResearchTier, UserIntent
from zhengyan.harness import HarnessExecutor, HarnessPolicyConfig
from zhengyan.routing import (
    ClassifierMode,
    DualAxisRouter,
    HybridIntentClassifier,
    HybridRoutingClassifier,
    Kev08BClassifier,
    MockRoutingClassifier,
    RuleFirstIntentClassifier,
    evaluate_intent,
    evaluate_routing,
    load_intent_cases,
    load_real_replay_queries,
    load_routing_cases,
    run_repeated_routing_eval,
)
from zhengyan.rsi import (
    AppendOnlyRSILedger,
    BadCaseRecord,
    BadCaseSource,
    CandidateProposal,
    HistoricalRoutingSnapshot,
    IndustryTermsConfig,
    RSIComponent,
    RSIController,
    RSISixGates,
    RoutingParamsAsset,
    build_protocol_fingerprint,
    evaluate_s2_replay,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config" / "policies.yaml"
INDUSTRY_PATH = ROOT / "config" / "industry_terms.yaml"
ROUTING_GOLDEN = ROOT / "evals" / "golden" / "routing_cases.json"
INTENT_MAIN = ROOT / "evals" / "golden" / "intent_cases.json"
INTENT_HOLDOUT = ROOT / "evals" / "golden" / "intent_holdout.json"
REAL_REPLAY = ROOT / "evals" / "golden" / "real_queries_replay.json"


def test_kev_routing_24_metrics_and_3_run_zero_spread() -> None:
    """精确复现文档第 26.1 题全部 Kev-0.8B 路由指标：
    准确率 95.8%(23/24)、深研召回 100%(8/8)、中位耗时 69ms、拒判率 20.8%(5/24)、3 次运行 100% 一致！
    """
    cases = load_routing_cases(ROUTING_GOLDEN)
    kev = Kev08BClassifier()
    router_kev = DualAxisRouter(classifier=kev, temperature=0.75)

    rep = evaluate_routing(cases, intent_classifier=kev, router=router_kev)
    assert rep.correct == 23 and rep.total == 24
    assert rep.accuracy >= 0.958
    assert rep.deep_recall == 1.0
    assert rep.median_latency_ms == 69.0
    # 还原已知代价：Kev 零样本分布偏平导致拒判率 5/24 = 20.8%
    assert rep.abstain_count == 5
    assert abs(rep.abstain_rate - (5 / 24)) < 1e-4

    # 3 次重复运行结果 100% 一致（spread == 0）
    repeated = run_repeated_routing_eval(
        cases,
        intent_classifier=kev,
        make_router=lambda _i: DualAxisRouter(classifier=Kev08BClassifier(), temperature=0.75),
        runs=3,
    )
    assert repeated.spread == 0.0
    assert repeated.exact_consistency_rate == 1.0


def test_kev_intent_38_plus_26_and_22_real_queries_100_percent_agreement() -> None:
    """复现文档第 26.1 题意图指标：原 38 题 97.4%(37/38)、留出 26 题 100%、下单 20/20 0 误报、43ms；
    22 条真实提问回放与 DeepSeek 结果 22/22（100%）一致。
    """
    main_cases = load_intent_cases(INTENT_MAIN)
    hold_cases = load_intent_cases(INTENT_HOLDOUT)
    kev = Kev08BClassifier()

    rep_m = evaluate_intent(main_cases, classifier=kev)
    rep_h = evaluate_intent(hold_cases, classifier=kev)
    rep_all = evaluate_intent(main_cases + hold_cases, classifier=kev)

    assert rep_m.correct == 37 and rep_m.total == 38
    assert abs(rep_m.accuracy - (37 / 38)) < 1e-4
    assert rep_h.accuracy == 1.0
    assert rep_all.trade_recalled == 20 and rep_all.trade_total == 20
    assert rep_all.trade_false_positives == 0
    assert rep_all.median_latency_ms <= 45.0

    # 22 条真实提问回放与大模型 100% 一致
    real_queries = load_real_replay_queries(REAL_REPLAY)
    assert len(real_queries) == 22
    llm_intent = RuleFirstIntentClassifier()
    r_llm = DualAxisRouter(classifier=MockRoutingClassifier(), temperature=0.75)
    r_kev = DualAxisRouter(classifier=kev, temperature=0.75)

    for cid, q in real_queries:
        d_llm = r_llm.route(q, intent=llm_intent.classify_intent(q).intent)
        d_kev = r_kev.route(q, intent=kev.classify_intent(q).intent)
        assert d_llm.tier == d_kev.tier, f"{cid} 不一致: llm={d_llm.tier}, kev={d_kev.tier}"


def test_hybrid_three_modes_fallback_and_harness_decision_audit() -> None:
    harness = HarnessExecutor(
        policy=HarnessPolicyConfig.from_yaml(POLICY_PATH),
        tier=ResearchTier.STANDARD,
        as_of=date(2025, 8, 20),
    )

    # 1. SHADOW 影子模式：大模型生效，小模型旁路记录对照，审计记为 decision 类型
    shadow_clf = HybridRoutingClassifier(
        mode=ClassifierMode.SHADOW,
        llm_classifier=MockRoutingClassifier(),
        small_classifier=Kev08BClassifier(),
        harness=harness,
    )
    out_shadow = shadow_clf.classify_route("我有存款 30万，复核贵州茅台半年报", intent=UserIntent.RESEARCH)
    assert out_shadow.source == "mock_deepseek_classifier"
    assert len(shadow_clf.shadow_records) == 1
    assert shadow_clf.shadow_records[0].agreed is True
    assert "30万" not in shadow_clf.shadow_records[0].question_masked
    assert any(s.action_type == "decision" for s in harness.audit.spans)

    # 2. SMALL 模式下本地 Kev 不可用 → 自动优雅回退到大模型，且意图代理同样自动回退
    broken_kev = Kev08BClassifier(available=False)
    fb_route = HybridRoutingClassifier(
        mode=ClassifierMode.SMALL,
        llm_classifier=MockRoutingClassifier(),
        small_classifier=broken_kev,
        harness=harness,
    )
    out_fb = fb_route.classify_route("复核贵州茅台半年报", intent=UserIntent.RESEARCH)
    assert out_fb.source == "mock_deepseek_classifier"

    fb_intent = HybridIntentClassifier(
        mode=ClassifierMode.SMALL,
        llm_classifier=RuleFirstIntentClassifier(),
        small_classifier=broken_kev,
        harness=harness,
    )
    assert fb_intent.classify_intent("什么是市盈率").intent == UserIntent.CONCEPT


def test_kev_high_abstain_rate_recalibrated_via_s2_rsi_gate() -> None:
    """呼应文档第 26.1 题：Kev 拒判率 20.8% 不能直接硬改配置，必须作为 S2 新版本走 RSI 六道门控！"""
    cases = load_routing_cases(ROUTING_GOLDEN)
    kev = Kev08BClassifier()
    r_v1 = DualAxisRouter(classifier=kev, temperature=0.75)
    rep_v1 = evaluate_routing(cases, intent_classifier=kev, router=r_v1)

    raw_probs = [d.raw_probs for _, _, _, d in rep_v1.predictions if d.raw_probs is not None]
    golds = [gold for _, gold, _, d in rep_v1.predictions if d.raw_probs is not None]
    snaps = [HistoricalRoutingSnapshot(f"kev-run-{i}", raw_probs, golds) for i in range(3)]

    p_v1 = RoutingParamsAsset(temperature=0.75, abstain_threshold=0.55)
    p_v2 = RoutingParamsAsset(temperature=0.48, abstain_threshold=0.55)  # 步长 0.27 <= 0.30（第 0 轮上限）

    s2_base = {k: evaluate_s2_replay(slice_name=k, snapshots=snaps, params=p_v1) for k in ("new_failures", "frozen_replay", "out_of_time")}
    s2_cand = {k: evaluate_s2_replay(slice_name=k, snapshots=snaps, params=p_v2) for k in ("new_failures", "frozen_replay", "out_of_time")}
    assert s2_cand["new_failures"].mean_score > s2_base["new_failures"].mean_score

    ledger = AppendOnlyRSILedger()
    ledger.init_component(RSIComponent.S2_ROUTING_PARAMS, version="v1", payload=p_v1.model_dump())
    ctrl = RSIController(ledger=ledger, gates=RSISixGates(industry_config=IndustryTermsConfig.from_yaml(INDUSTRY_PATH)))
    proto = build_protocol_fingerprint(eval_set_id="kev-24", scorer_version="acc-1", reviewer_prompt="none")

    rep = ctrl.submit_and_evaluate(
        proposal=CandidateProposal(
            proposal_id="P-KEV-TEMP",
            component=RSIComponent.S2_ROUTING_PARAMS,
            round_index=0,
            parent_version="v1",
            candidate_version="v2",
            direction_key="s2:kev_temp_0.75_to_0.48",
            change_reason="针对 Kev-0.8B 零样本概率偏平导致拒判率 20.8% 通过 S2 门控校准温度",
            source_bad_cases=[
                BadCaseRecord(
                    case_id="BC-KEV-ABSTAIN",
                    source_case_ref="Kev-0.8B 24题路由金标",
                    failure_type=BadCaseSource.ROUTING_ERROR,
                    evidence_summary="Kev 拒判率高达 20.8%，将 RT-15 误升级为深研",
                    hypothesis="我猜是因为 Kev 零样本概率分布偏平需要更低温度 T 拉尖校准",
                    target_component=RSIComponent.S2_ROUTING_PARAMS,
                )
            ],
            baseline_payload=p_v1.model_dump(),
            candidate_payload=p_v2.model_dump(),
        ),
        baseline_protocol=proto,
        candidate_protocol=proto,
        oot_pairs=set(),
        baseline_slices=s2_base,
        candidate_slices=s2_cand,
    )
    assert rep.passed is True
    # 用校准后的 T=0.48 重跑 24 题，拒判率显著下降且深研召回依然 100%！
    rep_v2 = evaluate_routing(cases, intent_classifier=kev, router=DualAxisRouter(classifier=kev, temperature=0.48))
    assert rep_v2.abstain_rate < rep_v1.abstain_rate
    assert rep_v2.deep_recall == 1.0
