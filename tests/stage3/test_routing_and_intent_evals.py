"""24 题路由金标、38+26 意图留出集评测、以及 LLM 随机波动复现测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from zhengyan.graph import ResearchRunner
from zhengyan.routing import (
    DualAxisRouter,
    MockRoutingClassifier,
    ModelAxisConfig,
    RuleFirstIntentClassifier,
    evaluate_intent,
    evaluate_routing,
    fit_temperature,
    load_intent_cases,
    load_routing_cases,
    replay_raw_probs_accuracy,
    run_repeated_routing_eval,
)

ROOT = Path(__file__).resolve().parents[2]
ROUTING_GOLDEN = ROOT / "evals" / "golden" / "routing_cases.json"
INTENT_MAIN = ROOT / "evals" / "golden" / "intent_cases.json"
INTENT_HOLDOUT = ROOT / "evals" / "golden" / "intent_holdout.json"
MODELS_PATH = ROOT / "config" / "models.yaml"
POLICY_PATH = ROOT / "config" / "policies.yaml"
SEARCH_PATH = ROOT / "config" / "search_sources.yaml"


def test_routing_golden_24_cases_accuracy_deep_recall_and_calibration() -> None:
    cases = load_routing_cases(ROUTING_GOLDEN)
    assert len(cases) == 24

    intent_clf = RuleFirstIntentClassifier()
    model_axis = ModelAxisConfig.from_yaml(MODELS_PATH)

    # 1. T=1.0（未校准）与 T=0.75（校准后）对比
    router_t1 = DualAxisRouter(
        classifier=MockRoutingClassifier(underconfident_scale=0.65, noise_std=0.0),
        model_axis=model_axis,
        temperature=1.0,
    )
    router_t075 = DualAxisRouter(
        classifier=MockRoutingClassifier(underconfident_scale=0.65, noise_std=0.0),
        model_axis=model_axis,
        temperature=0.75,
    )

    rep_t1 = evaluate_routing(cases, intent_classifier=intent_clf, router=router_t1)
    rep_t075 = evaluate_routing(cases, intent_classifier=intent_clf, router=router_t075)

    # 核心硬指标：准确率 >= 95.8%（23/24），深研召回 100%（8/8）
    assert rep_t075.correct >= 23
    assert rep_t075.accuracy >= 0.958
    assert rep_t075.deep_recall == 1.0
    assert rep_t075.deep_recalled == rep_t075.deep_total == 8
    # 校准后 ECE 显著低于未校准（偏保守分布经 T=0.75 拉尖后校准误差大幅下降）
    assert rep_t075.ece_calibrated < rep_t1.ece_raw

    # 离线拟合最优温度确实落在 < 1.0 区间，且能进一步压低 ECE
    raw_probs = [d.raw_probs for _, _, _, d in rep_t1.predictions if d.raw_probs is not None]
    labels = [gold.value for _, gold, _, d in rep_t1.predictions if d.raw_probs is not None]
    fit = fit_temperature(raw_probs, labels, objective="ece")
    assert fit.best_temperature < 1.0
    assert fit.ece_after < fit.ece_before

    # 同一次运行的原始概率回放三种温度，纯 argmax 准确率恒定不变
    replay = replay_raw_probs_accuracy(raw_probs, labels, [0.5, 0.75, 1.0])
    assert len({m["argmax_accuracy"] for m in replay.values()}) == 1


def test_intent_38_main_and_26_holdout_trade_20_of_20_zero_fp() -> None:
    """对应文档第 1、1.1、26.1 题：原测试集 38 题 + 留出集 26 题（12 新下单说法 + 14 买卖干扰题），
    合计 20 道下单题达到 20/20 识别、0 误报。
    """
    main_cases = load_intent_cases(INTENT_MAIN)
    holdout_cases = load_intent_cases(INTENT_HOLDOUT)
    assert len(main_cases) == 38
    assert len(holdout_cases) == 26
    assert sum(1 for c in holdout_cases if c.group == "new_trade_phrasing") == 12
    assert sum(1 for c in holdout_cases if c.group == "trade_distractor") == 14

    clf = RuleFirstIntentClassifier()
    rep_main = evaluate_intent(main_cases, classifier=clf)
    rep_holdout = evaluate_intent(holdout_cases, classifier=clf)
    rep_all = evaluate_intent(main_cases + holdout_cases, classifier=clf)

    assert rep_main.accuracy >= 0.97
    assert rep_holdout.accuracy == 1.0
    assert rep_all.trade_total == 20
    assert rep_all.trade_recalled == 20
    assert rep_all.trade_false_positives == 0


def test_stochastic_llm_drift_vs_deterministic_consistency() -> None:
    """复现文档核心教训：大模型判别器带随机采样波动，3 次运行准确率会漂移；
    确定性判别器 3 次运行结果 100% 一致（spread == 0）。
    """
    cases = load_routing_cases(ROUTING_GOLDEN)
    intent_clf = RuleFirstIntentClassifier()

    # 随机扰动 LLM 判别器：多次运行出现非零波动 spread > 0 且逐题一致性 < 1.0
    stochastic_summary = run_repeated_routing_eval(
        cases,
        intent_classifier=intent_clf,
        make_router=lambda i: DualAxisRouter(
            classifier=MockRoutingClassifier(underconfident_scale=0.65, noise_std=0.85, seed=100 + i),
            temperature=0.75,
        ),
        runs=3,
    )
    assert stochastic_summary.exact_consistency_rate < 1.0

    # 确定性判别器：3 次运行 100% 一致，spread == 0
    det_summary = run_repeated_routing_eval(
        cases,
        intent_classifier=intent_clf,
        make_router=lambda _i: DualAxisRouter(
            classifier=MockRoutingClassifier(underconfident_scale=0.65, noise_std=0.0),
            temperature=0.75,
        ),
        runs=3,
    )
    assert det_summary.spread == 0.0
    assert det_summary.exact_consistency_rate == 1.0


def test_langgraph_route_node_records_calibrated_probs_and_model_bindings() -> None:
    runner = ResearchRunner(
        as_of=date(2025, 8, 20),
        policy_path=POLICY_PATH,
        search_sources_path=SEARCH_PATH,
    )
    out = runner.start("贵州茅台现在估值合理吗，还要不要继续持有", thread_id="th-stage3-deep")
    assert out["tier"].value == "deep"
    assert out["route_raw_probs"] is not None
    assert out["route_calibrated_probs"] is not None
    assert "chair_adjudicator" in out["model_bindings"]
    assert "reason(" in out["model_bindings"]["chair_adjudicator"]
