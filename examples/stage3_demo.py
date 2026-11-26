"""Stage 3 算法岗实战演示：离线温度拟合、可靠性分桶、LLM 随机波动真相、24题+64题评测报告。

运行：uv run python examples/stage3_demo.py
"""

from __future__ import annotations

from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    routing_cases = load_routing_cases(ROOT / "evals" / "golden" / "routing_cases.json")
    intent_main = load_intent_cases(ROOT / "evals" / "golden" / "intent_cases.json")
    intent_holdout = load_intent_cases(ROOT / "evals" / "golden" / "intent_holdout.json")
    model_axis = ModelAxisConfig.from_yaml(ROOT / "config" / "models.yaml")
    intent_clf = RuleFirstIntentClassifier()

    print("=== 1. 24 题路由金标评测：未校准 (T=1.0) vs 温度校准 (T=0.75) ===\n")
    r_t1 = DualAxisRouter(classifier=MockRoutingClassifier(), model_axis=model_axis, temperature=1.0)
    r_t075 = DualAxisRouter(classifier=MockRoutingClassifier(), model_axis=model_axis, temperature=0.75)
    rep1 = evaluate_routing(routing_cases, intent_classifier=intent_clf, router=r_t1)
    rep075 = evaluate_routing(routing_cases, intent_classifier=intent_clf, router=r_t075)

    print(
        f"  T=1.00 | 准确率={rep1.accuracy*100:.1f}% ({rep1.correct}/{rep1.total}) | "
        f"深研召回={rep1.deep_recall*100:.0f}% | 拒判率={rep1.abstain_rate*100:.1f}% | "
        f"ECE={rep1.ece_raw:.4f} | NLL={rep1.nll_raw:.4f}"
    )
    print(
        f"  T=0.75 | 准确率={rep075.accuracy*100:.1f}% ({rep075.correct}/{rep075.total}) | "
        f"深研召回={rep075.deep_recall*100:.0f}% | 拒判率={rep075.abstain_rate*100:.1f}% | "
        f"ECE={rep075.ece_calibrated:.4f} | NLL={rep075.nll_calibrated:.4f}\n"
    )

    print("=== 2. 离线拟合最优温度 T* 与同概率回放验证（复现文档第 5.1 题教训） ===\n")
    raw_probs = [d.raw_probs for _, _, _, d in rep1.predictions if d.raw_probs is not None]
    labels = [gold.value for _, gold, _, d in rep1.predictions if d.raw_probs is not None]
    fit_clean = fit_temperature(raw_probs, labels, objective="ece")
    print(
        f"  [现象提示] 当前 20 道入模题纯 argmax 准确率为 100%，无约束拟合会向极小温度坍缩：T*={fit_clean.best_temperature:.2f}"
    )

    # 模拟更贴近真实 LLM 判别器的验证集（19/20 正确、1 题错判且整体偏保守）：
    # 此时 T 过小会导致错判样本的 NLL 爆炸、ECE 上升，最优 T* 自然落在 0.5~0.75 区间
    realistic_probs = list(raw_probs[:-1]) + [{"quick": 0.12, "standard": 0.56, "deep": 0.32}]
    realistic_labels = list(labels[:-1]) + ["deep"]
    fit_ece = fit_temperature(realistic_probs, realistic_labels, objective="ece")
    fit_nll = fit_temperature(realistic_probs, realistic_labels, objective="nll")
    print(
        f"  [含 1 错例的真实校准集] 按 ECE 拟合 T*={fit_ece.best_temperature:.2f} "
        f"(ECE {fit_ece.ece_before:.4f} -> {fit_ece.ece_after:.4f}) | "
        f"按 NLL 拟合 T*={fit_nll.best_temperature:.2f} "
        f"(NLL {fit_nll.nll_before:.4f} -> {fit_nll.nll_after:.4f}, ECE={fit_nll.ece_after:.4f})"
    )
    print("  [可靠性曲线非空分桶（按 NLL 校准后）]")
    for b in fit_nll.bins_after:
        if b.count > 0:
            print(
                f"    桶 ({b.lower:.1f}, {b.upper:.1f}] | 样本数={b.count:2d} | "
                f"平均置信度={b.avg_confidence:.3f} | 实际准确率={b.accuracy:.3f} | 偏差={b.gap:.3f}"
            )
    print("  [同一次运行的原始概率回放三种温度]")
    replay = replay_raw_probs_accuracy(realistic_probs, realistic_labels, [0.50, 0.75, 1.00])
    for t, metrics in replay.items():
        print(
            f"    T={t:.2f} -> 纯 argmax 准确率={metrics['argmax_accuracy']*100:.1f}% "
            f"| ECE={metrics['ece']:.4f} | NLL={metrics['nll']:.4f}"
        )

    print("\n  [为什么文档里改完温度重跑准确率会从 92% 掉到 83%？模拟 LLM 判别随机波动 3 次运行]")
    noisy = run_repeated_routing_eval(
        routing_cases,
        intent_classifier=intent_clf,
        make_router=lambda i: DualAxisRouter(
            classifier=MockRoutingClassifier(noise_std=0.85, seed=42 + i),
            temperature=0.75,
        ),
        runs=3,
    )
    clean = run_repeated_routing_eval(
        routing_cases,
        intent_classifier=intent_clf,
        make_router=lambda _i: DualAxisRouter(
            classifier=MockRoutingClassifier(noise_std=0.0),
            temperature=0.75,
        ),
        runs=3,
    )
    print(
        f"    随机波动 LLM 判别器：3 次准确率={[round(a*100, 1) for a in noisy.accuracies]}% | "
        f"波动范围 spread={noisy.spread*100:.1f}% | 逐题一致率={noisy.exact_consistency_rate*100:.1f}%"
    )
    print(
        f"    确定性判别器       ：3 次准确率={[round(a*100, 1) for a in clean.accuracies]}% | "
        f"波动范围 spread={clean.spread*100:.1f}% | 逐题一致率={clean.exact_consistency_rate*100:.1f}%\n"
    )

    print("=== 3. 意图测试集（38 题原集 + 26 题留出干扰集 = 64 题）评测 ===\n")
    rep_main = evaluate_intent(intent_main, classifier=intent_clf)
    rep_hold = evaluate_intent(intent_holdout, classifier=intent_clf)
    rep_all = evaluate_intent(intent_main + intent_holdout, classifier=intent_clf)
    print(f"  原测试集 (38 题)：准确率={rep_main.accuracy*100:.1f}% ({rep_main.correct}/{rep_main.total})")
    print(f"  留出集   (26 题)：准确率={rep_hold.accuracy*100:.1f}% ({rep_hold.correct}/{rep_hold.total})（含 12 新下单 + 14 买卖干扰题）")
    print(
        f"  合并统计 (64 题)：下单意图召回={rep_all.trade_recalled}/{rep_all.trade_total}，"
        f"误报={rep_all.trade_false_positives} 次"
    )


if __name__ == "__main__":
    main()
