"""路由 24 题金标与意图 38+26 测试集评测器（对应文档第 1、1.1、1.2、26 题）。"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from zhengyan.core.enums import ResearchTier, UserIntent
from zhengyan.routing.calibration import compute_ece, compute_nll
from zhengyan.routing.classifiers import IntentClassifier
from zhengyan.routing.router import DualAxisRouter, RoutingDecision


@dataclass(frozen=True)
class RoutingCase:
    case_id: str
    question: str
    expected_tier: ResearchTier
    category: str


@dataclass(frozen=True)
class IntentCase:
    case_id: str
    question: str
    expected_intent: UserIntent
    group: str = "default"


@dataclass(frozen=True)
class RoutingEvalReport:
    total: int
    correct: int
    accuracy: float
    deep_total: int
    deep_recalled: int
    deep_recall: float
    abstain_count: int
    abstain_rate: float
    ece_raw: float
    ece_calibrated: float
    nll_raw: float
    nll_calibrated: float
    median_latency_ms: float
    predictions: list[tuple[str, ResearchTier, ResearchTier, RoutingDecision]]


@dataclass(frozen=True)
class IntentEvalReport:
    total: int
    correct: int
    accuracy: float
    trade_total: int
    trade_recalled: int
    trade_false_positives: int
    median_latency_ms: float
    errors: list[tuple[str, str, UserIntent, UserIntent]]


@dataclass(frozen=True)
class RepeatedEvalSummary:
    runs: int
    accuracies: list[float]
    mean_accuracy: float
    spread: float  # max - min 波动范围（供 Stage 6 噪声门控直接复用）
    exact_consistency_rate: float  # 同一题目在多次运行中预测完全一致的比例


def load_routing_cases(path: Path) -> list[RoutingCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        RoutingCase(
            case_id=item["case_id"],
            question=item["question"],
            expected_tier=ResearchTier(item["expected_tier"]),
            category=item.get("category", "general"),
        )
        for item in raw
    ]


def load_intent_cases(path: Path) -> list[IntentCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        IntentCase(
            case_id=item["case_id"],
            question=item["question"],
            expected_intent=UserIntent(item["expected_intent"]),
            group=item.get("group", "default"),
        )
        for item in raw
    ]


def evaluate_routing(
    cases: Sequence[RoutingCase],
    *,
    intent_classifier: IntentClassifier,
    router: DualAxisRouter,
    temperature_override: float | None = None,
) -> RoutingEvalReport:
    predictions: list[tuple[str, ResearchTier, ResearchTier, RoutingDecision]] = []
    raw_probs_list: list[dict[str, float]] = []
    cal_probs_list: list[dict[str, float]] = []
    prob_labels: list[str] = []
    latencies: list[float] = []

    correct = 0
    deep_total = 0
    deep_recalled = 0
    abstain_count = 0

    for case in cases:
        intent_pred = intent_classifier.classify_intent(case.question)
        decision = router.route(
            case.question,
            intent=intent_pred.intent,
            temperature_override=temperature_override,
        )
        predictions.append((case.case_id, case.expected_tier, decision.tier, decision))
        if decision.tier == case.expected_tier:
            correct += 1
        if case.expected_tier == ResearchTier.DEEP:
            deep_total += 1
            if decision.tier == ResearchTier.DEEP:
                deep_recalled += 1
        if decision.abstained:
            abstain_count += 1
        if decision.raw_probs is not None and decision.calibrated_probs is not None:
            raw_probs_list.append(decision.raw_probs)
            cal_probs_list.append(decision.calibrated_probs)
            prob_labels.append(case.expected_tier.value)
            latencies.append(decision.latency_ms)

    n = len(cases)
    return RoutingEvalReport(
        total=n,
        correct=correct,
        accuracy=round(correct / n, 6) if n else 0.0,
        deep_total=deep_total,
        deep_recalled=deep_recalled,
        deep_recall=round(deep_recalled / deep_total, 6) if deep_total else 1.0,
        abstain_count=abstain_count,
        abstain_rate=round(abstain_count / n, 6) if n else 0.0,
        ece_raw=round(compute_ece(raw_probs_list, prob_labels), 6) if prob_labels else 0.0,
        ece_calibrated=round(compute_ece(cal_probs_list, prob_labels), 6) if prob_labels else 0.0,
        nll_raw=round(compute_nll(raw_probs_list, prob_labels), 6) if prob_labels else 0.0,
        nll_calibrated=round(compute_nll(cal_probs_list, prob_labels), 6) if prob_labels else 0.0,
        median_latency_ms=round(statistics.median(latencies), 2) if latencies else 0.0,
        predictions=predictions,
    )


def evaluate_intent(
    cases: Sequence[IntentCase],
    *,
    classifier: IntentClassifier,
) -> IntentEvalReport:
    correct = 0
    trade_total = 0
    trade_recalled = 0
    trade_fp = 0
    latencies: list[float] = []
    errors: list[tuple[str, str, UserIntent, UserIntent]] = []

    for case in cases:
        pred = classifier.classify_intent(case.question)
        latencies.append(pred.latency_ms)
        if pred.intent == case.expected_intent:
            correct += 1
        else:
            errors.append((case.case_id, case.question, case.expected_intent, pred.intent))

        if case.expected_intent == UserIntent.TRADE_ORDER:
            trade_total += 1
            if pred.intent == UserIntent.TRADE_ORDER:
                trade_recalled += 1
        elif pred.intent == UserIntent.TRADE_ORDER:
            trade_fp += 1

    n = len(cases)
    return IntentEvalReport(
        total=n,
        correct=correct,
        accuracy=round(correct / n, 6) if n else 0.0,
        trade_total=trade_total,
        trade_recalled=trade_recalled,
        trade_false_positives=trade_fp,
        median_latency_ms=round(statistics.median(latencies), 2) if latencies else 0.0,
        errors=errors,
    )


def run_repeated_routing_eval(
    cases: Sequence[RoutingCase],
    *,
    make_router: Callable[[int], DualAxisRouter],
    intent_classifier: IntentClassifier,
    runs: int = 3,
) -> RepeatedEvalSummary:
    """重复运行多次路由评测，量化均值、波动范围（spread）与逐题一致性。"""
    accuracies: list[float] = []
    per_case_preds: dict[str, list[ResearchTier]] = {c.case_id: [] for c in cases}

    for run_idx in range(runs):
        router = make_router(run_idx)
        report = evaluate_routing(cases, intent_classifier=intent_classifier, router=router)
        accuracies.append(report.accuracy)
        for cid, _, pred_tier, _ in report.predictions:
            per_case_preds[cid].append(pred_tier)

    consistent_cases = sum(1 for preds in per_case_preds.values() if len(set(preds)) == 1)
    return RepeatedEvalSummary(
        runs=runs,
        accuracies=accuracies,
        mean_accuracy=round(sum(accuracies) / len(accuracies), 6),
        spread=round(max(accuracies) - min(accuracies), 6),
        exact_consistency_rate=round(consistent_cases / len(cases), 6) if cases else 1.0,
    )
