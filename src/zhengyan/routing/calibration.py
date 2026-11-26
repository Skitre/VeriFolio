"""概率校准核心算法：Temperature Scaling、ECE、NLL、可靠性分桶与离线拟合。

对应文档第 5.1 题与八股“模型校准与评测”：
1. Temperature Scaling（温度缩放）：
   对概率 p 取对数得到 log-prob z_k = log(max(p_k, eps))，除以温度 T 后做 softmax：
     p_cal_k(T) = exp(z_k / T) / sum_j exp(z_j / T)
   - T > 1 使分布更平（抑制过度自信）
   - T < 1 使分布更尖（修正保守/欠自信，如文档中 DeepSeek 拟合出 T=0.75）
   - 单调严格保序：不改变 argmax 类别（因此同一次运行保存下来的原始概率换 T 回放，纯 argmax 准确率恒定！）
2. ECE（Expected Calibration Error，期望校准误差）：
   按最高预测置信度分成 M 个等宽桶（默认 10 桶），加权平均每个桶的 |acc(B_m) - conf(B_m)|。
3. NLL（Negative Log-Likelihood，负对数似然）：
   -1/N * sum(log(p_cal[y_i]))，对“高置信但判错”给予强惩罚。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

_EPS = 1e-12


@dataclass(frozen=True)
class CalibrationBin:
    bin_index: int
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float
    gap: float


@dataclass(frozen=True)
class TemperatureFitResult:
    best_temperature: float
    objective: Literal["ece", "nll"]
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    bins_before: list[CalibrationBin]
    bins_after: list[CalibrationBin]


def temperature_scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """对离散类别概率分布执行温度缩放（含数值稳定 log-sum-exp 技巧）。"""
    if temperature <= 0:
        raise ValueError(f"温度 T 必须 > 0，当前收到 {temperature}")
    if not probs:
        raise ValueError("概率分布不能为空")

    scaled_logits = {
        label: math.log(max(float(p), _EPS)) / temperature
        for label, p in probs.items()
    }
    max_logit = max(scaled_logits.values())
    exps = {label: math.exp(val - max_logit) for label, val in scaled_logits.items()}
    total = sum(exps.values())
    return {label: val / total for label, val in exps.items()}


def argmax_label(probs: dict[str, float]) -> tuple[str, float]:
    """返回最高概率对应的标签及其概率值（键排序稳定打破平局）。"""
    best_label = max(sorted(probs.keys()), key=lambda k: probs[k])
    return best_label, probs[best_label]


def top2_labels(probs: dict[str, float]) -> list[tuple[str, float]]:
    """按概率降序返回前两个候选类别（用于拒判时在前两名中选更重流程）。"""
    ordered = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:2]


def compute_reliability_bins(
    probs_list: Sequence[dict[str, float]],
    labels: Sequence[str],
    *,
    num_bins: int = 10,
) -> list[CalibrationBin]:
    """按最高预测置信度分桶统计平均置信度与真实准确率（可靠性曲线数据）。"""
    if len(probs_list) != len(labels):
        raise ValueError("预测分布数与标签数不一致")
    if num_bins <= 0:
        raise ValueError("num_bins 必须为正整数")

    buckets: list[list[tuple[float, float]]] = [[] for _ in range(num_bins)]
    for probs, gold in zip(probs_list, labels):
        pred, conf = argmax_label(probs)
        correct = 1.0 if pred == gold else 0.0
        # 置信度落在 (lower, upper]，conf == 0.0 归入第 0 桶
        idx = min(num_bins - 1, max(0, int(math.ceil(conf * num_bins)) - 1)) if conf > 0 else 0
        buckets[idx].append((conf, correct))

    bins: list[CalibrationBin] = []
    for i, items in enumerate(buckets):
        lower = i / num_bins
        upper = (i + 1) / num_bins
        if not items:
            bins.append(CalibrationBin(i, lower, upper, 0, 0.0, 0.0, 0.0))
            continue
        avg_conf = sum(c for c, _ in items) / len(items)
        acc = sum(ok for _, ok in items) / len(items)
        bins.append(
            CalibrationBin(
                bin_index=i,
                lower=round(lower, 4),
                upper=round(upper, 4),
                count=len(items),
                avg_confidence=round(avg_conf, 6),
                accuracy=round(acc, 6),
                gap=round(abs(acc - avg_conf), 6),
            )
        )
    return bins


def compute_ece(
    probs_list: Sequence[dict[str, float]],
    labels: Sequence[str],
    *,
    num_bins: int = 10,
) -> float:
    """计算期望校准误差 ECE = sum(|B_m|/N * |acc(B_m) - conf(B_m)|)。"""
    n = len(labels)
    if n == 0:
        return 0.0
    bins = compute_reliability_bins(probs_list, labels, num_bins=num_bins)
    return sum((b.count / n) * b.gap for b in bins)


def compute_nll(
    probs_list: Sequence[dict[str, float]],
    labels: Sequence[str],
) -> float:
    """计算平均负对数似然 NLL（多分类交叉熵）。"""
    n = len(labels)
    if n == 0:
        return 0.0
    total = 0.0
    for probs, gold in zip(probs_list, labels):
        p_gold = max(float(probs.get(gold, _EPS)), _EPS)
        total -= math.log(p_gold)
    return total / n


def fit_temperature(
    raw_probs_list: Sequence[dict[str, float]],
    labels: Sequence[str],
    *,
    objective: Literal["ece", "nll"] = "ece",
    t_min: float = 0.20,
    t_max: float = 3.00,
    step: float = 0.01,
    num_bins: int = 10,
) -> TemperatureFitResult:
    """在验证/金标集上离线网格搜索最优温度 T*（最小化 ECE 或 NLL）。"""
    if len(raw_probs_list) == 0 or len(raw_probs_list) != len(labels):
        raise ValueError("离线拟合样本不能为空且长度必须匹配")

    ece_before = compute_ece(raw_probs_list, labels, num_bins=num_bins)
    nll_before = compute_nll(raw_probs_list, labels)
    bins_before = compute_reliability_bins(raw_probs_list, labels, num_bins=num_bins)

    best_t = 1.0
    best_score = float("inf")
    best_nll_tie = float("inf")

    steps = int(round((t_max - t_min) / step))
    for i in range(steps + 1):
        t = round(t_min + i * step, 4)
        if t <= 0:
            continue
        scaled = [temperature_scale(p, t) for p in raw_probs_list]
        ece = compute_ece(scaled, labels, num_bins=num_bins)
        nll = compute_nll(scaled, labels)
        primary = ece if objective == "ece" else nll
        secondary = nll if objective == "ece" else ece
        if (primary < best_score - 1e-12) or (
            abs(primary - best_score) <= 1e-12 and secondary < best_nll_tie
        ):
            best_score = primary
            best_nll_tie = secondary
            best_t = t

    final_scaled = [temperature_scale(p, best_t) for p in raw_probs_list]
    return TemperatureFitResult(
        best_temperature=best_t,
        objective=objective,
        ece_before=round(ece_before, 6),
        ece_after=round(compute_ece(final_scaled, labels, num_bins=num_bins), 6),
        nll_before=round(nll_before, 6),
        nll_after=round(compute_nll(final_scaled, labels), 6),
        bins_before=bins_before,
        bins_after=compute_reliability_bins(final_scaled, labels, num_bins=num_bins),
    )


def replay_raw_probs_accuracy(
    raw_probs_list: Sequence[dict[str, float]],
    labels: Sequence[str],
    temperatures: Sequence[float],
) -> dict[float, dict[str, float]]:
    """在同一份保存下来的原始概率上用不同温度回放，验证“换 T 不改纯 argmax 准确率，只改校准度”。"""
    summary: dict[float, dict[str, float]] = {}
    n = len(labels)
    for t in temperatures:
        scaled = [temperature_scale(p, t) for p in raw_probs_list]
        correct = sum(1 for p, y in zip(scaled, labels) if argmax_label(p)[0] == y)
        summary[t] = {
            "argmax_accuracy": round(correct / n, 6) if n else 0.0,
            "ece": round(compute_ece(scaled, labels), 6),
            "nll": round(compute_nll(scaled, labels), 6),
        }
    return summary
