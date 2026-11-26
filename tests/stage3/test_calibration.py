"""概率校准算法单测：Temperature Scaling 保序性、ECE / NLL 计算、离线拟合与同概率回放。"""

from __future__ import annotations

import math

import pytest

from zhengyan.routing.calibration import (
    argmax_label,
    compute_ece,
    compute_nll,
    fit_temperature,
    replay_raw_probs_accuracy,
    temperature_scale,
)


def test_temperature_scaling_preserves_order_and_controls_sharpness() -> None:
    raw = {"quick": 0.15, "standard": 0.55, "deep": 0.30}

    sharper = temperature_scale(raw, 0.5)   # T < 1 分布变尖
    flatter = temperature_scale(raw, 2.0)   # T > 1 分布变平

    assert math.isclose(sum(sharper.values()), 1.0, rel_tol=1e-9)
    assert math.isclose(sum(flatter.values()), 1.0, rel_tol=1e-9)
    assert argmax_label(sharper)[0] == argmax_label(raw)[0] == argmax_label(flatter)[0] == "standard"
    assert sharper["standard"] > raw["standard"] > flatter["standard"]

    with pytest.raises(ValueError):
        temperature_scale(raw, 0.0)


def test_ece_and_nll_exact_values() -> None:
    # 2 条样本都判对，平均置信度 0.70，真实准确率 1.00 -> ECE 必须精确等于 0.30
    probs = [
        {"quick": 0.70, "standard": 0.20, "deep": 0.10},
        {"quick": 0.10, "standard": 0.70, "deep": 0.20},
    ]
    labels = ["quick", "standard"]
    assert math.isclose(compute_ece(probs, labels, num_bins=10), 0.30, abs_tol=1e-6)
    assert math.isclose(compute_nll(probs, labels), -math.log(0.70), abs_tol=1e-6)


def test_fit_temperature_reduces_ece_on_underconfident_predictions() -> None:
    # 构造偏保守（欠自信）的判别器输出：23/24 正确，但平均置信度仅约 0.75
    raw_probs = [{"quick": 0.74, "standard": 0.16, "deep": 0.10}] * 8
    raw_probs += [{"quick": 0.12, "standard": 0.76, "deep": 0.12}] * 8
    raw_probs += [{"quick": 0.10, "standard": 0.15, "deep": 0.75}] * 7
    raw_probs += [{"quick": 0.15, "standard": 0.52, "deep": 0.33}]  # 1 题错判（真实是 deep）
    labels = ["quick"] * 8 + ["standard"] * 8 + ["deep"] * 8

    fit = fit_temperature(raw_probs, labels, objective="ece")
    assert fit.best_temperature < 1.0  # 欠自信需要 T < 1 拉尖
    assert fit.ece_before > 0.15
    assert fit.ece_after < 0.06
    assert fit.ece_after < fit.ece_before


def test_same_run_raw_probs_replay_keeps_argmax_accuracy_identical() -> None:
    """还原文档第 5.1 题：同一次运行保存下来的原始概率换三种温度回放，纯 argmax 准确率完全不变。"""
    raw_probs = [
        {"quick": 0.70, "standard": 0.20, "deep": 0.10},
        {"quick": 0.10, "standard": 0.60, "deep": 0.30},
        {"quick": 0.15, "standard": 0.50, "deep": 0.35},
        {"quick": 0.05, "standard": 0.25, "deep": 0.70},
    ]
    labels = ["quick", "standard", "deep", "deep"]  # 第 3 题错判，准确率恒为 0.75

    replay = replay_raw_probs_accuracy(raw_probs, labels, temperatures=[0.5, 0.75, 1.0, 1.5])
    accs = {metrics["argmax_accuracy"] for metrics in replay.values()}
    assert accs == {0.75}
    # 准确率不变，但 ECE / NLL 随温度改变
    assert len({metrics["ece"] for metrics in replay.values()}) > 1
