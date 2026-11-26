"""14 例计算类金标回归测试（对应文档第 1 题与第 1.1 节 evals/golden/compute_cases.json）。"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from zhengyan.core.enums import EvidenceGrade, MetricUnit
from zhengyan.core.models import GoldenComputeCase
from zhengyan.data import (
    AccountingBasisMismatchError,
    MockMarketDataProvider,
    compute_pe_ttm_pit,
    compute_ratio,
    compute_single_quarter,
    compute_yoy_growth,
    half_ulp_tolerance,
)

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "evals" / "golden" / "compute_cases.json"


def _load_golden_cases() -> list[GoldenComputeCase]:
    raw = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return [GoldenComputeCase.model_validate(item) for item in raw]


GOLDEN_CASES = _load_golden_cases()


def test_golden_case_count_matches_doc() -> None:
    """文档明确要求计算类金标为 14 例。"""
    assert len(GOLDEN_CASES) == 14


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.case_id)
def test_compute_golden_regression(case: GoldenComputeCase) -> None:
    provider = MockMarketDataProvider()
    symbol = case.params["symbol"]
    as_of = date.fromisoformat(case.params["as_of"])
    cards = {c.evidence_id: c for c in provider.to_evidence_cards(symbol, as_of)}

    if case.category == "basis_mismatch":
        curr = cards[f"EV-FIN-{symbol}-{case.params['current_period']}"]
        prev = cards[f"EV-FIN-{symbol}-{case.params['prior_period']}"]
        with pytest.raises(AccountingBasisMismatchError):
            compute_yoy_growth(curr, prev, case.params["metric"], as_of=as_of)
        return

    if case.category == "yoy_growth":
        curr = cards[f"EV-FIN-{symbol}-{case.params['current_period']}"]
        prev = cards[f"EV-FIN-{symbol}-{case.params['prior_period']}"]
        out = compute_yoy_growth(curr, prev, case.params["metric"], as_of=as_of)
        actual_metric = out.metrics[0]
    elif case.category == "single_quarter":
        cum = cards[f"EV-FIN-{symbol}-{case.params['cumulative_period']}"]
        prev_cum = cards[f"EV-FIN-{symbol}-{case.params['previous_cumulative_period']}"]
        out = compute_single_quarter(
            cum,
            prev_cum,
            case.params["metric"],
            quarter_label=case.params["quarter_label"],
        )
        actual_metric = out.metrics[0]
    elif case.category == "ratio":
        card = cards[f"EV-FIN-{symbol}-{case.params['period']}"]
        out = compute_ratio(
            card,
            case.params["numerator"],
            case.params["denominator"],
            output_name=case.params["output_name"],
            output_display_name=case.params["output_display_name"],
            output_unit=MetricUnit(case.params["output_unit"]),
        )
        actual_metric = out.metrics[0]
    elif case.category == "pe_ttm":
        stmts = provider.get_financial_statements(symbol, as_of)
        price = provider.get_price_on_or_before(symbol, as_of)
        assert price is not None
        out = compute_pe_ttm_pit(stmts, price, as_of)
        actual_metric = next(m for m in out.metrics if m.name == "pe_ttm")
    else:
        pytest.fail(f"未知金标类别: {case.category}")

    assert out.grade == EvidenceGrade.A
    assert len(out.derived_from) >= 1
    assert out.published_at is not None and out.published_at <= as_of
    assert actual_metric.unit == case.expected_unit
    assert case.expected_value is not None

    expected = Decimal(case.expected_value)
    tol = half_ulp_tolerance(case.expected_value)
    assert abs(actual_metric.value - expected) <= tol, (
        f"{case.case_id} ({case.description}): "
        f"实际={actual_metric.value}, 期望={expected} ± {tol}"
    )
