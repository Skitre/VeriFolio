"""时点纪律（Point-in-Time / 防前视偏差）专项测试（对应文档第 10 题）。"""

from __future__ import annotations

from datetime import date

import pytest

from zhengyan.data import ComputeSkillError, MockMarketDataProvider, compute_pe_ttm_pit


def test_statements_filtered_by_disclosed_at_not_period_end() -> None:
    """2024FY 报告期截止日是 2024-12-31，但披露日是 2025-04-03。
    在 2025-03-20 研究时，严禁取到 2024FY（否则即前视偏差）。
    """
    provider = MockMarketDataProvider()
    stmts_before = provider.get_financial_statements("600519", as_of=date(2025, 3, 20))
    periods_before = {s.period for s in stmts_before}
    assert "2024Q3" in periods_before
    assert "2024FY" not in periods_before

    stmts_after = provider.get_financial_statements("600519", as_of=date(2025, 4, 3))
    periods_after = {s.period for s in stmts_after}
    assert "2024FY" in periods_after
    assert "2025H1" not in periods_after


def test_pe_ttm_refuses_missing_prior_fy_before_disclosure() -> None:
    """2025-03-20 时 2024FY 尚未披露，仅有 2024Q1/H1/Q3，缺少 2023FY，
    按披露日对齐计算 PE(TTM) 必须明确拒绝，而不是偷看未来年报。
    """
    provider = MockMarketDataProvider()
    as_of = date(2025, 3, 20)
    stmts = provider.get_financial_statements("600519", as_of)
    price = provider.get_price_on_or_before("600519", as_of)
    assert price is not None

    with pytest.raises(ComputeSkillError, match="缺少推算 TTM 所需的 2023FY"):
        compute_pe_ttm_pit(stmts, price, as_of)


def test_pe_ttm_rejects_future_price() -> None:
    """不允许拿晚于研究时点的行情去算历史时点 PE。"""
    provider = MockMarketDataProvider()
    stmts = provider.get_financial_statements("600519", as_of=date(2025, 8, 20))
    future_price = provider.get_price_on_or_before("600519", as_of=date(2025, 11, 5))
    assert future_price is not None

    with pytest.raises(ComputeSkillError, match="晚于研究时点"):
        compute_pe_ttm_pit(stmts, future_price, as_of=date(2025, 8, 20))
