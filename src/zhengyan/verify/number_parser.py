"""中文金融主张数字与单位抽取器（对应文档第 7 题）。

支持单位：亿元、万元、元、%、个百分点、倍。
关键细节：
1. 只抽取携带明确金融单位的数字，避免把股票代码（600519）、年份（2025年）误判为待核验指标。
2. 结合前置语义词识别符号方向（如“下降 42%”视为 -42%，防止把下跌写成上涨的符号反转幻觉）。
3. 按书写精度自动计算半单位容差（half-ULP）：写“12.3%”允许 ±0.05，写“42%”允许 ±0.5。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from zhengyan.core.enums import MetricUnit
from zhengyan.data.compute_skills import half_ulp_tolerance

# 按单位长度降序排列，防止 “亿元” 被拆成 “亿”+“元” 或 “个百分点” 漏匹配
_NUM_UNIT_RE = re.compile(
    r"(?P<sign>[+-]?)"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*"
    r"(?P<unit>个百分点|亿元|万元|元|%|倍)"
)

_NEGATIVE_CUE_RE = re.compile(r"(?:下降|减少|下滑|下跌|同比降|环比降|收窄|缩减|亏损|负增长|降幅|跌去)\s*(?:约|近|了)?$")
_POSITIVE_CUE_RE = re.compile(r"(?:上升|增长|提升|提高|扩大|增加|上涨|同比增|环比增|涨幅)\s*(?:约|近|了)?$")

_UNIT_MAP: dict[str, MetricUnit] = {
    "亿元": MetricUnit.HUNDRED_MILLION_CNY,
    "万元": MetricUnit.TEN_THOUSAND_CNY,
    "元": MetricUnit.CNY,
    "%": MetricUnit.PERCENT,
    "个百分点": MetricUnit.PERCENTAGE_POINT,
    "倍": MetricUnit.MULTIPLE,
}

AMOUNT_UNITS = frozenset({
    MetricUnit.CNY,
    MetricUnit.TEN_THOUSAND_CNY,
    MetricUnit.HUNDRED_MILLION_CNY,
})


@dataclass(frozen=True)
class ExtractedNumber:
    raw_text: str
    value: Decimal
    unit: MetricUnit
    tolerance: Decimal
    direction: str  # "positive" | "negative" | "neutral"


def to_base_cny(value: Decimal, unit: MetricUnit) -> Decimal:
    """将金额类单位统一换算为元（精确 Decimal，不丢精度）。"""
    if unit == MetricUnit.CNY:
        return value
    if unit == MetricUnit.TEN_THOUSAND_CNY:
        return value * Decimal("10000")
    if unit == MetricUnit.HUNDRED_MILLION_CNY:
        return value * Decimal("100000000")
    raise ValueError(f"非金额单位无法换算为元：{unit}")


def extract_financial_numbers(text: str) -> list[ExtractedNumber]:
    """从主张或网页正文中抽取全部带金融单位的数字及其半单位容差。"""
    results: list[ExtractedNumber] = []
    for match in _NUM_UNIT_RE.finditer(text):
        sign_str = match.group("sign")
        num_str = match.group("num")
        unit_str = match.group("unit")
        unit = _UNIT_MAP[unit_str]

        cleaned_num = num_str.replace(",", "")
        magnitude = Decimal(cleaned_num)
        tol = half_ulp_tolerance(cleaned_num)

        prefix = text[max(0, match.start() - 12) : match.start()]
        if sign_str == "-":
            value = -magnitude
            direction = "negative"
        elif sign_str == "+":
            value = magnitude
            direction = "positive"
        elif _NEGATIVE_CUE_RE.search(prefix):
            value = -magnitude
            direction = "negative"
        elif _POSITIVE_CUE_RE.search(prefix):
            value = magnitude
            direction = "positive"
        else:
            value = magnitude
            direction = "neutral"

        results.append(
            ExtractedNumber(
                raw_text=match.group(0),
                value=value,
                unit=unit,
                tolerance=tol,
                direction=direction,
            )
        )
    return results
