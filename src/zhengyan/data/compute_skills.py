"""确定性金融计算 Skill（全链路使用 Decimal，杜绝浮点误差）。

核心要点（对应文档第 1、6、10、30 题）：
1. 全部运算使用 decimal.Decimal，不使用 float。
2. 同口径校验：对比两期或相减推算单季时，强制检查会计口径（AccountingBasis）与单位（MetricUnit）一致。
3. 溯源链：每张计算卡记录 `derived_from`，并取所有输入卡中最晚的 `published_at` 作为计算卡时点。
4. PE(TTM) 严格按披露日对齐：仅使用 `disclosed_at <= as_of` 的财报推算最近四个季度利润。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from zhengyan.core.enums import (
    AccountingBasis,
    EvidenceGrade,
    EvidenceSourceType,
    MetricUnit,
)
from zhengyan.core.models import (
    EvidenceCard,
    FinancialStatementRecord,
    MetricEntry,
    PricePoint,
)


class ComputeSkillError(ValueError):
    """计算 Skill 通用异常。"""


class AccountingBasisMismatchError(ComputeSkillError):
    """会计口径或单位不一致，禁止跨口径计算。"""


def half_ulp_tolerance(written_number: str) -> Decimal:
    """按书写精度的半个单位计算允许误差（对应文档第 1.2 题与第 7 题）。

    示例：
    - "12.3"  -> 小数位 1 -> 单位 0.1  -> 半单位容差 0.05
    - "42"    -> 小数位 0 -> 单位 1    -> 半单位容差 0.5
    - "12.34" -> 小数位 2 -> 单位 0.01 -> 半单位容差 0.005
    """
    cleaned = written_number.strip().replace(",", "").lstrip("+-")
    if "." in cleaned:
        decimals = len(cleaned.split(".", maxsplit=1)[1])
    else:
        decimals = 0
    return Decimal("0.5") * (Decimal("10") ** (-decimals))


def _quantize(value: Decimal, places: int = 4) -> Decimal:
    quantum = Decimal("10") ** (-places)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _find_metric(card: EvidenceCard, metric_name: str) -> MetricEntry:
    for entry in card.metrics:
        if entry.name == metric_name:
            return entry
    raise ComputeSkillError(f"证据卡 {card.evidence_id} 中未找到指标 {metric_name!r}")


def _latest_date(*dates: date | None) -> date | None:
    valid = [d for d in dates if d is not None]
    return max(valid) if valid else None


def compute_yoy_growth(
    current_card: EvidenceCard,
    prior_card: EvidenceCard,
    metric_name: str,
    *,
    as_of: date | None = None,
    evidence_id: str | None = None,
) -> EvidenceCard:
    """同口径同比增长率（百分比 %）：(current - prior) / abs(prior) * 100。"""
    curr = _find_metric(current_card, metric_name)
    prev = _find_metric(prior_card, metric_name)

    if curr.basis != prev.basis:
        raise AccountingBasisMismatchError(
            f"口径不一致：{curr.period} 为 {curr.basis}，{prev.period} 为 {prev.basis}"
        )
    if curr.unit != prev.unit:
        raise AccountingBasisMismatchError(
            f"单位不一致：{curr.unit} vs {prev.unit}"
        )
    if prev.value == 0:
        raise ComputeSkillError("基期数值为 0，无法计算同比增长率")

    if as_of is not None:
        for card in (current_card, prior_card):
            if card.published_at and card.published_at > as_of:
                raise ComputeSkillError(
                    f"证据 {card.evidence_id} 披露日 {card.published_at} 晚于研究时点 {as_of}"
                )

    growth_pct = _quantize(((curr.value - prev.value) / abs(prev.value)) * Decimal("100"), places=2)
    out_id = evidence_id or f"EV-CMP-YOY-{current_card.symbol}-{metric_name}-{curr.period}"

    return EvidenceCard(
        evidence_id=out_id,
        grade=EvidenceGrade.A,
        source_type=EvidenceSourceType.COMPUTE,
        source_name="compute.yoy_growth",
        title=f"{current_card.symbol} {curr.display_name or metric_name} {curr.period} 同比增速",
        symbol=current_card.symbol,
        published_at=_latest_date(current_card.published_at, prior_card.published_at),
        period=curr.period,
        derived_from=[current_card.evidence_id, prior_card.evidence_id],
        metrics=[
            MetricEntry(
                name=f"{metric_name}_yoy",
                display_name=f"{curr.display_name or metric_name}同比增速",
                value=growth_pct,
                unit=MetricUnit.PERCENT,
                period=curr.period,
                period_end=curr.period_end,
                basis=curr.basis,
            )
        ],
    )


def compute_single_quarter(
    cumulative_card: EvidenceCard,
    previous_cumulative_card: EvidenceCard,
    metric_name: str,
    *,
    quarter_label: str,
    evidence_id: str | None = None,
) -> EvidenceCard:
    """单季推算：本期累计值 - 上期累计值（如 Q3 累计 - H1 累计 = Q3 单季）。"""
    curr = _find_metric(cumulative_card, metric_name)
    prev = _find_metric(previous_cumulative_card, metric_name)

    if curr.basis != AccountingBasis.CUMULATIVE or prev.basis != AccountingBasis.CUMULATIVE:
        raise AccountingBasisMismatchError("单季推算要求两期输入均为累计口径（CUMULATIVE）")
    if curr.unit != prev.unit:
        raise AccountingBasisMismatchError(f"单位不一致：{curr.unit} vs {prev.unit}")

    single_val = _quantize(curr.value - prev.value, places=2)
    out_id = evidence_id or f"EV-CMP-SQ-{cumulative_card.symbol}-{metric_name}-{quarter_label}"

    return EvidenceCard(
        evidence_id=out_id,
        grade=EvidenceGrade.A,
        source_type=EvidenceSourceType.COMPUTE,
        source_name="compute.single_quarter",
        title=f"{cumulative_card.symbol} {quarter_label} 单季{curr.display_name or metric_name}",
        symbol=cumulative_card.symbol,
        published_at=_latest_date(cumulative_card.published_at, previous_cumulative_card.published_at),
        period=quarter_label,
        derived_from=[cumulative_card.evidence_id, previous_cumulative_card.evidence_id],
        metrics=[
            MetricEntry(
                name=metric_name,
                display_name=f"单季{curr.display_name or metric_name}",
                value=single_val,
                unit=curr.unit,
                period=quarter_label,
                period_end=curr.period_end,
                basis=AccountingBasis.SINGLE_QUARTER,
            )
        ],
    )


def compute_ratio(
    card: EvidenceCard,
    numerator_metric: str,
    denominator_metric: str,
    *,
    output_name: str,
    output_display_name: str,
    output_unit: MetricUnit = MetricUnit.PERCENT,
    evidence_id: str | None = None,
) -> EvidenceCard:
    """比率计算（毛利率、净利率、净现比、ROE 等）。

    当 `output_unit == MetricUnit.PERCENT` 时乘以 100，保留 2 位小数；
    当 `output_unit == MetricUnit.MULTIPLE` 时保留 2 位小数。
    """
    num = _find_metric(card, numerator_metric)
    den = _find_metric(card, denominator_metric)

    if num.unit != den.unit:
        raise AccountingBasisMismatchError(f"分子分母单位不一致：{num.unit} vs {den.unit}")
    if den.value == 0:
        raise ComputeSkillError(f"分母 {denominator_metric!r} 为 0，无法计算比率")

    raw_ratio = num.value / den.value
    if output_unit == MetricUnit.PERCENT:
        result = _quantize(raw_ratio * Decimal("100"), places=2)
    else:
        result = _quantize(raw_ratio, places=2)

    out_id = evidence_id or f"EV-CMP-RATIO-{card.symbol}-{output_name}-{card.period}"

    return EvidenceCard(
        evidence_id=out_id,
        grade=EvidenceGrade.A,
        source_type=EvidenceSourceType.COMPUTE,
        source_name="compute.ratio",
        title=f"{card.symbol} {card.period} {output_display_name}",
        symbol=card.symbol,
        published_at=card.published_at,
        period=card.period,
        derived_from=[card.evidence_id],
        metrics=[
            MetricEntry(
                name=output_name,
                display_name=output_display_name,
                value=result,
                unit=output_unit,
                period=card.period,
                period_end=num.period_end,
                basis=num.basis,
            )
        ],
    )


def compute_pe_ttm_pit(
    statements: list[FinancialStatementRecord],
    price: PricePoint,
    as_of: date,
    *,
    evidence_id: str | None = None,
) -> EvidenceCard:
    """按披露日逐日对齐计算 PE(TTM)（对应文档第 10 题）。

    规则：
    1. 严格过滤 `disclosed_at <= as_of` 且 `basis == CUMULATIVE` 的财报。
    2. 找到最新一份已披露财报 `latest`：
       - 若 `latest.period` 以 `FY` 结尾，直接用该年报 `net_profit` 作为 TTM 利润；
       - 否则需要 `上一完整年报(prev_fy)` + `本期累计(latest)` - `上年同期累计(prior_same_period)`；
       - 若缺失上一完整年报或上年同期，抛出明确错误，绝不猜测或外推！
    3. PE(TTM) = 当日总市值（亿元） / TTM 归母净利润（亿元）。
    """
    if price.trade_date > as_of:
        raise ComputeSkillError(
            f"行情日期 {price.trade_date} 晚于研究时点 {as_of}，禁止使用未来行情"
        )

    available = [
        s for s in statements
        if s.symbol == price.symbol
        and s.disclosed_at <= as_of
        and s.basis == AccountingBasis.CUMULATIVE
        and "net_profit" in s.metrics
    ]
    if not available:
        raise ComputeSkillError(f"在 {as_of} 之前未找到 {price.symbol} 已披露的累计口径利润表")

    by_period = {s.period: s for s in available}
    latest = max(available, key=lambda s: (s.period_end, s.disclosed_at))
    used_stmts: list[FinancialStatementRecord] = [latest]

    if latest.period.endswith("FY"):
        ttm_profit = latest.metrics["net_profit"].value
    else:
        year = int(latest.period[:4])
        suffix = latest.period[4:]  # Q1 / H1 / Q3
        prev_fy_key = f"{year - 1}FY"
        prior_same_key = f"{year - 1}{suffix}"
        if prev_fy_key not in by_period or prior_same_key not in by_period:
            raise ComputeSkillError(
                f"截至 {as_of} 缺少推算 TTM 所需的 {prev_fy_key} 或 {prior_same_key} 已披露财报"
            )
        prev_fy = by_period[prev_fy_key]
        prior_same = by_period[prior_same_key]
        used_stmts.extend([prev_fy, prior_same])
        ttm_profit = (
            prev_fy.metrics["net_profit"].value
            + latest.metrics["net_profit"].value
            - prior_same.metrics["net_profit"].value
        )

    if ttm_profit <= 0:
        raise ComputeSkillError(f"TTM 归母净利润为 {ttm_profit}，PE(TTM) 无意义")

    pe_ttm = _quantize(price.market_cap_yi / ttm_profit, places=2)
    derived_ids = [f"EV-FIN-{s.symbol}-{s.period}" for s in used_stmts]
    derived_ids.append(f"EV-MKT-{price.symbol}-{price.trade_date.isoformat()}")

    out_id = evidence_id or f"EV-CMP-PETTM-{price.symbol}-{as_of.isoformat()}"
    return EvidenceCard(
        evidence_id=out_id,
        grade=EvidenceGrade.A,
        source_type=EvidenceSourceType.COMPUTE,
        source_name="compute.pe_ttm_pit",
        title=f"{price.symbol} 截至 {as_of.isoformat()} 按披露日对齐 PE(TTM)",
        symbol=price.symbol,
        published_at=_latest_date(price.trade_date, *(s.disclosed_at for s in used_stmts)),
        period=latest.period,
        derived_from=derived_ids,
        metrics=[
            MetricEntry(
                name="ttm_net_profit",
                display_name="TTM归母净利润",
                value=_quantize(ttm_profit, places=2),
                unit=MetricUnit.HUNDRED_MILLION_CNY,
                period=latest.period,
                period_end=latest.period_end,
            ),
            MetricEntry(
                name="pe_ttm",
                display_name="市盈率PE(TTM)",
                value=pe_ttm,
                unit=MetricUnit.MULTIPLE,
                period=latest.period,
                period_end=price.trade_date,
            ),
        ],
    )
