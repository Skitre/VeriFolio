"""离线确定性数据源（MockMarketDataProvider）。

内置贵州茅台（600519）与宁德时代（300750）多报告期结构化财报、日度行情与公告样例，
特意构造了以下真实金融工程边界场景：
1. 报告期截止日（如 2024-12-31）与实际披露日（2025-04-03）相差数月，用于测试防前视偏差。
2. 贵州茅台 2025H1 合同负债同比约 -42.0%，呼应文档中深研质询与经验卡 L-007 真实案例。
3. 累计值（Q1 / H1 / Q3 / FY）齐备，支持单季推算与按披露日对齐的 PE(TTM) 计算。
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from zhengyan.core.enums import (
    AccountingBasis,
    EvidenceGrade,
    EvidenceSourceType,
    MetricUnit,
    PITGuarantee,
)
from zhengyan.core.models import (
    AnnouncementRecord,
    EvidenceCard,
    FinancialStatementRecord,
    MetricEntry,
    PricePoint,
)


def _metric(
    name: str,
    display_name: str,
    value: str,
    unit: MetricUnit,
    period: str,
    period_end: date,
    basis: AccountingBasis = AccountingBasis.CUMULATIVE,
) -> MetricEntry:
    return MetricEntry(
        name=name,
        display_name=display_name,
        value=Decimal(value),
        unit=unit,
        period=period,
        period_end=period_end,
        basis=basis,
    )


def _build_default_statements() -> list[FinancialStatementRecord]:
    return [
        # ==================== 贵州茅台 600519 ====================
        FinancialStatementRecord(
            symbol="600519",
            company_name="贵州茅台",
            period="2024Q1",
            period_end=date(2024, 3, 31),
            disclosed_at=date(2024, 4, 27),
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "457.76", MetricUnit.HUNDRED_MILLION_CNY, "2024Q1", date(2024, 3, 31)),
                "net_profit": _metric("net_profit", "归母净利润", "240.65", MetricUnit.HUNDRED_MILLION_CNY, "2024Q1", date(2024, 3, 31)),
                "gross_profit": _metric("gross_profit", "毛利", "424.16", MetricUnit.HUNDRED_MILLION_CNY, "2024Q1", date(2024, 3, 31)),
                "ocf": _metric("ocf", "经营活动现金流净额", "91.87", MetricUnit.HUNDRED_MILLION_CNY, "2024Q1", date(2024, 3, 31)),
            },
        ),
        FinancialStatementRecord(
            symbol="600519",
            company_name="贵州茅台",
            period="2024H1",
            period_end=date(2024, 6, 30),
            disclosed_at=date(2024, 8, 9),
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "819.31", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
                "net_profit": _metric("net_profit", "归母净利润", "416.96", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
                "gross_profit": _metric("gross_profit", "毛利", "752.13", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
                "gross_margin": _metric("gross_margin", "毛利率", "91.80", MetricUnit.PERCENT, "2024H1", date(2024, 6, 30)),
                "ocf": _metric("ocf", "经营活动现金流净额", "366.22", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
                "contract_liabilities": _metric("contract_liabilities", "合同负债", "99.93", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
            },
        ),
        FinancialStatementRecord(
            symbol="600519",
            company_name="贵州茅台",
            period="2024Q3",
            period_end=date(2024, 9, 30),
            disclosed_at=date(2024, 10, 26),
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "1207.76", MetricUnit.HUNDRED_MILLION_CNY, "2024Q3", date(2024, 9, 30)),
                "net_profit": _metric("net_profit", "归母净利润", "608.28", MetricUnit.HUNDRED_MILLION_CNY, "2024Q3", date(2024, 9, 30)),
                "gross_profit": _metric("gross_profit", "毛利", "1105.50", MetricUnit.HUNDRED_MILLION_CNY, "2024Q3", date(2024, 9, 30)),
                "ocf": _metric("ocf", "经营活动现金流净额", "444.21", MetricUnit.HUNDRED_MILLION_CNY, "2024Q3", date(2024, 9, 30)),
            },
        ),
        FinancialStatementRecord(
            symbol="600519",
            company_name="贵州茅台",
            period="2024FY",
            period_end=date(2024, 12, 31),
            disclosed_at=date(2025, 4, 3),  # 注意：披露日比报告期截止日晚 3 个月！
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "1708.99", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
                "net_profit": _metric("net_profit", "归母净利润", "862.28", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
                "gross_profit": _metric("gross_profit", "毛利", "1571.07", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
                "ocf": _metric("ocf", "经营活动现金流净额", "924.64", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
                "equity": _metric("equity", "归母净资产", "2331.06", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
            },
        ),
        FinancialStatementRecord(
            symbol="600519",
            company_name="贵州茅台",
            period="2025H1",
            period_end=date(2025, 6, 30),
            disclosed_at=date(2025, 8, 13),
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "893.89", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
                "net_profit": _metric("net_profit", "归母净利润", "454.03", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
                "gross_profit": _metric("gross_profit", "毛利", "816.12", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
                "gross_margin": _metric("gross_margin", "毛利率", "91.30", MetricUnit.PERCENT, "2025H1", date(2025, 6, 30)),
                "ocf": _metric("ocf", "经营活动现金流净额", "398.15", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
                "contract_liabilities": _metric("contract_liabilities", "合同负债", "57.96", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
            },
        ),
        # 扣非口径记录，用于测试同口径校验（不能把 NON_GAAP 和 CUMULATIVE/GAAP 直接相除算同比）
        FinancialStatementRecord(
            symbol="600519",
            company_name="贵州茅台",
            period="2025H1_NON_GAAP",
            period_end=date(2025, 6, 30),
            disclosed_at=date(2025, 8, 13),
            basis=AccountingBasis.NON_GAAP,
            metrics={
                "net_profit": _metric(
                    "net_profit", "扣非归母净利润", "453.10",
                    MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30),
                    basis=AccountingBasis.NON_GAAP,
                ),
            },
        ),
        FinancialStatementRecord(
            symbol="600519",
            company_name="贵州茅台",
            period="2025Q3",
            period_end=date(2025, 9, 30),
            disclosed_at=date(2025, 10, 29),
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "1284.54", MetricUnit.HUNDRED_MILLION_CNY, "2025Q3", date(2025, 9, 30)),
                "net_profit": _metric("net_profit", "归母净利润", "646.27", MetricUnit.HUNDRED_MILLION_CNY, "2025Q3", date(2025, 9, 30)),
                "gross_profit": _metric("gross_profit", "毛利", "1172.78", MetricUnit.HUNDRED_MILLION_CNY, "2025Q3", date(2025, 9, 30)),
                "ocf": _metric("ocf", "经营活动现金流净额", "512.40", MetricUnit.HUNDRED_MILLION_CNY, "2025Q3", date(2025, 9, 30)),
            },
        ),
        # ==================== 宁德时代 300750 ====================
        FinancialStatementRecord(
            symbol="300750",
            company_name="宁德时代",
            period="2024H1",
            period_end=date(2024, 6, 30),
            disclosed_at=date(2024, 7, 27),
            basis=AccountingBasis.CUMULATIVE,
            pit_guarantee=PITGuarantee.WEAK_RESTATED,
            metrics={
                "revenue": _metric("revenue", "营业收入", "1667.67", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
                "net_profit": _metric("net_profit", "归母净利润", "228.65", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
                "gross_profit": _metric("gross_profit", "毛利", "442.45", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
                "ocf": _metric("ocf", "经营活动现金流净额", "447.09", MetricUnit.HUNDRED_MILLION_CNY, "2024H1", date(2024, 6, 30)),
            },
        ),
        FinancialStatementRecord(
            symbol="300750",
            company_name="宁德时代",
            period="2024FY",
            period_end=date(2024, 12, 31),
            disclosed_at=date(2025, 3, 15),
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "3620.13", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
                "net_profit": _metric("net_profit", "归母净利润", "507.45", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
                "gross_profit": _metric("gross_profit", "毛利", "884.76", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
                "ocf": _metric("ocf", "经营活动现金流净额", "969.90", MetricUnit.HUNDRED_MILLION_CNY, "2024FY", date(2024, 12, 31)),
            },
        ),
        FinancialStatementRecord(
            symbol="300750",
            company_name="宁德时代",
            period="2025H1",
            period_end=date(2025, 6, 30),
            disclosed_at=date(2025, 7, 31),
            basis=AccountingBasis.CUMULATIVE,
            metrics={
                "revenue": _metric("revenue", "营业收入", "1788.86", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
                "net_profit": _metric("net_profit", "归母净利润", "304.85", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
                "gross_profit": _metric("gross_profit", "毛利", "478.52", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
                "ocf": _metric("ocf", "经营活动现金流净额", "520.10", MetricUnit.HUNDRED_MILLION_CNY, "2025H1", date(2025, 6, 30)),
            },
        ),
    ]


def _build_default_prices() -> list[PricePoint]:
    return [
        # 茅台在 2024FY 披露前（2025-03-20）与披露后（2025-08-20、2025-11-05）的行情
        PricePoint(
            symbol="600519",
            trade_date=date(2025, 3, 20),
            close_price=Decimal("1620.00"),
            total_shares=Decimal("1256197800"),
            market_cap_yi=Decimal("20350.40"),
        ),
        PricePoint(
            symbol="600519",
            trade_date=date(2025, 8, 20),
            close_price=Decimal("1460.00"),
            total_shares=Decimal("1256197800"),
            market_cap_yi=Decimal("18340.49"),
        ),
        PricePoint(
            symbol="600519",
            trade_date=date(2025, 11, 5),
            close_price=Decimal("1485.00"),
            total_shares=Decimal("1256197800"),
            market_cap_yi=Decimal("18654.54"),
        ),
        PricePoint(
            symbol="300750",
            trade_date=date(2025, 4, 1),
            close_price=Decimal("220.00"),
            total_shares=Decimal("4403466458"),
            market_cap_yi=Decimal("9687.63"),
        ),
        PricePoint(
            symbol="300750",
            trade_date=date(2025, 8, 20),
            close_price=Decimal("260.00"),
            total_shares=Decimal("4403466458"),
            market_cap_yi=Decimal("11449.01"),
        ),
    ]


def _build_default_announcements() -> list[AnnouncementRecord]:
    return [
        AnnouncementRecord(
            announcement_id="ANN-600519-20250813",
            symbol="600519",
            title="贵州茅台2025年半年度报告摘要",
            summary="2025年上半年公司实现营业收入893.89亿元，归母净利润454.03亿元；公司优化部分直营与经销商打款节奏。",
            disclosed_at=date(2025, 8, 13),
            url="https://www.cninfo.com.cn/new/disclosure/detail?stockCode=600519&ann=20250813",
        ),
        AnnouncementRecord(
            announcement_id="ANN-600519-20251029",
            symbol="600519",
            title="贵州茅台2025年第三季度报告",
            summary="2025年前三季度营业收入1284.54亿元，归母净利润646.27亿元。",
            disclosed_at=date(2025, 10, 29),
            url="https://www.cninfo.com.cn/new/disclosure/detail?stockCode=600519&ann=20251029",
        ),
        AnnouncementRecord(
            announcement_id="ANN-300750-20250731",
            symbol="300750",
            title="宁德时代2025年半年度报告摘要",
            summary="2025年上半年营业收入1788.86亿元，归母净利润304.85亿元。",
            disclosed_at=date(2025, 7, 31),
            url="https://www.cninfo.com.cn/new/disclosure/detail?stockCode=300750&ann=20250731",
        ),
    ]


class MockMarketDataProvider:
    """确定性离线数据适配器（实现 MarketDataProvider 协议）。"""

    def __init__(
        self,
        statements: list[FinancialStatementRecord] | None = None,
        prices: list[PricePoint] | None = None,
        announcements: list[AnnouncementRecord] | None = None,
    ) -> None:
        self._statements = statements if statements is not None else _build_default_statements()
        self._prices = prices if prices is not None else _build_default_prices()
        self._announcements = announcements if announcements is not None else _build_default_announcements()

    def get_financial_statements(
        self,
        symbol: str,
        as_of: date,
    ) -> list[FinancialStatementRecord]:
        """严格按 disclosed_at <= as_of 过滤，彻底阻断前视偏差。"""
        matched = [
            s for s in self._statements
            if s.symbol == symbol and s.disclosed_at <= as_of
        ]
        return sorted(matched, key=lambda s: (s.period_end, s.disclosed_at))

    def get_statement_by_period(
        self,
        symbol: str,
        period: str,
        as_of: date,
        basis: AccountingBasis = AccountingBasis.CUMULATIVE,
    ) -> FinancialStatementRecord | None:
        for stmt in self.get_financial_statements(symbol, as_of):
            if stmt.period == period and stmt.basis == basis:
                return stmt
        return None

    def get_price_on_or_before(
        self,
        symbol: str,
        as_of: date,
    ) -> PricePoint | None:
        candidates = [
            p for p in self._prices
            if p.symbol == symbol and p.trade_date <= as_of
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.trade_date)

    def get_announcements(
        self,
        symbol: str,
        as_of: date,
        lookback_days: int = 120,
    ) -> list[AnnouncementRecord]:
        cutoff = as_of - timedelta(days=lookback_days)
        matched = [
            a for a in self._announcements
            if a.symbol == symbol and cutoff <= a.disclosed_at <= as_of
        ]
        return sorted(matched, key=lambda a: a.disclosed_at)

    def to_evidence_cards(
        self,
        symbol: str,
        as_of: date,
        lookback_days: int = 120,
    ) -> list[EvidenceCard]:
        """生成 A 级结构化证据卡（财报 + 行情 + 巨潮公告）。"""
        cards: list[EvidenceCard] = []

        for stmt in self.get_financial_statements(symbol, as_of):
            cards.append(
                EvidenceCard(
                    evidence_id=f"EV-FIN-{stmt.symbol}-{stmt.period}",
                    grade=EvidenceGrade.A,
                    source_type=EvidenceSourceType.FINANCIAL_REPORT,
                    source_name=stmt.source_name,
                    title=f"{stmt.company_name}({stmt.symbol}) {stmt.period} 财务报表",
                    symbol=stmt.symbol,
                    published_at=stmt.disclosed_at,
                    period=stmt.period,
                    pit_guarantee=stmt.pit_guarantee,
                    metrics=list(stmt.metrics.values()),
                )
            )

        price = self.get_price_on_or_before(symbol, as_of)
        if price is not None:
            cards.append(
                EvidenceCard(
                    evidence_id=f"EV-MKT-{price.symbol}-{price.trade_date.isoformat()}",
                    grade=EvidenceGrade.A,
                    source_type=EvidenceSourceType.MARKET_QUOTE,
                    source_name=price.source_name,
                    title=f"{price.symbol} {price.trade_date.isoformat()} 收盘行情",
                    symbol=price.symbol,
                    published_at=price.trade_date,
                    metrics=[
                        MetricEntry(
                            name="close_price",
                            display_name="收盘价",
                            value=price.close_price,
                            unit=MetricUnit.CNY,
                            period_end=price.trade_date,
                        ),
                        MetricEntry(
                            name="market_cap",
                            display_name="总市值",
                            value=price.market_cap_yi,
                            unit=MetricUnit.HUNDRED_MILLION_CNY,
                            period_end=price.trade_date,
                        ),
                    ],
                )
            )

        for ann in self.get_announcements(symbol, as_of, lookback_days=lookback_days):
            cards.append(
                EvidenceCard(
                    evidence_id=f"EV-{ann.announcement_id}",
                    grade=EvidenceGrade.A,
                    source_type=EvidenceSourceType.ANNOUNCEMENT,
                    source_name=ann.source_name,
                    title=ann.title,
                    symbol=ann.symbol,
                    published_at=ann.disclosed_at,
                    url=ann.url,
                    authority=5,
                    has_full_snapshot=True,
                    excerpt=ann.summary,
                )
            )

        return cards
