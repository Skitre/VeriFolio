"""结构化数据源适配器抽象接口。

设计目的：
- 默认使用 MockMarketDataProvider 离线确定性运行（单测与评测不联网）。
- 后续无缝接入 AKShare / 巨潮公告适配器时，只需实现本 Protocol，所有计算 Skill 和核验逻辑零改动。
- 接口层强制传入 `as_of: date`，把时点纪律（防前视偏差）固化在数据访问边界。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from zhengyan.core.models import (
    AnnouncementRecord,
    EvidenceCard,
    FinancialStatementRecord,
    PricePoint,
)


class MarketDataProvider(Protocol):
    """结构化行情、财报与公告统一适配器协议。"""

    def get_financial_statements(
        self,
        symbol: str,
        as_of: date,
    ) -> list[FinancialStatementRecord]:
        """仅返回 `disclosed_at <= as_of` 的已披露财报（严格按披露日过滤）。"""
        ...

    def get_price_on_or_before(
        self,
        symbol: str,
        as_of: date,
    ) -> PricePoint | None:
        """返回 `trade_date <= as_of` 的最近一个交易日行情与股本。"""
        ...

    def get_announcements(
        self,
        symbol: str,
        as_of: date,
        lookback_days: int = 120,
    ) -> list[AnnouncementRecord]:
        """返回研究时点之前窗口期内已披露的巨潮公告。"""
        ...

    def to_evidence_cards(
        self,
        symbol: str,
        as_of: date,
        lookback_days: int = 120,
    ) -> list[EvidenceCard]:
        """将已通过时点过滤的结构化数据转为 A 级 EvidenceCard。"""
        ...
