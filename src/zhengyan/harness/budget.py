"""预算追踪器（对应文档第 11 & 28 题：超限只降研究深度，绝不跳过核验）。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from zhengyan.core.enums import ResearchTier
from zhengyan.core.models import BudgetSnapshot


class BudgetExceededError(RuntimeError):
    """预算耗尽异常（上层应优雅降级为部分结论，不得跳过核验）。"""


@dataclass
class BudgetTracker:
    tier: ResearchTier
    max_tokens: int
    max_searches: int
    max_fetches: int
    max_duration_seconds: int
    clock: Callable[[], float] = field(default=time.monotonic)
    started_at: float = field(init=False)

    tokens_used: int = 0
    searches_used: int = 0
    fetches_used: int = 0

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def check_time_budget(self) -> None:
        if self.elapsed_seconds > self.max_duration_seconds:
            raise BudgetExceededError(
                f"总耗时已超过上限（{self.elapsed_seconds:.1f}s > {self.max_duration_seconds}s）"
            )

    def can_search(self) -> bool:
        return self.searches_used < self.max_searches and self.elapsed_seconds <= self.max_duration_seconds

    def can_fetch(self) -> bool:
        return self.fetches_used < self.max_fetches and self.elapsed_seconds <= self.max_duration_seconds

    def can_use_tokens(self, estimated_tokens: int = 1) -> bool:
        return (
            self.tokens_used + estimated_tokens <= self.max_tokens
            and self.elapsed_seconds <= self.max_duration_seconds
        )

    def consume_search(self, count: int = 1) -> None:
        self.check_time_budget()
        if self.searches_used + count > self.max_searches:
            raise BudgetExceededError(
                f"搜索预算不足（已用 {self.searches_used}/{self.max_searches}）"
            )
        self.searches_used += count

    def consume_fetch(self, count: int = 1) -> None:
        self.check_time_budget()
        if self.fetches_used + count > self.max_fetches:
            raise BudgetExceededError(
                f"抓取预算不足（已用 {self.fetches_used}/{self.max_fetches}）"
            )
        self.fetches_used += count

    def consume_tokens(self, tokens: int) -> None:
        self.check_time_budget()
        if self.tokens_used + tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Token 预算不足（已用 {self.tokens_used} + {tokens} > {self.max_tokens}）"
            )
        self.tokens_used += tokens

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            tier=self.tier,
            tokens_used=self.tokens_used,
            max_tokens=self.max_tokens,
            searches_used=self.searches_used,
            max_searches=self.max_searches,
            fetches_used=self.fetches_used,
            max_fetches=self.max_fetches,
            elapsed_seconds=round(self.elapsed_seconds, 2),
            max_duration_seconds=self.max_duration_seconds,
        )
