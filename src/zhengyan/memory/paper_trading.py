"""策略卡、前向模拟盘（严禁历史回填）与纯代码晋级门控/回滚（对应文档第 32 题）。

核心原则：
1. 策略卡显式声明“纳入 / 移出观察组合”的确定性规则，所有字段来自档案 A 级指标与状态：
   PE 分位、波动率、已核验主张数、待复核提醒数。
2. 前向模拟盘从 start_date 起逐交易日向前推进，严禁回填历史（BackfillForbiddenError）；
   只用当天及以前可得的档案版本与行情，虚拟资金全用 Decimal、等权、整手（100 股整数倍）。
3. 晋级门控纯代码执行 5 条硬标准：
   - 前向模拟 >= 10 个交易日
   - 零未来数据（每笔交易引用的档案 as_of <= trade_date）
   - 最大回撤不超上限
   - 每笔交易均可追溯到 dossier_id 与版本号
   - 同期收益不落后于在用策略版本超过 2%
   通过门控后仍必须人工显式批准才能晋级，且支持一键回滚到上一在用版本！
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_DOWN
from typing import Literal

from pydantic import BaseModel, Field


class BackfillForbiddenError(ValueError):
    """试图回填或改写历史交易日时抛出（前向模拟盘严禁历史回测/回填）。"""


class PromotionDeniedError(PermissionError):
    """未通过晋级门控或缺少人工显式批准时抛出。"""


class StrategyCard(BaseModel):
    """显式规则策略卡（决定纳入 / 移出观察组合）。"""

    strategy_id: str
    version: str
    description: str = ""
    max_pe_percentile: Decimal = Field(default=Decimal("60"), description="纳入最大 PE 历史分位（%）")
    max_volatility_pct: Decimal = Field(default=Decimal("35"), description="纳入最大年化波动率（%）")
    min_verified_claims: int = Field(default=3, description="纳入最少已核验主张数")
    max_pending_alerts: int = Field(default=0, description="允许最多待复核提醒数")
    max_drawdown_limit: Decimal = Field(default=Decimal("0.15"), description="晋级允许最大回撤上限（如 0.15 = 15%）")


class DailyUniverseSnapshot(BaseModel):
    """前向模拟盘单个交易日可得的客观快照（仅含当日及以前已披露信息）。"""

    symbol: str
    trade_date: date
    close_price: Decimal
    dossier_id: str
    dossier_version: int
    dossier_as_of: date
    pe_percentile: Decimal
    volatility_pct: Decimal
    verified_claims_count: int
    pending_alerts_count: int


class PaperTrade(BaseModel):
    """前向模拟盘单笔可追溯虚拟成交记录（整手 100 股）。"""

    trade_date: date
    symbol: str
    side: Literal["buy", "sell"]
    shares: int
    price: Decimal
    amount: Decimal
    dossier_id: str
    dossier_version: int
    dossier_as_of: date
    reason: str


class DailyNavPoint(BaseModel):
    trade_date: date
    cash: Decimal
    holdings_value: Decimal
    total_nav: Decimal
    cumulative_return: Decimal
    drawdown: Decimal


class PromotionGateReport(BaseModel):
    """纯代码 5 条晋级门控判定结果。"""

    candidate_version: str
    active_version: str
    trading_days: int
    has_future_leak: bool
    max_drawdown: Decimal
    all_trades_traceable: bool
    candidate_return: Decimal
    active_return: Decimal
    return_diff: Decimal
    passed: bool
    failed_checks: list[str] = Field(default_factory=list)


class ForwardPaperSimulator:
    """等权、整手（100 股）、Decimal 精确记账的前向模拟盘引擎。"""

    LOT_SIZE = 100

    def __init__(
        self,
        *,
        strategy: StrategyCard,
        start_date: date,
        initial_cash: Decimal = Decimal("1000000.00"),
    ) -> None:
        self.strategy = strategy
        self.start_date = start_date
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, int] = {}
        self.last_prices: dict[str, Decimal] = {}
        self.trades: list[PaperTrade] = []
        self.nav_history: list[DailyNavPoint] = []
        self.last_simulated_date: date | None = None
        self._peak_nav = initial_cash

    def qualifies(self, snap: DailyUniverseSnapshot) -> bool:
        """根据策略卡显式规则判断标的当日是否符合观察组合纳入条件。"""
        s = self.strategy
        return (
            snap.pe_percentile <= s.max_pe_percentile
            and snap.volatility_pct <= s.max_volatility_pct
            and snap.verified_claims_count >= s.min_verified_claims
            and snap.pending_alerts_count <= s.max_pending_alerts
        )

    def step_day(self, trade_date: date, snapshots: list[DailyUniverseSnapshot]) -> DailyNavPoint:
        """逐交易日向前推进，严禁回填早于 start_date 或已推进过的历史日期。"""
        if trade_date < self.start_date:
            raise BackfillForbiddenError(
                f"禁止历史回填：请求日期 {trade_date} 早于前向模拟起始日 {self.start_date}"
            )
        if self.last_simulated_date is not None and trade_date <= self.last_simulated_date:
            raise BackfillForbiddenError(
                f"禁止历史回填或改写：请求日期 {trade_date} 不晚于上次模拟日 {self.last_simulated_date}"
            )

        snap_by_sym = {s.symbol: s for s in snapshots}
        for sym, snap in snap_by_sym.items():
            self.last_prices[sym] = snap.close_price

        # 1. 先移出不再满足策略卡条件的持仓（全仓卖出释放现金）
        qualified_syms = [s.symbol for s in snapshots if self.qualifies(s)]
        qualified_set = set(qualified_syms)

        for sym in list(self.positions.keys()):
            if sym not in qualified_set and self.positions[sym] > 0:
                snap = snap_by_sym.get(sym)
                if snap is None:
                    continue
                qty = self.positions[sym]
                proceeds = (Decimal(qty) * snap.close_price).quantize(Decimal("0.01"))
                self.cash += proceeds
                del self.positions[sym]
                self.trades.append(
                    PaperTrade(
                        trade_date=trade_date,
                        symbol=sym,
                        side="sell",
                        shares=qty,
                        price=snap.close_price,
                        amount=proceeds,
                        dossier_id=snap.dossier_id,
                        dossier_version=snap.dossier_version,
                        dossier_as_of=snap.dossier_as_of,
                        reason="不再满足策略卡纳入条件，移出观察组合",
                    )
                )

        # 2. 对新纳入的符合条件标的按等权、整手买入
        new_buys = [sym for sym in qualified_syms if self.positions.get(sym, 0) == 0]
        if new_buys:
            total_equity = self.cash + sum(
                Decimal(qty) * self.last_prices[sym] for sym, qty in self.positions.items()
            )
            target_per_stock = total_equity / Decimal(len(qualified_syms))
            for sym in new_buys:
                snap = snap_by_sym[sym]
                alloc = min(target_per_stock, self.cash)
                raw_lots = (alloc / (snap.close_price * Decimal(self.LOT_SIZE))).to_integral_value(
                    rounding=ROUND_DOWN
                )
                shares = int(raw_lots) * self.LOT_SIZE
                if shares <= 0:
                    continue
                cost = (Decimal(shares) * snap.close_price).quantize(Decimal("0.01"))
                if cost > self.cash:
                    continue
                self.cash -= cost
                self.positions[sym] = shares
                self.trades.append(
                    PaperTrade(
                        trade_date=trade_date,
                        symbol=sym,
                        side="buy",
                        shares=shares,
                        price=snap.close_price,
                        amount=cost,
                        dossier_id=snap.dossier_id,
                        dossier_version=snap.dossier_version,
                        dossier_as_of=snap.dossier_as_of,
                        reason="满足策略卡规则，等权整手纳入观察组合",
                    )
                )

        # 3. 结算当日净值、累计收益与回撤
        holdings_val = sum(
            (Decimal(qty) * self.last_prices[sym] for sym, qty in self.positions.items()),
            Decimal("0.00"),
        ).quantize(Decimal("0.01"))
        total_nav = (self.cash + holdings_val).quantize(Decimal("0.01"))
        if total_nav > self._peak_nav:
            self._peak_nav = total_nav
        dd = ((self._peak_nav - total_nav) / self._peak_nav).quantize(Decimal("0.0001")) if self._peak_nav > 0 else Decimal("0")
        cum_ret = ((total_nav - self.initial_cash) / self.initial_cash).quantize(Decimal("0.0001"))

        point = DailyNavPoint(
            trade_date=trade_date,
            cash=self.cash.quantize(Decimal("0.01")),
            holdings_value=holdings_val,
            total_nav=total_nav,
            cumulative_return=cum_ret,
            drawdown=dd,
        )
        self.nav_history.append(point)
        self.last_simulated_date = trade_date
        return point

    @property
    def trading_days(self) -> int:
        return len(self.nav_history)

    @property
    def max_drawdown(self) -> Decimal:
        if not self.nav_history:
            return Decimal("0")
        return max(p.drawdown for p in self.nav_history)

    @property
    def cumulative_return(self) -> Decimal:
        if not self.nav_history:
            return Decimal("0")
        return self.nav_history[-1].cumulative_return


def evaluate_strategy_promotion_gate(
    *,
    candidate_sim: ForwardPaperSimulator,
    active_sim: ForwardPaperSimulator,
    min_trading_days: int = 10,
    max_underperformance: Decimal = Decimal("0.02"),
) -> PromotionGateReport:
    """纯代码 5 条策略晋级门控（对应文档第 32 题）。"""
    failed: list[str] = []

    # 1. 至少 10 个前向交易日
    days = candidate_sim.trading_days
    if days < min_trading_days:
        failed.append(f"前向模拟交易日不足 {min_trading_days} 天（当前 {days} 天）")

    # 2. 零未来数据检查（每笔交易引用的档案 as_of <= trade_date）
    has_future_leak = any(t.dossier_as_of > t.trade_date for t in candidate_sim.trades)
    if has_future_leak:
        failed.append("检测到前视偏差：存在交易引用了晚于交易日的未来档案")

    # 3. 回撤不超过策略上限
    max_dd = candidate_sim.max_drawdown
    if max_dd > candidate_sim.strategy.max_drawdown_limit:
        failed.append(
            f"最大回撤 {max_dd * 100:.2f}% 超过上限 {candidate_sim.strategy.max_drawdown_limit * 100:.2f}%"
        )

    # 4. 每笔成交均完整可追溯到 dossier_id 与版本号
    all_traceable = all(
        bool(t.dossier_id) and t.dossier_version >= 1 and t.shares % ForwardPaperSimulator.LOT_SIZE == 0
        for t in candidate_sim.trades
    )
    if not all_traceable:
        failed.append("存在不可追溯或非整手的异常成交记录")

    # 5. 与在用版本同期收益不落后超过 2%
    cand_ret = candidate_sim.cumulative_return
    act_ret = active_sim.cumulative_return
    diff = (cand_ret - act_ret).quantize(Decimal("0.0001"))
    if diff < -max_underperformance:
        failed.append(
            f"相对在用版本同期收益落后 {abs(diff) * 100:.2f}%，超过允许上限 {max_underperformance * 100:.2f}%"
        )

    return PromotionGateReport(
        candidate_version=candidate_sim.strategy.version,
        active_version=active_sim.strategy.version,
        trading_days=days,
        has_future_leak=has_future_leak,
        max_drawdown=max_dd,
        all_trades_traceable=all_traceable,
        candidate_return=cand_ret,
        active_return=act_ret,
        return_diff=diff,
        passed=(len(failed) == 0),
        failed_checks=failed,
    )


@dataclass
class StrategyRegistry:
    """只追加策略版本注册与晋级/一键回滚账本（晋级必须人工批准）。"""

    active_strategy: StrategyCard
    versions: dict[str, StrategyCard] = field(default_factory=dict)
    events: list[dict[str, str]] = field(default_factory=list)
    _history_stack: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.versions[self.active_strategy.version] = self.active_strategy
        self._history_stack.append(self.active_strategy.version)
        self.events.append({"event": "init", "version": self.active_strategy.version})

    def promote(
        self,
        *,
        candidate: StrategyCard,
        gate_report: PromotionGateReport,
        approved_by: str,
        human_approved: bool,
    ) -> StrategyCard:
        if not gate_report.passed:
            raise PromotionDeniedError(f"晋级门控未通过：{gate_report.failed_checks}")
        if not human_approved or not approved_by.strip():
            raise PromotionDeniedError("策略晋级必须经人工显式批准，禁止自动上线")

        self.versions[candidate.version] = candidate
        self.active_strategy = candidate
        self._history_stack.append(candidate.version)
        self.events.append(
            {"event": "promote", "version": candidate.version, "approved_by": approved_by}
        )
        return candidate

    def rollback(self, *, operator: str, reason: str) -> StrategyCard:
        """一键回滚到上一在用策略版本，账本只追加回滚事件不删历史。"""
        if len(self._history_stack) < 2:
            raise ValueError("当前已在初始版本，没有可回滚的上一版本")
        self._history_stack.pop()
        prev_ver = self._history_stack[-1]
        self.active_strategy = self.versions[prev_ver]
        self.events.append(
            {"event": "rollback", "version": prev_ver, "operator": operator, "reason": reason}
        )
        return self.active_strategy
