"""FastMCP 只读对外接口层（对应文档第 30 题 & 八股 Skills vs MCP）。

安全边界：
- 仅向外部 Agent 暴露只读的取数与 Decimal 计算 Skill；
- 严禁暴露任何交易类（action.*）或状态修改类工具；
- 所有工具强制传入 `as_of` 研究时点并经过 Harness 审计与前置检查；
- 支持直接内嵌调用与（安装 `fastmcp` 后）一键导出为标准 MCP Server。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

from zhengyan.core.enums import MetricUnit, ResearchTier
from zhengyan.data import (
    MarketDataProvider,
    MockMarketDataProvider,
    compute_pe_ttm_pit,
    compute_ratio,
    compute_yoy_growth,
)
from zhengyan.harness import HarnessExecutor, HarnessPolicyConfig, RegisteredSkill


class ReadOnlyMCPToolbox:
    """对外只读 MCP 工具箱（所有调用走 HarnessExecutor 与时点纪律）。"""

    def __init__(
        self,
        *,
        policy_path: Path,
        data_provider: MarketDataProvider | None = None,
    ) -> None:
        self.policy = HarnessPolicyConfig.from_yaml(policy_path)
        self.provider = data_provider or MockMarketDataProvider()
        self._tools: dict[str, dict[str, Any]] = {
            "get_financial_cards": {
                "description": "按披露日 <= as_of 只读获取 A 级结构化财报证据卡",
                "read_only": True,
                "handler": self.get_financial_cards,
            },
            "get_market_quote": {
                "description": "按 trade_date <= as_of 只读获取最近交易日收盘价与总市值",
                "read_only": True,
                "handler": self.get_market_quote,
            },
            "compute_yoy_growth": {
                "description": "使用 Decimal 精确计算同口径同比增长率（%）并返回 A 级计算卡",
                "read_only": True,
                "handler": self.compute_yoy_growth,
            },
            "compute_ratio": {
                "description": "使用 Decimal 精确计算财报比率（毛利率/净利率/净现比等）",
                "read_only": True,
                "handler": self.compute_ratio,
            },
            "compute_pe_ttm_pit": {
                "description": "严格按披露日逐日对齐计算市盈率 PE(TTM)",
                "read_only": True,
                "handler": self.compute_pe_ttm_pit,
            },
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": meta["description"], "read_only": meta["read_only"]}
            for name, meta in self._tools.items()
        ]

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        if name.startswith("action.") or name not in self._tools:
            raise PermissionError(f"MCP 只读接口拒绝调用未注册或非只读工具：{name!r}")
        handler: Callable[..., Any] = self._tools[name]["handler"]
        return handler(**kwargs)

    def _make_harness(self, as_of: date) -> HarnessExecutor:
        return HarnessExecutor(policy=self.policy, tier=ResearchTier.STANDARD, as_of=as_of)

    def get_financial_cards(self, *, symbol: str, as_of: str) -> list[dict[str, Any]]:
        as_of_d = date.fromisoformat(as_of)
        h = self._make_harness(as_of_d)
        h.register_skill(
            RegisteredSkill(
                "data.get_financials",
                "1.0.0",
                lambda symbol, as_of: [
                    c for c in self.provider.to_evidence_cards(symbol, as_of)
                    if c.source_type.value == "financial_report"
                ],
            )
        )
        cards = h.invoke_skill(
            "data.get_financials",
            {"symbol": symbol, "as_of": as_of_d},
            node_name="mcp",
            caller="external_mcp_agent",
        )
        return [c.model_dump(mode="json") for c in cards]

    def get_market_quote(self, *, symbol: str, as_of: str) -> dict[str, Any] | None:
        as_of_d = date.fromisoformat(as_of)
        h = self._make_harness(as_of_d)
        h.register_skill(
            RegisteredSkill("data.get_quote", "1.0.0", self.provider.get_price_on_or_before)
        )
        pt = h.invoke_skill(
            "data.get_quote",
            {"symbol": symbol, "as_of": as_of_d},
            node_name="mcp",
            caller="external_mcp_agent",
        )
        return pt.model_dump(mode="json") if pt is not None else None

    def compute_yoy_growth(
        self,
        *,
        symbol: str,
        as_of: str,
        current_period: str,
        prior_period: str,
        metric: str,
    ) -> dict[str, Any]:
        as_of_d = date.fromisoformat(as_of)
        cards = {c.evidence_id: c for c in self.provider.to_evidence_cards(symbol, as_of_d)}
        curr = cards[f"EV-FIN-{symbol}-{current_period}"]
        prev = cards[f"EV-FIN-{symbol}-{prior_period}"]
        h = self._make_harness(as_of_d)
        h.register_skill(RegisteredSkill("compute.yoy_growth", "1.0.0", compute_yoy_growth))
        out = h.invoke_skill(
            "compute.yoy_growth",
            {"current_card": curr, "prior_card": prev, "metric_name": metric, "as_of": as_of_d},
            node_name="mcp",
            caller="external_mcp_agent",
        )
        return out.model_dump(mode="json")

    def compute_ratio(
        self,
        *,
        symbol: str,
        as_of: str,
        period: str,
        numerator: str,
        denominator: str,
        output_name: str,
        output_display_name: str,
        output_unit: str = "%",
    ) -> dict[str, Any]:
        as_of_d = date.fromisoformat(as_of)
        cards = {c.evidence_id: c for c in self.provider.to_evidence_cards(symbol, as_of_d)}
        card = cards[f"EV-FIN-{symbol}-{period}"]
        h = self._make_harness(as_of_d)
        h.register_skill(RegisteredSkill("compute.ratio", "1.0.0", compute_ratio))
        out = h.invoke_skill(
            "compute.ratio",
            {
                "card": card,
                "numerator_metric": numerator,
                "denominator_metric": denominator,
                "output_name": output_name,
                "output_display_name": output_display_name,
                "output_unit": MetricUnit(output_unit),
            },
            node_name="mcp",
            caller="external_mcp_agent",
        )
        return out.model_dump(mode="json")

    def compute_pe_ttm_pit(self, *, symbol: str, as_of: str) -> dict[str, Any]:
        as_of_d = date.fromisoformat(as_of)
        stmts = self.provider.get_financial_statements(symbol, as_of_d)
        price = self.provider.get_price_on_or_before(symbol, as_of_d)
        if price is None:
            raise ValueError(f"未找到 {symbol} 截至 {as_of} 的行情快照")
        h = self._make_harness(as_of_d)
        h.register_skill(RegisteredSkill("compute.pe_ttm_pit", "1.0.0", compute_pe_ttm_pit))
        out = h.invoke_skill(
            "compute.pe_ttm_pit",
            {"statements": stmts, "price": price, "as_of": as_of_d},
            node_name="mcp",
            caller="external_mcp_agent",
        )
        return out.model_dump(mode="json")

    def to_fastmcp(self) -> Any:
        """若已安装可选依赖 `fastmcp`，导出标准 FastMCP Server 实例。"""
        from fastmcp import FastMCP  # type: ignore[import-not-found]

        mcp = FastMCP("zhengyan-readonly-research-tools")
        for name, meta in self._tools.items():
            mcp.tool(name=name, description=meta["description"])(meta["handler"])
        return mcp
