"""Harness 安全控制台核心机制单测（权限、Fail-Closed、幂等、指数退避重试、审批、脱敏、撤销）。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from zhengyan.core.enums import ApprovalLevel, ResearchTier
from zhengyan.core.models import ResearchPlan
from zhengyan.harness import (
    ApprovalRequiredError,
    ApprovalTicket,
    BudgetExceededError,
    HarnessExecutor,
    HarnessPolicyConfig,
    NonRetryableSkillError,
    PermissionDeniedError,
    PreCheckFailedError,
    PrivacyGuard,
    RegisteredSkill,
    RetryableSkillError,
    TaskRevokedError,
)
from zhengyan.llm import MockLLMClient

POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "policies.yaml"


def _make_harness(
    tier: ResearchTier = ResearchTier.STANDARD,
    sleep_log: list[float] | None = None,
    now_box: list[datetime] | None = None,
    llm_client: MockLLMClient | None = None,
    privacy_guard: PrivacyGuard | None = None,
) -> HarnessExecutor:
    policy = HarnessPolicyConfig.from_yaml(POLICY_PATH)
    sleeps = sleep_log if sleep_log is not None else []
    clock_box = now_box if now_box is not None else [datetime(2025, 8, 20, 10, 0, tzinfo=timezone.utc)]
    return HarnessExecutor(
        policy=policy,
        tier=tier,
        as_of=date(2025, 8, 20),
        llm_client=llm_client or MockLLMClient(),
        privacy_guard=privacy_guard,
        sleep_fn=lambda sec: sleeps.append(sec),
        now_fn=lambda: clock_box[0],
    )


def test_action_prefix_and_whitelist_denied_with_audit() -> None:
    h = _make_harness(tier=ResearchTier.QUICK)
    h.register_skill(RegisteredSkill("search.fetch", "1.0.0", lambda: "ok"))

    # 1. 所有 action.* 交易类动作一律拒绝
    with pytest.raises(PermissionDeniedError, match="严禁的交易"):
        h.invoke_skill("action.place_order", {"symbol": "600519"}, node_name="draft")

    # 2. quick 档白名单不包含 search.fetch → 越权拒绝
    with pytest.raises(PermissionDeniedError, match="越权调用"):
        h.invoke_skill("search.fetch", {}, node_name="search")

    # 3. 审计日志完整留痕
    denied_spans = [s for s in h.audit.spans if s.status == "denied"]
    assert len(denied_spans) == 2


def test_precheck_fail_closed_and_future_as_of_blocked() -> None:
    h = _make_harness()

    def broken_precheck(_params: dict) -> None:
        raise RuntimeError("前置校验服务崩溃")

    h.register_skill(
        RegisteredSkill(
            "data.get_financials",
            "1.0.0",
            lambda **kw: "should_not_run",
            precheck=broken_precheck,
        )
    )
    h.register_skill(
        RegisteredSkill(
            "search.web",
            "1.0.0",
            lambda query, as_of: f"ok:{query}",
            budget_kind="search",
            outbound_text_param="query",
        )
    )

    # 1. 前置检查自身抛异常 → Fail-Closed 默认阻断，绝不放行
    with pytest.raises(PreCheckFailedError, match="前置检查阻断"):
        h.invoke_skill("data.get_financials", {"symbol": "600519"}, node_name="collect")

    # 2. 请求时点晚于 as_of → 阻断
    with pytest.raises(PreCheckFailedError, match="禁止超时点取数"):
        h.invoke_skill("search.web", {"query": "茅台", "as_of": date(2025, 9, 1)}, node_name="search")

    # 3. 外发查询含个人隐私 → 阻断，且不扣搜索预算
    with pytest.raises(PreCheckFailedError, match="外发查询命中个人隐私"):
        h.invoke_skill(
            "search.web",
            {"query": "我的持仓 30万 茅台怎么办", "as_of": date(2025, 8, 20)},
            node_name="search",
        )
    assert h.budget.searches_used == 0


def test_idempotency_does_not_double_charge_budget() -> None:
    h = _make_harness(tier=ResearchTier.QUICK)
    calls = 0

    def fake_search(query: str) -> str:
        nonlocal calls
        calls += 1
        return f"res:{query}"

    h.register_skill(
        RegisteredSkill("search.web", "1.0.0", fake_search, budget_kind="search", outbound_text_param="query")
    )

    r1 = h.invoke_skill("search.web", {"query": "贵州茅台 半年报"}, node_name="search")
    r2 = h.invoke_skill("search.web", {"query": "贵州茅台 半年报"}, node_name="search")
    assert r1 == r2 == "res:贵州茅台 半年报"
    assert calls == 1
    assert h.budget.searches_used == 1  # 同任务内幂等命中不重复扣搜索次数！
    assert any(s.status == "cached" for s in h.audit.spans)


def test_retry_exponential_backoff_vs_non_retryable_402() -> None:
    sleeps: list[float] = []
    h = _make_harness(sleep_log=sleeps)
    attempts = 0

    def flaky_skill() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RetryableSkillError("429 Too Many Requests")
        return "recovered"

    def out_of_balance() -> str:
        raise NonRetryableSkillError("402 Payment Required")

    h.register_skill(RegisteredSkill("data.get_quote", "1.0.0", flaky_skill))
    h.register_skill(RegisteredSkill("data.get_financials", "1.0.0", out_of_balance))

    # 最多重试 2 次（共 3 次尝试），指数退避 1.0s, 2.0s
    assert h.invoke_skill("data.get_quote", {}, node_name="collect") == "recovered"
    assert attempts == 3
    assert sleeps == [1.0, 2.0]

    # 402 不可重试错误直接失败，不退避不重试
    sleeps.clear()
    with pytest.raises(NonRetryableSkillError, match="402"):
        h.invoke_skill("data.get_financials", {}, node_name="collect")
    assert sleeps == []


def test_confirm_skill_approval_ticket_and_expiry() -> None:
    clock = [datetime(2025, 8, 20, 10, 0, tzinfo=timezone.utc)]
    h = _make_harness(now_box=clock)
    h.register_skill(
        RegisteredSkill(
            "search.fetch",
            "1.0.0",
            lambda url: f"fetched:{url}",
            approval_level=ApprovalLevel.CONFIRM,
            budget_kind="fetch",
        )
    )

    params = {"url": "https://www.stcn.com/1.html"}
    # 无票据 → 需要审批
    with pytest.raises(ApprovalRequiredError):
        h.invoke_skill("search.fetch", params, node_name="search")

    ticket = ApprovalTicket.issue(
        ticket_id="TKT-1",
        skill_name="search.fetch",
        version="1.0.0",
        params=params,
        approver="reviewer",
        ttl_seconds=600,
        now=clock[0],
    )
    assert h.invoke_skill("search.fetch", params, node_name="search", approval_ticket=ticket) == "fetched:https://www.stcn.com/1.html"

    # 参数篡改（未过期但 URL 变了）→ 参数哈希校验失败
    with pytest.raises(ApprovalRequiredError, match="参数哈希不一致"):
        h.invoke_skill(
            "search.fetch",
            {"url": "https://www.stcn.com/2.html"},
            node_name="search",
            approval_ticket=ticket,
        )

    # 超过 600 秒后即使原参数再次提交也触发过期拒签（审批校验在幂等缓存之前）
    clock[0] = clock[0] + timedelta(seconds=605)
    with pytest.raises(ApprovalRequiredError, match="已过期"):
        h.invoke_skill("search.fetch", params, node_name="search", approval_ticket=ticket)


def test_llm_masking_fail_closed_and_revocation() -> None:
    llm = MockLLMClient()
    h = _make_harness(llm_client=llm)

    # 1. 发给 LLM 的 prompt 自动脱敏
    plan = h.invoke_llm(
        node_name="plan",
        role="draft",
        prompt="我有存款 50万，手机号 13900001111，研究 600519",
        response_model=ResearchPlan,
        context={"symbol": "600519"},
    )
    assert isinstance(plan, ResearchPlan)
    assert "50万" not in llm.received_prompts[0]
    assert "13900001111" not in llm.received_prompts[0]
    assert "[已脱敏:手机号]" in llm.received_prompts[0]
    assert h.budget.tokens_used == 900

    # 2. 脱敏器本身抛异常 → Fail-Closed 阻断 LLM 调用
    class BrokenPrivacy(PrivacyGuard):
        def mask_for_llm(self, text: str) -> str:
            raise RuntimeError("脱敏模块故障")

    h_broken = _make_harness(llm_client=MockLLMClient(), privacy_guard=BrokenPrivacy())
    with pytest.raises(PreCheckFailedError, match="模型脱敏失败"):
        h_broken.invoke_llm(
            node_name="plan",
            role="draft",
            prompt="研究 600519",
            response_model=ResearchPlan,
        )

    # 3. 任务被用户撤销 → 立即阻断后续任何调用
    h.revoke("用户点击停止研究")
    with pytest.raises(TaskRevokedError, match="用户点击停止研究"):
        h.invoke_llm(
            node_name="draft",
            role="draft",
            prompt="继续起草",
            response_model=ResearchPlan,
        )


def test_budget_limit_enforced() -> None:
    h = _make_harness(tier=ResearchTier.QUICK)
    h.register_skill(RegisteredSkill("search.web", "1.0.0", lambda q: q, budget_kind="search"))

    # quick 档最多 3 次搜索
    for i in range(3):
        h.invoke_skill("search.web", {"q": f"query-{i}"}, node_name="search")
    with pytest.raises(BudgetExceededError, match="搜索预算不足"):
        h.invoke_skill("search.web", {"q": "query-3"}, node_name="search")
