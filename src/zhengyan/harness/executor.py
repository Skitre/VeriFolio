"""Harness 安全控制台与统一执行器（对应文档第 2、11、11.1、11.2、29 题）。

核心治理闭环（模型永远绕不过去）：
一次 Skill 调用必须依次通过：
1. 撤销检查（任务已取消/撤权 → 立即终止）
2. Skill 注册检查 + 全局禁止前缀检查（action.* 一律拒绝）+ 当前流程白名单权限检查
3. 审批等级检查（FORBID 拒绝；CONFIRM 须持有效且未过期且参数哈希匹配的 ApprovalTicket）
4. 前置检查（Fail-Closed：参数时点 <= as_of、外发隐私拦截；检查器自身报错也默认阻断！）
5. 同任务内幂等缓存检查（sha256(skill + version + params) 命中则直接返回，不重复扣预算）
6. 预算扣减（搜索/抓取/总耗时）
7. 执行 + 有限指数退避重试（仅重试网络/429/5xx，不重试 402 等不可重试错误）
8. 写入幂等缓存 + 全链路审计 Span

LLM 调用同样经过 Harness：
撤销检查 → Token 预算检查 → 输入隐私脱敏 → 调用模型 → Pydantic 结构化契约校验 → 扣减预算 → 写审计 Span。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel

from zhengyan.core.enums import ApprovalLevel, ResearchTier
from zhengyan.harness.approval import ApprovalTicket, canonical_params_hash
from zhengyan.harness.audit import AuditTrail
from zhengyan.harness.budget import BudgetExceededError, BudgetTracker
from zhengyan.harness.policy import HarnessPolicyConfig
from zhengyan.harness.privacy import PrivacyGuard
from zhengyan.llm.base import LLMClient

TModel = TypeVar("TModel", bound=BaseModel)


class HarnessError(RuntimeError):
    """Harness 执行治理基类异常。"""


class TaskRevokedError(HarnessError):
    """任务已被用户取消或撤销授权。"""


class PermissionDeniedError(HarnessError):
    """Skill 未注册、被全局禁止（如 action.*）或不在当前流程白名单中。"""


class ApprovalRequiredError(HarnessError):
    """需要人工审批或审批票据无效/已过期。"""


class PreCheckFailedError(HarnessError):
    """前置检查失败或前置检查器本身发生异常（Fail-Closed 阻断）。"""


class RetryableSkillError(HarnessError):
    """可重试错误：网络超时、429 限流、5xx 服务端异常。"""


class NonRetryableSkillError(HarnessError):
    """不可重试错误：402 余额不足、参数错误等，直接失败。"""


@dataclass(frozen=True)
class RegisteredSkill:
    name: str
    version: str
    handler: Callable[..., Any]
    approval_level: ApprovalLevel = ApprovalLevel.AUTO
    budget_kind: Literal["none", "search", "fetch"] = "none"
    outbound_text_param: str | None = None
    precheck: Callable[[dict[str, Any]], None] | None = None


class HarnessExecutor:
    """统一管控 Skill 与模型调用的 Harness 控制台。"""

    def __init__(
        self,
        *,
        policy: HarnessPolicyConfig,
        tier: ResearchTier,
        as_of: date,
        llm_client: LLMClient | None = None,
        privacy_guard: PrivacyGuard | None = None,
        audit: AuditTrail | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = policy
        self.tier = tier
        self.as_of = as_of
        self.llm_client = llm_client
        self.privacy = privacy_guard or PrivacyGuard()
        self.audit = audit or AuditTrail()
        self.budget: BudgetTracker = policy.create_budget(tier)
        self._sleep = sleep_fn
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

        self._skills: dict[str, RegisteredSkill] = {}
        self._idempotency_cache: dict[str, Any] = {}
        self._revoked: bool = False
        self._revoke_reason: str = ""

    def switch_tier(self, new_tier: ResearchTier) -> None:
        """在路由阶段确定研究档位后切换白名单与预算上限（保留已消耗计数）。"""
        tp = self.policy.tiers[new_tier]
        self.tier = new_tier
        self.budget.tier = new_tier
        self.budget.max_tokens = tp.max_tokens
        self.budget.max_searches = tp.max_searches
        self.budget.max_fetches = tp.max_fetches
        self.budget.max_duration_seconds = tp.max_duration_seconds

    def revoke(self, reason: str = "用户撤销授权或取消任务") -> None:
        self._revoked = True
        self._revoke_reason = reason

    def check_not_revoked(self, node_name: str, caller: str = "harness") -> None:
        if self._revoked:
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="node",
                target="revocation_check",
                status="denied",
                detail={"reason": self._revoke_reason},
            )
            raise TaskRevokedError(self._revoke_reason)

    def register_skill(self, skill: RegisteredSkill) -> None:
        self._skills[skill.name] = skill

    def invoke_skill(
        self,
        skill_name: str,
        params: dict[str, Any],
        *,
        node_name: str,
        caller: str = "graph",
        approval_ticket: ApprovalTicket | None = None,
    ) -> Any:
        t0 = time.monotonic()

        # 1. 撤销检查
        self.check_not_revoked(node_name, caller=caller)

        # 2. 全局禁令（如 action.* 交易类 Skill 一律禁止）
        if any(skill_name.startswith(prefix) for prefix in self.policy.forbidden_prefixes):
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="skill",
                target=skill_name,
                status="denied",
                detail={"reason": "命中全局禁止前缀（禁止交易类操作）"},
            )
            raise PermissionDeniedError(f"Skill {skill_name!r} 属于系统严禁的交易/危险动作，已拒绝")

        # 3. 注册检查 + 当前流程白名单检查
        skill = self._skills.get(skill_name)
        if skill is None:
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="skill",
                target=skill_name,
                status="denied",
                detail={"reason": "Skill 未注册"},
            )
            raise PermissionDeniedError(f"Skill {skill_name!r} 未在 Harness 注册")

        allowed = self.policy.tiers[self.tier].allowed_skills
        if skill_name not in allowed:
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="skill",
                target=skill_name,
                status="denied",
                detail={"reason": f"不在 {self.tier.value} 流程白名单内"},
            )
            raise PermissionDeniedError(
                f"越权调用：Skill {skill_name!r} 不在 {self.tier.value} 流程白名单内"
            )

        # 4. 审批检查
        if skill.approval_level == ApprovalLevel.FORBID:
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="approval",
                target=skill_name,
                status="denied",
                detail={"reason": "Skill 审批等级为 FORBID"},
            )
            raise PermissionDeniedError(f"Skill {skill_name!r} 被配置为禁止调用")

        if skill.approval_level == ApprovalLevel.CONFIRM:
            if approval_ticket is None:
                self.audit.record(
                    node_name=node_name,
                    caller=caller,
                    action_type="approval",
                    target=skill_name,
                    status="interrupted",
                    detail={"reason": "需要人工审批票据"},
                )
                raise ApprovalRequiredError(f"Skill {skill_name!r} 需要人工审批确认")
            ok, reason = approval_ticket.verify(
                skill_name=skill.name,
                version=skill.version,
                params=params,
                now=self._now(),
            )
            if not ok:
                self.audit.record(
                    node_name=node_name,
                    caller=caller,
                    action_type="approval",
                    target=skill_name,
                    status="denied",
                    detail={"reason": reason, "ticket_id": approval_ticket.ticket_id},
                )
                raise ApprovalRequiredError(f"审批校验失败：{reason}")

        # 5. 前置检查（Fail-Closed：检查本身抛出任何异常都默认阻断！）
        try:
            self._run_prechecks(skill, params)
        except Exception as exc:
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="skill",
                target=skill_name,
                status="denied",
                detail={"reason": f"前置检查阻断（fail-closed）：{exc}"},
            )
            raise PreCheckFailedError(f"前置检查阻断：{exc}") from exc

        # 6. 同任务内幂等缓存（sha256(skill + version + params) 命中不重复扣预算）
        idem_key = f"{skill.name}@{skill.version}:{canonical_params_hash(params)}"
        if idem_key in self._idempotency_cache:
            duration_ms = (time.monotonic() - t0) * 1000
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="skill",
                target=skill_name,
                status="cached",
                duration_ms=duration_ms,
                detail={"idempotency_key": idem_key[:24]},
            )
            return self._idempotency_cache[idem_key]

        # 7. 预算扣减
        try:
            self.budget.check_time_budget()
            if skill.budget_kind == "search":
                self.budget.consume_search(1)
            elif skill.budget_kind == "fetch":
                self.budget.consume_fetch(1)
        except BudgetExceededError as exc:
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="skill",
                target=skill_name,
                status="denied",
                detail={"reason": str(exc)},
            )
            raise

        # 8. 执行 + 指数退避有限重试（仅 RetryableSkillError 重试，最多 max_retries 次）
        attempts = 0
        last_error: Exception | None = None
        while attempts <= self.policy.max_retries:
            attempts += 1
            try:
                result = skill.handler(**params)
                self._idempotency_cache[idem_key] = result
                duration_ms = (time.monotonic() - t0) * 1000
                self.audit.record(
                    node_name=node_name,
                    caller=caller,
                    action_type="skill",
                    target=skill_name,
                    status="ok",
                    duration_ms=duration_ms,
                    detail={"attempts": attempts, "version": skill.version},
                )
                return result
            except RetryableSkillError as exc:
                last_error = exc
                if attempts <= self.policy.max_retries:
                    delay = self.policy.backoff_base_seconds * (2 ** (attempts - 1))
                    self._sleep(delay)
                    continue
                break
            except NonRetryableSkillError as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                break

        duration_ms = (time.monotonic() - t0) * 1000
        self.audit.record(
            node_name=node_name,
            caller=caller,
            action_type="skill",
            target=skill_name,
            status="error",
            duration_ms=duration_ms,
            detail={"attempts": attempts, "error": str(last_error)},
        )
        assert last_error is not None
        raise last_error

    def invoke_llm(
        self,
        *,
        node_name: str,
        role: str,
        prompt: str,
        response_model: type[TModel],
        context: dict[str, Any] | None = None,
        caller: str = "graph",
        estimated_tokens: int = 1500,
    ) -> TModel:
        """所有 LLM 调用同样必须过 Harness：撤销检查 → 预算 → 隐私脱敏 → 契约校验 → 审计。"""
        t0 = time.monotonic()
        self.check_not_revoked(node_name, caller=caller)

        if self.llm_client is None:
            raise HarnessError("Harness 未配置 llm_client")

        if not self.budget.can_use_tokens(estimated_tokens):
            err = BudgetExceededError(
                f"Token 预算不足以执行 {node_name}（已用 {self.budget.tokens_used}/{self.budget.max_tokens}）"
            )
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="llm",
                target=role,
                status="denied",
                detail={"reason": str(err)},
            )
            raise err

        # 脱敏失败同样遵循 fail-closed 原则直接阻断
        try:
            masked_prompt = self.privacy.mask_for_llm(prompt)
        except Exception as exc:
            self.audit.record(
                node_name=node_name,
                caller=caller,
                action_type="llm",
                target=role,
                status="denied",
                detail={"reason": f"脱敏模块异常（fail-closed）：{exc}"},
            )
            raise PreCheckFailedError(f"模型脱敏失败，已阻断调用：{exc}") from exc

        resp = self.llm_client.generate_structured(
            role=role,
            prompt=masked_prompt,
            response_model=response_model,
            context=context or {},
        )
        self.budget.consume_tokens(resp.tokens_used)
        duration_ms = (time.monotonic() - t0) * 1000
        self.audit.record(
            node_name=node_name,
            caller=caller,
            action_type="llm",
            target=f"{role}:{resp.model_name}",
            status="ok",
            duration_ms=duration_ms,
            tokens_used=resp.tokens_used,
            detail={"masked": masked_prompt != prompt, "schema": response_model.__name__},
        )
        return resp.parsed

    def _run_prechecks(self, skill: RegisteredSkill, params: dict[str, Any]) -> None:
        # 1. 时点纪律检查：若参数包含 as_of 且晚于当前研究时点，直接拒绝
        req_as_of = params.get("as_of")
        if isinstance(req_as_of, date) and req_as_of > self.as_of:
            raise ValueError(f"请求时点 {req_as_of} 晚于研究时点 {self.as_of}，禁止超时点取数")

        # 2. 外发隐私拦截：发往外部搜索/抓取的文本命中隐私立即阻断
        if skill.outbound_text_param and skill.outbound_text_param in params:
            outbound_val = str(params[skill.outbound_text_param])
            self.privacy.assert_safe_outbound(outbound_val)

        # 3. Skill 自定义前置检查钩子
        if skill.precheck is not None:
            skill.precheck(params)
