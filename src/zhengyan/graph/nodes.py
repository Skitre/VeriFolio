"""LangGraph 主链路 10 个节点实现（对应文档第 3 题与第 16 题）。

固定状态图节点：
1. intake       -> 识别标的、意图、关注点、个人敏感信息
2. authorize    -> PII 命中时调用 LangGraph interrupt() 暂停等人确认；下单意图附加边界声明
3. route        -> 确定研究档位（Stage 2 基础规则；Stage 3 升级为双轴路由 + 温度校准）
4. plan         -> 调用 LLM 生成结构化 ResearchPlan（强制含风险/反证查询）
5. collect      -> 通过 Harness 调用结构化取数与 Decimal 计算 Skill 生成 A 级证据
6. search       -> 通过 Harness 调用搜索与抓取流水线（自动防隐私外发、SimHash 去重、注入清洗）
7. draft        -> 通过 Harness 调用 LLM 起草（输入自动脱敏，网页证据显式标记为非指令资料）
8. verify       -> 纯代码规则核验器 RuleVerifier（模型绕不过去）
9. quality_gate -> 通过 / 退回补证（上限 1 轮）/ 升级深研信号 / 如实提交部分结论
10. commit      -> 纯代码提交只追加档案（严禁存疑/驳回主张进入结论）
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langgraph.types import interrupt

from zhengyan.core.enums import (
    DossierStatus,
    EvidenceGrade,
    EvidenceSourceType,
    MetricUnit,
    QualityGateAction,
    ResearchTier,
    UserIntent,
    VerificationStatus,
)
from zhengyan.core.models import (
    DossierRecord,
    DraftReport,
    EvidenceCard,
    QualityGateReport,
    ResearchPlan,
)
from zhengyan.data.compute_skills import ComputeSkillError
from zhengyan.deep import (
    DeepResearchEngine,
    DeepSubgraphRuntime,
    MockDeepResearchEngine,
    compile_deep_research_subgraph,
)
from zhengyan.graph.state import ResearchState
from zhengyan.harness import BudgetExceededError, HarnessExecutor, PreCheckFailedError
from zhengyan.routing import (
    DualAxisRouter,
    IntentClassifier,
    MockRoutingClassifier,
    RuleFirstIntentClassifier,
)
from zhengyan.rsi import (
    AppendOnlyRSILedger,
    ExperienceCard,
    RSIComponent,
    format_method_hints_block,
    recall_experience_cards,
)
from zhengyan.search import SourcePolicy
from zhengyan.search.sanitizer import format_as_untrusted_data_block
from zhengyan.verify import RuleVerifier

# 已知公司简称反查表（对应文档第 3 题：正则不行再用已有档案简称反查）
DEFAULT_ALIAS_MAP: dict[str, tuple[str, str]] = {
    "贵州茅台": ("600519", "贵州茅台"),
    "茅台": ("600519", "贵州茅台"),
    "宁德时代": ("300750", "宁德时代"),
    "宁王": ("300750", "宁德时代"),
}

_STOCK_CODE_RE = re.compile(r"(?<!\d)(60\d{4}|00\d{4}|30\d{4}|688\d{3})(?!\d)")

# 严格区分“真实下单意图”与“提到买卖但其实在问研判”
_TRADE_ORDER_RE = re.compile(
    r"(?:帮我|替我|请|立刻|马上|现在|直接|一键)(?:买入|卖出|下单|建仓|清仓|加仓|减仓|抄底|全仓|满仓)"
    r"|(?:买入|卖出|下单)\s*\d+\s*(?:股|手|万|元)"
    r"|(?:按现价|市价|限价)(?:买入|卖出|挂单)"
)
_CONCEPT_RE = re.compile(r"(?:什么是|什么叫|解释一下|名词解释|是什么意思|怎么算|计算公式是什么)")
_FOCUS_KEYWORDS = ("利润质量", "现金流", "毛利率", "合同负债", "估值", "市盈率", "PE", "成长性", "风险", "分红", "持有")
_HIGH_CONTROVERSY_KEYWORDS = ("合不合理", "合理吗", "要不要", "该不该", "继续持有", "继续拿", "贵不贵", "值不值得")

BOUNDARY_NOTICE_TEXT = (
    "【边界声明】证研仅提供可追溯事实与多维度研究分析，不预测涨跌、不给出买卖指令，"
    "且系统在 Harness 权限层完全禁止任何交易类操作（action.*）。"
)


class InMemoryDossierStore:
    """只追加版本化档案存储（Stage 2 轻量版；Stage 5 扩展依赖图与增量复核）。"""

    def __init__(self) -> None:
        self._records: list[DossierRecord] = []

    def append(self, record: DossierRecord) -> DossierRecord:
        existing_for_symbol = [
            r for r in self._records
            if r.symbol == record.symbol and record.symbol is not None
        ]
        next_ver = len(existing_for_symbol) + 1
        versioned = record.model_copy(update={"version": next_ver})
        self._records.append(versioned)
        return versioned

    def list_all(self) -> list[DossierRecord]:
        return list(self._records)


@dataclass
class GraphRuntime:
    """图节点共享的非序列化运行时依赖容器（与检查点状态严格隔离）。"""

    harness: HarnessExecutor
    verifier: RuleVerifier = field(default_factory=RuleVerifier)
    dossier_store: InMemoryDossierStore = field(default_factory=InMemoryDossierStore)
    intent_classifier: IntentClassifier = field(default_factory=RuleFirstIntentClassifier)
    router: DualAxisRouter = field(
        default_factory=lambda: DualAxisRouter(classifier=MockRoutingClassifier())
    )
    search_policy: SourcePolicy | None = None
    deep_engine: DeepResearchEngine = field(default_factory=MockDeepResearchEngine)
    deep_runtime: DeepSubgraphRuntime | None = None
    rsi_ledger: AppendOnlyRSILedger | None = None
    alias_map: dict[str, tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_ALIAS_MAP))
    node_calls: list[str] = field(default_factory=list)


def _extract_symbol_and_name(question: str, alias_map: dict[str, tuple[str, str]]) -> tuple[str | None, str | None]:
    code_match = _STOCK_CODE_RE.search(question)
    if code_match:
        code = code_match.group(1)
        for _, (sym, cname) in alias_map.items():
            if sym == code:
                return code, cname
        return code, code

    for alias, (sym, cname) in alias_map.items():
        if alias in question:
            return sym, cname
    return None, None


def _classify_intent_rule(question: str, symbol: str | None) -> UserIntent:
    if _TRADE_ORDER_RE.search(question):
        return UserIntent.TRADE_ORDER
    if _CONCEPT_RE.search(question) and symbol is None:
        return UserIntent.CONCEPT
    if symbol is not None or any(kw in question for kw in ("财报", "半年报", "年报", "估值", "业绩", "利润")):
        return UserIntent.RESEARCH
    return UserIntent.OTHER


def build_nodes(rt: GraphRuntime) -> dict[str, Any]:
    """构造绑定到 GraphRuntime 的 10 个节点函数。"""

    def intake_node(state: ResearchState) -> ResearchState:
        rt.harness.check_not_revoked("intake")
        rt.node_calls.append("intake")

        question = state["question"]
        symbol, company_name = _extract_symbol_and_name(question, rt.alias_map)
        intent_pred = rt.intent_classifier.classify_intent(question)
        intent = intent_pred.intent
        focus_points = [kw for kw in _FOCUS_KEYWORDS if kw in question]
        pii_hits = [
            {"category": h.category, "matched_text": h.matched_text}
            for h in rt.harness.privacy.detect(question)
        ]

        rt.harness.audit.record(
            node_name="intake",
            caller="graph",
            action_type="decision",
            target=intent.value,
            status="ok",
            duration_ms=intent_pred.latency_ms,
            detail={
                "symbol": symbol,
                "intent": intent.value,
                "intent_source": intent_pred.source,
                "pii_count": len(pii_hits),
                "focus_points": focus_points,
            },
        )
        return {
            "symbol": symbol,
            "company_name": company_name,
            "intent": intent,
            "focus_points": focus_points,
            "pii_hits": pii_hits,
            "supplement_rounds": state.get("supplement_rounds", 0),
            "evidence_store": state.get("evidence_store", {}),
            "budget_exhausted": False,
            "needs_deep_escalation": False,
            "escalated_to_deep": False,
            "aborted": False,
        }

    def authorize_node(state: ResearchState) -> ResearchState:
        rt.harness.check_not_revoked("authorize")
        rt.node_calls.append("authorize")

        pii_hits = state.get("pii_hits") or []
        pii_approved = state.get("pii_approved")
        boundary_notice = BOUNDARY_NOTICE_TEXT if state.get("intent") == UserIntent.TRADE_ORDER else None

        if pii_hits and pii_approved is None:
            rt.harness.audit.record(
                node_name="authorize",
                caller="graph",
                action_type="approval",
                target="pii_confirmation",
                status="interrupted",
                detail={"categories": sorted({h["category"] for h in pii_hits})},
            )
            # LangGraph 原生人在环暂停：挂起当前状态到检查点，等待 Command(resume=...)
            human_reply = interrupt(
                {
                    "type": "pii_confirmation",
                    "message": "检测到提问中包含个人敏感/财务信息。是否同意在脱敏保护下继续本次研究？",
                    "pii_categories": sorted({h["category"] for h in pii_hits}),
                }
            )
            approved = bool(human_reply.get("approved", False)) if isinstance(human_reply, dict) else bool(human_reply)
            pii_approved = approved

        if pii_hits and pii_approved is False:
            rt.harness.audit.record(
                node_name="authorize",
                caller="graph",
                action_type="approval",
                target="pii_confirmation",
                status="denied",
                detail={"reason": "用户拒绝使用包含个人敏感信息的问题继续研究"},
            )
            return {
                "pii_approved": False,
                "boundary_notice": boundary_notice,
                "aborted": True,
                "abort_reason": "用户未授权使用包含个人隐私信息的提问，任务已安全终止。",
            }

        rt.harness.audit.record(
            node_name="authorize",
            caller="graph",
            action_type="node",
            target="authorize",
            status="ok",
            detail={"pii_approved": pii_approved, "has_boundary_notice": boundary_notice is not None},
        )
        return {
            "pii_approved": pii_approved,
            "boundary_notice": boundary_notice,
            "aborted": False,
        }

    def route_node(state: ResearchState) -> ResearchState:
        rt.harness.check_not_revoked("route")
        rt.node_calls.append("route")

        decision = rt.router.route(
            state["question"],
            intent=state["intent"],
            user_selected_tier=state.get("user_selected_tier"),
        )
        tier = decision.tier

        rt.harness.switch_tier(tier)
        model_bindings_map = {
            slot: f"{spec.role}({spec.model})"
            for slot, spec in decision.model_bindings.items()
        }
        rt.harness.audit.record(
            node_name="route",
            caller=decision.classifier_source,
            action_type="decision",
            target=tier.value,
            status="ok",
            duration_ms=decision.latency_ms,
            detail={
                "reason": decision.reason,
                "step": decision.step_triggered,
                "temperature": decision.temperature,
                "abstained": decision.abstained,
                "raw_probs": decision.raw_probs,
                "calibrated_probs": decision.calibrated_probs,
                "model_bindings": model_bindings_map,
            },
        )
        return {
            "tier": tier,
            "route_reason": decision.reason,
            "route_raw_probs": decision.raw_probs,
            "route_calibrated_probs": decision.calibrated_probs,
            "route_abstained": decision.abstained,
            "model_bindings": model_bindings_map,
        }

    def plan_node(state: ResearchState) -> ResearchState:
        rt.harness.check_not_revoked("plan")
        rt.node_calls.append("plan")

        prompt = (
            f"请为标的 {state.get('company_name') or state.get('symbol')} 制定研究检索与计算计划。\n"
            f"用户问题：{state['question']}\n"
            f"当前补证轮次：{state.get('supplement_rounds', 0)}\n"
            f"已有缺口：{state.get('evidence_gaps', [])}"
        )
        try:
            plan = rt.harness.invoke_llm(
                node_name="plan",
                role="draft",
                prompt=prompt,
                response_model=ResearchPlan,
                context={
                    "symbol": state.get("symbol"),
                    "company_name": state.get("company_name"),
                    "evidence_gaps": state.get("evidence_gaps", []),
                    "supplement_rounds": state.get("supplement_rounds", 0),
                },
            )
            return {"plan": plan}
        except BudgetExceededError:
            gaps = list(state.get("evidence_gaps", []))
            gaps.append("规划阶段预算耗尽，按已有材料降级继续")
            return {"budget_exhausted": True, "evidence_gaps": gaps}

    def collect_node(state: ResearchState) -> ResearchState:
        rt.harness.check_not_revoked("collect")
        rt.node_calls.append("collect")

        symbol = state.get("symbol")
        as_of = date.fromisoformat(state["as_of"])
        store = dict(state.get("evidence_store") or {})
        gaps = list(state.get("evidence_gaps") or [])
        budget_exhausted = bool(state.get("budget_exhausted", False))

        if not symbol:
            gaps.append("未识别到明确股票代码，跳过结构化财报取数")
            return {"evidence_store": store, "evidence_gaps": gaps}

        try:
            fin_cards: list[EvidenceCard] = rt.harness.invoke_skill(
                "data.get_financials",
                {"symbol": symbol, "as_of": as_of},
                node_name="collect",
            )
            quote_cards: list[EvidenceCard] = rt.harness.invoke_skill(
                "data.get_quote",
                {"symbol": symbol, "as_of": as_of},
                node_name="collect",
            )
            for c in fin_cards + quote_cards:
                store[c.evidence_id] = c

            if state["tier"] != ResearchTier.QUICK:
                ann_cards: list[EvidenceCard] = rt.harness.invoke_skill(
                    "data.get_announcements",
                    {"symbol": symbol, "as_of": as_of},
                    node_name="collect",
                )
                for c in ann_cards:
                    store[c.evidence_id] = c

            # 针对已取到的同期财报，通过 Harness 调用 Decimal 计算 Skill 生成派生指标卡
            by_period = {
                c.period: c for c in fin_cards
                if c.period and not c.period.endswith("NON_GAAP")
            }
            if "2025H1" in by_period and "2024H1" in by_period:
                c_curr, c_prev = by_period["2025H1"], by_period["2024H1"]
                for metric in ("revenue", "net_profit", "contract_liabilities"):
                    if any(m.name == metric for m in c_curr.metrics) and any(m.name == metric for m in c_prev.metrics):
                        yoy_card = rt.harness.invoke_skill(
                            "compute.yoy_growth",
                            {
                                "current_card": c_curr,
                                "prior_card": c_prev,
                                "metric_name": metric,
                                "as_of": as_of,
                            },
                            node_name="collect",
                        )
                        store[yoy_card.evidence_id] = yoy_card

                if any(m.name == "ocf" for m in c_curr.metrics) and any(m.name == "net_profit" for m in c_curr.metrics):
                    ratio_card = rt.harness.invoke_skill(
                        "compute.ratio",
                        {
                            "card": c_curr,
                            "numerator_metric": "ocf",
                            "denominator_metric": "net_profit",
                            "output_name": "ocf_to_np",
                            "output_display_name": "净现比",
                            "output_unit": MetricUnit.MULTIPLE,
                        },
                        node_name="collect",
                    )
                    store[ratio_card.evidence_id] = ratio_card

            try:
                pe_card = rt.harness.invoke_skill(
                    "compute.pe_ttm_pit",
                    {"symbol": symbol, "as_of": as_of},
                    node_name="collect",
                )
                if pe_card is not None:
                    store[pe_card.evidence_id] = pe_card
            except ComputeSkillError as exc:
                gaps.append(f"PE(TTM) 暂无法按披露日对齐计算：{exc}")

        except BudgetExceededError as exc:
            budget_exhausted = True
            gaps.append(f"取数阶段预算耗尽：{exc}")

        rt.harness.audit.record(
            node_name="collect",
            caller="graph",
            action_type="node",
            target="collect",
            status="ok",
            detail={"a_grade_cards": sum(1 for c in store.values() if c.grade == EvidenceGrade.A)},
        )
        return {
            "evidence_store": store,
            "evidence_gaps": gaps,
            "budget_exhausted": budget_exhausted,
        }

    def search_node(state: ResearchState) -> ResearchState:
        rt.harness.check_not_revoked("search")
        rt.node_calls.append("search")

        plan = state.get("plan")
        store = dict(state.get("evidence_store") or {})
        gaps = list(state.get("evidence_gaps") or [])
        budget_exhausted = bool(state.get("budget_exhausted", False))

        if plan is None or budget_exhausted:
            return {"evidence_store": store, "evidence_gaps": gaps, "budget_exhausted": budget_exhausted}

        as_of = date.fromisoformat(state["as_of"])
        symbol = state.get("symbol")
        queries = list(plan.queries) + list(plan.risk_queries)

        for q in queries:
            try:
                raw_results = rt.harness.invoke_skill(
                    "search.web",
                    {"query": q, "symbol": symbol, "as_of": as_of},
                    node_name="search",
                )
                # 快查白名单默认不含 search.fetch，仅在允许抓取时调用
                if "search.fetch" in rt.harness.policy.tiers[rt.harness.tier].allowed_skills:
                    web_cards: list[EvidenceCard] = rt.harness.invoke_skill(
                        "search.fetch",
                        {"raw_results": raw_results, "symbol": symbol, "as_of": as_of},
                        node_name="search",
                    )
                    for wc in web_cards:
                        store[wc.evidence_id] = wc
            except BudgetExceededError as exc:
                budget_exhausted = True
                gaps.append(f"检索阶段预算耗尽：{exc}")
                break
            except PreCheckFailedError as exc:
                gaps.append(f"搜索词触发安全前置拦截：{exc}")
                continue

        rt.harness.audit.record(
            node_name="search",
            caller="graph",
            action_type="node",
            target="search",
            status="ok",
            detail={"total_evidence": len(store), "budget_exhausted": budget_exhausted},
        )
        return {
            "evidence_store": store,
            "evidence_gaps": gaps,
            "budget_exhausted": budget_exhausted,
        }

    def draft_node(state: ResearchState) -> ResearchState:
        rt.harness.check_not_revoked("draft")
        rt.node_calls.append("draft")

        store = state.get("evidence_store") or {}
        web_blocks = [
            format_as_untrusted_data_block(c.evidence_id, c.source_name, c.excerpt)
            for c in store.values()
            if c.grade in (EvidenceGrade.B, EvidenceGrade.C)
        ]
        hints_block = ""
        if rt.rsi_ledger is not None:
            active_exp = rt.rsi_ledger.get_active_asset(RSIComponent.S1_EXPERIENCE_CARDS)
            if active_exp is not None:
                cards = [ExperienceCard.model_validate(c) for c in (active_exp.payload.get("cards") or [])]
                recalled = recall_experience_cards(state["question"], cards, max_cards=3)
                hints_block = format_method_hints_block(recalled)

        prompt = (
            f"用户问题：{state['question']}\n"
            f"意图：{state['intent'].value}\n"
            f"A 级证据数：{sum(1 for c in store.values() if c.grade == EvidenceGrade.A)}\n"
            + (f"{hints_block}\n" if hints_block else "")
            + "外部网页资料：\n"
            + "\n".join(web_blocks)
        )
        try:
            draft = rt.harness.invoke_llm(
                node_name="draft",
                role="draft",
                prompt=prompt,
                response_model=DraftReport,
                context={
                    "question": state["question"],
                    "intent": state["intent"],
                    "evidence_store": store,
                    "supplement_rounds": state.get("supplement_rounds", 0),
                },
            )
            combined_gaps = list(dict.fromkeys(list(state.get("evidence_gaps") or []) + list(draft.evidence_gaps)))
            return {
                "summary": draft.summary,
                "claims": draft.claims,
                "evidence_gaps": combined_gaps,
                "counterpoints": draft.counterpoints,
            }
        except BudgetExceededError as exc:
            gaps = list(state.get("evidence_gaps") or []) + [f"起草阶段预算耗尽：{exc}"]
            return {
                "summary": "预算耗尽，无法生成完整草稿，仅保留已有可核验记录。",
                "claims": [],
                "evidence_gaps": gaps,
                "counterpoints": [],
                "budget_exhausted": True,
            }

    def verify_node(state: ResearchState) -> ResearchState:
        """纯规则代码执行，不调模型，模型绝对绕不过去。"""
        rt.harness.check_not_revoked("verify")
        rt.node_calls.append("verify")

        as_of = date.fromisoformat(state["as_of"])
        claims = state.get("claims") or []
        store = state.get("evidence_store") or {}

        results = rt.verifier.verify_all(claims, store, as_of=as_of)
        rt.harness.audit.record(
            node_name="verify",
            caller="code_rule_verifier",
            action_type="node",
            target="verify",
            status="ok",
            detail={
                "total": len(results),
                "verified": sum(1 for r in results if r.status == VerificationStatus.VERIFIED),
                "doubtful": sum(1 for r in results if r.status == VerificationStatus.DOUBTFUL),
                "rejected": sum(1 for r in results if r.status == VerificationStatus.REJECTED),
            },
        )
        return {"verification_results": results}

    def quality_gate_node(state: ResearchState) -> ResearchState:
        """纯规则质量门控（对应文档第 16 题与第 5 题标准→深研升级检查点）。"""
        rt.harness.check_not_revoked("quality_gate")
        rt.node_calls.append("quality_gate")

        results = state.get("verification_results") or []
        store = state.get("evidence_store") or {}
        tier = state["tier"]
        intent = state["intent"]
        gaps = state.get("evidence_gaps") or []
        counterpoints = state.get("counterpoints") or []
        supplement_rounds = int(state.get("supplement_rounds", 0))
        max_rounds = rt.harness.policy.tiers[tier].max_supplement_rounds

        verified_facts = [r for r in results if r.status == VerificationStatus.VERIFIED and r.is_fact]
        verified_any = [r for r in results if r.status == VerificationStatus.VERIFIED]
        doubtful_cnt = sum(1 for r in results if r.status == VerificationStatus.DOUBTFUL)
        rejected_cnt = sum(1 for r in results if r.status == VerificationStatus.REJECTED)
        has_derived = any(
            c.grade == EvidenceGrade.A and c.source_type == EvidenceSourceType.COMPUTE
            for c in store.values()
        )

        reasons: list[str] = []
        passed = False

        if intent == UserIntent.CONCEPT:
            passed = len(verified_any) >= 1 and doubtful_cnt == 0 and rejected_cnt == 0
            if not passed:
                reasons.append("概念解释输出未通过核验")
        elif tier == ResearchTier.QUICK:
            passed = len(verified_facts) >= 1 and doubtful_cnt == 0
            if not passed:
                reasons.append("快查未达到 >=1 条已核验事实且 0 存疑的标准")
        else:
            if len(verified_facts) < 3:
                reasons.append(f"已核验事实不足 3 条（当前 {len(verified_facts)} 条）")
            if doubtful_cnt > 0:
                reasons.append(f"存在 {doubtful_cnt} 条存疑主张")
            if not has_derived:
                reasons.append("缺少同口径派生计算指标")
            passed = not reasons

        next_rounds = supplement_rounds
        needs_escalation = bool(state.get("needs_deep_escalation", False))

        if passed:
            action = QualityGateAction.PASS
        elif (
            tier == ResearchTier.STANDARD
            and not bool(state.get("escalated_to_deep", False))
            and (doubtful_cnt >= 2 or len(counterpoints) >= 3)
        ):
            action = QualityGateAction.ESCALATE_DEEP
            needs_escalation = True
            reasons.append("分歧/存疑超阈值（存疑>=2 或反方>=3），复用已有证据升级至争议深研（限 1 次）")
        elif (
            supplement_rounds < max_rounds
            and rt.harness.budget.can_search()
            and not state.get("budget_exhausted", False)
            and bool(gaps)
        ):
            action = QualityGateAction.RETRY_EVIDENCE
            next_rounds = supplement_rounds + 1
            reasons.append(f"触发第 {next_rounds} 轮退回补证")
        else:
            action = QualityGateAction.PARTIAL
            reasons.append("无法继续补证或预算已尽，按已有证据如实提交部分结论")

        report = QualityGateReport(
            action=action,
            verified_fact_count=len(verified_facts),
            doubtful_count=doubtful_cnt,
            rejected_count=rejected_cnt,
            has_derived_metric=has_derived,
            counterpoint_count=len(counterpoints),
            reasons=reasons,
        )
        rt.harness.audit.record(
            node_name="quality_gate",
            caller="code_quality_gate",
            action_type="node",
            target=action.value,
            status="ok",
            detail=report.model_dump(),
        )
        return {
            "quality_gate": report,
            "supplement_rounds": next_rounds,
            "needs_deep_escalation": needs_escalation,
        }

    def deep_research_node(state: ResearchState) -> ResearchState:
        """多专家争议深研节点：调用 LangGraph Send 并行子图（复用已取结构化证据，最多升级一次）。"""
        rt.harness.check_not_revoked("deep_research")
        rt.node_calls.append("deep_research")

        # 升级进入深研时切换为 deep 档权限与预算（保留已消耗计数）
        if rt.harness.tier != ResearchTier.DEEP:
            rt.harness.switch_tier(ResearchTier.DEEP)

        policy = rt.search_policy or SourcePolicy(whitelist={}, blacklist=set())
        deep_rt = DeepSubgraphRuntime(
            harness=rt.harness,
            search_policy=policy,
            engine=rt.deep_engine,
            verifier=rt.verifier,
        )
        rt.deep_runtime = deep_rt
        # checkpointer=None 令子图继承父图 Checkpointer，确保高危争议 interrupt 原生冒泡！
        deep_subgraph = compile_deep_research_subgraph(deep_rt, checkpointer=None)

        sub_out = deep_subgraph.invoke(
            {
                "question": state["question"],
                "symbol": state.get("symbol"),
                "as_of": state["as_of"],
                "shared_evidence_store": dict(state.get("evidence_store") or {}),
                "expert_submissions": [],
            }
        )

        adj = sub_out.get("adjudication")
        aborted = bool(sub_out.get("aborted_by_human_review", False))
        return {
            "tier": ResearchTier.DEEP,
            "escalated_to_deep": True,
            "evidence_store": sub_out.get("blackboard_evidence") or state.get("evidence_store") or {},
            "summary": adj.summary if adj else "多专家深研完成",
            "claims": sub_out.get("blackboard_claims") or [],
            "verification_results": sub_out.get("verification_results") or [],
            "deep_challenges": sub_out.get("challenges") or [],
            "deep_challenge_responses": sub_out.get("challenge_responses") or [],
            "deep_adjudication": adj,
            "deep_isolation_violations": sub_out.get("isolation_violations") or [],
            "deep_final_claims": sub_out.get("final_claims") or [],
            "deep_excluded_claims": sub_out.get("excluded_claims") or [],
            "aborted": aborted,
            "abort_reason": "用户拒绝提交存在未决高严重度争议的深研结论。" if aborted else None,
        }

    def commit_node(state: ResearchState) -> ResearchState:
        """纯代码提交节点：只允许 VERIFIED 且未被质询推翻的主张进入结论，旧版本永不覆盖。"""
        rt.node_calls.append("commit")

        as_of = date.fromisoformat(state["as_of"])
        if state.get("aborted", False):
            record = DossierRecord(
                dossier_id=f"DOS-{uuid.uuid4().hex[:8]}",
                symbol=state.get("symbol"),
                as_of=as_of,
                question=state["question"],
                intent=state.get("intent", UserIntent.OTHER),
                tier=state.get("tier", ResearchTier.STANDARD),
                status=DossierStatus.ABORTED,
                boundary_notice=state.get("boundary_notice"),
                summary=state.get("abort_reason") or "任务已终止",
                budget_snapshot=rt.harness.budget.snapshot(),
            )
            saved = rt.dossier_store.append(record)
            return {"dossier": saved, "final_status": DossierStatus.ABORTED}

        claims = state.get("claims") or []
        results = state.get("verification_results") or []
        res_map = {r.claim_id: r for r in results}
        adj = state.get("deep_adjudication")

        if adj is not None:
            # 深研裁决路径：严格采纳主席在规则核验 + 反证官质询之后放行的主张
            verified_claims = list(state.get("deep_final_claims") or [])
            excluded_claims = list(state.get("deep_excluded_claims") or [])
            dissents = [
                f"{d.topic}：主观点={d.bull_or_main_view}｜反方={d.bear_or_counter_view}"
                for d in adj.dissenting_opinions
            ]
            upheld = list(adj.upheld_challenge_ids)
            if adj.human_review_required:
                # 高严重度未决争议经人工确认后，以“部分结论”提交（文档第 11.2 题）
                final_status = DossierStatus.PARTIAL
            elif len(verified_claims) >= 3:
                final_status = DossierStatus.COMPLETE
            else:
                final_status = DossierStatus.PARTIAL
        else:
            verified_claims = [c for c in claims if res_map.get(c.claim_id) and res_map[c.claim_id].can_enter_conclusion]
            excluded_claims = [c for c in claims if not (res_map.get(c.claim_id) and res_map[c.claim_id].can_enter_conclusion)]
            dissents = []
            upheld = []
            qg = state.get("quality_gate")
            final_status = (
                DossierStatus.COMPLETE
                if (qg is not None and qg.action == QualityGateAction.PASS)
                else DossierStatus.PARTIAL
            )

        record = DossierRecord(
            dossier_id=f"DOS-{uuid.uuid4().hex[:8]}",
            symbol=state.get("symbol"),
            as_of=as_of,
            question=state["question"],
            intent=state["intent"],
            tier=state["tier"],
            status=final_status,
            boundary_notice=state.get("boundary_notice"),
            summary=state.get("summary", ""),
            verified_claims=verified_claims,
            excluded_claims=excluded_claims,
            verification_results=results,
            evidence_ids=sorted((state.get("evidence_store") or {}).keys()),
            evidence_gaps=list(state.get("evidence_gaps") or []),
            dissenting_opinions=dissents,
            upheld_challenges=upheld,
            method_asset_versions=(
                rt.rsi_ledger.get_active_versions_map() if rt.rsi_ledger is not None else {}
            ),
            needs_deep_escalation=bool(state.get("needs_deep_escalation", False)),
            budget_snapshot=rt.harness.budget.snapshot(),
        )
        saved = rt.dossier_store.append(record)

        rt.harness.audit.record(
            node_name="commit",
            caller="code_commit",
            action_type="node",
            target=saved.dossier_id,
            status="ok",
            detail={
                "version": saved.version,
                "status": saved.status.value,
                "verified_claims": len(verified_claims),
                "excluded_claims": len(excluded_claims),
                "dissents": len(dissents),
            },
        )
        return {
            "dossier": saved,
            "final_status": final_status,
            "trace_summary": {
                "trace_id": rt.harness.audit.trace_id,
                "span_count": len(rt.harness.audit.spans),
            },
        }

    return {
        "intake": intake_node,
        "authorize": authorize_node,
        "route": route_node,
        "plan": plan_node,
        "collect": collect_node,
        "search": search_node,
        "draft": draft_node,
        "verify": verify_node,
        "quality_gate": quality_gate_node,
        "deep_research": deep_research_node,
        "commit": commit_node,
    }
