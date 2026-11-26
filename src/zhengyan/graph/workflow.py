"""LangGraph 10 节点固定状态图装配与 Checkpointer 续跑封装。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from zhengyan.core.enums import QualityGateAction, ResearchTier, UserIntent
from zhengyan.data import MarketDataProvider, MockMarketDataProvider
from zhengyan.deep import DeepResearchEngine, MockDeepResearchEngine
from zhengyan.graph.nodes import GraphRuntime, InMemoryDossierStore, build_nodes
from zhengyan.graph.skills_registry import (
    MockWebSearchBackend,
    WebSearchBackend,
    register_default_skills,
)
from zhengyan.graph.state import ResearchState
from zhengyan.harness import HarnessExecutor, HarnessPolicyConfig
from zhengyan.llm import LLMClient, MockLLMClient
from zhengyan.routing import (
    DualAxisRouter,
    IntentClassifier,
    MockRoutingClassifier,
    ModelAxisConfig,
    RuleFirstIntentClassifier,
)
from zhengyan.rsi import AppendOnlyRSILedger
from zhengyan.search import SourcePolicy


def _after_authorize(state: ResearchState) -> str:
    return "commit" if state.get("aborted", False) else "route"


def _after_route(state: ResearchState) -> str:
    if state.get("intent") == UserIntent.CONCEPT:
        return "draft"
    return "plan"


def _after_collect(state: ResearchState) -> str:
    """深研档位在 collect 取完共享 A 级结构化快照后直接进入多专家争议深研子图（文档第 3 题）。"""
    if state.get("tier") == ResearchTier.DEEP:
        return "deep_research"
    return "search"


def _after_quality_gate(state: ResearchState) -> str:
    qg = state.get("quality_gate")
    if qg is not None and qg.action == QualityGateAction.RETRY_EVIDENCE:
        return "plan"
    if qg is not None and qg.action == QualityGateAction.ESCALATE_DEEP and not state.get("escalated_to_deep", False):
        return "deep_research"
    return "commit"


def create_default_checkpointer() -> MemorySaver:
    """构造显式白名单序列化的 MemorySaver（消除 LangGraph 1.2 反序列化警告并守住安全边界）。"""
    allowed = [
        ("zhengyan.core.enums", "AccountingBasis"),
        ("zhengyan.core.enums", "ApprovalLevel"),
        ("zhengyan.core.enums", "ChallengeResponseAction"),
        ("zhengyan.core.enums", "ChallengeSeverity"),
        ("zhengyan.core.enums", "ClaimType"),
        ("zhengyan.core.enums", "DossierStatus"),
        ("zhengyan.core.enums", "EvidenceGrade"),
        ("zhengyan.core.enums", "EvidenceSourceType"),
        ("zhengyan.core.enums", "ExpertRole"),
        ("zhengyan.core.enums", "MetricUnit"),
        ("zhengyan.core.enums", "PITGuarantee"),
        ("zhengyan.core.enums", "QualityGateAction"),
        ("zhengyan.core.enums", "ResearchTier"),
        ("zhengyan.core.enums", "UserIntent"),
        ("zhengyan.core.enums", "VerificationStatus"),
        ("zhengyan.core.models", "AdjudicationDecision"),
        ("zhengyan.core.models", "BudgetSnapshot"),
        ("zhengyan.core.models", "ChallengeItem"),
        ("zhengyan.core.models", "ChallengeResponse"),
        ("zhengyan.core.models", "Claim"),
        ("zhengyan.core.models", "DissentingOpinion"),
        ("zhengyan.core.models", "DossierRecord"),
        ("zhengyan.core.models", "EvidenceCard"),
        ("zhengyan.core.models", "ExpertSubBudget"),
        ("zhengyan.core.models", "ExpertSubmission"),
        ("zhengyan.core.models", "MetricEntry"),
        ("zhengyan.core.models", "NumberCheckDetail"),
        ("zhengyan.core.models", "QualityGateReport"),
        ("zhengyan.core.models", "ResearchPlan"),
        ("zhengyan.core.models", "VerificationResult"),
    ]
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))


def compile_research_graph(
    runtime: GraphRuntime,
    *,
    checkpointer: Any | None = None,
):
    """编译带检查点的完整研究状态图（含快查/标准主链路 + 深研子图直达与升级路径）。"""
    nodes = build_nodes(runtime)
    builder = StateGraph(ResearchState)

    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "authorize")
    builder.add_conditional_edges(
        "authorize",
        _after_authorize,
        {"commit": "commit", "route": "route"},
    )
    builder.add_conditional_edges(
        "route",
        _after_route,
        {"draft": "draft", "plan": "plan"},
    )
    builder.add_edge("plan", "collect")
    builder.add_conditional_edges(
        "collect",
        _after_collect,
        {"deep_research": "deep_research", "search": "search"},
    )
    builder.add_edge("search", "draft")
    builder.add_edge("draft", "verify")
    builder.add_edge("verify", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        _after_quality_gate,
        {"plan": "plan", "deep_research": "deep_research", "commit": "commit"},
    )
    builder.add_edge("deep_research", "commit")
    builder.add_edge("commit", END)

    cp = checkpointer if checkpointer is not None else create_default_checkpointer()
    return builder.compile(checkpointer=cp)


class ResearchRunner:
    """面向调用方与测试的便捷运行封装（支持中断检测与 Command(resume=...) 续跑）。"""

    def __init__(
        self,
        *,
        as_of: date,
        policy_path: Path,
        search_sources_path: Path,
        models_config_path: Path | None = None,
        llm_client: LLMClient | None = None,
        data_provider: MarketDataProvider | None = None,
        search_backend: WebSearchBackend | None = None,
        dossier_store: InMemoryDossierStore | None = None,
        intent_classifier: IntentClassifier | None = None,
        router: DualAxisRouter | None = None,
        deep_engine: DeepResearchEngine | None = None,
        rsi_ledger: AppendOnlyRSILedger | None = None,
        checkpointer: Any | None = None,
        initial_tier: ResearchTier = ResearchTier.STANDARD,
    ) -> None:
        self.policy = HarnessPolicyConfig.from_yaml(policy_path)
        self.search_policy = SourcePolicy.from_yaml(search_sources_path)
        default_models_path = models_config_path or (policy_path.parent / "models.yaml")
        self.model_axis = (
            ModelAxisConfig.from_yaml(default_models_path)
            if default_models_path.exists()
            else None
        )
        self.llm_client = llm_client or MockLLMClient()
        self.data_provider = data_provider or MockMarketDataProvider()
        self.search_backend = search_backend or MockWebSearchBackend()

        self.harness = HarnessExecutor(
            policy=self.policy,
            tier=initial_tier,
            as_of=as_of,
            llm_client=self.llm_client,
        )
        register_default_skills(
            self.harness,
            data_provider=self.data_provider,
            search_policy=self.search_policy,
            search_backend=self.search_backend,
        )
        effective_router = router or DualAxisRouter(
            classifier=MockRoutingClassifier(),
            model_axis=self.model_axis,
        )
        self.runtime = GraphRuntime(
            harness=self.harness,
            dossier_store=dossier_store or InMemoryDossierStore(),
            intent_classifier=intent_classifier or RuleFirstIntentClassifier(),
            router=effective_router,
            search_policy=self.search_policy,
            deep_engine=deep_engine or MockDeepResearchEngine(),
            rsi_ledger=rsi_ledger,
        )
        self.checkpointer = checkpointer if checkpointer is not None else create_default_checkpointer()
        self.graph = compile_research_graph(self.runtime, checkpointer=self.checkpointer)
        self.as_of = as_of

    def start(
        self,
        question: str,
        *,
        thread_id: str,
        user_selected_tier: ResearchTier | None = None,
    ) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        initial_state: ResearchState = {
            "question": question,
            "as_of": self.as_of.isoformat(),
            "user_selected_tier": user_selected_tier,
        }
        return self.graph.invoke(initial_state, config=config)

    def resume(self, resume_payload: Any, *, thread_id: str) -> dict[str, Any]:
        """审批回复后从检查点续跑（不重跑 interrupt 之前已完成的节点）。"""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(Command(resume=resume_payload), config=config)

    def get_pending_interrupts(self, *, thread_id: str) -> list[Any]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        interrupts: list[Any] = []
        for task in snapshot.tasks:
            for intr in getattr(task, "interrupts", ()):
                interrupts.append(intr.value)
        return interrupts
