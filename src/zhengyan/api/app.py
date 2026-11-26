"""FastAPI 后端服务（覆盖研究台、档案版本链、订阅增量复核、RSI 账本与回滚、Kev 评测看板）。"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from zhengyan.core.enums import ResearchTier
from zhengyan.data import MockMarketDataProvider
from zhengyan.graph import ResearchRunner
from zhengyan.memory import FiveLayerMemoryStore, SubscriptionScanner
from zhengyan.routing import (
    ClassifierMode,
    DualAxisRouter,
    HybridIntentClassifier,
    HybridRoutingClassifier,
    Kev08BClassifier,
    MockRoutingClassifier,
    ModelAxisConfig,
    RuleFirstIntentClassifier,
    evaluate_intent,
    evaluate_routing,
    load_intent_cases,
    load_real_replay_queries,
    load_routing_cases,
)
from zhengyan.rsi import AppendOnlyRSILedger, ExperienceCard, RSIComponent, SixGatesReport


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    as_of: str = Field(default="2025-08-20")
    user_selected_tier: ResearchTier | None = None
    classifier_mode: ClassifierMode = ClassifierMode.SMALL


class ResumeRequest(BaseModel):
    approved: bool


class RollbackRequest(BaseModel):
    component: RSIComponent
    operator: str = "researcher"
    reason: str = Field(min_length=10)


class ScanRequest(BaseModel):
    scan_as_of: str = "2025-08-20"


def create_app(root_dir: Path | None = None) -> FastAPI:
    """工厂函数创建完整可测试的证研 FastAPI 应用。"""
    root = root_dir or Path(__file__).resolve().parents[3]
    policy_path = root / "config" / "policies.yaml"
    search_path = root / "config" / "search_sources.yaml"
    models_path = root / "config" / "models.yaml"
    web_index = root / "web" / "index.html"

    app = FastAPI(title="证研（ZhengYan）可信投研多 Agent 控制台", version="0.1.0")

    data_provider = MockMarketDataProvider()
    memory_store = FiveLayerMemoryStore()
    rsi_ledger = AppendOnlyRSILedger()
    rsi_ledger.init_component(
        RSIComponent.S1_EXPERIENCE_CARDS,
        version="v1",
        payload={"cards": []},
    )
    # 预置 v2 经验卡 L-007 供控制台与研究台开箱展示
    l007 = ExperienceCard(
        card_id="L-007",
        version="v1",
        tags=["合同负债", "预收款", "消费品"],
        applicable_scope="消费品公司，合同负债或预收款大幅变动时",
        lesson="先排除打款政策、渠道结构调整，再推断需求变化",
        must_check="公告或业绩说明会中关于渠道、打款政策的表述",
        invalidation_condition="公司已明确说明打款政策未变",
        source_ref="深研中被裁定成立的质询（档案 600519 v5）",
    )
    rsi_ledger.publish_candidate(
        component=RSIComponent.S1_EXPERIENCE_CARDS,
        candidate_version="v2",
        parent_version="v1",
        change_reason="沉淀消费品合同负债变动归因排查泛化经验卡 L-007",
        payload={"cards": [l007.model_dump()]},
        gate_report=SixGatesReport(proposal_id="BOOT-L007", component=RSIComponent.S1_EXPERIENCE_CARDS, passed=True),
        approver="solo_dev",
        human_approved=True,
        approval_mode="self_approved",
    )
    rsi_ledger.init_component(
        RSIComponent.S2_ROUTING_PARAMS,
        version="v1",
        payload={"temperature": 0.75, "abstain_threshold": 0.55},
    )
    rsi_ledger.init_component(
        RSIComponent.S3_SEARCH_TEMPLATES,
        version="v1",
        payload={"whitelist_domains": ["stcn.com", "caixin.com", "cnstock.com"], "relevance_threshold": 0.50},
    )

    active_runners: dict[str, ResearchRunner] = {}

    @app.get("/", response_class=HTMLResponse)
    def serve_ui() -> str:
        if web_index.exists():
            return web_index.read_text(encoding="utf-8")
        return "<h1>证研控制台</h1><p>API 文档请访问 /docs</p>"

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "active_rsi_versions": rsi_ledger.get_active_versions_map(),
        }

    @app.post("/api/research")
    def start_research(req: ResearchRequest) -> dict[str, Any]:
        as_of_d = date.fromisoformat(req.as_of)
        tid = f"th-{uuid.uuid4().hex[:8]}"
        model_axis = ModelAxisConfig.from_yaml(models_path)
        kev = Kev08BClassifier()
        hybrid_intent = HybridIntentClassifier(
            mode=req.classifier_mode,
            llm_classifier=RuleFirstIntentClassifier(),
            small_classifier=kev,
        )
        hybrid_route = HybridRoutingClassifier(
            mode=req.classifier_mode,
            llm_classifier=MockRoutingClassifier(),
            small_classifier=kev,
        )
        runner = ResearchRunner(
            as_of=as_of_d,
            policy_path=policy_path,
            search_sources_path=search_path,
            data_provider=data_provider,
            intent_classifier=hybrid_intent,
            router=DualAxisRouter(classifier=hybrid_route, model_axis=model_axis),
            rsi_ledger=rsi_ledger,
        )
        active_runners[tid] = runner
        out = runner.start(req.question, thread_id=tid, user_selected_tier=req.user_selected_tier)
        pending = runner.get_pending_interrupts(thread_id=tid)
        if pending:
            return {
                "thread_id": tid,
                "status": "interrupted",
                "interrupt": pending[0],
            }

        dossier = out["dossier"]
        if dossier is not None and dossier.symbol:
            memory_store.append_dossier(
                dossier,
                evidence_store=out.get("evidence_store") or {},
                method_asset_versions=rsi_ledger.get_active_versions_map(),
            )
        return {
            "thread_id": tid,
            "status": "completed",
            "dossier": dossier.model_dump(mode="json") if dossier else None,
            "node_calls": runner.runtime.node_calls,
            "shadow_records": [r.__dict__ for r in hybrid_route.shadow_records],
        }

    @app.post("/api/research/{thread_id}/resume")
    def resume_research(thread_id: str, req: ResumeRequest) -> dict[str, Any]:
        runner = active_runners.get(thread_id)
        if runner is None:
            raise HTTPException(status_code=404, detail="未找到对应的研究会话 thread_id")
        out = runner.resume({"approved": req.approved}, thread_id=thread_id)
        dossier = out.get("dossier")
        if dossier is not None and dossier.symbol and dossier.status.value != "aborted":
            memory_store.append_dossier(
                dossier,
                evidence_store=out.get("evidence_store") or {},
                method_asset_versions=rsi_ledger.get_active_versions_map(),
            )
        return {
            "thread_id": thread_id,
            "status": "completed",
            "dossier": dossier.model_dump(mode="json") if dossier else None,
            "node_calls": runner.runtime.node_calls,
        }

    @app.get("/api/dossiers")
    def list_dossiers() -> dict[str, Any]:
        latest = memory_store.list_latest_dossiers_all_symbols()
        return {
            "count": len(latest),
            "items": [
                {
                    "dossier_id": n.dossier.dossier_id,
                    "symbol": n.dossier.symbol,
                    "version": n.dossier.version,
                    "as_of": n.dossier.as_of.isoformat(),
                    "tier": n.dossier.tier.value,
                    "status": n.dossier.status.value,
                    "verified_claims_count": len(n.dossier.verified_claims),
                    "method_asset_versions": n.method_asset_versions,
                }
                for n in latest
            ],
        }

    @app.post("/api/subscriptions/{symbol}/scan")
    def scan_subscription(symbol: str, req: ScanRequest) -> dict[str, Any]:
        scanner = SubscriptionScanner(memory_store=memory_store, data_provider=data_provider)
        alert = scanner.scan_symbol(symbol, scan_as_of=date.fromisoformat(req.scan_as_of))
        if alert is None:
            return {"has_alert": False, "symbol": symbol}
        return {"has_alert": True, "alert": alert.model_dump(mode="json")}

    @app.get("/api/rsi/ledger")
    def get_rsi_ledger() -> dict[str, Any]:
        return {
            "active_versions": rsi_ledger.get_active_versions_map(),
            "assets_count": len(rsi_ledger.assets),
            "runs_count": len(rsi_ledger.runs),
            "events": [e.model_dump(mode="json") for e in rsi_ledger.events],
        }

    @app.post("/api/rsi/rollback")
    def rollback_rsi(req: RollbackRequest) -> dict[str, Any]:
        try:
            target_ver = rsi_ledger.rollback(req.component, operator=req.operator, reason=req.reason)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "component": req.component.value,
            "rolled_back_to": target_ver,
            "active_versions": rsi_ledger.get_active_versions_map(),
        }

    @app.get("/api/evals/kev-comparison")
    def kev_comparison() -> dict[str, Any]:
        routing_cases = load_routing_cases(root / "evals" / "golden" / "routing_cases.json")
        intent_main = load_intent_cases(root / "evals" / "golden" / "intent_cases.json")
        intent_hold = load_intent_cases(root / "evals" / "golden" / "intent_holdout.json")
        real_queries = load_real_replay_queries(root / "evals" / "golden" / "real_queries_replay.json")

        llm_intent = RuleFirstIntentClassifier()
        kev = Kev08BClassifier()
        r_llm = DualAxisRouter(classifier=MockRoutingClassifier(), temperature=0.75)
        r_kev = DualAxisRouter(classifier=kev, temperature=0.75)

        rep_r_llm = evaluate_routing(routing_cases, intent_classifier=llm_intent, router=r_llm)
        rep_r_kev = evaluate_routing(routing_cases, intent_classifier=kev, router=r_kev)
        rep_i_main = evaluate_intent(intent_main, classifier=kev)
        rep_i_hold = evaluate_intent(intent_hold, classifier=kev)
        rep_i_all = evaluate_intent(intent_main + intent_hold, classifier=kev)

        replay_agree = 0
        for _, q in real_queries:
            d_llm = r_llm.route(q, intent=llm_intent.classify_intent(q).intent)
            d_kev = r_kev.route(q, intent=kev.classify_intent(q).intent)
            if d_llm.tier == d_kev.tier:
                replay_agree += 1

        return {
            "routing_24": {
                "llm": {
                    "accuracy": rep_r_llm.accuracy,
                    "deep_recall": rep_r_llm.deep_recall,
                    "abstain_rate": rep_r_llm.abstain_rate,
                    "median_latency_ms": rep_r_llm.median_latency_ms,
                },
                "kev_0_8b": {
                    "accuracy": rep_r_kev.accuracy,
                    "deep_recall": rep_r_kev.deep_recall,
                    "abstain_rate": rep_r_kev.abstain_rate,
                    "median_latency_ms": rep_r_kev.median_latency_ms,
                },
            },
            "intent_64_kev": {
                "main_38_accuracy": rep_i_main.accuracy,
                "holdout_26_accuracy": rep_i_hold.accuracy,
                "trade_recalled": rep_i_all.trade_recalled,
                "trade_total": rep_i_all.trade_total,
                "trade_false_positives": rep_i_all.trade_false_positives,
                "median_latency_ms": rep_i_all.median_latency_ms,
            },
            "real_replay_22": {
                "total": len(real_queries),
                "agreed": replay_agree,
                "agreement_rate": round(replay_agree / len(real_queries), 4),
            },
        }

    return app


app = create_app()
