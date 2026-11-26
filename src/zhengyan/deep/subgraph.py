"""多专家争议深研 LangGraph 子图（真正使用 Send 并行 + 首轮隔离 + 非投票裁决 + 高危争议 interrupt）。

拓扑结构（对应文档图 3 与第 13、14 题）：
START
  → chair_convene          （主席按证据缺口召集最多 5 位专家 + 反证官）
  → [Send("expert_worker", 隔离输入) x N]  （并行扇出，每人独立 2 搜 2 抓子预算，看不到彼此结论）
  → blackboard_verify      （归约黑板 + 首轮隔离越权引用拦截 + RuleVerifier 纯规则核验）
  → adversary_challenge    （反证官针对已核验主张挑毛病，最多 8 条）
  → expert_respond         （被质询专家回应 1 轮）
  → chair_adjudicate       （主席按独立证据非投票裁决；质询成立/认错主张禁止入结论；保留分歧；
                            若存在高严重度未决争议 → 调用 interrupt() 暂停等人复核！）
  → END
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from zhengyan.core.enums import (
    ChallengeResponseAction,
    ChallengeSeverity,
    EvidenceGrade,
    EvidenceSourceType,
    ExpertRole,
    VerificationStatus,
)
from zhengyan.core.models import (
    AdjudicationDecision,
    Claim,
    DissentingOpinion,
    EvidenceCard,
    ExpertSubBudget,
    ExpertSubmission,
    VerificationResult,
)
from zhengyan.deep.prompts_mock import (
    DeepResearchEngine,
    MockDeepResearchEngine,
    build_expert_private_web_results,
)
from zhengyan.deep.state import DeepResearchState, ExpertWorkerInput
from zhengyan.harness import HarnessExecutor
from zhengyan.search import SourcePolicy, WebSearchPipeline
from zhengyan.verify import RuleVerifier


@dataclass
class DeepSubgraphRuntime:
    harness: HarnessExecutor
    search_policy: SourcePolicy
    engine: DeepResearchEngine = field(default_factory=MockDeepResearchEngine)
    verifier: RuleVerifier = field(default_factory=RuleVerifier)
    expert_executions: list[str] = field(default_factory=list)


def count_independent_sources(
    evidence_ids: list[str],
    evidence_store: dict[str, EvidenceCard],
) -> int:
    """计算独立证据来源数（对应文档第 14 题：三个专家看同一篇转载稿只算 1 条独立来源）。

    去重规则：
    - A 级结构化证据按 evidence_id 独立计数；
    - B/C 级网页若共享同一 event_cluster_id 或同一 original_source，合并为 1 个独立来源！
    """
    independent_keys: set[str] = set()
    for eid in evidence_ids:
        card = evidence_store.get(eid)
        if card is None:
            continue
        if card.grade == EvidenceGrade.A:
            independent_keys.add(f"A:{card.evidence_id}")
        else:
            if card.original_source:
                independent_keys.add(f"WEB_SRC:{card.original_source}")
            elif card.event_cluster_id:
                independent_keys.add(f"WEB_CLU:{card.event_cluster_id}")
            else:
                independent_keys.add(f"WEB_ID:{card.evidence_id}")
    return len(independent_keys)


def compile_deep_research_subgraph(
    rt: DeepSubgraphRuntime,
    *,
    checkpointer: Any | None = None,
):
    """编译多专家争议深研子图（支持独立调用，也可嵌入主图作为节点）。"""
    web_pipeline = WebSearchPipeline(rt.search_policy)

    def chair_convene_node(state: DeepResearchState) -> DeepResearchState:
        rt.harness.check_not_revoked("chair_convene", caller="chair")
        shared = state.get("shared_evidence_store") or {}
        selected, rationale = rt.engine.convene_experts(state["question"], shared)
        # 硬性治理上限：最多召集 5 位专家（文档 L63 & L305）
        selected = selected[:5]

        rt.harness.audit.record(
            node_name="chair_convene",
            caller="chair",
            action_type="node",
            target="convene_experts",
            status="ok",
            detail={"experts": [e.value for e in selected], "rationale": rationale},
        )
        return {
            "selected_experts": selected,
            "convene_rationale": rationale,
            "deep_tokens_used": 600,
        }

    def fanout_to_experts(state: DeepResearchState) -> list[Send]:
        """LangGraph Send 并行扇出：仅传递共享 A 级结构化事实快照，实现首轮绝对隔离。"""
        shared = state.get("shared_evidence_store") or {}
        shared_a_only = {
            eid: card for eid, card in shared.items() if card.grade == EvidenceGrade.A
        }
        sends: list[Send] = []
        for role in state.get("selected_experts") or []:
            model_role = "reason" if role == ExpertRole.VALUATION else "draft"
            payload: ExpertWorkerInput = {
                "question": state["question"],
                "symbol": state.get("symbol"),
                "as_of": state["as_of"],
                "expert_role": role,
                "model_role": model_role,
                "shared_structured_evidence": shared_a_only,
                "max_searches": 2,
                "max_fetches": 2,
            }
            sends.append(Send("expert_worker", payload))
        return sends

    def expert_worker_node(worker_input: ExpertWorkerInput) -> dict[str, Any]:
        """单专家隔离取证工作节点（独立 2 搜 2 抓子预算，看不到其他专家任何产出）。"""
        role = worker_input["expert_role"]
        rt.harness.check_not_revoked(f"expert_{role.value}", caller=role.value)
        rt.expert_executions.append(role.value)

        as_of = date.fromisoformat(worker_input["as_of"])
        symbol = worker_input.get("symbol")
        sub_budget = ExpertSubBudget(
            max_searches=worker_input["max_searches"],
            max_fetches=worker_input["max_fetches"],
        )

        queries = rt.engine.expert_queries(role, symbol, worker_input["question"])
        private_web_cards: list[EvidenceCard] = []
        reprint_mode = bool(getattr(rt.engine, "reprint_echo_mode", False))

        for q in queries:
            if sub_budget.searches_used >= sub_budget.max_searches:
                break
            # 隐私外发前置校验（同样遵守 Harness 外发安全规则）
            rt.harness.privacy.assert_safe_outbound(q)
            sub_budget.searches_used += 1
            raw_web = build_expert_private_web_results(
                role, symbol, as_of, reprint_echo_mode=reprint_mode
            )
            if raw_web and sub_budget.fetches_used < sub_budget.max_fetches:
                sub_budget.fetches_used += 1
                graded = web_pipeline.process(raw_web, as_of=as_of, symbol=symbol)
                # 用专家角色前缀区分私有证据 ID，以便核验器检测跨视角偷看违规
                for idx, card in enumerate(graded, start=1):
                    scoped = card.model_copy(
                        update={"evidence_id": f"EV-EXP-{role.value}-WEB-{idx:03d}"}
                    )
                    private_web_cards.append(scoped)

        shared_a = worker_input["shared_structured_evidence"]
        claims, dims, tokens = rt.engine.expert_analyze(
            role,
            question=worker_input["question"],
            symbol=symbol,
            shared_a_cards=shared_a,
            own_web_cards=private_web_cards,
        )
        rt.harness.budget.consume_tokens(tokens)

        visible_ids = sorted(set(shared_a.keys()) | {c.evidence_id for c in private_web_cards})
        submission = ExpertSubmission(
            expert_role=role,
            model_role=worker_input["model_role"],
            visible_evidence_ids=visible_ids,
            private_evidence=private_web_cards,
            claims=claims,
            key_dimensions_covered=dims,
            sub_budget=sub_budget,
            tokens_used=tokens,
        )
        rt.harness.audit.record(
            node_name="expert_worker",
            caller=role.value,
            action_type="node",
            target=f"expert:{role.value}",
            status="ok",
            tokens_used=tokens,
            detail={
                "model_role": worker_input["model_role"],
                "claims": len(claims),
                "searches_used": sub_budget.searches_used,
                "fetches_used": sub_budget.fetches_used,
            },
        )
        return {"expert_submissions": [submission]}

    def blackboard_verify_node(state: DeepResearchState) -> DeepResearchState:
        """汇总证据黑板 → 检查首轮隔离违规引用 → 执行纯规则 RuleVerifier 逐条核验。"""
        rt.harness.check_not_revoked("blackboard_verify", caller="blackboard")
        as_of = date.fromisoformat(state["as_of"])
        submissions = state.get("expert_submissions") or []

        # 汇总全局证据黑板（跨专家网页再做一次同簇关联，防止多专家各搜到同篇转载稿）
        blackboard_ev: dict[str, EvidenceCard] = dict(state.get("shared_evidence_store") or {})
        all_private_web: list[EvidenceCard] = []
        for sub in submissions:
            for card in sub.private_evidence:
                all_private_web.append(card)
                blackboard_ev[card.evidence_id] = card

        # 跨专家网页转载聚类统一（如果不同专家抓到了相同 SimHash / 同原始来源的网页，合并到同一事件簇 ID）
        _unify_cross_expert_clusters(all_private_web, blackboard_ev)

        all_claims: list[Claim] = []
        verification_results: list[VerificationResult] = []
        isolation_violations: list[str] = []
        total_expert_tokens = int(state.get("deep_tokens_used", 0))

        for sub in submissions:
            total_expert_tokens += sub.tokens_used
            visible_set = set(sub.visible_evidence_ids)
            for claim in sub.claims:
                all_claims.append(claim)
                # 首轮隔离硬约束：引用了不在自己可见集合里的证据 ID → 直接驳回（对应文档第 13 题）
                forbidden_refs = [eid for eid in claim.evidence_ids if eid not in visible_set]
                if forbidden_refs:
                    msg = f"专家 {sub.expert_role.value} 的主张 {claim.claim_id} 违规引用不可见证据：{forbidden_refs}"
                    isolation_violations.append(msg)
                    verification_results.append(
                        VerificationResult(
                            claim_id=claim.claim_id,
                            status=VerificationStatus.REJECTED,
                            valid_evidence_ids=[],
                            ignored_evidence_ids=forbidden_refs,
                            reasons=[f"违反首轮隔离约束：{msg}"],
                        )
                    )
                else:
                    vr = rt.verifier.verify_claim(claim, blackboard_ev, as_of=as_of)
                    verification_results.append(vr)

        rt.harness.audit.record(
            node_name="blackboard_verify",
            caller="code_rule_verifier",
            action_type="node",
            target="blackboard_verify",
            status="ok",
            detail={
                "total_claims": len(all_claims),
                "verified": sum(1 for r in verification_results if r.status == VerificationStatus.VERIFIED),
                "doubtful": sum(1 for r in verification_results if r.status == VerificationStatus.DOUBTFUL),
                "rejected": sum(1 for r in verification_results if r.status == VerificationStatus.REJECTED),
                "isolation_violations": len(isolation_violations),
            },
        )
        return {
            "blackboard_evidence": blackboard_ev,
            "blackboard_claims": all_claims,
            "verification_results": verification_results,
            "isolation_violations": isolation_violations,
            "deep_tokens_used": total_expert_tokens,
        }

    def adversary_challenge_node(state: DeepResearchState) -> DeepResearchState:
        """反证官专门挑毛病（模型轴绑定 reason，最多提 8 条质询）。"""
        rt.harness.check_not_revoked("adversary_challenge", caller="adversary")
        challenges, tokens = rt.engine.generate_challenges(
            question=state["question"],
            claims=state.get("blackboard_claims") or [],
            verification_results=state.get("verification_results") or [],
            evidence_store=state.get("blackboard_evidence") or {},
        )
        challenges = challenges[:8]  # 硬性上限 8 条
        rt.harness.budget.consume_tokens(tokens)
        rt.harness.audit.record(
            node_name="adversary_challenge",
            caller="adversary_challenger",
            action_type="llm",
            target="reason:adversary",
            status="ok",
            tokens_used=tokens,
            detail={"challenge_count": len(challenges)},
        )
        return {
            "challenges": challenges,
            "deep_tokens_used": int(state.get("deep_tokens_used", 0)) + tokens,
        }

    def expert_respond_node(state: DeepResearchState) -> DeepResearchState:
        """被质询专家回应 1 轮。"""
        rt.harness.check_not_revoked("expert_respond", caller="experts")
        challenges = state.get("challenges") or []
        if not challenges:
            return {"challenge_responses": []}

        responses, tokens = rt.engine.respond_to_challenges(
            challenges=challenges,
            claims=state.get("blackboard_claims") or [],
            evidence_store=state.get("blackboard_evidence") or {},
        )
        rt.harness.budget.consume_tokens(tokens)
        rt.harness.audit.record(
            node_name="expert_respond",
            caller="experts",
            action_type="llm",
            target="draft:expert_respond",
            status="ok",
            tokens_used=tokens,
            detail={"response_count": len(responses)},
        )
        return {
            "challenge_responses": responses,
            "deep_tokens_used": int(state.get("deep_tokens_used", 0)) + tokens,
        }

    def chair_adjudicate_node(state: DeepResearchState) -> DeepResearchState:
        """主席按独立证据裁决（非多数投票制 + 高危未决争议 interrupt 暂停等人复核）。"""
        rt.harness.check_not_revoked("chair_adjudicate", caller="chair")

        claims = state.get("blackboard_claims") or []
        v_results = state.get("verification_results") or []
        v_map = {r.claim_id: r for r in v_results}
        ev_store = state.get("blackboard_evidence") or {}
        challenges = state.get("challenges") or []
        resp_map = {r.challenge_id: r for r in (state.get("challenge_responses") or [])}

        claim_map = {c.claim_id: c for c in claims}
        upheld_ids: list[str] = []
        blocked_by_challenge: set[str] = set()
        unresolved_high_ids: list[str] = []
        dissents: list[DissentingOpinion] = []

        for chg in challenges:
            resp = resp_map.get(chg.challenge_id)
            target_claim = claim_map.get(chg.target_claim_id)
            target_text = target_claim.text if target_claim else chg.target_claim_id

            if chg.is_causal_fallacy or (resp and resp.action == ChallengeResponseAction.CONCEDE):
                # 质询成立或专家认错 → 主席绝对不能采纳该主张！
                upheld_ids.append(chg.challenge_id)
                blocked_by_challenge.add(chg.target_claim_id)
            elif chg.severity == ChallengeSeverity.HIGH and (
                resp is None or resp.action == ChallengeResponseAction.DEFEND
            ):
                # 高严重度且未解决争议 → 记入未决列表，提交前必须人工复核
                unresolved_high_ids.append(chg.challenge_id)
                dissents.append(
                    DissentingOpinion(
                        topic=f"高严重度未决争议（{chg.challenge_id}）",
                        bull_or_main_view=target_text,
                        bear_or_counter_view=chg.critique,
                        independent_evidence_ids=list(
                            dict.fromkeys((target_claim.evidence_ids if target_claim else []) + chg.counter_evidence_ids)
                        ),
                    )
                )
            else:
                # 双方各有证据支撑的中低分歧 → 保留意见写进档案，不强行统一
                dissents.append(
                    DissentingOpinion(
                        topic=f"保留分歧（{chg.challenge_id}）",
                        bull_or_main_view=target_text,
                        bear_or_counter_view=chg.critique,
                        independent_evidence_ids=list(
                            dict.fromkeys((target_claim.evidence_ids if target_claim else []) + chg.counter_evidence_ids)
                        ),
                    )
                )

        # 按独立来源计数（非多数投票）并严格筛选可进入结论的主张
        indep_counts: dict[str, int] = {}
        final_claims: list[Claim] = []
        excluded_claims: list[Claim] = []

        # 额外：按独立来源聚合同主题网页主张（即便 3 位专家复述同一转载源，也标明独立来源=1）
        for claim in claims:
            vr = v_map.get(claim.claim_id)
            indep_counts[claim.claim_id] = count_independent_sources(claim.evidence_ids, ev_store)
            if vr is None or not vr.can_enter_conclusion or claim.claim_id in blocked_by_challenge:
                excluded_claims.append(claim)
            else:
                final_claims.append(claim)

        chair_tokens = 1800
        rt.harness.budget.consume_tokens(chair_tokens)

        human_review_required = bool(unresolved_high_ids)
        human_review_approved: bool | None = None
        aborted = False

        # 对应文档第 11.2 题 & 第 14 题：高严重度未决争议提交前 interrupt 暂停等人复核
        if human_review_required:
            rt.harness.audit.record(
                node_name="chair_adjudicate",
                caller="chair",
                action_type="approval",
                target="unresolved_high_severity_dispute",
                status="interrupted",
                detail={"unresolved_challenge_ids": unresolved_high_ids},
            )
            human_reply = interrupt(
                {
                    "type": "unresolved_high_severity_dispute",
                    "message": "深研存在高严重度未解决争议，是否确认以“部分结论（保留分歧）”形式提交档案？",
                    "unresolved_challenge_ids": unresolved_high_ids,
                    "dissents": [d.model_dump() for d in dissents],
                }
            )
            human_review_approved = (
                bool(human_reply.get("approved", False))
                if isinstance(human_reply, dict)
                else bool(human_reply)
            )
            if not human_review_approved:
                aborted = True

        adjudication = AdjudicationDecision(
            summary=(
                f"多专家深研完成：召集 {len(state.get('selected_experts') or [])} 位专家 + 反证官；"
                f"采纳 {len(final_claims)} 条独立证据支撑主张，拦截 {len(excluded_claims)} 条"
                f"（含 {len(upheld_ids)} 条质询成立主张），保留 {len(dissents)} 条分歧意见。"
            ),
            adopted_claim_ids=[c.claim_id for c in final_claims],
            blocked_claim_ids=[c.claim_id for c in excluded_claims],
            upheld_challenge_ids=upheld_ids,
            unresolved_high_severity_ids=unresolved_high_ids,
            dissenting_opinions=dissents,
            independent_source_counts=indep_counts,
            human_review_required=human_review_required,
            human_review_approved=human_review_approved,
        )

        rt.harness.audit.record(
            node_name="chair_adjudicate",
            caller="chair_adjudicator",
            action_type="llm",
            target="reason:chair",
            status="denied" if aborted else "ok",
            tokens_used=chair_tokens,
            detail={
                "adopted": len(final_claims),
                "blocked": len(excluded_claims),
                "upheld_challenges": upheld_ids,
                "dissents": len(dissents),
                "human_review_required": human_review_required,
                "human_review_approved": human_review_approved,
            },
        )
        return {
            "adjudication": adjudication,
            "final_claims": final_claims,
            "excluded_claims": excluded_claims,
            "aborted_by_human_review": aborted,
            "deep_tokens_used": int(state.get("deep_tokens_used", 0)) + chair_tokens,
        }

    builder = StateGraph(DeepResearchState)
    builder.add_node("chair_convene", chair_convene_node)
    builder.add_node("expert_worker", expert_worker_node)
    builder.add_node("blackboard_verify", blackboard_verify_node)
    builder.add_node("adversary_challenge", adversary_challenge_node)
    builder.add_node("expert_respond", expert_respond_node)
    builder.add_node("chair_adjudicate", chair_adjudicate_node)

    builder.add_edge(START, "chair_convene")
    builder.add_conditional_edges("chair_convene", fanout_to_experts, ["expert_worker"])
    builder.add_edge("expert_worker", "blackboard_verify")
    builder.add_edge("blackboard_verify", "adversary_challenge")
    builder.add_edge("adversary_challenge", "expert_respond")
    builder.add_edge("expert_respond", "chair_adjudicate")
    builder.add_edge("chair_adjudicate", END)

    return builder.compile(checkpointer=checkpointer)


def _unify_cross_expert_clusters(
    web_cards: list[EvidenceCard],
    blackboard_ev: dict[str, EvidenceCard],
) -> None:
    """跨专家统一转载事件簇：避免 A 专家和 B 专家分别抓到同一转载稿时 cluster_id 各自独立。"""
    from zhengyan.search.simhash import hamming_distance, normalize_title

    reps: list[tuple[str, str, int | None, str | None]] = []
    for card in web_cards:
        norm_t = normalize_title(card.title)
        matched_cid: str | None = None
        for cid, r_title, r_fp, r_orig in reps:
            same_title = bool(norm_t) and norm_t == r_title
            near_dup = (
                card.simhash is not None
                and r_fp is not None
                and card.simhash != 0
                and r_fp != 0
                and hamming_distance(card.simhash, r_fp) <= 10
            )
            same_orig = bool(card.original_source) and card.original_source == r_orig
            if same_title or near_dup or same_orig:
                matched_cid = cid
                break

        if matched_cid is None:
            matched_cid = f"BB-CLU-{len(reps) + 1:03d}"
            reps.append((matched_cid, norm_t, card.simhash, card.original_source))

        blackboard_ev[card.evidence_id] = card.model_copy(update={"event_cluster_id": matched_cid})
