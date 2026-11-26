"""确定性离线 MockLLMClient（用于测试与无 Key 本地学习）。

支持通过构造参数切换可控测试行为：
- `inject_hallucination=True`：在起草阶段故意编造一条“毛利率同比提升 3 个百分点”的幻觉主张，用于直观验证核验器拦截。
- `simulate_gap_on_round_0=True`：第 0 轮故意只起草少量主张并留下证据缺口，用于验证 quality_gate 自动退回 plan 补证 1 轮。
- `simulate_high_disagreement=True`：输出 >=3 条反方观点与 >=2 条存疑主张，用于验证标准研究向深研升级检查点。
- 记录所有收到的 prompt（`received_prompts`），便于测试验证 Harness 脱敏确实生效。
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from zhengyan.core.enums import ClaimType, EvidenceGrade, EvidenceSourceType, UserIntent
from zhengyan.core.models import Claim, DraftReport, EvidenceCard, ResearchPlan
from zhengyan.llm.base import StructuredLLMResponse

TModel = TypeVar("TModel", bound=BaseModel)


class MockLLMClient:
    """确定性离线结构化模型模拟器。"""

    def __init__(
        self,
        *,
        inject_hallucination: bool = False,
        simulate_gap_on_round_0: bool = False,
        simulate_high_disagreement: bool = False,
    ) -> None:
        self.inject_hallucination = inject_hallucination
        self.simulate_gap_on_round_0 = simulate_gap_on_round_0
        self.simulate_high_disagreement = simulate_high_disagreement
        self.received_prompts: list[str] = []

    def generate_structured(
        self,
        *,
        role: str,
        prompt: str,
        response_model: type[TModel],
        context: dict[str, Any],
    ) -> StructuredLLMResponse[TModel]:
        self.received_prompts.append(prompt)

        if response_model is ResearchPlan:
            plan = self._build_plan(context)
            return StructuredLLMResponse(
                parsed=plan,  # type: ignore[arg-type]
                tokens_used=900,
                model_name=f"mock-{role}",
            )

        if response_model is DraftReport:
            draft = self._build_draft(context)
            return StructuredLLMResponse(
                parsed=draft,  # type: ignore[arg-type]
                tokens_used=2400,
                model_name=f"mock-{role}",
            )

        raise NotImplementedError(f"MockLLMClient 暂未配置对 {response_model.__name__} 的模拟逻辑")

    def _build_plan(self, context: dict[str, Any]) -> ResearchPlan:
        symbol = context.get("symbol") or "未知标的"
        company = context.get("company_name") or symbol
        gaps: list[str] = context.get("evidence_gaps") or []
        round_idx: int = int(context.get("supplement_rounds", 0))

        if gaps and round_idx > 0:
            return ResearchPlan(
                queries=[f"{company} {gaps[0]} 财报披露"],
                risk_queries=[f"{company} 渠道库存 风险 反证"],
                target_metrics=["revenue_yoy", "net_profit_yoy", "gross_margin", "pe_ttm"],
                rationale=f"第 {round_idx} 轮补证：针对缺口补充权威披露与反证线索",
            )

        return ResearchPlan(
            queries=[f"{company} 最新定期报告 业绩 现金流"],
            risk_queries=[f"{company} 合同负债下降 渠道风险 质疑"],
            target_metrics=["revenue_yoy", "net_profit_yoy", "contract_liabilities_yoy", "ocf_to_np", "pe_ttm"],
            rationale="首轮取数与风险反证检索规划",
        )

    def _build_draft(self, context: dict[str, Any]) -> DraftReport:
        intent = context.get("intent", UserIntent.RESEARCH)
        if intent == UserIntent.CONCEPT:
            question = str(context.get("question", ""))
            return DraftReport(
                summary=f"概念解释：{question}。市盈率（PE）等于总市值除以净利润，反映市场为每 1 元盈利支付的估值倍数。",
                claims=[
                    Claim(
                        claim_id="CLM-CONCEPT-1",
                        text="市盈率（PE）是总市值与净利润的比值，常用于衡量估值高低。",
                        claim_type=ClaimType.HYPOTHESIS,
                        evidence_ids=[],
                    )
                ],
                evidence_gaps=[],
                counterpoints=[],
            )

        evidence_store: dict[str, EvidenceCard] = context.get("evidence_store") or {}
        round_idx: int = int(context.get("supplement_rounds", 0))
        a_cards = [c for c in evidence_store.values() if c.grade == EvidenceGrade.A]
        b_cards = [c for c in evidence_store.values() if c.grade == EvidenceGrade.B]

        # 模拟首轮证据不足，触发 quality_gate 退回补证
        if self.simulate_gap_on_round_0 and round_idx == 0:
            fin_cards = [c for c in a_cards if c.source_type == EvidenceSourceType.FINANCIAL_REPORT]
            first_fin = fin_cards[-1] if fin_cards else None
            claims: list[Claim] = []
            if first_fin and first_fin.metrics:
                rev = next((m for m in first_fin.metrics if m.name == "revenue"), first_fin.metrics[0])
                claims.append(
                    Claim(
                        claim_id="CLM-R0-1",
                        text=f"{first_fin.period} {rev.display_name}为 {rev.value}{rev.unit.value}。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=[first_fin.evidence_id],
                    )
                )
            return DraftReport(
                summary="首轮材料不足，尚缺利润质量与反证核实。",
                claims=claims,
                evidence_gaps=["缺少同口径现金流质量与合同负债变动原因核实"],
                counterpoints=[],
            )

        claims = []
        idx = 1

        # 从 A 级财报和计算卡中生成可复算事实主张
        for card in a_cards:
            if card.source_type == EvidenceSourceType.COMPUTE and card.metrics:
                m = card.metrics[-1]
                if m.value < 0:
                    text = f"{card.period or ''} {m.display_name}下降 {abs(m.value)}{m.unit.value}。"
                else:
                    text = f"{card.period or ''} {m.display_name}为 {m.value}{m.unit.value}。"
                claims.append(
                    Claim(
                        claim_id=f"CLM-{idx}",
                        text=text.strip(),
                        claim_type=ClaimType.FACT,
                        evidence_ids=[card.evidence_id],
                    )
                )
                idx += 1
            elif card.source_type == EvidenceSourceType.FINANCIAL_REPORT and card.period and card.period.startswith("2025"):
                rev = next((m for m in card.metrics if m.name == "revenue"), None)
                np_m = next((m for m in card.metrics if m.name == "net_profit"), None)
                if rev and np_m:
                    claims.append(
                        Claim(
                            claim_id=f"CLM-{idx}",
                            text=f"{card.period} 实现营业收入 {rev.value}{rev.unit.value}，归母净利润 {np_m.value}{np_m.unit.value}。",
                            claim_type=ClaimType.FACT,
                            evidence_ids=[card.evidence_id],
                        )
                    )
                    idx += 1

        # 若存在 B 级网页证据，生成一条说明“发生了什么”的不含数字事件主张
        if b_cards:
            claims.append(
                Claim(
                    claim_id=f"CLM-{idx}",
                    text="渠道调研报道指出，公司近期调整了经销商打款与排期节奏。",
                    claim_type=ClaimType.FACT,
                    evidence_ids=[c.evidence_id for c in b_cards[:2]],
                )
            )
            idx += 1

        # 添加一条情景推演（非事实）
        claims.append(
            Claim(
                claim_id=f"CLM-{idx}",
                text="若后续终端动销保持平稳，则盈利与现金流质量有望延续当前水平。",
                claim_type=ClaimType.SCENARIO,
                evidence_ids=[],
            )
        )
        idx += 1

        # 可选：故意注入模型经典数字幻觉（供核验器拦截演示）
        if self.inject_hallucination:
            fin_ids = [c.evidence_id for c in a_cards if c.source_type == EvidenceSourceType.FINANCIAL_REPORT][:2]
            claims.append(
                Claim(
                    claim_id=f"CLM-HALLUCINATED",
                    text="本期毛利率同比提升 3 个百分点，盈利能力大幅跃升。",
                    claim_type=ClaimType.FACT,
                    evidence_ids=fin_ids or [a_cards[0].evidence_id] if a_cards else [],
                )
            )

        counterpoints = ["需关注合同负债变动是否反映渠道去库压力"]
        if self.simulate_high_disagreement:
            fin_ids = [c.evidence_id for c in a_cards if c.source_type == EvidenceSourceType.FINANCIAL_REPORT][:1]
            for j in range(2):
                claims.append(
                    Claim(
                        claim_id=f"CLM-UNVERIFIED-{j+1}",
                        text=f"未经财报证实的估算显示渠道压货率上升 {18 + j}%。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=fin_ids,
                    )
                )
            counterpoints = [
                "合同负债同比大幅下降，存在终端需求走弱风险",
                "批价承压可能侵蚀渠道利润空间",
                "当前估值隐含永续增长假设偏乐观",
            ]

        return DraftReport(
            summary=f"基于截至研究时点的已披露财报、计算指标与清洗后报道形成研究草稿（共 {len(claims)} 条主张）。",
            claims=claims,
            evidence_gaps=[] if not self.simulate_high_disagreement else ["多方与空方核心分歧尚未经对抗质询收敛"],
            counterpoints=counterpoints,
        )
