"""多专家深研确定性模拟器与协议（对应文档第 13、14、15、27 题）。

完整还原文档中的多个真实核心场景：
1. 贵州茅台合同负债 -42% 经典对抗：
   - 基本面专家写出“合同负债同比下降 42.00%，说明终端需求走弱”（数字可复算、核验器放行，但因果推断有漏洞！）
   - 反证官提出高严重度质询：“合同负债同比下降 42% 也可能是渠道打款模式与配额节奏调整，不能直接推出需求走弱”
   - 基本面专家回应 concede / 主席裁定质询成立 → 该因果主张被精准拦截，不进入结论！
2. 专家派生数字被规则核验器标存疑（文档第 15 题观察）：
   - 估值专家可能写出证据里未直接出现的推断型派生数字（如“估算未来三年复合增速 11%”），核验器标为 DOUBTFUL 拦在结论外，复现“深研数字可复算率略低于单 Agent，但整体覆盖与质量更高”的真实指标特征。
3. 可控故障/边界注入（供测试验证治理机制）：
   - `inject_isolation_violation=True`：模拟某专家违规引用其他专家私有证据 ID → 被隔离核验拦截。
   - `inject_unresolved_high_dispute=True`：模拟高严重度争议未获解决 → 触发 LangGraph `interrupt()` 暂停等人复核。
   - `inject_reprint_echo=True`：模拟 3 位专家全部引用同一篇转载新闻的不同站点 URL → 独立来源计数仍为 1（打破多数投票幻觉）。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from zhengyan.core.enums import (
    ChallengeResponseAction,
    ChallengeSeverity,
    ClaimType,
    EvidenceGrade,
    EvidenceSourceType,
    ExpertRole,
    VerificationStatus,
)
from zhengyan.core.models import (
    ChallengeItem,
    ChallengeResponse,
    Claim,
    DissentingOpinion,
    EvidenceCard,
    ExpertSubmission,
    RawWebResult,
    VerificationResult,
)

# 按文档第 13 题：6 位可选专家，主席每次最多召集 5 位
ALL_OPTIONAL_EXPERTS: tuple[ExpertRole, ...] = (
    ExpertRole.FUNDAMENTAL,
    ExpertRole.INDUSTRY,
    ExpertRole.VALUATION,
    ExpertRole.TECHNICAL,
    ExpertRole.EVENT,
    ExpertRole.RISK,
)


class DeepResearchEngine(Protocol):
    """深研各角色推理逻辑抽象接口（离线用 MockDeepResearchEngine，线上可接 DeepSeek draft/reason）。"""

    def convene_experts(
        self,
        question: str,
        shared_evidence: dict[str, EvidenceCard],
    ) -> tuple[list[ExpertRole], str]:
        ...

    def expert_queries(self, role: ExpertRole, symbol: str | None, question: str) -> list[str]:
        ...

    def expert_analyze(
        self,
        role: ExpertRole,
        *,
        question: str,
        symbol: str | None,
        shared_a_cards: dict[str, EvidenceCard],
        own_web_cards: list[EvidenceCard],
    ) -> tuple[list[Claim], list[str], int]:
        ...

    def generate_challenges(
        self,
        *,
        question: str,
        claims: list[Claim],
        verification_results: list[VerificationResult],
        evidence_store: dict[str, EvidenceCard],
    ) -> tuple[list[ChallengeItem], int]:
        ...

    def respond_to_challenges(
        self,
        *,
        challenges: list[ChallengeItem],
        claims: list[Claim],
        evidence_store: dict[str, EvidenceCard],
    ) -> tuple[list[ChallengeResponse], int]:
        ...


def build_expert_private_web_results(
    role: ExpertRole,
    symbol: str | None,
    as_of: date,
    *,
    reprint_echo_mode: bool = False,
) -> list[RawWebResult]:
    """为不同专家生成隔离检索到的原始网页（每名专家独立子预算最多 2 搜 2 抓）。"""
    pub = min(date(2025, 8, 14), as_of)

    if reprint_echo_mode:
        # 故意让多位专家搜到同一篇转载稿的不同媒体站点，测试非投票同源合并
        site_map = {
            ExpertRole.FUNDAMENTAL: ("https://www.stcn.com/echo/1.html", "渠道调研称茅台打款节奏调整 | 证券时报网"),
            ExpertRole.INDUSTRY: ("https://finance.sina.com.cn/echo/2.shtml", "渠道调研称茅台打款节奏调整_新浪财经"),
            ExpertRole.EVENT: ("https://finance.eastmoney.com/echo/3.html", "渠道调研称茅台打款节奏调整 - 东方财富财经"),
        }
        url, title = site_map.get(
            role,
            ("https://www.cnstock.com/echo/4.html", "渠道调研称茅台打款节奏调整 | 上海证券报"),
        )
        return [
            RawWebResult(
                url=url,
                title=title,
                published_at=pub,
                fetched=True,
                raw_html="<p>来源：证券时报。渠道调研称公司近期优化经销商打款与排期节奏，终端库存结构分化。</p>",
            )
        ]

    if symbol == "300750":
        mapping: dict[ExpertRole, RawWebResult] = {
            ExpertRole.INDUSTRY: RawWebResult(
                url="https://www.caixin.com/catl/industry.html",
                title="动力电池海外装车量稳步增长但行业价格竞争加剧 | 财新网",
                published_at=pub,
                fetched=True,
                raw_html="<p>来源：财新网。欧洲及储能市场需求带动头部电池厂出货增长，但二三线厂商降价使价格竞争延续。</p>",
            ),
            ExpertRole.EVENT: RawWebResult(
                url="https://www.stcn.com/catl/event.html",
                title="宁德时代匈牙利工厂推进产线调试 | 证券时报网",
                published_at=pub,
                fetched=True,
                raw_html="<p>来源：证券时报。公司海外本土化产能稳步推进，有助于应对贸易壁垒。</p>",
            ),
            ExpertRole.RISK: RawWebResult(
                url="https://www.yicai.com/catl/risk.html",
                title="碳酸锂波动与海外关税政策成为电池产业链主要不确定性 | 第一财经",
                published_at=pub,
                fetched=True,
                raw_html="<p>来源：第一财经。行业人士提示原材料价格波动与欧美政策变化需持续跟踪。</p>",
            ),
        }
        item = mapping.get(role)
        return [item] if item else []

    # 贵州茅台 600519 各专家独立检索结果
    mapping_mt: dict[ExpertRole, RawWebResult] = {
        ExpertRole.FUNDAMENTAL: RawWebResult(
            url="https://www.cnstock.com/mt/fundamental.html",
            title="白酒半年报复盘：头部酒企现金流与毛利率维持高位 | 上海证券报",
            published_at=pub,
            fetched=True,
            raw_html="<p>来源：上海证券报。头部白酒企业营收与归母净利润保持正增长，毛利率处于行业领先水平。</p>",
        ),
        ExpertRole.INDUSTRY: RawWebResult(
            url="https://www.stcn.com/mt/channel.html",
            title="渠道调研：茅台合同负债变动受经销商打款节奏与配额调整影响 | 证券时报网",
            published_at=pub,
            fetched=True,
            raw_html="<p>来源：证券时报。多位经销商调研反馈，合同负债波动与公司调整打款政策及直营配额节奏密切相关，终端动销表现分化。</p>",
        ),
        ExpertRole.VALUATION: RawWebResult(
            url="https://www.caixin.com/mt/val.html",
            title="消费龙头估值回落至历史中低分位引发多空分歧 | 财新网",
            published_at=pub,
            fetched=True,
            raw_html="<p>来源：财新网。机构对高股息消费龙头的合理估值倍数存在明显分歧，部分资金关注增速换挡风险。</p>",
        ),
        ExpertRole.EVENT: RawWebResult(
            url="https://www.cls.cn/mt/event.html",
            title="批价跟踪：飞天茅台渠道批价阶段性波动 | 财联社",
            published_at=pub,
            fetched=True,
            raw_html="<p>来源：财联社。近期经销商反馈批价呈阶段性波动，节前动销节奏成为市场观察重点。</p>",
        ),
        ExpertRole.RISK: RawWebResult(
            url="https://www.yicai.com/mt/risk.html",
            title="白酒周期下行期需警惕渠道库存与价格倒挂风险 | 第一财经",
            published_at=pub,
            fetched=True,
            raw_html="<p>来源：第一财经。分析人士指出宏观商务需求偏弱与渠道库存消化进度是主要下行风险。</p>",
        ),
    }
    item = mapping_mt.get(role)
    return [item] if item else []


class MockDeepResearchEngine:
    """确定性离线多专家深研引擎（含可控测试开关）。"""

    def __init__(
        self,
        *,
        inject_isolation_violation: bool = False,
        inject_unresolved_high_dispute: bool = False,
        inject_expert_derived_number: bool = True,
        reprint_echo_mode: bool = False,
    ) -> None:
        self.inject_isolation_violation = inject_isolation_violation
        self.inject_unresolved_high_dispute = inject_unresolved_high_dispute
        self.inject_expert_derived_number = inject_expert_derived_number
        self.reprint_echo_mode = reprint_echo_mode

    def convene_experts(
        self,
        question: str,
        shared_evidence: dict[str, EvidenceCard],
    ) -> tuple[list[ExpertRole], str]:
        """按证据缺口召集最多 5 位专家（绝不超过 5 位）。"""
        selected = [
            ExpertRole.FUNDAMENTAL,
            ExpertRole.INDUSTRY,
            ExpertRole.VALUATION,
            ExpertRole.EVENT,
            ExpertRole.RISK,
        ]
        if "资金" in question or "技术" in question:
            selected[3] = ExpertRole.TECHNICAL
        return selected[:5], "围绕业绩质量、行业渠道、估值安全边际、事件跟踪与下行风险召集 5 位专家并行隔离取证"

    def expert_queries(self, role: ExpertRole, symbol: str | None, question: str) -> list[str]:
        tag = symbol or "标的"
        # 每位专家至多 2 条查询（契合独立子预算 max_searches=2）
        role_kw = {
            ExpertRole.FUNDAMENTAL: [f"{tag} 定期报告 利润质量 现金流"],
            ExpertRole.INDUSTRY: [f"{tag} 渠道调研 行业景气 库存"],
            ExpertRole.VALUATION: [f"{tag} 估值 PE 分位 机构分歧"],
            ExpertRole.TECHNICAL: [f"{tag} 资金流向 成交量 趋势"],
            ExpertRole.EVENT: [f"{tag} 公告 批价 产能 最新进展"],
            ExpertRole.RISK: [f"{tag} 下行风险 质疑 反例"],
        }
        return role_kw[role]

    def expert_analyze(
        self,
        role: ExpertRole,
        *,
        question: str,
        symbol: str | None,
        shared_a_cards: dict[str, EvidenceCard],
        own_web_cards: list[EvidenceCard],
    ) -> tuple[list[Claim], list[str], int]:
        a_list = list(shared_a_cards.values())
        fin_cards = [c for c in a_list if c.source_type == EvidenceSourceType.FINANCIAL_REPORT]
        cmp_cards = [c for c in a_list if c.source_type == EvidenceSourceType.COMPUTE]
        latest_fin = max(fin_cards, key=lambda c: c.published_at or date.min) if fin_cards else None
        own_b_ids = [c.evidence_id for c in own_web_cards if c.grade == EvidenceGrade.B]

        claims: list[Claim] = []
        dims: list[str] = []

        if role == ExpertRole.FUNDAMENTAL:
            dims.extend(["profit_quality", "cashflow"])
            if latest_fin:
                rev = next((m for m in latest_fin.metrics if m.name == "revenue"), None)
                np_m = next((m for m in latest_fin.metrics if m.name == "net_profit"), None)
                if rev and np_m:
                    claims.append(
                        Claim(
                            claim_id="EXP-FUN-1",
                            text=f"{latest_fin.period} 营业收入 {rev.value}{rev.unit.value}，归母净利润 {np_m.value}{np_m.unit.value}。",
                            claim_type=ClaimType.FACT,
                            evidence_ids=[latest_fin.evidence_id],
                            author=role.value,
                        )
                    )
            ocf_card = next((c for c in cmp_cards if "ocf_to_np" in c.evidence_id), None)
            if ocf_card and ocf_card.metrics:
                m = ocf_card.metrics[0]
                claims.append(
                    Claim(
                        claim_id="EXP-FUN-2",
                        text=f"本期{m.display_name}为 {m.value}{m.unit.value}，现金回款质量保持稳健。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=[ocf_card.evidence_id],
                        author=role.value,
                    )
                )
            # 还原文档经典案例：合同负债 -42% 的因果归因主张（数字完全对，但因果推断将被反证官质询拦下！）
            cl_yoy = next((c for c in cmp_cards if "contract_liabilities" in c.evidence_id), None)
            if cl_yoy and cl_yoy.metrics:
                m = cl_yoy.metrics[0]
                claims.append(
                    Claim(
                        claim_id="EXP-FUN-CL-CAUSAL",
                        text=f"{cl_yoy.period} 合同负债同比下降 {abs(m.value)}%，直接说明终端消费需求明显走弱。",
                        claim_type=ClaimType.INFERENCE,
                        evidence_ids=[cl_yoy.evidence_id],
                        author=role.value,
                    )
                )
            elif own_b_ids:
                claims.append(
                    Claim(
                        claim_id="EXP-FUN-WEB",
                        text="公开报道显示公司核心主业盈利能力保持行业前列。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=own_b_ids,
                        author=role.value,
                    )
                )
            if self.reprint_echo_mode and own_b_ids:
                claims.append(
                    Claim(
                        claim_id="EXP-FUN-ECHO",
                        text="媒体渠道调研反映公司近期优化经销商打款节奏。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=own_b_ids,
                        author=role.value,
                    )
                )

        elif role == ExpertRole.INDUSTRY:
            dims.append("channel_industry")
            if own_b_ids:
                claims.append(
                    Claim(
                        claim_id="EXP-IND-1",
                        text="行业与渠道调研显示企业正在主动调整渠道排期与竞争策略，下游需求呈结构性分化。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=own_b_ids,
                        author=role.value,
                    )
                )
            # 可控注入：故意引用另一位专家（RISK）的私有证据 ID，测试首轮隔离硬拦截
            if self.inject_isolation_violation:
                claims.append(
                    Claim(
                        claim_id="EXP-IND-LEAK",
                        text="跨视角偷看风险专家私有材料得出的结论。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=["EV-EXP-risk-WEB-001"],
                        author=role.value,
                    )
                )

        elif role == ExpertRole.VALUATION:
            dims.append("valuation")
            pe_card = next((c for c in cmp_cards if "PETTM" in c.evidence_id), None)
            if pe_card:
                pe_m = next((m for m in pe_card.metrics if m.name == "pe_ttm"), None)
                if pe_m:
                    claims.append(
                        Claim(
                            claim_id="EXP-VAL-1",
                            text=f"按披露日对齐计算的当前市盈率 PE(TTM) 为 {pe_m.value}{pe_m.unit.value}。",
                            claim_type=ClaimType.FACT,
                            evidence_ids=[pe_card.evidence_id],
                            author=role.value,
                        )
                    )
            if own_b_ids:
                claims.append(
                    Claim(
                        claim_id="EXP-VAL-2",
                        text="市场机构对龙头估值中枢存在多空分歧，核心争议集中在增速换挡后的合理倍数。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=own_b_ids,
                        author=role.value,
                    )
                )
            # 复现文档第 15 题：深研专家偶尔写出未经 A 级证据证实的估算派生数字，被核验器标为存疑拦在结论外
            if self.inject_expert_derived_number and latest_fin:
                claims.append(
                    Claim(
                        claim_id="EXP-VAL-DERIVED-EST",
                        text="按折现模型反推市场隐含未来三年业绩复合增速约 11.5%。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=[latest_fin.evidence_id],
                        author=role.value,
                    )
                )

        elif role in (ExpertRole.EVENT, ExpertRole.TECHNICAL):
            dims.append("channel_industry")
            if own_b_ids:
                claims.append(
                    Claim(
                        claim_id="EXP-EVT-1",
                        text="近期公开跟踪报道显示渠道价格与产能推进节奏仍是市场短期博弈焦点。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=own_b_ids,
                        author=role.value,
                    )
                )

        elif role == ExpertRole.RISK:
            dims.append("risk_counterpoint")
            if own_b_ids:
                claims.append(
                    Claim(
                        claim_id="EXP-RSK-1",
                        text="行业周期波动、渠道库存消化或外部政策变化构成主要下行风险，需持续跟踪。",
                        claim_type=ClaimType.FACT,
                        evidence_ids=own_b_ids,
                        author=role.value,
                    )
                )
            claims.append(
                Claim(
                    claim_id="EXP-RSK-SCENARIO",
                    text="若未来两个季度核心产品批价或毛利率连续下滑，则原有盈利预期将面临下修风险。",
                    claim_type=ClaimType.SCENARIO,
                    evidence_ids=[],
                    author=role.value,
                )
            )

        # 估值研究员走 reason 模型消耗略高，其余专家走 draft 模型
        tokens = 1800 if role == ExpertRole.VALUATION else 1200
        return claims, dims, tokens

    def generate_challenges(
        self,
        *,
        question: str,
        claims: list[Claim],
        verification_results: list[VerificationResult],
        evidence_store: dict[str, EvidenceCard],
    ) -> tuple[list[ChallengeItem], int]:
        """反证官挑毛病（上限 8 条）：专攻已过规则核验但因果推断有漏洞或分歧大的主张。"""
        verified_ids = {r.claim_id for r in verification_results if r.status == VerificationStatus.VERIFIED}
        challenges: list[ChallengeItem] = []

        ann_or_channel_ids = [
            c.evidence_id
            for c in evidence_store.values()
            if ("打款" in c.excerpt or "打款" in c.title or "渠道" in c.excerpt)
        ]

        for c in claims:
            if c.claim_id not in verified_ids:
                continue

            # 1. 经典案例：合同负债下降 42% 推出需求走弱 → 高严重度因果谬误质询
            if c.claim_id == "EXP-FUN-CL-CAUSAL":
                challenges.append(
                    ChallengeItem(
                        challenge_id="CHG-01",
                        target_claim_id=c.claim_id,
                        target_expert=ExpertRole.FUNDAMENTAL,
                        severity=ChallengeSeverity.HIGH,
                        critique=(
                            "合同负债同比下降 42% 可能是公司主动调整经销商打款政策和直营配额节奏所致，"
                            "在排除打款政策变化前，不能直接推断终端需求明显走弱。"
                        ),
                        counter_evidence_ids=ann_or_channel_ids[:2],
                        is_causal_fallacy=True,
                    )
                )

            # 2. 估值观点分歧质询（用于产生保留意见，或在可控开关下模拟未决高危争议触发 interrupt）
            elif c.claim_id == "EXP-VAL-2":
                sev = ChallengeSeverity.HIGH if self.inject_unresolved_high_dispute else ChallengeSeverity.MEDIUM
                challenges.append(
                    ChallengeItem(
                        challenge_id="CHG-02",
                        target_claim_id=c.claim_id,
                        target_expert=ExpertRole.VALUATION,
                        severity=sev,
                        critique=(
                            "多方看重历史低分位估值与高现金流，空方担忧增速换挡与渠道库存去化，"
                            "双方在永续增速假设上存在根本分歧，不能单一视角定论。"
                        ),
                        counter_evidence_ids=[
                            eid for eid, card in evidence_store.items()
                            if "风险" in card.title or "库存" in card.excerpt
                        ][:1],
                        is_causal_fallacy=False,
                    )
                )

            # 3. 行业/渠道多空分歧质询
            elif c.claim_id == "EXP-IND-1":
                challenges.append(
                    ChallengeItem(
                        challenge_id="CHG-03",
                        target_claim_id=c.claim_id,
                        target_expert=ExpertRole.INDUSTRY,
                        severity=ChallengeSeverity.MEDIUM,
                        critique="渠道调研样本覆盖有限，需区分头部经销商短期排期调整与全渠道真实终端动销。",
                        counter_evidence_ids=[],
                        is_causal_fallacy=False,
                    )
                )

        return challenges[:8], 1900

    def respond_to_challenges(
        self,
        *,
        challenges: list[ChallengeItem],
        claims: list[Claim],
        evidence_store: dict[str, EvidenceCard],
    ) -> tuple[list[ChallengeResponse], int]:
        """专家针对质询回应 1 轮。"""
        responses: list[ChallengeResponse] = []
        for chg in challenges:
            if chg.is_causal_fallacy:
                # 面对合同负债打款政策反证，基本面专家承认原因果推断过于武断（CONCEDE）
                responses.append(
                    ChallengeResponse(
                        challenge_id=chg.challenge_id,
                        expert_role=chg.target_expert,
                        action=ChallengeResponseAction.CONCEDE,
                        response_text="接受质询：仅凭合同负债下降 42% 无法排除打款政策与渠道结构调整，撤回“直接说明需求走弱”的因果结论。",
                        supporting_evidence_ids=chg.counter_evidence_ids,
                    )
                )
            elif chg.severity == ChallengeSeverity.HIGH and self.inject_unresolved_high_dispute:
                # 高严重度分歧下专家坚持己见、双方无妥协 → 形成未决高危争议，提交前暂停等人复核
                responses.append(
                    ChallengeResponse(
                        challenge_id=chg.challenge_id,
                        expert_role=chg.target_expert,
                        action=ChallengeResponseAction.DEFEND,
                        response_text="坚持估值已充分计入增速放缓预期，但无法在现有财报内证伪空方关于明年渠道出清的担忧。",
                        supporting_evidence_ids=[],
                    )
                )
            else:
                responses.append(
                    ChallengeResponse(
                        challenge_id=chg.challenge_id,
                        expert_role=chg.target_expert,
                        action=ChallengeResponseAction.NARROW,
                        response_text="同意将分歧与前提条件显式保留在结论附注中，不强行统一为单边结论。",
                        supporting_evidence_ids=[],
                    )
                )
        return responses, 900 * max(1, len(challenges))
