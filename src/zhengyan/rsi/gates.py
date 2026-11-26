"""RSI 六道独立代码门控与治理红线保护（对应文档第 20、21、25、27 题）。

六道门控防住评测中最常见的六种“自欺欺人”：
1. 协议一致（protocol_consistency） -> 防换尺子：评测集/评分器/审稿提示词哈希必须与基线完全一致
2. 防泄漏（anti_leakage）            -> 防考题混进练习题：OOT 时间外测试集的 (symbol, period) 不能出现在坏案例里
3. 防过拟合（anti_overfitting）      -> 防背答案：
   - S1 经验卡禁具体股票名/代码、禁具体数字、禁买卖结论；出现行业专用词须在 applicable_scope 写明对应行业
   - S2 路由参数改动不得超过本轮衰减步长上限
   - S3 站点白名单每次只加 1 个、只加不删、禁黑名单和自媒体平台
4. 噪声门控（noise_gate）            -> 防把运气当进步：>=3 次重复，mean(cand) - mean(base) > max(spread_base, spread_cand)
5. 成本门控（cost_gate）             -> 防花钱买分：token 相对涨幅超免费额度时，质量相对提升必须 >= token 相对涨幅
6. 回归门控（regression_gate）       -> 防拆东墙补西墙：固定回放集和时间外测试集均不得退化超过波动范围
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from zhengyan.rsi.models import (
    CandidateProposal,
    EvalProtocolFingerprint,
    ExperienceCard,
    FORBIDDEN_RSI_TARGETS,
    GateCheckResult,
    RSIComponent,
    RoutingParamsAsset,
    SearchTemplateAsset,
    SixGatesReport,
    SliceEvalStats,
)

_STOCK_CODE_RE = re.compile(r"(?<!\d)(?:60\d{4}|00\d{4}|30\d{4}|688\d{3})(?!\d)")
# 经验卡正文禁止具体财报/涨跌数值（如 42%、893.89、15倍）
_CONCRETE_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*(?:%|亿元|万元|元|倍|个百分点)|[+-]\d+(?:\.\d+)?)")


class RSIRedlineViolationError(PermissionError):
    """试图让 RSI 修改治理红线（权限、隐私、核验规则、交易边界、模型权重等）时抛出。"""


def assert_not_redline_target(target_name: str) -> None:
    if target_name in FORBIDDEN_RSI_TARGETS:
        raise RSIRedlineViolationError(
            f"红线拦截：{target_name!r} 属于系统安全/治理红线，严禁进入 RSI 自进化闭环！"
        )


def max_step_for_round(round_index: int, *, initial_step: float = 0.30, decay: float = 0.70, floor: float = 0.05) -> float:
    """防学过头：步长上限随轮次指数衰减（前几轮步长略大，后续越来越收敛）。"""
    return round(max(floor, initial_step * (decay ** round_index)), 4)


@dataclass(frozen=True)
class IndustryTermsConfig:
    industries: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]  # industry -> (scope_aliases, terms)
    forbidden_company_names: tuple[str, ...]
    forbidden_advice_terms: tuple[str, ...]
    forbidden_self_media_domains: frozenset[str]

    @classmethod
    def from_yaml(cls, path: Path) -> IndustryTermsConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        inds: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        for ind_name, info in (data.get("industries") or {}).items():
            aliases = tuple(str(x) for x in (info.get("scope_aliases") or []))
            terms = tuple(str(x) for x in (info.get("terms") or []))
            inds[str(ind_name)] = (aliases, terms)
        return cls(
            industries=inds,
            forbidden_company_names=tuple(str(x) for x in (data.get("forbidden_company_names") or [])),
            forbidden_advice_terms=tuple(str(x) for x in (data.get("forbidden_investment_advice_terms") or [])),
            forbidden_self_media_domains=frozenset(
                str(x).lower() for x in (data.get("forbidden_self_media_domains") or [])
            ),
        )


class RSISixGates:
    """纯代码六道门控（三权分立：优化器无法篡改或绕过）。"""

    def __init__(
        self,
        *,
        industry_config: IndustryTermsConfig,
        min_repeats: int = 3,
        free_token_overhead: float = 0.05,
    ) -> None:
        self.industry_config = industry_config
        self.min_repeats = min_repeats
        self.free_token_overhead = free_token_overhead

    def evaluate(
        self,
        *,
        proposal: CandidateProposal,
        baseline_protocol: EvalProtocolFingerprint,
        candidate_protocol: EvalProtocolFingerprint,
        oot_pairs: set[tuple[str, str]],
        baseline_slices: dict[str, SliceEvalStats],
        candidate_slices: dict[str, SliceEvalStats],
    ) -> SixGatesReport:
        checks = [
            self._gate1_protocol_consistency(baseline_protocol, candidate_protocol),
            self._gate2_anti_leakage(proposal, oot_pairs),
            self._gate3_anti_overfitting(proposal),
            self._gate4_noise_gate(baseline_slices["new_failures"], candidate_slices["new_failures"]),
            self._gate5_cost_gate(baseline_slices["new_failures"], candidate_slices["new_failures"]),
            self._gate6_regression_gate(baseline_slices, candidate_slices),
        ]
        failed = [c.gate_name for c in checks if not c.passed]
        return SixGatesReport(
            proposal_id=proposal.proposal_id,
            component=proposal.component,
            passed=(len(failed) == 0),
            checks=checks,
            failed_gates=failed,
        )

    def _gate1_protocol_consistency(
        self,
        base: EvalProtocolFingerprint,
        cand: EvalProtocolFingerprint,
    ) -> GateCheckResult:
        if base == cand:
            return GateCheckResult(gate_name="protocol_consistency", passed=True, detail="评测集/评分器/审稿提示词哈希完全一致")
        return GateCheckResult(
            gate_name="protocol_consistency",
            passed=False,
            detail=f"评测协议不一致：基线={base.model_dump()} vs 候选={cand.model_dump()}",
        )

    def _gate2_anti_leakage(
        self,
        proposal: CandidateProposal,
        oot_pairs: set[tuple[str, str]],
    ) -> GateCheckResult:
        leaked: list[str] = []
        for bc in proposal.source_bad_cases:
            if bc.symbol and bc.period and (bc.symbol, bc.period) in oot_pairs:
                leaked.append(f"{bc.symbol}/{bc.period}(来自 {bc.case_id})")
        if leaked:
            return GateCheckResult(
                gate_name="anti_leakage",
                passed=False,
                detail=f"检测到时间外测试集泄漏：坏案例包含 OOT 标的与报告期 {', '.join(leaked)}",
            )
        return GateCheckResult(gate_name="anti_leakage", passed=True, detail="坏案例与时间外测试集（股票+报告期）零重叠")

    def _gate3_anti_overfitting(self, proposal: CandidateProposal) -> GateCheckResult:
        if proposal.component == RSIComponent.S1_EXPERIENCE_CARDS:
            return self._check_s1_overfitting(proposal)
        if proposal.component == RSIComponent.S2_ROUTING_PARAMS:
            return self._check_s2_overfitting(proposal)
        if proposal.component == RSIComponent.S3_SEARCH_TEMPLATES:
            return self._check_s3_overfitting(proposal)
        return GateCheckResult(gate_name="anti_overfitting", passed=False, detail="未知组件类型")

    def _check_s1_overfitting(self, proposal: CandidateProposal) -> GateCheckResult:
        raw_cards = proposal.candidate_payload.get("cards") or []
        cards = [ExperienceCard.model_validate(c) for c in raw_cards]
        violations: list[str] = []

        for card in cards:
            body = f"{card.applicable_scope} {card.lesson} {card.must_check} {card.invalidation_condition}"
            # 1. 禁止具体股票代码与公司名
            if _STOCK_CODE_RE.search(body):
                violations.append(f"卡片 {card.card_id} 包含具体股票代码")
            for cname in self.industry_config.forbidden_company_names:
                if cname in body:
                    violations.append(f"卡片 {card.card_id} 包含具体公司名“{cname}”（疑似背答案）")
            # 2. 禁止具体财报/涨跌数字
            num_match = _CONCRETE_NUMBER_RE.search(body)
            if num_match:
                violations.append(f"卡片 {card.card_id} 包含具体数值“{num_match.group(0)}”（疑似背答案）")
            # 3. 禁止买卖推荐结论
            for adv in self.industry_config.forbidden_advice_terms:
                if adv in body:
                    violations.append(f"卡片 {card.card_id} 包含违规投资建议词“{adv}”")
            # 4. 行业专用词须匹配适用范围（config/industry_terms.yaml）
            for ind_name, (aliases, terms) in self.industry_config.industries.items():
                hit_terms = [t for t in terms if t in body]
                if hit_terms and not any(a in card.applicable_scope for a in aliases):
                    violations.append(
                        f"卡片 {card.card_id} 使用了行业专用词 {hit_terms}，但适用范围“{card.applicable_scope}”未标明行业（{ind_name}）"
                    )

        if violations:
            return GateCheckResult(gate_name="anti_overfitting", passed=False, detail="；".join(violations))
        return GateCheckResult(gate_name="anti_overfitting", passed=True, detail=f"S1 经验卡（{len(cards)} 张）全部通过泛化与行业范围校验")

    def _check_s2_overfitting(self, proposal: CandidateProposal) -> GateCheckResult:
        base = RoutingParamsAsset.model_validate(proposal.baseline_payload)
        cand = RoutingParamsAsset.model_validate(proposal.candidate_payload)
        limit = max_step_for_round(proposal.round_index)
        temp_delta = abs(cand.temperature - base.temperature)
        thresh_delta = abs(cand.abstain_threshold - base.abstain_threshold)
        if temp_delta > limit + 1e-9 or thresh_delta > limit + 1e-9:
            return GateCheckResult(
                gate_name="anti_overfitting",
                passed=False,
                detail=f"S2 参数变动超本轮步长上限 {limit}（ΔT={temp_delta:.3f}, Δthresh={thresh_delta:.3f}）",
            )
        return GateCheckResult(
            gate_name="anti_overfitting",
            passed=True,
            detail=f"S2 参数变动在步长上限 {limit} 内（ΔT={temp_delta:.3f}, Δthresh={thresh_delta:.3f}）",
        )

    def _check_s3_overfitting(self, proposal: CandidateProposal) -> GateCheckResult:
        base = SearchTemplateAsset.model_validate(proposal.baseline_payload)
        cand = SearchTemplateAsset.model_validate(proposal.candidate_payload)

        base_set = {d.lower() for d in base.whitelist_domains}
        cand_set = {d.lower() for d in cand.whitelist_domains}
        removed = base_set - cand_set
        added = cand_set - base_set

        if removed:
            return GateCheckResult(
                gate_name="anti_overfitting",
                passed=False,
                detail=f"S3 站点名单只加不删，禁止删除已有白名单站点：{sorted(removed)}",
            )
        if len(added) > 1:
            return GateCheckResult(
                gate_name="anti_overfitting",
                passed=False,
                detail=f"S3 高风险限制：单次只允许新增最多 1 个站点，本次试图新增 {len(added)} 个：{sorted(added)}",
            )
        for domain in added:
            if any(
                domain == bad or domain.endswith(f".{bad}")
                for bad in self.industry_config.forbidden_self_media_domains
            ):
                return GateCheckResult(
                    gate_name="anti_overfitting",
                    passed=False,
                    detail=f"S3 严禁将黑名单或自媒体平台加入白名单：{domain}",
                )

        step_limit = max_step_for_round(proposal.round_index, initial_step=0.20)
        if abs(cand.relevance_threshold - base.relevance_threshold) > step_limit + 1e-9:
            return GateCheckResult(
                gate_name="anti_overfitting",
                passed=False,
                detail=f"S3 相关度阈值改动超步长上限 {step_limit}",
            )
        return GateCheckResult(
            gate_name="anti_overfitting",
            passed=True,
            detail=f"S3 站点名单与阈值改动合法（新增站点={sorted(added)}）",
        )

    def _gate4_noise_gate(self, base: SliceEvalStats, cand: SliceEvalStats) -> GateCheckResult:
        if len(base.run_scores) < self.min_repeats or len(cand.run_scores) < self.min_repeats:
            return GateCheckResult(
                gate_name="noise_gate",
                passed=False,
                detail=f"重复次数不足 {self.min_repeats} 次（基线={len(base.run_scores)}, 候选={len(cand.run_scores)}）",
            )
        gain = cand.mean_score - base.mean_score
        max_spread = max(base.spread, cand.spread)
        if gain > max_spread + 1e-9:
            return GateCheckResult(
                gate_name="noise_gate",
                passed=True,
                detail=f"净提升 {gain:+.4f} 严格超过最大波动范围 {max_spread:.4f}",
            )
        return GateCheckResult(
            gate_name="noise_gate",
            passed=False,
            detail=f"噪声门控拦截：净提升 {gain:+.4f} 未超过两边波动范围较大值 {max_spread:.4f}（可能为随机波动）",
        )

    def _gate5_cost_gate(self, base: SliceEvalStats, cand: SliceEvalStats) -> GateCheckResult:
        if base.mean_tokens <= 0:
            return GateCheckResult(gate_name="cost_gate", passed=True, detail="离线确定性回放无额外 Token 消耗")
        tok_rel = (cand.mean_tokens - base.mean_tokens) / base.mean_tokens
        if tok_rel <= self.free_token_overhead:
            return GateCheckResult(
                gate_name="cost_gate",
                passed=True,
                detail=f"Token 相对涨幅 {tok_rel*100:+.1f}% 在免费额度 {self.free_token_overhead*100:.0f}% 内",
            )
        q_rel = (cand.mean_score - base.mean_score) / max(base.mean_score, 1e-6)
        if q_rel >= tok_rel:
            return GateCheckResult(
                gate_name="cost_gate",
                passed=True,
                detail=f"质量相对提升 {q_rel*100:+.1f}% >= Token 相对涨幅 {tok_rel*100:+.1f}%",
            )
        return GateCheckResult(
            gate_name="cost_gate",
            passed=False,
            detail=f"成本门控拦截：Token 涨幅 {tok_rel*100:+.1f}% 超过质量相对提升 {q_rel*100:+.1f}%",
        )

    def _gate6_regression_gate(
        self,
        base_slices: dict[str, SliceEvalStats],
        cand_slices: dict[str, SliceEvalStats],
    ) -> GateCheckResult:
        regressions: list[str] = []
        for sname in ("frozen_replay", "out_of_time"):
            b = base_slices[sname]
            c = cand_slices[sname]
            delta = c.mean_score - b.mean_score
            allowed_noise = max(b.spread, c.spread)
            if delta < -allowed_noise - 1e-9:
                regressions.append(
                    f"切片 {sname} 退化 {delta:+.4f}，超过允许波动 -{allowed_noise:.4f}"
                )
        if regressions:
            return GateCheckResult(gate_name="regression_gate", passed=False, detail="；".join(regressions))
        return GateCheckResult(gate_name="regression_gate", passed=True, detail="固定回放集与时间外测试集均无退化")
