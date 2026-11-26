"""双轴路由器（对应文档第 4、5、5.1、5.2 题）。

严格按文档 7 步优先级执行：
1. 用户手动指定档位 → 直接使用，且不升级
2. 硬规则：概念解释类（CONCEPT）→ 直接走 quick
3. 调用判别器输出三类原始概率 + consequence_level + disagreement_level（挂掉则默认 standard）
4. 温度缩放校准（默认 T=0.75）
5. 拒判升级：校准后最高概率 < 0.55 → 在前两名（top-2）中选更重的流程（宁可多花 token 也不降档漏判）
6. 高后果升级：consequence=high 且 disagreement=high 但非 deep → 强制升到 deep
7. 交易意图兜底：intent == TRADE_ORDER 时至少走 standard
最后根据最终任务档位从 ModelAxisConfig 解析模型轴角色映射，实现双轴彻底解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zhengyan.core.enums import ResearchTier, UserIntent
from zhengyan.routing.calibration import argmax_label, temperature_scale, top2_labels
from zhengyan.routing.classifiers import RoutingClassifier, SeverityLevel
from zhengyan.routing.models_config import ModelAxisConfig, ModelRoleSpec

_TIER_WEIGHT: dict[ResearchTier, int] = {
    ResearchTier.QUICK: 0,
    ResearchTier.STANDARD: 1,
    ResearchTier.DEEP: 2,
}


@dataclass(frozen=True)
class RoutingDecision:
    tier: ResearchTier
    reason: str
    step_triggered: str
    raw_probs: dict[str, float] | None = None
    calibrated_probs: dict[str, float] | None = None
    max_calibrated_prob: float | None = None
    temperature: float = 0.75
    abstained: bool = False
    escalated_by_consequence: bool = False
    floored_by_trade_intent: bool = False
    classifier_failed_fallback: bool = False
    consequence_level: SeverityLevel | None = None
    disagreement_level: SeverityLevel | None = None
    classifier_source: str = "rule"
    latency_ms: float = 0.0
    model_bindings: dict[str, ModelRoleSpec] = field(default_factory=dict)


class DualAxisRouter:
    """双轴路由器：任务轴（7 步治理路由）+ 模型轴（角色解耦解析）。"""

    def __init__(
        self,
        *,
        classifier: RoutingClassifier,
        model_axis: ModelAxisConfig | None = None,
        temperature: float = 0.75,
        abstain_threshold: float = 0.55,
    ) -> None:
        self.classifier = classifier
        self.model_axis = model_axis
        self.temperature = temperature
        self.abstain_threshold = abstain_threshold

    def _bindings_for(self, tier: ResearchTier) -> dict[str, ModelRoleSpec]:
        if self.model_axis is None:
            return {}
        return self.model_axis.resolve_bindings(tier)

    def route(
        self,
        question: str,
        *,
        intent: UserIntent,
        user_selected_tier: ResearchTier | None = None,
        temperature_override: float | None = None,
        abstain_threshold_override: float | None = None,
    ) -> RoutingDecision:
        temp = temperature_override if temperature_override is not None else self.temperature
        threshold = abstain_threshold_override if abstain_threshold_override is not None else self.abstain_threshold

        # 第 1 步：用户手动指定，直接采用且不升级
        if user_selected_tier is not None:
            return RoutingDecision(
                tier=user_selected_tier,
                reason="用户手动指定研究档位，不触发自动升级",
                step_triggered="step1_user_override",
                temperature=temp,
                model_bindings=self._bindings_for(user_selected_tier),
            )

        # 第 2 步：硬规则 — 概念解释类直接走快查
        if intent == UserIntent.CONCEPT:
            return RoutingDecision(
                tier=ResearchTier.QUICK,
                reason="硬规则：概念解释类直接走快查",
                step_triggered="step2_concept_hard_rule",
                temperature=temp,
                model_bindings=self._bindings_for(ResearchTier.QUICK),
            )

        # 第 3 步：调用判别器（故障时默认走 standard，不硬判崩溃）
        try:
            clf_out = self.classifier.classify_route(question, intent=intent)
        except Exception as exc:
            return RoutingDecision(
                tier=ResearchTier.STANDARD,
                reason=f"判别器异常（{exc}），安全降级为标准研究",
                step_triggered="step3_classifier_fallback",
                temperature=temp,
                classifier_failed_fallback=True,
                classifier_source="fallback",
                model_bindings=self._bindings_for(ResearchTier.STANDARD),
            )

        # 第 4 步：温度缩放校准
        calibrated = temperature_scale(clf_out.raw_probs, temp)
        top_label, top_prob = argmax_label(calibrated)
        chosen = ResearchTier(top_label)
        step = "step4_calibrated_argmax"
        reason = f"温度校准(T={temp})后最高概率 {top_prob:.3f} >= {threshold}，路由至 {chosen.value}"
        abstained = False

        # 第 5 步：拒判升级 — 校准后最高概率 < 0.55，在前两名中选更重流程
        if top_prob < threshold:
            abstained = True
            pair = top2_labels(calibrated)
            heavier = max((ResearchTier(lbl) for lbl, _ in pair), key=lambda t: _TIER_WEIGHT[t])
            chosen = heavier
            step = "step5_abstain_upgrade"
            pair_desc = ", ".join(f"{k}:{v:.2f}" for k, v in pair)
            reason = f"最高概率 {top_prob:.3f} < {threshold} 触发拒判，在前两名({pair_desc})中选择更重流程 {chosen.value}"

        # 第 6 步：高后果 + 高分歧强制升级深研
        escalated = False
        if (
            clf_out.consequence_level == "high"
            and clf_out.disagreement_level == "high"
            and chosen != ResearchTier.DEEP
        ):
            chosen = ResearchTier.DEEP
            escalated = True
            step = "step6_high_consequence_escalation"
            reason = "后果高且分歧高，强制升级至争议深研"

        # 第 7 步：交易意图兜底 — 含下单意图至少走标准研究
        floored = False
        if intent == UserIntent.TRADE_ORDER and _TIER_WEIGHT[chosen] < _TIER_WEIGHT[ResearchTier.STANDARD]:
            chosen = ResearchTier.STANDARD
            floored = True
            step = "step7_trade_intent_floor"
            reason = "交易意图兜底：至少走标准研究讲清事实、情景与边界"

        return RoutingDecision(
            tier=chosen,
            reason=reason,
            step_triggered=step,
            raw_probs=clf_out.raw_probs,
            calibrated_probs=calibrated,
            max_calibrated_prob=round(top_prob, 6),
            temperature=temp,
            abstained=abstained,
            escalated_by_consequence=escalated,
            floored_by_trade_intent=floored,
            consequence_level=clf_out.consequence_level,
            disagreement_level=clf_out.disagreement_level,
            classifier_source=clf_out.source,
            latency_ms=clf_out.latency_ms,
            model_bindings=self._bindings_for(chosen),
        )
