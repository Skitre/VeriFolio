"""双轴路由器 7 步优先级链与模型轴解耦测试（对应文档第 4、5、5.2 题）。"""

from __future__ import annotations

from pathlib import Path

from zhengyan.core.enums import ResearchTier, UserIntent
from zhengyan.routing import (
    DualAxisRouter,
    ModelAxisConfig,
    RoutingClassifierOutput,
)

MODELS_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class _StubClassifier:
    def __init__(self, out: RoutingClassifierOutput | None = None, fail: bool = False) -> None:
        self._out = out
        self._fail = fail

    def classify_route(self, question: str, *, intent: UserIntent) -> RoutingClassifierOutput:
        if self._fail:
            raise RuntimeError("判别器服务超时")
        assert self._out is not None
        return self._out


def test_step1_user_override_never_escalates() -> None:
    model_axis = ModelAxisConfig.from_yaml(MODELS_PATH)
    router = DualAxisRouter(
        classifier=_StubClassifier(
            RoutingClassifierOutput(
                raw_probs={"quick": 0.05, "standard": 0.15, "deep": 0.80},
                consequence_level="high",
                disagreement_level="high",
                source="stub",
            )
        ),
        model_axis=model_axis,
    )
    decision = router.route(
        "贵州茅台现在估值合理吗",
        intent=UserIntent.RESEARCH,
        user_selected_tier=ResearchTier.QUICK,
    )
    assert decision.tier == ResearchTier.QUICK
    assert decision.step_triggered == "step1_user_override"
    assert decision.model_bindings["draft"].model == "deepseek-chat"


def test_step2_concept_hard_rule_bypasses_classifier() -> None:
    # 即使判别器挂了，概念解释类依然由第 2 步硬规则直接判为 quick
    router = DualAxisRouter(classifier=_StubClassifier(fail=True))
    decision = router.route("什么是市盈率", intent=UserIntent.CONCEPT)
    assert decision.tier == ResearchTier.QUICK
    assert decision.step_triggered == "step2_concept_hard_rule"


def test_step3_classifier_failure_falls_back_to_standard() -> None:
    router = DualAxisRouter(classifier=_StubClassifier(fail=True))
    decision = router.route("复核贵州茅台半年报", intent=UserIntent.RESEARCH)
    assert decision.tier == ResearchTier.STANDARD
    assert decision.classifier_failed_fallback is True
    assert decision.step_triggered == "step3_classifier_fallback"


def test_step5_abstain_picks_heavier_tier_from_top2() -> None:
    # 原始概率接近均匀：校准(T=1.0)后最高是 quick=0.45 < 0.55，第二是 standard=0.38 -> 拒判并升级为更重的 standard
    router = DualAxisRouter(
        classifier=_StubClassifier(
            RoutingClassifierOutput(
                raw_probs={"quick": 0.45, "standard": 0.38, "deep": 0.17},
                consequence_level="low",
                disagreement_level="low",
                source="stub",
            )
        ),
        temperature=1.0,
        abstain_threshold=0.55,
    )
    decision = router.route("查一下茅台最近怎么样", intent=UserIntent.RESEARCH)
    assert decision.abstained is True
    assert decision.tier == ResearchTier.STANDARD
    assert decision.step_triggered == "step5_abstain_upgrade"

    # 若前两名是 standard(0.46) 和 deep(0.40) -> 拒判并升到 deep
    router_sd = DualAxisRouter(
        classifier=_StubClassifier(
            RoutingClassifierOutput(
                raw_probs={"quick": 0.14, "standard": 0.46, "deep": 0.40},
                consequence_level="medium",
                disagreement_level="medium",
                source="stub",
            )
        ),
        temperature=1.0,
        abstain_threshold=0.55,
    )
    dec_sd = router_sd.route("拆解茅台渠道风险", intent=UserIntent.RESEARCH)
    assert dec_sd.abstained is True
    assert dec_sd.tier == ResearchTier.DEEP


def test_step6_high_consequence_and_high_disagreement_forces_deep() -> None:
    model_axis = ModelAxisConfig.from_yaml(MODELS_PATH)
    router = DualAxisRouter(
        classifier=_StubClassifier(
            RoutingClassifierOutput(
                raw_probs={"quick": 0.05, "standard": 0.85, "deep": 0.10},
                consequence_level="high",
                disagreement_level="high",
                source="stub",
            )
        ),
        model_axis=model_axis,
    )
    decision = router.route("重仓持有茅台，业绩分歧很大", intent=UserIntent.RESEARCH)
    assert decision.tier == ResearchTier.DEEP
    assert decision.escalated_by_consequence is True
    assert decision.step_triggered == "step6_high_consequence_escalation"
    # 验证模型轴：深研中估值研究员、反证官、主席绑定 reason 角色（deepseek-reasoner），基本面专家绑定 draft
    assert decision.model_bindings["valuation_expert"].role == "reason"
    assert decision.model_bindings["adversary_challenger"].role == "reason"
    assert decision.model_bindings["chair_adjudicator"].role == "reason"
    assert decision.model_bindings["fundamental_expert"].role == "draft"


def test_step7_trade_order_intent_floors_at_standard() -> None:
    router = DualAxisRouter(
        classifier=_StubClassifier(
            RoutingClassifierOutput(
                raw_probs={"quick": 0.90, "standard": 0.07, "deep": 0.03},
                consequence_level="low",
                disagreement_level="low",
                source="stub",
            )
        ),
    )
    decision = router.route("帮我买入 100 股贵州茅台", intent=UserIntent.TRADE_ORDER)
    assert decision.tier == ResearchTier.STANDARD
    assert decision.floored_by_trade_intent is True
    assert decision.step_triggered == "step7_trade_intent_floor"
