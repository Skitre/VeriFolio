"""个人情境官（纯本地确定性规则，绝不调用大模型）与约束变更联动（对应文档第 12 & 18 题）。

设计要点：
1. 100% 在本机用纯规则评估用户确认约束（可承受回撤、资金期限、估值容忍度）与档案客观指标的匹配度；
   用户财务与风险偏好数据绝不离开本机发给云端模型！
2. 当用户修改自己的约束（如把可承受回撤从 20% 改成 10%），自动扫描所有标的最新档案，
   精准标出个人情境结论因此发生变化的档案（`detect_constraint_change_impacts`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from zhengyan.memory.store import (
    FiveLayerMemoryStore,
    UserConfirmedConstraints,
    VersionedDossier,
)


class SuitabilityLevel(StrEnum):
    ALIGNED = "aligned"        # 契合用户约束
    CAUTION = "caution"        # 存在需关注项
    MISALIGNED = "misaligned"  # 明显超出用户可承受风险/约束边界


@dataclass(frozen=True)
class PersonalContextAssessment:
    dossier_id: str
    symbol: str
    suitability: SuitabilityLevel
    triggers: list[str]


@dataclass(frozen=True)
class ConstraintChangeImpact:
    dossier_id: str
    symbol: str
    old_suitability: SuitabilityLevel
    new_suitability: SuitabilityLevel
    new_triggers: list[str]


class PersonalContextAdvisor:
    """纯本地规则个人情境官（零 LLM 调用）。"""

    def assess(
        self,
        dossier_node: VersionedDossier,
        constraints: UserConfirmedConstraints,
    ) -> PersonalContextAssessment:
        metrics = dossier_node.metrics_snapshot
        triggers: list[str] = []
        severe = False

        # 1. 最大回撤 / 年化波动率对照可承受回撤
        hist_dd = metrics.get("historical_drawdown_pct")
        if hist_dd is not None and hist_dd > constraints.max_drawdown_pct:
            severe = True
            triggers.append(
                f"历史最大回撤 {hist_dd}% 超过个人可承受回撤上限 {constraints.max_drawdown_pct}%"
            )

        # 2. 资金期限对照估值与高分歧周期属性
        if constraints.horizon_months < 12 and dossier_node.dossier.dissenting_opinions:
            triggers.append(
                f"资金期限仅 {constraints.horizon_months} 个月，但该标的存在多空分歧，短期波动风险较高"
            )

        # 3. 估值容忍度对照当前 PE(TTM)
        pe_ttm = metrics.get("pe_ttm")
        if (
            constraints.max_pe_tolerance is not None
            and pe_ttm is not None
            and pe_ttm > constraints.max_pe_tolerance
        ):
            severe = True
            triggers.append(
                f"当前 PE(TTM) {pe_ttm} 倍超过个人估值容忍上限 {constraints.max_pe_tolerance} 倍"
            )

        # 4. 存在未处理的待复核提醒
        pending = metrics.get("pending_alerts", Decimal("0"))
        if pending > 0:
            triggers.append(f"存在 {int(pending)} 条待复核变动提醒，建议更新研究后再评估")

        if severe:
            level = SuitabilityLevel.MISALIGNED
        elif triggers:
            level = SuitabilityLevel.CAUTION
        else:
            level = SuitabilityLevel.ALIGNED

        return PersonalContextAssessment(
            dossier_id=dossier_node.dossier.dossier_id,
            symbol=dossier_node.dossier.symbol or "UNKNOWN",
            suitability=level,
            triggers=triggers,
        )

    def detect_constraint_change_impacts(
        self,
        *,
        memory_store: FiveLayerMemoryStore,
        old_constraints: UserConfirmedConstraints,
        new_constraints: UserConfirmedConstraints,
    ) -> list[ConstraintChangeImpact]:
        """用户修改个人约束（如回撤 20% → 10%）后，精准找出个人情境结论因此改变的档案。"""
        impacts: list[ConstraintChangeImpact] = []
        for node in memory_store.list_latest_dossiers_all_symbols():
            before = self.assess(node, old_constraints)
            after = self.assess(node, new_constraints)
            if before.suitability != after.suitability:
                impacts.append(
                    ConstraintChangeImpact(
                        dossier_id=node.dossier.dossier_id,
                        symbol=node.dossier.symbol or "UNKNOWN",
                        old_suitability=before.suitability,
                        new_suitability=after.suitability,
                        new_triggers=after.triggers,
                    )
                )
        return impacts
