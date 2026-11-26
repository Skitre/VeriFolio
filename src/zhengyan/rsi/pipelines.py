"""S1 经验卡 / S2 路由自校准 / S3 检索策略进化完整三通道评测器与受控闭环控制器。

对应文档第 20–24 题：
- S1（经验卡）：三组对照（基线 / 候选 / 关掉经验库），每组重复 3 次，平衡准确率检验；
- S2（路由自校准）：在历史上多次独立运行保存的原始概率快照上回放不同参数，天然暴露随机波动；
- S3（检索策略进化）：在缓存的原始搜索结果上确定性回放，指标 = Precision(保留中可用B级占比) × Recall(权威结果中被保留且可抓占比)。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal, Sequence

from zhengyan.core.enums import ResearchTier, UserIntent
from zhengyan.routing.calibration import argmax_label, temperature_scale, top2_labels
from zhengyan.rsi.experience import (
    ClaimReviewSample,
    compute_balanced_accuracy,
    recall_experience_cards,
)
from zhengyan.rsi.gates import RSISixGates
from zhengyan.rsi.ledger import AppendOnlyRSILedger
from zhengyan.rsi.models import (
    CandidateProposal,
    EvalProtocolFingerprint,
    ExperienceCard,
    RSIComponent,
    RoutingParamsAsset,
    SearchTemplateAsset,
    SixGatesReport,
    SliceEvalStats,
)

_TIER_WEIGHT = {ResearchTier.QUICK: 0, ResearchTier.STANDARD: 1, ResearchTier.DEEP: 2}


def build_protocol_fingerprint(
    *,
    eval_set_id: str,
    scorer_version: str,
    reviewer_prompt: str,
) -> EvalProtocolFingerprint:
    return EvalProtocolFingerprint(
        eval_set_hash=hashlib.sha256(eval_set_id.encode("utf-8")).hexdigest()[:16],
        scorer_version_hash=hashlib.sha256(scorer_version.encode("utf-8")).hexdigest()[:16],
        reviewer_prompt_hash=hashlib.sha256(reviewer_prompt.encode("utf-8")).hexdigest()[:16],
    )


# ==================== S1：经验卡审稿评测器（固定提示词 + 3 次重复 + 开/关对照） ====================


def evaluate_s1_slice(
    *,
    slice_name: Literal["new_failures", "frozen_replay", "out_of_time"],
    samples: Sequence[ClaimReviewSample],
    cards: Sequence[ExperienceCard],
    runs: int = 3,
    reviewer_noise_prob: float = 0.0,
    seed: int = 42,
    include_off_ablation: bool = True,
) -> SliceEvalStats:
    """模拟固定提示词审稿模型对 S1 样本集打分，支持开/关对照（cards=[]）。"""
    rng = random.Random(seed)

    def _run_once(active_cards: Sequence[ExperienceCard]) -> tuple[float, float]:
        y_true: list[bool] = []
        y_pred_block: list[bool] = []
        tokens = 0.0
        for s in samples:
            recalled = recall_experience_cards(s.claim_text, active_cards, max_cards=3)
            # 每召回 1 张卡约增加 ~25 token 方法提示（相对 ~2400 token 远在 5% 免费额度内）
            tokens += 2400.0 + len(recalled) * 25.0
            y_true.append(s.is_bad_claim)

            if s.is_bad_claim:
                # 坏主张：若召回的经验卡覆盖了对应方法论标签，审稿模型大概率成功拦截；
                # 否则只有基础直觉能力（容易放过表面数字无误的因果谬误！）
                recalled_tags = {t for c in recalled for t in c.tags}
                covered = bool(s.required_hint_tags) and any(t in recalled_tags for t in s.required_hint_tags)
                if covered:
                    pred_block = True
                elif not s.required_hint_tags:
                    # 常规明显坏主张（基础规则/常识即可拦住）
                    pred_block = True
                else:
                    pred_block = False
            else:
                # 好主张：默认放过
                pred_block = False

            if reviewer_noise_prob > 0 and rng.random() < reviewer_noise_prob:
                pred_block = not pred_block
            y_pred_block.append(pred_block)

        return compute_balanced_accuracy(y_true, y_pred_block), tokens / max(1, len(samples))

    scores: list[float] = []
    tok_list: list[float] = []
    for _ in range(runs):
        sc, tok = _run_once(cards)
        scores.append(sc)
        tok_list.append(tok)

    off_mean: float | None = None
    if include_off_ablation:
        off_scores = [_run_once([])[0] for _ in range(runs)]
        off_mean = round(sum(off_scores) / len(off_scores), 6)

    return SliceEvalStats(
        slice_name=slice_name,
        run_scores=scores,
        mean_score=round(sum(scores) / len(scores), 6),
        spread=round(max(scores) - min(scores), 6),
        mean_tokens=round(sum(tok_list) / len(tok_list), 2),
        off_mean_score=off_mean,
    )


# ==================== S2：路由历史原始概率多轮回放自校准 ====================


@dataclass(frozen=True)
class HistoricalRoutingSnapshot:
    """一次独立历史路由运行保存下来的原始概率快照（每次历史运行天然构成一次重复）。"""

    run_label: str
    raw_probs_list: list[dict[str, float]]
    gold_tiers: list[ResearchTier]


def evaluate_s2_replay(
    *,
    slice_name: Literal["new_failures", "frozen_replay", "out_of_time"],
    snapshots: Sequence[HistoricalRoutingSnapshot],
    params: RoutingParamsAsset,
) -> SliceEvalStats:
    """在历史上多次独立运行保存的原始概率上回放 (temperature, abstain_threshold)。"""
    scores: list[float] = []
    for snap in snapshots:
        correct = 0
        for raw_p, gold in zip(snap.raw_probs_list, snap.gold_tiers):
            cal = temperature_scale(raw_p, params.temperature)
            top_lbl, top_p = argmax_label(cal)
            if top_p < params.abstain_threshold:
                pair = top2_labels(cal)
                pred = max((ResearchTier(lbl) for lbl, _ in pair), key=lambda t: _TIER_WEIGHT[t])
            else:
                pred = ResearchTier(top_lbl)
            if pred == gold:
                correct += 1
        scores.append(round(correct / len(snap.gold_tiers), 6) if snap.gold_tiers else 0.0)

    return SliceEvalStats(
        slice_name=slice_name,
        run_scores=scores,
        mean_score=round(sum(scores) / len(scores), 6),
        spread=round(max(scores) - min(scores), 6),
        mean_tokens=0.0,  # 原始概率离线确定性回放无额外 LLM Token 成本
    )


# ==================== S3：检索策略确定性回放进化（Precision × Recall） ====================


@dataclass(frozen=True)
class CachedSearchDoc:
    """缓存的原始搜索条目，用于 S3 阈值与白名单的确定性回放评测。"""

    doc_id: str
    domain: str
    relevance_score: float
    is_authoritative: bool
    is_usable_b_grade_if_fetched: bool


def evaluate_s3_search_replay(
    *,
    slice_name: Literal["new_failures", "frozen_replay", "out_of_time"],
    cached_docs: Sequence[CachedSearchDoc],
    template: SearchTemplateAsset,
    runs: int = 3,
) -> SliceEvalStats:
    """在缓存的原始搜索结果上确定性回放（文档第 23 题：3 次结果完全一致）。

    指标 = (保留下来的结果里可用 B 级占比) × (权威结果里被保留且能抓取的占比)
    """
    wl = {d.lower() for d in template.whitelist_domains}

    def _score_once() -> float:
        kept = [d for d in cached_docs if d.relevance_score >= template.relevance_threshold]
        usable_kept = [
            d for d in kept
            if d.is_usable_b_grade_if_fetched and any(d.domain.lower() == w or d.domain.lower().endswith(f".{w}") for w in wl)
        ]
        precision = len(usable_kept) / len(kept) if kept else 0.0

        auth_docs = [d for d in cached_docs if d.is_authoritative]
        auth_kept_fetchable = [
            d for d in auth_docs
            if d.relevance_score >= template.relevance_threshold
            and any(d.domain.lower() == w or d.domain.lower().endswith(f".{w}") for w in wl)
        ]
        recall = len(auth_kept_fetchable) / len(auth_docs) if auth_docs else 1.0
        return round(precision * recall, 6)

    val = _score_once()
    scores = [val] * runs
    return SliceEvalStats(
        slice_name=slice_name,
        run_scores=scores,
        mean_score=val,
        spread=0.0,
        mean_tokens=0.0,
    )


# ==================== 三权分立受控自进化控制器 ====================


class RSIController:
    """串联“优化器提交 → 独立代码评测 → 六道代码门控 → 账本记录 → 人工批准发布/回滚”。"""

    def __init__(self, *, ledger: AppendOnlyRSILedger, gates: RSISixGates) -> None:
        self.ledger = ledger
        self.gates = gates

    def submit_and_evaluate(
        self,
        *,
        proposal: CandidateProposal,
        baseline_protocol: EvalProtocolFingerprint,
        candidate_protocol: EvalProtocolFingerprint,
        oot_pairs: set[tuple[str, str]],
        baseline_slices: dict[str, SliceEvalStats],
        candidate_slices: dict[str, SliceEvalStats],
    ) -> SixGatesReport:
        """执行失败方向去重检查 + 六道代码门控，并无条件记账（含失败记录）。"""
        # 防止重复试错已失败过的方向
        if proposal.direction_key in self.ledger.failed_directions(proposal.component):
            report = SixGatesReport(
                proposal_id=proposal.proposal_id,
                component=proposal.component,
                passed=False,
                checks=[],
                failed_gates=["duplicate_failed_direction"],
            )
            self.ledger.record_run(
                proposal_id=proposal.proposal_id,
                component=proposal.component,
                candidate_version=proposal.candidate_version,
                direction_key=proposal.direction_key,
                protocol=candidate_protocol,
                baseline_slices=baseline_slices,
                candidate_slices=candidate_slices,
                gate_report=report,
            )
            return report

        report = self.gates.evaluate(
            proposal=proposal,
            baseline_protocol=baseline_protocol,
            candidate_protocol=candidate_protocol,
            oot_pairs=oot_pairs,
            baseline_slices=baseline_slices,
            candidate_slices=candidate_slices,
        )
        self.ledger.record_run(
            proposal_id=proposal.proposal_id,
            component=proposal.component,
            candidate_version=proposal.candidate_version,
            direction_key=proposal.direction_key,
            protocol=candidate_protocol,
            baseline_slices=baseline_slices,
            candidate_slices=candidate_slices,
            gate_report=report,
        )
        return report

    def recommend_next_component(self, current: RSIComponent, *, stagnation_limit: int = 2) -> RSIComponent:
        """防学过头：若当前组件连续失败达到上限，建议切换到下一个可进化组件。"""
        if self.ledger.consecutive_failed_rounds(current) < stagnation_limit:
            return current
        order = [
            RSIComponent.S1_EXPERIENCE_CARDS,
            RSIComponent.S2_ROUTING_PARAMS,
            RSIComponent.S3_SEARCH_TEMPLATES,
        ]
        idx = order.index(current)
        return order[(idx + 1) % len(order)]
