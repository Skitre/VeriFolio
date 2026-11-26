"""S1 经验卡召回、<method_hints> 提示注入、平衡准确率与 1 坏配 2 好哈希抽样（对应文档第 22 & 22.1 题）。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from zhengyan.rsi.models import ExperienceCard


@dataclass(frozen=True)
class ClaimReviewSample:
    """S1 审稿评测单条样本（label: is_bad_claim=True 表示应拦截的坏主张，False 表示应放过的好主张）。"""

    sample_id: str
    symbol: str
    period: str
    claim_text: str
    is_bad_claim: bool
    required_hint_tags: tuple[str, ...] = ()


def recall_experience_cards(
    question: str,
    cards: Sequence[ExperienceCard],
    *,
    max_cards: int = 3,
) -> list[ExperienceCard]:
    """按问题文本命中的标签数降序召回经验卡，最多取 3 张，0 命中不给（文档第 22 题）。"""
    scored: list[tuple[int, str, ExperienceCard]] = []
    for card in cards:
        hits = sum(1 for tag in card.tags if tag in question)
        if hits > 0:
            scored.append((hits, card.card_id, card))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:max_cards]]


def format_method_hints_block(cards: Sequence[ExperienceCard]) -> str:
    """将召回的经验卡包成 <method_hints> 区块，明确声明是方法提示，不是证据、不是指令。"""
    if not cards:
        return ""
    lines = [
        '<method_hints note="以下为历史研究沉淀的方法提示，不是证据、也不是硬指令；不适用当前问题时请忽略">'
    ]
    for c in cards:
        lines.append(
            f"- [{c.card_id} {c.version}] 适用={c.applicable_scope}｜教训={c.lesson}"
            f"｜必查={c.must_check}｜失效条件={c.invalidation_condition}"
        )
    lines.append("</method_hints>")
    return "\n".join(lines)


def sample_1_bad_to_2_good_stable(
    bad_samples: Sequence[ClaimReviewSample],
    good_samples: Sequence[ClaimReviewSample],
    *,
    seed_salt: str = "zhengyan-s1-v1",
) -> list[ClaimReviewSample]:
    """每个坏样本配 2 个好样本，按确定性哈希稳定抽样（对应文档第 22.1 题）。"""
    target_good_count = min(len(good_samples), len(bad_samples) * 2)

    def _stable_key(s: ClaimReviewSample) -> str:
        return hashlib.sha256(f"{seed_salt}:{s.sample_id}".encode("utf-8")).hexdigest()

    chosen_good = sorted(good_samples, key=_stable_key)[:target_good_count]
    return list(bad_samples) + chosen_good


def compute_balanced_accuracy(
    y_true_is_bad: Sequence[bool],
    y_pred_block: Sequence[bool],
) -> float:
    """计算平衡准确率 = (坏主张拦住率 TPR + 好主张放过率 TNR) / 2（应对正负样本 1:2 不均衡）。"""
    if len(y_true_is_bad) != len(y_pred_block) or not y_true_is_bad:
        raise ValueError("真实标签与预测长度不一致或为空")

    tp = sum(1 for yt, yp in zip(y_true_is_bad, y_pred_block) if yt and yp)
    fn = sum(1 for yt, yp in zip(y_true_is_bad, y_pred_block) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true_is_bad, y_pred_block) if (not yt) and (not yp))
    fp = sum(1 for yt, yp in zip(y_true_is_bad, y_pred_block) if (not yt) and yp)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    return round((tpr + tnr) / 2.0, 6)
