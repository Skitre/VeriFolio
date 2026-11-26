"""纯规则核验器（对应文档第 7 题：不调模型，100% 确定性）。

严格按文档 8 步执行：
1. 引用的证据 ID 不存在的，忽略这些引用；
2. 假设（hypothesis）和情景（scenario）类主张不算事实，直接放行，但标明“不是事实”；
3. 一条可用证据都没有 → 驳回（REJECTED）；
4. 引用的证据发布时间晚于研究时点 as_of → 驳回（REJECTED，防未来数据）；
5. 只有 C 级证据 → 驳回（REJECTED，保证 C 级线索进入结论 0 次）；
6. 主张里有数字的，逐个抽出并在引用的 A 级证据中匹配（计算卡所依据的财报递归计入）：
   - 金额/百分比/倍数在同类指标匹配，容差按书写精度半个单位；
   - “个百分点”拿两个百分比之差匹配；
7. 数字未在 A 级匹配到：
   - 仅在 B 级网页出现 → 存疑（DOUBTFUL，“须回到财报复核”）；
   - 哪里都找不到 → 存疑（DOUBTFUL，“无法由引用证据复算”）；
8. 多个网页来源属于同一事件簇（或同原始发布方）→ 标注“按单一来源计”。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from zhengyan.core.enums import ClaimType, EvidenceGrade, MetricUnit, VerificationStatus
from zhengyan.core.models import (
    Claim,
    EvidenceCard,
    MetricEntry,
    NumberCheckDetail,
    VerificationResult,
)
from zhengyan.verify.number_parser import (
    AMOUNT_UNITS,
    ExtractedNumber,
    extract_financial_numbers,
    to_base_cny,
)


class RuleVerifier:
    """确定性规则核验器。"""

    def verify_claim(
        self,
        claim: Claim,
        evidence_store: dict[str, EvidenceCard],
        *,
        as_of: date,
    ) -> VerificationResult:
        # 第 1 步：过滤不存在的证据 ID
        valid_cards: list[EvidenceCard] = []
        ignored_ids: list[str] = []
        seen: set[str] = set()
        for eid in claim.evidence_ids:
            if eid in seen:
                continue
            seen.add(eid)
            card = evidence_store.get(eid)
            if card is None:
                ignored_ids.append(eid)
            else:
                valid_cards.append(card)

        valid_ids = [c.evidence_id for c in valid_cards]

        # 第 2 步：假设和情景类主张不算事实，直接放行但标明非事实
        if claim.claim_type in (ClaimType.HYPOTHESIS, ClaimType.SCENARIO):
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.VERIFIED,
                is_fact=False,
                valid_evidence_ids=valid_ids,
                ignored_evidence_ids=ignored_ids,
                notes=["假设/情景类主张，不是事实陈述"],
            )

        # 第 3 步：一条可用证据都没有 → 驳回
        if not valid_cards:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.REJECTED,
                valid_evidence_ids=[],
                ignored_evidence_ids=ignored_ids,
                reasons=["无可用证据支撑"],
            )

        # 第 4 步：任意引用证据（含计算卡递归依赖）晚于研究时点 → 驳回
        expanded_a_cards = self._expand_a_grade_cards(valid_cards, evidence_store)
        all_checked_cards = self._dedup_cards(valid_cards + expanded_a_cards)
        future_cards = [
            c for c in all_checked_cards
            if c.published_at is not None and c.published_at > as_of
        ]
        if future_cards:
            bad = ", ".join(f"{c.evidence_id}({c.published_at})" for c in future_cards)
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.REJECTED,
                valid_evidence_ids=valid_ids,
                ignored_evidence_ids=ignored_ids,
                reasons=[f"引用证据发布时间晚于研究时点 {as_of}：{bad}"],
            )

        # 第 5 步：只有 C 级证据 → 驳回（铁律：C 级线索进入结论 0 次）
        if all(c.grade == EvidenceGrade.C for c in valid_cards):
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.REJECTED,
                valid_evidence_ids=valid_ids,
                ignored_evidence_ids=ignored_ids,
                reasons=["仅由 C 级线索支撑，不能进入结论"],
            )

        # 第 8 步检测提前准备：多条网页是否属于同一事件簇或同原始发布方
        single_source_merged, merge_notes = self._check_web_source_merging(valid_cards)

        # 第 6 & 7 步：抽取主张中的数字并核验
        extracted_numbers = extract_financial_numbers(claim.text)
        b_cards = [c for c in valid_cards if c.grade == EvidenceGrade.B]

        number_checks: list[NumberCheckDetail] = []
        doubtful_reasons: list[str] = []

        for num in extracted_numbers:
            a_match = self._match_in_a_cards(num, expanded_a_cards)
            if a_match is not None:
                eid, detail = a_match
                number_checks.append(
                    NumberCheckDetail(
                        raw_text=num.raw_text,
                        parsed_value=num.value,
                        unit=num.unit,
                        tolerance=num.tolerance,
                        matched=True,
                        matched_grade=EvidenceGrade.A,
                        matched_evidence_id=eid,
                        matched_detail=detail,
                    )
                )
                continue

            b_match = self._match_in_b_cards(num, b_cards)
            if b_match is not None:
                eid, detail = b_match
                number_checks.append(
                    NumberCheckDetail(
                        raw_text=num.raw_text,
                        parsed_value=num.value,
                        unit=num.unit,
                        tolerance=num.tolerance,
                        matched=False,
                        matched_grade=EvidenceGrade.B,
                        matched_evidence_id=eid,
                        matched_detail=detail,
                    )
                )
                doubtful_reasons.append(f"数字 {num.raw_text} 仅出现在 B 级网页({eid})，须回到财报复核")
            else:
                number_checks.append(
                    NumberCheckDetail(
                        raw_text=num.raw_text,
                        parsed_value=num.value,
                        unit=num.unit,
                        tolerance=num.tolerance,
                        matched=False,
                        matched_detail="未在引用证据中匹配到",
                    )
                )
                doubtful_reasons.append(f"数字 {num.raw_text} 无法由引用证据复算")

        if doubtful_reasons:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.DOUBTFUL,
                valid_evidence_ids=valid_ids,
                ignored_evidence_ids=ignored_ids,
                number_checks=number_checks,
                single_source_merged=single_source_merged,
                reasons=doubtful_reasons,
                notes=merge_notes,
            )

        notes = list(merge_notes)
        if not expanded_a_cards and b_cards:
            notes.append("无数值主张，仅由 B 级网页说明发生了什么，须标明来源")

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.VERIFIED,
            valid_evidence_ids=valid_ids,
            ignored_evidence_ids=ignored_ids,
            number_checks=number_checks,
            single_source_merged=single_source_merged,
            notes=notes,
        )

    def verify_all(
        self,
        claims: list[Claim],
        evidence_store: dict[str, EvidenceCard],
        *,
        as_of: date,
    ) -> list[VerificationResult]:
        return [self.verify_claim(c, evidence_store, as_of=as_of) for c in claims]

    @staticmethod
    def _expand_a_grade_cards(
        valid_cards: list[EvidenceCard],
        evidence_store: dict[str, EvidenceCard],
    ) -> list[EvidenceCard]:
        """收集引用的 A 级卡，并递归纳入计算卡 derived_from 所依据的 A 级财报/行情卡。"""
        result: list[EvidenceCard] = []
        visited: set[str] = set()
        stack = [c for c in valid_cards if c.grade == EvidenceGrade.A]

        while stack:
            card = stack.pop()
            if card.evidence_id in visited:
                continue
            visited.add(card.evidence_id)
            result.append(card)
            for parent_id in card.derived_from:
                parent = evidence_store.get(parent_id)
                if parent is not None and parent.grade == EvidenceGrade.A and parent_id not in visited:
                    stack.append(parent)
        return result

    @staticmethod
    def _dedup_cards(cards: list[EvidenceCard]) -> list[EvidenceCard]:
        seen: set[str] = set()
        out: list[EvidenceCard] = []
        for c in cards:
            if c.evidence_id not in seen:
                seen.add(c.evidence_id)
                out.append(c)
        return out

    @staticmethod
    def _match_in_a_cards(
        target: ExtractedNumber,
        a_cards: list[EvidenceCard],
    ) -> tuple[str, str] | None:
        if not a_cards:
            return None

        # 特殊规则：“个百分点”拿两个百分比指标之差去匹配
        if target.unit == MetricUnit.PERCENTAGE_POINT:
            pct_entries: list[tuple[str, MetricEntry]] = []
            for card in a_cards:
                for m in card.metrics:
                    if m.unit == MetricUnit.PERCENT:
                        pct_entries.append((card.evidence_id, m))

            for i, (eid1, m1) in enumerate(pct_entries):
                for j, (eid2, m2) in enumerate(pct_entries):
                    if i == j:
                        continue
                    # 优先要求同名指标跨期比较（如果两张卡都有 period_end，确保新期 - 旧期方向一致）
                    if m1.period_end and m2.period_end and m1.period_end <= m2.period_end:
                        continue
                    diff = m1.value - m2.value
                    if target.direction == "neutral":
                        matched = abs(abs(diff) - abs(target.value)) <= target.tolerance
                    else:
                        matched = abs(diff - target.value) <= target.tolerance
                    if matched:
                        return (
                            f"{eid1},{eid2}",
                            f"{m1.name}({m1.period}:{m1.value}%) - {m2.name}({m2.period}:{m2.value}%) = {diff}pp",
                        )
            # 若没有 period_end，退化为任意两个不同百分比之差匹配
            for i, (eid1, m1) in enumerate(pct_entries):
                for j, (eid2, m2) in enumerate(pct_entries):
                    if i == j or (m1.period_end and m2.period_end):
                        continue
                    diff = m1.value - m2.value
                    if abs(diff - target.value) <= target.tolerance:
                        return (
                            f"{eid1},{eid2}",
                            f"{m1.name}({m1.value}%) - {m2.name}({m2.value}%) = {diff}pp",
                        )
            return None

        # 常规指标：金额 / 百分比 / 倍数 按同类量纲匹配
        for card in a_cards:
            for m in card.metrics:
                if target.unit in AMOUNT_UNITS and m.unit in AMOUNT_UNITS:
                    target_cny = to_base_cny(target.value, target.unit)
                    tol_cny = to_base_cny(target.tolerance, target.unit)
                    metric_cny = to_base_cny(m.value, m.unit)
                    if abs(metric_cny - target_cny) <= tol_cny:
                        return (
                            card.evidence_id,
                            f"匹配 A 级指标 {m.name}={m.value}{m.unit.value}",
                        )
                elif target.unit == m.unit:
                    if abs(m.value - target.value) <= target.tolerance:
                        return (
                            card.evidence_id,
                            f"匹配 A 级指标 {m.name}={m.value}{m.unit.value}",
                        )
        return None

    @staticmethod
    def _match_in_b_cards(
        target: ExtractedNumber,
        b_cards: list[EvidenceCard],
    ) -> tuple[str, str] | None:
        for card in b_cards:
            b_nums = extract_financial_numbers(f"{card.title}\n{card.excerpt}")
            for bn in b_nums:
                if target.unit in AMOUNT_UNITS and bn.unit in AMOUNT_UNITS:
                    if abs(to_base_cny(bn.value, bn.unit) - to_base_cny(target.value, target.unit)) <= to_base_cny(target.tolerance, target.unit):
                        return card.evidence_id, f"仅命中 B 级网页数字 {bn.raw_text}"
                elif target.unit == bn.unit and abs(bn.value - target.value) <= target.tolerance:
                    return card.evidence_id, f"仅命中 B 级网页数字 {bn.raw_text}"
        return None

    @staticmethod
    def _check_web_source_merging(valid_cards: list[EvidenceCard]) -> tuple[bool, list[str]]:
        web_cards = [c for c in valid_cards if c.grade in (EvidenceGrade.B, EvidenceGrade.C)]
        if len(web_cards) < 2:
            return False, []

        cluster_counts: dict[str, list[str]] = {}
        orig_counts: dict[str, list[str]] = {}
        for c in web_cards:
            if c.event_cluster_id:
                cluster_counts.setdefault(c.event_cluster_id, []).append(c.evidence_id)
            if c.original_source:
                orig_counts.setdefault(c.original_source, []).append(c.evidence_id)

        notes: list[str] = []
        merged = False
        for cid, eids in cluster_counts.items():
            if len(eids) > 1:
                merged = True
                notes.append(f"网页 {', '.join(eids)} 属于同一事件簇 {cid}，按单一来源计")
        for orig, eids in orig_counts.items():
            if len(eids) > 1:
                merged = True
                notes.append(f"网页 {', '.join(eids)} 同源自“{orig}”，按单一来源计")
        return merged, notes
