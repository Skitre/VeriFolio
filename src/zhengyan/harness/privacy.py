"""隐私识别、外发拦截与大模型输入脱敏（对应文档第 12 题）。

覆盖四类个人敏感信息：
- 身份证号（18 位）
- 银行卡号（16~19 位数字）
- 手机号（中国大陆 11 位）
- 个人财务陈述（“我的持仓/存款/房贷/资产/仓位/本金 + 金额”）
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "id_card",
        re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
        "[已脱敏:身份证号]",
    ),
    (
        "bank_card",
        re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3,4}\d{3,4}(?!\d)"),
        "[已脱敏:银行卡号]",
    ),
    (
        "phone",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "[已脱敏:手机号]",
    ),
    (
        "personal_finance",
        re.compile(
            r"(?:我的|本人|家里|目前|现在)?(?:持仓|仓位|存款|房贷|月供|本金|身家|总资产|可投资资金|负债)"
            r"(?:有|约|大概|大概有|每月|一共|总共|为|是)?\s*\d+(?:\.\d+)?\s*(?:万|万元|元|亿|亿元|k|K|千|块)?"
        ),
        "[已脱敏:个人财务信息]",
    ),
]


@dataclass(frozen=True)
class PIIHit:
    category: str
    matched_text: str


class PrivacyGuard:
    """隐私检测、外发拦截与脱敏器。"""

    def detect(self, text: str) -> list[PIIHit]:
        hits: list[PIIHit] = []
        # 避免身份证号/手机号被银行卡号重复匹配，按优先级记录已覆盖区间
        occupied: list[tuple[int, int]] = []
        for category, pattern, _ in _PII_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                if category == "bank_card":
                    digits_only = re.sub(r"\D", "", match.group(0))
                    if not (16 <= len(digits_only) <= 19):
                        continue
                if any(not (end <= s or start >= e) for s, e in occupied):
                    continue
                occupied.append((start, end))
                hits.append(PIIHit(category=category, matched_text=match.group(0)))
        return hits

    def has_pii(self, text: str) -> bool:
        return bool(self.detect(text))

    def mask_for_llm(self, text: str) -> str:
        """发往大模型前对所有命中的敏感信息进行脱敏替换。"""
        masked = text
        for category, pattern, placeholder in _PII_PATTERNS:
            if category == "bank_card":
                def _replace_card(m: re.Match[str], ph: str = placeholder) -> str:
                    digits = re.sub(r"\D", "", m.group(0))
                    return ph if 16 <= len(digits) <= 19 else m.group(0)
                masked = pattern.sub(_replace_card, masked)
            else:
                masked = pattern.sub(placeholder, masked)
        return masked

    def assert_safe_outbound(self, text: str) -> None:
        """外发搜索前硬校验：命中隐私直接禁止发出。"""
        hits = self.detect(text)
        if hits:
            cats = ", ".join(sorted({h.category for h in hits}))
            raise ValueError(f"外发查询命中个人隐私信息（{cats}），已阻断发送")
