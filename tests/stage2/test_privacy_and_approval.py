"""隐私保护与审批票据单元测试（对应文档第 11、11.2、12 题）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from zhengyan.harness import ApprovalTicket, PrivacyGuard


def test_privacy_detect_mask_and_outbound_block() -> None:
    guard = PrivacyGuard()
    text = (
        "我有存款 30万，房贷每月 8000元，手机号 13812345678，"
        "银行卡 6222021234567890123，贵州茅台 600519 还要不要拿？"
    )
    hits = guard.detect(text)
    categories = {h.category for h in hits}
    assert "personal_finance" in categories
    assert "phone" in categories
    assert "bank_card" in categories

    masked = guard.mask_for_llm(text)
    assert "13812345678" not in masked
    assert "6222021234567890123" not in masked
    assert "30万" not in masked
    assert "8000元" not in masked
    # 股票代码和公司名绝不能被误脱敏
    assert "600519" in masked
    assert "贵州茅台" in masked

    with pytest.raises(ValueError, match="外发查询命中个人隐私"):
        guard.assert_safe_outbound("贵州茅台 我的持仓 50万 走势")

    # 正常投研查询顺利放行
    guard.assert_safe_outbound("贵州茅台 600519 2025年半年报 合同负债")


def test_approval_ticket_binding_and_600s_expiry() -> None:
    now = datetime(2025, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
    params = {"symbol": "600519", "limit": 5}
    ticket = ApprovalTicket.issue(
        ticket_id="TKT-001",
        skill_name="custom.export",
        version="1.0.0",
        params=params,
        approver="user_01",
        ttl_seconds=600,
        now=now,
    )

    # 有效期内、参数完全一致 → 通过
    ok, _ = ticket.verify(
        skill_name="custom.export",
        version="1.0.0",
        params={"limit": 5, "symbol": "600519"},  # key 顺序不同但规范哈希一致
        now=now + timedelta(seconds=599),
    )
    assert ok is True

    # 参数篡改 → 拒绝
    ok_tampered, reason_tampered = ticket.verify(
        skill_name="custom.export",
        version="1.0.0",
        params={"symbol": "600519", "limit": 50},
        now=now + timedelta(seconds=10),
    )
    assert ok_tampered is False
    assert "参数哈希不一致" in reason_tampered

    # 超过 600 秒 → 过期拒绝
    ok_expired, reason_expired = ticket.verify(
        skill_name="custom.export",
        version="1.0.0",
        params=params,
        now=now + timedelta(seconds=601),
    )
    assert ok_expired is False
    assert "已过期" in reason_expired
