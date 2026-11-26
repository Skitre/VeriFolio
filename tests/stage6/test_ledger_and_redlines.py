"""RSI 只追加三表账本、四要素坏案例与红线拦截单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zhengyan.rsi import (
    AppendOnlyRSILedger,
    BadCaseRecord,
    BadCaseSource,
    LedgerViolationError,
    RSIComponent,
    RSIRedlineViolationError,
    SixGatesReport,
    assert_not_redline_target,
)


def test_bad_case_requires_explicit_hypothesis() -> None:
    """文档第 20 题：没有根因假设的问题不允许登记，避免为了改而改。"""
    with pytest.raises(ValidationError, match="根因假设"):
        BadCaseRecord(
            case_id="BC-BAD",
            source_case_ref="600519 v5",
            failure_type=BadCaseSource.UPHELD_CHALLENGE,
            evidence_summary="合同负债下降被质询成立",
            hypothesis="太差",
        )

    ok = BadCaseRecord(
        case_id="BC-OK",
        source_case_ref="600519 v5",
        failure_type=BadCaseSource.UPHELD_CHALLENGE,
        evidence_summary="基本面专家凭合同负债下降直接推出需求走弱，被反证官以打款节奏调整推翻",
        hypothesis="我猜是因为起草阶段缺少消费品合同负债归因核查提示，未先排查打款政策变化",
        symbol="600519",
        period="2025H1",
    )
    assert ok.case_id == "BC-OK"


def test_redline_targets_cannot_enter_rsi() -> None:
    """文档第 25 题：权限、隐私、核验规则、交易边界、用户约束、模型权重永不进入自进化。"""
    for forbidden in (
        "permissions",
        "privacy_rules",
        "verifier_rules",
        "trading_boundary",
        "user_constraints",
        "model_weights",
    ):
        with pytest.raises(RSIRedlineViolationError):
            assert_not_redline_target(forbidden)


def test_append_only_ledger_event_sourcing_reason_length_and_rollback() -> None:
    ledger = AppendOnlyRSILedger()
    ledger.init_component(RSIComponent.S1_EXPERIENCE_CARDS, version="v1", payload={"cards": []})
    assert ledger.get_active_version(RSIComponent.S1_EXPERIENCE_CARDS) == "v1"

    passed_gate = SixGatesReport(proposal_id="P1", component=RSIComponent.S1_EXPERIENCE_CARDS, passed=True)
    failed_gate = SixGatesReport(
        proposal_id="P2", component=RSIComponent.S1_EXPERIENCE_CARDS, passed=False, failed_gates=["noise_gate"]
    )

    # 1. 门控未通过禁止发布
    with pytest.raises(LedgerViolationError, match="门控未通过"):
        ledger.publish_candidate(
            component=RSIComponent.S1_EXPERIENCE_CARDS,
            candidate_version="v2",
            parent_version="v1",
            change_reason="沉淀消费品合同负债变动归因核查经验卡",
            payload={"cards": ["c1"]},
            gate_report=failed_gate,
            approver="alice",
            human_approved=True,
        )

    # 2. 缺人工批准禁止发布（三权分立：优化器不能自己上线）
    with pytest.raises(LedgerViolationError, match="人工批准"):
        ledger.publish_candidate(
            component=RSIComponent.S1_EXPERIENCE_CARDS,
            candidate_version="v2",
            parent_version="v1",
            change_reason="沉淀消费品合同负债变动归因核查经验卡",
            payload={"cards": ["c1"]},
            gate_report=passed_gate,
            approver="optimizer",
            human_approved=False,
        )

    # 3. 改动理由少于 10 个字拒绝写入
    with pytest.raises(LedgerViolationError, match="少于 10 个字"):
        ledger.publish_candidate(
            component=RSIComponent.S1_EXPERIENCE_CARDS,
            candidate_version="v2",
            parent_version="v1",
            change_reason="加卡片",
            payload={"cards": ["c1"]},
            gate_report=passed_gate,
            approver="alice",
            human_approved=True,
        )

    # 4. 合法发布 v2（如实标注单人自批 self_approved），内容哈希不可篡改
    v2 = ledger.publish_candidate(
        component=RSIComponent.S1_EXPERIENCE_CARDS,
        candidate_version="v2",
        parent_version="v1",
        change_reason="从茅台深研成立质询沉淀消费品合同负债归因核查经验卡",
        payload={"cards": ["L-007"]},
        gate_report=passed_gate,
        approver="solo_dev",
        human_approved=True,
        approval_mode="self_approved",
    )
    assert v2.parent_version == "v1"
    assert len(v2.content_hash) == 64
    assert ledger.get_active_version(RSIComponent.S1_EXPERIENCE_CARDS) == "v2"

    # 5. 禁止覆盖已有版本 v2
    with pytest.raises(LedgerViolationError, match="禁止修改或覆盖"):
        ledger.publish_candidate(
            component=RSIComponent.S1_EXPERIENCE_CARDS,
            candidate_version="v2",
            parent_version="v2",
            change_reason="试图覆盖已经发布的第二版经验卡资产",
            payload={"cards": ["tampered"]},
            gate_report=passed_gate,
            approver="solo_dev",
            human_approved=True,
        )

    # 6. 一键回滚：追加 rollback 事件，由事件流自动推导回到 v1，历史资产完整保留
    rolled = ledger.rollback(
        RSIComponent.S1_EXPERIENCE_CARDS,
        operator="solo_dev",
        reason="上线观察期触发异常，一键回滚到已验证基线版本",
    )
    assert rolled == "v1"
    assert ledger.get_active_version(RSIComponent.S1_EXPERIENCE_CARDS) == "v1"
    assert ledger.get_asset(RSIComponent.S1_EXPERIENCE_CARDS, "v2") is not None
    assert [e.event_type for e in ledger.events] == ["init", "publish", "rollback"]
