"""MongoDB 风格只追加三表 RSI 账本（对应文档第 24 题）。

核心不可变与事件溯源约束：
1. 三张只追加表：`assets`（资产版本）、`runs`（评测与门控运行记录）、`events`（发布/回滚/拒绝事件）；
2. 每个资产版本一旦写入永不可改，记录 parent_version、>=10字改动理由、sha256 content_hash；
3. 当前在用版本绝不是一个随手可改的变量，而是由 `events` 表按时间顺序重放（Event Sourcing）推导得出；
4. 被拒绝的候选连同 direction_key 和拒绝原因一起记账，供优化器避开已失败方向；
5. 支持一条命令一键回滚，并完整保留历史轨迹。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from zhengyan.rsi.models import (
    EvalProtocolFingerprint,
    LedgerAssetRecord,
    LedgerEventRecord,
    LedgerRunRecord,
    RSIComponent,
    SixGatesReport,
    SliceEvalStats,
)


class LedgerViolationError(ValueError):
    """违反只追加账本纪律（如覆盖旧版本、理由少于 10 字、未过门控强行发布）时抛出。"""


def compute_content_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class AppendOnlyRSILedger:
    """只追加三表 RSI 版本账本（离线内存实现，表结构与 MongoDB 集合 1:1 对应）。"""

    def __init__(self) -> None:
        self._assets: list[LedgerAssetRecord] = []
        self._runs: list[LedgerRunRecord] = []
        self._events: list[LedgerEventRecord] = []

    @property
    def assets(self) -> tuple[LedgerAssetRecord, ...]:
        return tuple(self._assets)

    @property
    def runs(self) -> tuple[LedgerRunRecord, ...]:
        return tuple(self._runs)

    @property
    def events(self) -> tuple[LedgerEventRecord, ...]:
        return tuple(self._events)

    def init_component(
        self,
        component: RSIComponent,
        *,
        version: str = "v1",
        payload: dict[str, Any],
        reason: str = "初始化系统基线版本资产，建立可追溯版本起点",
    ) -> LedgerAssetRecord:
        asset = self._append_asset(
            component=component,
            version=version,
            parent_version=None,
            change_reason=reason,
            payload=payload,
        )
        self._events.append(
            LedgerEventRecord(
                event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                event_type="init",
                component=component,
                version=version,
                actor="system",
                approval_mode="system",
                reason=reason,
            )
        )
        return asset

    def _append_asset(
        self,
        *,
        component: RSIComponent,
        version: str,
        parent_version: str | None,
        change_reason: str,
        payload: dict[str, Any],
    ) -> LedgerAssetRecord:
        if len(change_reason.strip()) < 10:
            raise LedgerViolationError(
                f"改动理由不得少于 10 个字（当前仅 {len(change_reason.strip())} 字）"
            )
        if any(a.component == component and a.version == version for a in self._assets):
            raise LedgerViolationError(
                f"资产 {component.value}@{version} 已存在：账本只允许追加新版本，禁止修改或覆盖！"
            )
        rec = LedgerAssetRecord(
            component=component,
            version=version,
            parent_version=parent_version,
            change_reason=change_reason.strip(),
            content_hash=compute_content_hash(payload),
            payload=payload,
        )
        self._assets.append(rec)
        return rec

    def get_asset(self, component: RSIComponent, version: str) -> LedgerAssetRecord | None:
        for a in self._assets:
            if a.component == component and a.version == version:
                return a
        return None

    def get_active_version(self, component: RSIComponent) -> str | None:
        """严格由 events 表顺序推导当前在用版本（Event Sourcing，不设可变 active 字段）。"""
        active: str | None = None
        for ev in self._events:
            if ev.component != component:
                continue
            if ev.event_type in ("init", "publish", "rollback"):
                active = ev.version
        return active

    def get_active_asset(self, component: RSIComponent) -> LedgerAssetRecord | None:
        ver = self.get_active_version(component)
        if ver is None:
            return None
        return self.get_asset(component, ver)

    def get_active_versions_map(self) -> dict[str, str]:
        """返回各组件当前在用版本号字典，供研究档案记录继承快照（如 经验库 v2 · 路由 v1 · 检索 v2）。"""
        out: dict[str, str] = {}
        for comp in RSIComponent:
            ver = self.get_active_version(comp)
            if ver is not None:
                out[comp.value] = ver
        return out

    def record_run(
        self,
        *,
        proposal_id: str,
        component: RSIComponent,
        candidate_version: str,
        direction_key: str,
        protocol: EvalProtocolFingerprint,
        baseline_slices: dict[str, SliceEvalStats],
        candidate_slices: dict[str, SliceEvalStats],
        gate_report: SixGatesReport,
    ) -> LedgerRunRecord:
        run = LedgerRunRecord(
            run_id=f"RUN-{uuid.uuid4().hex[:8]}",
            proposal_id=proposal_id,
            component=component,
            candidate_version=candidate_version,
            direction_key=direction_key,
            protocol=protocol,
            baseline_slices=baseline_slices,
            candidate_slices=candidate_slices,
            gate_report=gate_report,
        )
        self._runs.append(run)
        if not gate_report.passed:
            self._events.append(
                LedgerEventRecord(
                    event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                    event_type="reject",
                    component=component,
                    version=candidate_version,
                    actor="six_gates",
                    approval_mode="system",
                    reason=f"门控未通过：{', '.join(gate_report.failed_gates)}",
                )
            )
        return run

    def publish_candidate(
        self,
        *,
        component: RSIComponent,
        candidate_version: str,
        parent_version: str,
        change_reason: str,
        payload: dict[str, Any],
        gate_report: SixGatesReport,
        approver: str,
        human_approved: bool,
        approval_mode: Literal["independent_approved", "self_approved"] = "independent_approved",
        approval_note: str = "",
    ) -> LedgerAssetRecord:
        """三权分立发布约束：必须六道门控全通过 + 人工显式批准 + 改动理由 >= 10 字。"""
        if not gate_report.passed:
            raise LedgerViolationError(f"门控未通过，禁止发布：{gate_report.failed_gates}")
        if not human_approved or not approver.strip():
            raise LedgerViolationError("RSI 资产发布必须经过人工批准，优化器无权自行上线")
        current = self.get_active_version(component)
        if current != parent_version:
            raise LedgerViolationError(
                f"父版本冲突：当前在用版本为 {current}，但候选声明的 parent={parent_version}"
            )

        effective_reason = approval_note.strip() or change_reason.strip()
        if len(effective_reason) < 10:
            raise LedgerViolationError("批准/发布理由不得少于 10 个字")

        asset = self._append_asset(
            component=component,
            version=candidate_version,
            parent_version=parent_version,
            change_reason=change_reason,
            payload=payload,
        )
        self._events.append(
            LedgerEventRecord(
                event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                event_type="publish",
                component=component,
                version=candidate_version,
                actor=approver,
                approval_mode=approval_mode,
                reason=effective_reason,
            )
        )
        return asset

    def rollback(
        self,
        component: RSIComponent,
        *,
        operator: str,
        reason: str,
    ) -> str:
        """一键回滚到当前版本的 parent_version（追加 rollback 事件，不删任何历史记录）。"""
        if len(reason.strip()) < 10:
            raise LedgerViolationError("回滚理由不得少于 10 个字")
        current_ver = self.get_active_version(component)
        if current_ver is None:
            raise LedgerViolationError(f"组件 {component.value} 尚未初始化")
        current_asset = self.get_asset(component, current_ver)
        if current_asset is None or current_asset.parent_version is None:
            raise LedgerViolationError(f"组件 {component.value}@{current_ver} 已在初始版本，无可回滚父版本")

        target_ver = current_asset.parent_version
        self._events.append(
            LedgerEventRecord(
                event_id=f"EVT-{uuid.uuid4().hex[:8]}",
                event_type="rollback",
                component=component,
                version=target_ver,
                actor=operator,
                approval_mode="self_approved",
                reason=reason.strip(),
            )
        )
        return target_ver

    def failed_directions(self, component: RSIComponent) -> set[str]:
        """返回账本中已被门控拒绝过的改动方向集合，避免优化器重复试错。"""
        return {
            r.direction_key
            for r in self._runs
            if r.component == component and not r.gate_report.passed
        }

    def consecutive_failed_rounds(self, component: RSIComponent) -> int:
        """统计某组件最近连续失败轮次（长期无进展时切换组件）。"""
        comp_runs = [r for r in self._runs if r.component == component]
        streak = 0
        for run in reversed(comp_runs):
            if run.gate_report.passed:
                break
            streak += 1
        return streak
