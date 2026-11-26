"""OpenTelemetry 风格的全链路审计收集器（对应文档第 11 & 30 题）。"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from zhengyan.core.models import AuditSpan


class AuditTrail:
    """记录单个研究 trace 下的全部节点、Skill、模型与审批 Span。"""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or f"tr-{uuid.uuid4().hex[:12]}"
        self.spans: list[AuditSpan] = []

    def record(
        self,
        *,
        node_name: str,
        caller: str,
        action_type: Literal["node", "skill", "llm", "decision", "approval"],
        target: str,
        status: Literal["ok", "denied", "error", "cached", "interrupted"],
        duration_ms: float = 0.0,
        tokens_used: int = 0,
        parent_span_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> AuditSpan:
        span = AuditSpan(
            trace_id=self.trace_id,
            span_id=f"sp-{uuid.uuid4().hex[:8]}",
            parent_span_id=parent_span_id,
            node_name=node_name,
            caller=caller,
            action_type=action_type,
            target=target,
            status=status,
            duration_ms=round(duration_ms, 2),
            tokens_used=tokens_used,
            detail=detail or {},
        )
        self.spans.append(span)
        return span
