"""审批票据与参数哈希绑定（对应文档第 11 & 11.2 题）。

设计细节：
- 审批票据绑定 (skill_name, version, params_hash)，任何参数篡改都会导致验签失败；
- 默认 600 秒过期，绝不默认同意；
- 交易类 Skill（action.*）在策略层一律禁止，即便持有票据也不放行。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


def canonical_params_hash(params: dict[str, Any]) -> str:
    """对参数字典做规范 JSON 序列化后计算 sha256，供幂等与审批绑定共用。"""
    encoded = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ApprovalTicket:
    ticket_id: str
    skill_name: str
    version: str
    params_hash: str
    approver: str
    approved: bool
    issued_at: datetime
    expires_at: datetime

    @classmethod
    def issue(
        cls,
        *,
        ticket_id: str,
        skill_name: str,
        version: str,
        params: dict[str, Any],
        approver: str,
        approved: bool = True,
        ttl_seconds: int = 600,
        now: datetime | None = None,
    ) -> ApprovalTicket:
        ts = now or datetime.now(timezone.utc)
        return cls(
            ticket_id=ticket_id,
            skill_name=skill_name,
            version=version,
            params_hash=canonical_params_hash(params),
            approver=approver,
            approved=approved,
            issued_at=ts,
            expires_at=ts + timedelta(seconds=ttl_seconds),
        )

    def verify(
        self,
        *,
        skill_name: str,
        version: str,
        params: dict[str, Any],
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        ts = now or datetime.now(timezone.utc)
        if not self.approved:
            return False, "审批未通过（已拒绝）"
        if self.skill_name != skill_name or self.version != version:
            return False, f"审批票据绑定目标不匹配（期望 {skill_name}@{version}）"
        actual_hash = canonical_params_hash(params)
        if self.params_hash != actual_hash:
            return False, "调用参数与审批票据参数哈希不一致，疑似参数变更"
        if ts > self.expires_at:
            return False, f"审批票据已过期（有效期截至 {self.expires_at.isoformat()}）"
        return True, "ok"
