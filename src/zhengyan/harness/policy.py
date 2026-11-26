"""加载 config/policies.yaml 的治理策略配置对象。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from zhengyan.core.enums import ResearchTier
from zhengyan.harness.budget import BudgetTracker


@dataclass(frozen=True)
class TierPolicy:
    tier: ResearchTier
    max_tokens: int
    max_searches: int
    max_fetches: int
    max_duration_seconds: int
    max_supplement_rounds: int
    allowed_skills: frozenset[str]


@dataclass(frozen=True)
class HarnessPolicyConfig:
    ticket_ttl_seconds: int
    forbidden_prefixes: tuple[str, ...]
    max_retries: int
    backoff_base_seconds: float
    tiers: dict[ResearchTier, TierPolicy]

    @classmethod
    def from_yaml(cls, path: Path) -> HarnessPolicyConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        approval_cfg = data.get("approval") or {}
        retry_cfg = data.get("retry") or {}
        tiers_raw = data.get("tiers") or {}

        tiers: dict[ResearchTier, TierPolicy] = {}
        for tier_key, info in tiers_raw.items():
            tier_enum = ResearchTier(tier_key)
            tiers[tier_enum] = TierPolicy(
                tier=tier_enum,
                max_tokens=int(info["max_tokens"]),
                max_searches=int(info["max_searches"]),
                max_fetches=int(info["max_fetches"]),
                max_duration_seconds=int(info["max_duration_seconds"]),
                max_supplement_rounds=int(info["max_supplement_rounds"]),
                allowed_skills=frozenset(info.get("allowed_skills") or []),
            )

        return cls(
            ticket_ttl_seconds=int(approval_cfg.get("ticket_ttl_seconds", 600)),
            forbidden_prefixes=tuple(approval_cfg.get("forbidden_prefixes") or ("action.",)),
            max_retries=int(retry_cfg.get("max_retries", 2)),
            backoff_base_seconds=float(retry_cfg.get("backoff_base_seconds", 1.0)),
            tiers=tiers,
        )

    def create_budget(self, tier: ResearchTier) -> BudgetTracker:
        tp = self.tiers[tier]
        return BudgetTracker(
            tier=tier,
            max_tokens=tp.max_tokens,
            max_searches=tp.max_searches,
            max_fetches=tp.max_fetches,
            max_duration_seconds=tp.max_duration_seconds,
        )
