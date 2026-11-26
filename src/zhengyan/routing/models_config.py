"""模型轴配置加载器（对应文档第 4 题：双轴路由解耦任务轴与模型轴）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from zhengyan.core.enums import ResearchTier


@dataclass(frozen=True)
class ModelRoleSpec:
    role: str
    provider: str
    model: str
    fallback_model: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ModelAxisConfig:
    roles: dict[str, ModelRoleSpec]
    tier_role_bindings: dict[ResearchTier, dict[str, str]]

    @classmethod
    def from_yaml(cls, path: Path) -> ModelAxisConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        roles: dict[str, ModelRoleSpec] = {}
        for role_name, info in (data.get("roles") or {}).items():
            roles[role_name] = ModelRoleSpec(
                role=role_name,
                provider=str(info["provider"]),
                model=str(info["model"]),
                fallback_model=info.get("fallback_model"),
                description=str(info.get("description", "")),
            )

        bindings: dict[ResearchTier, dict[str, str]] = {}
        for tier_key, mapping in (data.get("tier_role_bindings") or {}).items():
            bindings[ResearchTier(tier_key)] = {str(k): str(v) for k, v in (mapping or {}).items()}
        return cls(roles=roles, tier_role_bindings=bindings)

    def resolve_bindings(self, tier: ResearchTier) -> dict[str, ModelRoleSpec]:
        """返回某研究档位下各节点/专家对应的具体模型规格。"""
        mapping = self.tier_role_bindings.get(tier) or {}
        return {slot: self.roles[role_name] for slot, role_name in mapping.items()}

    def resolve_slot(self, tier: ResearchTier, slot: str) -> ModelRoleSpec:
        role_name = self.tier_role_bindings[tier][slot]
        return self.roles[role_name]
