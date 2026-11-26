"""LLM 统一抽象协议（按角色调用，预留 DeepSeek 等真实模型接入）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True)
class StructuredLLMResponse(Generic[TModel]):
    parsed: TModel
    tokens_used: int
    model_name: str


class LLMClient(Protocol):
    """结构化输出模型调用接口。"""

    def generate_structured(
        self,
        *,
        role: str,
        prompt: str,
        response_model: type[TModel],
        context: dict[str, Any],
    ) -> StructuredLLMResponse[TModel]:
        ...
