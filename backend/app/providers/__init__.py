from .a1111 import A1111Provider
from .base import BaseProvider, ProviderResult
from .diffusers_local import DiffusersProvider
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider

REGISTRY = {
    MockProvider.id: MockProvider,
    DiffusersProvider.id: DiffusersProvider,
    OpenAICompatProvider.id: OpenAICompatProvider,
    A1111Provider.id: A1111Provider,
}


def get_provider(provider_id: str, cfg: dict) -> BaseProvider:
    cls = REGISTRY.get(provider_id)
    if not cls:
        raise KeyError(f"未知 Provider: {provider_id}")
    return cls(cfg or {})
