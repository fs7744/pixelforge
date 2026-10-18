from .base import BaseProvider, ProviderResult
from .diffusers_local import DiffusersProvider

# 只保留本地双底座：Z-Image-Turbo（文生图）与 Qwen-Image-Edit-2511（指令编辑）。
# 模拟引擎与 SD/CLIP 系（A1111、云端 OpenAI 兼容接口）已整体移除。
REGISTRY = {
    DiffusersProvider.id: DiffusersProvider,
}


def get_provider(provider_id: str, cfg: dict) -> BaseProvider:
    cls = REGISTRY.get(provider_id)
    if not cls:
        raise KeyError(f"未知 Provider: {provider_id}")
    return cls(cfg or {})
