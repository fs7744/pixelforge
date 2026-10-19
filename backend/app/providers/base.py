"""Provider 抽象：所有生成后端统一接口。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class ProviderResult:
    images: List[Image.Image]
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseProvider:
    id = "base"
    label = "Base"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}

    async def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 28,
        cfg: float = 7.5,
        seed: Optional[int] = None,
        n: int = 1,
        ref: Optional[Image.Image] = None,
    ) -> ProviderResult:
        raise NotImplementedError

    async def test(self) -> Dict[str, Any]:
        return {"ok": True, "message": "未实现连接测试"}
