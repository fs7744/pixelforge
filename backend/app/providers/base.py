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
        refs: Optional[List[Image.Image]] = None,
    ) -> ProviderResult:
        """ref / refs 都是参考图入参：

        - Z-Image 是纯文生图，喂参考图会抛错（不会静默忽略）；
        - Qwen-Image-Edit 走指令式编辑，refs 支持多张（最多 4 张）语义组合，
          例如「角色正面 + 侧面」「角色 + 场景」。
        """
        raise NotImplementedError

    async def test(self) -> Dict[str, Any]:
        return {"ok": True, "message": "未实现连接测试"}
