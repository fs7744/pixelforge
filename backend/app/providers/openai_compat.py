"""OpenAI 兼容 Images API Provider（也适用于多数云端/自建兼容服务）。"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

import httpx
from PIL import Image

from .base import BaseProvider, ProviderResult


class OpenAICompatProvider(BaseProvider):
    id = "openai_compat"
    label = "云端 API（OpenAI 兼容）"
    # 云端多模态模型（Gemini / 即梦等）中文理解良好，直接吃中文模板
    supports_zh = True

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
        base = (self.cfg.get("baseUrl") or "").rstrip("/")
        key = self.cfg.get("apiKey") or ""
        model = self.cfg.get("model") or "gpt-image-1"
        if not base or not key:
            raise RuntimeError("云端 API 未配置 baseUrl 或 apiKey")

        payload = {
            "model": model,
            "prompt": prompt,
            "n": max(1, min(n, 4)),
            "size": f"{width}x{height}",
        }
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{base}/images/generations",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"云端返回 {resp.status_code}: {resp.text[:300]}")
            data = resp.json()

        images = []
        for item in data.get("data", []):
            if item.get("b64_json"):
                raw = base64.b64decode(item["b64_json"])
                images.append(Image.open(io.BytesIO(raw)).convert("RGBA"))
            elif item.get("url"):
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.get(item["url"])
                    r.raise_for_status()
                    images.append(Image.open(io.BytesIO(r.content)).convert("RGBA"))
        if not images:
            raise RuntimeError("云端未返回图片")
        return ProviderResult(images=images, meta={"provider": self.id, "model": model})

    async def test(self) -> Dict[str, Any]:
        base = (self.cfg.get("baseUrl") or "").rstrip("/")
        key = self.cfg.get("apiKey") or ""
        if not base or not key:
            return {"ok": False, "message": "请先填写 baseUrl 与 apiKey"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
            if r.status_code < 400:
                return {"ok": True, "message": "连接成功"}
            return {"ok": False, "message": f"鉴权失败（{r.status_code}）"}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "message": f"连接失败：{e}"}
