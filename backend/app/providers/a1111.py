"""本地 A1111 / Forge Provider：走 /sdapi/v1/txt2img。"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

import httpx
from PIL import Image

from .base import BaseProvider, ProviderResult


class A1111Provider(BaseProvider):
    id = "a1111"
    label = "本地 A1111 / Forge"

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
        strength: float = 0.6,
    ) -> ProviderResult:
        base = (self.cfg.get("baseUrl") or "http://127.0.0.1:7860").rstrip("/")
        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, jpeg artifacts",
            "steps": int(steps),
            "cfg_scale": float(cfg),
            "width": int(width),
            "height": int(height),
            "seed": -1 if seed is None else int(seed),
            "n_iter": max(1, min(n, 4)),
            "sampler_name": "DPM++ 2M Karras",
            "override_settings": {"sd_model_checkpoint": self.cfg.get("model")}
            if self.cfg.get("model") else {},
        }
        endpoint = "txt2img"
        if ref is not None:
            # 图生图：把参考图 base64 塞进 init_images，用 denoising_strength 控制改动力度
            buf = io.BytesIO()
            ref.convert("RGB").resize((int(width), int(height))).save(buf, "PNG")
            payload["init_images"] = [base64.b64encode(buf.getvalue()).decode()]
            payload["denoising_strength"] = float(min(max(strength, 0.1), 0.95))
            endpoint = "img2img"
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{base}/sdapi/v1/{endpoint}", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"A1111 返回 {r.status_code}: {r.text[:300]}")
        data = r.json()
        images = [Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
                  for b64 in data.get("images", [])]
        if not images:
            raise RuntimeError("A1111 未返回图片")
        return ProviderResult(images=images, meta={"provider": self.id, "info": str(data.get("info", ""))[:200]})

    async def test(self) -> Dict[str, Any]:
        base = (self.cfg.get("baseUrl") or "http://127.0.0.1:7860").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{base}/sdapi/v1/sd-models")
            if r.status_code < 400:
                models = r.json()
                return {"ok": True, "message": f"连接成功，检测到 {len(models)} 个模型"}
            return {"ok": False, "message": f"服务响应异常（{r.status_code}）"}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "message": f"连接失败：{e}"}
