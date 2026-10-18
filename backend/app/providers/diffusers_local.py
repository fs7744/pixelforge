"""内置 diffusers 推理服务：本地真模型（无需第三方 UI）。

设计要点：
- 懒加载：不装 torch/diffusers 也能启动服务，只在首次生成时报错提示。
- 单例缓存：同名模型只加载一次；切换模型时释放旧管线。
- 显存自适应：<8GB 自动开启 attention slicing + vae slicing，必要时 cpu offload。
- 异步：阻塞推理放到线程池，不卡事件循环。
"""
from __future__ import annotations

import asyncio
import gc
import os
from typing import Any, Dict, Optional

from .base import BaseProvider, ProviderResult

_CACHE: Dict[str, Any] = {}
_LOCK = asyncio.Lock()


def _torch_status() -> Dict[str, Any]:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        return {"installed": False, "error": str(e)}
    import torch

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vramGB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        if torch.cuda.is_available() else 0,
    }


class DiffusersProvider(BaseProvider):
    id = "diffusers"
    label = "内置推理服务（diffusers）"

    def _resolve_model(self) -> str:
        model = (self.cfg.get("model") or "").strip()
        if model:
            return model
        # 默认：SD-Turbo，1-4 步出图，6GB 显存友好
        return os.environ.get("PIXELFORGE_MODEL", "stabilityai/sd-turbo")

    def _load(self, model_id: str):
        if model_id in _CACHE:
            return _CACHE[model_id]

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "未安装 AI 依赖。请执行：\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "  pip install diffusers transformers accelerate safetensors\n"
                f"原始错误：{e}"
            )

        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        kwargs: Dict[str, Any] = {"torch_dtype": dtype, "safety_checker": None}
        if use_cuda:
            kwargs["variant"] = None

        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
        if use_cuda:
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pipe = pipe.to("cuda")
            if vram < 8:
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        else:
            pipe = pipe.to("cpu")

        pipe.set_progress_bar_config(disable=True)
        _CACHE[model_id] = pipe
        return pipe

    def _run(self, model_id: str, prompt: str, negative: str, width: int, height: int,
             steps: int, cfg: float, seed: Optional[int], n: int):
        import torch

        pipe = self._load(model_id)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

        # Turbo 系列 guidance_scale 必须为 0，步数少
        is_turbo = "turbo" in model_id.lower()
        use_steps = max(1, min(int(steps), 8)) if is_turbo else max(4, min(int(steps), 60))
        use_cfg = 0.0 if is_turbo else float(cfg)

        images = []
        for i in range(max(1, min(n, 4))):
            out = pipe(
                prompt=prompt,
                negative_prompt=negative or None,
                num_inference_steps=use_steps,
                guidance_scale=use_cfg,
                width=int(width) // 8 * 8,
                height=int(height) // 8 * 8,
                generator=generator,
            )
            images.append(out.images[0].convert("RGBA"))
            if generator is not None:
                generator = torch.Generator(device=generator.device).manual_seed(int(seed) + i + 1)
        return images, {"steps": use_steps, "cfg": use_cfg, "turbo": is_turbo}

    async def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: Optional[int] = None,
        cfg: float = 7.5,
        seed: Optional[int] = None,
        n: int = 1,
        ref=None,
    ) -> ProviderResult:
        model_id = self._resolve_model()
        if steps is None:
            # 蒸馏模型（turbo / LCM）只需极少步数，28 步纯属浪费
            steps = 4 if "turbo" in model_id.lower() else 28
        async with _LOCK:
            images, meta = await asyncio.to_thread(
                self._run, model_id, prompt, self.cfg.get("negativePrompt", ""),
                width, height, steps, cfg, seed, n,
            )
        status = _torch_status()
        return ProviderResult(
            images=images,
            meta={"provider": "diffusers", "model": model_id, "device": status.get("device"),
                  **meta},
        )

    async def test(self) -> Dict[str, Any]:
        st = _torch_status()
        if not st["installed"]:
            return {"ok": False, "message": "未安装 torch，请按 README 安装 AI 依赖"}
        model = self._resolve_model()
        import importlib.util

        has_diffusers = importlib.util.find_spec("diffusers") is not None
        if not has_diffusers:
            return {"ok": False, "message": "未安装 diffusers"}
        return {
            "ok": True,
            "message": f"{st['device']} · {st['vramGB']}GB 显存 · 模型 {model}"
                       + ("" if model not in _CACHE else " · 已加载"),
        }

    @staticmethod
    def unload() -> None:
        _CACHE.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
