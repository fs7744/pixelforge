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
from typing import Any, Dict, List, Optional

from .base import BaseProvider, ProviderResult

_CACHE: Dict[str, Any] = {}
_CAPS: Dict[str, Dict[str, Any]] = {}
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

    def _resolve_ip_adapter(self) -> Optional[str]:
        """IP-Adapter 目录（含 models/ip-adapter_sd15.safetensors 与 image_encoder）。"""
        path = (self.cfg.get("ipAdapter") or "").strip()
        if not path:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "models", "ip-adapter-sd15")
        if not os.path.isdir(path):
            return None
        adapter = os.path.join(path, "models", "ip-adapter_sd15.safetensors")
        encoder_dir = os.path.join(path, "models", "image_encoder")
        encoder = any(os.path.exists(os.path.join(encoder_dir, f))
                      for f in ("model.safetensors", "pytorch_model.bin"))
        # 缺任何一个都别硬加载：半套权重只会抛异常，还白占内存
        return path if os.path.exists(adapter) and encoder else None

    def _resolve_lora(self) -> Optional[str]:
        """像素风格 LoRA 文件或目录。"""
        path = (self.cfg.get("lora") or "").strip()
        if path and os.path.exists(path):
            return path
        default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "models", "lora-pixel-redmond")
        if os.path.isdir(default):
            files = [f for f in os.listdir(default) if f.endswith(".safetensors")]
            if files:
                return os.path.join(default, files[0])
        return None

    def _resolve_lcm(self) -> Optional[str]:
        """LCM 蒸馏 LoRA：把 SD1.5 的 25 步压到 4 步，CPU 才跑得动。"""
        path = (self.cfg.get("lcmLora") or "").strip()
        if path and os.path.exists(path):
            return path
        default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "models", "lcm-lora-sdv1-5")
        if os.path.isdir(default):
            files = [f for f in os.listdir(default) if f.endswith(".safetensors")]
            if files:
                return os.path.join(default, files[0])
        return None

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
        # 本地目录只有 fp16 权重时必须显式声明 variant，否则会找不到 fp32 分片
        variant = None
        if os.path.isdir(model_id):
            fp32 = any(os.path.exists(os.path.join(model_id, sub, f)) for sub, f in (
                ("unet", "diffusion_pytorch_model.safetensors"),
                ("text_encoder", "model.safetensors")))
            fp16 = any(os.path.exists(os.path.join(model_id, sub, f)) for sub, f in (
                ("unet", "diffusion_pytorch_model.fp16.safetensors"),
                ("text_encoder", "model.fp16.safetensors")))
            if fp16 and not fp32:
                variant = "fp16"
        if variant:
            kwargs["variant"] = variant
        elif use_cuda:
            kwargs["variant"] = None

        # 允许单独指定 VAE（缺 VAE 权重时可用同架构 VAE 顶替，先跑通再换回）
        vae_path = (self.cfg.get("vae") or "").strip()
        if vae_path and os.path.isdir(vae_path):
            try:
                from diffusers import AutoencoderKL

                kwargs["vae"] = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
            except Exception:  # noqa: BLE001
                pass

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

        caps: Dict[str, Any] = {"lora": None, "ipAdapter": None, "lcm": None, "notes": []}
        adapters: List[str] = []
        weights: List[float] = []

        # IP-Adapter 必须先挂：它会给 UNet 加 encoder_hid_dim_type，
        # 一旦先包了 LoRA 的 peft 包装，再挂 IP 会半途失败并留下"脏"UNet
        # （之后每次前向都缺 image_embeds，直接报 NoneType not iterable）
        ip_dir = self._resolve_ip_adapter()
        if ip_dir:
            try:
                pipe.load_ip_adapter(ip_dir, subfolder="models",
                                     weight_name="ip-adapter_sd15.safetensors")
                cap_scale = float(self.cfg.get("ipAdapterScale", 0.7))
                pipe.set_ip_adapter_scale(cap_scale)
                caps["ipAdapter"] = {"dir": ip_dir, "scale": cap_scale}
            except Exception as e:  # noqa: BLE001
                caps["notes"].append(f"IP-Adapter 加载失败：{e}")
                # 回滚：不让半个 IP-Adapter 把管线弄成不可用
                try:
                    pipe.unet.config.encoder_hid_dim_type = None
                except Exception:  # noqa: BLE001
                    pass

        # LCM 蒸馏 LoRA 要先挂：它决定采样器与步数策略
        lcm = self._resolve_lcm()
        if lcm:
            try:
                pipe.load_lora_weights(lcm, adapter_name="lcm")
                adapters.append("lcm")
                weights.append(1.0)
                caps["lcm"] = {"path": lcm}
            except Exception as e:  # noqa: BLE001
                caps["notes"].append(f"LCM LoRA 加载失败：{e}")

        # 像素风格 LoRA：26MB，直接把「插画感」压成像素块
        lora = self._resolve_lora()
        if lora:
            try:
                scale = float(self.cfg.get("loraScale", 0.85))
                pipe.load_lora_weights(lora, adapter_name="pixel")
                adapters.append("pixel")
                weights.append(scale)
                caps["lora"] = {"path": lora, "scale": scale}
            except Exception as e:  # noqa: BLE001
                caps["notes"].append(f"LoRA 加载失败：{e}")

        if adapters:
            try:
                pipe.set_adapters(adapters, adapter_weights=weights)
            except Exception as e:  # noqa: BLE001
                caps["notes"].append(f"set_adapters 失败：{e}")

        if caps["lcm"]:
            try:
                from diffusers import LCMScheduler

                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                caps["lcm"]["scheduler"] = "LCMScheduler"
            except Exception as e:  # noqa: BLE001
                caps["notes"].append(f"LCM 采样器设置失败：{e}")

        # IP-Adapter：参考图控身份，提示词控姿势，互不抢权重

        _CAPS[model_id] = caps
        _CACHE[model_id] = pipe
        return pipe

    def _load_i2i(self, model_id: str):
        """图生图管线：与文生图共享同一份权重（from_pipe），不额外占显存。"""
        key = f"{model_id}::i2i"
        if key in _CACHE:
            return _CACHE[key]
        from diffusers import AutoPipelineForImage2Image

        base = self._load(model_id)
        pipe = AutoPipelineForImage2Image.from_pipe(base)
        pipe.set_progress_bar_config(disable=True)
        _CACHE[key] = pipe
        return pipe

    def _run(self, model_id: str, prompt: str, negative: str, width: int, height: int,
             steps: int, cfg: float, seed: Optional[int], n: int,
             ref: Optional[Any] = None, strength: float = 0.6):
        import torch
        from PIL import Image as _Image

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

        w = int(width) // 8 * 8
        h = int(height) // 8 * 8

        # 必须先加载：caps 是 _load 里写入的，先读会拿到空字典，
        # 导致 IP-Adapter 明明挂上了却不传参考图 → UNet 报 NoneType
        base_pipe = self._load(model_id)
        caps = _CAPS.get(model_id, {})
        is_lcm = bool(caps.get("lcm"))
        is_turbo = "turbo" in model_id.lower()
        # Turbo / LCM 这类蒸馏模型都用极少步数与低 guidance
        if is_turbo:
            use_steps = max(1, min(int(steps), 8))
            use_cfg = 0.0
        elif is_lcm:
            use_steps = max(2, min(int(steps), 8))
            use_cfg = 1.0
        else:
            use_steps = max(4, min(int(steps), 60))
            use_cfg = float(cfg)

        # 有 IP-Adapter 就走「文生图 + 参考图控身份」：姿势由提示词自由发挥，
        # 不再受 img2img 初始图布局的牵制。refMode=img2img 可退回老路子。
        use_ip = (ref is not None and bool(caps.get("ipAdapter"))
                  and str(self.cfg.get("refMode") or "ip") != "img2img")
        i2i = ref is not None and not use_ip

        pipe = self._load_i2i(model_id) if i2i else base_pipe
        init = None
        if i2i:
            init = ref.convert("RGB").resize((w, h), _Image.LANCZOS)
            strength = float(min(max(strength, 0.2), 0.95))
        ip_image = ref.convert("RGB") if use_ip else None

        # 一旦挂上 IP-Adapter，UNet 每次前向都要求 image_embeds：
        # 没有参考图时必须传一张占位图并把强度调到 0，否则直接抛 NoneType 错误
        if caps.get("ipAdapter"):
            pipe.set_ip_adapter_scale(caps["ipAdapter"]["scale"] if use_ip else 0.0)
            if ip_image is None:
                ip_image = _Image.new("RGB", (64, 64), (127, 127, 127))

        images = []
        for i in range(max(1, min(n, 4))):
            kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": negative or None,
                "num_inference_steps": use_steps,
                "guidance_scale": use_cfg,
                "generator": generator,
            }
            if i2i:
                kwargs.update({"image": init, "strength": strength, "width": w, "height": h})
            else:
                kwargs.update({"width": w, "height": h})
            if ip_image is not None:
                kwargs["ip_adapter_image"] = ip_image
            out = pipe(**kwargs)
            images.append(out.images[0].convert("RGBA"))
            if generator is not None:
                generator = torch.Generator(device=generator.device).manual_seed(int(seed) + i + 1)
        mode = "ip-adapter" if use_ip else ("img2img" if i2i else "txt2img")
        return images, {"steps": use_steps, "cfg": use_cfg, "turbo": is_turbo,
                        "mode": mode,
                        "lora": bool(caps.get("lora")),
                        **({"strength": round(strength, 2)} if i2i else {}),
                        **({"ipScale": caps["ipAdapter"]["scale"]} if use_ip else {})}

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
        strength: float = 0.6,
    ) -> ProviderResult:
        model_id = self._resolve_model()
        if steps is None:
            # 蒸馏模型（turbo / LCM）只需极少步数，28 步纯属浪费
            steps = 4 if "turbo" in model_id.lower() else 28
        async with _LOCK:
            images, meta = await asyncio.to_thread(
                self._run, model_id, prompt, self.cfg.get("negativePrompt", ""),
                width, height, steps, cfg, seed, n, ref, strength,
            )
        status = _torch_status()
        return ProviderResult(
            images=images,
            meta={"provider": "diffusers", "model": model_id, "device": status.get("device"),
                  **meta, "caps": _CAPS.get(model_id)},
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
        caps = _CAPS.get(model, {})
        extra = []
        if caps.get("lora"):
            extra.append(f"LoRA {os.path.basename(caps['lora']['path'])}@{caps['lora']['scale']}")
        if caps.get("ipAdapter"):
            extra.append(f"IP-Adapter@{caps['ipAdapter']['scale']}")
        return {
            "ok": True,
            "message": f"{st['device']} · {st['vramGB']}GB 显存 · 模型 {model}"
                       + ("" if model not in _CACHE else " · 已加载")
                       + (f" · {' · '.join(extra)}" if extra else ""),
            "caps": caps,
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
