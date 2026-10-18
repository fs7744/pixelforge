"""内置 diffusers 推理服务：本地真模型（无需第三方 UI）。

设计要点：
- 懒加载：不装 torch/diffusers 也能启动服务，只在首次生成时报错提示。
- 单例缓存：同名模型只加载一次；切换模型时释放旧管线。
- 显存自适应：<8GB 自动开启 attention slicing + vae slicing，必要时 cpu offload。
- 异步：阻塞推理放到线程池，不卡事件循环。

模型族（2026-10 新增 Z-Image / Qwen-Image-Edit）：
- sd：SD1.5/SDXL/turbo 系（CLIP 编码器，中文须走 zh2en 翻译链路；
      可挂 LCM/像素 LoRA + IP-Adapter）
- zimage：阿里 Z-Image-Turbo（S3-DiT 单流 + Qwen3-4B 中文编码器）：
      * 中文提示词直喂（supports_zh=True），翻译链路整个绕过
      * bf16 · 8 步蒸馏 · guidance_scale 固定 0.0 · 1024² 原生分辨率
      * SD1.5 的 LoRA/IP-Adapter/LCM 均不适用（UNet 结构不同），
        一致性请用参考图 img2img 切回 SD 族，或后续接 Z-Image-Editor
- qwen_edit：阿里 Qwen-Image-Edit-2511（20B MMDiT + Qwen2.5-VL 编码器）：
      * 指令式图像编辑（必须传参考图），中文指令直喂
      * 角色一致性第二底座：帧 1 用 Z-Image 生成，帧 2+ 用它改姿势
      * 40 步 · true_cfg_scale=4.0 · guidance_scale=1.0 · FP8/bf16
      * 2511 版专项优化角色一致性与抗漂移，内置多视角/光照 LoRA
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseProvider, ProviderResult

_CACHE: Dict[str, Any] = {}
_CAPS: Dict[str, Dict[str, Any]] = {}
_FAMILY: Dict[str, str] = {}
_LOCK = asyncio.Lock()


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def detect_family(model_id: str) -> str:
    """模型族识别：zimage / qwen_edit / sd。

    先读 model_index.json 的 _class_name（最可靠），再看目录名兜底。
    """
    if model_id in _FAMILY:
        return _FAMILY[model_id]
    family = "sd"
    if os.path.isdir(model_id):
        idx = _read_json(os.path.join(model_id, "model_index.json"))
        cls = str(idx.get("_class_name") or "")
        if "ZImage" in cls:
            family = "zimage"
        elif "QwenImageEdit" in cls:
            family = "qwen_edit"
        elif cls:
            family = "sd"
    if family == "sd":
        name = os.path.basename(model_id.rstrip("/\\")).lower()
        if "z-image" in name or "zimage" in name:
            family = "zimage"
        elif "qwen-image-edit" in name or "qwen_image_edit" in name:
            family = "qwen_edit"
    _FAMILY[model_id] = family
    return family


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

    @property
    def supports_zh(self) -> bool:
        """Z-Image / Qwen-Image-Edit 的中文编码器原生吃中文；SD 系（CLIP）仍走翻译链路。"""
        try:
            return detect_family(self._resolve_model()) in ("zimage", "qwen_edit")
        except Exception:
            return False

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
        family = detect_family(model_id)
        if family == "zimage":
            return self._load_zimage(model_id)
        if family == "qwen_edit":
            return self._load_qwen_edit(model_id)
        return self._load_sd(model_id)

    def _load_zimage(self, model_id: str):
        """Z-Image-Turbo：bf16 单流 S3-DiT + Qwen3-4B 中文编码器，8 步蒸馏。"""
        try:
            import torch
            from diffusers import ZImagePipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Z-Image 需要较新版本的 diffusers（含 ZImagePipeline）。请执行：\n"
                "  pip install -U diffusers\n"
                f"原始错误：{e}"
            )
        # cuDNN 卷积算法启发式在显存紧张时会选中非 Tensor Core 慢速路径
        # （实测 4090 上 VAE 解码从 5s 劣化到 4 分钟以上，非确定性复现），
        # 开 benchmark 按实际 shape 自动调优，确定性走最快算法
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Z-Image 是 bf16 单流 DiT，需要 CUDA GPU（如 RTX 4090）；CPU 推理不可行。"
                "无 GPU 请切回 SD1.5 系模型。"
            )
        # low_cpu_mem_usage 保持默认 True：实测 False 会让推理慢 10 倍以上
        # （主线程直连对照实验：True 加载 20s + 推理 10s；False 推理 4-17 分钟），
        # 且 True 加载完整性无问题（同种子出图与 False 完全一致）。
        pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        caps: Dict[str, Any] = {
            "family": "zimage", "lora": None, "ipAdapter": None, "lcm": None,
            "notes": ["Z-Image：中文提示词直喂无需翻译；SD1.5 LoRA/IP-Adapter 不适用"],
        }
        _CAPS[model_id] = caps
        _CACHE[model_id] = pipe
        return pipe

    def _load_qwen_edit(self, model_id: str):
        """Qwen-Image-Edit-2511：bf16 20B MMDiT + Qwen2.5-VL 编码器，指令式编辑。

        与 Z-Image 同为阿里系，但分工不同：它只做「参考图 + 中文指令 → 编辑图」，
        是角色一致性（帧 2+ 换姿势）的主力底座。bf16 全量约 42GB，超出 24GB，
        因此开 enable_model_cpu_offload：transformer/text_encoder 按层换入换出，
        速度比全载慢约 30%，但 24GB 卡可跑；有 FP8 量化版（约 20GB）可去掉 offload。
        """
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Qwen-Image-Edit 需要较新版本的 diffusers（含 QwenImageEditPlusPipeline）。请执行：\n"
                "  pip install -U diffusers\n"
                f"原始错误：{e}"
            )
        # 同 _load_zimage：防止 cuDNN 选中慢速卷积路径劣化 VAE 解码
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen-Image-Edit 是 20B DiT，需要 CUDA GPU（如 RTX 4090）；CPU 推理不可行。"
                "无 GPU 请切回 SD1.5 系模型。"
            )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        caps: Dict[str, Any] = {
            "family": "qwen_edit", "lora": None, "ipAdapter": None, "lcm": None,
            "notes": ["Qwen-Image-Edit：中文指令直喂无需翻译；需传参考图；cpu offload 模式"],
        }
        _CAPS[model_id] = caps
        _CACHE[model_id] = pipe
        return pipe

    def _load_sd(self, model_id: str):
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
        if use_cuda:
            # 同 _load_zimage：防止 cuDNN 选中慢速卷积路径劣化 VAE 解码
            torch.backends.cudnn.benchmark = True
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

    def _run_zimage(self, model_id: str, prompt: str, width: int, height: int,
                    steps: Optional[int], seed: Optional[int], generator: Any, n: int):
        """Z-Image 文生图：8 次 DiT 前向、CFG 固定 0、1024² 原生分辨率。

        注意：
        - 调度器把 step 0 当热身，num_inference_steps=9 才对应 8 次前向；
        - Turbo 蒸馏已内化 CFG，guidance_scale 给正值反而掉画质；
        - 参考图（ref）在当前版本忽略：Z-Image 暂无 img2img 管线，
          角色一致性请切回 SD 族 refMode=ip/img2img，或后续接 Z-Image-Editor。
        """
        import torch

        pipe = self._load(model_id)
        use_steps = 9 if not steps else max(4, min(int(steps), 12))
        w = int(width) // 16 * 16
        h = int(height) // 16 * 16
        images = []
        for i in range(max(1, min(n, 4))):
            out = pipe(
                prompt=prompt,
                num_inference_steps=use_steps,
                guidance_scale=0.0,
                height=h,
                width=w,
                generator=generator,
            )
            images.append(out.images[0].convert("RGBA"))
            if generator is not None and seed is not None:
                generator = torch.Generator(device=generator.device).manual_seed(int(seed) + i + 1)
        return images, {"steps": use_steps, "cfg": 0.0, "family": "zimage",
                        "mode": "txt2img", "turbo": True}

    def _run_qwen_edit(self, model_id: str, prompt: str, negative: str,
                       width: int, height: int, steps: Optional[int],
                       seed: Optional[int], generator: Any, n: int, ref: Any):
        """Qwen-Image-Edit 指令式编辑：参考图 + 中文指令 → 编辑图。

        与 txt2img 的本质区别：prompt 是「编辑指令」而非「画面描述」。
        角色一致性指令模板：
          「保持角色外观、服装、配色完全不变，将姿势改为：{action}，纯色背景」
        - 必须传 ref（没有参考图时编辑器无输入，直接报错而不是静默出图）；
        - 输出尺寸跟随参考图（对齐 16），不另外传 width/height；
        - true_cfg_scale=4.0 是官方推荐（正负提示词对比强度）；
          guidance_scale=1.0 关掉二次 CFG（蒸馏已内化）。
        """
        import torch

        if ref is None:
            raise ValueError(
                "Qwen-Image-Edit 是指令式编辑器，必须传参考图（ref）。"
                "请通过工作流的 useRef 步骤或生成接口传入 ref。"
            )

        pipe = self._load(model_id)
        use_steps = 40 if not steps else max(20, min(int(steps), 50))
        from PIL import Image as _Image

        src = ref.convert("RGB")
        w = int(width or src.width) // 16 * 16
        h = int(height or src.height) // 16 * 16
        src = src.resize((w, h), _Image.LANCZOS)
        images = []
        for i in range(max(1, min(n, 4))):
            out = pipe(
                image=src,
                prompt=prompt,
                negative_prompt=negative or " ",
                num_inference_steps=use_steps,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                generator=generator,
            )
            images.append(out.images[0].convert("RGBA"))
            if generator is not None and seed is not None:
                generator = torch.Generator(device=generator.device).manual_seed(int(seed) + i + 1)
        return images, {"steps": use_steps, "cfg": 1.0, "trueCfg": 4.0,
                        "family": "qwen_edit", "mode": "edit"}

    def _run(self, model_id: str, prompt: str, negative: str, width: int, height: int,
             steps: int, cfg: float, seed: Optional[int], n: int,
             ref: Optional[Any] = None, strength: float = 0.6):
        import torch
        from PIL import Image as _Image

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

        family = _CAPS.get(model_id, {}).get("family") or detect_family(model_id)
        if family == "zimage":
            return self._run_zimage(model_id, prompt, width, height, steps, seed, generator, n)
        if family == "qwen_edit":
            return self._run_qwen_edit(model_id, prompt, negative, width, height,
                                       steps, seed, generator, n, ref)
        # SD 族：必须先加载——caps 是 _load 里写入的，先读会拿到空字典，
        # 导致 IP-Adapter 明明挂上了却不传参考图 → UNet 报 NoneType
        base_pipe = self._load(model_id)
        w = int(width) // 8 * 8
        h = int(height) // 8 * 8
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
        family = detect_family(model_id)
        if steps is None:
            # 蒸馏模型只需极少步数：Z-Image 9 步（8 次前向）、turbo 4 步、
            # Qwen-Image-Edit 40 步（非蒸馏，2511 有 Lightning 4 步变体可后续换）、
            # 普通 SD 28 步；28 步纯属浪费
            if family == "zimage":
                steps = 9
            elif family == "qwen_edit":
                steps = 40
            elif "turbo" in model_id.lower():
                steps = 4
            else:
                steps = 28
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
        family = detect_family(model)
        zh_tag = "（中文直喂）" if family in ("zimage", "qwen_edit") else ""
        return {
            "ok": True,
            "message": f"{st['device']} · {st['vramGB']}GB 显存 · 模型 {model} · 族 {family}"
                       + zh_tag
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
