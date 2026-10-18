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

_FP8_LINEAR_CLS: Any = None


def _get_fp8_linear_cls() -> Any:
    """懒加载 _FP8Linear 类（本模块遵循「无 torch 也能 import」的设计）。

    fp8(E4M3) 权重量化 Linear：权重存 fp8 + 逐输出通道 scale，前向时反量化
    成 bf16 再算。与 torchao Float8WeightOnlyConfig 数学等价，但权重是普通
    fp8 张量（safetensors 0.4.2+ 原生支持、bit 级往返一致），因此可落盘缓存；
    量化在 GPU 上逐层完成（峰值显存=单层），20B 模型约 10s（CPU 全量 58s）。
    """
    global _FP8_LINEAR_CLS
    if _FP8_LINEAR_CLS is None:
        import torch
        import torch.nn.functional as F

        class _FP8Linear(torch.nn.Linear):
            def forward(self, x):  # type: ignore[override]
                w = self.weight.to(x.dtype)
                scale = getattr(self, "w_scale", None)
                if scale is not None:
                    w = w * scale.to(x.dtype)[:, None]
                return F.linear(x, w, self.bias)

        _FP8_LINEAR_CLS = _FP8Linear
    return _FP8_LINEAR_CLS


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
        """Qwen-Image-Edit-2511：20B MMDiT + Qwen2.5-VL 编码器，指令式编辑。

        与 Z-Image 同为阿里系，但分工不同：它只做「参考图 + 中文指令 → 编辑图」，
        是角色一致性（帧 2+ 换姿势）的主力底座。

        显存策略（实测踩坑）：bf16 全量约 54GB（transformer 单独 39GB），
        远超 24GB。原设想「bf16 + enable_model_cpu_offload」不可行——
        offload 钩子在前向时要把整个 transformer 搬上 GPU，39GB 必爆 OOM
        （torch.OutOfMemoryError: tried to allocate ... 23.99GiB total）。
        解决：transformer 做 fp8 权重量化（39GB → 约 20GB），再开 cpu offload
        让 text_encoder(14GB)/VAE 按阶段换入换出。4090 Ada 架构上 fp8 开销可忽略。
        量化结果落盘缓存（transformer-fp8/），冷启动从 ~100s 降到 ~35s。
        可用环境变量 PIXELFORGE_QWEN_FP8=0 关闭量化（需 >39GB 显存的卡）。
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
        use_fp8 = os.environ.get("PIXELFORGE_QWEN_FP8", "1") != "0"
        transformer = None
        if use_fp8:
            try:
                transformer = self._load_qwen_transformer_fp8(model_id)
            except Exception as e:
                print(f"[pixelforge] fp8 transformer 快速路径失败（{e}），退回完整加载", flush=True)
                transformer = None
        if transformer is not None:
            # 注入已加载的 transformer：from_pretrained 对已实例化组件跳过磁盘读取
            #（省掉管线侧 39GB bf16 读），其余组件（text_encoder/VAE）正常加载
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, transformer=transformer)
        else:
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16)
            if use_fp8:
                # 兜底：torchao CPU 量化（58s，慢但经过完整冒烟验证）
                self._quantize_fp8(pipe.transformer, "qwen_edit.transformer")
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        caps: Dict[str, Any] = {
            "family": "qwen_edit", "lora": None, "ipAdapter": None, "lcm": None,
            "notes": ["Qwen-Image-Edit：中文指令直喂无需翻译；需传参考图；transformer fp8 + cpu offload"],
        }
        _CAPS[model_id] = caps
        _CACHE[model_id] = pipe
        return pipe

    # ---------- Qwen-Image-Edit fp8 transformer：量化 + 落盘缓存 ----------

    def _load_qwen_transformer_fp8(self, model_id: str):
        """加载 fp8 版 transformer（约 20GB，进 24GB 卡的关键）。

        两条路径：
        - 缓存命中（<model>/transformer-fp8/）：safetensors mmap 零拷贝 assign，
          约 20s（整机 31.8GB 内存下 bf16 39GB 本就靠 mmap，缓存同理）；
        - 缓存未命中：bf16 完整加载（mmap）→ GPU 逐层 fp8 量化（约 10s）
          → 落盘缓存（约 20s，下次冷启动直接命中）。
        任何失败都抛异常，由 _load_qwen_edit 退回「完整加载 + torchao 量化」。
        """
        import time
        from pathlib import Path

        import torch
        from diffusers import QwenImageTransformer2DModel

        model_dir = Path(model_id)
        cache_dir = model_dir / "transformer-fp8"
        if self._fp8_cache_valid(cache_dir, model_dir):
            t0 = time.time()
            t = self._load_fp8_cache(cache_dir)
            print(f"[pixelforge] fp8 缓存命中：{cache_dir.name}/（{time.time() - t0:.0f}s，"
                  f"跳过 39GB bf16 读取与量化）", flush=True)
            return t

        t0 = time.time()
        t = QwenImageTransformer2DModel.from_pretrained(
            str(model_dir / "transformer"), torch_dtype=torch.bfloat16)
        t_load = time.time() - t0
        n = self._quantize_transformer_gpu(t)
        t_quant = time.time() - t0 - t_load
        try:
            self._save_fp8_cache(t, cache_dir, model_dir)
        except Exception as e:
            print(f"[pixelforge] fp8 缓存落盘失败（{e}），不影响本次运行", flush=True)
        print(f"[pixelforge] transformer fp8 量化完成（{n} 个 Linear，"
              f"bf16 加载 {t_load:.0f}s + GPU 量化 {t_quant:.0f}s，缓存已写 "
              f"{cache_dir.name}/）", flush=True)
        return t

    @staticmethod
    def _fp8_cache_valid(cache_dir, model_dir) -> bool:
        """缓存有效性：三件套齐全 + 源 transformer 权重文件指纹未变。

        环境变量 PIXELFORGE_QWEN_CACHE=0 可强制 miss（调试/重新量化用）。
        """
        if os.environ.get("PIXELFORGE_QWEN_CACHE", "1") == "0":
            return False
        if not ((cache_dir / "model.safetensors").is_file()
                and (cache_dir / "config.json").is_file()
                and (cache_dir / "fp8_cache_meta.json").is_file()):
            return False
        try:
            meta = json.loads((cache_dir / "fp8_cache_meta.json").read_text())
            for name, (size, mtime) in meta.get("files", {}).items():
                f = model_dir / "transformer" / name
                if not f.is_file() or f.stat().st_size != size or int(f.stat().st_mtime) != mtime:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _save_fp8_cache(transformer, cache_dir, model_dir) -> None:
        """把量化后的 transformer 存成 diffusers 风格缓存目录（约 20GB）。

        键布局与 diffusers 权重一致：<路径>.weight 为 float8_e4m3fn，
        <路径>.w_scale 为 fp32 逐输出通道 scale，其余（norm/adaLN 的 bf16 等）
        原样保存。safetensors 原生支持 fp8 且往返 bit 级一致（已实测）。
        """
        import shutil

        from safetensors.torch import save_file

        cache_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().contiguous() for k, v in transformer.state_dict().items()}
        save_file(sd, str(cache_dir / "model.safetensors"))
        shutil.copy(str(model_dir / "transformer" / "config.json"),
                    str(cache_dir / "config.json"))
        # 源指纹：权重文件名 + 大小 + mtime（重新下载/换版本即失效，自动重建）
        files = {}
        for f in sorted((model_dir / "transformer").glob("*.safetensors")):
            st = f.stat()
            files[f.name] = [st.st_size, int(st.st_mtime)]
        (cache_dir / "fp8_cache_meta.json").write_text(json.dumps(
            {"source": str(model_dir), "format": "fp8-e4m3-rowwise", "files": files},
            ensure_ascii=False, indent=1))

    @staticmethod
    def _load_fp8_cache(cache_dir):
        """从缓存快速搭出 fp8 transformer：meta 骨架 + mmap assign + 换类。

        用 diffusers 同款低内存加载路径（accelerate init_empty_weights +
        load_state_dict(assign=True)），20B 模型不分配 bf16 空张量；
        safetensors load_file 返回 mmap 视图，assign 后参数直接指向映射页。
        """
        import torch
        from accelerate import init_empty_weights
        from diffusers import QwenImageTransformer2DModel
        from safetensors.torch import load_file

        fp8_cls = _get_fp8_linear_cls()
        config = QwenImageTransformer2DModel.load_config(str(cache_dir / "config.json"))
        sd = load_file(str(cache_dir / "model.safetensors"))
        try:
            with init_empty_weights():
                model = QwenImageTransformer2DModel.from_config(config)
        except Exception:
            # meta 路径失败时退回普通 from_config（CPU 上空张量，仅多耗内存）
            model = QwenImageTransformer2DModel.from_config(config)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # unexpected 必须全部是 .w_scale（骨架无此 buffer）；missing 必须为空
        scales = [k for k in unexpected if k.endswith(".w_scale")]
        if missing or len(scales) != len(unexpected):
            raise RuntimeError(
                f"缓存键不匹配（missing={len(missing)} unexpected={len(unexpected)}），"
                "删除 transformer-fp8/ 可重建")
        for key in scales:
            mod = model.get_submodule(key[: -len(".w_scale")])
            mod.register_buffer("w_scale", sd[key])
        n = 0
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear) and getattr(mod, "w_scale", None) is not None:
                mod.__class__ = fp8_cls
                n += 1
        if n == 0:
            raise RuntimeError("缓存中没有任何 fp8 Linear，格式可能损坏")
        return model

    @staticmethod
    def _quantize_transformer_gpu(transformer) -> int:
        """逐 Linear 在 GPU 上做 fp8 权重量化，原地换类为 _FP8Linear。

        E4M3 + 逐输出通道 scale（scale = amax(|W|, dim=1) / 448）。
        峰值显存仅为单层权重（最大约 75MB），20B 模型全程约 10s；
        CPU 全量量化实测 58s。与 torchao Float8WeightOnlyConfig 数学等价。
        """
        import torch

        fp8_cls = _get_fp8_linear_cls()
        n = 0
        n_linear = sum(1 for m in transformer.modules()
                       if type(m) is torch.nn.Linear)
        for mod in transformer.modules():
            if type(mod) is not torch.nn.Linear:
                continue
            w = mod.weight.detach().to("cuda", non_blocking=True)
            scale = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12) / 448.0
            wq = (w / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
            mod.weight = torch.nn.Parameter(wq.to("cpu"), requires_grad=False)
            mod.register_buffer("w_scale", scale.squeeze(1).to("cpu").float())
            mod.__class__ = fp8_cls
            n += 1
        if n != n_linear:  # 理论上相等；不等说明有 Linear 子类，日志可见
            print(f"[pixelforge] 警告：{n_linear} 个 Linear 中仅量化 {n} 个", flush=True)
        return n

    @staticmethod
    def _quantize_fp8(module: Any, tag: str) -> Any:
        """torchao fp8 权重量化：bf16 → fp8（显存减半，权重误差 <4%）。

        用于把 39GB 的 Qwen-Image-Edit transformer 压进 24GB 卡。
        失败时静默退回 bf16（打印警告），保证功能可用。
        """
        import time

        import torch

        try:
            from torchao.quantization import Float8WeightOnlyConfig, quantize_
        except Exception as e:
            print(f"[pixelforge] torchao 不可用（{e}），{tag} 保持 bf16", flush=True)
            return module
        try:
            n_lin = sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))
            t0 = time.time()
            quantize_(module, Float8WeightOnlyConfig())
            print(f"[pixelforge] {tag} fp8 权重量化完成（{n_lin} 个 Linear，"
                  f"{time.time() - t0:.0f}s，显存约减半）", flush=True)
        except Exception as e:
            print(f"[pixelforge] {tag} fp8 量化失败（{e}），保持 bf16", flush=True)
        return module

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
