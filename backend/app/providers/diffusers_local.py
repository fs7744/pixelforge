"""内置 diffusers 推理服务：本地真模型（无需第三方 UI）。

设计要点：
- 懒加载：不装 torch/diffusers 也能启动服务，只在首次生成时报错提示。
- 单例缓存：同名模型只加载一次；切换模型时释放旧管线。
- 显存自适应：<8GB 自动开启 attention slicing + vae slicing，必要时 cpu offload。
- 异步：阻塞推理放到线程池，不卡事件循环。

支持两个模型族（都是阿里系，中文原生，不支持其它族）：
- zimage：Z-Image-Turbo（S3-DiT 单流 + Qwen3-4B 中文编码器）
      * 中文提示词直喂，9 步蒸馏 · guidance_scale 固定 0.0 · 1024² 原生分辨率
- qwen_edit：Qwen-Image-Edit-2511（20B MMDiT + Qwen2.5-VL 编码器）
      * 指令式图像编辑（必须传参考图），中文指令直喂
      * 角色一致性第二底座：帧 1 用 Z-Image 生成，帧 2+ 用它改视角/改姿势
      * 40 步 · true_cfg_scale=4.0 · guidance_scale=1.0 · FP8/bf16

SD1.5 系（CLIP 只认英文、依赖翻译链路、靠 LCM/LoRA/IP-Adapter 补质量）
已整体移除：两个底座都是中文原生，那条链路是死代码。指向不支持的
模型目录时会直接报错，不再悄悄按 SD 加载。
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
from typing import Any, Dict, Optional

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
    """模型族识别：zimage / qwen_edit / unknown。

    先读 model_index.json 的 _class_name（最可靠），再看目录名兜底；
    认不出来返回 "unknown"（不再默认当 SD 处理，由调用方报错）。
    """
    if model_id in _FAMILY:
        return _FAMILY[model_id]
    family = "unknown"
    if os.path.isdir(model_id):
        idx = _read_json(os.path.join(model_id, "model_index.json"))
        cls = str(idx.get("_class_name") or "")
        if "ZImage" in cls:
            family = "zimage"
        elif "QwenImageEdit" in cls:
            family = "qwen_edit"
    if family == "unknown":
        name = os.path.basename(model_id.rstrip("/\\")).lower()
        if "z-image" in name or "zimage" in name:
            family = "zimage"
        elif "qwen-image-edit" in name or "qwen_image_edit" in name:
            family = "qwen_edit"
    _FAMILY[model_id] = family
    return family


# 单卡 24GB 的现实：Z-Image bf16 约 21GB、Qwen-Image-Edit fp8 前向峰值 20.3GB，
# 两个模型族在同进程共存必 OOM（实测 37.6GiB > 23.99GiB），必须互斥驻留。
_BIG_FAMILIES = {"zimage", "qwen_edit"}


def _evict_other_families(incoming: str) -> None:
    """加载大族模型前，把缓存里另一大族的驻留模型卸掉，给显存让位。"""
    if incoming not in _BIG_FAMILIES:
        return
    victims = [m for m in list(_CACHE)
               if (_FAMILY.get(m) or detect_family(m)) in _BIG_FAMILIES
               and (_FAMILY.get(m) or detect_family(m)) != incoming]
    if not victims:
        return
    for mid in victims:
        _CACHE.pop(mid, None)
        _CAPS.pop(mid, None)
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[pixelforge] 显存换出：卸载 {'/'.join(victims)}，为 {incoming} 让位",
          flush=True)


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
    label = "本地双底座（Z-Image / Qwen-Edit）"

    def _resolve_model(self) -> str:
        model = (self.cfg.get("model") or "").strip()
        if model:
            return model
        # 默认：Z-Image-Turbo（本地目录；首次使用请在设置里确认实际路径）
        return os.environ.get("PIXELFORGE_MODEL", "models/z-image-turbo")

    def _load(self, model_id: str):
        if model_id in _CACHE:
            return _CACHE[model_id]
        family = detect_family(model_id)
        if family == "unknown":
            raise RuntimeError(
                f"不支持的模型：{model_id}。目前只支持 Z-Image-Turbo 与 "
                "Qwen-Image-Edit-2511（两个底座都是中文原生）。"
            )
        # 单卡 24GB 只容一个大家族：切换前先把另一大族卸了，
        # 否则 Z-Image(21GB) 与 Qwen-Edit fp8(峰值20.3GB) 同进程共存必 OOM
        _evict_other_families(family)
        if family == "zimage":
            return self._load_zimage(model_id)
        return self._load_qwen_edit(model_id)

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
            )
        # low_cpu_mem_usage 保持默认 True：实测 False 会让推理慢 10 倍以上
        # （主线程直连对照实验：True 加载 20s + 推理 10s；False 推理 4-17 分钟），
        # 且 True 加载完整性无问题（同种子出图与 False 完全一致）。
        pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        caps: Dict[str, Any] = {
            "family": "zimage",
            "notes": ["Z-Image：中文提示词直喂，9 步蒸馏，guidance 固定 0"],
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
            "family": "qwen_edit",
            "notes": ["Qwen-Image-Edit：中文指令直喂；必须传参考图；fp8 + cpu offload"],
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

    def _run_zimage(self, model_id: str, prompt: str, width: int, height: int,
                    steps: Optional[int], seed: Optional[int], generator: Any, n: int):
        """Z-Image 文生图：8 次 DiT 前向、CFG 固定 0、1024² 原生分辨率。

        注意：
        - 调度器把 step 0 当热身，num_inference_steps=9 才对应 8 次前向；
        - Turbo 蒸馏已内化 CFG，guidance_scale 给正值反而掉画质；
        - 参考图（ref）被忽略：Z-Image 没有 img2img 管线，角色一致性
          走 Qwen-Image-Edit 指令接力（先出帧 1，再用指令改视角/姿势）。
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
             ref: Optional[Any] = None):
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

        # caps 是 _load 里写入的，先读会拿到空字典，所以先加载一次
        self._load(model_id)
        family = _CAPS.get(model_id, {}).get("family") or detect_family(model_id)
        if family == "zimage":
            # Z-Image 蒸馏模型不吃负向、guidance 固定 0、无 img2img 管线，
            # 角色一致性交给 Qwen-Edit 指令接力（见 threeview / 动画工作流）
            return self._run_zimage(model_id, prompt, width, height, steps, seed, generator, n)
        if family == "qwen_edit":
            return self._run_qwen_edit(model_id, prompt, negative, width, height,
                                       steps, seed, generator, n, ref)
        raise RuntimeError(
            f"不支持的模型族：{model_id}。只支持 Z-Image-Turbo 与 Qwen-Image-Edit-2511。"
        )

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
        family = detect_family(model_id)
        if steps is None:
            # 蒸馏模型只需极少步数：Z-Image 9 步（8 次前向）、
            # Qwen-Image-Edit 40 步（非蒸馏）；给 28 步纯属浪费
            steps = 9 if family == "zimage" else 40
        async with _LOCK:
            images, meta = await asyncio.to_thread(
                self._run, model_id, prompt, self.cfg.get("negativePrompt", ""),
                width, height, steps, cfg, seed, n, ref,
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
        family = detect_family(model)
        zh_tag = "（中文直喂）" if family in ("zimage", "qwen_edit") else "（不支持）"
        return {
            "ok": True,
            "message": f"{st['device']} · {st['vramGB']}GB 显存 · 模型 {model} · 族 {family}"
                       + zh_tag
                       + ("" if model not in _CACHE else " · 已加载"),
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
