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
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseProvider, ProviderResult
from ..config import resolve_path
from .. import progress as progress_mod

_CACHE: Dict[str, Any] = {}
_CAPS: Dict[str, Dict[str, Any]] = {}
_FAMILY: Dict[str, str] = {}
_LOCK = asyncio.Lock()

_FP8_LINEAR_CLS: Any = None

# fp8 缓存的布局版本号。改了 state_dict 的键布局（fp8 Parameter → fp8 buffer
# w_fp8 + 空 weight）就必须换号，否则旧缓存会被误判成命中并在加载时 KeyError。
_FP8_CACHE_FORMAT = "fp8-e4m3-rowwise-buf/v2"


def _get_fp8_linear_cls() -> Any:
    """懒加载 _FP8Linear 类（本模块遵循「无 torch 也能 import」的设计）。

    fp8(E4M3) 权重量化 Linear：权重存 fp8 + 逐输出通道 scale，前向时反量化
    成 bf16 再算。与 torchao Float8WeightOnlyConfig 数学等价，但权重是普通
    fp8 张量（safetensors 0.4.2+ 原生支持、bit 级往返一致），因此可落盘缓存；
    量化在 GPU 上逐层完成（峰值显存=单层），20B 模型约 10s（CPU 全量 58s）。

    为什么 fp8 权重不能直接放 weight Parameter：

    diffusers 到处用「取第一个 Parameter 的 dtype」来推断该子模块的计算精度——
    Wan 里就有两处硬依赖：
      - transformer_wan.py: time_embedder_dtype = next(iter(parameters())).dtype
      - pipeline_wan_i2v.py: transformer_dtype = self.transformer.dtype
    如果 weight 本身就是 fp8 Parameter，这两个地方会把时间步张量也转成 fp8；
    而 float8_e4m3fn 的上限只有 448，时间步是 0-1000，一转就溢出成 inf/NaN，
    正向时直接报 "mul_cuda not implemented for Float8_e4m3fn"（真实踩过）。
    另有一层：自定义 Tensor 子类无法包成 Parameter（detach() 返回基类即报错）。

    所以布局是：真数据放 buffer ``w_fp8``，weight 置成一个 0 字节的 bf16
    Parameter 专供上面那些 dtype 探针读取。代价是 0（空张量不占内存）。
    """
    global _FP8_LINEAR_CLS
    if _FP8_LINEAR_CLS is None:
        import torch
        import torch.nn.functional as F

        class _FP8Linear(torch.nn.Linear):
            """权重存 fp8 buffer（w_fp8）+ 一个 bf16 空 Parameter 当 dtype 探针。"""

            def forward(self, x):  # type: ignore[override]
                w_fp8 = getattr(self, "w_fp8", None)
                if w_fp8 is None:                    # 非量化路径（理论上不会到）
                    w = self.weight.to(x.dtype)
                    return F.linear(x, w, self.bias)
                # 计算 dtype 只认 bf16/fp16：fp32 会让 F.linear 在 mixed dtype
                # 上报错，而 x 的 dtype 是上游（可能 fp32 的时间步）决定的。
                dt = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
                if x.dtype != dt:
                    x = x.to(dt)
                # w_scale 与 bias 都是量化/构建时留在 CPU 的常驻 buffer，而权重
                # 会被 cpu offload 搬到 GPU，直接相乘/相加会撞 device mismatch。
                w = w_fp8.to(device=x.device, dtype=dt)
                scale = getattr(self, "w_scale", None)
                if scale is not None:
                    w = w * scale.to(device=w.device, dtype=dt)[:, None]
                b = self.bias
                if b is not None:
                    b = b.to(device=w.device, dtype=dt)
                return F.linear(x, w, b)

        _FP8_LINEAR_CLS = _FP8Linear
    return _FP8_LINEAR_CLS


def _wrap_fp8_weight(linear: "torch.nn.Linear") -> None:
    """把 Linear 的裸 fp8 weight 转成 fp8 buffer + bf16 空 Parameter。

    调用前提：weight 已是 float8_e4m3fn。转换后 state_dict 里会出现
    ``w_fp8``（真数据）与空的 ``weight``（仅供 dtype 探针），缓存读写两侧
    都按这个约定处理。
    """
    import torch

    w = linear.weight
    if isinstance(w, torch.nn.Parameter):
        w = w.data
    if not isinstance(w, torch.Tensor) or w.dtype != torch.float8_e4m3fn:
        return
    linear.register_buffer("w_fp8", w)
    # 空 weight（0 字节）只用于让 next(iter(parameters())).dtype 返回 bf16
    linear.weight = torch.nn.Parameter(
        torch.empty(0, dtype=torch.bfloat16, device=w.device), requires_grad=False)


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def detect_family(model_id: str) -> str:
    """模型族识别：zimage / qwen_edit / qwen_layered / wan / unknown。

    先读 model_index.json 的 _class_name（最可靠），再看目录名兜底；
    认不出来返回 "unknown"（不再默认当 SD 处理，由调用方报错）。

    Layered 必须排在 qwen_edit 之前判：两者都是 Qwen-Image 系、transformer
    组件类名也相同（QwenImageTransformer2DModel），只有管线类名不同。
    """
    if model_id in _FAMILY:
        return _FAMILY[model_id]
    family = "unknown"
    if os.path.isdir(model_id):
        idx = _read_json(os.path.join(model_id, "model_index.json"))
        cls = str(idx.get("_class_name") or "")
        if "ZImage" in cls:
            family = "zimage"
        elif "QwenImageLayered" in cls:
            family = "qwen_layered"
        elif "QwenImageEdit" in cls:
            family = "qwen_edit"
        elif "Wan" in cls:
            family = "wan"
    if family == "unknown":
        name = os.path.basename(model_id.rstrip("/\\")).lower()
        if "z-image" in name or "zimage" in name:
            family = "zimage"
        elif "qwen-image-layered" in name or "qwen_image_layered" in name:
            family = "qwen_layered"
        elif "qwen-image-edit" in name or "qwen_image_edit" in name:
            family = "qwen_edit"
        elif "wan" in name:
            family = "wan"
    _FAMILY[model_id] = family
    return family


# 单卡 24GB 的现实：Z-Image bf16 约 21GB、Qwen-Image-Edit fp8 前向峰值 20.3GB、
# Wan2.2 TI2V-5B（transformer fp8 + umT5 fp8 + VAE）常驻约 17GB，
# 任两个模型族在同进程共存都可能 OOM 或被 WDDM 挤到共享内存，必须互斥驻留。
_BIG_FAMILIES = {"zimage", "qwen_edit", "qwen_layered", "wan"}


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


MAX_REFS = 4


def _normalize_refs(ref=None, refs=None) -> List[Any]:
    """把 ref（单张）与 refs（多张）合并成统一的参考图列表。

    上限 4 张：Qwen-Image-Edit-2511 官方支持多图输入，但再多既拉长编码时间，
    也很难让模型分清主次。顺序有意义——第一张决定输出宽高。
    """
    out: List[Any] = []
    if ref is not None:
        out.append(ref)
    if refs:
        out.extend([r for r in refs if r is not None])
    return out[:MAX_REFS]


def _frames_to_pil(frames: Any) -> List[Any]:
    """把管线的输出统一成一维 PIL 列表（RGBA）。

    Wan I2V 的输出结构随 output_type 变：output_type="pil" 是 [batch][frame] 的
    PIL 列表，="np" 是 numpy 数组 [batch][frame][h][w][c]。这里两种都收，
    免得为了换输出类型去改调用点——类型差异不是业务逻辑该操心的事。

    顺带解决了一个真实踩坑：把 [0]（取 batch 0）后的结果当成 PIL 用，
    在默认 np 模式下会报 'numpy.ndarray' object has no attribute 'convert'。
    """
    import numpy as np
    from PIL import Image

    seq = frames[0] if (hasattr(frames, "__len__") and len(frames)
                        and isinstance(frames[0], (list, tuple))) else frames
    # numpy 输出带 batch 维（[b][f][h][w][c]）。这里把所有前置 batch 维压掉：
    # 判据是「单个元素的维度 ≥ 4」——只有帧数据才有 5 维，灰度图只有 2 维、
    # 单帧 HWC 只有 3 维，不会被误当 batch。逐元素 isinstance 判断既啰嗦又易错。
    if not hasattr(seq, "ndim") and len(seq) and hasattr(seq[0], "ndim"):
        if seq[0].ndim >= 4:          # [b][f][h][w][c]
            seq = [seq[b] for b in range(len(seq))]
    out: List[Any] = []
    for f in seq:
        if isinstance(f, Image.Image):
            out.append(f.convert("RGBA"))
        else:
            arr = np.asarray(f)
            if arr.ndim >= 4:          # 残留 batch 维（单个视频对象）
                arr = arr[0]
            # 只在「最后一维明显不是通道」时才转置。必须同时满足首末维都是
            # 小通道数（3/4）才算 CHW——只用 shape[0] 判会把 HWC 的小尺寸图
            # （比如 8×10×3）误转置成 (3,8,10) 再喂给 fromarray，直接抛 TypeError。
            if (arr.ndim == 3 and arr.shape[-1] not in (3, 4)
                    and arr.shape[0] in (3, 4)):
                arr = np.transpose(arr, (1, 2, 0))   # CHW → HWC
            if arr.ndim == 2:
                # 灰度图：必须真的复制成 3 通道。用 arr[:, :, None] 补一维会得到
                # (h, w, 1)，PIL 的 fromarray 不认（KeyError (1,1,1)）。
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            out.append(Image.fromarray(arr.astype("uint8")).convert("RGBA"))
    return out


# torch 不在 backend/requirements.txt 里：它是 2.9GB 的推理依赖，只有真要出图才需要，
# 塞进基础依赖会让「只想用抠图/像素化」的人白等十几分钟。
# 代价是没装 torch 时点生成会抛 ModuleNotFoundError——这里统一换成可执行的指引。
_TORCH_HINT = (
    "未安装 PyTorch，AI 生成不可用（抠图 / 像素化 / 延展等确定性管线不受影响）。\n"
    "安装推理依赖（约 2.9GB）：\n"
    "  Windows:            start.bat --gpu\n"
    "  macOS / Linux:      ./start.sh --gpu\n"
    "或手动安装（NVIDIA 显卡，CUDA 12.4）：\n"
    "  pip install torch torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124\n"
    "  pip install diffusers transformers accelerate safetensors huggingface_hub peft torchao==0.16.0"
)


def _require_torch():
    """取 torch；缺失时抛带安装指引的 RuntimeError，而不是裸的 ModuleNotFoundError。"""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(_TORCH_HINT) from e
    return torch


def _torch_status() -> Dict[str, Any]:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        return {"installed": False, "error": str(e)}

    import torch

    free_gb = 0.0
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            free_gb = round(free / 1024**3, 1)
        except Exception:
            pass
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vramGB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        if torch.cuda.is_available() else 0,
        "freeGB": free_gb,
    }


# 三个大族在 24GB 卡上的实测/估算占用（bf16 / fp8+offload）：
# Z-Image 19.3GB、Qwen-Edit fp8 前向峰值 20.3GB、Wan2.2 TI2V-5B（transformer
# fp8 9.3GB + umT5 fp8 5.3GB + VAE 2.6GB ≈ 17GB 常驻，视频前向另有激活）。
# 留 1GB 余量给 CUDA context。
_NEED_GB = {"zimage": 20.5, "qwen_edit": 21.0, "qwen_layered": 22.0, "wan": 19.0}


def _vram_note(family: str) -> Optional[str]:
    """可用显存不够时返回一句人话告警，够了返回 None。

    为什么值得单独查一次：Windows(WDDM) 显存打满**不会 OOM**，而是把超出部分
    放到系统内存（共享显存），DiT 每一步都从 PCIe 拉权重。实测同一张 512²：
    桌面占 1GB 时 2.1s（0.1s/步），桌面占 6GB 时 203s（20s/步）——差 100 倍，
    而报错信息一个字都没有，用户只会以为「模型很慢」。
    这里不拒绝请求（慢总比不能跑好），只把原因和处置写进日志与 meta。
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
        need = _NEED_GB.get(family, 20.5)
        if free_gb >= need:
            return None
        return (
            f"可用显存只剩 {free_gb:.1f}GB（{family} 需要约 {need:.0f}GB，"
            f"总共 {total / 1024**3:.0f}GB）。超出部分会被 Windows 放进共享内存，"
            "推理会慢 50-100 倍（实测 2s → 200s+）。"
            "请关闭占显卡的程序（浏览器硬件加速、游戏、视频播放、其他 AI 工具）后重试。"
        )
    except Exception:  # pragma: no cover
        return None


def _resolve_layered_model(explicit: Optional[str] = None) -> str:
    """解析 Qwen-Image-Layered 底座路径，解析不到就抛带下载指引的错误。

    顺序：显式参数 → 环境变量 → 设置里配的 qwen_layered provider →
    models/qwen-image-layered。分层底座是「可选能力」，不配也能用其它功能，
    所以这里不静默回退到 Z-Image（那会拆出一堆不相干的东西）。
    """
    if explicit:
        hit = resolve_path(explicit)
        if os.path.isdir(hit):
            return hit
    env = (os.environ.get("PIXELFORGE_LAYERED_MODEL") or "").strip()
    if env:
        hit = resolve_path(env)
        if os.path.isdir(hit):
            return hit
    try:
        from ..config import load_settings

        for cfg in (load_settings().get("providers") or {}).values():
            if not isinstance(cfg, dict):
                continue
            mid = resolve_path(str(cfg.get("model") or ""))
            if mid and os.path.isdir(mid) and detect_family(mid) == "qwen_layered":
                return mid
    except Exception:
        pass
    default = resolve_path("models/qwen-image-layered")
    if os.path.isdir(default):
        return default
    raise RuntimeError(
        "未找到 Qwen-Image-Layered 底座（约 57.7GB）。请先下载：\n"
        "  PYTHONPATH=. python backend/fetch_qwen_layered.py\n"
        "或在设置里配置一个 qwen_layered 族的本地模型路径。"
    )


class DiffusersProvider(BaseProvider):
    id = "diffusers"
    label = "本地双底座（Z-Image / Qwen-Edit）"

    def _resolve_model(self) -> str:
        model = (self.cfg.get("model") or "").strip()
        if model:
            # 相对路径按项目根解析：从别的目录启动服务也能找到 models/ 下的底座
            return resolve_path(model)
        # 默认：Z-Image-Turbo（本地目录；首次使用请在设置里确认实际路径）
        return resolve_path(os.environ.get("PIXELFORGE_MODEL", "models/z-image-turbo"))

    def _load(self, model_id: str):
        if model_id in _CACHE:
            return _CACHE[model_id]
        family = detect_family(model_id)
        if family == "unknown":
            raise RuntimeError(
                f"不支持的模型：{model_id}。目前只支持 Z-Image-Turbo / "
                "Qwen-Image-Edit-2511 / Wan2.2 TI2V-5B（都是中文原生）。"
            )
        # 单卡 24GB 只容一个大家族：切换前先把另一大族卸了，
        # 否则 Z-Image(21GB) 与 Qwen-Edit fp8(峰值20.3GB) 同进程共存必 OOM
        _evict_other_families(family)
        if family == "zimage":
            return self._load_zimage(model_id)
        if family == "qwen_layered":
            return self._load_qwen_layered(model_id)
        if family == "wan":
            return self._load_wan(model_id)
        return self._load_qwen_edit(model_id)

    def _load_zimage(self, model_id: str):
        """Z-Image-Turbo：bf16 单流 S3-DiT + Qwen3-4B 中文编码器，8 步蒸馏。"""
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(_TORCH_HINT) from e
        try:
            from diffusers import ZImagePipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Z-Image 需要较新版本的 diffusers（含 ZImagePipeline）。请执行：\n"
                "  ./start.sh --gpu      （Windows：start.bat --gpu）\n"
                "或  pip install -U diffusers\n"
                f"原始错误：{e}"
            ) from e
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
        except Exception as e:  # pragma: no cover
            raise RuntimeError(_TORCH_HINT) from e
        try:
            from diffusers import QwenImageEditPlusPipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Qwen-Image-Edit 需要较新版本的 diffusers（含 QwenImageEditPlusPipeline）。"
                "请执行：\n"
                "  ./start.sh --gpu      （Windows：start.bat --gpu）\n"
                "或  pip install -U diffusers\n"
                f"原始错误：{e}"
            ) from e
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

    # ---------- Qwen-Image-Layered：单图语义分层 ----------

    def _load_qwen_layered(self, model_id: str):
        """Qwen-Image-Layered：把一张场景图拆成语义图层（每层一张 RGBA）。

        与 Qwen-Edit 的关系：同为 Qwen-Image 系，transformer 组件类名也一样
        （QwenImageTransformer2DModel），区别只在管线——它输出的是 layers 张
        图而不是一张图。所以显存策略、fp8 量化、落盘缓存三条全部照搬
        _load_qwen_edit：transformer 40.9GB bf16 → fp8 约 20.5GB + cpu offload。

        为什么值得引入：视差工作流原先把同一场景**生成四遍**，四次独立推理
        的透视和轮廓对不上，只能靠纯色键硬抠。语义分层是拆同一张图，天然
        坐标对齐。
        """
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(_TORCH_HINT) from e
        try:
            from diffusers import QwenImageLayeredPipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Qwen-Image-Layered 需要较新版本的 diffusers（含 QwenImageLayeredPipeline）。"
                "请执行：\n"
                "  ./start.sh --gpu      （Windows：start.bat --gpu）\n"
                "或  pip install -U diffusers\n"
                f"原始错误：{e}"
            ) from e
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen-Image-Layered 是 20B DiT，需要 CUDA GPU（如 RTX 4090）；CPU 推理不可行。"
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
            pipe = QwenImageLayeredPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, transformer=transformer)
        else:
            pipe = QwenImageLayeredPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16)
            if use_fp8:
                self._quantize_fp8(pipe.transformer, "qwen_layered.transformer")
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        _CAPS[model_id] = {
            "family": "qwen_layered",
            "notes": ["Qwen-Image-Layered：单图拆语义层，输出 N 张 RGBA；fp8 + cpu offload"],
        }
        _CACHE[model_id] = pipe
        return pipe

    def _run_layers(self, model_id: str, image: Any, layers: int, steps: Optional[int],
                    resolution: int, seed: Optional[int], generator: Any,
                    task_id: Optional[str] = None):
        """单次推理拆层。返回（[RGBA...], meta）。

        输出形状容易踩坑：管线返回的 ``out.images`` 是**按 batch 分组的二维
        列表**（每个 batch 一项、每项是 layers 张图），不是一维图列表。取
        ``out.images[0]`` 才是「这张图的 N 层」。
        """
        import torch

        pipe = self._load(model_id)
        use_steps = 50 if not steps else max(10, min(int(steps), 60))
        n_layers = max(2, min(int(layers), 8))
        # 必须是 RGBA：该管线的 VAE 按 4 通道（RGB + alpha）编码输入，给 RGB
        # 会直接崩在 encode 的 Conv3d 上（weight [96,4,3,3,3] vs input 3 通道）。
        # 不透明场景 alpha 全 255，等价于「这一层是原图内容」。
        src = image.convert("RGBA")

        # 自动图像描述必须在 CPU 上跑，不能交给管线内部做：
        # get_image_caption() 把输入张量 .to(device)，而 enable_model_cpu_offload
        # 下 text_encoder 的权重此刻还在 CPU —— 实测直接崩在 Qwen2.5-VL 的
        # patch_embed Conv3d：「Input type (CUDABFloat16Type) and weight type
        # (CPUBFloat16Type) should be the same」。这里先自己拿到描述再传 prompt，
        # 管线就不会再走那条内部路径。
        progress_mod.report(task_id, 0, use_steps, "生成图像描述…")
        caption = ""
        try:
            caption = pipe.get_image_caption(src.convert("RGB"), use_en_prompt=True, device="cpu")
        except Exception as e:
            print(f"[pixelforge] 自动图像描述失败（{type(e).__name__}: {e}），用通用描述兜底",
                  flush=True)
            caption = "A detailed game scene with distinct background, midground and foreground elements."

        kwargs: Dict[str, Any] = dict(
            image=src,
            prompt=caption,
            layers=n_layers,
            num_inference_steps=use_steps,
            true_cfg_scale=4.0,
            # 负向词必须给（哪怕一个空格）：true_cfg_scale=4 的 CFG 只在有
            # negative_prompt 时生效，否则管线只打一句 warning 就退回单分支，
            # 出图质量与官方参数不是一回事。
            negative_prompt=" ",
            resolution=int(resolution),
            # 官方 README 的推荐组合：cfg_normalize=True 稳定对比强度。
            # use_en_prompt 在这里不再生效（prompt 已经给定了）。
            cfg_normalize=True,
            generator=generator,
            callback_on_step_end=progress_mod.make_step_callback(
                task_id, use_steps, label=f"拆 {n_layers} 层"),
        )
        out = pipe(**kwargs)
        first = out.images[0]
        raw = first if isinstance(first, (list, tuple)) else [first]
        images = [im.convert("RGBA") for im in raw]
        return images, {"steps": use_steps, "trueCfg": 4.0, "family": "qwen_layered",
                        "layers": len(images), "resolution": int(resolution),
                        "seed": seed, "caption": caption,
                        "model": os.path.basename(model_id.rstrip("/\\"))}

    async def generate_layers(
        self,
        image: Any,
        *,
        model: Optional[str] = None,
        layers: int = 4,
        steps: Optional[int] = None,
        resolution: int = 640,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """场景图 → 语义图层列表（不经过 generate()）。

        与 generate() 分开：输入是一张图、输出是 N 张图，语义是「分解」而不是
        「生成」，塞进同一个方法只会让调用方猜返回结构。
        """
        _require_torch()
        model_id = resolve_path(model) if model else _resolve_layered_model()
        if detect_family(model_id) != "qwen_layered":
            raise RuntimeError(
                f"当前模型不是分层底座：{model_id}。"
                "请先下载 Qwen-Image-Layered：PYTHONPATH=. python backend/fetch_qwen_layered.py"
            )
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(
                device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
        progress_mod.check_cancel(task_id)
        async with _LOCK:
            progress_mod.check_cancel(task_id)
            images, meta = await asyncio.to_thread(
                self._run_layers, model_id, image, layers, steps, resolution,
                seed, generator, task_id,
            )
        note = _vram_note("qwen_layered")
        if note:
            meta["vramWarning"] = note
        return images, meta

    # ---------- Wan2.2 TI2V-5B：图生视频（角色参考图 + 动作提示词 → 帧序列） ----------

    def _load_wan(self, model_id: str):
        """Wan2.2 TI2V-5B：5B DiT + umT5-xxl + Wan VAE，diffusers WanImageToVideoPipeline。

        为什么用视频底座做精灵：逐帧编辑（Qwen-Edit 一帧一帧改姿势）每帧都是
        独立推理，帧间没有运动连续性，快速动作会抖；视频底座一次生成整段动作，
        肢体路径天然连贯，抽帧后就是现成的循环素材。h3-character-studio 用的
        MiniMax H3 是同一思路（首帧 = 方向母版，27 帧加中间引导）。

        管线类必须用 I2V 版：仓库的 model_index.json 写的是 ``WanPipeline``（纯文生
        视频，__call__ 不接受 image），但 TI2V-5B 的 transformer 是
        ``in_channels=48``（16 噪声 + 32 图像 latent），只有 WanImageToVideoPipeline
        会把参考图编码进第一帧的 latent 通道。照 model_index 加载会直接报
        ``unexpected keyword argument 'image'``。

        显存策略（照 Qwen-Edit 的作业）：
        - transformer bf16 18.6GB → GPU 逐层 fp8（约 9.3GB），落盘缓存
          transformer-fp8/，冷启动省掉 18GB 读取；
        - umT5-xxl bf16 10.6GB → torchao fp8（约 5.3GB），只在条件编码时用，
          enable_model_cpu_offload 会让它按阶段换出；
        - VAE 2.6GB 保持 bf16（解码质量敏感，且只在首尾用一次）。
        合计常驻约 17GB，24GB 卡可行；14B I2V 在同一张卡上要 block swap，
        所以先用 5B 验证链路。
        """
        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(_TORCH_HINT) from e
        try:
            from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
        except Exception as e:
            raise RuntimeError(
                "Wan2.2 视频生成需要 diffusers ≥ 0.36（含 "
                "WanImageToVideoPipeline / WanTransformer3DModel）。"
                "请执行 pip install -U diffusers。\n"
                f"原始错误：{e}"
            ) from e
        # ftfy 是 encode_prompt → prompt_clean 的硬依赖，但 diffusers 不默认带它。
        # 缺了会在真正开始去噪时才炸（NameError 埋在 diffusers 内部，位置很难找），
        # 所以在这里拦一次并说清楚装什么。
        try:
            import ftfy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "缺 ftfy：Wan 的提示词编码器会调用 ftfy.fix_text 清洗文本。\n"
                "请执行：pip install ftfy（或重跑 bash backend/install_gpu_deps.sh）"
            ) from e
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.2 是视频 DiT，需要 CUDA GPU；CPU 推理不可行。")

        transformer = None
        if os.environ.get("PIXELFORGE_QWEN_FP8", "1") != "0":
            try:
                transformer = self._load_component_fp8(
                    model_id, "transformer", WanTransformer3DModel, "wan.transformer")
            except Exception as e:
                print(f"[pixelforge] fp8 transformer 快速路径失败（{e}），退回完整加载",
                      flush=True)
                transformer = None

        # WanImageToVideoPipeline 与 WanPipeline 的组件字段一致，可以直接复用
        # 仓库里的 config/权重；不存在的 transformer_2/transformer_3 传 None。
        if transformer is not None:
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, transformer=transformer)
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16)
            if os.environ.get("PIXELFORGE_QWEN_FP8", "1") != "0":
                self._quantize_fp8(pipe.transformer, "wan.transformer")
        # umT5-xxl 10.6GB：fp8 后约 5.3GB，offload 会在编码完成后把它搬出显存
        if os.environ.get("PIXELFORGE_QWEN_FP8", "1") != "0":
            self._quantize_fp8(pipe.text_encoder, "wan.text_encoder")
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        caps: Dict[str, Any] = {
            "family": "wan",
            "notes": ["Wan2.2 TI2V-5B：参考图 + 中文提示词 → 视频帧序列；"
                      "fp8 + cpu offload"],
        }
        _CAPS[model_id] = caps
        _CACHE[model_id] = pipe
        return pipe

    def _run_wan(self, model_id: str, prompt: str, negative: str,
                 width: int, height: int, steps: Optional[int],
                 seed: Optional[int], generator: Any,
                 ref: Optional[Any], num_frames: int, fps: int, max_side: int,
                 task_id: Optional[str] = None):
        """图生视频：参考图 + 动作提示词 → 帧序列（PIL 列表）。

        - num_frames 必须是 4k+1（Wan 时序 VAE 的要求），内部向下取整；
        - 输入图按 16 对齐缩放到目标尺寸，长边超过 max_side 时先降（编辑成本
          随像素数涨，视频尤其明显——帧数 × 分辨率双重放大）；
        - guidance_scale=5.0 / steps=30 是 TI2V-5B 的官方推荐量级；步数可调，
          但不低于 20（再低动作会跳变）。
        """
        import torch

        if ref is None:
            raise ValueError("Wan 图生视频需要一张参考图（ref）")
        pipe = self._load(model_id)
        use_steps = 30 if not steps else max(20, min(int(steps), 50))
        from PIL import Image as _Image

        src = ref.convert("RGB")
        w = int(width or src.width) // 16 * 16
        h = int(height or src.height) // 16 * 16
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            w, h = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
        src = src.resize((w, h), _Image.LANCZOS)
        # 4k+1：Wan 时序 VAE 把帧数压成 (n-1)/4+1 个潜帧，非 4k+1 会补齐或报错
        frames = max(5, (int(num_frames) - 1) // 4 * 4 + 1)
        out = pipe(
            image=src,
            prompt=prompt,
            negative_prompt=negative or " ",
            height=h,
            width=w,
            num_frames=frames,
            num_inference_steps=use_steps,
            guidance_scale=5.0,
            generator=generator,
            # Wan 的 I2V 管线默认 output_type="np"（返回 ndarray），不是 "pil"。
            # 显式要 PIL：下游统一按 Image 处理，少一层 numpy 往返也少一处错位风险。
            output_type="pil",
            # 步进回调：30 步的视频生成是全场最长的等待，进度与取消都靠它
            callback_on_step_end=progress_mod.make_step_callback(task_id, use_steps),
        )
        imgs = _frames_to_pil(out.frames)
        return imgs, {
            "steps": use_steps, "cfg": 5.0, "family": "wan",
            "mode": "i2v", "frames": len(imgs), "fps": fps,
            "seconds": round(len(imgs) / max(1, int(fps)), 2),
            "size": [w, h],
        }

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
    def _fp8_cache_valid(cache_dir, model_dir, subdir: str = "transformer") -> bool:
        """缓存有效性：三件套齐全 + 源权重文件指纹未变。

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
            # 布局变了（fp8 Parameter → fp8 buffer + 空 weight）就必须重建：
            # 源权重指纹没变，光看源文件会误判成命中，然后用旧布局加载直接 KeyError。
            if meta.get("format") != _FP8_CACHE_FORMAT:
                return False
            for name, (size, mtime) in meta.get("files", {}).items():
                f = model_dir / subdir / name
                if not f.is_file() or f.stat().st_size != size or int(f.stat().st_mtime) != mtime:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _save_fp8_cache(transformer, cache_dir, model_dir, subdir: str = "transformer") -> None:
        """把量化后的组件存成 diffusers 风格缓存目录。

        键布局与 diffusers 权重一致：<路径>.weight 为 float8_e4m3fn，
        <路径>.w_scale 为 fp32 逐输出通道 scale，其余（norm/adaLN 的 bf16 等）
        原样保存。safetensors 原生支持 fp8 且往返 bit 级一致（已实测）。
        """
        import shutil

        from safetensors.torch import save_file

        cache_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: (v.data if hasattr(v, "data") else v).detach().contiguous()
              for k, v in transformer.state_dict().items()}
        save_file(sd, str(cache_dir / "model.safetensors"))
        shutil.copy(str(model_dir / subdir / "config.json"),
                    str(cache_dir / "config.json"))
        # 源指纹：权重文件名 + 大小 + mtime（重新下载/换版本即失效，自动重建）
        files = {}
        for f in sorted((model_dir / subdir).glob("*.safetensors")):
            st = f.stat()
            files[f.name] = [st.st_size, int(st.st_mtime)]
        (cache_dir / "fp8_cache_meta.json").write_text(json.dumps(
            {"source": str(model_dir), "format": _FP8_CACHE_FORMAT,
             "files": files},
            ensure_ascii=False, indent=1))

    @staticmethod
    def _load_fp8_cache(cache_dir, cls=None):
        """从缓存快速搭出 fp8 组件：meta 骨架 + mmap assign + 换类。

        用 diffusers 同款低内存加载路径（accelerate init_empty_weights +
        load_state_dict(assign=True)），大模型不分配 bf16 空张量；
        safetensors load_file 返回 mmap 视图，assign 后参数直接指向映射页。
        cls 缺省取 Qwen 的 transformer 类（历史调用路径）。
        """
        import torch
        from accelerate import init_empty_weights
        from safetensors.torch import load_file

        if cls is None:
            from diffusers import QwenImageTransformer2DModel as cls

        fp8_cls = _get_fp8_linear_cls()
        config = cls.load_config(str(cache_dir / "config.json"))
        sd = load_file(str(cache_dir / "model.safetensors"))
        # 缓存里的量化 Linear 权重存两处：w_fp8（fp8 裸张量，真数据）和
        # weight（0 字节 bf16，只为运行时 dtype 探针）。骨架是纯 nn.Linear，
        # 它要满尺寸 weight，两者形状对不上——必须先把 weight 从待加载字典里
        # 拿掉，否则 load_state_dict(assign=True) 会报 size mismatch 直接失败。
        fp8_targets = {k[: -len(".w_fp8")] for k in sd if k.endswith(".w_fp8")}
        # 这些 Linear 的 weight 会被换成 0 字节 bf16，对应的 state_dict 键
        # 正好是 <模块路径>.weight —— 加载前先剔除，否则 assign 时形状不匹配。
        fp8_weight_keys = {f"{k}.weight" for k in fp8_targets}
        sd = {k: v for k, v in sd.items() if k not in fp8_weight_keys}
        try:
            with init_empty_weights():
                model = cls.from_config(config)
        except Exception:
            # meta 路径失败时退回普通 from_config（CPU 上空张量，仅多耗内存）
            model = cls.from_config(config)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # unexpected 只允许 w_scale / w_fp8（骨架没有这两个 buffer）。
        # missing 里必然有一批 .weight：那正是上面剔除掉的量化 Linear 权重，
        # 骨架是满尺寸 nn.Linear 却没有真数据（真数据在 w_fp8），后面统一换成
        # 0 字节 bf16。除这些以外不允许再有缺失——那才是缓存损坏。
        extras = [k for k in unexpected if k.endswith((".w_scale", ".w_fp8"))]
        stray = [k for k in missing if k not in fp8_weight_keys]
        if stray or len(extras) != len(unexpected):
            raise RuntimeError(
                f"缓存键不匹配（missing={len(missing)} stray={len(stray)} "
                f"unexpected={len(unexpected)}），删除 {cache_dir.name}/ 可重建")
        for key in unexpected:
            mod = model.get_submodule(key.rsplit(".", 1)[0])
            mod.register_buffer(key.rsplit(".", 1)[1], sd[key])
        n = 0
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear) and getattr(mod, "w_fp8", None) is not None:
                mod.__class__ = fp8_cls
                # weight 换成 0 字节 bf16：让 diffusers 的 dtype 探针看到 bf16
                mod.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=torch.bfloat16, device=mod.w_fp8.device),
                    requires_grad=False)
                n += 1
        if n == 0:
            raise RuntimeError("缓存中没有任何 fp8 Linear，格式可能损坏")
        return model

    def _load_component_fp8(self, model_id: str, subdir: str, cls, tag: str):
        """任意 diffusers 组件的 fp8 加载（量化 + 落盘缓存），Qwen/Wan 共用。

        与 _load_qwen_transformer_fp8 的区别只是组件目录和模型类可指定，
        逻辑完全一样：命中缓存 mmap 直读，未命中 GPU 逐层量化后写缓存。
        """
        import time
        from pathlib import Path

        import torch

        model_dir = Path(model_id)
        cache_dir = model_dir / f"{subdir}-fp8"
        if self._fp8_cache_valid(cache_dir, model_dir, subdir):
            t0 = time.time()
            t = self._load_fp8_cache(cache_dir, cls)
            print(f"[pixelforge] fp8 缓存命中：{cache_dir.name}/（{time.time() - t0:.0f}s）",
                  flush=True)
            return t

        t0 = time.time()
        t = cls.from_pretrained(str(model_dir / subdir), torch_dtype=torch.bfloat16)
        t_load = time.time() - t0
        n = self._quantize_transformer_gpu(t)
        try:
            self._save_fp8_cache(t, cache_dir, model_dir, subdir)
        except Exception as e:
            print(f"[pixelforge] {tag} fp8 缓存落盘失败（{e}），不影响本次运行", flush=True)
        print(f"[pixelforge] {tag} fp8 量化完成（{n} 个 Linear，bf16 加载 {t_load:.0f}s，"
              f"缓存已写 {cache_dir.name}/）", flush=True)
        return t

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
            # 包成 _FP8Weight：对外报 bf16，避免 diffusers 的 dtype 探针把
            # 时间步张量也转成 fp8（e4m3 上限 448，时间步 0-1000 会溢出）
            _wrap_fp8_weight(mod)
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
                    steps: Optional[int], seed: Optional[int], generator: Any, n: int,
                    task_id: Optional[str] = None):
        """Z-Image 文生图：8 次 DiT 前向、CFG 固定 0、1024² 原生分辨率。

        注意：
        - 调度器把 step 0 当热身，num_inference_steps=9 才对应 8 次前向；
        - Turbo 蒸馏已内化 CFG，guidance_scale 给正值反而掉画质；
        - 它是纯文生图，没有 img2img / 参考图输入。传了参考图会在 generate()
          里改走 Qwen-Image-Edit（autoRouted），或在这里抛错，不再静默忽略。
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
                # 步进回调：SSE 进度 + 取消检查（diffusers 每步结束调一次）
                callback_on_step_end=progress_mod.make_step_callback(
                    task_id, use_steps, label=f"第 {i + 1} 张"),
            )
            images.append(out.images[0].convert("RGBA"))
            if generator is not None and seed is not None:
                generator = torch.Generator(device=generator.device).manual_seed(int(seed) + i + 1)
        return images, {"steps": use_steps, "cfg": 0.0, "family": "zimage",
                        "mode": "txt2img", "turbo": True}

    def _run_qwen_edit(self, model_id: str, prompt: str, negative: str,
                       width: int, height: int, steps: Optional[int],
                       seed: Optional[int], generator: Any, n: int,
                       refs: Optional[Any] = None, task_id: Optional[str] = None):
        """Qwen-Image-Edit 指令式编辑：参考图（可多张）+ 中文指令 → 编辑图。

        与 txt2img 的本质区别：prompt 是「编辑指令」而非「画面描述」。
        角色一致性指令模板：
          「保持角色外观、服装、配色完全不变，将姿势改为：{action}，纯色背景」

        - 必须传参考图（没有输入时直接报错，而不是静默出图）；
        - **多图**：Qwen-Image-Edit-2511 的 Plus 管线原生支持多张输入，用于
          「角色正面 + 侧面」「角色 + 场景」这类组合参考；所有参考图统一缩放到
          目标尺寸后再送编码器；
        - 输出尺寸：显式 width/height 生效（16 对齐）；缺省时跟随第一张参考图。
          注意管线自己的默认是「按最后一张参考图的长宽比算 1024² 面积」，
          所以必须显式传参，否则用户设的宽高会被忽略；
        - true_cfg_scale=4.0 是官方推荐（正负提示词对比强度）；
          guidance_scale=1.0 关掉二次 CFG（蒸馏已内化）。
        """
        import torch

        refs = [r for r in (refs or []) if r is not None]
        if not refs:
            raise ValueError(
                "Qwen-Image-Edit 是指令式编辑器，必须传参考图（ref/refs）。"
                "请通过 /api/generate/ref 或工作流的 useRef 步骤传入。"
            )

        pipe = self._load(model_id)
        use_steps = 40 if not steps else max(20, min(int(steps), 50))
        from PIL import Image as _Image

        srcs = [r.convert("RGB") for r in refs]
        w = int(width or srcs[0].width) // 16 * 16
        h = int(height or srcs[0].height) // 16 * 16
        srcs = [s.resize((w, h), _Image.LANCZOS) for s in srcs]
        # 单图传 Image，多图传 list（Plus 管线两者都吃）
        image_arg = srcs[0] if len(srcs) == 1 else srcs
        images = []
        for i in range(max(1, min(n, 4))):
            out = pipe(
                image=image_arg,
                prompt=prompt,
                negative_prompt=negative or " ",
                height=h,
                width=w,
                num_inference_steps=use_steps,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                generator=generator,
                # 步进回调：SSE 进度 + 取消检查（40 步里每一步都报，长任务的
                # 进度条就靠它；取消时抛 CancelledError 中止推理）
                callback_on_step_end=progress_mod.make_step_callback(
                    task_id, use_steps, label=f"第 {i + 1} 张"),
            )
            images.append(out.images[0].convert("RGBA"))
            if generator is not None and seed is not None:
                generator = torch.Generator(device=generator.device).manual_seed(int(seed) + i + 1)
        return images, {"steps": use_steps, "cfg": 1.0, "trueCfg": 4.0,
                        "family": "qwen_edit",
                        "mode": "edit" if len(srcs) == 1 else "ref2img",
                        "refCount": len(srcs), "size": [w, h]}

    def _run(self, model_id: str, prompt: str, negative: str, width: int, height: int,
             steps: int, cfg: float, seed: Optional[int], n: int,
             ref: Optional[Any] = None, refs: Optional[Any] = None,
             task_id: Optional[str] = None):
        torch = _require_torch()

        # 先看显存够不够，再决定要不要让用户干等：桌面程序占着显存时，模型会被
        # Windows 挤到共享内存，同一张图从 2s 变 200s，而界面只会显示「生成中」。
        family_hint = _CAPS.get(model_id, {}).get("family") or detect_family(model_id)
        vram_note = _vram_note(family_hint)
        if vram_note:
            print(f"[pixelforge] 警告：{vram_note}", flush=True)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

        # caps 是 _load 里写入的，先读会拿到空字典，所以先加载一次
        self._load(model_id)
        family = _CAPS.get(model_id, {}).get("family") or detect_family(model_id)
        if family == "zimage":
            # 参考图在 generate() 里已经改走编辑底座；走到这里说明调用方绕过了
            # 那层判断，明确报错比静默出图更容易排查
            if ref is not None or refs:
                raise RuntimeError(
                    "Z-Image 是纯文生图底座，不支持参考图。请用 /api/generate/ref "
                    "（会自动改用 Qwen-Image-Edit）或下载编辑底座。"
                )
            images, meta = self._run_zimage(model_id, prompt, width, height,
                                            steps, seed, generator, n, task_id=task_id)
        elif family == "qwen_edit":
            images, meta = self._run_qwen_edit(model_id, prompt, negative, width, height,
                                               steps, seed, generator, n,
                                               refs=_normalize_refs(ref, refs),
                                               task_id=task_id)
        elif family == "wan":
            raise RuntimeError(
                "Wan 是视频底座，请走 generate_video()（参考图 + 提示词 → 帧序列），"
                "不要走 generate()。"
            )
        elif family == "qwen_layered":
            raise RuntimeError(
                "Qwen-Image-Layered 是分层底座，请走 generate_layers()"
                "（一张场景图 → N 张语义图层），不要走 generate()。"
            )
        else:
            raise RuntimeError(
                f"不支持的模型族：{model_id}。只支持 Z-Image-Turbo / "
                "Qwen-Image-Edit-2511 / Qwen-Image-Layered / Wan2.2 TI2V-5B。"
            )
        if vram_note:
            meta["vramWarning"] = vram_note
        return images, meta

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
        refs: Optional[Any] = None,
        task_id: Optional[str] = None,
    ) -> ProviderResult:
        # 入口先卡一次：缺 torch 时直接把安装指引抛给前端，
        # 而不是等 _run 里 import 出一句光秃秃的 "No module named 'torch'"
        _require_torch()
        requested = self._resolve_model()
        family = detect_family(requested)
        all_refs = _normalize_refs(ref, refs)
        model_id = requested
        auto_routed = False

        # 带参考图 = 图生图/参考图生成；Z-Image 没有这条管线，自动改走编辑底座，
        # 而不是把参考图悄悄丢掉（历史上就是这样静默忽略的）。
        if all_refs and family == "zimage":
            try:
                from ..pipelines.ai_edit import resolve_edit_model

                model_id = resolve_edit_model()
                family = "qwen_edit"
                auto_routed = True
            except Exception as e:
                raise RuntimeError(
                    "Z-Image 是纯文生图底座，不支持参考图。要按参考图生成，"
                    "请先下载编辑底座：bash backend/fetch_qwen_edit.sh"
                ) from e

        if steps is None:
            # 蒸馏模型只需极少步数：Z-Image 9 步（8 次前向）、
            # Qwen-Image-Edit 40 步（非蒸馏）；给 28 步纯属浪费
            steps = 9 if family == "zimage" else 40
        # 取消检查点：排队等锁之前 + 拿到锁之后。模型加载（fp8 量化可达
        # 十几分钟）和文本编码阶段无法中断——它们不在 diffusers 的步进
        # 回调里，只能等进入采样后的第一步。
        progress_mod.check_cancel(task_id)
        async with _LOCK:
            progress_mod.check_cancel(task_id)
            images, meta = await asyncio.to_thread(
                self._run, model_id, prompt, self.cfg.get("negativePrompt", ""),
                width, height, steps, cfg, seed, n, ref, all_refs or None,
                task_id,
            )
        status = _torch_status()
        out_meta = {"provider": "diffusers", "model": model_id, "device": status.get("device"),
                    **meta, "caps": _CAPS.get(model_id)}
        if auto_routed:
            out_meta["autoRouted"] = True
            out_meta["requestedModel"] = requested
        return ProviderResult(images=images, meta=out_meta)

    async def generate_video(
        self,
        prompt: str,
        ref,
        width: int = 0,
        height: int = 0,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        num_frames: int = 81,
        fps: int = 24,
        max_side: int = 832,
        task_id: Optional[str] = None,
    ) -> ProviderResult:
        """图生视频：参考图 + 提示词 → 帧序列（ProviderResult.images = 帧）。

        与 generate() 分开而不是加参数：视频的输入输出语义都不同（一张图进、
        几十帧出、按时长而非张数描述），塞进同一个方法只会让调用方猜。
        """
        _require_torch()
        model_id = self._resolve_model()
        if detect_family(model_id) != "wan":
            raise RuntimeError(
                f"当前模型不是视频底座：{model_id}。"
                "请先下载 Wan2.2 TI2V-5B：PYTHONPATH=. python backend/fetch_wan22.py"
            )
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(
                device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
        progress_mod.check_cancel(task_id)
        async with _LOCK:
            progress_mod.check_cancel(task_id)
            images, meta = await asyncio.to_thread(
                self._run_wan, model_id, prompt, self.cfg.get("negativePrompt", ""),
                width, height, steps, seed, generator, ref,
                int(num_frames), int(fps), int(max_side), task_id,
            )
        status = _torch_status()
        out_meta = {"provider": "diffusers", "model": model_id,
                    "device": status.get("device"), **meta,
                    "caps": _CAPS.get(model_id)}
        return ProviderResult(images=images, meta=out_meta)

    async def test(self) -> Dict[str, Any]:
        st = _torch_status()
        if not st["installed"]:
            return {"ok": False,
                    "message": "未安装 PyTorch，AI 生成不可用。运行 ./start.sh --gpu"
                               "（Windows：start.bat --gpu）安装推理依赖，约 2.9GB"}
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
            "message": f"{st['device']} · {st['vramGB']}GB 显存（当前可用 "
                       f"{st.get('freeGB', 0)}GB） · 模型 {model} · 族 {family}"
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
