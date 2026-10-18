"""图片调整管线的 AI 编辑共享辅助（Qwen-Image-Edit-2511 指令式编辑）。

三个 AI 管线——去水印 / 无缝延展 / 画风重统一——都是同一模式：
「参考图 + 中文编辑指令 → 编辑图」。本模块统一处理：

- **底座解析**：设置里配了 qwen_edit 族的 diffusers provider 就用它，
  否则退回默认路径 models/qwen-image-edit-2511；都没有就报明确错误；
- **异步执行**：provider.generate 内部已 asyncio.to_thread（GPU 阻塞不占
  事件循环），端点直接 await 即可，前端轮询/等待数分钟不会卡死其他请求；
- **seed 可复现**：同种子 + 同指令 + 同参数出同一张，方便迭代调指令；
- **尺寸控制**：Edit 模型成本随像素数涨（1024² 约 4 分钟），超大图先降
  到 max_side 再编辑、结束后 Lanczos 放大回原尺寸（AI 操作本就允许重采样，
  与像素管线的最近邻哲学不冲突——像素类请走确定性管线）。

工程约束（单卡 24GB 实测）：
- 单次 40 步约 2-4 分钟，调用方异步等，期间不要叠加其他 GPU 任务；
- provider._load 已加家族互斥换出：跑本管线前若驻留着 Z-Image，
  会自动卸载让位（日志「显存换出」），跑完再生成 txt2img 会重新加载。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from ..config import ROOT, load_settings

# 默认 Edit 底座（与 workflows.py 的 txt2imgModel 默认值同一套约定）
DEFAULT_EDIT_MODEL = "models/qwen-image-edit-2511"


def resolve_edit_model(explicit: Optional[str] = None) -> str:
    """解析 Qwen-Image-Edit 底座路径，解析不到就抛带指引的错误。"""
    if explicit and os.path.isdir(explicit):
        return explicit
    try:
        s = load_settings()
        for cfg in (s.get("providers") or {}).values():
            if not isinstance(cfg, dict):
                continue
            mid = str(cfg.get("model") or "").strip()
            if not mid or not os.path.isdir(mid):
                continue
            from ..providers.diffusers_local import detect_family

            if detect_family(mid) == "qwen_edit":
                return mid
    except Exception:
        pass
    if os.path.isdir(str(ROOT / DEFAULT_EDIT_MODEL)):
        return str(ROOT / DEFAULT_EDIT_MODEL)
    raise RuntimeError(
        "未找到 Qwen-Image-Edit 底座。请先下载 models/qwen-image-edit-2511"
        "（backend/fetch_qwen_edit.sh），或在设置里配置一个"
        " qwen_edit 族的 diffusers provider。"
    )


def _fit(img: Image.Image, max_side: int) -> Tuple[Image.Image, Tuple[int, int]]:
    """限制最长边（16 对齐），返回缩放后的图与原尺寸。"""
    w, h = img.size
    if max(w, h) <= max_side:
        return img, (w, h)
    scale = max_side / max(w, h)
    nw, nh = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
    return img.resize((nw, nh), Image.LANCZOS), (w, h)


async def run_edit(
    image: Image.Image,
    instruction: str,
    *,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    max_side: int = 1024,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """对单张图执行一次 AI 编辑，返回（结果图, meta）。

    meta 含 model/steps/seed/instruction，前端可展示、可复现。
    """
    from ..providers import get_provider
    from ..providers.diffusers_local import DiffusersProvider

    model_id = resolve_edit_model(model)
    work, orig = _fit(image.convert("RGB"), max_side)
    provider = DiffusersProvider({"model": model_id})
    res = await provider.generate(
        prompt=instruction,
        width=work.width,
        height=work.height,
        steps=steps,
        seed=seed,
        ref=work,
    )
    out = res.images[0].convert("RGB")
    if out.size != orig:
        out = out.resize(orig, Image.LANCZOS)
    meta = {
        "pipeline": "ai_edit",
        "model": model_id,
        "steps": res.meta.get("steps"),
        "seed": seed,
        "instruction": instruction,
        "origSize": list(orig),
        "editSize": list(work.size),
    }
    return out, meta


# --------------------------------------------------------------------------- #
# 管线 1：AI 去水印
# --------------------------------------------------------------------------- #

WATERMARK_PROMPT = (
    "移除画面中所有的文字水印、logo、角标和网址，按照周围画面内容自然补全"
    "被遮挡的区域，保持画面其余部分完全不变。不要添加任何新元素，不要改变"
    "主体、构图和光照。"
)


async def remove_watermark_ai(
    img: Image.Image,
    *,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    max_side: int = 1024,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """AI 去水印：指令式重绘覆盖区域，按周围内容补全。

    与反向 Alpha 混合（watermark.py）互补：那条路需要精确的水印掩码；
    本路不需要掩码，适合 AI 出图自带的、拿不到 alpha 的合成水印。
    """
    out, meta = await run_edit(
        img, WATERMARK_PROMPT, steps=steps, seed=seed, model=model, max_side=max_side)
    meta["pipeline"] = "watermark_ai"
    return out, meta


# --------------------------------------------------------------------------- #
# 管线 2：AI 左右延展（outpainting）
# --------------------------------------------------------------------------- #

def _extend_prompt(times: int) -> str:
    return (
        f"将画面内容向左右两侧自然延展，填满两侧新增的空白区域，使总宽度约为"
        f"原来的{times}倍。保持原有的透视、光照、色彩风格和画面主题完全一致，"
        f"延展内容要与原图边缘的结构自然衔接，不要重复原图内容，不要改变原图"
        f"已有区域，不要添加新的主体元素。"
    )


async def extend_horizontal_ai(
    img: Image.Image,
    times: int = 2,
    blend: int = 6,
    *,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    max_side: int = 1024,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """AI 左右延展：两侧补中性色画布 → outpainting → 原图像素贴回 + 接缝交叉淡化。

    相比镜面拼接（seamless.py）：没有对称重复感，内容是「续画」出来的；
    代价是 40 步推理（数分钟）且仅适合写实/手绘/动漫等连续色调画风——
    像素风 AI 延展会抹掉硬边 medium，请走确定性镜面管线。
    """
    import numpy as np

    src = img.convert("RGB")
    w, h = src.size
    pad = max(0, int(w * (times - 1) / 2))
    if pad == 0:
        return src, {"pipeline": "seamless_ai", "times": times, "skipped": True}
    # 中性画布：取原图均值色，比纯白/纯黑更不干扰模型的色彩先验
    arr = np.asarray(src, dtype=np.float32)
    mean_rgb = tuple(int(x) for x in arr.reshape(-1, 3).mean(axis=0))
    canvas = Image.new("RGB", (w + pad * 2, h), mean_rgb)
    canvas.paste(src, (pad, 0))

    out, meta = await run_edit(
        canvas, _extend_prompt(times), steps=steps, seed=seed, model=model, max_side=max_side)

    # 接缝交叉淡化 + 原图内部贴回：
    # 指令要求不动原区域，但模型边缘未必严丝合缝，这里把接缝带（±b 像素）
    # 在新内容与原图之间做线性淡化（原图越界列 clamp 到边缘色），再把原图
    # 内部区域原样贴回——保证已有画面零损伤，接缝不硬碰。
    res = np.asarray(out, dtype=np.float32).copy()
    orig_arr = np.asarray(src, dtype=np.float32)
    _, W = res.shape[:2]
    b = max(1, blend // 2)

    def _seam_band(x0: int) -> None:
        x_lo, x_hi = max(0, x0 - b), min(W, x0 + b)
        if x_hi <= x_lo:
            return
        ramp = np.linspace(0.0, 1.0, x_hi - x_lo, dtype=np.float32)[None, :, None]
        # 原图列号 = 画布列号 - pad（原图左边缘在画布上的偏移）；
        # 越界（接缝外侧的新内容区）clamp 到原图边缘色
        xi = np.clip(np.arange(x_lo, x_hi) - pad, 0, w - 1)
        orig_cols = orig_arr[:, xi, :]
        res[:, x_lo:x_hi] = res[:, x_lo:x_hi] * (1.0 - ramp) + orig_cols * ramp

    _seam_band(pad)
    _seam_band(pad + w)
    if w > 2 * b:
        res[:, pad + b:pad + w - b] = orig_arr[:, b:w - b]
    out = Image.fromarray(np.clip(res, 0, 255).astype(np.uint8), "RGB")
    meta.update({"pipeline": "seamless_ai", "times": times, "pad": pad, "blend": blend,
                 "mean_rgb": mean_rgb})
    return out, meta


# --------------------------------------------------------------------------- #
# 管线 3：画风重统一
# --------------------------------------------------------------------------- #

def _restyle_prompt(style: Dict[str, Any]) -> str:
    return (
        f"将这张图片整体转换为{style['name']}（{style['en']}）风格："
        f"{style['prompt']}。保持主体内容、构图、角色外观和姿态完全不变，"
        f"只改变绘画风格、线条和上色方式，不要添加或移除画面元素。"
    )


async def restyle(
    img: Image.Image,
    style_id: str,
    *,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    max_side: int = 1024,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """画风重统一：把已有资产/旧帧重绘到选定画风。

    像素类画风在 AI 重绘后会自动接 pixelize 归一化（画风库自带的
    suggestPost 参数）——AI 直接出不了硬边像素，必须确定性管线收尾。
    """
    from ..styles import get_style
    from . import pixelize as pixelize_pipe

    style = get_style(style_id)
    if style is None:
        raise ValueError(f"未知画风: {style_id}")
    out, meta = await run_edit(
        img, _restyle_prompt(style), steps=steps, seed=seed, model=model, max_side=max_side)
    px = style.get("pixelize")
    if px:
        out, px_meta = pixelize_pipe.pixelize(
            out, colors=px["colors"], pixel_size=px["block"], dither=False)
        meta["pixelize"] = px_meta
    meta.update({"pipeline": "restyle", "style": style_id, "styleName": style["name"]})
    return out, meta
