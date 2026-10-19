"""精灵动作帧：以角色参考图为准，一帧一帧改姿势，再确定性拼成精灵表。

为什么不用「一张图排 4×4」：
- 文生图底座没有「这一格是哪个动作」的定位能力，实测 16 格的姿势大面积重复，
  用户看到的是一张「同一个姿势贴了 16 遍」的图，动作循环根本不能用；
- 排布还会受模型自由发挥影响（格子大小、分割线、缺格），拆帧时全对不上。

改成「一帧一次编辑 + 确定性拼板」后：
- 每一帧都以同一张角色参考图为条件（编辑底座），外观/配色/画风不会漂；
- 每帧的动作由用户单独填，模型只需要做「换个姿势」这一件事，可控性高得多；
- 拼板、切帧、基线对齐都是本地确定性的，不依赖模型的网格能力。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from . import backdrop as backdrop_pipe
from . import sheet as sheet_pipe
from . import sprite as sprite_pipe

# 用户没填动作时的兜底（也只用于提示，不写进生成结果）
DEFAULT_POSES = ["站立", "迈步", "抬手", "落脚"]


# 常见动作的肢体描述。编辑底座对「迈步」这种两字词几乎没反应——实测 12 步时
# 生成结果和参考图站姿一模一样，加上四肢怎么动才真的会动。
_POSE_HINTS = [
    (("迈步", "行走", "走路", "走"), "一条腿屈膝向前迈出、脚掌落地，另一条腿在后侧蹬直，双臂自然前后摆动"),
    (("站立", "待机", "站"), "双腿并拢直立，双臂自然垂在身体两侧"),
    (("抬手", "举手", "举"), "一只手臂抬起到肩膀高度，手掌张开朝向画面"),
    (("落脚", "收步", "落地"), "前脚落地踩实，后脚跟上与它并拢，重心落回两脚之间"),
    (("跳跃", "滞空", "腾空", "起跳"), "双腿屈膝，身体腾起离地，双臂向上抬起"),
    (("蹲",), "双膝弯曲下蹲，身体压低，背部保持挺直"),
    (("受击", "挨打", "受创"), "身体向后仰，一只手臂抬起来护住头部"),
    (("倒地", "倒下", "死亡"), "身体侧向倒向地面，四肢摊开"),
    (("攻击", "挥剑", "挥砍", "出拳", "挥刀"), "双手持械向前挥出，身体重心前压"),
]


def pose_hint(pose: str) -> str:
    """把「迈步」这类两字词扩成「四肢怎么动」的具体描述。"""
    p = (pose or "").strip()
    for keys, hint in _POSE_HINTS:
        if any(k in p for k in keys):
            return hint
    return ""


def pose_instruction(pose: str, index: int, total: int,
                     backdrop: str = "neutral") -> str:
    """单帧编辑指令：只讲「改成什么姿势」，不讲「保持什么不变」。

    实测结论（同一张角色图、同 seed、20 步）：
    - 指令里先写「保持外观完全不变／衣服装备不要动」，编辑底座把整条当成
      「什么都别改」，姿势纹丝不动 —— 出来 16 格全是一个样，正是用户看到的
      「生成都一样」；
    - 把这些约束全删掉、只写「把角色改成正在迈步的样子：一腿前迈屈膝、
      另一腿后蹬、手臂摆动」，姿势立刻真的变了，而且衣服/装备/配色/画风
      本来就是编辑底座默认保留的，不会丢。
    所以这里不写任何「保持不变」。全身入画那条不能省——跨步姿势最容易把脚
    裁出画面。

    底色默认走中性灰而不是绿幕：实测模型声明 #00FF00 实际画出薄荷绿
    [90,199,126] + 径向渐变，还会把绿染到角色边缘（内圈偏底色像素 76.5%，
    中性灰只有 33.3%）。理由与数据见 backdrop 模块文档。
    """
    hint = pose_hint(pose)
    detail = f"：{hint}" if hint else ""
    return (
        f"把角色改成{pose}的动作{detail}。"
        f"全身完整入画，双脚不超出画面，{backdrop_pipe.backdrop_prompt(backdrop)}。"
    )


def pose_prompt(pose: str, index: int, total: int, subject: str,
                style_words: str = "", backdrop: str = "neutral") -> str:
    """没有编辑底座时的文生图兜底提示词（一致性会弱一些，但不会没输出）。"""
    body = (
        f"游戏角色正面立绘，全身，角色设定：{(subject or '').strip()}。"
        f"动作：{pose}。这是同一套动作循环的第 {index + 1} 帧（共 {total} 帧），"
        "角色朝向正前方，双脚踩在同一条水平线上，身体水平居中。"
        f"画面中只有一个角色，{backdrop_pipe.backdrop_prompt(backdrop)}，"
        "无文字、无水印、无边框。"
    )
    return f"画风：{style_words}。{body}" if style_words else body


async def run_frame(
    ref: Optional[Image.Image],
    pose: str,
    index: int,
    total: int,
    *,
    subject: str = "",
    style: str = "",
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    max_side: int = 768,
    engines: Optional[Dict[str, Any]] = None,
    backdrop: str = "neutral",
) -> Tuple[Image.Image, Dict[str, Any]]:
    """生成一帧。有参考图走编辑底座（保外观），没有则退回文生图。

    返回的图**不做抠图**：抠图/缩放/基线对齐统一放到拼板时做，
    这样重出一帧只影响那一帧，所有帧共享同一个缩放系数（不会忽大忽小）。
    """
    from ..styles import style_words
    from .threeview import _engines

    eng = engines or _engines()
    t = time.time()
    bd = backdrop_pipe.get_backdrop(backdrop)
    inst = pose_instruction(pose, index, total, backdrop)

    if ref is not None and eng["editModel"]:
        from . import ai_edit

        out, emeta = await ai_edit.run_edit(
            ref, inst, steps=steps, seed=seed, model=eng["editModel"], max_side=max_side)
        return out, {
            "mode": "edit", "instruction": inst, "pose": pose,
            "index": index, "total": total, "steps": emeta.get("steps"),
            "seed": seed, "seconds": round(time.time() - t, 1),
            "editModel": eng["editModel"], "backdrop": bd["id"],
        }

    sw = style_words(style)
    prompt = pose_prompt(pose, index, total, subject, sw, backdrop)
    res = await eng["txt"].generate(
        prompt=prompt, width=max(64, width // 16 * 16), height=max(64, height // 16 * 16),
        steps=steps, cfg=7.5, seed=seed, n=1)
    meta: Dict[str, Any] = {
        "mode": "txt2img", "prompt": prompt, "pose": pose,
        "index": index, "total": total, "steps": res.meta.get("steps"),
        "seed": seed, "seconds": round(time.time() - t, 1),
        "subject": subject, "backdrop": bd["id"],
    }
    if ref is not None and eng["editFallback"]:
        meta["editFallback"] = eng["editFallback"]
    elif ref is None:
        meta["note"] = "没有参考图，走文生图：同一角色的帧间一致性会明显变差"
    return res.images[0].convert("RGB"), meta


def compose(
    images: List[Image.Image],
    *,
    cols: int = 4,
    size: int = 256,
    anchor: str = "bottom",
    pad: int = 2,
    matte: bool = True,
    tolerance: int = 80,
    feather: int = 0,
    hard: bool = True,
    matte_method: str = "auto",
    colors: int = 0,
) -> Tuple[Image.Image, List[Image.Image], Dict[str, Any], Dict[str, Any]]:
    """把逐帧结果规整（共享缩放）后拼成精灵表。

    shared_scale：下蹲/抬腿的帧包围盒更小，逐帧各自撑满画布会让角色在
    整段循环里忽大忽小——一播就露馅。统一按最大包围盒算缩放，角色尺寸恒定。

    colors > 0：先取一块**全局色板**再让每帧映射回它（移植自
    game-asset-video-sprites）。逐帧各自量化时每帧色板都不同，同一个肤色
    第 1 帧偏橙、第 3 帧偏红，动画一播就闪；帧数越多越明显。

    最后算一遍一致性指标（首尾衔接 / 帧间差 / 色板外颜色 / 半透明占比），
    把「稳不稳」变成数字——AI 逐帧生成最常见的就是某一帧姿势跑飞，肉眼看
    不出来，指标能直接指到第几帧。
    """
    from . import metrics as metrics_pipe
    from . import palette as palette_pipe

    if not images:
        raise ValueError("没有可合成的帧")
    frames, fmeta = sprite_pipe.normalize_frames(
        images, shared_scale=True, size=int(size), anchor=anchor, pad=int(pad),
        matte=matte, tolerance=tolerance, feather=feather,
        hard=hard, matte_method=matte_method,
    )

    pal: List[Sequence[int]] = []
    if colors and int(colors) >= 2:
        # 没抠图时绿色背景还在，取色要把键色排除掉，否则色板半版是绿
        exclude = (0, 255, 0) if not matte else None
        pal = palette_pipe.extract_palette(frames, k=int(colors), exclude=exclude)
        if pal:
            frames = [palette_pipe.apply_palette(f, pal) for f in frames]

    sheet, index = sheet_pipe.compose_sheet(frames, max(1, int(cols)))
    m = metrics_pipe.sheet_metrics(frames, pal or None)
    meta = {
        "cols": max(1, int(cols)),
        "frames": len(frames),
        "normalize": fmeta,
        "frameSize": index["frame_size"],
        "sheetSize": index["sheet_size"],
        "metrics": m,
        "metricsVerdict": metrics_pipe.metrics_verdict(m),
    }
    if pal:
        meta["palette"] = {"colors": len(pal), "swatches": [list(c) for c in pal[:32]]}
    return sheet, frames, index, meta
