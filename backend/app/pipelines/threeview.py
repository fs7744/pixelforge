"""角色三视图：正/侧/背三张独立生成 → 确定性拼横向板。

为什么不再「一张图里画三个视角」：
单图提示词让模型一次画齐正/侧/背，在 turbo 蒸馏模型上基本不可控——
三个角色经常糊成一团（重叠）、数量不对、视角缺失，裁切也没法裁
（ workflows 里为此放弃过从三视图裁参考图的方案）。这里改成：

  帧 1 正视图  → txt2img 底座自由创作（Z-Image-Turbo，中文直喂）
  帧 2 侧视图  → Qwen-Image-Edit 指令接力：「转成侧面视图，外观不变」
  帧 3 背视图  → 同上，「转成背面视图」
  拼板        → 抠掉各自底色 → 统一暗绿底 → 基线/水平居中对齐

「三张、不重叠、同一角色」由结构保证：每张图只有一个角色的约束 +
编辑底座的外观保持指令 + 拼板是确定性像素操作，不靠模型自觉。
没有编辑底座时退回「同种子 txt2img × 3」：一致性弱一些，但仍满足
三张独立不重叠（总比糊成一团强）。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from . import matte as matte_pipe

# 统一底板色：暗绿 #1B4332（与模板/工作流提示词里的「暗绿色纯色背景」一致）
DEFAULT_BG = (27, 67, 50)

# 视角元数据：名称 + txt2img 姿势词（兜底路径用）
_VIEW_META: Dict[str, Dict[str, str]] = {
    "front": {
        "cn": "正视图", "en": "front view",
        "pose_cn": "角色面向观众，全身站立，双脚踩地，身体居中",
        "pose_en": "character facing the viewer, full body standing, feet on the ground, centered",
    },
    "side": {
        "cn": "侧视图", "en": "side view",
        "pose_cn": "角色面向画面左侧，展现完整侧面轮廓，全身站立，身体居中",
        "pose_en": "character in profile facing left, full side silhouette, full body standing, centered",
    },
    "back": {
        "cn": "背视图", "en": "back view",
        "pose_cn": "角色背对观众，展现完整背部轮廓，全身站立，身体居中",
        "pose_en": "character seen from behind, full back silhouette, full body standing, centered",
    },
}

VIEW_ORDER = ("front", "side", "back")


def _view_prompt(subject: str, view: str, chinese: bool) -> str:
    """单视角 txt2img 提示词：核心是「画面中只有一个角色」的反重叠约束。"""
    m = _VIEW_META[view]
    if chinese:
        return (
            f"游戏角色{m['cn']}立绘。画面中只有一个角色，绝对不能出现两个或多个人物。"
            f"角色设定：{subject}。"
            f"{m['pose_cn']}。纯色暗绿色背景，无渐变、无纹理、无文字、无水印、无边框、无其他物体。"
        )
    return (
        f"game character {m['en']} full body illustration. "
        f"exactly one single character in the frame, never two or more figures. "
        f"character: {subject}. {m['pose_en']}. "
        f"plain solid dark green background, no gradient, no texture, no text, "
        f"no watermark, no frame, no other objects."
    )


def _view_instruction(view: str) -> str:
    """编辑底座的中文指令：只转视角，外观/服装/配色/画风零改动。

    编辑器（Qwen-Image-Edit-2511）吃中文指令，不随 txt2img 底座语言切换。
    背视图是最容易翻车的视角：模型常见失败是「只抹掉脸」——后脑勺上
    留着正面额头条纹、脚掌仍朝前。指令里逐条钉死这些背面特征。
    """
    if view == "side":
        return (
            "把画面中的角色转换成侧面视图：角色整体转向画面左侧，展现完整的侧面轮廓，"
            "画面中仍然只有一个角色。严格保持角色的外观、发型、服装、配色、体型比例和画风"
            "完全不变，不要增删任何装备和细节，背景保持纯色暗绿色不变，角色居中站立，"
            "不要添加任何文字和水印。"
        )
    return (
        "把画面中的角色转换成背面视图：角色整体旋转180度，背对观众站立，从正背后观察。"
        "画面中只能看到一个角色。背面特征：后脑勺是一整片纯灰色，没有条纹、没有花纹、"
        "没有五官；两只耳朵从背后看只露出外侧的灰色耳廓；红色披风的背部完整可见；"
        "手中的兵器仍然握在原来那只手里，随身体一起旋转到背面，不要换手。"
        "角色的外观、服装、配色、体型比例和画风"
        "与原来完全一致，不要增删任何装备和细节，背景保持纯色暗绿色不变，"
        "角色居中站立，不要添加任何文字和水印。"
    )


# --------------------------------------------------------------------------- #
# 拼板：抠像 → 统一底色 → 基线/居中对齐
# --------------------------------------------------------------------------- #
def _border_bg(img: Image.Image) -> Tuple[int, int, int]:
    """取画面边框上出现最多的颜色作为底色（角色碰边时也比中位数稳）。"""
    import numpy as np

    arr = np.asarray(img.convert("RGB"))
    border = np.concatenate([
        arr[0, :, :].reshape(-1, 3), arr[-1, :, :].reshape(-1, 3),
        arr[:, 0, :].reshape(-1, 3), arr[:, -1, :].reshape(-1, 3),
    ])
    q = border // 16 * 16
    colors, counts = np.unique(q, axis=0, return_counts=True)
    base = colors[counts.argmax()]
    return tuple(int(v) for v in base + 8)


def _matte_padded(img: Image.Image, pad_frac: float, tolerance: int
                  ) -> Tuple[Image.Image, dict]:
    """铺一圈底色边距后再抠图，裁回原尺寸。

    为什么必须铺边：AI 出图的角色经常踩到画布底边（脚/影子被裁切），
    u2net 显著性抠图会把「触及画面边界的肢体」当成出画部分直接截掉——
    实测背视图的白色脚掌抠完只剩黑色描边（脚部 alpha 保留仅 8.6%），
    拼板上看就是脚被抠掉了颜色。铺 12% 边距让脚不碰边再抠，裁回原尺寸，
    脚掌完整保留。
    """
    w, h = img.size
    p = max(8, int(min(w, h) * pad_frac))
    canvas = Image.new("RGB", (w + 2 * p, h + 2 * p), _border_bg(img))
    canvas.paste(img.convert("RGB"), (p, p))
    cut, meta = matte_pipe.connected_matte(
        canvas, tolerance=tolerance, feather=0, method="auto")
    cut = cut.crop((p, p, p + w, p + h))
    return _drop_bg_islands(cut), meta


def _drop_bg_islands(cut: Image.Image, tol: float = 70.0) -> Image.Image:
    """丢掉 mask 里「孤立小块且整体都是背景色」的伪影岛。

    u2net 偶发把一小片背景当成显著目标保留（实测：前视图脚边一块
    248px 的亮绿岛）——原图里那是纯净背景，拼到统一底色上却成了一块
    扎眼的亮绿斑。判定：连通域面积小（< 主体 2%，上限 3000px）且域内
    90%+ 像素接近透明区主色 → alpha 置 0。

    连在主体上的伪影（如头顶一小块绿）属于主体大连通域，不受影响——
    那块肉眼不可辨，按需求保持原样。
    """
    import numpy as np

    arr = np.asarray(cut)
    kept = arr[:, :, 3] > 8
    if not kept.any():
        return cut
    try:
        import cv2
    except Exception:
        return cut

    rgb = arr[:, :, :3].astype(np.int32)
    trans = rgb[~kept]
    if len(trans) < 64:
        return cut
    q = trans // 16 * 16
    colors, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    refs = (colors[counts.argsort()[::-1][:3]] + 8).astype(np.int32)

    bg_like = np.zeros(kept.shape, dtype=bool)
    for c in refs:
        bg_like |= (np.abs(rgb - c).sum(axis=2) < tol)
    bg_like &= kept
    if not bg_like.any():
        return cut

    n, lab = cv2.connectedComponents(kept.astype(np.uint8), connectivity=4)
    if n <= 1:
        return cut
    subject_area = int(kept.sum())
    max_area = max(48, min(int(0.02 * subject_area), 3000))
    out = arr.copy()
    for i in range(1, n):
        comp = lab == i
        area = int(comp.sum())
        if area == 0 or area > max_area:
            continue
        if (bg_like & comp).sum() < 0.9 * area:
            continue
        out[comp, 3] = 0
    return Image.fromarray(out, "RGBA")


def compose_three_view(
    views: List[Image.Image],
    *,
    cell: Optional[Tuple[int, int]] = None,
    bg: Tuple[int, int, int] = DEFAULT_BG,
    gap: int = 12,
    baseline: float = 0.93,
    tolerance: int = 40,
    pad_frac: float = 0.12,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """把正/侧/背三张单视角图拼成一张横向三视图板。

    - 每张先抠底色，消除三张图之间暗绿色差的可见接缝。抠图走
      auto 链（u2net 显著性 → grabcut → 颜色键）：实测颜色键容差 72
      会把「深色裤子/黑披风」这类与暗绿底距离近的配件当成背景咬掉
      （侧视图裤子只剩 20%），u2net 语义抠图无此问题；key 兜底容差
      收紧到 40，保住深色小配件；抠图前先铺边距，防踩边肢体被截；
    - 主体底部对齐到共同基线、水平居中——模型出图的角色高低不齐，
      平移对齐后才是可用的三视图基准；
    - 抠空（背景没识别出来）时退回原图等比贴入，绝不丢视图。
    """
    import numpy as np

    imgs = [v.convert("RGB") for v in views]
    if len(imgs) != 3:
        raise ValueError(f"三视图需要 3 张图，收到 {len(imgs)} 张")

    if cell is None:
        cw = max(i.width for i in imgs)
        ch = max(i.height for i in imgs)
    else:
        cw, ch = int(cell[0]), int(cell[1])

    mats: List[Tuple[Image.Image, Optional[Tuple[int, int, int, int]], str]] = []
    for im in imgs:
        try:
            # auto 链：u2net → grabcut → key(tolerance)。深色配件（裤子/披风）
            # 与暗绿底 RGB 距离近，纯颜色键无论松紧都会二选一，优先语义抠图；
            # 先铺边距再抠，防踩边肢体（脚/影子）被 u2net 截掉。
            cut, mmeta = _matte_padded(im, pad_frac, tolerance)
            a = np.asarray(cut)[:, :, 3]
            ys, xs = np.where(a > 8)
            if len(xs) == 0:
                raise ValueError("matte empty")
            mats.append((cut, (int(xs.min()), int(ys.min()),
                               int(xs.max()), int(ys.max())), mmeta.get("method", "key")))
        except Exception:
            # 抠空兜底：整图当主体（不裁不缩放，只做平移对齐）
            mats.append((im.convert("RGBA"), None, "raw"))

    baseline_y = int(ch * baseline)
    strip = Image.new("RGB", (cw * 3 + gap * 2, ch), bg)
    infos: List[Dict[str, Any]] = []
    for i, (rgba, box, method) in enumerate(mats):
        x0 = i * (cw + gap)
        if box is None:
            # 兜底：等比缩放置中（只缩不放，避免小图被放大发虚）
            scale = min((cw - 8) / rgba.width, (ch - 8) / rgba.height, 1.0)
            nw, nh = max(1, int(rgba.width * scale)), max(1, int(rgba.height * scale))
            small = rgba.resize((nw, nh), Image.LANCZOS)
            strip.paste(small, (x0 + (cw - nw) // 2, (ch - nh) // 2), small)
            infos.append({"view": VIEW_ORDER[i], "matte": method, "bbox": None,
                          "offset": [(cw - nw) // 2, (ch - nh) // 2], "scaled": [nw, nh]})
            continue
        bx0, by0, bx1, by1 = box
        # 目标：主体水平中心 → 格子中心；主体底部 → 共同基线
        dx = int(round(cw / 2 - (bx0 + bx1) / 2))
        dy = int(round(baseline_y - by1))
        # 钳位：任何情况下不许把主体推出画布
        dx = max(min(dx, cw - bx1 - 1), -bx0)
        dy = max(min(dy, ch - by1 - 1), -by0)
        strip.paste(rgba, (x0 + dx, dy), rgba)
        infos.append({"view": VIEW_ORDER[i], "matte": method, "bbox": list(box),
                      "offset": [dx, dy]})
    return strip, {
        "cell": [cw, ch], "gap": gap, "baseline": baseline_y,
        "bg": "#%02X%02X%02X" % tuple(int(v) for v in bg),
        "size": list(strip.size), "views": infos,
    }


# --------------------------------------------------------------------------- #
# 主管线
# --------------------------------------------------------------------------- #
async def run_three_view(
    subject: str,
    *,
    style: Optional[str] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 768,
    max_side: int = 768,
    provider_id: Optional[str] = None,
    provider_cfg: Optional[Dict[str, Any]] = None,
    txt2img_model: Optional[str] = None,
) -> Tuple[Image.Image, List[Image.Image], Dict[str, Any]]:
    """生成角色三视图。返回（拼板图, [正, 侧, 背], meta）。

    - subject：角色描述（中文优先；SD 系底座会自动翻译）；
    - style：画风 id（见 app/styles.py），只作用于帧 1，帧 2+ 靠编辑指令继承画风；
    - 帧 2+ 优先走 Qwen-Image-Edit 指令接力（外观保持最好）；
      编辑底座缺失时退回同种子 txt2img × 3（meta.editFallback 记录原因）。
    """
    from .. import zh2en
    from ..config import ROOT, load_settings
    from ..providers import get_provider
    from ..providers.diffusers_local import DiffusersProvider, detect_family
    from ..styles import apply_style
    from . import ai_edit

    t0 = time.time()
    settings = load_settings()
    pid = provider_id or settings.get("activeProvider") or "mock"
    pcfg = provider_cfg if provider_cfg is not None else \
        (settings.get("providers") or {}).get(pid) or {}
    active = get_provider(pid, pcfg)

    # 双底座接力：当前 Provider 是编辑器（不能从零出图）时，
    # 帧 1 借 txt2img 底座生成——与动画工作流 action_frames 同一套约定。
    is_editor = (
        getattr(active, "id", "") == "diffusers"
        and detect_family(active._resolve_model()) == "qwen_edit"
    )
    txt_model = txt2img_model or str(ROOT / "models" / "z-image-turbo")
    if is_editor and not os.path.isdir(txt_model):
        raise RuntimeError(
            "当前 Provider 是 Qwen-Image-Edit 编辑器（无法从零出图），"
            f"但找不到 txt2img 底座：{txt_model}。请先下载 Z-Image-Turbo，"
            "或在设置里把默认 Provider 换成生成底座。"
        )

    # 语言能力以 txt2img 底座为准（编辑器帧不吃这段提示词）。
    # 注意 supports_zh 是 property，直接对路径判定，避免为探活临时构造 provider。
    if is_editor:
        chinese = detect_family(txt_model) in ("zimage", "qwen_edit")
    else:
        chinese = bool(getattr(active, "supports_zh", False))
    subj = (subject or "").strip()
    translation = None
    if not chinese and zh2en.has_chinese(subj):
        subj, translation = zh2en.translate_prompt(subj, add_style=False)

    styled = apply_style(subj, style, chinese)

    # 画风的 SD 负向项并入 provider cfg（Z-Image/Qwen-Edit 不吃负向，无需处理）
    txt_cfg = dict(pcfg) if isinstance(pcfg, dict) else {}
    if styled.get("negative"):
        base = (txt_cfg.get("negativePrompt") or "").strip()
        txt_cfg["negativePrompt"] = ", ".join(
            x for x in (base, styled["negative"]) if x)
    if is_editor:
        txt_cfg["model"] = txt_model
        txt = DiffusersProvider(txt_cfg)
    elif txt_cfg != pcfg:
        txt = get_provider(pid, txt_cfg)
    else:
        txt = active

    cw = max(64, int(width) // 16 * 16)
    ch = max(64, int(height) // 16 * 16)
    gen_cfg = cfg if cfg is not None else 7.5

    # ---- 帧 1：正视图（txt2img） ----
    t1 = time.time()
    p_front = _view_prompt(styled["prompt"], "front", chinese)
    res = await txt.generate(prompt=p_front, width=cw, height=ch,
                             steps=steps, cfg=gen_cfg, seed=seed, n=1)
    front = res.images[0].convert("RGB")
    views = [front]
    per_view: List[Dict[str, Any]] = [{
        "view": "front", "name": _VIEW_META["front"]["cn"], "mode": "txt2img",
        "prompt": p_front, "steps": res.meta.get("steps"), "seed": seed,
        "seconds": round(time.time() - t1, 1),
    }]

    # ---- 帧 2/3：侧/背视图 ----
    edit_model: Optional[str] = None
    edit_fallback: Optional[str] = None
    try:
        edit_model = ai_edit.resolve_edit_model(None)
    except Exception as e:  # noqa: BLE001
        edit_fallback = str(e)

    if edit_model:
        for key in ("side", "back"):
            t1 = time.time()
            out, emeta = await ai_edit.run_edit(
                front, _view_instruction(key), steps=steps, seed=seed,
                model=edit_model, max_side=max_side)
            views.append(out)
            per_view.append({
                "view": key, "name": _VIEW_META[key]["cn"], "mode": "edit",
                "instruction": _view_instruction(key),
                "steps": emeta.get("steps"), "seed": seed,
                "seconds": round(time.time() - t1, 1),
            })
        mode = "txt2img→edit-relay"
    else:
        # 无编辑底座：同种子 txt2img × 3。一致性弱，但三张独立、绝不重叠。
        for key in ("side", "back"):
            t1 = time.time()
            p = _view_prompt(styled["prompt"], key, chinese)
            r = await txt.generate(prompt=p, width=cw, height=ch,
                                   steps=steps, cfg=gen_cfg, seed=seed, n=1)
            views.append(r.images[0].convert("RGB"))
            per_view.append({
                "view": key, "name": _VIEW_META[key]["cn"], "mode": "txt2img",
                "prompt": p, "steps": r.meta.get("steps"), "seed": seed,
                "seconds": round(time.time() - t1, 1),
            })
        mode = "txt2img×3"

    strip, cmeta = compose_three_view(views, cell=(cw, ch))
    meta: Dict[str, Any] = {
        "pipeline": "threeview",
        "subject": subj,
        "style": styled.get("style"),
        "mode": mode,
        "txt2imgModel": txt._resolve_model() if hasattr(txt, "_resolve_model") else pid,
        "editModel": edit_model,
        "cell": [cw, ch],
        "views": per_view,
        "compose": cmeta,
        "seed": seed,
        "promptLang": "zh" if chinese else "en",
        "seconds": round(time.time() - t0, 1),
    }
    if translation:
        meta["translation"] = translation
    if edit_fallback:
        meta["editFallback"] = edit_fallback
    if styled.get("suggestPost"):
        meta["suggestPost"] = styled["suggestPost"]
    return strip, views, meta
