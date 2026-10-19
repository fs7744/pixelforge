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

抠图边缘实测审计（u2net，512×768 真实出图，两轮）：
- 角色被咬：0.05% / 0% / 0.14%（可忽略）；细结构（武士刀）保留 99.9%；
- 第一轮审计（底色接近 #1B4332）看不到外圈绿边，据此判定"不做膨胀/收缩"；
- **第二轮真实出图推翻了一半结论**：模型这次给的底色是亮绿 #24A455，
  描边外圈全是「黑+亮绿」的抗锯齿混色（正视图边缘环 2696px 里 1183px 偏绿，
  占比 44%）。connected_matte 自带的 despill 按「与底色距离」判定，亮绿底
  与边框取样色差 197，一格都洗不掉——所以肉眼能看到一圈绿线。
- 背视图白色脚掌又被 u2net 判成背景：mask 在脚上只剩一圈黑描边，
  两片脚掌成为 1760/1744px 的**封闭孔洞**，脚白保留 0.2%。
修法（都不动 alpha 形状，所以不会啃掉胡须/触角）：
1. `_fill_enclosed_holes`：补回被描边围死的封闭孔洞（≤2% 画布），RGB 原样；
2. `_clean_edges`：外圈 2px 里「比离自身内侧颜色更接近底色」的像素，取内侧
   颜色重绘——纯黑描边离内侧为 0 永远不会被洗掉，抗锯齿混色则必然命中；
   细结构上残留的偏底色像素再从最近的干净邻居取色。
仍**不做** mask 膨胀/收缩：那会改轮廓并啃掉细结构，而这两类问题的根因
分别是 mask 孔洞和边缘颜色，与轮廓形状无关。

指令层面的坑（已修）：`_view_instruction("back")` 曾把示例角色（灰白英短
猫武士）的造型写死进指令——「后脑勺纯灰色」「灰色耳廓」「红色披风背部
完整可见」「手中兵器」。这些句子对任何角色都会生效，于是正面没有披风的
角色转背面也一定长出红披风、后脑被刷成灰。现在指令只描述「视角怎么变」
和「造型以参考图为准」，不点名任何具体服装、配件或颜色。
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


def _clean_subject(subject: str) -> str:
    """去掉调用方自己带的「角色设定：」前缀与结尾标点。

    工作流 three_view 步骤的 prompt 本来就是「角色设定：{input}…」，管线又
    会补同一个前缀，直接拼就成了「角色设定：角色设定：…」。重复前缀不只是
    浪费提示词预算：模型会把它读成「前半段是标题、后半段是补充说明」，
    后半段的画风描述容易被弱化。
    """
    s = (subject or "").strip()
    for _ in range(3):  # 防御多重前缀，不无限循环
        for prefix in ("角色设定：", "角色设定:", "角色：", "角色:"):
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip()
    return s.rstrip("。.，, ").strip()


def _view_prompt(subject: str, view: str, style_words: str = "") -> str:
    """单视角 txt2img 提示词：画风句在前，主体与「只画一个角色」约束在后。

    风格句必须占开头：Qwen3 编码器是因果结构，越靠后的 token 对整体语义
    影响越弱，而这条 prompt 后面还有一大段视角/背景约束，风格词排在里面
    等于没写（线上表现为「选了画风出图没变化」）。
    """
    m = _VIEW_META[view]
    subj = _clean_subject(subject)
    head = f"画风：{style_words}。" if style_words else ""
    return (
        f"{head}"
        f"游戏角色{m['cn']}立绘。画面中只有一个角色，绝对不能出现两个或多个人物。"
        f"角色设定：{subj}。"
        f"{m['pose_cn']}。纯色暗绿色背景，无渐变、无纹理、无文字、无水印、无边框、无其他物体。"
    )


# 「只改角度、不改造型」的通用尾巴：不点名任何具体服装/配件/颜色。
# 点名会直接变成生成指令——这一条踩过坑：背视图指令里写死「红色披风的背部
# 完整可见、后脑勺是一整片纯灰色」，于是任何角色转背面都会长出一件红披风、
# 后脑还被刷成灰色，哪怕正面根本没有披风。
_KEEP_APPEARANCE = (
    "只改变观察角度：角色的造型、配色、体型比例和画风全部以参考图为准，"
    "不新增也不删除任何部件。背景保持纯色暗绿色不变，角色居中站立，"
    "画面中仍然只有一个角色，不要添加文字和水印。"
)


def _view_instruction(view: str) -> str:
    """编辑底座的中文指令：只转视角，外观/服装/配色/画风零改动。

    编辑器（Qwen-Image-Edit-2511）吃中文指令，不随 txt2img 底座语言切换。
    背视图是最容易翻车的视角：模型常见失败是「只抹掉脸」——后脑勺上留着
    正面的五官或图案。指令里用「背面该看到什么」的正向描述钉住，而不是
    复述某个具体角色的造型（那会成为凭空添加装备的指令）。
    """
    if view == "side":
        return (
            "把画面中的角色转换成侧面视图：角色整体转向画面左侧，展现完整的侧面轮廓，"
            "全身站立。侧面的造型全部来自参考图中这个角色本身，身体的各个部位按它"
            "本来的样子推演，参考图里没有的东西不要出现。"
            + _KEEP_APPEARANCE
        )
    return (
        "把画面中的角色转换成背面视图：角色整体旋转180度，背对观众站立，从正背后观察。"
        "脸部、眼睛和鼻子完全不可见；头部的造型按参考图中这个角色的原样自然延续到背面。"
        "背面只呈现角色自己已有的造型，参考图里没有的衣物、配件和装饰一件都不要出现。"
        "角色手里原本拿着东西就随身体一起转到背面，仍然握在同一只手里；原本空着手就"
        "保持空手，不要凭空添一件道具。"
        + _KEEP_APPEARANCE
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
    bg = _border_bg(img)
    canvas = Image.new("RGB", (w + 2 * p, h + 2 * p), bg)
    canvas.paste(img.convert("RGB"), (p, p))
    cut, meta = matte_pipe.connected_matte(
        canvas, tolerance=tolerance, feather=0, method="auto")
    cut = cut.crop((p, p, p + w, p + h))
    cut = _drop_bg_islands(cut)
    cut, filled = _fill_enclosed_holes(cut)
    cut, recolored = _clean_edges(cut, bg)
    meta = dict(meta, holesRestored=filled, edgePixelsRecolored=recolored)
    return cut, meta


def _fill_enclosed_holes(cut: Image.Image, max_ratio: float = 0.02
                         ) -> Tuple[Image.Image, int]:
    """补回被角色完全围住的透明孔洞（只改 alpha，RGB 原样）。

    实测背视图两片白色脚掌被 u2net 判成背景：mask 在脚上只剩黑描边，
    脚掌内部成了 1760/1744px 的封闭孔洞，拼板上看就是"脚的白色被抠没了"。
    孔洞边界完全在画面内部（不接边）即可判定为角色内部——补回不透明后
    露出的就是原图的白脚掌，不需要任何颜色猜测。

    上限 2% 画布：更大的封闭区域（如身侧空隙）保留透明，避免糊成一块。
    """
    import numpy as np

    a = np.asarray(cut).copy()
    kept = a[:, :, 3] > 8
    holes = (~kept).astype(np.uint8)
    try:
        import cv2
        n, lab = cv2.connectedComponents(holes, connectivity=4)
    except Exception:
        # 无 cv2：scipy 的 4 邻域连通域兜底（标签从 1 开始，与 cv2 对齐）
        try:
            from scipy import ndimage
        except Exception:
            return cut, 0  # 两个都没有就放弃补洞，别把整条拼板链路带崩
        lab, count = ndimage.label(holes, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        n = int(count) + 1
    h, w = kept.shape
    limit = int(max_ratio * h * w)
    filled = 0
    for i in range(1, n):
        comp = lab == i
        ys, xs = np.nonzero(comp)
        if not len(ys):
            continue
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            continue  # 触边 = 外部背景
        area = int(comp.sum())
        if area > limit:
            continue
        a[comp, 3] = 255
        filled += area
    if not filled:
        return cut, 0
    return Image.fromarray(a, "RGBA"), filled


def _clean_edges(cut: Image.Image, bg: Tuple[int, int, int], depth: int = 2
                 ) -> Tuple[Image.Image, int]:
    """边缘溢色清洗——实现已提到 matte.clean_edge_spill（sprite 管线也要用）。

    留这个薄包装是因为拼板逻辑里 _clean_edges 这个名字更贴合上下文，
    同时避免两套实现各自漂移。
    """
    return matte_pipe.clean_edge_spill(cut, bg, depth=depth)


def _drop_bg_islands(cut: Image.Image, tol: float = 70.0) -> Image.Image:
    """丢掉 mask 里的孤立背景伪影岛（实现已提到 matte.drop_bg_islands）。

    留这个薄包装是因为拼板逻辑里 _drop_bg_islands 这个名字更贴合上下文，
    同时避免两套实现各自漂移。
    """
    return matte_pipe.drop_bg_islands(cut, tol=tol)


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
# 单步执行：正视图 / 侧背视图 / （拼板走 compose_three_view）
# 分步界面按这三步推进，任何一步都能单独重跑；run_three_view 只是把它们串起来。
# --------------------------------------------------------------------------- #
def cell(width: int, height: int) -> Tuple[int, int]:
    """对齐到 16 的倍数（下采样约束），最小 64。"""
    return max(64, int(width) // 16 * 16), max(64, int(height) // 16 * 16)


def _engines(*, provider_id: Optional[str] = None,
             provider_cfg: Optional[Dict[str, Any]] = None,
             txt2img_model: Optional[str] = None) -> Dict[str, Any]:
    """解析这一步要用的底座：正视图走 txt2img，侧/背视图走编辑底座。

    分步生成时每个请求都重新解析一次——中途可能刚在设置页换过默认
    Provider，也可能有别的请求把模型换进换出了。代价只有一次设置读取
    加几个目录判断，不值得做缓存。
    """
    from ..config import ROOT, load_settings, resolve_path
    from ..providers import get_provider
    from ..providers.diffusers_local import DiffusersProvider, detect_family

    settings = load_settings()
    pid = provider_id or settings.get("activeProvider") or "diffusers"
    pcfg = provider_cfg if provider_cfg is not None else \
        (settings.get("providers") or {}).get(pid) or {}
    active = get_provider(pid, pcfg)

    # 当前 Provider 是编辑器（不能从零出图）时，正视图借 txt2img 底座生成
    is_editor = (
        getattr(active, "id", "") == "diffusers"
        and detect_family(active._resolve_model()) == "qwen_edit"
    )
    # txt2img_model 可能是相对路径（工作流里写的就是 "models/z-image-turbo"），
    # 按项目根解析，从别的目录启动服务也找得到
    txt_model = resolve_path(txt2img_model or str(ROOT / "models" / "z-image-turbo"))
    if is_editor and not os.path.isdir(txt_model):
        raise RuntimeError(
            "当前 Provider 是 Qwen-Image-Edit 编辑器（无法从零出图），"
            f"但找不到 txt2img 底座：{txt_model}。请先下载 Z-Image-Turbo，"
            "或在设置里把默认 Provider 换成生成底座。"
        )
    txt = DiffusersProvider({**pcfg, "model": txt_model}) if is_editor else active

    edit_model: Optional[str] = None
    edit_fallback: Optional[str] = None
    try:
        from . import ai_edit

        edit_model = ai_edit.resolve_edit_model(None)
    except Exception as e:  # noqa: BLE001
        edit_fallback = str(e)
    return {
        "pid": pid, "txt": txt, "isEditor": is_editor,
        "editModel": edit_model, "editFallback": edit_fallback,
    }


async def run_front(
    subject: str,
    *,
    style: Optional[str] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 768,
    provider_id: Optional[str] = None,
    provider_cfg: Optional[Dict[str, Any]] = None,
    txt2img_model: Optional[str] = None,
    engines: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """第一步：正视图（txt2img）。

    三视图的起点必须是文生图——编辑器只能改已有图，不能从零创作。
    后两张的全部造型都由这一张决定，所以它值得单独重跑。
    """
    from ..styles import get_style, style_words

    eng = engines or _engines(provider_id=provider_id, provider_cfg=provider_cfg,
                              txt2img_model=txt2img_model)
    st = get_style(style)
    sw = style_words(style)
    cw, ch = cell(width, height)
    prompt = _view_prompt((subject or "").strip(), "front", sw)

    t = time.time()
    res = await eng["txt"].generate(
        prompt=prompt, width=cw, height=ch, steps=steps,
        cfg=cfg if cfg is not None else 7.5, seed=seed, n=1)
    img = res.images[0].convert("RGB")
    meta: Dict[str, Any] = {
        "view": "front", "name": _VIEW_META["front"]["cn"], "mode": "txt2img",
        "prompt": prompt, "steps": res.meta.get("steps"), "seed": seed,
        "seconds": round(time.time() - t, 1), "cell": [cw, ch],
        "style": {"id": st["id"], "name": st["name"]} if st else None,
        "styleWords": sw or None,
        "txt2imgModel": eng["txt"]._resolve_model()
        if hasattr(eng["txt"], "_resolve_model") else eng["pid"],
    }
    if st and st.get("pixelize"):
        meta["suggestPost"] = st["pixelize"]
    return img, meta


async def run_derived(
    view: str,
    base: Image.Image,
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
    engines: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """第二/三步：侧视图、背视图。

    优先走编辑底座指令接力（外观保持最好）；编辑器不可用时退回同种子
    txt2img——一致性弱一些，但仍保证「一张图一个角色、三张不重叠」。
    """
    from ..styles import style_words

    if view not in ("side", "back"):
        raise ValueError(f"未知视角：{view}")

    eng = engines or _engines(provider_id=provider_id, provider_cfg=provider_cfg,
                              txt2img_model=txt2img_model)
    cw, ch = cell(width, height)
    t = time.time()

    if eng["editModel"]:
        from . import ai_edit

        out, emeta = await ai_edit.run_edit(
            base, _view_instruction(view), steps=steps, seed=seed,
            model=eng["editModel"], max_side=max_side)
        return out, {
            "view": view, "name": _VIEW_META[view]["cn"], "mode": "edit",
            "instruction": _view_instruction(view),
            "steps": emeta.get("steps"), "seed": seed,
            "seconds": round(time.time() - t, 1), "cell": [cw, ch],
            "editModel": eng["editModel"],
        }

    sw = style_words(style)
    prompt = _view_prompt((subject or "").strip(), view, sw)
    res = await eng["txt"].generate(
        prompt=prompt, width=cw, height=ch, steps=steps,
        cfg=cfg if cfg is not None else 7.5, seed=seed, n=1)
    meta: Dict[str, Any] = {
        "view": view, "name": _VIEW_META[view]["cn"], "mode": "txt2img",
        "prompt": prompt, "steps": res.meta.get("steps"), "seed": seed,
        "seconds": round(time.time() - t, 1), "cell": [cw, ch],
    }
    if eng["editFallback"]:
        meta["editFallback"] = eng["editFallback"]
    return res.images[0].convert("RGB"), meta


# --------------------------------------------------------------------------- #
# 主管线（一次跑完，CLI / 工作流用；分步界面走各自的函数）
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

    - subject：角色描述（中文，直接送中文原生底座，不翻译）；
    - style：画风 id（见 app/styles.py），只作用于帧 1，帧 2+ 靠编辑指令继承画风；
    - 帧 2+ 优先走 Qwen-Image-Edit 指令接力（外观保持最好）；
      编辑底座缺失时退回同种子 txt2img × 3（meta.editFallback 记录原因）。
    """
    from ..styles import get_style

    t0 = time.time()
    eng = _engines(provider_id=provider_id, provider_cfg=provider_cfg,
                   txt2img_model=txt2img_model)
    kw: Dict[str, Any] = dict(style=style, steps=steps, cfg=cfg, seed=seed,
                              width=width, height=height,
                              provider_cfg=provider_cfg, engines=eng)

    # ---- 帧 1：正视图 ----
    front, fmeta = await run_front(subject, provider_id=provider_id, **kw)
    views = [front]
    per_view: List[Dict[str, Any]] = [fmeta]

    # ---- 帧 2/3：侧/背视图（编辑底座接力，缺失时退回 txt2img） ----
    for key in ("side", "back"):
        out, vmeta = await run_derived(
            key, front, subject, max_side=max_side, provider_id=provider_id, **kw)
        views.append(out)
        per_view.append(vmeta)

    cw, ch = cell(width, height)
    strip, cmeta = compose_three_view(views, cell=(cw, ch))
    st = get_style(style)
    meta: Dict[str, Any] = {
        "pipeline": "threeview",
        "subject": (subject or "").strip(),
        "style": {"id": st["id"], "name": st["name"]} if st else None,
        "mode": "txt2img→edit-relay" if eng["editModel"] else "txt2img×3",
        "txt2imgModel": fmeta.get("txt2imgModel"),
        "editModel": eng["editModel"],
        "cell": [cw, ch],
        "views": per_view,
        "compose": cmeta,
        "seed": seed,
        "promptLang": "zh",
        "seconds": round(time.time() - t0, 1),
    }
    if eng["editFallback"]:
        meta["editFallback"] = eng["editFallback"]
    if st and st.get("pixelize"):
        meta["suggestPost"] = st["pixelize"]
    return strip, views, meta
