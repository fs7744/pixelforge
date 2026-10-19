"""人物像素化：一张人物照片 → 像素角色立绘，三档风格化程度。

为什么不是「换个画风」：画风转换（/api/pipeline/restyle）只换笔触，人物比例
和五官写实度不动；照片要变成游戏里能用的像素角色，还得改**造型语言**——
头身比、眼睛大小、五官简化程度、阴影层数。所以这里是「一张照片 + 一个档位」
的独立管线，而不是画风库里再加几个画风。

两段式，缺一不可：
1. **AI 重绘**（Qwen-Image-Edit）：照片 → 该档位的像素角色画风。编辑底座
   能保住人物身份（发型、发色、服装），但**出不了真正的硬边像素**——
   它给的是「像素感的插画」；
2. **确定性像素化**（pixelize）：按「横向格数」算块大小，分块取单色 + 限定
   调色板，得到硬边、每格单色的真像素图。AI 那一步只负责造型，这一步负责
   像素语法；
3. **背景**（可选）：绿幕 / 透明都是本地抠图后处理出来的，不靠 AI 画绿底
   ——原因见 BACKGROUNDS 上方的注释（绿幕会把角色染绿）。

档位参数为什么用「横向格数」而不是「像素块大小」：同一档位在 1024 照片和
512 照片上应该得到一样的像素密度，写死 block=8 会让大图变成超细像素、
小图变成马赛克。格数是分辨率无关的，block 由 block_for() 反算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..config import CHAR_PIXEL_SAMPLES_DIR

SAMPLE_URL_PREFIX = "/char-pixel-samples"

# 三档：造型语言（写实 / 半 Q / Q 版）× 像素密度 × 色数
# - grid：成品横向格数（分辨率无关的像素密度）
# - colors：调色板色数。写实档需要更多色阶撑体积，Q 版靠平涂，色少反而干净
#
# 提示词里必须显式写到的四件事（缺一件就会退化成「照片加马赛克」）：
# 1. **粗黑轮廓线**——像素画的辨识度一半来自描边，不写模型就保留照片的软边；
# 2. **平涂色块 + 限定明暗层次**——照片级的细腻渐变被量化后就是一片脏噪点；
# 3. **深色瞳孔 + 白色高光**——不写清楚，眼睛会变成一坨灰或死白；
# 4. **高饱和明快配色 + 偏暖肤色**——模型默认往灰白走，出来像褪色照片。
LEVELS: List[Dict[str, Any]] = [
    {
        "id": "fine",
        "name": "轻度 · 写实比例",
        "short": "写实比例",
        "hint": "保留照片里的五官与真实头身比，只把笔触换成像素画；适合想要「像本人」的头像",
        "heads": "6-7 头身",
        "grid": 128,
        "colors": 32,
        "prompt": (
            "把这张照片重绘成像素画角色立绘（dot art）："
            "五官特征、脸型、发型、发色和服装照照片中的人物绘制，保留真实的人体比例；"
            "用粗黑轮廓线勾出发型、脸颊、肩线和衣服边缘；"
            "皮肤、头发、衣服全部平涂成色块，每块只有一种颜色，明暗最多两级硬边过渡，"
            "不要照片级的细腻渐变、不要噪点；"
            "眼睛画成清晰的深色瞳孔加一点白色高光，眼线明确，眉毛和嘴唇用色块概括；"
            "肤色用偏暖的肉色，脸颊带一点红晕；"
            "背景简化为像素游戏场景，树木草地用大色块概括，删掉细碎枝叶；"
            "像素边缘硬朗、无抗锯齿，限定调色板，配色鲜明、明暗对比清楚。"
        ),
    },
    {
        "id": "mid",
        "name": "中度 · 半 Q 版",
        "short": "半 Q 版",
        "hint": "约 4 头身，头略放大、眼睛放大，保留人物辨识度的游戏立绘比例",
        "heads": "约 4 头身",
        "grid": 104,
        "colors": 24,
        "prompt": (
            "把这张照片重绘成日系像素 RPG 角色立绘："
            "按照片中人物的发型、发色、脸型和服装配色绘制，比例调整到约 4 头身；"
            "头部略放大，用粗黑轮廓线勾出头发、脸、身体和衣服的边线；"
            "眼睛放大成深色瞳孔加白色高光，眼线清晰，鼻子和嘴用两三个色块概括；"
            "全身平涂，每块一种颜色，两级硬边阴影，不要渐变、不要照片质感；"
            "全身完整入画，头顶到脚底都在画面内；"
            "肤色偏暖，脸颊带红晕，配色鲜明、明暗对比清楚；"
            "背景简化为像素游戏场景的大色块；"
            "像素边缘硬朗、无抗锯齿，限定调色板。"
        ),
    },
    {
        "id": "chibi",
        "name": "重度 · Q 版",
        "short": "Q 版",
        "hint": "约 2 头身，大头大眼、四肢短粗，最像游戏里的行走小人",
        "heads": "约 2 头身",
        "grid": 76,
        "colors": 16,
        "prompt": (
            "把这张照片重绘成 Q 版像素角色："
            "保留照片中人物的发型、发色和服装配色，比例压缩到约 2 头身，头部约占全身一半；"
            "用粗黑轮廓线勾出整个角色的外轮廓和五官；"
            "眼睛大而圆，深色瞳孔占满眼白并带一点白色高光，眼线粗黑，鼻子省略，嘴画成小弧线；"
            "四肢短粗，手掌简化为圆块；"
            "大面积平涂，每块一种颜色，单级硬边阴影，不要渐变；"
            "画全身，头顶到脚底完整落在画面内，四周留出边距，不裁切；"
            "肤色偏暖，脸颊带红晕，配色鲜明；"
            "背景简化为纯色或简单色块；"
            "像素边缘硬朗、无抗锯齿，限定调色板。"
        ),
    },
]

_BY_ID: Dict[str, Dict[str, Any]] = {lv["id"]: lv for lv in LEVELS}

# 背景处理：AI 只负责画角色，**不要求它画绿幕**。
#
# 踩过的坑：让编辑底座「背景铺纯绿 #00FF00」时，模型为了画面协调会把绿色带进
# 人物本体——实测同一张人像、同种子，肤色被画成橄榄绿 (183,195,67)、白 T 恤
# 变成柠檬绿；换个措辞又反过来把 T 恤画成肤色。这类染色是概率性的，靠提示词
# 压不干净。
#
# 现在的做法：一律生成「带背景的正常图」（颜色最忠实），再用 u2net 把人物抠
# 出来——绿幕是我们自己合成的纯色（必然是纯 #00FF00），透明就是抠出来的
# alpha。好处：颜色不经过绿幕污染、绿幕绝对干净、三种背景可以随时切换而不用
# 重新生成。
BACKGROUNDS: List[Dict[str, Any]] = [
    {
        "id": "keep",
        "name": "保留原图",
        "hint": "背景一起像素化，保留照片原本的环境（参考图那种场景感）",
    },
    {
        "id": "green",
        "name": "绿幕",
        "hint": "本地抠出人物后合成纯 #00FF00 底，颜色绝不被绿幕污染；进引擎前可自己键控",
    },
    {
        "id": "alpha",
        "name": "透明",
        "hint": "本地抠图输出带 alpha 的 PNG（硬边、无白晕），可直接当精灵素材",
    },
]
_BG_IDS = {b["id"] for b in BACKGROUNDS}

GREEN_KEY = (0, 255, 0)


def get_level(level_id: Optional[str]) -> Dict[str, Any]:
    """按 id 取档位；空值取中度（最常用的「既像本人又能进游戏」的比例）。"""
    key = (level_id or "mid").strip().lower()
    lv = _BY_ID.get(key)
    if lv is None:
        raise ValueError(
            f"未知像素化档位：{key}。合法值：" + " / ".join(
                f"{x['id']}（{x['name']}）" for x in LEVELS))
    return lv


def list_levels() -> List[Dict[str, Any]]:
    """给前端的档位列表。

    sample：档位示例图（gen_char_pixel_samples.py 生成过才返回）。三档差别
    是「造型语言」，不摆图用户根本没法在 2-4 分钟的生成前做判断。
    """
    out = []
    for lv in LEVELS:
        f = CHAR_PIXEL_SAMPLES_DIR / f"{lv['id']}.png"
        out.append({
            "id": lv["id"],
            "name": lv["name"],
            "short": lv["short"],
            "hint": lv["hint"],
            "heads": lv["heads"],
            "grid": lv["grid"],
            "colors": lv["colors"],
            "sample": f"{SAMPLE_URL_PREFIX}/{lv['id']}.png" if f.is_file() else None,
        })
    return out


def build_prompt(level: Dict[str, Any], extra: str = "") -> str:
    """档位提示词 + 构图锚点 + 用户补充要求。

    锚点写成「沿用原图」而不是「保持不变」：编辑底座对否定约束极敏感，
    写成「不要改变构图」时它连画风一起不动（restyle 踩过的坑）。

    背景不写进生成指令：绿幕/透明都在出图后本地处理（见 BACKGROUNDS 注释），
    AI 只负责把角色画对、颜色画准。
    """
    text = level["prompt"] + "人物在画面中的位置与构图沿用原图，背景一并做同样的像素化处理。"
    extra = (extra or "").strip()
    if extra:
        text += f"补充要求：{extra}"
    return text


def block_for(width: int, grid: int) -> int:
    """横向格数 → 像素块大小（宽 / 格数，至少 1px）。"""
    return max(1, int(round(width / max(8, int(grid)))))


def apply_pixels(
    img: Image.Image, grid: int, colors: int, dither: bool = False
) -> Tuple[Image.Image, Dict[str, Any]]:
    """按横向格数做确定性像素化（分块单色 + 限定调色板，最近邻放大回原尺寸）。"""
    from . import pixelize as pixelize_pipe

    block = block_for(img.width, int(grid))
    out, meta = pixelize_pipe.pixelize(
        img, colors=int(colors), pixel_size=block, dither=dither)
    meta["block"] = block
    meta["requestedGrid"] = int(grid)
    return out, meta


def get_background(bg_id: Optional[str]) -> str:
    key = (bg_id or "keep").strip().lower()
    if key not in _BG_IDS:
        raise ValueError(
            f"未知背景处理：{key}。合法值：" + " / ".join(
                f"{b['id']}（{b['name']}）" for b in BACKGROUNDS))
    return key


def cut_out(img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """抠出人物，返回带硬 alpha 的 RGBA。

    走深度抠图链（u2net → grabcut → 色键）：背景是照片/像素场景，不是纯色幕，
    色键在这里没有可用基准。硬 alpha + 边缘去污保证像素素材没有半透明白晕；
    再清掉 u2net 偶发留下的小块背景伪影（长椅碎片、草丛色块）。
    """
    from . import matte as matte_pipe

    cut, info = matte_pipe.connected_matte(
        img, tolerance=60, feather=0, hard=True)
    cut = matte_pipe.drop_bg_islands(cut)
    return cut, info


def compose_green(cut: Image.Image) -> Image.Image:
    """把抠好的 RGBA 合成到纯 #00FF00 上——绿幕由我们生成，必然是纯色。"""
    canvas = Image.new("RGB", cut.size, GREEN_KEY)
    canvas.paste(cut, (0, 0), cut)
    return canvas


def _apply_background(
    img: Image.Image, bg: str
) -> Tuple[Image.Image, Dict[str, Any]]:
    """按背景选项处理像素化后的成品：keep 原样 / alpha 抠图 / green 抠图后铺绿。"""
    if bg == "keep":
        return img, {}
    cut, info = cut_out(img)
    return (cut if bg == "alpha" else compose_green(cut)), info


async def run(
    img: Image.Image,
    level_id: Optional[str] = None,
    prompt: str = "",
    *,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    model: Optional[str] = None,
    max_side: int = 1024,
    grid: Optional[int] = None,
    colors: Optional[int] = None,
    background: str = "keep",
    task_id: Optional[str] = None,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """照片 → 像素角色，返回（AI 原稿, 像素化成品, meta）。

    原稿单独返回并入库：像素化是纯 CPU 的毫秒级操作，用户调块大小/色数时
    没必要重跑 40 步编辑。前端「重新像素化」按钮就是拿原稿再来一次。
    """
    from . import ai_edit

    level = get_level(level_id)
    bg = get_background(background)
    instruction = build_prompt(level, prompt)
    draft, meta = await ai_edit.run_edit(
        img, instruction, steps=steps, seed=seed, model=model, max_side=max_side,
        task_id=task_id)

    g = int(grid or level["grid"])
    c = int(colors or level["colors"])
    out, px = apply_pixels(draft, g, c)
    out, minfo = _apply_background(out, bg)
    if minfo:
        meta["matte"] = minfo
    meta.update({
        "pipeline": "char-pixel",
        "level": level["id"],
        "levelName": level["name"],
        "heads": level["heads"],
        "grid": g,
        "colors": c,
        "background": bg,
        "pixelize": px,
        "userPrompt": (prompt or "").strip(),
    })
    return draft, out, meta


def repixel(
    draft: Image.Image,
    grid: int,
    colors: int,
    dither: bool = False,
    background: str = "keep",
) -> Tuple[Image.Image, Dict[str, Any]]:
    """对已有 AI 原稿重新像素化（不碰 GPU）。

    背景三种都能随意切：原稿是「带背景的正常图」，绿幕/透明都是本地抠出来再
    合成的，所以改背景不需要重新生成——这是「不要求 AI 画绿幕」换来的好处。
    """
    bg = get_background(background)
    out, px = apply_pixels(draft, int(grid), int(colors), dither)
    out, minfo = _apply_background(out, bg)
    meta: Dict[str, Any] = {
        "pipeline": "char-pixel",
        "grid": int(grid),
        "colors": int(colors),
        "background": bg,
        "pixelize": px,
        "reusedDraft": True,
    }
    if minfo:
        meta["matte"] = minfo
    return out, meta
