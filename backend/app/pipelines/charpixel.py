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
   像素语法。

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
LEVELS: List[Dict[str, Any]] = [
    {
        "id": "fine",
        "name": "轻度 · 写实比例",
        "short": "写实比例",
        "hint": "保留照片里的五官与真实头身比，只把笔触换成像素画；适合想要「像本人」的头像",
        "heads": "6-7 头身",
        "grid": 168,
        "colors": 48,
        "prompt": (
            "把这张照片重绘成高精度像素角色立绘："
            "五官特征、脸型、发型、发色和服装照照片中的人物绘制，保留真实的人体比例；"
            "面部用像素画方式刻画，眼睛、眉毛、嘴唇清晰可辨，头发分组块、有高光；"
            "光影柔和有体积感，明暗过渡用有限色阶表现；"
            "像素块细小、边缘硬朗，限定调色板，日式独立游戏角色立绘质感。"
        ),
    },
    {
        "id": "mid",
        "name": "中度 · 半 Q 版",
        "short": "半 Q 版",
        "hint": "约 4 头身，头略放大、眼睛放大，保留人物辨识度的游戏立绘比例",
        "heads": "约 4 头身",
        "grid": 120,
        "colors": 28,
        "prompt": (
            "把这张照片重绘成日系像素游戏角色："
            "按照片中人物的五官特征、发型、发色和服装配色绘制，比例调整到约 4 头身；"
            "头部略放大，眼睛放大并简化眼型，鼻子和嘴简化，手部与四肢比例简化；"
            "像素块中等偏粗、边缘硬朗，大面积平涂配两级硬边阴影，限定调色板。"
        ),
    },
    {
        "id": "chibi",
        "name": "重度 · Q 版",
        "short": "Q 版",
        "hint": "约 2 头身，大头大眼、四肢短粗，最像游戏里的行走小人",
        "heads": "约 2 头身",
        "grid": 84,
        "colors": 20,
        "prompt": (
            "把这张照片重绘成 Q 版像素角色："
            "保留照片中人物的发型、发色和服装配色，比例压缩到约 2 头身，头部约占全身一半；"
            "眼睛大而圆、占脸部三分之一，鼻子和嘴简化为小点，四肢短粗、手掌简化为圆块；"
            "像素块粗、边缘硬朗，大面积平涂配单级硬边阴影，限定调色板，可爱游戏角色图标质感。"
        ),
    },
]

_BY_ID: Dict[str, Dict[str, Any]] = {lv["id"]: lv for lv in LEVELS}

# 背景处理：游戏素材最终要透明 PNG，但 AI 一步出不了可靠 alpha，
# 所以「透明」的实际做法是——生成时铺纯绿幕，出图后按键色确定性抠掉。
# 这样 alpha 边缘是像素对齐的硬边，不是模型猜出来的半透明。
BACKGROUNDS: List[Dict[str, Any]] = [
    {
        "id": "keep",
        "name": "保留原图",
        "hint": "背景一起像素化，保留照片原本的环境（参考图那种场景感）",
    },
    {
        "id": "green",
        "name": "绿幕",
        "hint": "纯 #00FF00 满铺，自己进引擎/后处理键控；角色身上别有大块绿",
    },
    {
        "id": "alpha",
        "name": "透明",
        "hint": "生成绿幕后按色键抠掉，输出带 alpha 的 PNG（硬边、无白晕）",
    },
]
_BG_IDS = {b["id"] for b in BACKGROUNDS}

# 绿幕底提示词：只写正向要求，并明确「无渐变无纹理」——
# 渐变底会让色键只抠掉一部分，留下半透明脏边。
_GREEN_BG_PROMPT = (
    "背景为纯绿色 #00FF00 满铺，颜色均匀、无渐变、无纹理、无投影，"
    "人物边缘干净利落，不要出现环境元素。"
)


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


def build_prompt(level: Dict[str, Any], extra: str = "", background: str = "keep") -> str:
    """档位提示词 + 构图锚点 + 背景要求 + 用户补充要求。

    锚点写成「沿用原图」而不是「保持不变」：编辑底座对否定约束极敏感，
    写成「不要改变构图」时它连画风一起不动（restyle 踩过的坑）。
    """
    text = level["prompt"] + "人物在画面中的位置与构图沿用原图，"
    if background in ("green", "alpha"):
        text += _GREEN_BG_PROMPT
    else:
        text += "背景一并做同样的像素化处理。"
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
    """绿幕 → 透明：强制走色键（method=key），不用 u2net/grabcut。

    为什么不用深度抠图：这里的前景是 AI 刚画在纯绿底上的像素角色，色键是
    精确且可复现的；u2net 会按「语义」猜边缘，反而把描边啃掉。硬 alpha 之后
    再洗一圈边缘溢色，避免轮廓外留着「黑 + 亮绿」的混色。
    """
    from . import matte as matte_pipe

    cut, info = matte_pipe.connected_matte(
        img, tolerance=60, feather=0, sample="topleft", method="key", hard=True)
    bg_hex = info.get("background")
    if bg_hex:
        bg = tuple(int(bg_hex[i:i + 2], 16) for i in (1, 3, 5))
        cut, spilled = matte_pipe.clean_edge_spill(cut, bg)
        info["edgePixelsRecolored"] = spilled
    return cut, info


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
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """照片 → 像素角色，返回（AI 原稿, 像素化成品, meta）。

    原稿单独返回并入库：像素化是纯 CPU 的毫秒级操作，用户调块大小/色数时
    没必要重跑 40 步编辑。前端「重新像素化」按钮就是拿原稿再来一次。
    """
    from . import ai_edit

    level = get_level(level_id)
    bg = get_background(background)
    instruction = build_prompt(level, prompt, bg)
    draft, meta = await ai_edit.run_edit(
        img, instruction, steps=steps, seed=seed, model=model, max_side=max_side)

    g = int(grid or level["grid"])
    c = int(colors or level["colors"])
    out, px = apply_pixels(draft, g, c)
    if bg == "alpha":
        out, minfo = cut_out(out)
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

    background 只允许「保持原样」或「绿幕 → 透明」：原稿若是保留原图背景，
    再按键色去背只会把照片背景啃出一块。这条约束由调用方保证。
    """
    bg = get_background(background)
    out, px = apply_pixels(draft, int(grid), int(colors), dither)
    meta: Dict[str, Any] = {
        "pipeline": "char-pixel",
        "grid": int(grid),
        "colors": int(colors),
        "background": bg,
        "pixelize": px,
        "reusedDraft": True,
    }
    if bg == "alpha":
        out, minfo = cut_out(out)
        meta["matte"] = minfo
    return out, meta
