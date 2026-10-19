"""美术画风库：游戏行业典型风格，生成时拼接在提示词尾部。

设计要点：
- 风格词统一为中文（两个底座 Z-Image / Qwen-Image-Edit 都是中文原生），
  不再维护英文版，也没有中文→英文的翻译链路；
- 风格词放提示词尾部：注意力对尾部权重自然衰减，放开头容易被主体描述淹没；
- pixelize 字段是「后处理建议」：像素类画风出图后应接最近邻降采样 +
  调色板量化，参数随画风返回给前端，一键送后处理时预填。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import STYLE_SAMPLES_DIR

# 画风示例图 URL 前缀（main.py 挂载）；文件不存在时前端回落纯文字 tooltip
SAMPLE_URL_PREFIX = "/style-samples"

STYLES: List[Dict[str, Any]] = [
    # ---------------- 像素类 ----------------
    {
        "id": "pixel_16bit",
        "name": "复古 16-bit 像素",
        "tags": ["角色", "场景", "道具"],
        "hint": "FC/SFC 时代像素风，32×32 网格感，限定调色板，出图后建议 8× 像素化",
        "prompt": "复古16-bit像素艺术风格，硬边缘像素块，有限调色板（16色以内），无抗锯齿，无渐变，清晰像素网格",
        "pixelize": {"block": 8, "colors": 16},
    },
    {
        "id": "pixel_hd",
        "name": "高清像素（HD Pixel）",
        "tags": ["角色", "场景"],
        "hint": "现代独立游戏像素风（Owlboy/Sea of Stars 感），大色块+选择性高光，出图后建议 4× 像素化",
        "prompt": "高清像素艺术风格，大块饱满色块，选择性高光与轮廓光，细腻色阶但保持像素硬边，现代独立游戏美术",
        "pixelize": {"block": 4, "colors": 32},
    },
    # ---------------- 2D 动漫/卡通 ----------------
    {
        "id": "chibi",
        "name": "日系 Q 版卡通",
        "tags": ["角色"],
        "hint": "3 头身萌系，粗黑描边平涂，游戏角色立绘常用",
        "prompt": "二维日系Q版卡通风格，3头身萌系比例，头部放大，四肢简化圆润，粗黑色外轮廓线，明亮饱和色彩，大面积平涂色块，硬边阴影",
    },
    {
        "id": "anime_cel",
        "name": "日系动画赛璐璐",
        "tags": ["角色", "场景"],
        "hint": "TV 动画赛璐璐上色，干净线稿+两级阴影，Galgame/二次元手游立绘感",
        "prompt": "日系动画赛璐璐风格，干净流畅线稿，平涂上色，两级硬边阴影，高光点缀，动画原画质感",
    },
    {
        "id": "manga_bw",
        "name": "日式黑白漫画",
        "tags": ["角色", "场景"],
        "hint": "黑白网点纸漫画风，速写线条+拟声字排版空间",
        "prompt": "日式黑白漫画风格，墨线勾勒，网点纸灰阶，高对比黑白，漫画分镜感，无彩色",
    },
    {
        "id": "us_cartoon",
        "name": "美式卡通",
        "tags": ["角色", "场景"],
        "hint": "Cartoon Network / 迪士尼 TV 动画风，粗轮廓+夸张动态",
        "prompt": "美式卡通动画风格，粗黑轮廓线，夸张动态造型，明亮色块，富有表现力的面部，电视动画质感",
    },
    # ---------------- UI / 图标 ----------------
    {
        "id": "flat_vector",
        "name": "扁平矢量插画",
        "tags": ["UI", "场景", "道具"],
        "hint": "手游 UI/宣传图常用，无描边大色块，简洁现代",
        "prompt": "扁平矢量插画风格，无描边，简洁几何形，明快大色块，现代手游UI美术，干净留白",
    },
    {
        "id": "outline_icon",
        "name": "粗描边游戏图标",
        "tags": ["UI", "道具"],
        "hint": "App 图标/物品图标风，粗描边+高饱和+居中构图，透明底友好",
        "prompt": "游戏图标风格，粗黑描边，高饱和配色，居中构图，简洁造型，图标级可读性，平涂",
    },
    # ---------------- 写实/3D ----------------
    {
        "id": "painterly",
        "name": "厚涂半写实",
        "tags": ["角色", "场景"],
        "hint": "数字厚涂，笔触可见，暗黑奇幻/ARPG 立绘感",
        "prompt": "半写实数字厚涂风格，可见笔触，丰富中间色调，戏剧化光影，暗黑奇幻游戏原画质感",
    },
    {
        "id": "realistic",
        "name": "写实 3A 风",
        "tags": ["角色", "场景"],
        "hint": "3A 写实渲染方向，PBR 材质描述，概念设计稿",
        "prompt": "写实3A游戏概念设计，照片级材质细节，次表面散射皮肤，体积光，电影级构图，虚幻引擎渲染质感",
    },
    {
        "id": "lowpoly",
        "name": "低多边形 3D",
        "tags": ["场景", "道具"],
        "hint": "低面数 3D，等距视角友好，独立游戏场景常用",
        "prompt": "低多边形3D风格，少面数几何体，平面着色，等距视角，柔和 pastel 配色，独立游戏场景",
    },
    {
        "id": "voxel",
        "name": "体素风",
        "tags": ["角色", "场景"],
        "hint": "MagicaVoxel 立方体风（Crossy Road 感），等距场景出彩",
        "prompt": "体素艺术风格，均匀立方体单元，等距视角，明亮配色，3D像素方块构成，干净边缘",
    },
    # ---------------- 手绘 ----------------
    {
        "id": "watercolor",
        "name": "水彩手绘",
        "tags": ["场景", "UI"],
        "hint": "透明水彩层叠，纸纹留白，绘本/治愈系游戏感",
        "prompt": "水彩手绘风格，透明颜料层叠，湿画法晕染，纸纹与留白，柔和边缘，绘本插画感",
    },
    {
        "id": "concept_sketch",
        "name": "概念速写",
        "tags": ["角色", "场景"],
        "hint": "铅笔/炭笔速写，设计稿阶段用，线稿+灰调",
        "prompt": "概念设计速写风格，铅笔线稿，轻灰调，构图探索感，设计稿质感，未上色或局部上色",
    },
]

_BY_ID: Dict[str, Dict[str, Any]] = {s["id"]: s for s in STYLES}

# --------------------------------------------------------------------------- #
# 模板分类 → 推荐画风
# --------------------------------------------------------------------------- #
# 为什么不用 tags 过滤：tags 太粗，「场景」覆盖 13/14 个画风、「角色」覆盖
# 10 个，按交集筛完几乎没减少。这里按「这类模板实际会用到什么」单独维护
# 一份表，默认只露 5-6 个最常用的，其余折在「全部」里。
# 「工程」是文本任务（不出图），返回空列表，前端整块隐藏画风区。
CATEGORY_STYLES: Dict[str, List[str]] = {
    "角色": ["anime_cel", "chibi", "pixel_16bit", "pixel_hd", "painterly", "realistic"],
    "动画": ["chibi", "anime_cel", "pixel_16bit", "pixel_hd", "us_cartoon", "concept_sketch"],
    "场景": ["anime_cel", "painterly", "realistic", "watercolor", "pixel_hd", "lowpoly"],
    "视差": ["painterly", "realistic", "watercolor", "pixel_hd", "anime_cel", "lowpoly"],
    "瓦片": ["pixel_16bit", "pixel_hd", "lowpoly", "voxel", "painterly"],
    "工程": [],
}


def recommend_for(category: Optional[str]) -> List[str]:
    """某类模板默认展示的画风 id（顺序即 chip 顺序）；空列表 = 不展示画风区。"""
    if not category:
        return []
    return [i for i in (CATEGORY_STYLES.get(category) or []) if i in _BY_ID]


def list_styles() -> List[Dict[str, Any]]:
    """画风列表（去掉大字段，给前端 chips 用）。

    sample：示例图 URL（gen_style_samples.py 生成过才返回），供悬浮 tooltip 展示。
    """
    out = []
    for s in STYLES:
        sample_file = STYLE_SAMPLES_DIR / f"{s['id']}.png"
        out.append(
            {
                "id": s["id"],
                "name": s["name"],
                "tags": s["tags"],
                "hint": s["hint"],
                "pixelize": s.get("pixelize"),
                "sample": f"{SAMPLE_URL_PREFIX}/{s['id']}.png" if sample_file.is_file() else None,
            }
        )
    return out


def get_style(style_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return _BY_ID.get(style_id) if style_id else None


def apply_style(prompt: str, style_id: Optional[str]) -> Dict[str, Any]:
    """把中文风格词拼到提示词尾部，返回 {prompt, style, suggestPost}。

    风格词放尾部：注意力对尾部权重自然衰减，放开头容易被主体描述淹没。
    拼好后直接送中文原生底座，不需要任何语言转换。
    """
    style = get_style(style_id)
    if not style:
        return {"prompt": prompt, "style": None, "suggestPost": None}
    words = style["prompt"]
    merged = prompt.rstrip("。.，, ") + "。" + words if prompt.strip() else words
    return {
        "prompt": merged,
        "style": {"id": style["id"], "name": style["name"]},
        "suggestPost": style.get("pixelize"),
    }
