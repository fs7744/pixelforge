"""美术画风库：游戏行业典型风格，生成时拼接在提示词尾部。

设计要点：
- 风格词按 Provider 语言能力二选一：中文模型（Z-Image / Qwen-Image-Edit）
  用 prompt（中文），SD 系（CLIP）用 promptEn（英文，避免机翻损耗风格词）；
- pixelize 字段是「后处理建议」：像素类画风出图后应接最近邻降采样 +
  调色板量化，参数随画风返回给前端，一键送后处理时预填；
- negative 只对 SD 系生效（Z-Image/Qwen-Edit 蒸馏模型不吃负向提示词）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

STYLES: List[Dict[str, Any]] = [
    # ---------------- 像素类 ----------------
    {
        "id": "pixel_16bit",
        "name": "复古 16-bit 像素",
        "en": "16-bit pixel art",
        "tags": ["角色", "场景", "道具"],
        "hint": "FC/SFC 时代像素风，32×32 网格感，限定调色板，出图后建议 8× 像素化",
        "prompt": "复古16-bit像素艺术风格，硬边缘像素块，有限调色板（16色以内），无抗锯齿，无渐变，清晰像素网格",
        "promptEn": "16-bit pixel art style, hard-edged pixels, limited color palette, no anti-aliasing, no gradients, crisp pixel grid, retro game sprite",
        "negative": "blurry, smooth gradients, 3d render, photo",
        "pixelize": {"block": 8, "colors": 16},
    },
    {
        "id": "pixel_hd",
        "name": "高清像素（HD Pixel）",
        "en": "HD pixel art",
        "tags": ["角色", "场景"],
        "hint": "现代独立游戏像素风（Owlboy/Sea of Stars 感），大色块+选择性高光，出图后建议 4× 像素化",
        "prompt": "高清像素艺术风格，大块饱满色块，选择性高光与轮廓光，细腻色阶但保持像素硬边，现代独立游戏美术",
        "promptEn": "high definition pixel art style, large saturated color blocks, selective highlights and rim light, refined shading with hard pixel edges, modern indie game art",
        "negative": "blurry, photo realistic, smooth gradients",
        "pixelize": {"block": 4, "colors": 32},
    },
    # ---------------- 2D 动漫/卡通 ----------------
    {
        "id": "chibi",
        "name": "日系 Q 版卡通",
        "en": "chibi cartoon",
        "tags": ["角色"],
        "hint": "3 头身萌系，粗黑描边平涂，游戏角色立绘常用",
        "prompt": "二维日系Q版卡通风格，3头身萌系比例，头部放大，四肢简化圆润，粗黑色外轮廓线，明亮饱和色彩，大面积平涂色块，硬边阴影",
        "promptEn": "2D chibi japanese cartoon style, 3 heads tall, big head small body, thick black outline, flat cel shading, bright saturated colors, hard-edged shadows, game character art",
        "negative": "realistic, 3d render, dark, gritty",
    },
    {
        "id": "anime_cel",
        "name": "日系动画赛璐璐",
        "en": "anime cel-shaded",
        "tags": ["角色", "场景"],
        "hint": "TV 动画赛璐璐上色，干净线稿+两级阴影，Galgame/二次元手游立绘感",
        "prompt": "日系动画赛璐璐风格，干净流畅线稿，平涂上色，两级硬边阴影，高光点缀，动画原画质感",
        "promptEn": "japanese anime cel-shaded style, clean line art, flat coloring, two-tier hard shadows, accent highlights, anime key visual quality",
        "negative": "realistic, photo, 3d, western cartoon",
    },
    {
        "id": "manga_bw",
        "name": "日式黑白漫画",
        "en": "manga monochrome",
        "tags": ["角色", "场景"],
        "hint": "黑白网点纸漫画风，速写线条+拟声字排版空间",
        "prompt": "日式黑白漫画风格，墨线勾勒，网点纸灰阶，高对比黑白，漫画分镜感，无彩色",
        "promptEn": "japanese black and white manga style, ink line art, screentone shading, high contrast monochrome, comic panel composition",
        "negative": "color, colorful, 3d render",
    },
    {
        "id": "us_cartoon",
        "name": "美式卡通",
        "en": "western cartoon",
        "tags": ["角色", "场景"],
        "hint": "Cartoon Network / 迪士尼 TV 动画风，粗轮廓+夸张动态",
        "prompt": "美式卡通动画风格，粗黑轮廓线，夸张动态造型，明亮色块，富有表现力的面部，电视动画质感",
        "promptEn": "western cartoon animation style, bold outlines, exaggerated dynamic shapes, bright flat colors, expressive faces, saturday morning cartoon quality",
        "negative": "anime, realistic, muted colors",
    },
    # ---------------- UI / 图标 ----------------
    {
        "id": "flat_vector",
        "name": "扁平矢量插画",
        "en": "flat vector illustration",
        "tags": ["UI", "场景", "道具"],
        "hint": "手游 UI/宣传图常用，无描边大色块，简洁现代",
        "prompt": "扁平矢量插画风格，无描边，简洁几何形，明快大色块，现代手游UI美术，干净留白",
        "promptEn": "flat vector illustration style, no outlines, simple geometric shapes, bold clean color blocks, modern mobile game UI art, generous negative space",
        "negative": "detailed texture, realistic, grunge, outline",
    },
    {
        "id": "outline_icon",
        "name": "粗描边游戏图标",
        "en": "bold outline game icon",
        "tags": ["UI", "道具"],
        "hint": "App 图标/物品图标风，粗描边+高饱和+居中构图，透明底友好",
        "prompt": "游戏图标风格，粗黑描边，高饱和配色，居中构图，简洁造型，图标级可读性，平涂",
        "promptEn": "game icon style, thick dark outline, highly saturated colors, centered composition, simple readable silhouette, flat shading, icon design",
        "negative": "busy background, realistic, full scene",
    },
    # ---------------- 写实/3D ----------------
    {
        "id": "painterly",
        "name": "厚涂半写实",
        "en": "semi-realistic digital painting",
        "tags": ["角色", "场景"],
        "hint": "数字厚涂，笔触可见，暗黑奇幻/ARPG 立绘感",
        "prompt": "半写实数字厚涂风格，可见笔触，丰富中间色调，戏剧化光影，暗黑奇幻游戏原画质感",
        "promptEn": "semi-realistic digital painting style, visible brush strokes, rich midtones, dramatic lighting, dark fantasy game key art quality",
        "negative": "anime, flat, cel shaded, photo",
    },
    {
        "id": "realistic",
        "name": "写实 3A 风",
        "en": "AAA realistic",
        "tags": ["角色", "场景"],
        "hint": "3A 写实渲染方向，PBR 材质描述，概念设计稿",
        "prompt": "写实3A游戏概念设计，照片级材质细节，次表面散射皮肤，体积光，电影级构图，虚幻引擎渲染质感",
        "promptEn": "AAA game concept art, photorealistic material detail, subsurface scattering skin, volumetric lighting, cinematic composition, unreal engine render quality",
        "negative": "anime, cartoon, flat, low quality",
    },
    {
        "id": "lowpoly",
        "name": "低多边形 3D",
        "en": "low-poly 3D",
        "tags": ["场景", "道具"],
        "hint": "低面数 3D，等距视角友好，独立游戏场景常用",
        "prompt": "低多边形3D风格，少面数几何体，平面着色，等距视角，柔和 pastel 配色，独立游戏场景",
        "promptEn": "low-poly 3D style, faceted low-count geometry, flat shading, isometric perspective, soft pastel palette, indie game environment",
        "negative": "high poly, realistic, detailed texture",
    },
    {
        "id": "voxel",
        "name": "体素风",
        "en": "voxel art",
        "tags": ["角色", "场景"],
        "hint": "MagicaVoxel 立方体风（Crossy Road 感），等距场景出彩",
        "prompt": "体素艺术风格，均匀立方体单元，等距视角，明亮配色，3D像素方块构成，干净边缘",
        "promptEn": "voxel art style, uniform cubic units, isometric view, bright color palette, built from 3D pixel cubes, clean edges",
        "negative": "smooth surfaces, organic curves, realistic",
    },
    # ---------------- 手绘 ----------------
    {
        "id": "watercolor",
        "name": "水彩手绘",
        "en": "watercolor",
        "tags": ["场景", "UI"],
        "hint": "透明水彩层叠，纸纹留白，绘本/治愈系游戏感",
        "prompt": "水彩手绘风格，透明颜料层叠，湿画法晕染，纸纹与留白，柔和边缘，绘本插画感",
        "promptEn": "watercolor hand-painted style, translucent layered pigment, wet-on-wet bleeding, paper texture and white space, soft edges, storybook illustration feel",
        "negative": "digital, hard edges, 3d render, neon",
    },
    {
        "id": "concept_sketch",
        "name": "概念速写",
        "en": "concept sketch",
        "tags": ["角色", "场景"],
        "hint": "铅笔/炭笔速写，设计稿阶段用，线稿+灰调",
        "prompt": "概念设计速写风格，铅笔线稿，轻灰调，构图探索感，设计稿质感，未上色或局部上色",
        "promptEn": "concept design sketch style, pencil line art, light gray tones, exploratory composition, design draft feel, mostly uncolored",
        "negative": "full color, polished render, photo",
    },
]

_BY_ID: Dict[str, Dict[str, Any]] = {s["id"]: s for s in STYLES}


def list_styles() -> List[Dict[str, Any]]:
    """画风列表（去掉大字段，给前端 chips 用）。"""
    return [{"id": s["id"], "name": s["name"], "en": s["en"],
             "tags": s["tags"], "hint": s["hint"],
             "pixelize": s.get("pixelize")} for s in STYLES]


def get_style(style_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return _BY_ID.get(style_id) if style_id else None


def apply_style(prompt: str, style_id: Optional[str], chinese_model: bool) -> Dict[str, Any]:
    """把画风词拼到提示词尾部，返回 {prompt, negative, style, suggestPost}。

    - chinese_model=True（Z-Image / Qwen-Image-Edit）拼中文风格词；
    - False（SD 系 CLIP）拼英文风格词，避免机翻把风格词毁掉；
    - 风格词放尾部：SD 的注意力机制对尾部权重自然衰减，风格词放
      开头反而容易被主体描述淹没。
    """
    style = get_style(style_id)
    if not style:
        return {"prompt": prompt, "negative": None, "style": None, "suggestPost": None}
    words = style["prompt"] if chinese_model else style["promptEn"]
    merged = prompt.rstrip("。.，, ") + "。" + words if prompt.strip() else words
    return {
        "prompt": merged,
        "negative": style.get("negative"),
        "style": {"id": style["id"], "name": style["name"], "en": style["en"]},
        "suggestPost": style.get("pixelize"),
    }
