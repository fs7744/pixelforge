"""生成侧底色：AI 画什么底，抠图就从什么底入手。

这一层存在的理由是一次实测（640×896、同一提示词同一 seed，只改背景色）：

| 底色 | 模型实际画出 | 内圈偏底色像素 | 棋子残留 | 接地阴影 |
|---|---|---|---|---|
| 绿幕 #00FF00 | **[90,199,126] 薄荷绿 + 径向渐变** | **76.5%** | 有 | 有 |
| 中性灰 #808080 | [178,178,177] 极均匀 | **33.3%** | 无 | 有 |
| 白 #FFFFFF | [255,254,253] 极均匀 | 26.4% | 无 | 有 |

两个结论，都与直觉相反：

1. **模型画不出纯色底。** 声明 #00FF00，它给你的是薄荷绿 `[90,199,126]`，
   还带中心亮四周暗的径向渐变。色键的固定容差在这种底上必然留下一圈脏边——
   这不是容差调小了，是根本没有一个"纯"可以对齐。
2. **绿幕之所以脏，主因不是"难抠"，是模型主动把绿染进角色。** 绿色高饱和、
   与角色配色冲突，模型为了画面协调会把它带到人物身上：同位置绿底图均值色
   `[91,121,80]`（偏绿），灰底图 `[156,125,124]`（正常）。色键调参对此无能为力
   ——那些像素本身就是错的，抠掉反而是在毁图。

所以默认底色是**中性灰**而不是绿幕：中性色不参与角色配色，不会被"和谐化"
到人物身上；而抠图本来就该走语义分割（u2net），它根本不看背景是什么颜色。

绿幕保留为一个选项，因为它是行业惯例、且用户有时要在引擎里自己键控——
那时我们本地合成的纯 #00FF00（不是模型画的）比模型画的绿底干净得多。

接地阴影是三者都有、且色键必残留的问题（阴影比底色暗，色键判成前景）。
语义分割能正确忽略它。
"""
from typing import Any, Dict, List

BACKDROPS: List[Dict[str, Any]] = [
    {
        "id": "neutral",
        "name": "中性灰底（推荐）",
        "hint": "抠图最干净的底色：灰色不参与角色配色，模型不会把底色染到人物身上。"
                "走语义分割抠图，不受背景色影响",
        "hex": "#808080",
        "prompt": "背景为纯中性灰色 #808080 满铺，颜色均匀、无渐变、无纹理、"
                  "无投影、无接地阴影",
        "chromaKeyable": False,
    },
    {
        "id": "green",
        "name": "绿幕 #00FF00",
        "hint": "行业惯例色，便于在引擎里自己键控。⚠ 模型画不出纯绿（实测是薄荷绿"
                "[90,199,126] 且带渐变），且会把绿染到角色边缘——本地抠图走语义"
                "分割可缓解；要绝对纯的绿幕请用「人物像素化」，它的绿幕是我们本地"
                "合成的",
        "hex": "#00FF00",
        "prompt": "背景为纯绿色 #00FF00 满铺，颜色均匀、无渐变、无纹理、"
                  "无投影、无接地阴影",
        "chromaKeyable": True,
    },
    {
        "id": "white",
        "name": "白底",
        "hint": "白色同样不会被染到角色上，但浅色角色（白衣、白发）容易与底色混在一起",
        "hex": "#FFFFFF",
        "prompt": "背景为纯白色 #FFFFFF 满铺，颜色均匀、无渐变、无纹理、"
                  "无投影、无接地阴影",
        "chromaKeyable": False,
    },
]

_BY_ID = {b["id"]: b for b in BACKDROPS}
DEFAULT = "neutral"


def list_backdrops() -> List[Dict[str, Any]]:
    return [
        {k: v for k, v in b.items() if k != "prompt"} for b in BACKDROPS
    ]


def get_backdrop(backdrop_id: str | None) -> Dict[str, Any]:
    """取底色定义；未知 id 报错并列出合法值（别静默退回默认，用户会以为改了）。"""
    key = (backdrop_id or DEFAULT).strip()
    if key not in _BY_ID:
        raise ValueError(
            f"未知底色: {backdrop_id}。可选: {', '.join(_BY_ID)}"
        )
    return _BY_ID[key]


def backdrop_prompt(backdrop_id: str | None) -> str:
    return get_backdrop(backdrop_id)["prompt"]


def backdrop_rgb(backdrop_id: str | None) -> tuple:
    h = get_backdrop(backdrop_id)["hex"]
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))