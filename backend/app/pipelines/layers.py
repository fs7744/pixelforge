"""场景语义分层：一张场景图 → N 张 RGBA 图层（移植自 FrameBaker 的 imageLayers 流程）。

为什么需要：视差工作流要求「同尺寸、同坐标、可重合」的多层。原先的做法是把
同一个场景**生成 N 遍**——每遍只保留一层、其余抹成纯色键（见 workflows.py
里那句「MVP 用独立生成模拟」）。四次独立推理的透视、比例、轮廓都对不上，
叠起来只能靠纯色键硬抠，边缘留一圈杂色，所以后面才要跟「同尺寸规整 + 无缝
延展」去救。语义分层是拆**同一张**图：天然坐标对齐、无接缝、直接带 alpha。

层序：按模型输出原序保存；meta 另给一份 `order`（「远 → 近」），视差导引用它
分配 scroll_scale。排序是启发式而不是保证：按主体底边高度（横版 2.5D 的通行
约定，脚踩得越低越近），铺满画布的背景层固定最远。最初按 alpha 覆盖率排，
真实场景立刻证伪——石墙层面积 31% 被排到城堡层（2.8%）后面，远近整个反了。
每层覆盖率与底边也一并给出，前端能一眼看出哪层不对，不要让用户猜。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

MAX_SIDE = 1600          # 输入上限：再大只是白烧显存，分层模型内部按 resolution 分桶
MIN_LAYERS, MAX_LAYERS = 2, 8
RESOLUTIONS = (640, 1024)


def _coverage(img: Image.Image) -> Dict[str, Any]:
    """一层的 alpha 统计：覆盖占比 + 不透明包围盒。

    bbox 用 alpha>8 而不是 >0：分层模型边缘会留极淡的半透明噪点，
    按 0 算包围盒几乎总是整张图，等于没给信息。
    """
    alpha = img.getchannel("A")
    hist = alpha.histogram()
    total = sum(hist) or 1
    solid = sum(hist[9:])
    bbox = alpha.point(lambda v: 255 if v > 8 else 0).getbbox()
    return {
        "coverage": round(solid / total, 4),
        "bbox": list(bbox) if bbox else None,
        "w": img.width,
        "h": img.height,
    }


def _order_back_to_front(stats: Sequence[Dict[str, Any]]) -> List[int]:
    """给出「远 → 近」层序（启发式，实测修正过一次）。

    最初按 alpha 覆盖率降序排，真实场景立刻证伪：石墙层覆盖率 31%、城堡层
    2.8%，于是墙被排到城堡**后面**——在横版视差里远近完全反了。
    面积跟深度没有关系：前景遮挡物往往又大又低。

    改用 2.5D 横版的通行约定：**主体底边越靠上越远**（脚踩得越低越近）。
    铺满整张画布的层（coverage≈1 且 bbox 覆盖全图）固定当最远，否则背景
    会因为底边在画面最底部而被排到最前面。底边相同再按面积降序兜底。
    """
    def key(i: int):
        s = stats[i]
        cov = s["coverage"]
        bbox = s["bbox"]
        if cov >= 0.98 and bbox and bbox[3] - bbox[1] >= 0.95 * max(1, s.get("h", 0)):
            return (-1, 0)
        bottom = bbox[3] if bbox else 0
        return (bottom, -cov)

    return sorted(range(len(stats)), key=key)


async def decompose(
    image: Image.Image,
    *,
    layers: int = 4,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    resolution: int = 640,
    model: Optional[str] = None,
    task_id: Optional[str] = None,
    provider: Any = None,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """拆层主入口：返回（图层列表, meta）。

    provider 可注入：测试用假的 provider 顶掉 57GB 模型，把「拆完之后的
    校验、覆盖率、排序、警告」这些纯逻辑跑起来。
    """
    n = int(layers)
    if not (MIN_LAYERS <= n <= MAX_LAYERS):
        raise ValueError(f"层数需在 {MIN_LAYERS}-{MAX_LAYERS} 之间，收到 {n}")
    res = int(resolution)
    if res not in RESOLUTIONS:
        # 模型只在这两个分桶上训练过；别的值不会崩，但会掉质量且没人验证过
        raise ValueError(f"分辨率只支持 {RESOLUTIONS}（模型按这两个分桶训练），收到 {res}")

    src = image.convert("RGBA")
    if max(src.size) > MAX_SIDE:
        scale = MAX_SIDE / max(src.size)
        src = src.resize((max(1, round(src.width * scale)),
                          max(1, round(src.height * scale))), Image.LANCZOS)

    if provider is None:
        from ..providers.diffusers_local import DiffusersProvider

        provider = DiffusersProvider({"model": model} if model else {})

    images, pmeta = await provider.generate_layers(
        src, model=model, layers=n, steps=steps, resolution=res,
        seed=seed, task_id=task_id,
    )
    if not images:
        raise RuntimeError("分层模型没有返回任何图层")

    # 尺寸必须统一：管线理论上给同尺寸，但下游视差导出按「同坐标」假设拼层，
    # 这里显式对齐一次（以最大层为基准），把假设变成事实。
    bw = max(im.width for im in images)
    bh = max(im.height for im in images)
    fixed: List[Image.Image] = []
    resized = 0
    for im in images:
        if im.size != (bw, bh):
            im = im.resize((bw, bh), Image.LANCZOS)
            resized += 1
        fixed.append(im)

    stats = [_coverage(im) for im in fixed]
    covs = [s["coverage"] for s in stats]
    order = _order_back_to_front(stats)
    warnings: List[str] = []
    empty = [i for i, c in enumerate(covs) if c < 0.005]
    if empty:
        warnings.append(f"第 {'、'.join(str(i + 1) for i in empty)} 层几乎全透明，"
                        "可能是层数给多了，试试减一层")
    if covs and max(covs) < 0.6:
        warnings.append("没有一层铺满画面，背景可能被拆散了，视差叠层会有空洞")
    if resized:
        warnings.append(f"{resized} 层尺寸不一致，已统一到 {bw}×{bh}")

    meta: Dict[str, Any] = {
        "pipeline": "scene-layers",
        "layerCount": len(fixed),
        "size": [bw, bh],
        "coverage": covs,
        "layerStats": stats,
        "order": order,
        "orderNote": "order 为「远→近」启发式：主体底边越靠上越远，铺满画布的背景层固定最远；不是模型保证，导出前请看一眼缩略图",
        "warnings": warnings,
        **pmeta,
    }
    return fixed, meta
