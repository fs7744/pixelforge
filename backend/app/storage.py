"""本地资产库：图片落盘 + index.json 元数据（不上云）。"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from .config import ASSETS_DIR, ensure_dirs

INDEX_PATH = ASSETS_DIR / "index.json"

# 进程内索引缓存：index.json 已到 263KB/513 条，每次保存都全量读盘解析
# 再全量重写，工作流一步存 16 帧就是 16 轮全量 IO。缓存后保存仍是全量
# 原子写（正确性优先），但读路径不再碰盘。
# 前提：单进程部署（start.sh 默认单 worker uvicorn）。多 worker 时各进程
# 缓存会互相看不到，届时需要换回每次读盘或改存储后端。
_INDEX_CACHE: Optional[List[Dict[str, Any]]] = None

# ---------- 分类标签 ----------
# 中文名只在这里维护一份，前端拿 facets 直接渲染。
# 之前的教训：Library.tsx 里手写了 6 个 kind 的 chips，而实际写入的 kind 有 12 种，
# 于是「像素化 / 工作流 / 水印 / 换画风 / 拼图」这些资产在库里根本选不出来。
ALL = "全部"

KIND_LABELS: Dict[str, str] = {
    "generated": "生成",
    "threeview": "三视图",
    "workflow": "工作流",
    "montage": "拼图",
    "matte": "抠图",
    "resized": "规整",
    "refnorm": "参考图规范化",
    "pixelized": "像素化",
    "seamless": "无缝",
    "tile": "瓦片",
    "watermark": "水印",
    "restyled": "换画风",
    "charpixel": "像素角色",
    "frame": "帧",
    "sheet": "精灵表",
    "layer": "图层",
}

# 管线标签：三视图这类「同 kind 不同来源」的东西只能靠它区分——
# 正/侧/背/拼板全部落在 kind=generated 里，不按 pipeline 筛就永远混在生成里。
PIPELINE_LABELS: Dict[str, str] = {
    "threeview": "三视图",
    "sheet-split": "精灵表拆分",
    "sheet-compose": "精灵表合成",
    "pixelize": "像素化",
    "resize": "规整",
    "refnorm": "参考图规范化",
    "matte": "抠图",
    "tile": "瓦片",
    "seamless": "无缝",
    "seamless_ai": "无缝 · AI",
    "restyle": "换画风",
    "char-pixel": "人物像素化",
    "watermark": "水印",
    "watermark_ai": "水印 · AI",
    "scene-layers": "场景分层",
}

# meta 里没有 pipeline 的记录（/api/generate 直接出图）统一归到这个虚拟管线，
# 否则前端会收到一个 id 为 None 的 chip，点了必然 500。
DIRECT = "direct"
PIPELINE_LABELS[DIRECT] = "直接生成"

_KIND_ORDER = list(KIND_LABELS)
_PIPE_ORDER = list(PIPELINE_LABELS)


def asset_pipeline(item: Dict[str, Any]) -> str:
    """资产的来源管线；没有就归到 direct。"""
    return ((item.get("meta") or {}).get("pipeline")) or DIRECT


def _load_index() -> List[Dict[str, Any]]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        ensure_dirs()
        if INDEX_PATH.exists():
            try:
                _INDEX_CACHE = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                _INDEX_CACHE = []
        else:
            _INDEX_CACHE = []
    # 防御性拷贝：调用方的增删改不直接污染缓存，缓存只由 _save_index 推进
    return list(_INDEX_CACHE)


def _save_index(items: List[Dict[str, Any]]) -> None:
    global _INDEX_CACHE
    # 先写临时文件再原子替换：直接覆盖时若进程中断，index.json 会变成
    # 半截 JSON，_load_index 只能返回空列表，整个资产库等于丢失。
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    # allow_nan=False：NaN/Infinity 混进 meta 时直接炸在这里，而不是写出
    # 严格 JSON 解析器拒绝的索引文件，留着一颗雷等下次读取才爆
    # （h3-character-studio 的 write_json 同款原则：写坏不如写炸）
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(INDEX_PATH)
    _INDEX_CACHE = list(items)


def _digest_file(path: Path) -> str:
    """文件 sha256（移植自 h3-character-studio 的 core.digest）。

    用途：资产完整性校验（磁盘损坏时能发现）、导出包溯源、将来做
    「同内容去重」。分块读，大文件不吃内存。
    """
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp_ai_label(img: Image.Image, text: str = "AI") -> Image.Image:
    """显式 AI 标识：左下角小角标。

    默认不烧进存储——角标一旦画进像素，后续抠图 / 精灵表 / 拼板会把它
    当成画面内容一路带进最终素材。需要合规标识时走
    GET /api/assets/{id}/download?label=1，下载时才补。
    """
    out = img.convert("RGBA").copy()
    d = ImageDraw.Draw(out)
    pad = 4
    w, h = 26, 14
    x, y = pad, out.height - h - pad
    d.rectangle([x, y, x + w, y + h], fill=(10, 10, 10, 210))
    d.text((x + 5, y + 3), text, fill=(0, 255, 136, 255))
    return out


def save_asset(
    img: Image.Image,
    kind: str = "generated",
    name: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    # 默认不烧角标：存储阶段一旦烧进去，抠图/合成/拼板都会把它当内容带走。
    ai_label: bool = False,
) -> Dict[str, Any]:
    ensure_dirs()
    asset_id = uuid.uuid4().hex[:12]
    if ai_label:
        img = stamp_ai_label(img)

    filename = f"{asset_id}.png"
    path = ASSETS_DIR / filename
    img.save(path, "PNG")

    item = {
        "id": asset_id,
        "name": name or f"{kind}-{asset_id[:6]}",
        "kind": kind,
        "file": filename,
        "url": f"/files/{filename}",
        "width": img.width,
        "height": img.height,
        "createdAt": int(time.time()),
        "sha256": _digest_file(path),
        "meta": meta or {},
    }
    items = _load_index()
    items.insert(0, item)
    try:
        _save_index(items)
    except Exception:
        # 索引写失败（NaN、磁盘满）时把刚写的 PNG 一起收走，不留
        # 「磁盘有文件、索引无记录」的孤儿——导出和缩略图都会漏掉它
        path.unlink(missing_ok=True)
        raise
    return item


def list_assets(kind: Optional[str] = None, pipeline: Optional[str] = None) -> List[Dict[str, Any]]:
    """按 kind / pipeline 过滤，两个条件是 AND。

    "全部" 视同不过滤——前端 chip 状态直接透传，不用在两边各写一次判断。
    """
    items = _load_index()
    if kind and kind != ALL:
        items = [i for i in items if i.get("kind") == kind]
    if pipeline and pipeline != ALL:
        items = [i for i in items if asset_pipeline(i) == pipeline]
    return items


def asset_facets() -> Dict[str, Any]:
    """筛选项从索引实时派生，而不是前端写死。

    写死过一次的代价：新增 6 种 kind 后界面只认旧的 6 种，资产在库里「看不见」。
    所以这里按实际存在的值聚合，出现什么就能筛什么；标签表里没有的也用原值兜住。
    """
    from collections import Counter

    items = _load_index()
    kinds = Counter(i.get("kind") or "generated" for i in items)
    pipes = Counter(asset_pipeline(i) for i in items)

    def _rows(counter, order, labels):
        def key(kv):
            name = kv[0]
            pos = order.index(name) if name in order else len(order)
            return (pos, -kv[1], name)

        return [
            {"id": k, "label": labels.get(k, k), "count": c}
            for k, c in sorted(counter.items(), key=key)
        ]

    return {
        "total": len(items),
        "kinds": _rows(kinds, _KIND_ORDER, KIND_LABELS),
        "pipelines": _rows(pipes, _PIPE_ORDER, PIPELINE_LABELS),
    }


def get_asset(asset_id: str) -> Optional[Dict[str, Any]]:
    for item in _load_index():
        if item["id"] == asset_id:
            return item
    return None


def asset_path(item: Dict[str, Any]) -> Path:
    # 纵深防御（移植自 h3-character-studio 的 core.contained）：file 来自
    # index.json（本模块写入、uuid 命名），理论上不可能穿出 assets 目录；
    # 索引被手工改坏时在这里挡下，而不是把挂载点之外的文件读出去/删掉。
    p = (ASSETS_DIR / item["file"]).resolve()
    if not p.is_relative_to(ASSETS_DIR.resolve()):
        raise ValueError(f"资产路径越界：{item['file']}")
    return p


THUMB_DIR = ASSETS_DIR / "thumbs"

# 缩略图宽度做档位而不是任意值：前端只用 96/160/320 三档，
# 否则每个宽高比 × 任意宽都会生成一份缓存，thumbs 目录很快就失控。
THUMB_WIDTHS = (96, 160, 320)


def thumb_path(item: Dict[str, Any], width: int) -> Path:
    return THUMB_DIR / f"{item['id']}-{width}.png"


def ensure_thumb(item: Dict[str, Any], width: int = 160) -> Path:
    """按宽度取缩略图，没有就地生成并缓存。

    为什么需要：资产库和图生图的取图区一次要列上百张，直接铺原图意味着
    几百 MB 的 PNG 全下到浏览器；而且三视图拼板是 1560×512，用原图当缩略图
    既慢又浪费。缩略图保持宽高比（thumbnail 按长边等比缩），所以
    「宽条 = 三视图拼板」在列图里一眼就能看出来，不会被裁成方图。
    """
    width = min(THUMB_WIDTHS, key=lambda w: abs(w - int(width)))
    p = thumb_path(item, width)
    if p.is_file():
        return p

    src = asset_path(item)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGBA")
    img.thumbnail((width, width), Image.LANCZOS)
    img.save(p, "PNG", optimize=True)
    return p


def delete_asset(asset_id: str) -> bool:
    items = _load_index()
    keep = []
    removed = False
    for item in items:
        if item["id"] == asset_id:
            removed = True
            try:
                (ASSETS_DIR / item["file"]).unlink(missing_ok=True)
            except Exception:
                pass
            # 缩略图缓存跟着删，否则同一个 id 被复用时会拿到旧图
            for p in THUMB_DIR.glob(f"{asset_id}-*.png"):
                p.unlink(missing_ok=True)
        else:
            keep.append(item)
    if removed:
        _save_index(keep)
    return removed
