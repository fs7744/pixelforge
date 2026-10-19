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
    "pixelized": "像素化",
    "seamless": "无缝",
    "tile": "瓦片",
    "watermark": "水印",
    "restyled": "换画风",
    "frame": "帧",
    "sheet": "精灵表",
}

# 管线标签：三视图这类「同 kind 不同来源」的东西只能靠它区分——
# 正/侧/背/拼板全部落在 kind=generated 里，不按 pipeline 筛就永远混在生成里。
PIPELINE_LABELS: Dict[str, str] = {
    "threeview": "三视图",
    "sheet-split": "精灵表拆分",
    "sheet-compose": "精灵表合成",
    "pixelize": "像素化",
    "resize": "规整",
    "matte": "抠图",
    "tile": "瓦片",
    "seamless": "无缝",
    "seamless_ai": "无缝 · AI",
    "restyle": "换画风",
    "watermark": "水印",
    "watermark_ai": "水印 · AI",
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
    ensure_dirs()
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_index(items: List[Dict[str, Any]]) -> None:
    # 先写临时文件再原子替换：直接覆盖时若进程中断，index.json 会变成
    # 半截 JSON，_load_index 只能返回空列表，整个资产库等于丢失。
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(INDEX_PATH)


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
        "meta": meta or {},
    }
    items = _load_index()
    items.insert(0, item)
    _save_index(items)
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
    return ASSETS_DIR / item["file"]


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
        else:
            keep.append(item)
    if removed:
        _save_index(keep)
    return removed
