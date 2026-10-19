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


def list_assets(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    items = _load_index()
    if kind:
        items = [i for i in items if i.get("kind") == kind]
    return items


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
