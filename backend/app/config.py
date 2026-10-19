"""运行时配置：持久化到 data/settings.json（本地优先，不上云）。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ASSETS_DIR = DATA_DIR / "assets"
# 画风示例图（画风 chip 悬浮 tooltip 用），由 backend/gen_style_samples.py 生成
STYLE_SAMPLES_DIR = DATA_DIR / "style_samples"
# 人物像素化三档示例图（档位卡片用），由 backend/gen_char_pixel_samples.py 生成
CHAR_PIXEL_SAMPLES_DIR = DATA_DIR / "char_pixel_samples"
SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    # 只有本地双底座：Z-Image-Turbo（文生图）与 Qwen-Image-Edit-2511（指令编辑）
    "activeProvider": "diffusers",
    "providers": {
        "diffusers": {
            "label": "本地双底座（Z-Image / Qwen-Edit）",
            "baseUrl": "",
            "apiKey": "",
            # 生成底座（文生图）；编辑底座由 ai_edit.resolve_edit_model 解析，
            # 优先扫设置里的 qwen_edit 族模型，否则用 models/qwen-image-edit-2511
            "model": "models/z-image-turbo",
            "negativePrompt": "模糊, 低质量, 水印, 文字",
            "enabled": True,
        },
    },
    "defaults": {"steps": 28, "cfg": 7.5, "width": 512, "height": 512},
    # 默认不烧 AI 角标：角标一旦画进像素，后续抠图/合成会把它带进精灵表。
    # 需要合规标识时，下载端点可用 ?label=1 显式补标。
    "aiLabel": False,
}


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_path(p: str) -> str:
    """把设置里的相对路径按项目根目录解析。

    模型路径（如 "models/z-image-turbo"）存进 settings.json 后， historically
    依赖「从项目根目录启动」才找得到；换成别的目录启动 uvicorn 就会静默
    找不到模型。这里兜底：绝对路径 / 已存在的路径原样返回，否则按 ROOT
    拼一次，拼出来存在就用，不存在也返回原值（让调用方报它自己的错）。
    """
    p = (p or "").strip()
    if not p or os.path.isabs(p) or Path(p).exists():
        return p
    cand = ROOT / p
    return str(cand) if cand.exists() else p


def load_settings() -> Dict[str, Any]:
    ensure_dirs()
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        merged = {**DEFAULT_SETTINGS, **data}
        merged["providers"] = {**DEFAULT_SETTINGS["providers"], **data.get("providers", {})}
        merged["defaults"] = {**DEFAULT_SETTINGS["defaults"], **data.get("defaults", {})}
        return merged
    return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    ensure_dirs()
    current = load_settings()
    merged = {**current, **payload}
    merged["providers"] = {**current["providers"], **payload.get("providers", {})}
    merged["defaults"] = {**current["defaults"], **payload.get("defaults", {})}
    # allow_nan=False：NaN 写坏不如写炸（同 storage._save_index 的原则）
    SETTINGS_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return merged
