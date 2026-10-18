"""运行时配置：持久化到 data/settings.json（本地优先，不上云）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ASSETS_DIR = DATA_DIR / "assets"
SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "activeProvider": "mock",
    "providers": {
        "mock": {
            "label": "内置模拟引擎",
            "baseUrl": "",
            "apiKey": "",
            "model": "mock-pixel-v1",
            "enabled": True,
        },
        "diffusers": {
            "label": "内置推理服务（diffusers）",
            "baseUrl": "",
            "apiKey": "",
            "model": "models/z-image-turbo",
            "negativePrompt": "blurry, low quality, watermark, text",
            # 一致性增强（留空则自动探测 models/ 下的默认目录；仅 SD1.5 族有效，
            # Z-Image 的 UNet 结构不同，挂了也会在 caps.notes 里说明不适用）
            "ipAdapter": "",
            "ipAdapterScale": 0.7,
            "lora": "",
            "loraScale": 0.85,
            "lcmLora": "",
            "refMode": "ip",
            "enabled": True,
        },
        "openai_compat": {
            "label": "云端 API（OpenAI 兼容）",
            "baseUrl": "https://api.openai.com/v1",
            "apiKey": "",
            "model": "gpt-image-1",
            "enabled": False,
        },
        "a1111": {
            "label": "本地 A1111 / Forge",
            "baseUrl": "http://127.0.0.1:7860",
            "apiKey": "",
            "model": "",
            "enabled": False,
        },
    },
    "defaults": {"steps": 28, "cfg": 7.5, "width": 512, "height": 512},
    "aiLabel": True,
}


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


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
    SETTINGS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged
