"""本地神经机器翻译：中文 → 英文（Helsinki opus-mt-zh-en / MarianMT）。

词典翻译（zh2en）在领域词覆盖率高时最准，但遇到长句、生僻描写或整段叙述
就会漏译。这里用 ~300MB 的 Marian 模型兜底：整句翻译，再由领域术语表做
英文校正，保证「Q版 / 像素风 / 粗描边 / 武士刀」这类关键词不会译丢。

模型文件放 models/opus-mt-zh-en/（见 backend/fetch_support.py）。
首次调用才加载，避免拖慢服务启动。

注意：当前环境 transformers 为 v5，Auto 类映射已不认 `model_type: marian`，
所以这里直接用显式 MarianTokenizer / MarianMTModel，绕开 auto 解析。
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, Optional

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "opus-mt-zh-en",
)

# 机翻常见错译 → 游戏素材领域标准说法（大小写不敏感匹配）
GLOSSARY = [
    ("Q version", "chibi"),
    ("Q-version", "chibi"),
    ("q version", "chibi"),
    ("Q editor", "chibi"),
    ("Q-editor", "chibi"),
    ("Q edition", "chibi"),
    ("cartoon version", "chibi"),
    ("cartoon style", "chibi"),
    ("pixel style", "pixel art style"),
    ("pixel wind", "pixel art style"),
    ("pixel art wind", "pixel art style"),
    ("thick outline", "thick black outline"),
    ("black edge", "black outline"),
    ("black line", "black outline"),
    ("plain color background", "plain solid background"),
    ("solid color background", "plain solid background"),
    ("pure color background", "plain solid background"),
    ("pure green background", "plain solid green background"),
    ("three views", "character turnaround sheet"),
    ("three-view drawing", "character turnaround sheet"),
    ("three view", "character turnaround sheet"),
    ("side view", "side view"),
    ("Samurai knife", "katana"),
    ("samurai knife", "katana"),
    ("samurai sword", "katana"),
    ("cloak", "cape"),
    ("mantle", "cape"),
    ("moonlit night", "moonlit night"),
]

_LOCK = threading.Lock()
_PIPE = None
_ERROR: Optional[str] = None


def available() -> bool:
    return os.path.exists(os.path.join(_MODEL_DIR, "pytorch_model.bin")) and \
        os.path.exists(os.path.join(_MODEL_DIR, "vocab.json"))


def _load():
    global _PIPE, _ERROR
    with _LOCK:
        if _PIPE is not None or _ERROR is not None:
            return _PIPE
        try:
            import torch
            from transformers import MarianTokenizer, MarianMTModel

            tok = MarianTokenizer.from_pretrained(_MODEL_DIR)
            model = MarianMTModel.from_pretrained(
                _MODEL_DIR, torch_dtype=torch.float32)
            _PIPE = (tok, model)
        except Exception as e:  # noqa: BLE001
            _ERROR = f"{type(e).__name__}: {e}"[:240]
            return None
        return _PIPE


def apply_glossary(text: str) -> str:
    """把机翻常见错译替换成游戏素材领域标准说法（大小写不敏感）。"""
    out = text
    for wrong, right in GLOSSARY:
        # 大小写不敏感整词替换，避免误伤子串
        out = re.sub(re.escape(wrong), lambda m: right, out, flags=re.IGNORECASE)
    return out


def translate(text: str, max_len: int = 256) -> Dict[str, Any]:
    """中文 → 英文整句翻译。返回 {english, raw, engine, ok, error?}。"""
    if not (text or "").strip():
        return {"english": "", "engine": "none", "ok": False}
    if not available():
        return {"english": "", "engine": "unavailable", "ok": False}
    pipe = _load()
    if pipe is None:
        return {"english": "", "engine": "error", "ok": False, "error": _ERROR}
    tok, model = pipe
    import torch

    inputs = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_len, num_beams=2)
    raw = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    english = apply_glossary(raw)
    return {"english": english, "raw": raw, "engine": "opus-mt+glossary",
            "ok": bool(english)}


def status() -> Dict[str, Any]:
    return {"available": available(), "dir": _MODEL_DIR, "loaded": _PIPE is not None,
            "error": _ERROR}
