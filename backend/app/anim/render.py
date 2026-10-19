"""渲染：2D FK 骨骼 + 逐轨关键帧 + 3×3 自由变形 + 多轨合成 + 笔迹。

为什么用矩阵而不是「逐轨算角度」：骨骼是树，父骨骼一转，子骨骼的位置和
朝向都要跟着变；用 3×3 仿射矩阵做级联，父子关系就只是矩阵相乘，不需要
在每个功能里各写一遍三角函数（那种写法在「父骨骼带缩放」时必然出错）。

渲染是纯 CPU 的 PIL/OpenCV 操作：一张 512²、十来条轨道的帧在十几毫秒级，
所以播放时每次改参数都能立刻重画，不需要缓存失效策略。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import fx as fx_mod

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ---------- 仿射矩阵（3×3，行向量右乘） ----------
def _T(x: float, y: float) -> np.ndarray:
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float64)


def _R(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _S(k: float) -> np.ndarray:
    return np.array([[k, 0, 0], [0, k, 0], [0, 0, 1]], dtype=np.float64)


def _ease(t: float, kind: str) -> float:
    t = max(0.0, min(1.0, t))
    if kind == "ease-in":
        return t * t
    if kind == "ease-out":
        return 1 - (1 - t) * (1 - t)
    return t


def _sample(keys: Sequence[Dict[str, Any]], frame: int) -> Dict[str, float]:
    """关键帧采样：线性 + 缓动；首帧前/末帧后保持端点值（不外推）。"""
    if not keys:
        return {"x": 0.0, "y": 0.0, "rot": 0.0, "scale": 1.0, "opacity": 1.0}
    ks = sorted(keys, key=lambda k: k["frame"])
    if frame <= ks[0]["frame"]:
        return {k: float(ks[0].get(k, 0.0)) for k in ("x", "y", "rot")} | {
            "scale": float(ks[0].get("scale", 1.0)), "opacity": float(ks[0].get("opacity", 1.0))}
    if frame >= ks[-1]["frame"]:
        return {k: float(ks[-1].get(k, 0.0)) for k in ("x", "y", "rot")} | {
            "scale": float(ks[-1].get("scale", 1.0)), "opacity": float(ks[-1].get("opacity", 1.0))}
    for a, b in zip(ks, ks[1:]):
        if a["frame"] <= frame <= b["frame"]:
            span = b["frame"] - a["frame"]
            t = 0.0 if span == 0 else (frame - a["frame"]) / span
            # 缓动取**后一帧**的 ease：表示「走到这一帧时怎么减速」
            t = _ease(t, b.get("ease") or "linear")
            out = {}
            for k, d in (("x", 0.0), ("y", 0.0), ("rot", 0.0)):
                out[k] = float(a.get(k, d)) + (float(b.get(k, d)) - float(a.get(k, d))) * t
            out["scale"] = float(a.get("scale", 1.0)) + (float(b.get("scale", 1.0)) - float(a.get("scale", 1.0))) * t
            out["opacity"] = float(a.get("opacity", 1.0)) + (float(b.get("opacity", 1.0)) - float(a.get("opacity", 1.0))) * t
            return out
    return {"x": 0.0, "y": 0.0, "rot": 0.0, "scale": 1.0, "opacity": 1.0}


def bone_world(proj: Dict[str, Any], frame: int) -> Dict[str, np.ndarray]:
    """每根骨骼在该帧的世界矩阵（含父级级联）。"""
    bones = {b["id"]: b for b in proj.get("bones") or []}
    out: Dict[str, np.ndarray] = {}
    for bid, b in bones.items():
        rest = _T(b["x"], b["y"])
        k = _sample(b.get("keys") or [], frame)
        local = _T(k["x"], k["y"]) @ _R(k["rot"]) @ _S(k["scale"])
        parent = b.get("parent")
        if parent and parent in out:
            pb = bones[parent]
            # 子骨骼的静止位是绝对坐标：换算成相对父骨骼的偏移再接矩阵链
            off = _T(b["x"] - pb["x"], b["y"] - pb["y"])
            out[bid] = out[parent] @ off @ local
        else:
            out[bid] = rest @ local
    return out


# ---------- 自由变形 ----------
def warp_image(img: Image.Image, ctrl: Sequence[Sequence[float]]) -> Image.Image:
    """3×3 控制点做双线性自由变形（控制点是**像素偏移**，不是绝对坐标）。"""
    if not ctrl or len(ctrl) != 9 or all(abs(p[0]) < 0.01 and abs(p[1]) < 0.01 for p in ctrl):
        return img
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("自由变形需要 opencv-python：pip install opencv-python")
    w, h = img.size
    arr = np.asarray(img.convert("RGBA"))
    dx = np.array([[p[0] for p in ctrl[i * 3:(i + 1) * 3]] for i in range(3)], dtype=np.float32)
    dy = np.array([[p[1] for p in ctrl[i * 3:(i + 1) * 3]] for i in range(3)], dtype=np.float32)
    dx_full = cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR)
    dy_full = cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR)
    base_x, base_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    # 控制点是「把这一点的内容推向哪」：out(x) = src(x - dx)。
    # 写成 +dx 会变成内容朝反方向跑，用户往右拖、图往左走，手感直接错。
    out = cv2.remap(arr, base_x - dx_full, base_y - dy_full, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return Image.fromarray(out, "RGBA")


def _apply_matrix(img: Image.Image, m: np.ndarray, canvas: Tuple[int, int]) -> Image.Image:
    """按 src→canvas 的矩阵把图放到画布上（PIL 需要的是逆矩阵）。"""
    inv = np.linalg.inv(m)
    coeffs = tuple(inv[:2].reshape(-1).tolist())
    return img.transform(canvas, Image.AFFINE, coeffs, resample=Image.BICUBIC)


def _blend(base: Image.Image, layer: Image.Image, mode: str,
           opacity: float) -> Image.Image:
    if opacity <= 0:
        return base
    if opacity < 1:
        a = layer.getchannel("A").point(lambda v: int(v * opacity))
        layer = layer.copy()
        layer.putalpha(a)
    if mode == "normal":
        return Image.alpha_composite(base, layer)
    b = np.asarray(base).astype(np.float32) / 255.0
    l = np.asarray(layer).astype(np.float32) / 255.0
    la = l[..., 3:4]
    if mode == "add":
        rgb = np.clip(b[..., :3] + l[..., :3] * la, 0, 1)
    else:  # multiply：只压暗有内容的地方
        rgb = b[..., :3] * (1 - la) + b[..., :3] * l[..., :3] * la
    out = np.concatenate([rgb, b[..., 3:4]], axis=2)
    return Image.fromarray((out * 255).astype(np.uint8), "RGBA")


def render_frame(proj: Dict[str, Any], frame: int,
                 loader) -> Image.Image:
    """渲染单帧。loader(asset_id) -> PIL.Image（由调用方决定怎么取图）。"""
    cw, ch = proj["canvas"]
    base = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    worlds = bone_world(proj, frame)
    iw = np.eye(3)

    for t in proj.get("tracks") or []:
        if not t.get("visible", True):
            continue
        src = loader(t["assetId"])
        if src is None:
            continue
        src = src.convert("RGBA")
        src = warp_image(src, t.get("warp") or [])
        k = _sample(t.get("keyframes") or [], frame)
        px, py = t.get("pivot") or [src.width / 2, src.height / 2]
        world = worlds.get(t.get("bone") or "root", iw)
        # pivot 的语义是「贴图上这一点落在骨骼原点上」：先把它平移到原点，
        # 再旋转/缩放，最后按关键帧位移。写成 T(p)…T(-p) 的话，rotation=0、
        # scale=1 时两段平移互相抵消，pivot 就完全失效了（贴图左上角贴到骨骼）。
        m = (world @ _T(k["x"], k["y"]) @ _R(k["rot"])
             @ _S(k["scale"]) @ _T(-px, -py))
        placed = _apply_matrix(src, m, (cw, ch))
        base = _blend(base, placed, t.get("blend") or "normal",
                      float(t.get("opacity", 1.0)) * k["opacity"])
        if t.get("fx"):
            strokes = [s for s in t["fx"] if s.get("from", 0) <= frame <= s.get("to", frame)]
            if strokes:
                base = Image.alpha_composite(base, fx_mod.render(strokes, (cw, ch), frame))
    return base


def render_frames(proj: Dict[str, Any], frames: Optional[Sequence[int]] = None,
                  loader=None, scale: float = 1.0) -> List[Tuple[int, Image.Image]]:
    if loader is None:  # pragma: no cover - 正常由调用方注入
        raise ValueError("render_frames 需要 loader")
    idx = list(frames) if frames is not None else list(range(int(proj["frames"])))
    out: List[Tuple[int, Image.Image]] = []
    for i in idx:
        img = render_frame(proj, i, loader)
        if scale and abs(scale - 1.0) > 1e-3:
            img = img.resize((max(1, round(img.width * scale)),
                              max(1, round(img.height * scale))), Image.NEAREST)
        out.append((i, img))
    return out


def sheet(images: Sequence[Image.Image], cols: int = 4, pad: int = 2,
          bg=(24, 24, 28, 255)) -> Image.Image:
    """预览用拼图（导出 ZIP 里附一张，比逐个点开快得多）。"""
    if not images:
        return Image.new("RGBA", (8, 8), bg)
    w, h = images[0].size
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGBA", (cols * (w + pad) + pad, rows * (h + pad) + pad), bg)
    for i, im in enumerate(images):
        c, r = i % cols, i // cols
        out.alpha_composite(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return out
