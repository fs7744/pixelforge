"""动画剪辑工程：数据模型与磁盘存储。

这层只负责「工程长什么样」和「怎么落盘」，不碰像素——渲染在 render.py，
笔刷在 fx.py，法线在 normals.py。分开的理由：工程结构要能被前端直接编辑、
被测试直接构造，不该拖着 PIL 一起进来。

坐标约定（全工程统一，避免各处各自解释）：
- 画布坐标：左上原点，单位像素，y 向下。
- 骨骼：x/y 是**静止位**的绝对坐标；子骨骼的静止位也写绝对坐标，
  运行时用「相对父骨骼的偏移」参与 FK（见 render.bone_world）。
- 轨道的 pivot：贴图内的像素坐标，旋转/缩放都绕它。
- 关键帧：frame 为整数帧号；x/y 是**相对绑定骨骼**的平移，rot 为角度，
  scale 为等比缩放，opacity 0-1。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DATA_DIR

ANIM_DIR = DATA_DIR / "anim"
_ID_RE = re.compile(r"^[0-9a-f]{12}$")

MAX_FRAMES = 240
MAX_TRACKS = 32
MAX_BONES = 24

DEFAULT_PROJECT: Dict[str, Any] = {
    "name": "未命名工程",
    "canvas": [512, 512],
    "fps": 12,
    "frames": 12,
    "background": "transparent",
    "bones": [{"id": "root", "name": "根骨骼", "parent": None, "x": 256, "y": 420}],
    "tracks": [],
}

DEFAULT_KEYFRAME = {"frame": 0, "x": 0.0, "y": 0.0, "rot": 0.0,
                    "scale": 1.0, "opacity": 1.0, "ease": "linear"}


def ensure_dir() -> None:
    ANIM_DIR.mkdir(parents=True, exist_ok=True)


def _path(pid: str) -> Path:
    if not _ID_RE.match(pid or ""):
        # 路径穿越防线：id 只允许 12 位 hex（工程 id 由 uuid 截断生成）
        raise ValueError(f"非法工程 id：{pid!r}")
    return ANIM_DIR / f"{pid}.json"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def create(name: str = "", canvas=(512, 512), frames: int = 12) -> Dict[str, Any]:
    ensure_dir()
    proj = json.loads(json.dumps(DEFAULT_PROJECT))   # 深拷贝，别让默认值被子类改坏
    proj["id"] = new_id()
    proj["name"] = (name or "").strip() or "未命名工程"
    proj["canvas"] = [int(canvas[0]), int(canvas[1])]
    proj["frames"] = max(1, min(int(frames), MAX_FRAMES))
    # 根骨骼放到画布中下：默认值里的 256/420 只对 512² 成立，
    # 直接套到 120×120 的画布会把整条轨道甩到画外（渲染出一张空图）
    proj["bones"] = [{"id": "root", "name": "根骨骼", "parent": None,
                      "x": round(proj["canvas"][0] / 2, 2),
                      "y": round(proj["canvas"][1] * 0.85, 2), "keys": []}]
    proj["createdAt"] = int(time.time())
    proj["updatedAt"] = proj["createdAt"]
    save(proj)
    return proj


def list_projects() -> List[Dict[str, Any]]:
    ensure_dir()
    out = []
    for p in sorted(ANIM_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"id": d.get("id") or p.stem, "name": d.get("name") or p.stem,
                    "canvas": d.get("canvas"), "frames": d.get("frames"),
                    "tracks": len(d.get("tracks") or []),
                    "updatedAt": d.get("updatedAt")})
    return out


def get(pid: str) -> Optional[Dict[str, Any]]:
    try:
        p = _path(pid)
    except ValueError:
        return None
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save(proj: Dict[str, Any]) -> Dict[str, Any]:
    ensure_dir()
    proj["updatedAt"] = int(time.time())
    _path(proj["id"]).write_text(
        json.dumps(proj, ensure_ascii=False, indent=1), encoding="utf-8")
    return proj


def delete(pid: str) -> bool:
    try:
        p = _path(pid)
    except ValueError:
        return False
    if p.is_file():
        p.unlink()
        return True
    return False


# ---------- 校验 ----------
def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    # NaN/Inf 会一路污染矩阵，最后在 PIL 里变成一句看不懂的报错
    return f if f == f and abs(f) != float("inf") else default


def normalize(proj: Dict[str, Any]) -> Dict[str, Any]:
    """清洗工程：补齐字段、夹取范围、去重 id、关键帧排序。

    前端拖拽时可能传半截数据（只改了 x），这里保证落盘的永远是完整结构。
    """
    out = json.loads(json.dumps(proj))
    w, h = out.get("canvas") or [512, 512]
    out["canvas"] = [max(16, min(int(w), 4096)), max(16, min(int(h), 4096))]
    out["fps"] = max(1, min(int(out.get("fps") or 12), 60))
    out["frames"] = max(1, min(int(out.get("frames") or 12), MAX_FRAMES))

    bones = []
    seen = set()
    for b in (out.get("bones") or [])[:MAX_BONES]:
        bid = str(b.get("id") or "").strip()
        if not bid or bid in seen:
            continue
        seen.add(bid)
        bones.append({"id": bid, "name": str(b.get("name") or bid),
                      "parent": (str(b["parent"]) if b.get("parent") else None),
                      "x": round(_num(b.get("x"), out["canvas"][0] / 2), 2),
                      "y": round(_num(b.get("y"), out["canvas"][1] / 2), 2),
                      # 骨骼自身也能打关键帧（根骨骼整体位移、手臂摆动）
                      "keys": [{"frame": max(0, min(int(_num(k.get("frame"), 0)), out["frames"] - 1)),
                                "x": round(_num(k.get("x")), 2), "y": round(_num(k.get("y")), 2),
                                "rot": round(_num(k.get("rot")), 2),
                                "scale": max(0.01, round(_num(k.get("scale"), 1.0), 4)),
                                "ease": k.get("ease") if k.get("ease") in
                                ("linear", "ease-in", "ease-out") else "linear"}
                               for k in (b.get("keys") or [])[:MAX_FRAMES]]})
    if not bones:
        bones = [{"id": "root", "name": "根骨骼", "parent": None,
                  "x": out["canvas"][0] / 2, "y": out["canvas"][1] * 0.85}]
    out["bones"] = bones
    # 父骨骼必须存在，且不能形成环（环会让 FK 无限递归）
    ids = {b["id"] for b in bones}
    for b in bones:
        if b["parent"] and b["parent"] not in ids:
            b["parent"] = None
    for b in bones:
        seen_chain = set()
        cur = b
        while cur and cur.get("parent"):
            if cur["id"] in seen_chain:
                cur["parent"] = None
                break
            seen_chain.add(cur["id"])
            cur = next((x for x in bones if x["id"] == cur["parent"]), None)

    tracks = []
    for t in (out.get("tracks") or [])[:MAX_TRACKS]:
        tid = str(t.get("id") or "").strip()
        if not tid or not t.get("assetId"):
            continue
        ks = []
        for k in (t.get("keyframes") or [])[:MAX_FRAMES]:
            ks.append({"frame": max(0, min(int(_num(k.get("frame"), 0)), out["frames"] - 1)),
                       "x": round(_num(k.get("x")), 2), "y": round(_num(k.get("y")), 2),
                       "rot": round(_num(k.get("rot")), 2),
                       "scale": max(0.01, round(_num(k.get("scale"), 1.0), 4)),
                       "opacity": max(0.0, min(round(_num(k.get("opacity"), 1.0), 4), 1.0)),
                       "ease": k.get("ease") if k.get("ease") in ("linear", "ease-in", "ease-out") else "linear"})
        # 同一帧多个关键帧只留最后一个，渲染时不必再挑
        dedup: Dict[int, Dict[str, Any]] = {}
        for k in sorted(ks, key=lambda x: x["frame"]):
            dedup[k["frame"]] = k
        warp = t.get("warp")
        if isinstance(warp, list) and len(warp) == 9:
            warp = [[round(_num(p[0]), 2), round(_num(p[1]), 2)] for p in warp]
        else:
            warp = [[0.0, 0.0] for _ in range(9)]
        pivot = t.get("pivot") or [0, 0]
        fx = []
        for s in (t.get("fx") or [])[:64]:
            pts = []
            for p in (s.get("points") or [])[:512]:
                if not p or len(p) < 2:
                    continue
                # 压力是可选字段：只给 x/y 的笔迹也合法，缺省按 1.0
                if len(p) > 2:
                    pr = max(0.05, min(_num(p[2], 1.0), 1.0))
                else:
                    pr = 1.0
                pts.append([round(_num(p[0]), 2), round(_num(p[1]), 2), round(pr, 3)])
            if len(pts) < 2:
                continue
            fx.append({"brush": s.get("brush") if s.get("brush") in
                       ("slash", "bristle", "dry", "spark", "echo") else "slash",
                       "color": str(s.get("color") or "#ffffff"),
                       "size": max(1.0, min(_num(s.get("size"), 8), 256)),
                       "points": pts,
                       "from": max(0, min(int(_num(s.get("from"), 0)), out["frames"] - 1)),
                       "to": max(0, min(int(_num(s.get("to"), out["frames"] - 1)), out["frames"] - 1))})
        tracks.append({
            "id": tid,
            "name": str(t.get("name") or f"轨道 {len(tracks) + 1}"),
            "assetId": str(t["assetId"]),
            "bone": str(t.get("bone") or "root"),
            "pivot": [round(_num(pivot[0]), 2), round(_num(pivot[1]), 2)],
            "warp": warp,
            "keyframes": [dedup[i] for i in sorted(dedup)],
            "fx": fx,
            "opacity": max(0.0, min(round(_num(t.get("opacity"), 1.0), 4), 1.0)),
            "blend": t.get("blend") if t.get("blend") in ("normal", "add", "multiply") else "normal",
            "visible": bool(t.get("visible", True)),
        })
    out["tracks"] = tracks
    return out
