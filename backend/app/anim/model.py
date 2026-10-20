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


def _skin(v: Any, bone_ids: set) -> Optional[Dict[str, Any]]:
    """骨骼蒙皮配置：参与骨骼 + 影响半径 + 网格密度。

    权重不存盘：按顶点到骨段的距离在渲染时现算（确定性），
    省掉一个「权重图」编辑器和一堆同步问题。
    """
    if not isinstance(v, dict):
        return None
    ids = [str(x) for x in (v.get("bones") or [])][:8]
    ids = [i for i in ids if i in bone_ids]
    if not ids:
        return None
    grid = max(3, min(int(_num(v.get("grid"), 8)), 20))
    # 手刷权重按 **归一化坐标 (u, v)** 存，不按顶点下标。
    #
    # 为什么：顶点下标依赖网格密度，改了密度同一份数据就指向别的位置
    # （用户会看到「权重全错位」，只能全部重刷）。
    # (u,v) ∈ [0,1]² 是贴图上的固定位置，密度变了重新采样即可，权重跟着走。
    #
    # 格式：{骨骼id: {"u,v": 权重}}，键是 "0.25,0.5" 这样的字符串。
    # 旧数据（按整数下标存的）读进来时按旧密度换算成 (u,v)——
    # 换算需要知道旧密度，用 v 里的 paintGrid 字段；没有就按当前 grid 猜。
    paint: Dict[str, Dict[str, float]] = {}
    raw = v.get("paint")
    raw_grid = max(3, min(int(_num(v.get("paintGrid"), grid)), 20))
    if isinstance(raw, dict):
        for bid, cells in list(raw.items())[:8]:
            if bid not in ids or not isinstance(cells, dict):
                continue
            clean: Dict[str, float] = {}
            for key, val in list(cells.items())[:4096]:
                w = round(max(0.0, min(_num(val), 1.0)), 4)
                if "," in str(key):
                    # 已是 (u,v) 格式
                    try:
                        us, vs = str(key).split(",", 1)
                        pu, pv = float(us), float(vs)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= pu <= 1.0 and 0.0 <= pv <= 1.0:
                        clean[f"{pu:.4f},{pv:.4f}"] = w
                else:
                    # 旧格式：整数顶点下标 → 按旧密度换算
                    try:
                        i = int(key)
                    except (TypeError, ValueError):
                        continue
                    top = (raw_grid + 1) * (raw_grid + 1) - 1
                    if 0 <= i <= top:
                        gx = i % (raw_grid + 1)
                        gy = i // (raw_grid + 1)
                        pu = gx / raw_grid
                        pv = gy / raw_grid
                        clean[f"{pu:.4f},{pv:.4f}"] = w
            if clean:
                paint[str(bid)] = clean
    return {"bones": ids,
            "falloff": max(8.0, min(round(_num(v.get("falloff"), 60), 1), 400.0)),
            "grid": grid,
            "paintGrid": raw_grid,
            "paint": paint}


def _bez(v: Any) -> Optional[List[float]]:
    """cubic-bezier 控制点：只收 4 个数，x 夹到 [0,1]（CSS 同样约束）。

    y 允许超出 0..1 —— 过冲（anticipation / 回弹）就是靠它做的，
    夹掉 y 等于把曲线编辑器的表达力砍掉一半。
    """
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    except (TypeError, ValueError):
        return None
    return [round(max(0.0, min(x1, 1.0)), 3), round(y1, 3),
            round(max(0.0, min(x2, 1.0)), 3), round(y2, 3)]


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
                      # 骨长：IK 解算与末端伸出方向都靠它；缺省 60px
                      "len": max(4.0, round(_num(b.get("len"), 60), 2)),
                      # 骨骼自身也能打关键帧（根骨骼整体位移、手臂摆动）
                      "keys": [{"frame": max(0, min(int(_num(k.get("frame"), 0)), out["frames"] - 1)),
                                "x": round(_num(k.get("x")), 2), "y": round(_num(k.get("y")), 2),
                                "rot": round(_num(k.get("rot")), 2),
                                "scale": max(0.01, round(_num(k.get("scale"), 1.0), 4)),
                                "ease": k.get("ease") if k.get("ease") in
                                ("linear", "ease-in", "ease-out") else "linear",
                                "bez": _bez(k.get("bez"))}
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

    # IK 目标：只允许挂在已存在的骨骼上；链长按祖先数量截断，
    # 免得写了 6 节但只有 2 根父骨骼时解算出 NaN
    iks: List[Dict[str, Any]] = []
    for it in (out.get("ik") or [])[:8]:
        eid = str(it.get("end") or "").strip()
        if eid not in ids:
            continue
        depth = 1
        cur = next((x for x in bones if x["id"] == eid), None)
        while cur and cur.get("parent") and depth < 6:
            depth += 1
            cur = next((x for x in bones if x["id"] == cur["parent"]), None)
        chain = max(2, min(int(_num(it.get("chain"), 2)), depth))
        ks = []
        for k in (it.get("keys") or [])[:MAX_FRAMES]:
            ks.append({"frame": max(0, min(int(_num(k.get("frame"), 0)), out["frames"] - 1)),
                       "x": round(_num(k.get("x")), 2), "y": round(_num(k.get("y")), 2),
                       # 权重：0 = 完全听 FK，1 = 完全听 IK；中间是混合，
                       # 做「手扶墙时手臂半跟随」这类过渡靠它
                       "w": max(0.0, min(round(_num(k.get("w"), 1.0), 3), 1.0)),
                       "ease": k.get("ease") if k.get("ease") in
                       ("linear", "ease-in", "ease-out") else "linear",
                       "bez": _bez(k.get("bez"))})
        dedup_ik: Dict[int, Dict[str, Any]] = {}
        for k in sorted(ks, key=lambda x: x["frame"]):
            dedup_ik[k["frame"]] = k
        if not dedup_ik:
            continue
        pole = it.get("pole")
        pole_xy = None
        if isinstance(pole, (list, tuple)) and len(pole) >= 2:
            pole_xy = [round(_num(pole[0]), 2), round(_num(pole[1]), 2)]
        iks.append({"id": str(it.get("id") or f"ik{len(iks)}"),
                    "name": str(it.get("name") or f"IK {len(iks) + 1}"),
                    "end": eid, "chain": chain, "pole": pole_xy,
                    "keys": [dedup_ik[i] for i in sorted(dedup_ik)]})
    out["ik"] = iks

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
                       "ease": k.get("ease") if k.get("ease") in ("linear", "ease-in", "ease-out") else "linear",
                       "bez": _bez(k.get("bez"))})
        # 同一帧多个关键帧只留最后一个，渲染时不必再挑
        dedup: Dict[int, Dict[str, Any]] = {}
        for k in sorted(ks, key=lambda x: x["frame"]):
            dedup[k["frame"]] = k
        def _warp(v) -> List[List[float]]:
            if isinstance(v, list) and len(v) == 9 and all(
                    isinstance(q, (list, tuple)) and len(q) >= 2 for q in v):
                return [[round(_num(q[0]), 2), round(_num(q[1]), 2)] for q in v]
            return [[0.0, 0.0] for _ in range(9)]

        warp = _warp(t.get("warp"))
        warp_keys = []
        for k in (t.get("warpKeys") or [])[:MAX_FRAMES]:
            warp_keys.append({"frame": max(0, min(int(_num(k.get("frame"), 0)), out["frames"] - 1)),
                              "warp": _warp(k.get("warp")),
                              "ease": k.get("ease") if k.get("ease") in
                              ("linear", "ease-in", "ease-out") else "linear",
                              "bez": _bez(k.get("bez"))})
        wdedup: Dict[int, Dict[str, Any]] = {}
        for k in sorted(warp_keys, key=lambda x: x["frame"]):
            wdedup[k["frame"]] = k
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
            "warpKeys": [wdedup[i] for i in sorted(wdedup)],
            "skin": _skin(t.get("skin"), ids),
            "keyframes": [dedup[i] for i in sorted(dedup)],
            "fx": fx,
            "opacity": max(0.0, min(round(_num(t.get("opacity"), 1.0), 4), 1.0)),
            "blend": t.get("blend") if t.get("blend") in ("normal", "add", "multiply") else "normal",
            "visible": bool(t.get("visible", True)),
        })
    out["tracks"] = tracks
    return out
