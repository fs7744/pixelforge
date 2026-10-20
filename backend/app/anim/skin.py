"""骨骼权重：自动权重 + 手刷覆盖。

自动权重（按顶点到骨段的距离）够用，但总有失手的时候：比如手臂旁边的布
本该跟肩，却被自动算到跟着手臂甩。所以这里给出手刷覆盖：

    skin.paint = { "骨骼id": { "顶点下标": 权重 } }

只存**被改过的顶点**，其余仍走自动计算。好处是工程文件小、撤销简单
（删掉 paint 里的条目即可），坏处是网格密度变了以后下标会错位——
所以 normalize 会把下标夹在合法范围内，真要改密度建议重置后重刷。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def dist_to_segment(px: float, py: float, x1: float, y1: float,
                    x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def setup(proj: Dict[str, Any], track: Dict[str, Any], src_size: Tuple[int, int],
          frame: int) -> Optional[Dict[str, Any]]:
    """算出顶点静止位坐标、参与骨骼与自动权重（不含 paint 覆盖）。"""
    skin = track.get("skin") or {}
    bones = {b["id"]: b for b in proj.get("bones") or []}
    ids = [i for i in (skin.get("bones") or []) if i in bones]
    if not ids:
        return None
    grid = max(3, min(int(skin.get("grid") or 8), 20))
    falloff = float(skin.get("falloff") or 60)
    w, h = src_size

    bind = track.get("bone") or "root"
    k = _sample_local(track.get("keyframes") or [], frame)
    px, py = track.get("pivot") or [w / 2, h / 2]
    br = bones.get(bind)
    ox, oy = (br["x"], br["y"]) if br else (w / 2, h / 2)
    # 静止位摆放：绑定骨骼原点 + 关键帧位移/旋转/缩放，再把 pivot 移到原点
    ang = math.radians(k["rot"])
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    scale = max(1e-3, k["scale"])

    rest: List[List[float]] = []
    auto: List[List[float]] = []
    for j in range(grid + 1):
        for i in range(grid + 1):
            sx = w * i / grid - px
            sy = h * j / grid - py
            rx = ox + k["x"] + (sx * cos_a - sy * sin_a) * scale
            ry = oy + k["y"] + (sx * sin_a + sy * cos_a) * scale
            rest.append([round(rx, 2), round(ry, 2)])
            row = []
            for bid in ids:
                b = bones[bid]
                d = dist_to_segment(rx, ry, b["x"], b["y"],
                                    b["x"] + float(b.get("len") or 60), b["y"])
                row.append(1.0 / ((d / max(1.0, falloff)) ** 2 + 1.0))
            auto.append(row)
    return {"grid": grid, "bones": ids, "rest": rest, "auto": auto,
            "falloff": falloff, "index": {b: n for n, b in enumerate(ids)}}


def _sample_local(keys: Sequence[Dict[str, Any]], frame: int) -> Dict[str, float]:
    """关键帧采样（与 render._sample 同口径的精简版，只取 x/y/rot/scale）。"""
    if not keys:
        return {"x": 0.0, "y": 0.0, "rot": 0.0, "scale": 1.0}
    ks = sorted(keys, key=lambda k: k["frame"])
    if frame <= ks[0]["frame"]:
        return {a: float(ks[0].get(a, d)) for a, d in
                (("x", 0.0), ("y", 0.0), ("rot", 0.0), ("scale", 1.0))}
    if frame >= ks[-1]["frame"]:
        return {a: float(ks[-1].get(a, d)) for a, d in
                (("x", 0.0), ("y", 0.0), ("rot", 0.0), ("scale", 1.0))}
    for a, b in zip(ks, ks[1:]):
        if a["frame"] <= frame <= b["frame"]:
            span = (b["frame"] - a["frame"]) or 1
            t = (frame - a["frame"]) / span
            return {key: float(a.get(key, 0.0)) + (float(b.get(key, 0.0)) - float(a.get(key, 0.0))) * t
                    for key in ("x", "y", "rot", "scale")}
    return {"x": 0.0, "y": 0.0, "rot": 0.0, "scale": 1.0}


def apply_paint(info: Dict[str, Any], paint: Optional[Dict[str, Any]]) -> List[List[float]]:
    """自动权重 + 手刷覆盖 → 归一化后的最终权重（每顶点对每骨骼）。

    手刷的语义是「绝对权重」：刷成 1.0 就是完全交给这根骨骼，剩余质量按
    自动权重分给其它骨骼。早期版本把刷的值当相对值塞进 top2 里再归一化，
    结果「刷满」只到 0.6——因为另一根骨骼的自动权重还占着，用户会以为
    笔刷坏了。

    只保留权重最大的两根骨骼：更多骨骼同时作用在一点上，数学上没错，
    但 2D 里几乎总是「拉成一团」，且很难调出稳定的手感。
    """
    idx = info["index"]
    n = len(info["rest"])
    grid = int(info["grid"])

    # 手刷数据按 (u,v) 存，这里重采样到当前网格的每个顶点。
    #
    # 为什么是重采样而不是直接查表：顶点下标依赖网格密度，改了密度同一份
    # 数据就指向别的位置（权重全错位，只能全部重刷）。(u,v) 是贴图上的
    # 固定位置，密度变了重新采样，权重跟着走。
    #
    # 采样用**双线性**：权重是连续量（刷出来是一片区域，不是几个孤立点），
    # 最近邻会在网格变密时把边界锯齿化。
    sampled: Dict[int, List[Tuple[float, float, float]]] = {}
    if paint:
        for bid, cells in paint.items():
            col = idx.get(bid)
            if col is None or not cells:
                continue
            pts: List[Tuple[float, float, float]] = []
            for key, val in cells.items():
                try:
                    us, vs = str(key).split(",", 1)
                    u, v = float(us), float(vs)
                except (TypeError, ValueError):
                    continue
                pts.append((u, v, max(0.0, min(1.0, float(val)))))
            if pts:
                sampled[col] = pts

    def _sample_uv(u: float, v: float,
                    pts: List[Tuple[float, float, float]]) -> Optional[float]:
        """在 (u,v) 处双线性采样手刷权重；附近没有数据点则返回 None。"""
        best = None
        best_d = 1e9
        for (pu, pv, w) in pts:
            d = (pu - u) ** 2 + (pv - v) ** 2
            if d < best_d:
                best_d = d
                best = (pu, pv, w)
        if best is None:
            return None
        # 影响半径：1.5 个网格单位。超过这个距离就认为「这块没刷过」——
        # 网格变粗时不能把远处的刷点也算进来，否则整张图都会被染色。
        radius = 1.5 / max(1.0, float(grid))
        if best_d > radius * radius:
            return None
        # 双线性：找 (u,v) 周围的四个数据点
        us = sorted({p[0] for p in pts})
        vs = sorted({p[1] for p in pts})
        u0 = max([x for x in us if x <= u], default=us[0])
        u1 = min([x for x in us if x >= u], default=us[-1])
        v0 = max([y for y in vs if y <= v], default=vs[0])
        v1 = min([y for y in vs if y >= v], default=vs[-1])
        if (u1 - u0) < 1e-9 or (v1 - v0) < 1e-9:
            return best[2]
        fu = (u - u0) / (u1 - u0)
        fv = (v - v0) / (v1 - v0)

        def _w_at(pu: float, pv: float) -> float:
            for (a, b, w) in pts:
                if abs(a - pu) < 1e-6 and abs(b - pv) < 1e-6:
                    return w
            return best[2]

        w00 = _w_at(u0, v0)
        w10 = _w_at(u1, v0)
        w01 = _w_at(u0, v1)
        w11 = _w_at(u1, v1)
        return (w00 * (1 - fu) * (1 - fv) + w10 * fu * (1 - fv)
                + w01 * (1 - fu) * fv + w11 * fu * fv)

    out: List[List[float]] = []
    for v in range(n):
        auto = info["auto"][v]
        row = list(auto)
        painted: Dict[int, float] = {}
        if sampled:
            gx = v % (grid + 1)
            gy = v // (grid + 1)
            u = gx / grid
            vv = gy / grid
            for col, pts in sampled.items():
                got = _sample_uv(u, vv, pts)
                if got is not None:
                    painted[col] = got

        if painted:
            total = sum(painted.values())
            if total > 1.0:                     # 同一顶点刷了多根骨骼：先按比例缩
                painted = {c: w / total for c, w in painted.items()}
                total = 1.0
            left = 1.0 - total
            others = [(auto[c], c) for c in range(len(auto)) if c not in painted]
            rest_total = sum(w for w, _ in others) or 1.0
            norm = [0.0] * len(auto)
            for c, val in painted.items():
                norm[c] = val
            if left > 1e-4:
                for w, c in others:
                    norm[c] = (w / rest_total) * left
        else:
            order = sorted(range(len(auto)), key=lambda c: auto[c], reverse=True)[:2]
            top = [(auto[c], c) for c in order if auto[c] > 1e-4]
            tot = sum(w for w, _ in top) or 1.0
            norm = [0.0] * len(auto)
            for w, c in top:
                norm[c] = w / tot
        out.append(norm)
    return out


def deform(proj: Dict[str, Any], track: Dict[str, Any], src_size: Tuple[int, int],
           frame: int) -> Optional[List[List[Tuple[float, float]]]]:
    """线性混合蒙皮：返回 (grid+1)×(grid+1) 的画布坐标顶点。"""
    from . import render as render_mod  # 延迟导入，避免循环依赖

    info = setup(proj, track, src_size, frame)
    if info is None:
        return None
    weights = apply_paint(info, track.get("skin", {}).get("paint"))
    bones = {b["id"]: b for b in proj.get("bones") or []}
    worlds = render_mod.bone_world(proj, frame)
    ids = info["bones"]
    grid = info["grid"]

    deltas = []
    for bid in ids:
        b = bones[bid]
        m = worlds.get(bid, np.eye(3))
        deltas.append(m @ render_mod._T(-b["x"], -b["y"]))  # noqa: SLF001

    verts: List[List[Tuple[float, float]]] = []
    k = 0
    for j in range(grid + 1):
        row: List[Tuple[float, float]] = []
        for i in range(grid + 1):
            rx, ry = info["rest"][k]
            w = weights[k]
            ax = ay = 0.0
            for c, wc in enumerate(w):
                if wc <= 1e-4:
                    continue
                p = deltas[c] @ np.array([rx, ry, 1.0])
                ax += wc * float(p[0])
                ay += wc * float(p[1])
            row.append((ax, ay))
            k += 1
        verts.append(row)
    return verts


def paint_view(proj: Dict[str, Any], track: Dict[str, Any], src_size: Tuple[int, int],
               frame: int) -> Optional[Dict[str, Any]]:
    """给界面用的权重可视化数据（顶点坐标 + 每骨骼权重）。"""
    info = setup(proj, track, src_size, frame)
    if info is None:
        return None
    weights = apply_paint(info, track.get("skin", {}).get("paint"))
    painted = sum(len(c) for c in (track.get("skin", {}).get("paint") or {}).values())
    return {
        "grid": info["grid"],
        "bones": [{"id": b, "name": next((x["name"] for x in proj["bones"] if x["id"] == b), b)}
                  for b in info["bones"]],
        "rest": info["rest"],
        "weights": [[round(x, 4) for x in row] for row in weights],
        "painted": painted,
    }