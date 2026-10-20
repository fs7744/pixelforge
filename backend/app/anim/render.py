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
from . import sequence as seq_mod

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


def _bezier(t: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """cubic-bezier(x1,y1,x2,y2) 在时间 t 的取值（CSS 语义）。

    曲线是参数式 (x(u), y(u))，要的是「x = t 时的 y」，所以先解 u 再取 y。
    先牛顿迭代（快，通常 3-4 次收敛），失败退回二分（慢但不会发散）——
    只靠牛顿的话，控制点贴近端点时导数接近 0，会直接飞出去。
    """
    t = max(0.0, min(1.0, t))
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0

    def beziers(u: float, a: float, b: float) -> float:
        # 三次贝塞尔，端点固定在 0/1，中间两个控制点 a、b
        v = 1 - u
        return 3 * v * v * u * a + 3 * v * u * u * b + u ** 3

    def slope(u: float, a: float, b: float) -> float:
        v = 1 - u
        return 3 * v * v * a + 6 * v * u * (b - a) + 3 * u * u * (1 - b)

    # 先二分出粗解，再牛顿精修：单纯二分要 40+ 次才够精度
    lo, hi, u = 0.0, 1.0, t
    for _ in range(24):
        x = beziers(u, x1, x2)
        if abs(x - t) < 1e-5:
            break
        if x < t:
            lo = u
        else:
            hi = u
        u = (lo + hi) * 0.5
    for _ in range(8):
        d = slope(u, x1, x2)
        if abs(d) < 1e-6:
            break
        u -= (beziers(u, x1, x2) - t) / d
        u = max(0.0, min(1.0, u))
    return beziers(u, y1, y2)


def _sample(keys: Sequence[Dict[str, Any]], frame: int) -> Dict[str, float]:
    """关键帧采样：线性 + 缓动；首帧前/末帧后保持端点值（不外推）。"""
    if not keys:
        return {"x": 0.0, "y": 0.0, "rot": 0.0, "scale": 1.0, "opacity": 1.0, "w": 1.0}
    ks = sorted(keys, key=lambda k: k["frame"])
    if frame <= ks[0]["frame"]:
        return {k: float(ks[0].get(k, 0.0)) for k in ("x", "y", "rot")} | {
            "scale": float(ks[0].get("scale", 1.0)),
            "opacity": float(ks[0].get("opacity", 1.0)),
            "w": float(ks[0].get("w", 1.0))}
    if frame >= ks[-1]["frame"]:
        return {k: float(ks[-1].get(k, 0.0)) for k in ("x", "y", "rot")} | {
            "scale": float(ks[-1].get("scale", 1.0)),
            "opacity": float(ks[-1].get("opacity", 1.0)),
            "w": float(ks[-1].get("w", 1.0))}
    for a, b in zip(ks, ks[1:]):
        if a["frame"] <= frame <= b["frame"]:
            span = b["frame"] - a["frame"]
            t = 0.0 if span == 0 else (frame - a["frame"]) / span
            # 缓动取**后一帧**的 ease/bez：表示「走到这一帧时怎么减速」。
            # 有 bez 就优先 bez —— 曲线编辑器存的是 4 个控制点，
            # 命名缓动只是给不想调曲线的人准备的快捷档。
            bez = b.get("bez")
            t = _bezier(t, *bez) if bez else _ease(t, b.get("ease") or "linear")
            out = {}
            for k, d in (("x", 0.0), ("y", 0.0), ("rot", 0.0)):
                out[k] = float(a.get(k, d)) + (float(b.get(k, d)) - float(a.get(k, d))) * t
            out["scale"] = float(a.get("scale", 1.0)) + (float(b.get("scale", 1.0)) - float(a.get("scale", 1.0))) * t
            out["opacity"] = float(a.get("opacity", 1.0)) + (float(b.get("opacity", 1.0)) - float(a.get("opacity", 1.0))) * t
            # IK 权重也走同一条采样：漏掉它的话，w 永远回到默认 1.0，
            # 界面上拖权重会发现「怎么拖都没反应」
            out["w"] = float(a.get("w", 1.0)) + (float(b.get("w", 1.0)) - float(a.get("w", 1.0))) * t
            return out
    return {"x": 0.0, "y": 0.0, "rot": 0.0, "scale": 1.0, "opacity": 1.0, "w": 1.0}


def _fk_world(proj: Dict[str, Any], frame: int,
              override: Optional[Dict[str, tuple]] = None) -> Dict[str, np.ndarray]:
    """纯 FK：父级级联。override 用来替换 IK 解算出的骨骼世界位姿。

    override[bid] = (x, y, 角度, 缩放)：IK 直接把关节世界坐标喂进来，
    再按这个坐标反推矩阵 —— 比「改欧拉角再级联」稳，不会因为父级缩放
    把角度算歪（那条路上只要父骨骼带非等比缩放就必错）。
    """
    bones = {b["id"]: b for b in proj.get("bones") or []}
    out: Dict[str, np.ndarray] = {}
    override = override or {}

    def build(bid: str, seen: set) -> np.ndarray:
        if bid in out:
            return out[bid]
        if bid in seen:      # 环（normalize 已拦，这里兜底防递归爆栈）
            out[bid] = np.eye(3)
            return out[bid]
        seen.add(bid)
        b = bones[bid]
        if bid in override:
            x, y, ang, sc = override[bid]
            m = _T(x, y) @ _R(ang) @ _S(sc)
        else:
            k = _sample(b.get("keys") or [], frame)
            local = _T(k["x"], k["y"]) @ _R(k["rot"]) @ _S(k["scale"])
            parent = b.get("parent")
            if parent and parent in bones:
                pb = bones[parent]
                m = build(parent, seen) @ _T(b["x"] - pb["x"], b["y"] - pb["y"]) @ local
            else:
                m = _T(b["x"], b["y"]) @ local
        out[bid] = m
        return m

    for bid in bones:
        build(bid, set())
    return out


def _matrix_angle(m: np.ndarray) -> float:
    return math.atan2(float(m[1, 0]), float(m[0, 0]))


def _matrix_scale(m: np.ndarray) -> float:
    return max(0.01, math.hypot(float(m[0, 0]), float(m[1, 0])))


def _apply_pole(pts: List[np.ndarray], pole: np.ndarray) -> List[np.ndarray]:
    """极向校正：中间关节不在 pole 那一侧时，把链条沿 root→end 直线镜像。

    FABRIK 只保证「长度对、末端到位」，肘/膝该朝哪边它不管——不加这个，
    同一组目标在不同起始姿势下会随机翻肘。镜像不会移动末端（末端在直线上）。
    """
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    line = b - a
    L = float(np.linalg.norm(line))
    if L < 1e-3:
        return pts
    n = np.array([-line[1], line[0]]) / L

    def side(p: np.ndarray) -> float:
        return float(np.dot(p - a, n))

    if side(pts[1]) * side(pole) >= 0:
        return pts
    out = [a.copy()]
    for p in pts[1:-1]:
        out.append(a + (p - a) - 2 * side(p) * n)
    out.append(b.copy())
    return out


def _solve_ik(proj: Dict[str, Any], frame: int,
              base: Dict[str, np.ndarray], override: Dict[str, tuple],
              ik: Dict[str, Any], iterations: int = 12) -> None:
    """FABRIK 解 2D 骨链，把结果写进 override。

    为什么用 FABRIK 而不是解析二骨：解析式只对两节闭式可解，手臂一旦是
    三节（大臂/小臂/手）就得写分支；FABRIK 对 N 节统一，十几行就够，
    迭代 12 次在像素级已经收敛。

    关节取「骨骼原点的世界坐标」，末端 effector 再沿最后一节的朝向伸出
    该骨骼的 len —— 骨骼在数据里是一个点，长度靠 len 补足。
    """
    bones = {b["id"]: b for b in proj.get("bones") or []}
    chain: List[str] = []
    bid = ik.get("end")
    while bid and bid in bones and len(chain) < int(ik.get("chain") or 2):
        chain.append(bid)
        bid = bones[bid].get("parent")
    chain.reverse()
    if len(chain) < 2:
        return

    pos: List[np.ndarray] = []
    angs: List[float] = []
    lens: List[float] = []
    for b_id in chain:
        if b_id in override:
            x, y, ang, _sc = override[b_id]
            pos.append(np.array([x, y], dtype=np.float64))
            angs.append(math.radians(ang))
        else:
            m = base.get(b_id, np.eye(3))
            pos.append(np.array([m[0, 2], m[1, 2]], dtype=np.float64))
            angs.append(_matrix_angle(m))
        lens.append(float(bones[b_id].get("len") or 60))

    tip = pos[-1] + np.array([math.cos(angs[-1]), math.sin(angs[-1])]) * lens[-1]
    pts = pos + [tip]                      # 关节 0..n，pts[n] 为末端
    target = np.array([_sample(ik.get("keys") or [], frame)[k] for k in ("x", "y")],
                      dtype=np.float64)
    root = pts[0].copy()
    total = float(sum(lens))
    reach = float(np.linalg.norm(target - root))

    if reach >= total:
        # 够不着：整条链朝目标方向伸直（比反复迭代更稳定，也符合直觉）
        d = (target - root) / (reach or 1.0)
        cur = root.copy()
        pts = [cur.copy()]
        for L in lens:
            cur = cur + d * L
            pts.append(cur.copy())
    else:
        for _ in range(iterations):
            pts[-1] = target.copy()
            for i in range(len(pts) - 2, -1, -1):
                v = pts[i] - pts[i + 1]
                n = float(np.linalg.norm(v)) or 1.0
                pts[i] = pts[i + 1] + v / n * lens[i]
            pts[0] = root.copy()
            for i in range(len(pts) - 1):
                v = pts[i + 1] - pts[i]
                n = float(np.linalg.norm(v)) or 1.0
                pts[i + 1] = pts[i] + v / n * lens[i]
            if float(np.linalg.norm(pts[-1] - target)) < 0.5:
                break

    pole = ik.get("pole")
    if pole and len(pts) >= 3:
        pts = _apply_pole(pts, np.array(pole, dtype=np.float64))

    # 权重：把「IK 解」和「当前 FK 位姿」按 w 混合。
    # 位置线性插值、角度走最短弧（差角归一化到 ±180，否则 170°→-170°
    # 会被当成绕一大圈，手臂直接抡过去）。
    # 权重曲线可能过冲（回弹 bez），夹回 0..1：超出去会把姿态推到 IK 解之外，
    # 表现为「明明 w=1 却差几像素」
    w = max(0.0, min(float(_sample(ik.get("keys") or [], frame).get("w", 1.0)), 1.0))
    for i, b_id in enumerate(chain):
        cur = override.get(b_id)
        if cur:
            fk_pos = np.array([cur[0], cur[1]], dtype=np.float64)
            fk_ang, fk_sc = cur[2], cur[3]
        else:
            m = base.get(b_id, np.eye(3))
            fk_pos = np.array([m[0, 2], m[1, 2]], dtype=np.float64)
            fk_ang = math.degrees(_matrix_angle(m))
            fk_sc = _matrix_scale(m)
        d = pts[i + 1] - pts[i]
        ik_ang = math.degrees(math.atan2(float(d[1]), float(d[0])))
        pos = fk_pos + (pts[i] - fk_pos) * w
        diff = (ik_ang - fk_ang + 180.0) % 360.0 - 180.0
        override[b_id] = (float(pos[0]), float(pos[1]), fk_ang + diff * w, fk_sc)


def bone_world(proj: Dict[str, Any], frame: int) -> Dict[str, np.ndarray]:
    """每根骨骼在该帧的世界矩阵（FK + IK 覆盖）。"""
    base = _fk_world(proj, frame)
    iks = proj.get("ik") or []
    if not iks:
        return base
    override: Dict[str, tuple] = {}
    for ik in iks:
        _solve_ik(proj, frame, base, override, ik)
    return _fk_world(proj, frame, override) if override else base


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


def _sample_warp(track: Dict[str, Any], frame: int) -> List[List[float]]:
    """变形网格采样：有关键帧就插值，没有就用静态 warp。"""
    keys = track.get("warpKeys") or []
    if not keys:
        return track.get("warp") or []
    ks = sorted(keys, key=lambda k: k["frame"])
    if frame <= ks[0]["frame"]:
        return ks[0]["warp"]
    if frame >= ks[-1]["frame"]:
        return ks[-1]["warp"]
    for a, b in zip(ks, ks[1:]):
        if a["frame"] <= frame <= b["frame"]:
            span = b["frame"] - a["frame"] or 1
            t = (frame - a["frame"]) / span
            bez = b.get("bez")
            t = _bezier(t, *bez) if bez else _ease(t, b.get("ease") or "linear")
            return [[a["warp"][i][0] + (b["warp"][i][0] - a["warp"][i][0]) * t,
                     a["warp"][i][1] + (b["warp"][i][1] - a["warp"][i][1]) * t]
                    for i in range(9)]
    return ks[-1]["warp"]


def _dist_to_segment(px: float, py: float, x1: float, y1: float,
                     x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _skin_mesh(src: Image.Image, proj: Dict[str, Any], track: Dict[str, Any],
               frame: int) -> Optional[List[List[Tuple[float, float]]]]:
    """骨骼权重蒙皮：顶点权重 → 线性混合 → 画布坐标。

    权重算法（自动权重 + 手刷覆盖）已经挪到 anim/skin.py，那里同时供
    「权重可视化」使用；渲染只负责把权重变成坐标。

    变换链：先把顶点按「静止位」摆到画布上（p_rest），再让每根骨骼用它相对
    静止位的位移去拉这个点（M_i = world_i · T(-rest_i)），按权重加权。
    所有骨骼都不动时 M_i 是单位阵，结果与原图逐像素一致——这条恒等式
    就是最有力的回归测试。
    """
    from . import skin as skin_mod

    return skin_mod.deform(proj, track, src.size, frame)


def _blit_triangle(out: np.ndarray, src: np.ndarray,
                   src_tri, dst_tri) -> None:
    """把源三角形仿射贴到目标三角形上（只看目标包围盒，避免整幅重采样）。"""
    xs = [p[0] for p in dst_tri]
    ys = [p[1] for p in dst_tri]
    H, W = out.shape[:2]
    x0 = max(0, int(math.floor(min(xs))) - 1)
    x1 = min(W, int(math.ceil(max(xs))) + 2)
    y0 = max(0, int(math.floor(min(ys))) - 1)
    y1 = min(H, int(math.ceil(max(ys))) + 2)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    # warpAffine 的 M 是「源 → 目标」，内部自己求逆；传反了会采到图外，
    # 结果整块透明（还会把相邻三角写出来的像素擦掉）。
    m = cv2.getAffineTransform(
        np.float32(src_tri),
        np.float32([(p[0] - x0, p[1] - y0) for p in dst_tri]))
    roi = cv2.warpAffine(src, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, np.int32([(p[0] - x0, p[1] - y0) for p in dst_tri]), 255)
    sub = out[y0:y1, x0:x1]
    sel = mask.astype(bool)
    sub[sel] = roi[sel]


def _warp_grid(src: Image.Image, verts: List[List[Tuple[float, float]]],
               grid: int, canvas: Tuple[int, int]) -> Image.Image:
    """按「顶点网格」把贴图撕成三角形逐片仿射贴回去。

    为什么不用 PIL 的 Image.MESH：它的语义是「源四边形 → 目标**包围盒**」，
    形变信息只能体现在包围盒上，四边形内部怎么歪完全丢掉——蒙皮的弯曲会
    退化成方块拼接。三角形分片仿射是精确的（片内仿射，片间连续）。
    """
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("骨骼蒙皮需要 opencv-python：pip install opencv-python")
    sw, sh = src.size
    arr = np.asarray(src.convert("RGBA"))
    cw, ch = canvas
    out = np.zeros((ch, cw, 4), np.uint8)
    for j in range(grid):
        for i in range(grid):
            s00 = (sw * i / grid, sh * j / grid)
            s10 = (sw * (i + 1) / grid, sh * j / grid)
            s11 = (sw * (i + 1) / grid, sh * (j + 1) / grid)
            s01 = (sw * i / grid, sh * (j + 1) / grid)
            d00, d10 = verts[j][i], verts[j][i + 1]
            d11, d01 = verts[j + 1][i + 1], verts[j + 1][i]
            _blit_triangle(out, arr, (s00, s10, s11), (d00, d10, d11))
            _blit_triangle(out, arr, (s00, s11, s01), (d00, d11, d01))
    return Image.fromarray(out, "RGBA")


def _track_source(t: Dict[str, Any], loader, frame: int,
                  total_frames: int = 0) -> Optional[Image.Image]:
    """取轨道这一帧该用的贴图。

    普通轨道：整张贴图。
    序列帧轨道（带 seq）：按帧号从精灵表裁出对应格子。

    序列帧的 pivot 是**格子内坐标**（不是整表坐标）——裁出来的格子在
    后续流程里就是一张普通贴图，pivot / 变形 / 蒙皮全部按格子尺寸算。
    """
    src = loader(t["assetId"])
    if src is None:
        return None
    src = src.convert("RGBA")
    seq = t.get("seq")
    if not seq:
        return src
    boxes = seq_mod.cell_boxes(src, seq)
    if not boxes:
        return None
    idx = seq_mod.frame_index(seq, frame, len(boxes), int(total_frames or 0))
    x0, y0, x1, y1 = boxes[idx]
    return src.crop((x0, y0, x1, y1))


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
        src = _track_source(t, loader, frame, int(proj.get("frames") or 0))
        if src is None:
            continue
        src = warp_image(src, _sample_warp(t, frame))
        # 关键帧采样在两条路径（蒙皮 / 普通）都要用，提前算，别在分支里各算一次
        k = _sample(t.get("keyframes") or [], frame)
        if t.get("skin"):
            verts = _skin_mesh(src, proj, t, frame)
            placed = (_warp_grid(src, verts, int(t["skin"].get("grid") or 8), (cw, ch))
                      if verts else Image.new("RGBA", (cw, ch), (0, 0, 0, 0)))
        else:
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
