"""像素素材抠图：优先深度抠图（u2net），退化到 GrabCut 图割，最后才是颜色键。

为什么不再只靠颜色键：AI 出图常见「灰色渐变影棚底 + 灰白毛色角色」，
背景与白甲 / 白毛的 RGB 距离很近，纯颜色阈值无论怎么调都会二选一——
要么背景留一大片，要么把角色白甲抠成透明洞（实测误抠 75%）。

三级路径：
- **u2net**（默认，需 rembg + 权重）：显著性分割，灰底灰角色、软投影都能处理。
- **grabcut**（需 opencv）：边框作背景种子、中心作前景种子做图割；第二遍把
  脚底「低饱和且明显暗于局部底色」的像素标成可能背景，压掉地板软投影。
- **key**（最终兜底）：四角色板 + 连通域洪泛 + 有限梯度生长，仅靠颜色。

三条路径最终都走「硬边 alpha + 边缘去污（despill）」：
- 像素画的边只能是全不透明 / 全透明。旧版对 alpha 做高斯羽化，半透明像素
  在浅色底上显成一圈白晕，模糊半径还会把脸部 / 细肢等窄结构啃出洞。
- 紧贴背景的一圈像素颜色是背景与角色的混合，直接留下会形成白 / 绿色描边；
  这里把「仍接近背景色」的轮廓像素改写为相邻内部像素的颜色。
"""
from __future__ import annotations

import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def _neighbour_shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """把 bool mask 按 (dy,dx) 平移，越界填 False。"""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, ye = (dy, h) if dy >= 0 else (0, h + dy)
    xs, xe = (dx, w) if dx >= 0 else (0, w + dx)
    out[ys:ye, xs:xe] = mask[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def _shift_block(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """把任意通道数组按 (dy,dx) 平移，越界填 0。用于取邻居颜色。"""
    out = np.zeros_like(arr)
    h, w = arr.shape[:2]
    ys, ye = (dy, h) if dy >= 0 else (0, h + dy)
    xs, xe = (dx, w) if dx >= 0 else (0, w + dx)
    out[ys:ye, xs:xe] = arr[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        out &= _neighbour_shift(mask, dy, dx)
    return out


def _border_palette(arr: np.ndarray, bucket: int = 24, limit: int = 4) -> List[np.ndarray]:
    """四角取样得到背景色板（只取四角，避免把角色描边算成背景）。"""
    h, w = arr.shape[:2]
    pad = max(4, min(16, h // 8, w // 8))
    patches = [
        arr[0:pad, 0:pad, :3],
        arr[0:pad, w - pad:w, :3],
        arr[h - pad:h, 0:pad, :3],
        arr[h - pad:h, w - pad:w, :3],
    ]
    votes: dict = {}
    sums: dict = {}
    for patch in patches:
        px = patch.reshape(-1, 3).astype(np.int32)
        q = (px // bucket) * bucket
        uniq, counts = np.unique(q, axis=0, return_counts=True)
        for i in np.argsort(-counts)[:2]:
            key = tuple(int(v) for v in uniq[i])
            votes[key] = votes.get(key, 0) + 1
            sums.setdefault(key, []).append(uniq[i].astype(np.int32) + bucket // 2)

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    out: List[np.ndarray] = []
    for key, v in ranked:
        if v < 2 and out:
            continue
        out.append(np.mean(sums[key], axis=0).astype(np.int32))
        if len(out) >= limit:
            break
    if not out:
        out.append(arr[0, 0, :3].astype(np.int32))
    return out


def _palette_dist(rgb: np.ndarray, palette: List[np.ndarray]) -> np.ndarray:
    h, w = rgb.shape[:2]
    dist = np.full((h, w), 1e9, dtype=np.float32)
    for c in palette:
        d = np.sqrt(((rgb - c) ** 2).sum(axis=2)).astype(np.float32)
        dist = np.minimum(dist, d)
    return dist


def _row_bg(rgb: np.ndarray, strip: int = 8) -> np.ndarray:
    """用左右边框估计每一行的背景色（竖向渐变 / 暗角都能跟上）。"""
    h = rgb.shape[0]
    out = np.zeros((h, 3), dtype=np.float32)
    for y in range(h):
        side = np.concatenate([rgb[y, :strip], rgb[y, -strip:]], axis=0)
        out[y] = np.median(side, axis=0)
    return out


def _saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=2).astype(np.float32)
    mn = rgb.min(axis=2).astype(np.float32)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)


# --------------------------------------------------------------------------- #
# 边缘去污
# --------------------------------------------------------------------------- #
def _despill(arr: np.ndarray, kept: np.ndarray, palette: List[np.ndarray], tol: float) -> int:
    """把「紧贴透明区、颜色仍接近背景」的轮廓像素改成相邻内部像素的颜色。

    像素画的边界像素要么是角色本身的颜色，要么透明，不能是背景色的混合残影。
    """
    rgb = arr[:, :, :3].astype(np.int32)
    interior = kept & ~_erode(kept)
    edge = kept & ~interior
    if not edge.any():
        return 0
    dist = _palette_dist(rgb, palette)
    contaminated = edge & (dist <= tol * 1.25)

    out = rgb.copy()
    filled = np.zeros_like(kept)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        cand = contaminated & ~filled
        if not cand.any():
            continue
        near_int = _neighbour_shift(interior, dy, dx)
        if not near_int.any():
            continue
        take = cand & near_int
        if not take.any():
            continue
        out[take] = _shift_block(rgb, dy, dx)[take]
        filled |= take

    changed = int(filled.sum())
    if changed:
        arr[:, :, :3] = out.astype(np.uint8)
    return changed


def _fill_small_holes(kept: np.ndarray, max_area_ratio: float = 0.002) -> Tuple[np.ndarray, int]:
    """填掉被角色完全包围的小透明空洞（分割偶发斑点）。大缝隙会保留。"""
    try:
        import cv2
    except Exception:
        return kept, 0
    holes = (~kept).astype(np.uint8)
    n, lab = cv2.connectedComponents(holes, connectivity=4)
    if n <= 1:
        return kept, 0
    h, w = kept.shape
    max_area = int(max(4, max_area_ratio * h * w))
    filled = 0
    for i in range(1, n):
        comp = lab == i
        area = int(comp.sum())
        if area == 0 or area > max_area:
            continue
        ys, xs = np.nonzero(comp)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            continue
        kept |= comp
        filled += area
    return kept, filled


# --------------------------------------------------------------------------- #
# 路径 1：u2net 深度抠图
# --------------------------------------------------------------------------- #
_SESSION = None
_U2NET_ERR: Optional[str] = None
_U2NET_HOME = os.path.expanduser(os.getenv("U2NET_HOME", "~/.u2net"))
_U2NET_ONNX = os.path.join(_U2NET_HOME, "u2net.onnx")


def _u2net_session():
    global _SESSION, _U2NET_ERR
    if _SESSION is not None or _U2NET_ERR is not None:
        return _SESSION
    # 关键：权重文件不存在时 **绝不** 触发 rembg 的联网下载（会卡死数十分钟，
    # 直接拖垮整个请求）。只有本地权重就绪才尝试 new_session。
    if not os.path.exists(_U2NET_ONNX):
        _U2NET_ERR = f"u2net 权重缺失：{_U2NET_ONNX}（请先放置权重）"
        return None
    try:
        from rembg import new_session
        _SESSION = new_session("u2net")
    except Exception as e:  # 权重损坏 / onnxruntime 不可用
        _U2NET_ERR = f"{type(e).__name__}: {e}"
        _SESSION = None
    return _SESSION


def u2net_available() -> bool:
    return _u2net_session() is not None


def u2net_status() -> dict:
    _u2net_session()
    return {"available": _SESSION is not None, "error": _U2NET_ERR}


def _mask_keep(rgb: np.ndarray) -> np.ndarray:
    """显著性分割得到 kept 掩码（阈值二值化，像素素材不要软边）。"""
    from rembg import remove
    session = _u2net_session()
    if session is None:
        raise RuntimeError(_U2NET_ERR or "u2net session unavailable")
    mask = remove(Image.fromarray(rgb), session=session, only_mask=True)
    return np.array(mask) >= 128


# --------------------------------------------------------------------------- #
# 路径 2：GrabCut 图割
# --------------------------------------------------------------------------- #
def _grabcut_keep(rgb: np.ndarray, shadow_pass: bool = True, iters: int = 5) -> np.ndarray:
    """GrabCut 图割：返回 kept（前景）bool 掩码。

    外侧一圈为确定背景、中心为可能前景；第二遍把「角色 bbox 底部 30% 内、
    低饱和且明显暗于局部底色」的像素标成可能背景，压掉地板软投影
    （用 PR_BGD 而非 BGD：硬性种子会误切暗色腿甲）。
    """
    import cv2

    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = max(3, int(round(min(h, w) * 0.015)))
    rect = (m, m, w - 2 * m, h - 2 * m)

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[m:h - m, m:w - m] = cv2.GC_PR_FGD
    c = m + 2
    mask[:c, :c] = cv2.GC_BGD
    mask[:c, -c:] = cv2.GC_BGD
    mask[-c:, :c] = cv2.GC_BGD
    mask[-c:, -c:] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)

    if shadow_pass:
        kept = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
        ys, xs = np.nonzero(kept)
        if len(ys):
            top, bot = int(ys.min()), int(ys.max())
            y0 = top + int((bot - top) * 0.70)
            sat = _saturation(rgb)
            luma = rgb.mean(axis=2)
            bgl = _row_bg(rgb).mean(axis=1)
            band = np.zeros((h, w), dtype=bool)
            band[y0:bot + 1, :] = True
            seed = band & kept & (sat < 0.14) & (luma < bgl[:, None] - 15)
            if seed.any():
                mask[seed] = cv2.GC_PR_BGD
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)

    return np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])


# --------------------------------------------------------------------------- #
# 路径 3：颜色键兜底
# --------------------------------------------------------------------------- #
def _key_keep(rgb: np.ndarray, palette: List[np.ndarray], tolerance: float,
              edge_grow: int, grow_passes: int) -> Tuple[np.ndarray, int, int]:
    h, w = rgb.shape[:2]
    dist = _palette_dist(rgb, palette)
    match = dist <= float(tolerance)

    visited = np.zeros((h, w), dtype=bool)
    dq: deque = deque()
    for x in range(w):
        for y in (0, h - 1):
            if match[y, x] and not visited[y, x]:
                visited[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if match[y, x] and not visited[y, x]:
                visited[y, x] = True
                dq.append((y, x))

    removed = 0
    while dq:
        y, x = dq.popleft()
        removed += 1
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and match[ny, nx]:
                visited[ny, nx] = True
                dq.append((ny, nx))

    grown = 0
    if edge_grow > 0:
        near_bg = dist <= float(tolerance) * 1.5
        for _ in range(max(0, grow_passes)):
            added = np.zeros((h, w), dtype=bool)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                added |= _neighbour_shift(visited, dy, dx) & ~visited & near_bg
            if not added.any():
                break
            visited |= added
            grown += int(added.sum())

    return ~visited, removed, grown


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def connected_matte(
    img: Image.Image,
    tolerance: int = 80,
    feather: int = 0,
    sample: str = "auto",
    edge_grow: int = 10,
    grow_passes: int = 6,
    hard: bool = True,
    despill: bool = True,
    method: str = "auto",
) -> Tuple[Image.Image, dict]:
    """抠掉背景。

    hard=True（默认，像素素材）：二值 alpha + 边缘去污，无软边、无白晕。
    hard=False（照片 / 半透明素材）：保留高斯羽化，边缘更自然。
    method: auto | u2net | grabcut | key
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3].astype(np.int32)

    palette = [rgb[0, 0]] if sample == "topleft" else _border_palette(arr)
    already_clear = arr[:, :, 3] < 8

    # 解析实际使用的路径
    used = "key"
    chain: List[str] = []
    if method == "auto":
        chain = ["u2net", "grabcut", "key"] if hard else ["key"]
    else:
        chain = [method]

    kept = None
    removed = 0
    grown = 0
    holes_filled = 0
    last_err = None

    for cand in chain:
        if cand == "u2net":
            if not hard or min(h, w) < 64 or not u2net_available():
                continue
            try:
                kept = _mask_keep(arr[:, :, :3])
                used = "u2net"
            except Exception as e:
                last_err = f"u2net: {type(e).__name__}"
                kept = None
                continue
        elif cand == "grabcut":
            if not hard or min(h, w) < 96:
                continue
            try:
                import cv2  # noqa: F401
                kept = _grabcut_keep(arr[:, :, :3])
                used = "grabcut"
            except Exception as e:  # 无 cv2
                last_err = f"grabcut: {type(e).__name__}"
                kept = None
                continue
        else:
            kept, removed, grown = _key_keep(rgb, palette, tolerance, edge_grow, grow_passes)
            used = "key"
        break

    if kept is None:  # 兜底
        kept, removed, grown = _key_keep(rgb, palette, tolerance, edge_grow, grow_passes)
        used = "key"

    kept = kept & ~already_clear
    if hard and used != "key":
        kept, holes_filled = _fill_small_holes(kept)
    removed = int((~kept).sum())

    despilled = 0
    alpha = np.where(kept, 255, 0).astype(np.uint8)

    if hard:
        if despill:
            despilled = _despill(arr, kept, palette, float(tolerance))
        arr[:, :, 3] = alpha
    else:
        if feather > 0:
            softened = np.array(
                Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(max(0.5, feather / 2.0)))
            )
            alpha = np.where(softened > 250, 255, softened).astype(np.uint8)
        arr[:, :, 3] = alpha

    out = Image.fromarray(arr, "RGBA")
    meta = {
        "method": used,
        "background": "#%02X%02X%02X" % tuple(int(v) for v in palette[0]),
        "palette": ["#%02X%02X%02X" % tuple(int(v) for v in c) for c in palette],
        "tolerance": tolerance,
        "feather": feather,
        "hard": hard,
        "despilledPixels": despilled,
        "holesFilled": holes_filled,
        "edgeGrowPixels": grown,
        "removedRatio": round(removed / float(h * w), 4),
    }
    if last_err:
        meta["fallbackReason"] = last_err
    return out, meta
