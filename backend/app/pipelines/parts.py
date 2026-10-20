"""角色部件拆分：人涂部位，机器按颜色裁切。

为什么不自动拆：头/身/四肢的语义判断，本来就该由人给。自动方案要么依赖
「角色部件级分层」模型（现在只有场景版），要么靠几何启发式猜（宽高比、
连通域形状）——猜出来的部件在关节处照样会撕，用户还得回来手改一遍，
不如把笔刷给他。

所以这里的分工很干脆：**用户涂颜色，系统出图**。
颜色标定图（每种部件一个纯色）与角色图同尺寸，前端在角色图上直接涂；
服务端按颜色容差取区域，再乘上去背 alpha，得到干净的 RGBA 部件。

产物 meta 里带 groupId / partName，动画编辑器的「导入拆分件」直接认得，
不需要额外适配。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def hex_rgb(color: str) -> Tuple[int, int, int]:
    c = (color or "").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except (ValueError, IndexError):
        return (255, 255, 255)


def _hex_dist(a: np.ndarray, b: Tuple[int, int, int]) -> np.ndarray:
    """RGB 欧氏距离。

    用 int32 而不是 int16：差值平方最大 255²×3 ≈ 19.5 万，**远超 int16 的
    32767**，实测会在 numpy 里回绕成负数，sqrt 直接给出 NaN——不报错，
    只是让对应像素的 mask 静默失效。这里踩过一次（控制台一行 RuntimeWarning）。
    """
    target = np.array(b, dtype=np.int32)
    d = a.astype(np.int32) - target
    return np.sqrt((d * d).sum(axis=-1, dtype=np.int64)).astype(np.float32)


def split_parts(art: Image.Image, tag: Image.Image,
                parts: Sequence[Dict[str, Any]],
                tolerance: int = 40) -> List[Tuple[str, Image.Image]]:
    """按颜色把 tag 的区域从 art 里裁出来。

    tolerance 是**颜色距离**不是单通道阈值：用户涂的颜色可能和预设不完全一致，
    纯通道判等会脆到没法用。
    """
    art_rgba = _defringe(art.convert("RGBA"))
    aw, ah = art_rgba.size
    tag_rgb = _fit_tag(tag, (aw, ah))
    tarr = np.asarray(tag_rgb)
    aarr = np.asarray(art_rgba)
    alpha = aarr[..., 3]

    # 涂得太满/没涂到的部件直接跳过，但要在返回里保留名字，
    # 让前端能告诉用户「头部那块你好像没涂」
    # 先把每个部位的候选区域算出来，再做**互斥分配**。
    #
    # 为什么必须互斥：用户的框选难免重叠（框大的把框小的罩住了）。
    # 如果每个部位各自算 mask，同一个像素会被两个部件同时认领——
    # 叠回去看着还行，但在动画里两个部件会抢同一块区域，旋转时互相撕。
    # 分配规则：像素归给**颜色距离最近**的那个部位；距离一样时按列表顺序。
    cands: List[Tuple[str, np.ndarray, float]] = []
    for p in parts:
        name = str(p.get("name") or p.get("id") or "部件")
        rgb = hex_rgb(str(p.get("color") or "#ffffff"))
        dist = _hex_dist(tarr, rgb)
        mask = (dist <= max(4, int(tolerance))) & (alpha > 8)
        cands.append((name, mask, float(dist.min())))
    if not cands:
        return []

    # 距离矩阵：每个像素到每个部位颜色的距离
    dists = np.stack([_hex_dist(tarr, hex_rgb(str(p.get("color") or "#ffffff")))
                       for p in parts])            # (K, H, W)
    # 像素归属：到哪个部位颜色最近就用哪个（只在该部位候选区内）
    owner = np.full((ah, aw), -1, dtype=np.int16)
    best = np.full((ah, aw), np.inf, dtype=np.float32)
    for k, (_n, mask, _mn) in enumerate(cands):
        upd = mask & (dists[k] < best)
        owner[upd] = k
        best[upd] = dists[k][upd]

    out: List[Tuple[str, Image.Image]] = []
    for k, (name, mask, _mn) in enumerate(cands):
        mask = (owner == k) & mask
        area = int(mask.sum())
        if area < 8:
            out.append((name, Image.new("RGBA", (aw, ah), (0, 0, 0, 0))))
            continue
        part = aarr.copy()
        part[..., 3] = np.where(mask, alpha, 0).astype(np.uint8)
        # 边缘 1px 羽化：部件接缝处硬切会在游戏里显出锯齿亮边
        img = Image.fromarray(part, "RGBA")
        if p.get("feather", True):
            img = _feather_edge(img)
        out.append((name, img))
    return out


def _fit_tag(tag: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """把色标图放到原图坐标系上。

    不能直接 resize 到原图尺寸：色标图可能是缩略图（前端在编辑器里用缩略图
    涂色），宽高比一旦不同，拉伸就会把涂的部位拉歪。这里按**宽度等比缩放
    再居中**，比例一致时等价于直接 resize，比例不同也不会变形。
    """
    tw, th = tag.size
    aw, ah = size
    if tw <= 0 or th <= 0:
        return tag.convert("RGB").resize(size)
    if (tw, th) == (aw, ah):
        return tag.convert("RGB")
    scale = aw / float(tw)
    nh = max(1, int(round(th * scale)))
    out = Image.new("RGB", size, (0, 0, 0))
    resized = tag.convert("RGB").resize((aw, nh), Image.NEAREST)
    out.paste(resized, (0, (ah - nh) // 2))
    return out


def _defringe(art: Image.Image) -> Image.Image:
    """把透明区的颜色用邻近的不透明颜色填掉（像素动画里的 de-fringe）。

    不做这一步，每个部件都会带着「原来那里是什么颜色」——半透明边缘在
    缩放/叠加时就会显出一条与内容无关的色边（fringe）。这里先用 inpaint
    把边界环一圈补上，再原样保留 alpha。
    """
    try:
        import cv2
    except Exception:  # pragma: no cover
        return art
    arr = np.asarray(art)
    a = arr[..., 3]
    solid = a > 8
    if not solid.any() or solid.all():
        return art
    # 只补「透明但紧邻不透明」的一圈，避免整张图被 inpaint 拖慢
    band = (cv2.dilate(solid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (~solid)
    if not band.any():
        return art
    out = arr.copy()
    rgb = cv2.inpaint(arr[..., :3], band.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    out[..., :3] = rgb
    return Image.fromarray(out, "RGBA")


def _feather_edge(img: Image.Image) -> Image.Image:
    """柔化边缘，但**不把实心区往外扩**。

    直接对 alpha 做高斯模糊，边界外那一圈会从 0 变成非 0——
    相邻部件于是各扩了 1px，实心像素都重叠了。两个部件抢同一块区域，
    旋转时就会互相撕。

    做法：模糊后取「原值与模糊值的较小者」，模糊只会让边缘更柔（0 → 更小
    是不可能的，实际是把过渡带收窄），实心区保持原样。
    """
    a = img.getchannel("A")
    from PIL import ImageFilter
    import numpy as _np

    blurred = a.filter(ImageFilter.GaussianBlur(0.6))
    ba = _np.asarray(blurred, dtype=_np.int16)
    oa = _np.asarray(a, dtype=_np.int16)
    # 关键：**限制在原区域内**。模糊会把区域外的 0 抬成几十，
    # 相邻部件于是各扩 1px、实心像素互相重叠（旋转时会撕）。
    #
    # 柔化怎么做：先做「腐蚀」——把最外一圈实心像素降为半透明，
    # 再模糊。这样边缘是从 0 渐变到实心的过渡带，而区域外仍是 0。
    # 单独模糊做不到：原图的 alpha 本来就是硬边（0 或 255），
    # 模糊值在区域外也非 0，限制在区域内就等于没有柔化。
    import cv2 as _cv2
    solid = (oa >= 250).astype(_np.uint8)
    eroded = _cv2.erode(solid, _np.ones((3, 3), _np.uint8), iterations=1)
    # 腐蚀掉的边界环：既不属于实心也不属于外部，标记为过渡带
    ring = (solid > 0) & (eroded == 0)
    ba = _np.asarray(a.filter(ImageFilter.GaussianBlur(0.6)), dtype=_np.int16)
    merged = _np.where(eroded > 0, oa, 0)        # 实心区不变
    merged = _np.where(ring, ba, merged)         # 过渡带取模糊值（会低于 255）
    merged = merged.clip(0, 255).astype(_np.uint8)
    out = img.copy()
    out.putalpha(Image.fromarray(merged, "L"))
    return out


def trim_part(img: Image.Image, pad: int = 2) -> Tuple[Image.Image, List[int]]:
    """把部件裁到自己的包围盒，返回 (裁后图, [x, y, w, h])。

    不裁的话每个部件都是「整张画布大小、只有一小块不透明」——在动画里
    直接用会出现：pivot 落在画布底部（部件悬在半空）、缩放按整幅算、
    包围盒重叠判断全乱。裁到自身包围盒后，pivot 才是「部件的落点」。

    pad 是往外留的透明边：紧贴裁会把边缘羽化一起切掉，部件接缝处显硬边。
    """
    a = np.asarray(img.getchannel("A"))
    if a.size == 0:
        return img, [0, 0, img.width, img.height]
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return img, [0, 0, img.width, img.height]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(img.width, int(xs.max()) + 1 + pad)
    y1 = min(img.height, int(ys.max()) + 1 + pad)
    return img.crop((x0, y0, x1, y1)), [x0, y0, x1 - x0, y1 - y0]


def coverage(name: str, img: Image.Image) -> float:
    """部件占原图面积的比例——太小的多半是用户手滑涂了一笔。"""
    a = np.asarray(img.getchannel("A"))
    if a.size == 0:
        return 0.0
    return float((a > 8).sum()) / float(a.size)

# ---------------------------------------------------------------------------
# 装骨架：把一组部件摆成「轨道 + 骨骼」
#
# 放在后端而不是前端：pivot 规则、父子推断、整体缩放都是数学，
# 前端再写一遍必然和渲染对不上（这项目已经吃过好几次亏）。
# ---------------------------------------------------------------------------

# pivot 取部件的哪个角：旋转的原点必须在「关节」上，不是几何中心。
# 头绕脖子转 → 下边中点；手臂绕肩转 → 上边中点；躯干绕胯转 → 下边中点。
_PIVOT_TOP_WORDS = ("臂", "手", "腿", "脚", "翼", "爪", "尾", "触", "鞭",
                    "arm", "hand", "leg", "foot", "wing", "tail")
_PIVOT_BOTTOM_WORDS = ("头", "脸", "帽", "head", "face", "hat")
_BODY_WORDS = ("身体", "躯干", "胸", "body", "torso", "chest")


def _pivot_for(name: str, w: int, h: int) -> List[int]:
    n = (name or "").lower()
    if any(k in n for k in _PIVOT_BOTTOM_WORDS):
        return [w // 2, h]
    if any(k in n for k in _PIVOT_TOP_WORDS):
        return [w // 2, 0]
    return [w // 2, h]


def plan_rig(parts: Sequence[Dict[str, Any]],
             canvas: Sequence[int]) -> List[Dict[str, Any]]:
    """给一组部件规划「骨骼位置 + pivot + 缩放 + 父子关系」。

    parts 每项：{id, name, size:[w,h], bbox:[x,y,w,h]}（bbox 是原图坐标）。
    返回每项：{id, name, bone:{x,y,len,parent}, pivot:[x,y], scale}。

    规则（都是「约定」而不是「真相」，所以写清楚）：
    - 整体按部件并集包围盒缩放放进画布，居中；
    - 面积最大的部件当躯干（root 的子骨骼），其余挂到躯干下；
    - 头部挂躯干；四肢/尾巴挂躯干；
    - pivot：头/躯干取下边中点，四肢与尾巴取上边中点——旋转绕关节。
    """
    items = [p for p in parts if p.get("size")]
    if not items:
        return []
    cw = int(canvas[0]) if canvas else 512
    ch = int(canvas[1]) if canvas else 512

    x0 = min(int(p["bbox"][0]) for p in items)
    y0 = min(int(p["bbox"][1]) for p in items)
    x1 = max(int(p["bbox"][0]) + int(p["bbox"][2]) for p in items)
    y1 = max(int(p["bbox"][1]) + int(p["bbox"][3]) for p in items)
    aw = max(1, x1 - x0)
    ah = max(1, y1 - y0)
    scale = min(cw * 0.88 / aw, ch * 0.88 / ah)
    ox = (cw - aw * scale) / 2 - x0 * scale
    oy = (ch - ah * scale) / 2 - y0 * scale

    def _area(p: Dict[str, Any]) -> int:
        return int(p["bbox"][2]) * int(p["bbox"][3])

    body = None
    for p in items:
        n = (p.get("name") or "").lower()
        if any(k in n for k in _BODY_WORDS):
            body = p
            break
    if body is None:
        body = max(items, key=_area)

    out: List[Dict[str, Any]] = []
    for p in items:
        w, h = int(p["size"][0]), int(p["size"][1])
        piv = _pivot_for(p.get("name") or "", w, h)
        bx = (int(p["bbox"][0]) + piv[0]) * scale + ox
        by = (int(p["bbox"][1]) + piv[1]) * scale + oy
        parent = None if p is body else "body"
        out.append({
            "id": p["id"],
            "name": p.get("name") or "部件",
            "bone": {"x": round(bx, 1), "y": round(by, 1),
                     "len": max(24, int(max(w, h) * scale)),
                     "parent": parent},
            "pivot": piv,
            "scale": round(scale, 4),
            "isBody": p is body,
        })
    return out
