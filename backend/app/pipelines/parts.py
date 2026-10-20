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
    tag_rgb = tag.convert("RGB").resize(art_rgba.size)
    tarr = np.asarray(tag_rgb)
    aarr = np.asarray(art_rgba)
    alpha = aarr[..., 3]

    # 涂得太满/没涂到的部件直接跳过，但要在返回里保留名字，
    # 让前端能告诉用户「头部那块你好像没涂」
    out: List[Tuple[str, Image.Image]] = []
    for p in parts:
        name = str(p.get("name") or p.get("id") or "部件")
        rgb = hex_rgb(str(p.get("color") or "#ffffff"))
        dist = _hex_dist(tarr, rgb)
        mask = (dist <= max(4, int(tolerance))) & (alpha > 8)
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
    """只对 alpha 做一次轻模糊，RGB 保持不变（模糊 RGB 会把边缘颜色污染到内部）。"""
    a = img.getchannel("A")
    from PIL import ImageFilter

    a = a.filter(ImageFilter.GaussianBlur(0.6))
    out = img.copy()
    out.putalpha(a)
    return out


def coverage(name: str, img: Image.Image) -> float:
    """部件占原图面积的比例——太小的多半是用户手滑涂了一笔。"""
    a = np.asarray(img.getchannel("A"))
    if a.size == 0:
        return 0.0
    return float((a > 8).sum()) / float(a.size)