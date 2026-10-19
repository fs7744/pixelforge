"""光流插帧：相邻两帧之间生成中间帧（移植自 FrameRonin 的 opticalFlowInterpolate）。

为什么替掉 Image.blend：交叉淡化把两帧叠在一起，中间帧是一张「重影」——
角色的手臂会同时出现在两个位置，循环播放时看到的是鬼影而不是过渡。光流
先把每帧按运动方向 warp 到中间时刻再融合，肢体只有一条。

实现（cv2 Farneback 稠密光流）：
- 双向：分别算 I0→I1 与 I1→I0，两帧各自 warp 到 t=alpha 再加权融合，
  比单方向少一半拖影和空洞；
- 位移限幅：单像素位移超过画面短边 × max_motion 时按比例缩小整场，
  挡住光流在遮挡边界上的异常跳变；
- nearest=True：remap 用最近邻，像素素材不产生插值色；代价是运动边缘
  会缺一两颗像素，由后面的「锁色 / 二值 alpha」收尾。

没有 cv2 时调用方应退回交叉淡化（本模块不静默降级，直接抛错让调用方选）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Farneback 参数：pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
# balanced 档与 FrameRonin 默认一致；小图（角色 ≤256px）用不上 high 档
_FLOW_ARGS = (0.5, 4, 21, 5, 7, 1.5, 0)


def _warp(img: np.ndarray, flow: np.ndarray, nearest: bool) -> np.ndarray:
    """按 flow（dx, dy）把内容搬到「过了一半」的位置。

    符号是**减**不是加：remap 的 map 是「目标像素去源图哪里取」，内容在源图
    x 处、要搬到 x+d 处，就得让目标 x+d 去源图 x 取，即 map(x) = x - d。
    写成加号会把内容往反方向搬——实测（纯平移 24px）中间帧误差从 3 涨到 56，
    比不插还糟，而且不报任何错。
    """
    import cv2

    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[:, :, 0]
    map_y = grid_y - flow[:, :, 1]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(img, map_x, map_y, interp, borderMode=cv2.BORDER_REPLICATE)


def _flow(a_gray: np.ndarray, b_gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, *_FLOW_ARGS)


def _clamp_motion(flow: np.ndarray, max_motion: float) -> np.ndarray:
    """整场位移超过上限时按比例缩小，挡住遮挡边界的异常向量。"""
    if max_motion <= 0:
        return flow
    mag = np.sqrt((flow ** 2).sum(axis=2))
    limit = max_motion * max(flow.shape[0], flow.shape[1])
    peak = float(mag.max()) if mag.size else 0.0
    if peak > limit > 0:
        flow = flow * (limit / peak)
    return flow


def flow_mid(
    a: Image.Image,
    b: Image.Image,
    alpha: float = 0.5,
    *,
    nearest: bool = True,
    max_motion: float = 0.35,
) -> Tuple[Image.Image, dict]:
    """在 a（t=0）与 b（t=1）之间生成 t=alpha 的中间帧。"""
    import cv2

    if a.size != b.size:
        b = b.resize(a.size, Image.NEAREST)
    a_rgb = np.asarray(a.convert("RGB"))
    b_rgb = np.asarray(b.convert("RGB"))
    a_gray = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)

    f_ab = _clamp_motion(_flow(a_gray, b_gray), max_motion)
    f_ba = _clamp_motion(_flow(b_gray, a_gray), max_motion)

    wa = _warp(a_rgb, f_ab * alpha, nearest).astype(np.float32)
    wb = _warp(b_rgb, f_ba * (1.0 - alpha), nearest).astype(np.float32)
    out = wa * (1.0 - alpha) + wb * alpha

    mag = float(np.sqrt((f_ab ** 2).sum(axis=2)).mean())
    meta = {
        "method": "optical-flow-farneback",
        "alpha": round(float(alpha), 3),
        "nearest": bool(nearest),
        "meanMotionPx": round(mag, 2),
        "clamped": bool(mag > max_motion * max(a.size)),
    }
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), meta
