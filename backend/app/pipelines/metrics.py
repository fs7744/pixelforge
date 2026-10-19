"""精灵表一致性指标：把「帧之间稳不稳」变成可读的数字。

移植自 game-asset-video-sprites 的 `metrics` 子命令（D:/code/gen）。

AI 逐帧生成的精灵表有两类问题肉眼很难判：帧间颜色漂移（每帧各自量化的
色板不同）和首尾衔接不齐（循环动画接缝跳一下）。这些都能算：

- ``seam_ratio`` 首尾帧差 / 平均帧间差。<1 说明首尾接得比普通相邻帧还紧，
  循环播放不会「顿」一下；明显大于 1 就是接缝没对齐。
- ``step_max`` / ``step_max_at`` 帧间差峰值与位置——某一对相邻帧差得离谱，
  通常是那帧姿势跑飞或抠图失败。
- ``off_palette_colors_max`` 锁色后仍不在色板里的颜色数（锁色后应为 0）。
- ``alpha_partial_ratio`` 半透明像素占比——像素素材必须是二值 alpha，
  这个数不为 0 说明抠图出了软边。
- ``fg_mean_color_std`` 前景平均色的逐帧标准差——帧间偏色的直接度量。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


def _mad(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(x - y).mean())


def sheet_metrics(
    frames: Sequence[Image.Image],
    palette: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """一组帧的一致性指标；帧数 < 2 时只返回基础信息。"""
    if not frames:
        raise ValueError("没有可计算的帧")
    ims = [np.asarray(f.convert("RGBA")).astype(np.float32) for f in frames]
    rgb = [im[:, :, :3] for im in ims]
    al = [im[:, :, 3] for im in ims]

    res: Dict[str, Any] = {
        "frames": len(frames),
        "size": list(ims[0].shape[1::-1]),
    }
    if len(ims) < 2:
        res["note"] = "只有一帧，无法计算帧间一致性"
        return res

    steps = [_mad(rgb[i], rgb[i + 1]) for i in range(len(rgb) - 1)]
    res["seam_first_last"] = _mad(rgb[0], rgb[-1])
    res["step_mean"] = round(float(np.mean(steps)), 3)
    res["step_max"] = round(float(np.max(steps)), 3)
    res["step_max_at"] = int(np.argmax(steps))
    res["seam_ratio"] = round(res["seam_first_last"] / max(res["step_mean"], 1e-6), 3)
    res["motion_peak_vs_first"] = round(float(max(_mad(r, rgb[0]) for r in rgb)), 3)

    if any((x < 255).any() for x in al):
        flick = [_mad(al[i], al[i + 1]) for i in range(len(al) - 1)]
        res["alpha_flicker_mean"] = round(float(np.mean(flick)), 3)
        res["alpha_partial_ratio"] = round(
            float(np.mean([((x > 0) & (x < 255)).mean() for x in al])), 4)

    if palette:
        palset = {tuple(int(v) for v in c) for c in palette}
        off = []
        for r_, a_ in zip(rgb, al):
            px = r_[a_ > 0].astype(np.uint8)
            uniq = {tuple(c) for c in np.unique(px.reshape(-1, 3), axis=0).tolist()}
            off.append(len(uniq - palset))
        res["off_palette_colors_max"] = int(max(off))

    means = [r_[a_ > 0].mean(0) if (a_ > 0).any() else np.zeros(3) for r_, a_ in zip(rgb, al)]
    res["fg_mean_color_std"] = round(float(np.std(np.stack(means), 0).mean()), 2)
    return res


def metrics_verdict(m: Dict[str, Any]) -> List[str]:
    """把指标翻译成用户能行动的一句话（没有结论的指标等于没算）。"""
    out: List[str] = []
    if "seam_ratio" in m:
        r = m["seam_ratio"]
        if r > 2.0:
            out.append(f"首尾衔接差（接缝是普通帧间差的 {r:.1f} 倍）：循环播放会顿一下，"
                       "建议让首末帧姿势一致，或去掉与首帧重复的末帧")
        elif r < 1.0:
            out.append(f"首尾衔接良好（接缝比普通相邻帧还紧，{r:.2f}）")
    if "off_palette_colors_max" in m and m["off_palette_colors_max"] > 0:
        out.append(f"锁色后仍有 {m['off_palette_colors_max']} 种色板外颜色：帧间会偏色")
    if m.get("alpha_partial_ratio", 0) > 0.01:
        out.append(f"半透明像素占 {m['alpha_partial_ratio'] * 100:.1f}%：像素素材应为二值 alpha，"
                   "检查抠图容差")
    if "step_max_at" in m and m.get("step_max", 0) > 3 * max(m.get("step_mean", 0), 1e-6):
        out.append(f"第 {m['step_max_at'] + 1} 帧与相邻帧差异过大：那一帧可能姿势跑飞或抠图失败")
    return out
