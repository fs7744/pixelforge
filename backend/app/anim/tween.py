"""编辑器内插帧：两种含义都做，但走同一套后端数学。

一、**关键帧插帧**（tween keyframes）
   两个关键帧之间补中间关键帧，值按**当前缓动曲线**求出来。
   为什么必须放后端：缓动曲线（cubic-bezier）在前后端各有一份实现，
   预览和导出必须完全一致。插出来的关键帧如果前端自己算，就会出现
   「预览里看着对、导出后曲线变形」——因为两边对 bezier 的求值精度不同。
   所以这里直接复用 render._sample（渲染用的那个），保证插出来的点
   落在渲染曲线上。

二、**位图插帧**（tween raster）
   序列帧轨道在相邻格子之间补中间帧（双向光流）。
   这是「工作流里能插帧、编辑器里不能」的那一半：工作流的 blend 步骤
   产出的是**新资产**（一格一格存进素材库），编辑器要的是**轨道内部**
   插帧，不新建资产。

两种插帧的共同点：都不新建资产、不改时间轴长度语义——插帧只是让
「这一段」的关键帧/格子变密，帧总数不变。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import render as render_mod
from . import sequence as seq_mod

# 一次插帧最多补多少个关键帧：240 帧 × 32 轨 = 7680 个点，
# 再多前端会卡住（每次改动都要重算曲线）
MAX_TWEEN_KEYS = 400


def _key_list(target: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    """按 kind 取关键帧数组（轨道 keyframes / 骨骼 keys / IK keys）。"""
    if kind == "track":
        return list(target.get("keyframes") or [])
    if kind == "bone":
        return list(target.get("keys") or [])
    if kind == "ik":
        return list(target.get("keys") or [])
    if kind == "warp":
        return list(target.get("warpKeys") or [])
    return []


def _write_keys(target: Dict[str, Any], kind: str, keys: List[Dict[str, Any]]) -> None:
    if kind == "track":
        target["keyframes"] = keys
    elif kind in ("bone", "ik"):
        target["keys"] = keys
    elif kind == "warp":
        target["warpKeys"] = keys


def tween_keys(target: Dict[str, Any], kind: str, count: int = 1,
               frame: Optional[int] = None) -> Dict[str, Any]:
    """在关键帧之间补中间关键帧。

    count = 每两个相邻关键帧之间补几个点。
    frame = 只处理到这个帧号为止（None = 全部）。

    补出来的点用**后一帧的缓动/曲线**求值（和 render._sample 同一套），
    所以插完之后曲线形状不变——只是从「两点直线」变成「多点拟合原曲线」，
    用户就能单独拖动中间某一点做局部调整，而不影响整体节奏。
    """
    keys = _key_list(target, kind)
    if len(keys) < 2:
        return {"added": 0, "total": len(keys), "reason": "关键帧不足 2 个，无需插帧"}
    keys = sorted(keys, key=lambda k: int(k.get("frame", 0)))
    count = max(1, min(int(count), MAX_TWEEN_KEYS))
    limit = int(frame) if frame is not None else None

    # 记录原始关键帧集合：插出来的点要能区分开，重复插帧时只补新的空位
    existing = {int(k.get("frame", 0)) for k in keys}
    added: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []

    for i, k in enumerate(keys):
        out.append(dict(k))
        if i >= len(keys) - 1:
            break
        nxt = keys[i + 1]
        f0, f1 = int(k.get("frame", 0)), int(nxt.get("frame", 0))
        if f1 - f0 < 2:
            continue                      # 相邻帧没有空位可插
        if limit is not None and f0 > limit:
            continue
        # 插几个点，均匀分布在 (f0, f1) 里
        for j in range(1, count + 1):
            f = f0 + round((f1 - f0) * j / (count + 1))
            if f <= f0 or f >= f1 or f in existing:
                continue
            existing.add(f)
            val = render_mod._sample(keys, f)   # noqa: SLF001
            newk: Dict[str, Any] = {"frame": f}
            if kind == "ik":
                newk["w"] = round(float(val.get("w", 1.0)), 3)
            elif kind == "warp":
                # 变形关键帧是 3×3 网格，用 _sample_warp 求值
                newk["warp"] = render_mod._sample_warp(target, f)  # noqa: SLF001
            else:
                newk["x"] = round(float(val.get("x", 0.0)), 2)
                newk["y"] = round(float(val.get("y", 0.0)), 2)
                newk["rot"] = round(float(val.get("rot", 0.0)), 2)
                newk["scale"] = round(float(val.get("scale", 1.0)), 4)
                if kind == "track":
                    newk["opacity"] = round(float(val.get("opacity", 1.0)), 4)
            # 缓动继承后一帧：插出来的点与它后面的点同一条曲线上
            newk["ease"] = nxt.get("ease") or "linear"
            if nxt.get("bez"):
                newk["bez"] = nxt["bez"]
            added.append(newk)

    out.extend(added)
    out.sort(key=lambda k: int(k.get("frame", 0)))
    _write_keys(target, kind, out)
    return {"added": len(added), "total": len(out)}


def tween_raster(sheet, seq: Dict[str, Any], per_gap: int = 1
                 ) -> Dict[str, Any]:
    """序列帧轨道的位图插帧：格子之间补中间格。

    返回新的 seq 布局（含格子对应表 map），**不新建资产**——扩展后的
    精灵表会作为新资产入库（因为一张图没法原地变长），但轨道本身
    记住 map，导出时能还原出「哪个格子是插出来的」。
    """
    per_gap = max(0, min(int(per_gap), 8))
    if per_gap <= 0:
        return {"seq": dict(seq), "added": 0}
    _sheet2, meta = seq_mod.build_interpolated(sheet, seq, per_gap)
    new_seq = dict(seq)
    new_seq["cols"] = meta.get("cols", seq.get("cols", 1))
    new_seq["rows"] = meta.get("rows", seq.get("rows", 1))
    new_seq["count"] = meta.get("count", seq.get("count", 1))
    new_seq["cellW"] = meta.get("cellW", seq.get("cellW", 0))
    new_seq["cellH"] = meta.get("cellH", seq.get("cellH", 0))
    new_seq["map"] = meta.get("map")
    new_seq["inbetween"] = {"perGap": per_gap, "method": meta.get("method", "flow")}
    # 循环段要跟着重算：插帧把格子数从 N 变成 ~2N，旧的格号会指到别的格。
    # 新表每一格都记着「源格序号」（meta.map[i].src），据此把起止格
    # 映射到新编号。映射不到（源格被裁掉）就退回全播。
    old_range = seq_mod.range_of(seq, int(seq.get("count") or 1))
    cmap = meta.get("map") or []
    new_start = new_stop = None
    for i, m in enumerate(cmap):
        src = int(m.get("src", -1)) if isinstance(m, dict) else -1
        if src == old_range[0]:
            new_start = i
        if src == old_range[1]:
            new_stop = i
    if new_start is not None and new_stop is not None and new_stop >= new_start:
        new_seq["range"] = [new_start, new_stop]
    else:
        new_seq["range"] = [0, max(0, int(new_seq["count"]) - 1)]
    return {"seq": new_seq, "added": meta.get("count", 0) - int(seq.get("count", 0)),
            "method": meta.get("method"), "sheet": _sheet2}


def estimate_tween(track: Dict[str, Any], kind: str = "track",
                   count: int = 1) -> Dict[str, Any]:
    """预估插帧会补几个点（给界面做提示，不改数据）。"""
    keys = _key_list(track, kind)
    if len(keys) < 2:
        return {"added": 0}
    keys = sorted(keys, key=lambda k: int(k.get("frame", 0)))
    existing = {int(k.get("frame", 0)) for k in keys}
    count = max(1, min(int(count), MAX_TWEEN_KEYS))
    added = 0
    for i in range(len(keys) - 1):
        f0, f1 = int(keys[i].get("frame", 0)), int(keys[i + 1].get("frame", 0))
        if f1 - f0 < 2:
            continue
        for j in range(1, count + 1):
            f = f0 + round((f1 - f0) * j / (count + 1))
            if f <= f0 or f >= f1 or f in existing:
                continue
            existing.add(f)
            added += 1
    return {"added": added}
