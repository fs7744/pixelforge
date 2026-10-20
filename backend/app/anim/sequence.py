"""序列帧轨道：一条轨道挂一张精灵表，按帧号取对应格子。

为什么需要它
------------
工作流（sprite_loop / sprite-compose）产出的是**一张精灵表**——8 个动作帧
拼在一张图里，meta 里有 cols / frames / frameSize。但动画编辑器的轨道
原来只能挂**一张**贴图：把精灵表当普通贴图用，整条动画显示的都是
8 格拼在一起的大图。

于是「生成 → 编辑」断在这里：工作流跑完了，产物进不了编辑器精修。
本模块补上这一段：轨道带 seq 字段时，渲染时按帧号裁出对应格子。

格子布局从哪来
--------------
优先读资产 meta（sprite-compose 写的有 cols / frameSize / index），
读不到就退回「单格铺满」——用户也可以只给一张单帧图当 1 格的序列。

坐标语义
--------
seq 轨道渲染时，先裁出格子（cell），再把 cell 当作普通贴图走
骨骼 / 蒙皮 / 变形 / 混合。所以 pivot 用的是**格子内坐标**，
不是整张精灵表的坐标。这一点和普通轨道不同，界面上要说明。

插帧（in-between）
-----------------
两个相邻格子之间插入中间帧，用双向光流把两格 warp 到中间时刻再融合
（移植自 pipelines/inbetween.py 的做法）。为什么不用交叉淡化：淡化出来的
中间帧是「重影」——角色的手同时出现在两个位置，循环播放时看到的是鬼影。

光流只在这里用一次（生成中间帧），不做逐帧实时计算，所以可以接受它的开销。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def layout_from_meta(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """从资产 meta 推断精灵表布局。

    sprite-compose 写的 meta 形如：
        {"cols": 4, "frames": 8, "frameSize": {"w":64,"h":64}, "sheetSize": {...}}

    这些字段是工作流自己写的，所以这里读它是**同一份真相**，不用猜
    「这图是几列几行」。猜错的话切出来的格子会串行（拿到相邻格的像素）。
    """
    if not isinstance(meta, dict):
        return None
    cols = int(_num(meta.get("cols"), 0))
    frames = int(_num(meta.get("frames"), 0))
    fs = meta.get("frameSize") or {}
    fw = int(_num(fs.get("w"), 0))
    fh = int(_num(fs.get("h"), 0))
    if cols < 1 or frames < 1 or fw < 1 or fh < 1:
        return None
    rows = (frames + cols - 1) // cols
    return {"cols": cols, "rows": rows, "count": frames, "cellW": fw, "cellH": fh}


def cell_boxes(sheet: Image.Image, seq: Dict[str, Any]) -> List[Tuple[int, int, int, int]]:
    """按布局算出每个格子的裁切框。

    格子宽高优先用 seq 里写死的值（cellW/cellH），没有才按整图均分——
    均分是**兜底**，不是主路径：sprite 表常常最后一格是空的（不满格），
    均分会把空白也算成帧，切出来是全透明图。
    """
    cols = max(1, int(_num(seq.get("cols"), 1)))
    rows = max(1, int(_num(seq.get("rows"), 1)))
    count = max(1, int(_num(seq.get("count"), cols * rows)))
    cw = int(_num(seq.get("cellW"), 0))
    ch = int(_num(seq.get("cellH"), 0))
    if cw < 1 or ch < 1:
        cw = max(1, sheet.width // cols)
        ch = max(1, sheet.height // rows)
    boxes = []
    for i in range(count):
        r, c = divmod(i, cols)
        x, y = c * cw, r * ch
        if x >= sheet.width or y >= sheet.height:
            break
        boxes.append((x, y, min(x + cw, sheet.width), min(y + ch, sheet.height)))
    return boxes


def cut(sheet: Image.Image, seq: Dict[str, Any], index: int) -> Image.Image:
    """裁出第 index 格。越界返回 1×1 透明图（不抛错，渲染时好降级）。"""
    boxes = cell_boxes(sheet, seq)
    if not boxes:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    if index < 0:
        index = 0
    if index >= len(boxes):
        index = len(boxes) - 1
    return sheet.convert("RGBA").crop(boxes[index])


def range_of(seq: Dict[str, Any], count: int) -> Tuple[int, int]:
    """序列帧的循环段（闭区间）：[start, stop]。

    为什么需要：工作流的 sprite-select 能在**资产级**取一段当循环
    （「不插帧、不跳帧、不改时长」），但那要重跑一遍工作流。编辑器里
    导入了 8 格动作表后，想只播第 2–6 格，不该为此回到工作流重来。

    返回的区间一定落在 [0, count-1] 内，且 start <= stop
    （给错值就退化成全播，而不是崩掉或播出空段）。
    """
    if count < 1:
        return (0, 0)
    r = seq.get("range")
    if not isinstance(r, (list, tuple)) or len(r) < 2:
        return (0, count - 1)
    try:
        start = int(r[0])
        stop = int(r[1])
    except (TypeError, ValueError):
        return (0, count - 1)
    start = max(0, min(start, count - 1))
    stop = max(0, min(stop, count - 1))
    if stop < start:
        start, stop = stop, start
    return (start, stop)


def frame_index(seq: Dict[str, Any], frame: int, count: int,
                total: Optional[int] = None) -> int:
    """帧号 → 格子序号。

    序列帧均匀铺满整条时间轴：cell = frame × count ÷ total。

    为什么不用 `frame % count`（每格一帧、循环）：
    - 8 格配 24 帧时间轴时，取模是「快速播 3 遍」，而铺满是「慢速播 1 遍」。
      后者才是延长时间轴想要的。
    - 更要紧的是**位图插帧之后**：8 格插成 15 格，时间轴还是 8 帧。
      取模的话只显示前 8 格（正好是原始帧），插出来的中间帧一个都看不到；
      铺满则从 15 格里均匀取，画面真的变顺了。

    count == total 时就是 1:1 对应（最常见的情况，行为和取模一致）。

    选了循环段（seq.range）时，**段内的格子**均匀铺满时间轴，段外不参与
    —— 这就是「只播这一段当循环」。
    """
    if count < 1:
        return 0
    start, stop = range_of(seq, count)
    seg = stop - start + 1                       # 段内格子数
    if total is None or total < 1:
        # 没有总帧数时退回「每格一帧 + 循环」
        if seq.get("loop", True):
            return start + (int(frame) % seg)
        return start + max(0, min(int(frame), seg - 1))
    idx = int(frame) * seg // int(total)
    idx = max(0, min(idx, seg - 1))
    if not seq.get("loop", True):
        # 不循环：一次性动作（受击 → 倒地），时间轴尾部停在最后一格
        idx = min(idx, seg - 1)
    return start + idx


def inbetween(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    """两个相邻格子之间生成中间帧（RGBA）。

    有 cv2 就用双向光流；没有就退回交叉淡化。**不静默降级**：返回的图
    会带上标记吗？不会——但调用方拿到的画面差别很大（重影 vs 真运动），
    所以这里把「用了哪种方法」写进 track 的 meta 里，界面上提示用户。
    """
    alpha = max(0.0, min(float(alpha), 1.0))
    if a.size != b.size:
        b = b.resize(a.size, Image.NEAREST)
    try:
        from ..pipelines import inbetween as inbetween_pipe

        mid, _meta = inbetween_pipe.flow_mid(a, b, alpha, nearest=True)
        return mid.convert("RGBA")
    except Exception:
        # cv2 不可用：交叉淡化。透明度单独线性插值（RGBA 直接 blend 会
        # 把「有没有像素」和「多亮」搅在一起，边缘会糊成一团灰）。
        out = Image.blend(a.convert("RGB"), b.convert("RGB"), alpha).convert("RGBA")
        aa = a.getchannel("A")
        ab = b.getchannel("A")
        out.putalpha(Image.blend(aa, ab, alpha))
        return out


def build_interpolated(sheet: Image.Image, seq: Dict[str, Any],
                        per_gap: int = 1) -> Tuple[Image.Image, Dict[str, Any]]:
    """把序列扩成「每格之间再插 per_gap 帧」的新精灵表。

    返回（新精灵表, 元信息）。元信息里有原始格子数、扩展后的格子数、
    每个新格子对应的「源格序号 / 混合比例」，让编辑器能画出格子对应关系。
    """
    per_gap = max(0, int(per_gap))
    boxes = cell_boxes(sheet, seq)
    cells = [sheet.convert("RGBA").crop(bx) for bx in boxes]
    if not cells:
        return sheet, {"count": 0, "perGap": per_gap, "map": [], "method": "none"}
    if per_gap <= 0 or len(cells) < 2:
        return sheet, {"count": len(cells), "perGap": 0,
                       "map": [{"src": i, "alpha": 0.0} for i in range(len(cells))],
                       "method": "none"}

    out_cells: List[Image.Image] = []
    cmap: List[Dict[str, Any]] = []
    method = "flow"
    for i, c in enumerate(cells):
        out_cells.append(c)
        cmap.append({"src": i, "alpha": 0.0})
        if i >= len(cells) - 1:
            continue
        nxt = cells[i + 1]
        for g in range(per_gap):
            alpha = (g + 1) / (per_gap + 1)
            try:
                mid = inbetween(c, nxt, alpha)
            except Exception:
                method = "blend"
                mid = inbetween(c, nxt, alpha)
            out_cells.append(mid)
            cmap.append({"src": i, "alpha": round(alpha, 4)})

    fw, fh = out_cells[0].size
    cols = max(1, int(_num(seq.get("cols"), 1)))
    rows = (len(out_cells) + cols - 1) // cols
    sheet2 = Image.new("RGBA", (fw * cols, fh * rows), (0, 0, 0, 0))
    for i, im in enumerate(out_cells):
        r, c = divmod(i, cols)
        sheet2.alpha_composite(im, (c * fw, r * fh))
    return sheet2, {"count": len(out_cells), "perGap": per_gap, "cols": cols, "rows": rows,
                    "cellW": fw, "cellH": fh, "map": cmap, "method": method}


def detect_grid(img: Image.Image) -> Optional[Dict[str, Any]]:
    """从图片内容**猜**精灵表的行列（meta 读不到时的兜底）。

    原理：拼出来的精灵表，每格里的角色是居中的，格子边界附近通常是空的。
    所以沿 x / y 方向看「这一列/行有没有内容」，会得到一串**内容段**。
    段与段之间的空隙就是格子边界，**段之间的距离（周期）**就是格子尺寸。

    关键：格子尺寸是**周期**，不是段自身的宽度。
    早先按段宽算，64px 的格里放个 20px 宽的角色 → 算出 12 列，
    切出来全是碎片。周期才是对的。

    周期取「相邻段起点之差」的中位数：一排 8 格的角色起落位置略有差异，
    中位数能把它们对齐到真实格宽。

    局限：角色贴边、或格子之间内容连成一片时数不准。所以这条路只做**兜底**，
    优先还是读 meta（工作流自己写的 cols/frames 才是准的）。
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return None
    arr = np.asarray(img.convert("RGBA"))
    if arr.ndim != 3 or arr.shape[2] < 4:
        return None
    alpha = arr[..., 3]

    def segments(profile: np.ndarray) -> List[Tuple[int, int]]:
        """把「有内容」的连续段找出来，返回 [(起, 止), ...]（左闭右开）。"""
        segs: List[Tuple[int, int]] = []
        start = -1
        for i, v in enumerate(profile):
            if v > 0 and start < 0:
                start = i
            elif v == 0 and start >= 0:
                segs.append((start, i))
                start = -1
        if start >= 0:
            segs.append((start, len(profile)))
        return segs

    def period(segs: List[Tuple[int, int]], total: int) -> int:
        """从内容段推出格子尺寸（周期）。

        只有一段时推不出周期：单行表沿 y 方向整行都有内容，就是一段。
        这时行数必然是 1，格子高度 = 整图高（交给调用方处理）。
        """
        if len(segs) < 2:
            return 0
        diffs = [segs[i + 1][0] - segs[i][0] for i in range(len(segs) - 1)]
        srt = sorted(diffs)
        med = srt[len(srt) // 2]
        return med if med > 0 else 0

    cols_segs = segments((alpha > 0).any(axis=0))
    rows_segs = segments((alpha > 0).any(axis=1))
    cw = period(cols_segs, img.width)
    # 行方向只有一段 → 单行表，格高就是整图高
    ch = period(rows_segs, img.height) or (img.height if rows_segs else 0)
    if cw < 1 or ch < 1:
        return None
    cols = max(1, min(32, int(round(img.width / cw))))
    rows = max(1, min(32, int(round(img.height / ch))))
    if cols * rows > 512:
        return None
    return {"cols": cols, "rows": rows, "count": cols * rows,
            "cellW": img.width // cols, "cellH": img.height // rows}


def normalize_seq(v: Any, canvas: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """清洗 seq 配置。写错/缺字段就整体丢弃（宁可当普通轨道，也不要半坏的）。"""
    if not isinstance(v, dict):
        return None
    cols = int(_num(v.get("cols"), 0))
    if cols < 1:
        return None
    count = int(_num(v.get("count"), 0))
    if count < 1:
        return None
    rows = int(_num(v.get("rows"), 1)) or 1
    cw = int(_num(v.get("cellW"), 0))
    ch = int(_num(v.get("cellH"), 0))
    if cw < 1 or ch < 1:
        cw = int(_num(v.get("frameW"), 0)) or 0
        ch = int(_num(v.get("frameH"), 0)) or 0
    return {
        "cols": max(1, min(cols, 64)),
        "rows": max(1, min(rows, 64)),
        "count": max(1, min(count, 4096)),
        "cellW": max(0, min(cw, 4096)),
        "cellH": max(0, min(ch, 4096)),
        "loop": bool(v.get("loop", True)),
        # 循环段（闭区间，格子序号）：只播这一段。越界会在 frame_index
        # 里被夹回合法范围，这里先夹一次免得存下垃圾值。
        "range": _norm_range(v.get("range"), max(1, min(count, 4096))),
        # 插帧后的格子对应表：有它才能把扩展格子映射回源格
        "map": v.get("map") if isinstance(v.get("map"), list) else None,
    }


def _norm_range(v: Any, count: int) -> List[int]:
    """清洗循环段：必须是两个整数、start <= stop、都在 [0, count-1] 内。"""
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return [0, max(0, count - 1)]
    try:
        start = int(v[0])
        stop = int(v[1])
    except (TypeError, ValueError):
        return [0, max(0, count - 1)]
    start = max(0, min(start, count - 1))
    stop = max(0, min(stop, count - 1))
    if stop < start:
        start, stop = stop, start
    return [start, stop]
