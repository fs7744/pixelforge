"""序列帧轨道 + 编辑器内插帧 的端到端验证（隔离数据目录，不污染真实素材库）。

验证链路：
1. 造一张 4×2 的精灵表（8 格，每格一个可辨认的色块 + 位置标记）
2. 用 /api/anim/projects/{pid}/import-sequence 导入成序列帧轨道
3. 渲染第 0/1/7 帧 → 确认每帧取到的是**不同的格子**（这是核心）
4. 关键帧插帧 → 确认补出来的关键帧落在渲染曲线上
5. 位图插帧（perGap）→ 确认格子数变多、渲染仍是正确的中间帧
6. 导出 ZIP/Godot → 确认能跑通

跑法：python backend/seq_verify.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# 必须在 import app 之前设好：测试期间所有资产写到临时目录
_TMP = tempfile.mkdtemp(prefix="pixelforge-seq-")
os.environ["PIXELFORGE_DATA_DIR"] = _TMP

from PIL import Image, ImageDraw  # noqa: E402

from app import storage  # noqa: E402
from app.anim import model as anim_model  # noqa: E402
from app.anim import render as anim_render  # noqa: E402
from app.anim import sequence as seq_mod  # noqa: E402
from app.anim import tween as tween_mod  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    tag = "OK  " if cond else "FAIL"
    if not cond:
        FAILED.append(name)
    print(f"[{tag}] {name}" + (f"  —— {detail}" if detail else ""))


def make_sheet(cols=4, rows=2, cw=32, ch=32):
    """每格一个纯色 + 左上角一个标记点，肉眼能看出取到的是哪格。"""
    sheet = Image.new("RGBA", (cols * cw, rows * ch), (0, 0, 0, 0))
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        color = (40 + i * 25, 200 - i * 15, 90 + i * 20, 255)
        tile = Image.new("RGBA", (cw, ch), color)
        # 标记点：位置随格子序号变化，用来验证取格顺序
        for j in range(i % 5 + 1):
            tile.putpixel((2 + j * 3, 2), (255, 255, 255, 255))
        sheet.alpha_composite(tile, (c * cw, r * ch))
    return sheet


def main():
    sheet = make_sheet()
    item = storage.save_asset(
        sheet, kind="sheet", name="verify-sheet",
        meta={"pipeline": "sprite-compose", "cols": 4, "frames": 8,
              "frameSize": {"w": 32, "h": 32}, "sheetSize": {"w": 128, "h": 64}},
    )
    aid = item["id"]
    print(f"素材 {aid}  {sheet.size}")

    # ---------- 1. 布局推断 ----------
    lay = seq_mod.layout_from_meta(item.get("meta"))
    check("从 meta 推断出 4×2 共 8 格", lay is not None and lay["cols"] == 4
          and lay["count"] == 8, str(lay))

    # ---------- 2. 裁格子 ----------
    boxes = seq_mod.cell_boxes(sheet, lay)
    check("切出 8 个格子", len(boxes) == 8, f"{len(boxes)}")
    c0 = seq_mod.cut(sheet, lay, 0)
    c1 = seq_mod.cut(sheet, lay, 1)
    check("第 0 格与第 1 格内容不同", list(c0.getdata()) != list(c1.getdata()))
    check("格子尺寸 = 32×32", c0.size == (32, 32), str(c0.size))

    # ---------- 3. 工程 + 序列帧轨道 ----------
    proj = anim_model.create("序列验证", canvas=(256, 256), frames=8)
    pid = proj["id"]
    track = {
        "id": "seqtest", "name": "动作序列", "assetId": aid, "bone": "root",
        "pivot": [16, 16],
        "seq": {"cols": 4, "rows": 2, "count": 8, "cellW": 32, "cellH": 32, "loop": True},
        "keyframes": [{"frame": 0, "x": 0, "y": 0, "rot": 0, "scale": 1,
                       "opacity": 1, "ease": "linear"}],
    }
    proj["tracks"] = [track]
    proj = anim_model.normalize(proj)
    check("normalize 保留 seq 字段", bool(proj["tracks"][0].get("seq")),
          str(proj["tracks"][0].get("seq")))

    def loader(a):
        it = storage.get_asset(a)
        return Image.open(storage.asset_path(it)).convert("RGBA") if it else None

    # ---------- 3b. 无 meta 时从内容检测网格 ----------
    def _mk(cols, rows, cw=64, ch=64):
        sh = Image.new("RGBA", (cols * cw, rows * ch), (0, 0, 0, 0))
        for i in range(cols * rows):
            r0, c0 = divmod(i, cols)
            cell = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            dr = ImageDraw.Draw(cell)
            x, y = cw // 2 - 6 + (i % 3), ch // 2 - 18
            dr.ellipse((x + 4, y, x + 16, y + 12), fill=(240, 200, 160, 255))
            dr.rectangle((x, y + 12, x + 20, y + 30), fill=(200, 60, 60, 255))
            sh.alpha_composite(cell, (c0 * cw, r0 * ch))
        return sh

    bad = []
    for c_n, r_n in [(4, 2), (4, 1), (8, 1), (3, 1), (2, 2), (6, 1), (2, 3)]:
        g = seq_mod.detect_grid(_mk(c_n, r_n))
        if not g or g["cols"] != c_n or g["rows"] != r_n:
            bad.append((c_n, r_n, g))
    check("从内容检测网格（7 种布局全对）", not bad, str(bad))
    check("单格图检测不出（当普通轨道）", seq_mod.detect_grid(_mk(1, 1)) is None)

    # ---------- 3c. 循环段 ----------
    # 只播 [start, stop] 这几格：替代「回工作流重跑一次 sprite-select」
    seq_full = {"cols": 4, "rows": 2, "count": 8, "cellW": 32, "cellH": 32, "loop": True}
    check("未设循环段 → 全播",
          [seq_mod.frame_index(seq_full, f, 8, 8) for f in range(8)] == list(range(8)))
    seq_r = dict(seq_full)
    seq_r["range"] = [1, 5]
    got = [seq_mod.frame_index(seq_r, f, 8, 8) for f in range(8)]
    check("选第 2–6 格 → 只在这 5 格里循环", set(got) == {1, 2, 3, 4, 5}, str(got))
    check("循环段起点正确", got[0] == 1, str(got))
    check("循环段终点正确", got[-1] == 5, str(got))
    # 异常值兜底
    check("起止反了 → 自动交换", seq_mod.range_of({"range": [5, 1]}, 8) == (1, 5))
    check("越界 → 夹回合法范围", seq_mod.range_of({"range": [-3, 99]}, 8) == (0, 7))
    check("垃圾值 → 退化成全播", seq_mod.range_of({"range": "x"}, 8) == (0, 7))

    # ---------- 3d. 循环段随位图插帧重映射 ----------
    # 插帧把格子数从 8 变成 15，旧的循环段格号会指到别的格 —— 必须跟着重算
    _seq_before = dict(seq_r)
    _res = tween_mod.tween_raster(sheet, _seq_before, per_gap=1)
    _ns = _res["seq"]
    _cmap = _ns.get("map") or []
    check("插帧后循环段格号已重算",
          _ns.get("range") != _seq_before.get("range")
          and _ns.get("range") != [1, 5],
          f"{_seq_before.get('range')} → {_ns.get('range')}")
    _ok = (len(_cmap) > _ns["range"][1]
           and _cmap[_ns["range"][0]].get("src") == 1
           and _cmap[_ns["range"][1]].get("src") == 5)
    check("重映射后的段两端仍指向原来的源格", _ok,
          f"src={[ _cmap[i].get('src') for i in _ns['range'] ] if len(_cmap) > _ns['range'][1] else 'N/A'}")

    # ---------- 3e. 循环接缝诊断 ----------
    def _cell_at(x):
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        dr.ellipse((x + 18, 10, x + 30, 22), fill=(240, 200, 160, 255))
        dr.rectangle((x + 14, 22, x + 34, 48), fill=(200, 60, 60, 255))
        return im

    from app.anim import diagnose as diag_mod
    walk = diag_mod.seam_report([_cell_at(0), _cell_at(5), _cell_at(0), _cell_at(5)])
    check("走路循环：接缝与普通帧间差持平（不顿）",
          walk["seam_ratio"] < 1.2 and not walk["verdict"],
          f"ratio={walk['seam_ratio']} {walk['verdict']}")
    jump = diag_mod.seam_report([_cell_at(0), _cell_at(5), _cell_at(0), _cell_at(30)])
    check("接缝跳变：报「循环会顿一下」",
          any("顿一下" in v for v in jump["verdict"]), str(jump["verdict"]))
    off = diag_mod.seam_report([_cell_at(0), _cell_at(5), _cell_at(60), _cell_at(5)])
    check("姿势跑飞：指出具体是哪一格",
          any("第 3 格" in v for v in off["verdict"]), str(off["verdict"]))
    check("一格时没有接缝可言",
          "note" in diag_mod.seam_report([_cell_at(0)]))
    check("空序列不崩", diag_mod.seam_report([]).get("frames") == 0)

    # 诊断只针对循环段：选了 2-6 格，接缝就该是「第 6 格 → 第 2 格」
    _sheet2 = sheet
    _cells, _s, _e = diag_mod.cut_range(_sheet2, seq_r)
    check("诊断只取循环段内的格子", len(_cells) == 5 and _s == 1 and _e == 5,
          f"{len(_cells)} 格 [{_s},{_e}]")

    # ---------- 4. 逐帧取到不同格子 ----------
    # 格子贴在根骨骼原点处（pivot=格子中心 → 格子中心落在骨骼原点），
    # 所以取样点取「根骨骼原点 - 格子半径」附近：格子左上角那一块。
    # 不能取 (16,16)：那是画布左上角，格子根本不在那儿。
    def sample_at(img):
        # 找画面里第一个不透明像素的色值（比猜坐标可靠）
        arr = img.getchannel("A")
        bbox = arr.getbbox()
        if not bbox:
            return None
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        return img.getpixel((cx, cy))

    pixels = []
    for f in range(8):
        img = anim_render.render_frame(proj, f, loader)
        pixels.append(sample_at(img))
    uniq = len(set(pixels))
    check("8 帧取到 8 个不同格子", uniq == 8 and all(p is not None for p in pixels),
          f"唯一色 {uniq}：{pixels[:4]}…")

    # 循环：序列均匀铺满时间轴（cell = frame × count ÷ total）
    check("count==total 时 1:1 对应", seq_mod.frame_index({"loop": True}, 3, 8, 8) == 3)
    check("8 格铺 24 帧 → 每格 3 帧（慢速播 1 遍）",
          seq_mod.frame_index({"loop": True}, 0, 8, 24) == 0
          and seq_mod.frame_index({"loop": True}, 3, 8, 24) == 1
          and seq_mod.frame_index({"loop": True}, 23, 8, 24) == 7)
    # 位图插帧后 15 格铺 8 帧 → 从 15 格里均匀取，插出来的中间帧能看到
    idx15 = [seq_mod.frame_index({"loop": True}, f, 15, 8) for f in range(8)]
    check("15 格铺 8 帧：用到了插出来的中间格（索引 > 7）",
          any(i > 7 for i in idx15), str(idx15))
    check("loop=False 时停在最后一格", seq_mod.frame_index({"loop": False}, 99, 8, 8) == 7)

    # ---------- 5. 关键帧插帧 ----------
    t = proj["tracks"][0]
    t["keyframes"] = [
        {"frame": 0, "x": 0, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
        {"frame": 7, "x": 70, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
    ]
    res = tween_mod.tween_keys(t, "track", 2)
    check("关键帧插帧补出中间点", res["added"] == 2, str(res))
    frames = [k["frame"] for k in t["keyframes"]]
    check("插出的帧号落在 (0,7) 内且不重复",
          sorted(frames) == [0, 2, 5, 7], str(sorted(frames)))
    # 值必须落在直线上（linear 缓动，x 从 0 到 70）
    mid = [k for k in t["keyframes"] if k["frame"] == 2][0]
    check("插出的关键帧值落在原曲线上", abs(mid["x"] - 70 * 2 / 7) < 0.5,
          f"x={mid['x']} 期望≈{70 * 2 / 7:.2f}")

    # 缓动继承：后一帧有 bez 时插出的点也带 bez
    t2 = {"keyframes": [
        {"frame": 0, "x": 0, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
        {"frame": 8, "x": 80, "y": 0, "rot": 0, "scale": 1, "opacity": 1,
         "ease": "ease-in", "bez": [0.42, 0, 0.58, 1]},
    ]}
    tween_mod.tween_keys(t2, "track", 1)
    check("插出的点继承后一帧的 bez", bool(t2["keyframes"][1].get("bez")),
          str(t2["keyframes"][1]))

    # 已有中间关键帧时，不会插到**已占用的帧号**上：
    # 段(0,4) 插 1 个 → 候选 f=2（合法新点）；段(4,8) 插 1 个 → 候选 f=6。
    # 但若把 count 提到 3，段(0,4) 的候选是 1,2,3 —— 都不能撞上已有关键帧 4。
    t3 = {"keyframes": [
        {"frame": 0, "x": 0, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
        {"frame": 4, "x": 40, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
        {"frame": 8, "x": 80, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
    ]}
    r3 = tween_mod.tween_keys(t3, "track", 1)
    f3 = [k["frame"] for k in t3["keyframes"]]
    check("插出的点不与已有关键帧撞号", len(set(f3)) == len(f3), str(f3))
    check("原有关键帧一个没丢", all(f in f3 for f in (0, 4, 8)), str(f3))

    # 相邻帧（间隔 1）没有中间位置，不该插
    t4 = {"keyframes": [
        {"frame": 0, "x": 0, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
        {"frame": 1, "x": 10, "y": 0, "rot": 0, "scale": 1, "opacity": 1, "ease": "linear"},
    ]}
    r4 = tween_mod.tween_keys(t4, "track", 3)
    check("相邻关键帧之间插不进（无空位）", r4["added"] == 0, str(r4))

    # ---------- 6. 位图插帧 ----------
    sheet2, meta2 = seq_mod.build_interpolated(sheet, lay, per_gap=1)
    check("位图插帧后格子数 8 → 15", meta2["count"] == 15, str(meta2["count"]))
    check("插帧后仍是 4 列", meta2["cols"] == 4, str(meta2["cols"]))
    # 中间格不该等于任一端点（否则说明没真插）
    mid_cell = seq_mod.cut(sheet2, {"cols": 4, "rows": 4, "count": 15, "cellW": 32, "cellH": 32}, 1)
    end_cell = seq_mod.cut(sheet, lay, 0)
    check("中间格与端点不同", list(mid_cell.getdata()) != list(end_cell.getdata()))

    # ---------- 7. 序列帧轨道 + 骨骼/蒙皮叠加 ----------
    proj2 = anim_model.normalize({
        "id": pid, "name": "叠加验证", "canvas": [256, 256], "fps": 12, "frames": 8,
        "bones": [{"id": "root", "name": "根", "parent": None, "x": 128, "y": 200}],
        "tracks": [{
            "id": "sk", "name": "蒙皮序列", "assetId": aid, "bone": "root",
            "pivot": [16, 16],
            "seq": {"cols": 4, "rows": 2, "count": 8, "cellW": 32, "cellH": 32, "loop": True},
            "skin": {"grid": 4, "bones": ["root"], "falloff": 60},
            "keyframes": [{"frame": 0, "x": 0, "y": 0, "rot": 0, "scale": 1,
                           "opacity": 1, "ease": "linear"}],
        }],
    })
    img = anim_render.render_frame(proj2, 0, loader)
    check("序列帧 + 蒙皮能渲染", img is not None and img.size == (256, 256))
    check("蒙皮后画面非空", img.getchannel("A").getbbox() is not None)

    # ---------- 8. 导出 ----------
    from app.anim import export as export_mod
    zf = export_mod.build_zip(proj2, lambda a: storage.asset_path(storage.get_asset(a)))
    check("序列帧工程能导出 ZIP", len(zf) > 0, f"{len(zf)} 字节")
    import io, json, zipfile
    with zipfile.ZipFile(io.BytesIO(zf)) as z:
        names = z.namelist()
        data = json.loads(z.read("anim.json"))
    check("导出包含 anim.json 与贴图",
          "anim.json" in names and any(n.startswith("textures/") for n in names))

    # ---------- 9. 清理 ----------
    anim_model.delete(pid)
    print()
    if FAILED:
        print(f"失败 {len(FAILED)} 项：{FAILED}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
