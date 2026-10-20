"""MVP 冒烟测试：不依赖前端与模型，直接验证确定性管线。

    python backend/smoke_test.py

测试素材由本脚本自己画（4×4 纯色块网格 + 纯色底），不经过任何
生成模型——模拟引擎已移除，管线测试也不该依赖 GPU。
"""
from __future__ import annotations

import asyncio
import io
import math
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

# 测试数据落临时目录：绝不碰用户素材库（详见 coverage_api_test.py 同一段注释）
import tempfile  # noqa: E402

_TMP_DATA = Path(tempfile.mkdtemp(prefix="pf-test-data-"))
os.environ["PIXELFORGE_DATA_DIR"] = str(_TMP_DATA)

import atexit  # noqa: E402
import shutil  # noqa: E402

atexit.register(lambda: shutil.rmtree(_TMP_DATA, ignore_errors=True))

from backend.app import storage  # noqa: E402
from backend.app import godot_export  # noqa: E402
from backend.app.pipelines import matte, resize, sheet

OUT = Path(__file__).resolve().parents[1] / "data" / "tmp"
OUT.mkdir(parents=True, exist_ok=True)

CELL = 32
BG = (42, 21, 63)  # 暗紫底，与管线默认一致
BLOCKS = [
    (0, 255, 136), (255, 51, 102), (91, 87, 235), (255, 213, 0),
    (10, 10, 10), (255, 255, 255), (255, 180, 162), (0, 200, 200),
]


def make_fake_sheet() -> Image.Image:
    """画一张 4×4 的假精灵表：每格一个纯色方块 + 黑描边。"""
    img = Image.new("RGB", (4 * CELL, 4 * CELL), BG)
    d = ImageDraw.Draw(img)
    for r in range(4):
        for c in range(4):
            ox, oy = c * CELL, r * CELL
            d.rectangle([ox + 6, oy + 6, ox + 25, oy + 25],
                        fill=BLOCKS[(r * 4 + c) % len(BLOCKS)],
                        outline=(10, 10, 10))
    return img


async def main() -> None:
    # 基线：跑完把本轮新造的资产全删掉。以前只删「登记过的那几个」，
    # 管线内部产出的帧/表/图层全留了下来，跑一次胖几十张。
    _base = {a["id"] for a in storage.list_assets()}
    sheet_img = make_fake_sheet()
    sheet_img.save(OUT / "smoke-sheet.png")
    print(f"[1] 素材     → {sheet_img.size} 4×4 合成网格")

    cut, meta = matte.connected_matte(sheet_img, tolerance=80, feather=5)
    cut.save(OUT / "smoke-matte.png")
    print(f"[2] matte    → bg={meta['background']} removed={meta['removedRatio']*100:.1f}%")

    small = resize.hard_resize(cut, (144, 144))
    rpg, rmeta = resize.rpgmaker(cut, "v1")
    rpg.save(OUT / "smoke-rpgmaker.png")
    print(f"[3] resize   → hard={small.size} rpgmaker={rmeta['size']}")

    frames, index = sheet.split_sheet(sheet_img, 4, 4)
    print(f"[4] split    → {len(frames)} frames, each {index['frame_size']['w']}×{index['frame_size']['h']}")

    merged, _ = sheet.compose_sheet(frames, 4)
    print(f"[5] compose  → {merged.size}")

    # 透明区 RGB 清零校验（移植自 h3 的 a[~visible,:3]=0）：必须用抠过图的
    # 帧来检验——不透明网格走不到这条路径。合成表空白区 RGB 必须为 0，
    # 否则透明像素携带的底色会被引擎采样成晕边
    cut_frames, _ = sheet.split_sheet(cut, 4, 4)
    merged2, _ = sheet.compose_sheet(cut_frames, 4)
    m2 = np.asarray(merged2.convert("RGBA"))
    t2 = m2[:, :, 3] == 0
    assert t2.any(), "抠图帧没有透明区，清零路径未被检验"
    assert not m2[t2, :3].any(), "合成表存在透明区 RGB 非 0"
    print(f"[5b] zero-rgb → 透明区 {int(t2.sum())}px 全零 OK")

    # 合成自检（移植自 h3 的 check_bundle）：单元格与「清零后」的源帧逐位一致
    checks = sheet.verify_composed(merged2, cut_frames, 4)
    assert checks.get("cellsMatchFrames"), checks
    print(f"[5c] verify  → {checks['cells']} 单元格与源帧逐位一致 OK")

    # 透明率校验：抠图后四角必须透明
    alpha = cut.getchannel("A")
    corners = [alpha.getpixel((0, 0)), alpha.getpixel((cut.width - 1, 0)),
               alpha.getpixel((0, cut.height - 1)), alpha.getpixel((cut.width - 1, cut.height - 1))]
    assert all(a == 0 for a in corners), f"四角未透明: {corners}"
    print("[6] alpha    → 四角透明 OK")

    # 资产 sha256 溯源 + 写坏不如写炸（移植自 h3 的 digest / write_json）
    item = storage.save_asset(merged, kind="smoke", name="smoke-zero-rgb")
    assert item.get("sha256"), "save_asset 未记录 sha256"
    import hashlib
    actual = hashlib.sha256((storage.ASSETS_DIR / item["file"]).read_bytes()).hexdigest()
    assert actual == item["sha256"], "sha256 与文件内容不符"
    try:
        storage.save_asset(merged, kind="smoke", meta={"bad": float("nan")})
        raise AssertionError("NaN 元数据没有被拒绝")
    except ValueError:
        pass
    storage.delete_asset(item["id"])
    print("[7] digest   → sha256 记录一致 / NaN 写入被拒绝 OK")

    # Godot 导出包带 SHA256SUMS 清单（移植自 h3 的 delivery.assemble）
    sheet_item = storage.save_asset(merged, kind="sheet", name="smoke-godot-sheet")
    fake_run = {"id": "smoke00000000", "steps": [
        {"id": "sprite_sheet", "meta": {"frameSize": {"w": 32, "h": 32}, "frames": 16, "cols": 4}}]}
    info = godot_export.build_godot_package([sheet_item["id"]], fake_run)
    assert "SHA256SUMS.json" in info["files"], info["files"]
    with zipfile.ZipFile(godot_export.EXPORTS_DIR / info["zip"].rsplit("/", 1)[-1]) as zf:
        sums = json.loads(zf.read("SHA256SUMS.json"))
        for row in sums["files"]:
            assert hashlib.sha256(zf.read(row["path"])).hexdigest() == row["sha256"], row["path"]
        assert "SHA256SUMS.json" not in {r["path"] for r in sums["files"]}, "manifest 不该含自身"
    storage.delete_asset(sheet_item["id"])
    (godot_export.EXPORTS_DIR / info["zip"].rsplit("/", 1)[-1]).unlink(missing_ok=True)
    print(f"[7b] godot   → {len(sums['files'])} 个文件摘要全部校验通过 OK")

    # 视频/GIF 导入拆帧 + 审查图（移植自 FrameBaker extract / h3 review）
    from backend.app.pipelines import review as review_pipe
    from backend.app.pipelines import videoio

    assert videoio.sniff_media(b"GIF89a" + b"x" * 20) == "gif"
    assert videoio.sniff_media(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert videoio.sniff_media(b"\x00\x00\x00\x18ftypmp42" + b"x" * 20) == "mp4"
    assert videoio.sniff_media(b"not a media file at all") == ""
    print("[7c] sniff   → gif/webp/mp4/未知 全部识别正确 OK")

    # 造一个 4 帧动画 GIF（每帧一个色块），拆出来必须 4 帧、尺寸一致
    gif_buf = io.BytesIO()
    gframes = []
    for i in range(4):
        f = Image.new("RGB", (48, 48), (30, 30, 40))
        d = ImageDraw.Draw(f)
        d.rectangle([6 + i * 8, 12, 26 + i * 8, 36], fill=(0, 255, 136))
        gframes.append(f)
    gframes[0].save(gif_buf, format="GIF", save_all=True, append_images=gframes[1:],
                    duration=100, loop=0)
    vframes, vmeta = videoio.extract(gif_buf.getvalue(), "x.gif", fps=0, max_frames=64)
    assert len(vframes) == 4, f"GIF 拆帧数不对: {len(vframes)}"
    assert vmeta["source"] == "gif" and vmeta["frames"] == 4
    assert all(fr.size == (48, 48) for fr in vframes)
    # 抽帧步长：100ms/帧（10fps 源）要 2fps → 每 5 帧取 1 → 只剩 1 帧。
    # 这条断言锁 _sample_step 的公式（曾经把倒数写反，抽稀变全取）
    vframes2, vmeta2 = videoio.extract(gif_buf.getvalue(), "x.gif", fps=2, max_frames=64)
    assert vmeta2["sampleStep"] == 5 and len(vframes2) == 1, (vmeta2, len(vframes2))
    print(f"[7d] gif     → 4 帧全取 / 2fps 抽成 {len(vframes2)} 帧（步长 {vmeta2['sampleStep']}）OK")

    # 审查图：网格 + 标签，尺寸 = 列*(宽+pad)+pad
    cs = review_pipe.contact_sheet(vframes, cols=2)
    assert cs.width == 2 * (48 + 6) + 6 and cs.height == 2 * (48 + 26 + 6) + 6, cs.size
    print(f"[7e] review  → 审查图 {cs.size} OK")

    # 参考图规范化（移植自 h3 normalize-master）
    from backend.app.pipelines import refnorm as refnorm_pipe

    src_ref = Image.new("RGB", (400, 300), (30, 32, 40))  # 统一底色
    dr = ImageDraw.Draw(src_ref)
    dr.rectangle((300, 60, 360, 200), fill=(220, 80, 60))  # 主体偏右上
    norm, nmeta = refnorm_pipe.normalize_ref(src_ref)
    assert norm.width == norm.height and norm.width % 16 == 0, norm.size
    assert nmeta["backgroundDetected"] is True
    _arr = np.asarray(norm.convert("RGB"))
    _mask = (np.abs(_arr.astype(np.int16) - np.array((220, 80, 60))).sum(axis=2) < 60)
    _ys = np.where(_mask.any(axis=1))[0]
    assert _ys.size, "规范化后主体丢失"
    # 主体底边应落在 baseline（0.94）附近（±8px）
    assert abs(int(_ys.max()) - int(norm.height * 0.94)) <= 8, \
        f"底边 {int(_ys.max())} vs 基线 {int(norm.height * 0.94)}"
    print(f"[7f] refnorm  → {norm.size} 方形 · 主体底边对齐基线 OK")

    # 四方向 Godot 包（h3 delivery 四向清单）
    d_sheets = []
    for d in ("down", "up"):
        sh = storage.save_asset(Image.new("RGBA", (64, 32), (10, 200, 100, 255)),
                                kind="sheet", name=f"smoke-4dir-{d}")
        d_sheets.append({"id": d, "sheetAssetId": sh["id"], "frames": 2,
                         "frameSize": [32, 32]})
    finfo = godot_export.build_fourdir_package(d_sheets, {"id": "smoke4dir0000"})
    assert "SHA256SUMS.json" in finfo["files"], finfo["files"]
    with zipfile.ZipFile(godot_export.EXPORTS_DIR / finfo["zip"].rsplit("/", 1)[-1]) as zf:
        znames = zf.namelist()
        assert "scripts/direction_controller.gd" in znames
        assert "sprites/walk_down_frames.tres" in znames
        sums = json.loads(zf.read("SHA256SUMS.json"))
        for row in sums["files"]:
            assert hashlib.sha256(zf.read(row["path"])).hexdigest() == row["sha256"], row["path"]
    for sh in d_sheets:
        storage.delete_asset(sh["sheetAssetId"])
    (godot_export.EXPORTS_DIR / finfo["zip"].rsplit("/", 1)[-1]).unlink(missing_ok=True)
    print(f"[7g] fourdir  → 四向包 {len(znames)} 文件，清单校验通过 OK")

    # 场景语义分层（移植自 FrameBaker imageLayers）：注入假 provider 跑纯逻辑
    from backend.app.pipelines import layers as layers_pipe

    class _FakeLayered:
        """顶掉 57GB 底座：只验拆完之后的覆盖率/排序/尺寸对齐/校验。"""

        async def generate_layers(self, image, *, model=None, layers=4, steps=None,
                                  resolution=640, seed=None, task_id=None):
            w, h = image.size
            out = []
            for i in range(layers):
                canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                bw = w if i == 0 else max(2, w // (i + 2))   # 背景铺满，其余越来越窄
                ImageDraw.Draw(canvas).rectangle(
                    (0, 0, bw, h), fill=(40 + i * 20, 90, 120, 255))
                out.append(canvas)
            return out, {"steps": steps or 50, "family": "qwen_layered",
                         "layers": layers, "resolution": resolution}

    lay_src = Image.new("RGB", (240, 160), (20, 20, 20))
    limgs, lmeta = await layers_pipe.decompose(lay_src, layers=4, provider=_FakeLayered())
    assert len(limgs) == 4 and all(i.size == (240, 160) for i in limgs), [i.size for i in limgs]
    # 覆盖率必须严格递减（背景 100%，后面越来越窄），排序第一层是背景
    assert lmeta["coverage"] == sorted(lmeta["coverage"], reverse=True), lmeta["coverage"]
    assert lmeta["order"][0] == 0 and lmeta["coverage"][0] == 1.0, lmeta
    assert lmeta["pipeline"] == "scene-layers" and lmeta["size"] == [240, 160]
    try:
        await layers_pipe.decompose(lay_src, layers=9, provider=_FakeLayered())
        raise AssertionError("层数超限应报错")
    except ValueError:
        pass
    try:
        await layers_pipe.decompose(lay_src, resolution=800, provider=_FakeLayered())
        raise AssertionError("非法分辨率应报错")
    except ValueError:
        pass
    print(f"[7h] layers   → 4 层 {limgs[0].size} · 覆盖率 {lmeta['coverage']} · "
          f"远→近 {lmeta['order']} OK")

    # 层序启发式：面积大小不能当深度（真实场景里石墙 31% 会比城堡 2.8% 更靠前）
    class _FakeDepth:
        async def generate_layers(self, image, *, layers=3, **kw):
            w, h = image.size
            out = [
                Image.new("RGBA", (w, h), (0, 0, 0, 255)),          # 0 背景
                Image.new("RGBA", (w, h), (0, 0, 0, 0)),            # 1 小且高（远）
                Image.new("RGBA", (w, h), (0, 0, 0, 0)),            # 2 大且低（近）
            ]
            ImageDraw.Draw(out[1]).rectangle((10, 20, 60, 100), fill=(200, 60, 60, 255))
            ImageDraw.Draw(out[2]).rectangle((0, 300, w, h), fill=(60, 200, 60, 255))
            return out, {"steps": 50}

    _, _dm = await layers_pipe.decompose(Image.new("RGB", (200, 400), (0, 0, 0)),
                                         layers=3, provider=_FakeDepth())
    assert _dm["order"] == [0, 1, 2], _dm["order"]   # 背景 → 高处小物件 → 低处大遮挡
    print(f"[7i] layers   → 层序按底边高度：{_dm['order']}（面积大≠更近）OK")

    # ---------- 动画引擎：骨骼 FK / 关键帧 / 自由变形 / 多轨 / 笔刷 / 法线 ----------
    from backend.app.anim import fx as anim_fx
    from backend.app.anim import model as anim_model
    from backend.app.anim import normals as anim_normals
    from backend.app.anim import render as anim_render

    bar = Image.new("RGBA", (60, 20), (0, 0, 0, 0))
    ImageDraw.Draw(bar).rectangle((0, 0, 59, 19), fill=(240, 80, 60, 255))
    block = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    ImageDraw.Draw(block).rectangle((0, 0, 19, 19), fill=(60, 120, 240, 255))
    a_bar = storage.save_asset(bar, kind="generated", name="smoke-anim-bar")
    a_blk = storage.save_asset(block, kind="generated", name="smoke-anim-block")
    tex = {a_bar["id"]: bar, a_blk["id"]: block}.get

    proj = anim_model.normalize({
        "id": "abcdef012345", "name": "smoke", "canvas": [200, 200],
        "fps": 12, "frames": 3,
        "bones": [
            {"id": "root", "name": "根", "parent": None, "x": 100, "y": 100},
            {"id": "arm", "name": "臂", "parent": "root", "x": 140, "y": 100,
             "keys": [{"frame": 0, "rot": 0}, {"frame": 2, "rot": 90}]},
            {"id": "hand", "name": "手", "parent": "arm", "x": 160, "y": 100},
        ],
        "tracks": [
            {"id": "t1", "name": "横条", "assetId": a_bar["id"], "bone": "root",
             "pivot": [30, 10], "keyframes": [{"frame": 0, "x": 0}, {"frame": 2, "x": 20}]},
            {"id": "t2", "name": "方块", "assetId": a_blk["id"], "bone": "hand",
             "pivot": [10, 10], "keyframes": [{"frame": 0}]},
        ],
    })
    assert len(proj["tracks"]) == 2 and proj["bones"][1]["parent"] == "root"
    f0 = anim_render.render_frame(proj, 0, tex)
    assert f0.getpixel((160, 100))[3] > 200, f0.getpixel((160, 100))
    f1 = anim_render.render_frame(proj, 1, tex)
    # 横条：x 0→20 的第 1 帧应插值到 10，左端从 70 移到 80
    assert f1.getpixel((72, 100))[3] < 40 and f1.getpixel((82, 100))[3] > 200, \
        (f1.getpixel((72, 100)), f1.getpixel((82, 100)))
    f2 = anim_render.render_frame(proj, 2, tex)
    # 臂转 90°：手（相对臂 +20,0）绕臂关节转到 (140,120)
    assert f2.getpixel((140, 120))[3] > 200, f2.getpixel((140, 120))
    print("[9] anim     → 骨骼 FK（父转子跟）/ 关键帧插值 / 多轨合成 OK")

    # 自由变形：整格右移 5px → 内容整体右移，左侧 5 列露空
    # （方向约定：控制点是「把内容推向哪」，往右拖就该往右走）
    stripe = Image.new("RGBA", (20, 20), (60, 120, 240, 255))
    ImageDraw.Draw(stripe).rectangle((8, 0, 11, 19), fill=(255, 255, 255, 255))
    shifted = anim_render.warp_image(stripe, [[5, 0]] * 9)
    assert shifted.getpixel((0, 10))[3] == 0, shifted.getpixel((0, 10))
    assert shifted.getpixel((15, 10))[0] > 200, shifted.getpixel((15, 10))
    print("[9b] warp    → 3×3 控制点双线性变形（方向/边界正确）OK")

    # 笔刷：五种笔刷都要能画出不透明像素，且同数据两次渲染完全一致
    strokes = [{"brush": b, "color": "#ffcc33", "size": 10,
                "points": [[20, 40], [60, 60], [100, 45]], "from": 0, "to": 2}
               for b in anim_fx.BRUSHES]
    fx1 = anim_fx.render(strokes, (160, 120), frame=1)
    fx2 = anim_fx.render(strokes, (160, 120), frame=1)
    assert fx1.getbbox() is not None, "笔刷没画出东西"
    assert fx1.tobytes() == fx2.tobytes(), "同笔迹两次渲染不一致"
    # 帧区间之外不画
    assert anim_fx.render(strokes, (160, 120), frame=9).getbbox() is None
    print(f"[9c] fx      → {len(anim_fx.BRUSHES)} 种笔刷可复现渲染 OK")

    # 法线 + 材质掩码：平铺色块的中心应接近「正对镜头」(128,128,255)
    nrm_src = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    ImageDraw.Draw(nrm_src).rectangle((6, 6, 17, 17), fill=(200, 200, 200, 255))
    nrm, msk, nmeta = anim_normals.analyze(nrm_src)
    cx, cy, cz, ca = nrm.getpixel((12, 12))
    # 中心是「大致朝镜头」而不是精确 (128,128)：距离变换在偶数尺寸上峰值
    # 落在像素之间，梯度不会严格为零
    assert abs(cx - 128) < 45 and abs(cy - 128) < 45 and cz > 235, nrm.getpixel((12, 12))
    assert nrm.getpixel((0, 0))[3] == 0, "透明区应是 alpha=0"
    assert msk.size == nrm_src.size and nmeta["solidRatio"] < 0.6, nmeta
    print(f"[9d] normals → 法线 {nrm.size} · 中心 {nrm.getpixel((12, 12))} · "
          f"金属占比 {nmeta['metalRatio'] * 100:.0f}% OK")

    # IK：两节骨链伸向目标点，末端应落在目标上（2px 内）
    ik_proj = anim_model.normalize({
        "id": "abcdef012346", "name": "ik", "canvas": [400, 400], "frames": 2,
        "bones": [
            {"id": "root", "parent": None, "x": 100, "y": 100},
            {"id": "upper", "parent": "root", "x": 140, "y": 100, "len": 60},
            {"id": "lower", "parent": "upper", "x": 160, "y": 100, "len": 50},
        ],
        "ik": [{"id": "ik1", "end": "lower", "chain": 2,
                "keys": [{"frame": 0, "x": 200, "y": 180}]}],
    })
    w = anim_render.bone_world(ik_proj, 0)
    m = w["lower"]
    ang = math.atan2(float(m[1, 0]), float(m[0, 0]))
    tip = (m[0, 2] + math.cos(ang) * 50, m[1, 2] + math.sin(ang) * 50)
    err = math.hypot(tip[0] - 200, tip[1] - 180)
    assert err < 2.0, f"IK 末端偏差 {err:.2f}px（应在 2px 内）"

    # 够不着的目标：链条伸直朝向目标，末端距离 = 总骨长
    far = json.loads(json.dumps(ik_proj))
    far["ik"][0]["keys"][0].update({"x": 500, "y": 100})
    w2 = anim_render.bone_world(far, 0)
    m2 = w2["lower"]
    ang2 = math.atan2(float(m2[1, 0]), float(m2[0, 0]))
    tip2 = (m2[0, 2] + math.cos(ang2) * 50, m2[1, 2] + math.sin(ang2) * 50)
    root_xy = (w2["upper"][0, 2], w2["upper"][1, 2])
    assert abs(math.hypot(tip2[0] - root_xy[0], tip2[1] - root_xy[1]) - 110) < 2, \
        (tip2, root_xy)
    print(f"[9e] ik      → 两节链命中目标（误差 {err:.2f}px）/ 超距伸直 OK")

    # 贝塞尔缓动：端点固定、强 ease-in 在前半段明显落后于线性
    bz = anim_render._bezier
    assert bz(0, 0.9, 0, 1, 0.2) == 0.0 and bz(1, 0.9, 0, 1, 0.2) == 1.0
    assert bz(0.3, 0.9, 0, 1, 0.2) < 0.2, bz(0.3, 0.9, 0, 1, 0.2)
    assert abs(bz(0.5, 0.42, 0, 0.58, 1) - 0.5) < 0.02    # 对称曲线中点仍在 0.5
    # 采样确实用了 bez（而不是忽略它退回 ease）
    bproj = anim_model.normalize({
        "id": "abcdef012347", "name": "bez", "canvas": [100, 100], "frames": 3,
        "bones": [{"id": "root", "parent": None, "x": 50, "y": 50}],
        "tracks": [{"id": "t1", "assetId": "x", "bone": "root",
                    "keyframes": [{"frame": 0, "y": 0},
                                  {"frame": 2, "y": 100, "bez": [0.9, 0, 1, 0.2]}]}],
    })
    mid = anim_render._sample(bproj["tracks"][0]["keyframes"], 1)["y"]
    assert mid < 20, mid
    print(f"[9f] bezier  → 端点精确 / 中点 {mid:.1f}（线性应为 50）OK")

    # IK 权重：w=0 与纯 FK 完全一致，w=1 命中目标，w=0.5 介于两者之间
    def _tip(proj, frame, bone="lower", blen=50.0):
        m = anim_render.bone_world(proj, frame)[bone]
        a = math.atan2(float(m[1, 0]), float(m[0, 0]))
        return (m[0, 2] + math.cos(a) * blen, m[1, 2] + math.sin(a) * blen)

    def _with_w(w: float):
        q = json.loads(json.dumps(ik_proj))
        q["ik"][0]["keys"][0]["w"] = w
        return anim_model.normalize(q)

    p0 = _with_w(0.0)
    fk_only = json.loads(json.dumps(ik_proj))
    fk_only["ik"] = []
    fk_only = anim_model.normalize(fk_only)
    t0, tfk = _tip(p0, 0), _tip(fk_only, 0)
    assert abs(t0[0] - tfk[0]) < 0.5 and abs(t0[1] - tfk[1]) < 0.5, (t0, tfk)
    th = _tip(_with_w(0.5), 0)
    t1 = _tip(_with_w(1.0), 0)
    d_fk = math.hypot(tfk[0] - 200, tfk[1] - 180)
    d_half = math.hypot(th[0] - 200, th[1] - 180)
    assert d_half < d_fk and d_half > 1, (d_fk, d_half)
    assert math.hypot(t1[0] - 200, t1[1] - 180) < 2
    print(f"[9g] ik 权重 → w0 与 FK 一致 / w1 命中 / w0.5 距目标 {d_half:.0f}px"
          f"（FK {d_fk:.0f}px）OK")

    # pole：目标在正下方时，肘部应落在 pole 那一侧（否则会随机翻肘）
    pole_proj = anim_model.normalize({
        "id": "abcdef012348", "name": "pole", "canvas": [400, 400], "frames": 1,
        "bones": [
            {"id": "root", "parent": None, "x": 100, "y": 100},
            {"id": "upper", "parent": "root", "x": 140, "y": 100, "len": 60},
            {"id": "lower", "parent": "upper", "x": 160, "y": 100, "len": 50},
        ],
        "ik": [{"id": "ik1", "end": "lower", "chain": 2, "pole": [220, 100],
                "keys": [{"frame": 0, "x": 140, "y": 180}]}],
    })
    wm = anim_render.bone_world(pole_proj, 0)
    root_xy = (wm["upper"][0, 2], wm["upper"][1, 2])
    elbow = (wm["lower"][0, 2], wm["lower"][1, 2])
    # 叉积符号：root→target 与 root→elbow，应与 root→pole 同侧
    def _cross(p, q):
        return (140 - root_xy[0]) * (q[1] - root_xy[1]) - (180 - root_xy[1]) * (q[0] - root_xy[0])
    assert _cross(0, elbow) * _cross(0, (220, 100)) > 0, (elbow, root_xy)
    print(f"[9h] pole    → 肘部 {elbow[0]:.0f},{elbow[1]:.0f} 落在 pole 一侧 OK")

    # 变形关键帧：中间帧应插值到一半（控制点线性插值）
    wproj = anim_model.normalize({
        "id": "abcdef012349", "name": "warpk", "canvas": [100, 100], "frames": 3,
        "bones": [{"id": "root", "parent": None, "x": 50, "y": 50}],
        "tracks": [{"id": "t1", "assetId": "x", "bone": "root",
                    "warpKeys": [
                        {"frame": 0, "warp": [[0, 0]] * 9},
                        {"frame": 2, "warp": [[10, 0]] * 9, "ease": "linear"},
                    ]}],
    })
    midw = anim_render._sample_warp(wproj["tracks"][0], 1)
    assert abs(midw[4][0] - 5) < 0.01, midw
    assert anim_render._sample_warp(wproj["tracks"][0], 0)[4][0] == 0
    print(f"[9i] warpKey → 中间帧控制点 dx={midw[4][0]:.1f}（0→10 的一半）OK")

    # 蒙皮：静止位必须与原图一致（M_i 都是单位阵），转动一节只弯自己那片
    band = Image.new("RGBA", (160, 40), (0, 0, 0, 0))
    db = ImageDraw.Draw(band)
    db.rectangle((0, 0, 159, 39), fill=(80, 160, 220, 255))
    for x in range(0, 160, 16):
        db.rectangle((x, 0, x + 6, 39), fill=(240, 230, 120, 255))
    ta = storage.save_asset(band, kind="generated", name="smoke-skin-band")

    def _skin_proj(rot2: float, with_skin: bool):
        tr = {"id": "t1", "assetId": ta["id"], "bone": "b1", "pivot": [20, 20],
              "keyframes": [{"frame": 0, "scale": 1}]}
        if with_skin:
            tr["skin"] = {"bones": ["b1", "b2"], "falloff": 40, "grid": 10}
        return anim_model.normalize({
            "id": "abcdef012350", "name": "skin", "canvas": [240, 120], "frames": 1,
            "bones": [
                {"id": "b1", "parent": None, "x": 20, "y": 20, "len": 60},
                {"id": "b2", "parent": "b1", "x": 80, "y": 20, "len": 60,
                 "keys": [{"frame": 0, "rot": rot2}]},
            ],
            "tracks": [tr],
        })

    tex = {ta["id"]: band}.get
    still_plain = anim_render.render_frame(_skin_proj(0, False), 0, tex)
    still_skin = anim_render.render_frame(_skin_proj(0, True), 0, tex)
    a1 = np.asarray(still_plain.convert("RGB"), dtype=np.int16)
    a2 = np.asarray(still_skin.convert("RGB"), dtype=np.int16)
    still_mad = float(np.abs(a1 - a2).mean())
    assert still_mad < 3.0, f"静止位应与普通渲染一致，MAD={still_mad:.1f}"

    bent = anim_render.render_frame(_skin_proj(45, True), 0, tex)

    def _alpha_diff(img):
        return np.abs(np.asarray(img.convert("RGBA"))[..., 3].astype(np.int16)
                      - np.asarray(still_skin.convert("RGBA"))[..., 3].astype(np.int16))

    def _centroid_y(img):
        ys, _xs = np.where(np.asarray(img.convert("RGBA"))[..., 3] > 40)
        return float(ys.mean())

    da = _alpha_diff(bent)
    left = float(da[:, :60].sum())
    right = float(da[:, 110:].sum())
    dy = _centroid_y(bent) - _centroid_y(still_skin)
    # 权重是按距离混合的，左段也会被带动一点；但弯曲量应该明显集中在右段，
    # 而且整体质心要往下（b2 从 +X 转向 +X+Y）
    assert right > left * 2, f"右段形变应明显大于左段：左 {left:.0f} 右 {right:.0f}"
    assert dy > 3, f"弯下去后质心应下移，实际 {dy:.1f}px"
    print(f"[9j] skin    → 静止位与原渲染 MAD={still_mad:.2f}；转 45° 后 "
          f"右段形变 {right / max(1.0, left):.1f}× 左段，质心下移 {dy:.1f}px OK")

    # Godot 包：骨骼世界矩阵与蒙皮顶点都烘焙进 anim.json，运行时不再算
    from backend.app.anim import export as anim_export

    gproj = _skin_proj(30, True)
    gproj["tracks"].append({"id": "t2", "name": "刚体", "assetId": ta["id"],
                            "bone": "b2", "pivot": [5, 5],
                            "keyframes": [{"frame": 0, "x": 10}]})
    gproj = anim_model.normalize(gproj)
    blob = anim_export.build_zip(gproj, lambda aid: storage.asset_path(storage.get_asset(aid)))
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
        gdata = json.loads(zf.read("anim.json"))
    assert "anim.json" in names and "anim_player.gd" in names and "README.txt" in names
    assert f"textures/{ta['id']}.png" in names, names
    assert gdata["frames"] == 1 and len(gdata["boneFrames"]) == 1
    assert len(gdata["tracks"]) == 2, [t["name"] for t in gdata["tracks"]]

    skinned = next(t for t in gdata["tracks"] if t["skinned"])
    rigid = next(t for t in gdata["tracks"] if not t["skinned"])
    assert len(skinned["mesh"]["uv"]) == 11 * 11 and len(skinned["mesh"]["triangles"]) == 200
    assert len(skinned["frames"][0]["verts"]) == 11 * 11 * 2
    # 骨骼世界矩阵要和渲染层完全对得上（差 0.01px 以上就说明两套数学漂了）
    b2m = gdata["boneFrames"][0]["b2"]["m"]
    real = anim_render.bone_world(gproj, 0)["b2"]
    assert abs(b2m[4] - float(real[0, 2])) < 0.01 and abs(b2m[5] - float(real[1, 2])) < 0.01, \
        (b2m, real)
    # 刚体轨道：矩阵应等于「骨骼世界 × 关键帧位移」，这里 x=10 直接体现在 e 上
    assert rigid["frames"][0]["m"][4] > 0, rigid["frames"][0]["m"]
    print(f"[9k] export  → Godot 包 {len(blob) // 1024}KB："
          f"{len(names)} 文件 · 蒙皮 UV {len(skinned['mesh']['uv'])} / 三角 "
          f"{len(skinned['mesh']['triangles'])} · 骨骼矩阵一致 OK")

    # 用包里的数据重建一帧，应该和渲染结果几乎一致（MAD 只在插值误差量级）
    recon = np.zeros((120, 240, 4), np.uint8)
    uv = skinned["mesh"]["uv"]
    verts = skinned["frames"][0]["verts"]
    for tri in skinned["mesh"]["triangles"]:
        src_tri = [tuple(uv[i]) for i in tri]
        dst_tri = [(verts[i * 2], verts[i * 2 + 1]) for i in tri]
        anim_render._blit_triangle(recon, np.asarray(band), src_tri, dst_tri)
    # 只比蒙皮轨道：gproj 里还有一条刚体轨道，不参与这次重建
    solo = json.loads(json.dumps(gproj))
    solo["tracks"] = solo["tracks"][:1]
    exp = np.asarray(anim_render.render_frame(solo, 0, tex).convert("RGBA"), np.int16)
    got = recon.astype(np.int16)
    mad_new = float(np.abs(exp[..., :3] - got[..., :3]).mean())
    a_mad = float(np.abs(exp[..., 3] - got[..., 3]).mean())
    assert mad_new < 8 and a_mad < 8, f"导出数据重建偏差过大：RGB {mad_new:.1f} alpha {a_mad:.1f}"
    print(f"[9l] export  → 用 anim.json 重建帧与直出帧一致（RGB MAD {mad_new:.2f} / "
          f"alpha {a_mad:.2f}）OK")

    # 手刷权重：把最左边几个顶点强行刷给 b2，旋转时它们就该跟着 b2 走
    from backend.app.anim import skin as skin_mod

    pproj = _skin_proj(45, True)
    cells = {str(i): 1.0 for i in range(6)}
    pproj["tracks"][0]["skin"]["paint"] = {"b2": cells}
    pproj = anim_model.normalize(pproj)
    painted_img = anim_render.render_frame(pproj, 0, tex)
    view = skin_mod.paint_view(pproj, pproj["tracks"][0], band.size, 0)
    assert view["painted"] == 6 and len(view["weights"]) == 121, view["painted"]
    assert abs(view["weights"][0][1] - 1.0) < 1e-6, view["weights"][0]

    # 直接比顶点：刷过的顶点应该被 b2 拉走一大截
    twin = anim_model.normalize({
        **pproj, "tracks": [{**pproj["tracks"][0],
                             "skin": {**pproj["tracks"][0]["skin"], "paint": {}}}]})
    vp = skin_mod.deform(pproj, pproj["tracks"][0], band.size, 0)
    va = skin_mod.deform(twin, twin["tracks"][0], band.size, 0)
    moved = max(math.hypot(vp[0][i][0] - va[0][i][0], vp[0][i][1] - va[0][i][1])
                for i in range(6))
    assert moved > 20, f"刷过的顶点应被 b2 带走，实际最大位移 {moved:.1f}px"
    print(f"[9m] paint   → 6 个顶点刷给 b2，最大位移 {moved:.1f}px OK")

    # 部件拆分：角色图 + 色标图 → 按颜色裁出的 RGBA 部件
    from backend.app.pipelines import parts as parts_pipe

    art = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    da = ImageDraw.Draw(art)
    da.ellipse((24, 4, 40, 20), fill=(240, 200, 180, 255))     # 头
    da.rectangle((22, 22, 42, 50), fill=(80, 120, 200, 255))    # 身体
    da.rectangle((44, 24, 58, 34), fill=(200, 80, 60, 255))     # 右臂
    tag = Image.new("RGB", (64, 64), (0, 0, 0))
    dtag = ImageDraw.Draw(tag)
    dtag.ellipse((24, 4, 40, 20), fill=(255, 0, 0))
    dtag.rectangle((22, 22, 42, 50), fill=(0, 255, 0))
    dtag.rectangle((44, 24, 58, 34), fill=(0, 0, 255))
    got = parts_pipe.split_parts(art, tag, [
        {"id": "head", "name": "头", "color": "#ff0000"},
        {"id": "body", "name": "身体", "color": "#00ff00"},
        {"id": "arm", "name": "右臂", "color": "#0000ff"},
    ])
    assert [n for n, _ in got] == ["头", "身体", "右臂"], [n for n, _ in got]
    head_arr = np.asarray(got[0][1])
    assert head_arr[..., 3].max() > 0
    # 头的部件里不该出现身体的蓝色像素（颜色裁切串色）
    vis = head_arr[..., 3] > 8
    body_leak = int((((head_arr[..., 2] > 150) & (head_arr[..., 0] < 120)) & vis).sum())
    assert body_leak == 0, f"头部件混进了 {body_leak} 个身体像素"
    cov = parts_pipe.coverage("头", got[0][1])
    assert 0.01 < cov < 0.2, cov
    # 没涂到的部位返回空图而不是报错
    none = parts_pipe.split_parts(art, tag, [{"id": "tail", "name": "尾巴", "color": "#ffff00"}])[0][1]
    assert none.getbbox() is None
    print(f"[9n] parts   → 头/身体/右臂 分色正确（头占比 {cov * 100:.1f}%，未涂部位为空）OK")

    # 裁剪 + 装骨架：部件应裁到自身包围盒，骨骼位置落在画布内
    trimmed, bbox = parts_pipe.trim_part(got[0][1])
    assert trimmed.width < got[0][1].width, (trimmed.size, got[0][1].size)
    assert bbox[2] <= trimmed.width and bbox[3] <= trimmed.height, bbox

    rig = parts_pipe.plan_rig([
        {"id": "a", "name": "头", "size": [20, 16], "bbox": [24, 4, 20, 16]},
        {"id": "b", "name": "身体", "size": [24, 32], "bbox": [20, 22, 24, 32]},
        {"id": "c", "name": "右臂", "size": [16, 24], "bbox": [44, 24, 16, 24]},
    ], [200, 200])
    assert len(rig) == 3
    byname = {r["name"]: r for r in rig}
    # 头 pivot 在下边中点（绕脖子），手臂 pivot 在上边中点（绕肩）
    assert byname["头"]["pivot"][1] == 16, byname["头"]
    assert byname["右臂"]["pivot"][1] == 0, byname["右臂"]
    # 身体是躯干：自己无父，其余都挂躯干。
    # parent 现在是**部件 id**（不再是 "body" 占位）——服务端算好的层级。
    ids = {r["name"]: r["id"] for r in rig}
    body_id = byname["身体"]["id"]
    assert byname["身体"]["bone"]["parent"] is None
    assert byname["头"]["bone"]["parent"] == body_id
    assert byname["右臂"]["bone"]["parent"] == body_id
    # 骨骼坐标必须落在画布内
    for r in rig:
        assert 0 <= r["bone"]["x"] <= 200 and 0 <= r["bone"]["y"] <= 200, r
    # 整体缩放一致（同一张原图切出来的部件，缩放必须相同）
    assert abs(byname["头"]["scale"] - byname["右臂"]["scale"]) < 1e-6
    print(f"[9o] rig     → 部件裁到 {trimmed.size}（bbox {bbox}）；"
          f"头 pivot 下中点 / 臂 pivot 上中点 / 躯干为父 / 缩放 "
          f"{rig[0]['scale']:.3f} OK")

    # 框选涂色：色标是矩形、彼此重叠时，部件必须互不重叠。
    # 之前羽化把边界外也抬成非 0，相邻部件各扩 1px、实心像素互相重叠
    # ——动画里两个部件抢同一块区域，旋转时互相撕。
    tag2 = Image.new("RGB", (64, 64), (0, 0, 0))
    dt2 = ImageDraw.Draw(tag2)
    dt2.rectangle((22, 2, 40, 18), fill=(255, 0, 0))      # 头（与身体框重叠）
    dt2.rectangle((20, 14, 42, 44), fill=(0, 255, 0))     # 身体
    art2 = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    da2 = ImageDraw.Draw(art2)
    da2.ellipse((24, 4, 38, 18), fill=(240, 200, 180, 255))
    da2.rectangle((22, 20, 40, 44), fill=(80, 130, 200, 255))
    got2 = parts_pipe.split_parts(art2, tag2, [
        {"id": "head", "name": "头", "color": "#ff0000"},
        {"id": "body", "name": "身体", "color": "#00ff00"},
    ])
    m_head = np.asarray(got2[0][1].getchannel("A")) > 8
    m_body = np.asarray(got2[1][1].getchannel("A")) > 8
    overlap = int((m_head & m_body).sum())
    assert overlap == 0, f"部件实心像素重叠 {overlap} 个（旋转时会互相撕）"
    soft = int(((np.asarray(got2[0][1].getchannel("A")) > 0)
                & (np.asarray(got2[0][1].getchannel("A")) < 250)).sum())
    assert soft > 0, "边缘应保留半透明过渡带（羽化没生效）"
    print(f"[9p] 互斥    → 部件实心重叠 0 像素 / 边缘半透明过渡 {soft} 像素 OK")

    # 分节链：尾巴1/2/3 应串成父子链；左上臂/右上臂 不得被误串
    chain = parts_pipe.plan_rig([
        {"id": "a", "name": "身体", "size": [40, 60], "bbox": [80, 60, 40, 60]},
        {"id": "b", "name": "头", "size": [30, 30], "bbox": [85, 20, 30, 30]},
        {"id": "c", "name": "尾巴1", "size": [20, 20], "bbox": [120, 70, 20, 20]},
        {"id": "d", "name": "尾巴2", "size": [20, 20], "bbox": [140, 70, 20, 20]},
        {"id": "e", "name": "尾巴3", "size": [20, 20], "bbox": [160, 70, 20, 20]},
        {"id": "f", "name": "左上臂", "size": [16, 40], "bbox": [60, 62, 16, 40]},
        {"id": "g", "name": "右上臂", "size": [16, 40], "bbox": [124, 62, 16, 40]},
    ], [300, 300])
    par = {r["id"]: r["bone"]["parent"] for r in chain}
    assert par["c"] == "a", par          # 尾巴1 挂躯干
    assert par["d"] == "c", par          # 尾巴2 挂尾巴1
    assert par["e"] == "d", par          # 尾巴3 挂尾巴2
    assert par["f"] == "a" and par["g"] == "a", par   # 两条手臂各自挂躯干
    assert par["a"] is None, par         # 躯干是顶层

    # 链式旋转：转第一节，后面的节跟着摆成一条斜线（波浪的形状）
    cproj = anim_model.normalize({
        "id": "abcdef012351", "name": "chain", "canvas": [300, 200], "frames": 1,
        "bones": [
            {"id": "root", "parent": None, "x": 30, "y": 100, "len": 40},
            {"id": "t1", "parent": "root", "x": 60, "y": 100, "len": 40,
             "keys": [{"frame": 0, "rot": 30}]},
            {"id": "t2", "parent": "t1", "x": 100, "y": 100, "len": 40},
            {"id": "t3", "parent": "t2", "x": 140, "y": 100, "len": 40},
        ],
    })
    w = anim_render.bone_world(cproj, 0)
    ys = [float(w[b][1, 2]) for b in ("t1", "t2", "t3")]
    assert ys[0] < ys[1] < ys[2], ys     # 逐节下移 → 斜着摆开了
    assert abs(float(w["t3"][1, 2]) - 140.0) < 1.0, ys
    print(f"[9q] 分节链  → 尾巴1→2→3 串成链 / 手臂不误串；转 30° 后逐节下移 "
          f"{ys[0]:.0f}→{ys[1]:.0f}→{ys[2]:.0f} OK")

    # 父子成环：前端有拦截，但服务端必须自己也能挡住（API 可被直接调用）
    cyc = anim_model.normalize({
        "id": "abcdef012352", "name": "cyc", "canvas": [200, 200], "frames": 1,
        "bones": [
            {"id": "a", "name": "A", "parent": "b", "x": 10, "y": 10},
            {"id": "b", "name": "B", "parent": "c", "x": 20, "y": 20},
            {"id": "c", "name": "C", "parent": "a", "x": 30, "y": 30},
        ],
    })
    pm = {b["id"]: b.get("parent") for b in cyc["bones"]}
    assert pm["a"] is None, pm       # 环上有一根被断开
    assert not (pm["b"] == "c" and pm["c"] == "a" and pm["a"] == "b"), pm
    anim_render.render_frame(cyc, 0, lambda _: None)   # 不能崩
    # 父不存在也要被清掉
    orph = anim_model.normalize({
        "id": "abcdef012353", "name": "orph", "canvas": [200, 200], "frames": 1,
        "bones": [{"id": "a", "name": "A", "parent": "nope", "x": 10, "y": 10}],
    })
    assert orph["bones"][0].get("parent") is None, orph["bones"]
    print("[9r] 去环    → 环上断开一根 / 坏父级清空 / 渲染不崩 OK")

    # 几何识别：只给几块涂色（不写名字），判断它们是头/躯干/四肢/尾巴
    roles = {r["id"]: r for r in parts_pipe.classify_regions([
        {"id": "h", "name": "", "size": [30, 30], "bbox": [85, 10, 30, 30]},
        {"id": "b", "name": "", "size": [34, 50], "bbox": [83, 44, 34, 50]},
        {"id": "al", "name": "", "size": [14, 38], "bbox": [64, 46, 14, 38]},
        {"id": "ar", "name": "", "size": [14, 38], "bbox": [122, 46, 14, 38]},
        {"id": "ll", "name": "", "size": [15, 42], "bbox": [84, 96, 15, 42]},
        {"id": "lr", "name": "", "size": [15, 42], "bbox": [101, 96, 15, 42]},
        {"id": "t", "name": "", "size": [40, 14], "bbox": [120, 70, 40, 14]},
    ])}
    assert roles["h"]["role"] == "head", roles
    assert roles["b"]["role"] == "body", roles
    assert roles["al"]["role"] == "arm" and roles["al"]["side"] == "l", roles
    assert roles["ar"]["role"] == "arm" and roles["ar"]["side"] == "r", roles
    assert roles["ll"]["role"] == "leg" and roles["ll"]["side"] == "l", roles
    assert roles["lr"]["role"] == "leg" and roles["lr"]["side"] == "r", roles
    assert roles["t"]["role"] == "tail" and roles["t"]["side"] is None, roles
    # 关节位置：头绕下边（脖子）、臂绕上边（肩）、腿绕上边（髋）、
    # 尾巴绕靠身体那端
    assert roles["h"]["joint"] == "bottom", roles
    assert roles["al"]["joint"] == "top", roles
    assert roles["ll"]["joint"] == "top", roles
    assert roles["t"]["joint"] == "left", roles      # 尾巴在身体右侧 → 关节在左端

    # 自动命名的部件：pivot 按几何走，不按名字
    auto = parts_pipe.plan_rig([
        {"id": "h", "name": "", "autoNamed": True, "size": [30, 30], "bbox": [85, 10, 30, 30]},
        {"id": "b", "name": "", "autoNamed": True, "size": [34, 50], "bbox": [83, 44, 34, 50]},
        {"id": "al", "name": "", "autoNamed": True, "size": [14, 38], "bbox": [64, 46, 14, 38]},
    ], [300, 300])
    am = {r["name"]: r for r in auto}
    assert set(am) == {"头", "身体", "左臂"}, set(am)
    assert am["头"]["pivot"][1] == 30, am["头"]        # 下边（绕脖子）
    assert am["左臂"]["pivot"][1] == 0, am["左臂"]     # 上边（绕肩）
    # 显式命名的部件：pivot 按名字走（用户故意叫「上臂」画朝上的）
    manu = parts_pipe.plan_rig([
        {"id": "b", "name": "身体", "size": [34, 50], "bbox": [83, 44, 34, 50]},
        {"id": "al", "name": "上臂", "size": [14, 38], "bbox": [64, 46, 14, 38]},
    ], [300, 300])
    mm = {r["name"]: r for r in manu}
    assert mm["上臂"]["pivot"][1] == 0, mm["上臂"]
    print(f"[9s] 识别    → 头/身体/双臂/双腿/尾巴全部认对；"
          f"自动命名 pivot 按几何（头 {am['头']['pivot']}）、"
          f"显式命名按名字（{mm['上臂']['pivot']}）OK")

    # 颜色距离不能出 NaN：int16 算平方差会溢出回绕成负数，sqrt 得 NaN，
    # 不报错、只是让对应像素的 mask 静默失效（当时只在控制台看到一行 warning）
    far = parts_pipe._hex_dist(np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8),
                               (255, 255, 255))
    assert not np.isnan(far).any() and abs(float(far[0, 0]) - 360.62) < 0.1, far



    storage.delete_asset(ta["id"])



    for it in (a_bar, a_blk):
        storage.delete_asset(it["id"])

    png = io.BytesIO()
    rpg.save(png, "PNG")
    print(f"[8] export   → rpgmaker png {len(png.getvalue())} bytes")

    storage._INDEX_CACHE = None
    leaked = [a for a in storage.list_assets() if a["id"] not in _base]
    for a in leaked:
        try:
            storage.delete_asset(a["id"])
        except Exception:
            pass
    if leaked:
        print(f"[cleanup] 清掉本轮产物 {len(leaked)} 个")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
