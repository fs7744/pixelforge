"""MVP 冒烟测试：不依赖前端与模型，直接验证确定性管线。

    python backend/smoke_test.py

测试素材由本脚本自己画（4×4 纯色块网格 + 纯色底），不经过任何
生成模型——模拟引擎已移除，管线测试也不该依赖 GPU。
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

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
    from backend.app import storage
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
    from backend.app import godot_export
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

    png = io.BytesIO()
    rpg.save(png, "PNG")
    print(f"[8] export   → rpgmaker png {len(png.getvalue())} bytes")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
