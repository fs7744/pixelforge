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

    png = io.BytesIO()
    rpg.save(png, "PNG")
    print(f"[8] export   → rpgmaker png {len(png.getvalue())} bytes")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
