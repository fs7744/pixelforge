"""角色部件拆分的端到端验证：角色图 → 涂部位 → 部件 → 骨架 → 渲染。

独立成模块而不是塞进测试脚本，是因为它要跑完整的「拆分 → 装骨架 → 渲染 →
与原图比对」链路，输出是给人看的图，不是断言。

**数据目录必须隔离**：这个脚本会真的往素材库里写部件资产。上一版我
直接在交互里跑，用的是真实 data/，结果素材库多出 18 个部件资产还得手工
清。凡是「跑一下看看」性质的脚本，一律先指向临时目录。
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

# 必须在 import backend 之前设好：config 在导入时就把 DATA_DIR 定下来了
_TMP_DATA = Path(tempfile.mkdtemp(prefix="pf-parts-e2e-"))
os.environ["PIXELFORGE_DATA_DIR"] = str(_TMP_DATA)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import storage  # noqa: E402
from backend.app.anim import model as anim_model  # noqa: E402
from backend.app.anim import render as anim_render  # noqa: E402
from backend.app.main import app  # noqa: E402

CANVAS = [300, 300]
OUT = Path(__file__).resolve().parents[1] / "data" / "tmp"

PARTS = [
    {"id": "head", "name": "头", "color": "#ff0000"},
    {"id": "body", "name": "身体", "color": "#00ff00"},
    {"id": "arm_l", "name": "左上臂", "color": "#0000ff"},
    {"id": "arm_r", "name": "右上臂", "color": "#ffff00"},
    {"id": "leg_l", "name": "左腿", "color": "#ff00ff"},
    {"id": "leg_r", "name": "右腿", "color": "#00ffff"},
]


def make_character() -> Image.Image:
    img = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((48, 6, 76, 34), fill=(240, 200, 170, 255))      # 头
    d.rectangle((46, 36, 78, 78), fill=(70, 130, 210, 255))     # 身体
    d.rectangle((28, 38, 42, 72), fill=(230, 120, 80, 255))     # 左臂
    d.rectangle((82, 38, 96, 72), fill=(230, 120, 80, 255))     # 右臂
    d.rectangle((50, 80, 60, 112), fill=(60, 90, 140, 255))     # 左腿
    d.rectangle((64, 80, 74, 112), fill=(60, 90, 140, 255))     # 右腿
    return img


def make_tag() -> Image.Image:
    img = Image.new("RGB", (120, 120), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((48, 6, 76, 34), fill=(255, 0, 0))
    d.rectangle((46, 36, 78, 78), fill=(0, 255, 0))
    d.rectangle((28, 38, 42, 72), fill=(0, 0, 255))
    d.rectangle((82, 38, 96, 72), fill=(255, 255, 0))
    d.rectangle((50, 80, 60, 112), fill=(255, 0, 255))
    d.rectangle((64, 80, 74, 112), fill=(0, 255, 255))
    return img


def build_project(rig: list, textures: dict) -> dict:
    """按前端 applyRig 的规则建工程（服务端算方案，前端只写入）。"""
    body = next((x for x in rig if x.get("isBody")), rig[0])
    bid = {x["id"]: f"p{i}" for i, x in enumerate(rig)}

    def parent_of(x: dict) -> str:
        if not x["bone"].get("parent"):
            return "root"
        if x["bone"]["parent"] == "body":
            return bid[body["id"]]
        found = next((y for y in rig if y["name"] == x["bone"]["parent"]), None)
        return bid[found["id"]] if found else "root"

    return anim_model.normalize({
        "id": "abcdef012399", "name": "rig", "canvas": CANVAS, "frames": 1,
        "bones": [{"id": "root", "name": "根骨骼", "parent": None,
                   "x": CANVAS[0] // 2, "y": CANVAS[1] - 40, "len": 60}] + [
            {"id": bid[x["id"]], "name": x["name"], "parent": parent_of(x),
             "x": x["bone"]["x"], "y": x["bone"]["y"], "len": x["bone"]["len"],
             "keys": []} for x in rig],
        "tracks": [{"id": f"t{i}", "name": x["name"], "assetId": x["id"],
                    "bone": bid[x["id"]], "pivot": x["pivot"],
                    "keyframes": [{"frame": 0, "x": 0, "y": 0, "rot": 0,
                                   "scale": x["scale"], "opacity": 1,
                                   "ease": "linear"}]}
                   for i, x in enumerate(rig)],
    })


def reference(art: Image.Image, parts: list, scale: float) -> Image.Image:
    """同基准参考：按部件并集 bbox 裁原图再缩放居中。

    必须同基准——直接拿整幅原图居中做参考会差好几像素（部件并集比整幅小），
    那种偏差会掩盖真正的对齐问题。
    """
    boxes = [p["bbox"] for p in parts]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    crop = art.crop((x0, y0, x1, y1)).resize(
        (max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale))),
        Image.LANCZOS)
    out = Image.new("RGBA", tuple(CANVAS), (0, 0, 0, 0))
    out.alpha_composite(crop, ((CANVAS[0] - crop.width) // 2,
                               (CANVAS[1] - crop.height) // 2))
    return out


def main() -> int:
    client = TestClient(app)
    art = make_character()
    tag = make_tag()
    ab = io.BytesIO()
    art.save(ab, "PNG")
    tb = io.BytesIO()
    tag.save(tb, "PNG")

    res = client.post("/api/pipeline/parts",
                      files={"file": ("a.png", ab.getvalue(), "image/png"),
                             "tag": ("t.png", tb.getvalue(), "image/png")},
                      data={"parts": json.dumps(PARTS, ensure_ascii=False),
                            "canvas": json.dumps(CANVAS)})
    if res.status_code != 200:
        print("拆分失败", res.status_code, res.text[:200])
        return 1
    data = res.json()
    print(f"部件 {len(data['parts'])} 个：")
    for p in data["parts"]:
        print(f"  {p['name']:<6} 占比 {p['coverage'] * 100:5.1f}%  "
              f"bbox {p['bbox']}  尺寸 {p['size']}")
    print("骨架方案：")
    for r in data["rig"]:
        print(f"  {r['name']:<6} 骨骼 ({r['bone']['x']:.0f},{r['bone']['y']:.0f}) "
              f"父={r['bone']['parent'] or '-'}  pivot {r['pivot']}  缩放 {r['scale']}")

    textures = {}
    for p in data["parts"]:
        if p.get("asset"):
            item = storage.get_asset(p["asset"]["id"])
            textures[p["asset"]["id"]] = Image.open(
                storage.asset_path(item)).convert("RGBA")

    proj = build_project(data["rig"], textures)
    out = anim_render.render_frame(proj, 0, textures.get)
    ref = reference(art, data["parts"], data["rig"][0]["scale"])

    A = np.asarray(out.convert("RGBA"), np.int16)
    B = np.asarray(ref.convert("RGBA"), np.int16)
    inter = float(np.minimum(A[..., 3], B[..., 3]).sum())
    union = float(np.maximum(A[..., 3], B[..., 3]).sum())
    ys, xs = np.where(A[..., 3] > 40)
    ys2, xs2 = np.where(B[..., 3] > 40)
    print(f"\nalpha IoU {inter / union * 100:.1f}%   "
          f"质心偏差 dx={xs.mean() - xs2.mean():.1f} dy={ys.mean() - ys2.mean():.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGBA", (CANVAS[0] * 3 + 16, CANVAS[1]), (20, 20, 26, 255))
    for i, im in enumerate([art.convert("RGBA"), ref, out]):
        t = im if im.size == tuple(CANVAS) else im.resize(tuple(CANVAS), Image.LANCZOS)
        sheet.alpha_composite(t, (i * (CANVAS[0] + 8), 0))
    sheet.save(OUT / "parts-rig-compare.png")
    print("对比图（原角色 | 同基准参考 | 骨架渲染）→", OUT / "parts-rig-compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
