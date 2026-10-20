"""用 Godot --headless 实跑导出的运行时，验证「数据格式 + GDScript 实现」都对。

之前只做了 Python 侧的等价性验证（用 anim.json 重建帧 = 直出帧），
但那只证明**数据**够用，证明不了 **GDScript** 写得对——Transform2D 构造顺序反了、
UV 分母用错、数组索引错位，这些都得在 Godot 里跑一遍才暴露。

比什么、不比什么：
- 比：逐帧顶点/UV/矩阵在 GDScript 里还原出来是否与 anim.json 一致；
      用这些几何在 Python 侧重建帧，与后端直出帧的偏差（MAD）。
- 不比像素：headless 用的是 dummy 渲染器，出不了图，也不该假装能出图。

用法：
    GODOT_BIN=/path/to/Godot_v4.3-stable_win64_console.exe python backend/godot_verify.py
找不到 Godot 时打印「跳过」并返回 0，不阻塞其它测试。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.anim import export as anim_export  # noqa: E402
from backend.app.anim import model as anim_model  # noqa: E402
from backend.app.anim import render as anim_render  # noqa: E402

PROJECT_GODOT = """config_version=5

[application]
config/name="pf-anim-verify"

[rendering]
renderer/rendering_method="gl_compatibility"
"""

VERIFY_GD = """extends SceneTree
const Anim = preload("res://anim/anim_player.gd")

func _init():
	var node = Anim.new()
	root.add_child(node)
	node.load_package("res://anim/")
	var data: Dictionary = node.data
	var out := {"frameCount": int(data.get("frames", 0)), "tracks": []}
	for i in range(data["tracks"].size()):
		var t: Dictionary = data["tracks"][i]
		var skinned: bool = t.get("skinned", false)
		var entry: Dictionary = {"index": i, "name": t.get("name", ""), "skinned": skinned,
			"verts": [], "ms": [], "uv0": [], "uvCount": 0, "triCount": 0}
		if skinned:
			var uvs = node.frame_uvs(t)
			entry["uvCount"] = uvs.size()
			entry["triCount"] = node.track_triangles(t).size()
			entry["uv0"] = [uvs[0].x, uvs[0].y]
			entry["uvLast"] = [uvs[uvs.size() - 1].x, uvs[uvs.size() - 1].y]
		for f in range(int(data["frames"])):
			var fr: Dictionary = t["frames"][f]
			if skinned:
				var pts = node.frame_vertices(fr)
				entry["verts"].append([pts[0].x, pts[0].y,
					pts[pts.size() - 1].x, pts[pts.size() - 1].y])
			else:
				var tr = node.track_transform(fr)
				entry["ms"].append([tr.x.x, tr.x.y, tr.y.x, tr.y.y, tr.origin.x, tr.origin.y])
		out["tracks"].append(entry)
	print("PFVERIFY " + JSON.stringify(out))
	quit(0)
"""


def find_godot() -> Optional[Path]:
    env = os.environ.get("GODOT_BIN")
    if env and Path(env).is_file():
        return Path(env)
    for pat in ("Godot*_console.exe", "godot", "godot4", "Godot*"):
        for base in (ROOT / "data" / "tmp" / "godot", ROOT / "tools"):
            if base.is_dir():
                hit = sorted(base.glob(pat))
                if hit:
                    return hit[0]
    return None


def _make_assets() -> Tuple[Image.Image, Image.Image]:
    """造两张贴图：一条弯曲带（蒙皮）、一个小方块（刚性轨道）。"""
    band = Image.new("RGBA", (160, 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle((0, 0, 159, 39), fill=(80, 160, 220, 255))
    for x in range(0, 160, 16):
        d.rectangle((x, 0, x + 6, 39), fill=(240, 230, 120, 255))
    cube = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    ImageDraw.Draw(cube).rectangle((2, 2, 21, 21), fill=(240, 90, 70, 255))
    return band, cube


def build_project(dirpath: Path) -> Tuple[Dict[str, Any], Dict[str, Image.Image], Dict[str, Any]]:
    band, cube = _make_assets()
    textures = {"band": band, "cube": cube}
    proj = anim_model.normalize({
        "id": "abcdef012345", "name": "verify", "canvas": [240, 120], "fps": 10, "frames": 4,
        "bones": [
            {"id": "b1", "name": "尾根", "parent": None, "x": 20, "y": 20, "len": 60},
            {"id": "b2", "name": "尾尖", "parent": "b1", "x": 80, "y": 20, "len": 60,
             "keys": [{"frame": 0, "rot": 0}, {"frame": 3, "rot": 40}]},
        ],
        "tracks": [
            {"id": "t1", "name": "带子", "assetId": "band", "bone": "b1", "pivot": [20, 20],
             "skin": {"bones": ["b1", "b2"], "falloff": 40, "grid": 10},
             "keyframes": [{"frame": 0, "scale": 1}]},
            {"id": "t2", "name": "方块", "assetId": "cube", "bone": "b2", "pivot": [4, 4],
             "keyframes": [{"frame": 0, "x": 0}, {"frame": 3, "x": 20, "rot": 15}]},
        ],
    })
    data = anim_export.build_anim_data(proj, textures)
    (dirpath / "anim").mkdir(parents=True, exist_ok=True)
    (dirpath / "project.godot").write_text(PROJECT_GODOT, encoding="utf-8")
    (dirpath / "verify.gd").write_text(VERIFY_GD, encoding="utf-8")
    (dirpath / "anim" / "anim_player.gd").write_text(anim_export.GDSCRIPT, encoding="utf-8")
    (dirpath / "anim" / "anim.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (dirpath / "anim" / "textures").mkdir(exist_ok=True)
    for aid, img in textures.items():
        img.save(dirpath / "anim" / "textures" / f"{aid}.png")
    return data, textures, proj


def run_godot(godot: Path, dirpath: Path) -> Dict[str, Any]:
    cmd = [str(godot), "--headless", "--path", str(dirpath), "--script", "verify.gd"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    for line in out.splitlines():
        if line.startswith("PFVERIFY "):
            return json.loads(line[len("PFVERIFY "):])
    raise RuntimeError("Godot 没有输出验证结果：\n" + out[-2000:])


def compare(data: Dict[str, Any], got: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if got.get("frameCount") != data["frames"]:
        problems.append(f"帧数不符：godot {got.get('frameCount')} vs {data['frames']}")

    for gd, gd_track in zip(data["tracks"], got["tracks"]):
        if gd_track["skinned"]:
            if gd_track["uvCount"] != len(gd["mesh"]["uv"]):
                problems.append(f"{gd['name']}: UV 数 {gd_track['uvCount']} != {len(gd['mesh']['uv'])}")
            if gd_track["triCount"] != sum(len(t) for t in gd["mesh"]["triangles"]):
                problems.append(f"{gd['name']}: 三角形下标数不符")
            ts = gd["textureSize"]
            want_first = [gd["mesh"]["uv"][0][0] / ts[0], gd["mesh"]["uv"][0][1] / ts[1]]
            if max(abs(gd_track["uv0"][0] - want_first[0]),
                   abs(gd_track["uv0"][1] - want_first[1])) > 1e-4:
                problems.append(f"{gd['name']}: UV 归一化不一致 {gd_track['uv0']} vs {want_first}")
            for f, corner in enumerate(gd_track["verts"]):
                want = gd["frames"][f]["verts"]
                n = len(want)
                pairs = [(corner[0], want[0]), (corner[1], want[1]),
                         (corner[2], want[n - 2]), (corner[3], want[n - 1])]
                if any(abs(a - b) > 0.01 for a, b in pairs):
                    problems.append(f"{gd['name']} 第 {f} 帧顶点与 anim.json 不一致：{corner}")
                    break
        else:
            for f, m in enumerate(gd_track["ms"]):
                want = gd["frames"][f]["m"]
                # Godot Transform2D.x = (a,b)，.y = (c,d)，.origin = (e,f)
                if max(abs(m[0] - want[0]), abs(m[1] - want[1]), abs(m[2] - want[2]),
                       abs(m[3] - want[3]), abs(m[4] - want[4]), abs(m[5] - want[5])) > 1e-4:
                    problems.append(
                        f"{gd['name']} 第 {f} 帧 Transform2D 与 anim.json 不一致：{m} vs {want}")
                    break
    return problems


def verify_frames(data: Dict[str, Any], got: Dict[str, Any],
                  textures: Dict[str, Image.Image], proj: Dict[str, Any]) -> List[str]:
    """用 Godot 给的几何在 Python 侧重建帧，和后端直出比 MAD。

    headless 出不了像素图，但几何是一样的：这一步能证明
    「GDScript 交出来的顶点」能还原出后端那张画。
    """
    out: List[str] = []
    canvas = tuple(proj["canvas"])
    tex = {t["assetId"]: textures[t["assetId"]] for t in proj["tracks"]}
    for gd, gd_track in zip(data["tracks"], got["tracks"]):
        if not gd_track["skinned"]:
            continue
        track = next(t for t in proj["tracks"] if gd["name"] == t.get("name"))
        f = int(data["frames"]) - 1
        exp = np.asarray(anim_render.render_frame(
            anim_model.normalize({**proj, "tracks": [track]}), f,
            lambda _a: tex[track["assetId"]]).convert("RGBA"), np.int16)
        verts = gd["frames"][f]["verts"]
        uv = gd["mesh"]["uv"]
        out_img = np.zeros((canvas[1], canvas[0], 4), np.uint8)
        for tri in gd["mesh"]["triangles"]:
            src_tri = [tuple(uv[i]) for i in tri]
            dst_tri = [(verts[i * 2], verts[i * 2 + 1]) for i in tri]
            anim_render._blit_triangle(out_img, np.asarray(tex[track["assetId"]]),
                                       src_tri, dst_tri)
        got_img = out_img.astype(np.int16)
        mad = float(np.abs(exp[..., :3] - got_img[..., :3]).mean())
        a_mad = float(np.abs(exp[..., 3] - got_img[..., 3]).mean())
        if mad > 1.0 or a_mad > 1.0:
            out.append(f"{gd['name']} 第 {f} 帧几何重建偏差过大：RGB {mad:.2f} alpha {a_mad:.2f}")
        else:
            out.append(f"{gd['name']} 第 {f} 帧由 Godot 几何重建与直出帧一致"
                       f"（RGB MAD {mad:.3f} / alpha {a_mad:.3f}）")
    return out


def main() -> int:
    godot = find_godot()
    if not godot:
        print("[godot] 未找到 Godot 可执行文件，跳过引擎内验证。"
              "\n        设置 GODOT_BIN 后重跑；本机可放 data/tmp/godot/Godot*_console.exe。")
        return 0
    workdir = Path(tempfile.mkdtemp(prefix="pf-godot-verify-"))
    try:
        data, textures, proj = build_project(workdir)
        got = run_godot(godot, workdir)
        problems = compare(data, got)
        print(f"[godot] {godot.name}：{len(got['tracks'])} 条轨道 / {got['frameCount']} 帧已跑通")
        for line in problems:
            print("  x " + line)
        for line in verify_frames(data, got, textures, proj):
            print("  - " + line)
        if problems:
            print("[godot] 验证未通过")
            return 1
        print("[godot] 验证通过：GDScript 还原的顶点/UV/矩阵与 anim.json 一致")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
