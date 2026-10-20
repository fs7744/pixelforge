"""动画工程 → Godot 可用包（骨架 + 逐帧变换 + 烘焙蒙皮网格 + 纹理）。

为什么不在引擎里重算骨链和蒙皮：IK（FABRIK）、权重、变形都是本项目的实现细节，
一旦让运行时复刻，就等于把同一套数学维护两遍——两边版本一漂，画面就对不上。
所以这里**烘焙**：骨骼逐帧世界矩阵、蒙皮网格逐帧顶点坐标都算好写进 JSON，
运行时只做「按帧取数 → 设变换 → 画」。

代价是文件大一点（8 帧 × 169 顶点 ≈ 几万字节），换来的是"导出的和预览的一模一样"。

包结构：
    anim.json               全部数据（见下面 build_anim_data 的注释）
    textures/<assetId>.png  轨道贴图（原图，不烧角标）
    anim_player.gd          参考运行时（Godot 4.x）
    README.txt              字段说明与用法
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import render as render_mod

VERSION = 1


def _mat6(m: np.ndarray) -> List[float]:
    """3×3 仿射 → [a,b,c,d,e,f]，与前端/文档同一约定：x' = a·x + c·y + e。"""
    return [round(float(m[0, 0]), 4), round(float(m[1, 0]), 4),
            round(float(m[0, 1]), 4), round(float(m[1, 1]), 4),
            round(float(m[0, 2]), 3), round(float(m[1, 2]), 3)]


def _bone_frames(proj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """逐帧骨骼世界变换。IK 已经解算进 world 里，运行时不需要再解一遍。"""
    out = []
    for f in range(int(proj["frames"])):
        worlds = render_mod.bone_world(proj, f)
        row = {}
        for b in proj.get("bones") or []:
            m = worlds.get(b["id"])
            if m is None:
                continue
            row[b["id"]] = {
                "m": _mat6(m),
                # 另外给一份「原点 + 角度 + 缩放」，方便肉眼核对/调试
                "x": round(float(m[0, 2]), 2),
                "y": round(float(m[1, 2]), 2),
                "rot": round(math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0]))), 2),
                "scale": round(math.hypot(float(m[0, 0]), float(m[1, 0])), 4),
            }
        out.append(row)
    return out


def _track_frames(proj: Dict[str, Any], t: Dict[str, Any],
                  src: Image.Image) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """一条轨道的逐帧数据。

    非蒙皮：给 2D 仿射矩阵（运行时直接 set_transform）。
    蒙皮：给烘焙后的顶点坐标（uv/三角形只写一份）。两条路径都带 opacity。
    """
    skin = t.get("skin")
    frames: List[Dict[str, Any]] = []
    mesh: Optional[Dict[str, Any]] = None
    worlds = None

    if skin:
        grid = int(skin.get("grid") or 8)
        w, h = src.size
        uv = []
        for j in range(grid + 1):
            for i in range(grid + 1):
                uv.append([round(w * i / grid, 2), round(h * j / grid, 2)])
        tris = []
        def vid(i: int, j: int) -> int:
            return j * (grid + 1) + i
        for j in range(grid):
            for i in range(grid):
                tris.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
                tris.append([vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)])
        mesh = {"uv": uv, "triangles": tris}
        for f in range(int(proj["frames"])):
            verts = render_mod._skin_mesh(src, proj, t, f)  # noqa: SLF001
            flat = [round(v, 2) for row in (verts or []) for p in row for v in p]
            frames.append({"opacity": round(_opacity(t, f), 3), "verts": flat})

    if not skin:
        worlds = None
        for f in range(int(proj["frames"])):
            k = render_mod._sample(t.get("keyframes") or [], f)
            b = next((x for x in proj["bones"] if x["id"] == (t.get("bone") or "root")), None)
            world = (render_mod.bone_world(proj, f).get(b["id"]) if b else None)
            if world is None:
                world = np.eye(3)
            px, py = t.get("pivot") or [src.width / 2, src.height / 2]
            m = (world @ render_mod._T(k["x"], k["y"]) @ render_mod._R(k["rot"])
                 @ render_mod._S(k["scale"]) @ render_mod._T(-px, -py))
            frames.append({"opacity": round(_opacity(t, f), 3), "m": _mat6(m)})

    return frames, mesh


def _opacity(t: Dict[str, Any], frame: int) -> float:
    k = render_mod._sample(t.get("keyframes") or [], frame)  # noqa: SLF001
    return max(0.0, min(1.0, float(t.get("opacity", 1.0)) * float(k.get("opacity", 1.0))))


def build_anim_data(proj: Dict[str, Any], textures: Dict[str, Image.Image]) -> Dict[str, Any]:
    """组装 anim.json 的内容。

    tracks[i].frames[j] 的形状取决于是否蒙皮：
      - 刚性：{"opacity": float, "m": [a,b,c,d,e,f]}
      - 蒙皮：{"opacity": float, "verts": [x0,y0,x1,y1,...]}
    蒙皮轨道的 mesh.uv 与 mesh.triangles 只写一份，顶点顺序按行优先、
    与 uv 一一对应。
    """
    data: Dict[str, Any] = {
        "version": VERSION,
        "name": proj.get("name") or "",
        "canvas": list(proj["canvas"]),
        "fps": int(proj.get("fps") or 12),
        "frames": int(proj["frames"]),
        "bones": [{"id": b["id"], "name": b["name"], "parent": b.get("parent"),
                   "x": b["x"], "y": b["y"], "len": b.get("len", 60)}
                  for b in proj.get("bones") or []],
        "boneFrames": _bone_frames(proj),
        "tracks": [],
    }
    for t in proj.get("tracks") or []:
        src = textures.get(t["assetId"])
        if src is None:
            continue
        frames, mesh = _track_frames(proj, t, src.convert("RGBA"))
        item: Dict[str, Any] = {
            "name": t.get("name") or t["id"],
            "texture": f"textures/{t['assetId']}.png",
            "bone": t.get("bone") or "root",
            "pivot": t.get("pivot") or [src.width / 2, src.height / 2],
            "blend": t.get("blend") or "normal",
            "visible": bool(t.get("visible", True)),
            "skinned": bool(t.get("skin")),
            "frames": frames,
        }
        if mesh:
            item["mesh"] = mesh
            item["skin"] = t.get("skin")
        data["tracks"].append(item)
    return data


GDSCRIPT = '''# PixelForge 动画运行时（参考实现，Godot 4.x）
#
# 用法：
#   var anim = PixelForgeAnim.new()
#   anim.load_package("res://anim/")     # anim.json + textures/ 所在目录
#   add_child(anim)
#   anim.play("walk", true)              # 循环播放
#
# 说明：本文件是**参考实现**，数据格式以 anim.json 为准；
# 蒙皮轨道用 ArrayMesh 逐帧重建（顶点已烘焙好，不需要在引擎里算权重）。
extends Node2D
class_name PixelForgeAnim

var data: Dictionary = {}
var _time := 0.0
var _playing := false
var _loop := true
var _sprite_nodes: Array = []
var _mesh_nodes: Array = []
var _base := ""

func load_package(path: String) -> void:
	_base = path
	var f := FileAccess.open(path.path_join("anim.json"), FileAccess.READ)
	if f == null:
		push_error("找不到 anim.json：%s" % path)
		return
	data = JSON.parse_string(f.get_as_text())
	_build()

func _build() -> void:
	for t in data.get("tracks", []):
		if not t.get("visible", true):
			continue
		var tex_path: String = _base.path_join(t["texture"])
		if t.get("skinned", false):
			var mi := MeshInstance2D.new()
			mi.material = CanvasItemMaterial.new()
			add_child(mi)
			_mesh_nodes.append({"node": mi, "track": t})
		else:
			var sp := Sprite2D.new()
			sp.texture = load(tex_path)
			sp.centered = false
			sp.z_index = _sprite_nodes.size()
			add_child(sp)
			_sprite_nodes.append({"node": sp, "track": t})

func play(_name := "", loop := true) -> void:
	_playing = true
	_loop = loop
	_time = 0.0

func _process(delta: float) -> void:
	if not _playing or data.is_empty():
		return
	var frames: int = int(data["frames"])
	var fps: float = float(data.get("fps", 12))
	_time += delta
	var idx := int(floor(_time * fps))
	if idx >= frames:
		if _loop:
			idx = idx % frames
		else:
			idx = frames - 1
			_playing = false
	_apply(idx)

func _apply(idx: int) -> void:
	for entry in _sprite_nodes:
		var t: Dictionary = entry["track"]
		var fr: Dictionary = t["frames"][idx]
		var a: Array = fr["m"]
		# [a,b,c,d,e,f] → Godot Transform2D(x_axis, y_axis, origin)
		entry["node"].transform = Transform2D(
			Vector2(a[0], a[1]), Vector2(a[2], a[3]), Vector2(a[4], a[5]))
		entry["node"].modulate.a = float(fr.get("opacity", 1.0))
	for entry in _mesh_nodes:
		var t: Dictionary = entry["track"]
		var fr: Dictionary = t["frames"][idx]
		var verts: Array = fr["verts"]
		var uv: Array = t["mesh"]["uv"]
		var tris: Array = t["mesh"]["triangles"]
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		var pts := PackedVector2Array()
		for i in range(0, verts.size(), 2):
			pts.append(Vector2(verts[i], verts[i + 1]))
		for tri in tris:
			for vi in tri:
				st.set_uv(Vector2(uv[vi][0], uv[vi][1]) / Vector2(
					t["pivot"][0] * 2.0, t["pivot"][1] * 2.0))
				st.add_vertex(Vector3(pts[vi].x, pts[vi].y, 0.0))
		var mi: MeshInstance2D = entry["node"]
		mi.mesh = st.commit()
		mi.material = CanvasItemMaterial.new()
		mi.texture = load(_base.path_join(t["texture"]))
		mi.modulate.a = float(fr.get("opacity", 1.0))
'''

README = """PixelForge 动画包
=================

anim.json 结构
--------------
version     : 1
canvas      : [宽, 高]，所有坐标都在这个像素空间里（左上原点，y 向下）
fps, frames : 播放参数
bones       : 骨骼静态定义（id / name / parent / x / y / len），x,y 是静止位
boneFrames  : 每帧每根骨骼的世界变换
              m = [a,b,c,d,e,f]，x' = a*x + c*y + e，y' = b*x + d*y + f
              （IK / 关键帧都已解算进去，运行时不用再算骨链）
tracks      : 每条轨道
  texture   : textures/ 下的贴图路径
  pivot     : 贴图内像素坐标，pivot 落在骨骼原点上
  frames[j] : 第 j 帧
      opacity   0..1
      m         [a..f]  —— 非蒙皮轨道的 2D 仿射（直接设到 Sprite2D.transform）
      verts     [x0,y0,x1,y1,...] —— 蒙皮轨道的烘焙顶点（画布坐标）
  mesh      : 仅蒙皮轨道：uv 与 triangles 只写一份，顺序与 uv 对应
              uv 是**贴图像素坐标**，三角形用顶点下标

坐标约定
--------
- 画布左上为原点，y 向下（2D 游戏常规），与 Godot 的 CanvasItem 一致。
- 矩阵是行向量右乘：p' = [x,y,1] · 3x3。Godot 的 Transform2D 用
  Transform2D(Vector2(a,b), Vector2(c,d), Vector2(e,f))。

使用
----
1. 把整个文件夹拷进 Godot 工程（例如 res://anim/）。
2. 挂载 anim_player.gd 的 PixelForgeAnim 节点，调用：
       var a := PixelForgeAnim.new()
       a.load_package("res://anim/")
       add_child(a)
       a.play("", true)
3. 蒙皮轨道用 ArrayMesh 逐帧重建——顶点已经烘焙，引擎侧不需要权重计算。

注意
----
- 导出的是**逐帧烘焙**数据，不是骨骼驱动的运行时蒙皮。好处是所见即所得；
  代价是文件更大，且改了骨骼要重新导出。
- 贴图不带 AI 角标；需要合规标识请在导出后自行叠加。
"""


def build_zip(proj: Dict[str, Any], get_asset_path) -> bytes:
    """打包 anim.json + textures + 运行时 + 说明。"""
    textures: Dict[str, Image.Image] = {}
    for t in proj.get("tracks") or []:
        aid = t.get("assetId")
        if not aid or aid in textures:
            continue
        p = get_asset_path(aid)
        if p is None:
            continue
        textures[aid] = Image.open(p).convert("RGBA")

    data = build_anim_data(proj, textures)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("anim.json", json.dumps(data, ensure_ascii=False, indent=1))
        for aid, img in textures.items():
            b = io.BytesIO()
            img.save(b, "PNG")
            zf.writestr(f"textures/{aid}.png", b.getvalue())
        zf.writestr("anim_player.gd", GDSCRIPT)
        zf.writestr("README.txt", README)
    return buf.getvalue()
