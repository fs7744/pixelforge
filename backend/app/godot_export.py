"""Godot 工程导出：SpriteFrames 资源 + AnimatedSprite2D 场景 + 视差参数表。

产出 ZIP 可直接解压进 Godot 项目（res:// 相对路径已写好）。
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from typing import Any, Dict, List, Tuple

from PIL import Image

from . import storage
from .config import DATA_DIR, load_settings

EXPORTS_DIR = DATA_DIR / "exports"


def _write_zip(zf: zipfile.ZipFile, entries: List[Tuple[str, bytes]]) -> None:
    """写文件 + SHA256SUMS.json 清单（移植自 h3-character-studio 的
    delivery.assemble：交付包里每个文件都带摘要，接收方可以校验拿到的东西
    有没有被改坏；manifest 自身不进清单）。"""
    for arcname, data in entries:
        zf.writestr(arcname, data)
    zf.writestr("SHA256SUMS.json", json.dumps({
        "version": "1.0",
        "files": [{"path": a, "sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)}
                  for a, d in entries],
    }, ensure_ascii=False, indent=2))


def _ai_label_enabled() -> bool:
    """设置里的「下载时补 AI 标识」开关，同时控制导出包里的显式标记文件。

    默认关：图像像素永远不烧角标，导出包也不附带说明文件；打开后单张下载
    会叠角标，导出 ZIP 里补一份 AI_GENERATED.txt（都不改动画面像素）。
    """
    try:
        return bool(load_settings().get("aiLabel"))
    except Exception:
        return False


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "asset"


def _sheet_and_index(run: Dict[str, Any], artifacts: List[str]) -> Tuple[Image.Image, Dict[str, Any]]:
    """找到 compose 步骤的索引信息。"""
    for step in run["steps"]:
        meta = step.get("meta") or {}
        if meta.get("frameSize") and meta.get("frames"):
            item = storage.get_asset(artifacts[0])
            if item:
                img = Image.open(storage.asset_path(item)).convert("RGBA")
                fw = int(meta["frameSize"]["w"])
                fh = int(meta["frameSize"]["h"])
                cols = int(meta.get("cols", 4))
                rows = (int(meta["frames"]) + cols - 1) // cols
                return img, {"fw": fw, "fh": fh, "cols": cols, "rows": rows, "frames": int(meta["frames"]),
                             "sheet": img.size}
    # 兜底：单图当一帧
    item = storage.get_asset(artifacts[0])
    img = Image.open(storage.asset_path(item)).convert("RGBA")
    return img, {"fw": img.width, "fh": img.height, "cols": 1, "rows": 1, "frames": 1, "sheet": img.size}


def build_godot_package(artifacts: List[str], run: Dict[str, Any]) -> Dict[str, Any]:
    """动画包：sprites/*.png + SpriteFrames.tres + player.tscn + README。"""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe(f"pf_{run['id']}")
    sheet, info = _sheet_and_index(run, artifacts)
    fw, fh, cols, rows, total = info["fw"], info["fh"], info["cols"], info["rows"], info["frames"]
    fps = 10.0

    sub_resources = []
    frame_entries = []
    for i in range(total):
        c, r = i % cols, i // cols
        rid = f"AtlasTexture_{i}"
        sub_resources.append(
            f'[sub_resource type="AtlasTexture" id="{rid}"]\n'
            f'atlas = ExtResource("1_sheet")\n'
            f"region = Rect2({c * fw}, {r * fh}, {fw}, {fh})\n"
        )
        frame_entries.append('{\n"duration": 1.0,\n"texture": SubResource("%s")\n}' % rid)

    load_steps = len(sub_resources) + 2
    tres = (
        f'[gd_resource type="SpriteFrames" load_steps={load_steps} format=3]\n\n'
        f'[ext_resource type="Texture2D" path="res://sprites/{name}.png" id="1_sheet"]\n\n'
        + "\n".join(sub_resources)
        + "\n[resource]\nanimations = [{\n"
        + '"frames": [' + ", ".join(frame_entries) + "],\n"
        + f'"loop": true,\n"name": &"run",\n"speed": {fps}\n'
        + "}]\n"
    )

    tscn = (
        "[gd_scene load_steps=2 format=3]\n\n"
        f'[ext_resource type="SpriteFrames" path="res://sprites/{name}_frames.tres" id="1_frames"]\n\n'
        '[node name="Player" type="AnimatedSprite2D"]\n'
        'sprite_frames = ExtResource("1_frames")\n'
        'animation = &"run"\n'
        'autoplay = "run"\n'
    )

    readme = f"""# {name} · Godot 导入说明

1. 把 `sprites/` 与 `scenes/` 复制进你的 Godot 项目（保持目录结构）。
2. 打开 `scenes/{name}_player.tscn`，即可看到自动播放的 AnimatedSprite2D。
3. 帧规格：{fw}×{fh} · {total} 帧 · {fps}fps（每帧 {round(1/fps, 3)}s）。
4. 若需替换贴图：同名覆盖 `sprites/{name}.png`，保持 {cols} 列网格即可。
5. 动画名称为 `run`，可在 `{name}_frames.tres` 中改名或增加动作组。

由 PixelForge 生成 · 本地推理 · {time.strftime('%Y-%m-%d %H:%M')}
"""

    index_json = {
        "version": "1.0",
        "generator": "pixelforge",
        "frame_size": {"w": fw, "h": fh},
        "sheet_size": {"w": info["sheet"][0], "h": info["sheet"][1]},
        "grid": {"cols": cols, "rows": rows},
        "fps": fps,
        "frames": [
            {"i": i, "x": (i % cols) * fw, "y": (i // cols) * fh, "w": fw, "h": fh, "t": round(i / fps, 3)}
            for i in range(total)
        ],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        png_buf = io.BytesIO()
        sheet.save(png_buf, "PNG")
        entries: List[Tuple[str, bytes]] = [
            (f"sprites/{name}.png", png_buf.getvalue()),
            (f"sprites/{name}_frames.tres", tres.encode("utf-8")),
            (f"scenes/{name}_player.tscn", tscn.encode("utf-8")),
            ("pixelforge_index.json", json.dumps(index_json, ensure_ascii=False, indent=2).encode("utf-8")),
            ("README.md", readme.encode("utf-8")),
        ]
        # 中间产物不带角标（会污染像素）；显式标记文件跟随设置开关
        if _ai_label_enabled():
            entries.append((
                "AI_GENERATED.txt",
                ("本素材由 PixelForge 使用生成式人工智能模型产出。\n"
                 "依据《人工智能生成合成内容标识办法》，对外发布时请保留显式"
                 "「AI 生成」标识，并勿删除 pixelforge_index.json 中的 generator 字段（隐式标识）。\n"
                 f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n"
                 f"帧规格：{fw}x{fh} × {total} 帧 @ {fps}fps\n").encode("utf-8"),
            ))
        _write_zip(zf, entries)

    out = EXPORTS_DIR / f"{name}_godot.zip"
    out.write_bytes(buf.getvalue())

    return {
        "kind": "godot",
        "zip": f"/exports/{out.name}",
        "frames": total,
        "frameSize": [fw, fh],
        "fps": fps,
        "files": [a for a, _ in entries] + ["SHA256SUMS.json"],
        "note": f"ZIP 大小 {round(len(buf.getvalue()) / 1024, 1)} KB · 解压即可进 Godot",
    }


PARALLAX_SCALES = [0.05, 0.2, 0.5, 1.0, 1.15]


# --------------------------------------------------------------------------- #
# 四方向角色包（四方向行走工作流用；思路来自 h3 的 delivery 四方向清单）
# --------------------------------------------------------------------------- #
FOUR_DIRS = ("down", "up", "left", "right")
_DIR_CN = {"down": "下（正面）", "up": "上（背面）", "left": "左", "right": "右"}

_DIR_CONTROLLER = """\
extends Node2D
## 四方向角色控制器（PixelForge 导出）
## 用法：把本脚本挂到含四个 AnimatedSprite2D 子节点（$Down/$Up/$Left/$Right）的节点上，
## 调用 set_direction("left") 切换方向；未给定的方向沿用上一次。

const DIRS := ["down", "up", "left", "right"]

func set_direction(dir_name: String) -> void:
    assert(dir_name in DIRS, "unknown direction: " + dir_name)
    for d in DIRS:
        var node := get_node_or_null(d.capitalize())
        if node:
            node.visible = (d == dir_name)
"""


def build_fourdir_package(directions: List[Dict[str, Any]], run: Dict[str, Any]) -> Dict[str, Any]:
    """四方向角色包：每方向 1 个 SpriteFrames + 4 张表 + 场景 + 控制器 + 清单。

    directions: [{"id": "down", "sheetAssetId": ..., "frames": N, "frameSize": [w,h]}, ...]
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe(f"pf_{run['id']}_4dir")
    if not directions:
        raise RuntimeError("没有可导出的四方向素材")

    entries: List[Tuple[str, bytes]] = []
    tres_files: List[str] = []
    index_dirs: Dict[str, Any] = {}

    for d in directions:
        did = str(d["id"])
        item = storage.get_asset(str(d["sheetAssetId"]))
        if not item:
            raise RuntimeError(f"方向素材缺失：{did} → {d['sheetAssetId']}")
        img = Image.open(storage.asset_path(item)).convert("RGBA")
        fw, fh = int(d["frameSize"][0]), int(d["frameSize"][1])
        total = int(d["frames"])

        sheet_name = f"walk_{did}.png"
        tres_name = f"walk_{did}_frames.tres"
        b = io.BytesIO()
        img.save(b, "PNG")
        entries.append((f"sprites/{sheet_name}", b.getvalue()))

        subs, frames = [], []
        for i in range(total):
            rid = f"AtlasTexture_{did}_{i}"
            subs.append(
                f'[sub_resource type="AtlasTexture" id="{rid}"]\n'
                f'atlas = ExtResource("1_sheet")\n'
                f"region = Rect2({i * fw}, 0, {fw}, {fh})\n"
            )
            frames.append('{\n"duration": 1.0,\n"texture": SubResource("%s")\n}' % rid)
        tres = (
            f'[gd_resource type="SpriteFrames" load_steps={len(subs) + 2} format=3]\n\n'
            f'[ext_resource type="Texture2D" path="res://sprites/{sheet_name}" id="1_sheet"]\n\n'
            + "\n".join(subs)
            + "\n[resource]\nanimations = [{\n"
            + '"frames": [' + ", ".join(frames) + "],\n"
            + f'"loop": true,\n"name": &"walk_{did}",\n"speed": 10.0\n'
            + "}]\n"
        )
        entries.append((f"sprites/{tres_name}", tres.encode("utf-8")))
        tres_files.append(tres_name)
        index_dirs[did] = {"sheet": f"sprites/{sheet_name}", "frames": total,
                           "frameSize": [fw, fh], "animation": f"walk_{did}",
                           "label": _DIR_CN.get(did, did)}

    scene = (
        # load_steps = 资源数 + 1；本场景只有 ext_resource（SpriteFrames×N + 脚本）
        f"[gd_scene load_steps={len(directions) + 2} format=3]\n\n"
        + "\n".join(
            f'[ext_resource type="SpriteFrames" path="res://sprites/{f}" id="1_{d}"]'
            for d, f in zip([x["id"] for x in directions], tres_files))
        + "\n[ext_resource type=\"Script\" path=\"res://scripts/direction_controller.gd\" id=\"2_ctrl\"]\n\n"
        '[node name="Character" type="Node2D"]\n'
        'script = ExtResource("2_ctrl")\n\n'
        + "\n".join(
            f'[node name="{str(d["id"]).capitalize()}" type="AnimatedSprite2D" parent="."]\n'
            f'sprite_frames = ExtResource("1_{d["id"]}")\n'
            f'animation = &"walk_{d["id"]}"\n'
            f'autoplay = "walk_{d["id"]}"\n'
            f'visible = {"true" if d["id"] == "down" else "false"}\n'
            for d in directions)
    )
    entries.append((f"scenes/{name}.tscn", scene.encode("utf-8")))
    entries.append(("scripts/direction_controller.gd", _DIR_CONTROLLER.encode("utf-8")))
    entries.append(("pixelforge_directions.json", json.dumps({
        "version": "1.0", "generator": "pixelforge", "animations": index_dirs,
        "directions": [d["id"] for d in directions],
    }, ensure_ascii=False, indent=2).encode("utf-8")))
    entries.append(("README.md", (
        f"# {name} · 四方向角色导入说明\n\n"
        "1. 把 `sprites/`、`scenes/`、`scripts/` 复制进 Godot 项目。\n"
        "2. 打开 `scenes/" + name + ".tscn`：四个方向各一个 AnimatedSprite2D，默认显示 down。\n"
        "3. 用 `Character.set_direction(\"left\")` 切方向（见 scripts/direction_controller.gd）。\n"
        "4. 每个方向一条横排精灵表，帧数相同（每方向 " + str(directions[0].get("frames", "?")) + " 帧）。\n"
        "5. 动画名：`walk_down` / `walk_up` / `walk_left` / `walk_right`。\n\n"
        "由 PixelForge 生成 · " + time.strftime("%Y-%m-%d %H:%M") + "\n"
    ).encode("utf-8")))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_zip(zf, entries)
    out = EXPORTS_DIR / f"{name}.zip"
    out.write_bytes(buf.getvalue())
    return {
        "kind": "godot_fourdir",
        "zip": f"/exports/{out.name}",
        "directions": [d["id"] for d in directions],
        "files": [a for a, _ in entries] + ["SHA256SUMS.json"],
        "note": f"ZIP 大小 {round(len(buf.getvalue()) / 1024, 1)} KB · 四方向各一条精灵表",
    }


def build_parallax_package(artifacts: List[str], run: Dict[str, Any]) -> Dict[str, Any]:
    """视差包：layers/*.png + scroll_scale.json + parallax.tscn + README。"""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe(f"pf_{run['id']}_parallax")

    layers: List[Tuple[str, Image.Image]] = []
    for i, aid in enumerate(artifacts):
        item = storage.get_asset(aid)
        if not item:
            continue
        layers.append((f"layer_{i + 1:02d}.png", Image.open(storage.asset_path(item)).convert("RGBA")))
    if not layers:
        raise RuntimeError("没有可导出的图层")

    w, h = layers[0][1].size
    ext_lines, nodes = [], []
    scroll = []
    for i, (fname, img) in enumerate(layers):
        scale = PARALLAX_SCALES[min(i, len(PARALLAX_SCALES) - 1)]
        ext_lines.append(f'[ext_resource type="Texture2D" path="res://layers/{fname}" id="1_l{i}"]')
        nodes.append(
            f'[node name="Layer{i + 1}" type="ParallaxLayer" parent="."]\n'
            f"motion_scale = Vector2({scale}, 1.0)\n\n"
            f'[node name="Sprite{i + 1}" type="Sprite2D" parent="Layer{i + 1}"]\n'
            f'texture = ExtResource("1_l{i}")\n'
            f"centered = false\n"
        )
        scroll.append({"layer": i + 1, "file": fname, "scrollScaleX": scale,
                       "position": [0, 0], "size": list(img.size)})

    tscn = (
        f"[gd_scene load_steps={len(layers) + 1} format=3]\n\n"
        + "\n".join(ext_lines)
        + '\n\n[node name="Parallax" type="ParallaxBackground"]\n\n'
        + "\n".join(nodes)
    )

    readme = f"""# {name} · 视差场景导入说明

1. 复制 `layers/` 与 `scenes/` 进 Godot 项目。
2. 打开 `scenes/{name}.tscn`，相机横向移动即可看到景深。
3. 所有图层同尺寸 {w}×{h}、同原点，导入后初始位置一致，只靠 `motion_scale` 体现远近。
4. `scroll_scale.json` 给出每层建议系数：
{chr(10).join(f'   - {s["file"]} → {s["scrollScaleX"]}' for s in scroll)}
5. 若角色移动超一屏宽度，需为每层制作左右无缝延展版本（PixelForge「无缝延展」管线）。

注意：Engine 4.3+ 推荐改用 Parallax2D 节点；本模板用 ParallaxBackground，兼容 4.0–4.4。

由 PixelForge 生成 · {time.strftime('%Y-%m-%d %H:%M')}
"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        entries: List[Tuple[str, bytes]] = []
        for fname, img in layers:
            b = io.BytesIO()
            img.save(b, "PNG")
            entries.append((f"layers/{fname}", b.getvalue()))
        entries.append((f"scenes/{name}.tscn", tscn.encode("utf-8")))
        entries.append(("scroll_scale.json", json.dumps(
            {"canvas": {"w": w, "h": h}, "layers": scroll},
            ensure_ascii=False, indent=2).encode("utf-8")))
        entries.append(("README.md", readme.encode("utf-8")))
        if _ai_label_enabled():
            entries.append((
                "AI_GENERATED.txt",
                ("本素材由 PixelForge 使用生成式人工智能模型产出。\n"
                 "依据《人工智能生成合成内容标识办法》，对外发布时请保留显式"
                 "「AI 生成」标识，并勿删除 scroll_scale.json 中的 generator 字段（隐式标识）。\n"
                 f"生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n"
                 f"画布：{w}x{h} · {len(layers)} 层\n").encode("utf-8"),
            ))
        _write_zip(zf, entries)

    out = EXPORTS_DIR / f"{name}.zip"
    out.write_bytes(buf.getvalue())

    return {
        "kind": "godot_parallax",
        "zip": f"/exports/{out.name}",
        "layers": len(layers),
        "canvas": [w, h],
        "scrollScale": [s["scrollScaleX"] for s in scroll],
        "files": [a for a, _ in entries] + ["SHA256SUMS.json"],
        "note": f"ZIP 大小 {round(len(buf.getvalue()) / 1024, 1)} KB",
    }
