"""Godot 工程导出：SpriteFrames 资源 + AnimatedSprite2D 场景 + 视差参数表。

产出 ZIP 可直接解压进 Godot 项目（res:// 相对路径已写好）。
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from . import storage
from .config import DATA_DIR

EXPORTS_DIR = DATA_DIR / "exports"


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
        zf.writestr(f"sprites/{name}.png", png_buf.getvalue())
        zf.writestr(f"sprites/{name}_frames.tres", tres)
        zf.writestr(f"scenes/{name}_player.tscn", tscn)
        zf.writestr("pixelforge_index.json", json.dumps(index_json, ensure_ascii=False, indent=2))
        zf.writestr("README.md", readme)

    out = EXPORTS_DIR / f"{name}_godot.zip"
    out.write_bytes(buf.getvalue())

    return {
        "kind": "godot",
        "zip": f"/exports/{out.name}",
        "frames": total,
        "frameSize": [fw, fh],
        "fps": fps,
        "files": [f"sprites/{name}.png", f"sprites/{name}_frames.tres",
                  f"scenes/{name}_player.tscn", "pixelforge_index.json", "README.md"],
        "note": f"ZIP 大小 {round(len(buf.getvalue()) / 1024, 1)} KB · 解压即可进 Godot",
    }


PARALLAX_SCALES = [0.05, 0.2, 0.5, 1.0, 1.15]


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
        for fname, img in layers:
            b = io.BytesIO()
            img.save(b, "PNG")
            zf.writestr(f"layers/{fname}", b.getvalue())
        zf.writestr(f"scenes/{name}.tscn", tscn)
        zf.writestr("scroll_scale.json", json.dumps({"canvas": {"w": w, "h": h}, "layers": scroll},
                                                    ensure_ascii=False, indent=2))
        zf.writestr("README.md", readme)

    out = EXPORTS_DIR / f"{name}.zip"
    out.write_bytes(buf.getvalue())

    return {
        "kind": "godot_parallax",
        "zip": f"/exports/{out.name}",
        "layers": len(layers),
        "canvas": [w, h],
        "scrollScale": [s["scrollScaleX"] for s in scroll],
        "files": [f"layers/{f}" for f, _ in layers] + [f"scenes/{name}.tscn", "scroll_scale.json", "README.md"],
        "note": f"ZIP 大小 {round(len(buf.getvalue()) / 1024, 1)} KB",
    }
