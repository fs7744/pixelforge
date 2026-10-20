"""真实序列帧工程 → Godot 引擎实跑 → 与后端逐帧比对。

比 godot_verify.py 更进一步：那边用的是构造的合成工程，这边用**真的**
从编辑器导出的包（走 /export/godot 端点），验证序列帧轨道在引擎里
切格正确（hframes/vframes/frame 与后端裁的格子一致）。

跑法：python backend/seq_godot_verify.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import godot_verify as G  # noqa: E402
from backend.app import storage  # noqa: E402
from backend.app.anim import render as anim_render  # noqa: E402

BASE = os.environ.get("PF_BASE", "http://127.0.0.1:8787")


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=60, trust_env=False)
    # 找一个带序列帧轨道的工程
    projects = client.get("/api/anim/projects").json().get("projects") or []
    target = None
    for p in projects:
        full = client.get(f"/api/anim/projects/{p['id']}").json()
        if any(t.get("seq") for t in (full.get("tracks") or [])):
            target = (p["id"], full)
            break
    if not target:
        print("[seq-godot] 没有带序列帧轨道的工程，跳过")
        return 0
    pid, proj = target
    print(f"[seq-godot] 工程 {pid}：{len(proj.get('tracks') or [])} 条轨道")

    godot = G.find_godot()
    if not godot:
        print("[seq-godot] 未找到 Godot，跳过")
        return 0

    res = client.post(f"/api/anim/projects/{pid}/export/godot")
    work = Path(tempfile.mkdtemp(prefix="pf-seq-godot-"))
    # 导出包的结构是「包根即 anim/ 的内容」（anim.json / textures/ / anim_player.gd），
    # 而运行时里写死了 res://anim/anim.json。所以这里要放到 anim/ 子目录下，
    # 否则 Godot 找不到文件会一直等（headless 下不报错，就是不出图）。
    anim_dir = work / "anim"
    anim_dir.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        zf.extractall(anim_dir)

    def loader(aid):
        item = storage.get_asset(aid)
        return Image.open(storage.asset_path(item)).convert("RGBA") if item else None

    # 复用 godot_verify 的渲染流程：它用的是带显示上下文的 project.godot
    # （截图需要显示上下文，纯 headless 出不了图），这里别自己造。
    shots = G.run_godot_render(godot, work, proj)
    if not shots:
        print("[seq-godot] 引擎渲染不可用（多半是没显示上下文 / 贴图没导入）")
        return 1

    total = int(proj.get("frames") or 0)
    bad = 0
    for f in range(total):
        shot = work / "out" / f"f{f}.png"
        if not shot.exists():
            print(f"  x 第 {f} 帧缺图")
            bad += 1
            continue
        got = Image.open(shot).convert("RGBA")
        exp = anim_render.render_frame(proj, f, loader)
        if got.size != exp.size:
            print(f"  x 第 {f} 帧尺寸不同：引擎 {got.size} / 后端 {exp.size}")
            bad += 1
            continue
        a = np.asarray(exp, np.int16)
        b = np.asarray(got, np.int16)
        mad = float(np.abs(a[..., :3] - b[..., :3]).mean())
        # 序列帧轨道：引擎切错格的话颜色会差很多（不同姿势的色差）
        if mad > 6.0:
            print(f"  x 第 {f} 帧引擎与后端偏差过大：RGB MAD {mad:.2f}")
            bad += 1
        else:
            print(f"  - 第 {f} 帧一致（RGB MAD {mad:.2f}）")
    if bad:
        print(f"[seq-godot] {bad} 帧不一致")
        return 1
    print(f"[seq-godot] {total} 帧全部一致：引擎切格与后端裁格相同")
    return 0


if __name__ == "__main__":
    sys.exit(main())
