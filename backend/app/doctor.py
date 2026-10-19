"""运行环境体检（移植自 FrameBaker 的 apps/server/src/doctor.ts）。

把「为什么我的生成不工作」变成一组结构化检查：每项
``{id, ok, label, detail}``，detail 里直接给可执行的修复命令
（哪个脚本、装什么），而不是让用户自己翻文档。

与 start.sh --check 的分工：那个在启动前跑（bash 层面，文本输出）；
这个在服务运行中可随时调（GET /api/doctor），设置页一键体检用。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .config import ROOT, load_settings


def _check(cid: str, ok: bool, label: str, detail: str) -> Dict[str, Any]:
    return {"id": cid, "ok": bool(ok), "label": label, "detail": detail}


def _model_dir(rel: str) -> str:
    return str(ROOT / rel)


def run_doctor() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    # 1) 设置可读（settings.json 损坏时 load_settings 会静默退回默认，
    #    这里显式探一次，坏文件应该让人知道）
    try:
        s = load_settings()
        checks.append(_check("settings", True, "设置文件",
                             "data/settings.json 可读（损坏时会自动退回默认值）"))
    except Exception as e:
        s = {}
        checks.append(_check("settings", False, "设置文件", f"读取失败：{e}"))

    # 2) 资产目录可写（探针文件写后即删）
    try:
        from .config import ASSETS_DIR
        probe = ASSETS_DIR / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append(_check("assets-dir", True, "资产目录", f"{ASSETS_DIR} 可写"))
    except Exception as e:
        checks.append(_check("assets-dir", False, "资产目录", f"不可写：{e}"))

    # 3) Web 依赖（缺任何一样服务起不来，能跑到这里说明都在，仍然显式确认）
    missing = []
    for mod in ("fastapi", "uvicorn", "PIL", "numpy"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    checks.append(_check("web-deps", not missing, "Web 依赖",
                         "fastapi / uvicorn / pillow / numpy 齐全" if not missing
                         else f"缺失：{', '.join(missing)}（重跑 ./start.sh）"))

    # 4) cv2（抠图 GrabCut 路径用；缺了退纯颜色键，效果变差但不炸）
    try:
        import cv2  # noqa: F401
        checks.append(_check("cv2", True, "OpenCV（GrabCut 抠图）", "已安装"))
    except Exception:
        checks.append(_check("cv2", False, "OpenCV（GrabCut 抠图）",
                             "未安装：抠图会退到纯颜色键（pip install opencv-python-headless）"))

    # 5) u2net 权重（可选增强：不装退 GrabCut，遵循 FrameBaker 的
    #    「模型未缓存不算失败」原则——服务本身是好的）
    u2net = os.path.expanduser(os.getenv("U2NET_HOME", "~/.u2net"))
    onnx = os.path.join(u2net, "u2net.onnx")
    if os.path.exists(onnx):
        checks.append(_check("u2net", True, "u2net 深度抠图权重",
                             f"已就绪（{onnx}）"))
    else:
        checks.append(_check("u2net", True, "u2net 深度抠图权重（可选）",
                             f"未安装，抠图退 GrabCut：把 u2net.onnx 放到 {u2net}/（README 有命令）"))

    # 6) torch + GPU（AI 生成的前提；没装只是不能生成，其余功能照常）
    torch_ok = False
    try:
        import torch  # noqa: F401
        torch_ok = True
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
            checks.append(_check("gpu", True, "GPU 推理",
                                 f"{name} · {vram}GB 显存"))
        else:
            checks.append(_check("gpu", False, "GPU 推理",
                                 "torch 已装但 CUDA 不可用：CPU 上跑不动这些底座"))
    except Exception:
        checks.append(_check("gpu", False, "GPU 推理",
                             "未安装 torch：跑 ./start.sh --gpu 安装推理依赖（约 2.9GB）"))

    # 7) 模型底座（每个族一个检查；缺哪个给哪个的下载命令。
    #    Z-Image 是唯一必需底座——没有它任何生成都跑不了）
    models = [
        ("model-zimage", "models/z-image-turbo", "生成底座 Z-Image-Turbo",
         "bash backend/fetch_zimage.sh", True),
        ("model-qwen-edit", "models/qwen-image-edit-2511", "编辑底座 Qwen-Image-Edit-2511",
         "bash backend/fetch_qwen_edit.sh", False),
        ("model-wan", "models/wan2.2-ti2v-5b", "视频底座 Wan2.2 TI2V-5B",
         "PYTHONPATH=. python backend/fetch_wan22.py", False),
    ]
    for cid, rel, label, fix, required in models:
        p = _model_dir(rel)
        if os.path.isdir(p):
            checks.append(_check(cid, True, label, f"已就绪（{p}）"))
        else:
            tag = "（必需）" if required else "（可选）"
            checks.append(_check(cid, not required, f"{label} {tag}",
                                 f"未找到 {p}：{fix}"))

    ok_count = sum(1 for c in checks if c["ok"])
    return {
        "ok": ok_count == len(checks),
        "summary": f"{ok_count}/{len(checks)} 项就绪",
        "checks": checks,
        "activeProvider": (s or {}).get("activeProvider", "diffusers"),
        "torch": torch_ok,
    }
