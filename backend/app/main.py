"""PixelForge MVP · FastAPI 后端。

启动：
    uvicorn backend.app.main:app --reload --port 8787   （在 pixelforge 根目录）
"""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import storage, templates
from .config import ASSETS_DIR, DATA_DIR, ROOT, load_settings, save_settings
from .pipelines import matte as matte_pipe
from .pipelines import resize as resize_pipe
from .pipelines import sheet as sheet_pipe
from .pipelines import watermark as wm_pipe
from .providers import REGISTRY, get_provider

app = FastAPI(title="PixelForge MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 基础 ----------
@app.get("/api/health")
async def health():
    s = load_settings()
    return {"ok": True, "app": "pixelforge", "version": "0.1.0", "activeProvider": s.get("activeProvider")}


@app.get("/api/templates")
async def get_templates():
    return {"categories": templates.CATEGORIES, "templates": templates.TEMPLATES}


@app.get("/api/settings")
async def get_settings():
    s = load_settings()
    # 不回传明文 key
    safe = json.loads(json.dumps(s))
    for pid, cfg in s.get("providers", {}).items():
        if safe["providers"][pid].get("apiKey"):
            safe["providers"][pid]["apiKey"] = "••••••••"
        safe["providers"][pid]["hasKey"] = bool(cfg.get("apiKey"))
    return safe


class SettingsPayload(BaseModel):
    activeProvider: Optional[str] = None
    providers: Optional[Dict[str, Dict[str, Any]]] = None
    defaults: Optional[Dict[str, Any]] = None
    aiLabel: Optional[bool] = None


@app.put("/api/settings")
async def put_settings(payload: SettingsPayload):
    data = payload.model_dump(exclude_none=True)
    # 掩码值不覆盖真实 key
    for pid, cfg in (data.get("providers") or {}).items():
        if cfg.get("apiKey") and set(cfg["apiKey"]) == {"•"}:
            cfg.pop("apiKey", None)
    return save_settings(data)


@app.get("/api/providers")
async def get_providers():
    s = load_settings()
    out = []
    for pid, cls in REGISTRY.items():
        cfg = s["providers"].get(pid, {})
        out.append({
            "id": pid,
            "label": cls.label,
            "active": s.get("activeProvider") == pid,
            "configured": bool(cfg.get("baseUrl")) if pid != "mock" else True,
            "hasKey": bool(cfg.get("apiKey")),
            "model": cfg.get("model", ""),
        })
    return {"providers": out}


@app.post("/api/providers/{pid}/test")
async def test_provider(pid: str):
    s = load_settings()
    cfg = s["providers"].get(pid)
    if cfg is None:
        raise HTTPException(404, "未知 Provider")
    provider = get_provider(pid, cfg)
    return await provider.test()


# ---------- 生成 ----------
class GeneratePayload(BaseModel):
    prompt: str
    provider: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 28
    cfg: float = 7.5
    seed: Optional[int] = None
    n: int = 1
    templateId: Optional[str] = None


@app.post("/api/generate")
async def generate(payload: GeneratePayload):
    s = load_settings()
    pid = payload.provider or s.get("activeProvider") or "mock"
    cfg = s["providers"].get(pid)
    if cfg is None:
        raise HTTPException(400, f"未配置 Provider: {pid}")
    provider = get_provider(pid, cfg)
    try:
        result = await provider.generate(
            prompt=payload.prompt,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            cfg=payload.cfg,
            seed=payload.seed,
            n=payload.n,
        )
    except Exception as e:
        raise HTTPException(502, str(e))

    items = []
    for img in result.images:
        items.append(storage.save_asset(
            img,
            kind="generated",
            meta={**result.meta, "prompt": payload.prompt, "templateId": payload.templateId},
            ai_label=bool(s.get("aiLabel", True)),
        ))
    return {"images": items, "meta": result.meta}


# ---------- 后处理管线 ----------
async def _read_image(file: UploadFile) -> Image.Image:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空文件")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        raise HTTPException(400, "无法解析图片")


@app.post("/api/pipeline/matte")
async def pipeline_matte(
    file: UploadFile = File(...),
    tolerance: int = Form(80),
    feather: int = Form(5),
    sample: str = Form("topleft"),
):
    img = await _read_image(file)
    out, meta = matte_pipe.connected_matte(img, tolerance=tolerance, feather=feather, sample=sample)
    item = storage.save_asset(out, kind="matte", meta={"pipeline": "matte", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/pipeline/resize")
async def pipeline_resize(
    file: UploadFile = File(...),
    width: int = Form(144),
    height: int = Form(144),
    mode: str = Form("hard"),
    variant: str = Form("v1"),
):
    img = await _read_image(file)
    if mode == "rpgmaker":
        out, meta = resize_pipe.rpgmaker(img, variant=variant)
    elif mode == "fit":
        out = resize_pipe.fit_pad(img, (width, height))
        meta = {"mode": "fit", "size": [width, height]}
    else:
        out = resize_pipe.hard_resize(img, (width, height))
        meta = {"mode": "hard", "size": [width, height], "interpolation": "nearest"}
    item = storage.save_asset(out, kind="resized", meta={"pipeline": "resize", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/pipeline/sheet/split")
async def pipeline_sheet_split(
    file: UploadFile = File(...),
    cols: int = Form(4),
    rows: int = Form(4),
):
    img = await _read_image(file)
    frames, meta = sheet_pipe.split_sheet(img, cols, rows)
    items = [
        storage.save_asset(f, kind="frame", name=f"frame-{i:02d}", meta={"pipeline": "sheet-split", "index": i})
        for i, f in enumerate(frames)
    ]
    return {"frames": items, "index": meta}


@app.post("/api/pipeline/sheet/compose")
async def pipeline_sheet_compose(
    files: List[UploadFile] = File(...),
    cols: int = Form(4),
):
    frames = [await _read_image(f) for f in files]
    sheet, meta = sheet_pipe.compose_sheet(frames, cols)
    item = storage.save_asset(sheet, kind="sheet", meta={"pipeline": "sheet-compose", **meta.get("grid", {})})
    return {"asset": item, "index": meta}


@app.post("/api/pipeline/seamless")
async def pipeline_seamless(
    file: UploadFile = File(...),
    times: int = Form(2),
    blend: int = Form(24),
    mirror: bool = Form(True),
):
    from .pipelines import seamless as seamless_pipe

    img = await _read_image(file)
    out, meta = seamless_pipe.extend_horizontal(img, times=times, blend=blend, mirror=mirror)
    item = storage.save_asset(out, kind="seamless", meta={"pipeline": "seamless", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/pipeline/tile")
async def pipeline_tile(
    file: UploadFile = File(...),
    band: int = Form(12),
    feather: int = Form(3),
):
    from .pipelines import tile as tile_pipe

    img = await _read_image(file)
    out, meta = tile_pipe.make_seamless(img, band=band, feather=feather)
    item = storage.save_asset(out, kind="tile", meta={"pipeline": "tile", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/providers/diffusers/unload")
async def unload_diffusers():
    from .providers.diffusers_local import DiffusersProvider

    DiffusersProvider.unload()
    return {"ok": True, "message": "已卸载模型并清空显存缓存"}


@app.post("/api/pipeline/watermark")
async def pipeline_watermark(
    file: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    size: int = Form(48),
    strength: float = Form(1.0),
):
    img = await _read_image(file)
    mask_img = await _read_image(mask) if mask is not None else None
    out, meta = wm_pipe.remove_watermark(img, mask=mask_img, size=size, strength=strength)
    if not meta.get("applied"):
        return {"asset": None, "meta": meta}
    item = storage.save_asset(out, kind="watermark", meta={"pipeline": "watermark", **meta})
    return {"asset": item, "meta": meta}


# ---------- 资产库 ----------
@app.get("/api/assets")
async def assets(kind: Optional[str] = None):
    return {"assets": storage.list_assets(kind)}


@app.delete("/api/assets/{asset_id}")
async def delete_asset(asset_id: str):
    ok = storage.delete_asset(asset_id)
    if not ok:
        raise HTTPException(404, "资产不存在")
    return {"ok": True}


@app.post("/api/export/zip")
async def export_zip(payload: Dict[str, List[str]]):
    ids = payload.get("ids") or []
    items = [storage.get_asset(i) for i in ids]
    items = [i for i in items if i]
    if not items:
        raise HTTPException(400, "没有可导出的资产")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        index = []
        for item in items:
            p = storage.asset_path(item)
            zf.write(p, arcname=item["file"])
            index.append({k: item[k] for k in ("id", "name", "file", "width", "height", "kind")})
        zf.writestr("index.json", json.dumps(index, ensure_ascii=False, indent=2))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": 'attachment; filename="pixelforge-export.zip"'})


# ---------- 工作流 ----------
class RunPayload(BaseModel):
    input: str
    params: Optional[Dict[str, Any]] = None


class StepPayload(BaseModel):
    prompt: Optional[str] = None
    n: Optional[int] = None
    artifacts: Optional[List[str]] = None


@app.get("/api/workflows")
async def get_workflows():
    from .workflows import list_workflows
    return {"workflows": list_workflows()}


@app.post("/api/workflows/{wf_id}/runs")
async def create_workflow_run(wf_id: str, payload: RunPayload):
    from .workflows import create_run
    try:
        return create_run(wf_id, payload.input, payload.params)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/api/runs")
async def get_runs():
    from .workflows import list_runs
    return {"runs": list_runs()}


@app.get("/api/runs/{run_id}")
async def get_run_detail(run_id: str):
    from .workflows import get_run
    run = get_run(run_id)
    if not run:
        raise HTTPException(404, "run 不存在")
    return run


@app.post("/api/runs/{run_id}/steps/{step_id}/run")
async def run_single_step(run_id: str, step_id: str, payload: StepPayload):
    from .workflows import run_step
    try:
        return await run_step(run_id, step_id, payload.model_dump(exclude_none=True))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, str(e))


@app.post("/api/runs/{run_id}/run-all")
async def run_workflow_all(run_id: str):
    from .workflows import run_all
    try:
        return await run_all(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(500, str(e))


# ---------- 静态资源 ----------
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(ASSETS_DIR)), name="files")

EXPORTS_DIR = DATA_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")
