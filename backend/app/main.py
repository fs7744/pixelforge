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

from . import storage, styles, templates, zh2en
from .config import ASSETS_DIR, DATA_DIR, ROOT, STYLE_SAMPLES_DIR, load_settings, save_settings
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
    # 中文提示词自动译成英文（SD 系模型必需）；显式关闭可原样送模型
    autoTranslate: bool = True
    # 美术画风 id（见 /api/styles）；空 = 不拼风格词
    style: Optional[str] = None


@app.get("/api/styles")
async def styles_endpoint():
    """美术画风库（游戏行业典型风格，生成时拼接在提示词尾部）。"""
    return {"styles": styles.list_styles()}


@app.post("/api/translate")
async def translate_prompt_endpoint(payload: Dict[str, Any]):
    text = str(payload.get("text") or "")
    # useMt=False 时只走领域词典（毫秒级、零依赖），遇到漏译会返回 unmatched
    use_mt = bool(payload.get("useMt", True))
    english, meta = zh2en.translate_prompt(text, use_mt=use_mt)
    return {"english": english, **meta}


@app.post("/api/generate")
async def generate(payload: GeneratePayload):
    s = load_settings()
    pid = payload.provider or s.get("activeProvider") or "mock"
    cfg = s["providers"].get(pid)
    if cfg is None:
        raise HTTPException(400, f"未配置 Provider: {pid}")
    provider = get_provider(pid, cfg)
    # 画风链路（按 Provider 语言能力分流）：
    # - 中文模型（Z-Image/Qwen-Image-Edit）：中文风格词直接拼在提示词尾部，不走翻译；
    # - SD 系（CLIP）：先译中文（add_style=False，不让 zh2en 补它的默认风格词，
    #   否则会和用户选的画风打架），再把英文风格词拼到译后尾部——风格词放尾部
    #   符合 SD 注意力权重分布，也不被机翻毁掉。
    # 注意：SD 分支不复用 apply_style 合并的 prompt（那会给英文 prompt 拼两遍），
    # 只用它的 negative / style / suggestPost 字段。
    chinese_model = bool(getattr(provider, "supports_zh", False))
    styled = styles.apply_style(payload.prompt, payload.style, chinese_model)
    translation = None
    if chinese_model:
        effective_prompt = styled["prompt"]
    else:
        style = styles.get_style(payload.style)
        if payload.autoTranslate and zh2en.has_chinese(payload.prompt):
            # 选了画风才关掉 zh2en 的默认风格兜底（否则两段风格词打架）；
            # 没选画风时保持原行为，让它补 DEFAULT_STYLE
            effective_prompt, translation = zh2en.translate_prompt(
                payload.prompt, add_style=style is None)
        else:
            effective_prompt = payload.prompt
        if style:
            effective_prompt = f"{effective_prompt}, {style['promptEn']}"
    # SD 系的画风排除项叠加进 negativePrompt（Z-Image/Qwen-Edit 蒸馏模型不吃负向，
    # 且 BaseProvider.generate 没有 negative 参数——负向词只走 provider cfg）
    if styled["negative"]:
        base_neg = (cfg.get("negativePrompt") or "").strip()
        merged_neg = ", ".join(x for x in (base_neg, styled["negative"]) if x)
        cfg = {**cfg, "negativePrompt": merged_neg}
        provider = get_provider(pid, cfg)
    try:
        result = await provider.generate(
            prompt=effective_prompt,
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
            meta={**result.meta, "prompt": effective_prompt, "sourcePrompt": payload.prompt,
                  "templateId": payload.templateId, "style": styled["style"],
                  "suggestPost": styled["suggestPost"]},
            # 存储不烧角标：单张下载时由 /api/assets/{id}/download 补显式标识，
            # 否则角标会被后续抠图/合成带进精灵表
            ai_label=False,
        ))
    meta = {**result.meta, "prompt": effective_prompt, "promptSource": payload.prompt}
    if translation:
        meta["translation"] = translation
    if styled["style"]:
        meta["style"] = styled["style"]
    if styled["suggestPost"]:
        meta["suggestPost"] = styled["suggestPost"]
    return {"images": items, "meta": meta}


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


@app.post("/api/pipeline/pixelize")
async def pipeline_pixelize(
    file: UploadFile = File(...),
    colors: int = Form(16),
    pixel_size: int = Form(4),
    dither: bool = Form(False),
    palette: str = Form("auto"),
):
    from .pipelines import pixelize as pixelize_pipe

    img = await _read_image(file)
    out, meta = pixelize_pipe.pixelize(img, colors=colors, pixel_size=pixel_size,
                                       dither=dither, palette=palette)
    item = storage.save_asset(out, kind="pixelized", meta={"pipeline": "pixelize", **meta})
    return {"asset": item, "meta": meta}


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


# ---------- AI 编辑管线（Qwen-Image-Edit-2511 指令式编辑） ----------
# 共同特性：单次 40 步约 2-4 分钟；provider 内部 asyncio.to_thread 不阻塞
# 事件循环；加载前自动换出另一大族模型（Z-Image ↔ Qwen-Edit 互斥）。
@app.post("/api/pipeline/watermark/ai")
async def pipeline_watermark_ai(
    file: UploadFile = File(...),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
):
    from .pipelines import ai_edit

    img = await _read_image(file)
    out, meta = await ai_edit.remove_watermark_ai(img, steps=steps, seed=seed)
    item = storage.save_asset(out, kind="watermark", meta={"pipeline": "watermark_ai", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/pipeline/seamless/ai")
async def pipeline_seamless_ai(
    file: UploadFile = File(...),
    times: int = Form(2),
    blend: int = Form(6),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
):
    from .pipelines import ai_edit

    img = await _read_image(file)
    out, meta = await ai_edit.extend_horizontal_ai(
        img, times=times, blend=blend, steps=steps, seed=seed)
    item = storage.save_asset(out, kind="seamless", meta={"pipeline": "seamless_ai", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/pipeline/restyle")
async def pipeline_restyle(
    file: UploadFile = File(...),
    style: str = Form(...),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
):
    from .pipelines import ai_edit

    img = await _read_image(file)
    out, meta = await ai_edit.restyle(img, style, steps=steps, seed=seed)
    item = storage.save_asset(out, kind="restyled", meta={"pipeline": "restyle", **meta})
    return {"asset": item, "meta": meta}


# ---------- 角色三视图（正/侧/背独立生成 + 拼板） ----------
class ThreeViewPayload(BaseModel):
    prompt: str
    style: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None
    width: int = 512
    height: int = 768
    maxSide: int = 768


@app.post("/api/pipeline/threeview")
async def pipeline_threeview(payload: ThreeViewPayload):
    """角色三视图：正/侧/背三张独立生成（帧 2+ 编辑底座接力保一致性）后拼横向板。

    单图提示词让模型「一次画三个视角」会重叠/缺视图，本端点把三视拆成
    三次单角色生成再确定性拼板。返回与 /api/generate 同构：
    images[0] 是拼板，其后依次是正/侧/背单视图。
    """
    from .pipelines import threeview as tv_pipe

    strip, views, meta = await tv_pipe.run_three_view(
        payload.prompt,
        style=payload.style,
        steps=payload.steps,
        cfg=payload.cfg,
        seed=payload.seed,
        width=payload.width,
        height=payload.height,
        max_side=payload.maxSide,
    )
    labels = ["三视图拼板", "正视图", "侧视图", "背视图"]
    items = []
    for i, img in enumerate([strip, *views]):
        items.append(storage.save_asset(
            img,
            kind="generated",
            name=f"threeview-{labels[i]}",
            meta={"pipeline": "threeview", "view": labels[i],
                  "prompt": payload.prompt, "style": meta.get("style"),
                  "mode": meta.get("mode"), "seed": payload.seed},
            ai_label=False,
        ))
    return {"images": items, "meta": meta}


# ---------- 资产库 ----------
@app.get("/api/assets")
async def assets(kind: Optional[str] = None):
    return {"assets": storage.list_assets(kind)}


@app.get("/api/assets/{asset_id}/download")
async def download_asset(asset_id: str, label: int = 1):
    """单张下载：默认在下载时补上显式 AI 角标（存储不烧标，避免污染后续合成）。"""
    item = storage.get_asset(asset_id)
    if not item:
        raise HTTPException(404, "资产不存在")
    path = storage.asset_path(item)
    if not path.exists():
        raise HTTPException(404, "文件缺失")
    if not label:
        return FileResponse(str(path), media_type="image/png", filename=f"{item['name']}.png")
    img = Image.open(path).convert("RGBA")
    img = storage.stamp_ai_label(img)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{asset_id}.png"'})


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

# 画风示例图：gen_style_samples.py 生成后才挂载，避免目录不存在启动报错
if STYLE_SAMPLES_DIR.is_dir():
    app.mount("/style-samples", StaticFiles(directory=str(STYLE_SAMPLES_DIR)), name="style-samples")

EXPORTS_DIR = DATA_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")
