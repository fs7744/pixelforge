"""PixelForge MVP · FastAPI 后端。

启动：
    uvicorn backend.app.main:app --reload --port 8787   （在 pixelforge 根目录）
"""
from __future__ import annotations

import io
import json
import zipfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, field_validator

from . import doctor as doctor_mod
from . import storage, styles, templates
from .config import (ASSETS_DIR, CHAR_PIXEL_SAMPLES_DIR, DATA_DIR, ROOT,
                     STYLE_SAMPLES_DIR, load_settings, save_settings)
from .jsonlimits import validate_bounded
from .pipelines import matte as matte_pipe
from .pipelines import resize as resize_pipe
from .pipelines import sheet as sheet_pipe
from .pipelines import watermark as wm_pipe
from .providers import REGISTRY, get_provider


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 启动恢复（移植自 FrameBaker queue.ts）：上次进程崩溃时状态停在
    # running 的步骤标回 failed，否则重启后永远显示「执行中」且重跑被挡
    from .workflows import recover_interrupted_runs

    recovered = recover_interrupted_runs()
    if recovered:
        print(f"[startup] 重置了 {recovered} 个中断工作流的 running 步骤")
    yield


app = FastAPI(title="PixelForge MVP", version="0.1.0", lifespan=lifespan)
# 本地工作站：只可能来自本机浏览器（生产同源；开发态 vite 也代理 /api，
# 见 frontend/vite.config.ts）。原写法 allow_origins=["*"] 与
# allow_credentials=True 并存是 CORS 规范禁止的组合，浏览器会整组拒绝；
# 且本服务不使用 Cookie / 会话，没有任何 credentialed 请求，
# 所以显式列来源 + allow_credentials=False 既合法又够用。
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8787",
        "http://localhost:8787",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def _value_error(request, exc: ValueError):
    """管线里的 ValueError 都是「用户参数不对」（未知画风、无帧可合成等），
    统一按 400 返回；否则 FastAPI 会当 500 并抛出整页堆栈。"""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------- 基础 ----------
@app.get("/api/health")
async def health():
    s = load_settings()
    return {"ok": True, "app": "pixelforge", "version": "0.1.0", "activeProvider": s.get("activeProvider")}


@app.get("/api/doctor")
async def doctor():
    """运行环境体检（移植自 FrameBaker 的 doctor.ts）。

    逐项检查设置 / 资产目录 / 依赖 / GPU / 模型底座，detail 里带可执行的
    修复命令。设置页一键体检用这个；启动前的文本自检走 ./start.sh --check。
    """
    return await run_in_threadpool(doctor_mod.run_doctor)


@app.get("/api/templates")
async def get_templates():
    # 占位符在返回前就填成默认值：前端拿到的是「点了就能跑」的提示词，
    # 用户改「角色 / 动作序列」字段时再按字段值替换那一段。
    return {"categories": templates.CATEGORIES, "templates": templates.api_items()}


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

    @field_validator("providers", "defaults")
    @classmethod
    def _bounded(cls, v):
        # 有界校验（移植自 FrameBaker shared/json.ts）：pydantic 只验类型，
        # 不拦 NaN / 无限 / 深层嵌套——那些会写坏 settings.json
        if v is not None:
            validate_bounded(v)
        return v


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
            "configured": True,
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
    try:
        provider = get_provider(pid, cfg)
    except KeyError:
        raise HTTPException(404, "未知 Provider")
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
    # 美术画风 id（见 /api/styles）；空 = 不拼风格词
    style: Optional[str] = None


@app.get("/api/styles")
async def styles_endpoint():
    """美术画风库（游戏行业典型风格，生成时拼接在提示词尾部）。

    recommended：模板分类 → 默认展示的画风 id。前端按当前模板的 category
    取对应列表，避免 14 个画风无条件全列出来（画风与模板类型不匹配时
    用户根本无从判断该选哪个）。
    """
    return {"styles": styles.list_styles(), "recommended": styles.CATEGORY_STYLES}


@app.post("/api/generate")
async def generate(payload: GeneratePayload):
    s = load_settings()
    pid = payload.provider or s.get("activeProvider") or "diffusers"
    cfg = s["providers"].get(pid)
    if cfg is None:
        raise HTTPException(400, f"未配置 Provider: {pid}")
    provider = get_provider(pid, cfg)
    # 画风链路：中文风格词直接拼到提示词尾部（两个底座都是中文原生，
    # 没有翻译链路，也不需要负向词合并）
    styled = styles.apply_style(payload.prompt, payload.style)
    effective_prompt = styled["prompt"]
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
            # 存储不烧角标：角标会被后续抠图/合成带进精灵表；
            # 需要合规标识时由 /api/assets/{id}/download?label=1 显式补
            ai_label=False,
        ))
    meta = {**result.meta, "prompt": effective_prompt, "promptSource": payload.prompt}
    if styled["style"]:
        meta["style"] = styled["style"]
        meta["styleWords"] = styled["styleWords"]
    if styled["suggestPost"]:
        meta["suggestPost"] = styled["suggestPost"]
    return {"images": items, "meta": meta}


@app.post("/api/generate/ref")
async def generate_with_refs(
    prompt: str = Form(...),
    style: str = Form(""),
    width: int = Form(0),
    height: int = Form(0),
    steps: int = Form(0),
    cfg: float = Form(0.0),
    seed: Optional[int] = Form(None),
    n: int = Form(1),
    refAssetIds: str = Form(""),
    refs: Optional[List[UploadFile]] = File(None),
):
    """参考图生成（图生图）：1–4 张参考图 + 指令 → Qwen-Image-Edit。

    - 参考图两种来源可混用：直接上传（refs）、资产库 id（refAssetIds，逗号分隔）；
    - 宽度/高度缺省（0）时跟随第一张参考图，按 16 对齐；
    - 底座固定走编辑底座：Z-Image 是纯文生图，没有参考图输入，传了也只会
      让用户以为生效——所以这里直接由 provider 改走编辑底座（meta.autoRouted 标记）。
    """
    from .providers.diffusers_local import MAX_REFS

    images: List[Image.Image] = []
    for f in refs or []:
        if f is not None:
            images.append((await _read_image(f)).convert("RGB"))
    for aid in [x.strip() for x in (refAssetIds or "").split(",") if x.strip()]:
        item = storage.get_asset(aid)
        if not item:
            raise HTTPException(400, f"参考图资产不存在：{aid}")
        p = storage.asset_path(item)
        if not p.exists():
            raise HTTPException(400, f"参考图文件缺失：{aid}")
        images.append(Image.open(p).convert("RGB"))
    if not images:
        raise HTTPException(400, "至少需要一张参考图（上传 refs 或指定 refAssetIds）")
    if len(images) > MAX_REFS:
        raise HTTPException(400, f"参考图最多 {MAX_REFS} 张，当前 {len(images)} 张")

    s = load_settings()
    pid = s.get("activeProvider") or "diffusers"
    pcfg = s["providers"].get(pid)
    if pcfg is None:
        raise HTTPException(400, f"未配置 Provider: {pid}")
    provider = get_provider(pid, pcfg)

    styled = styles.apply_style(prompt, style or None)
    w = int(width or images[0].width) // 16 * 16 or 512
    h = int(height or images[0].height) // 16 * 16 or 512
    try:
        result = await provider.generate(
            prompt=styled["prompt"],
            width=w,
            height=h,
            steps=steps or None,
            cfg=cfg or 7.5,
            seed=seed,
            n=max(1, min(int(n or 1), 4)),
            refs=images,
        )
    except Exception as e:
        raise HTTPException(502, str(e))

    meta = {**result.meta, "prompt": styled["prompt"], "promptSource": prompt,
            "mode": "ref2img", "refCount": len(images)}
    if styled["style"]:
        meta["style"] = styled["style"]
        meta["styleWords"] = styled["styleWords"]
    items = [
        storage.save_asset(img, kind="generated", meta=meta, ai_label=False)
        for img in result.images
    ]
    return {"images": items, "meta": meta}


# ---------- 后处理管线 ----------
# 上传上限：误拖进来的几百 MB「图片」会整体读进内存，直接把服务打爆。
# 64MB 对 PNG 精灵表 / 场景图绰绰有余。分块读、超限即断——
# 不能先 read() 完再判断，那时内存已经吃满了。
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


async def _read_image(file: UploadFile) -> Image.Image:
    declared = getattr(file, "size", None)
    if declared and declared > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"文件过大（上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB）")
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"文件过大（上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB）")
        chunks.append(chunk)
    raw = b"".join(chunks)
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
    # 抠图可能跑数秒（u2net CPU 推理），卸载到线程池，别占着事件循环
    out, meta = await run_in_threadpool(
        matte_pipe.connected_matte, img,
        tolerance=tolerance, feather=feather, sample=sample)
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

    def _resize():
        if mode == "rpgmaker":
            out, meta = resize_pipe.rpgmaker(img, variant=variant)
        elif mode == "fit":
            out = resize_pipe.fit_pad(img, (width, height))
            meta = {"mode": "fit", "size": [width, height]}
        else:
            out = resize_pipe.hard_resize(img, (width, height))
            meta = {"mode": "hard", "size": [width, height], "interpolation": "nearest"}
        return out, meta

    out, meta = await run_in_threadpool(_resize)
    item = storage.save_asset(out, kind="resized", meta={"pipeline": "resize", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/pipeline/sheet/split")
async def pipeline_sheet_split(
    file: UploadFile = File(...),
    cols: int = Form(4),
    rows: int = Form(4),
    # 精灵表拆完通常还要逐帧去背；16 帧手动跑 16 次抠图没人受得了。
    # 默认关，避免改变既有调用方的行为。
    matte: bool = Form(False),
    tolerance: int = Form(80),
):
    img = await _read_image(file)

    def _split():
        frames, meta = sheet_pipe.split_sheet(img, cols, rows)
        if matte:
            cleaned = []
            for i, f in enumerate(frames):
                m, minfo = matte_pipe.connected_matte(f, tolerance=tolerance)
                cleaned.append(m)
                meta["frames"][i]["matte"] = minfo
            frames = cleaned
        return frames, meta

    frames, meta = await run_in_threadpool(_split)
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
    sheet, meta = await run_in_threadpool(sheet_pipe.compose_sheet, frames, cols)
    # 交付前自检：单元格与源帧逐位一致（移植自 h3 的 check_bundle）
    checks = await run_in_threadpool(sheet_pipe.verify_composed, sheet, frames, cols)
    item = storage.save_asset(sheet, kind="sheet",
                              meta={"pipeline": "sheet-compose", **meta.get("grid", {}),
                                    "checks": checks})
    return {"asset": item, "index": meta}


@app.post("/api/pipeline/sprite/frame")
async def pipeline_sprite_frame(
    pose: str = Form(...),
    index: int = Form(0),
    total: int = Form(1),
    refAssetId: str = Form(""),
    file: Optional[UploadFile] = File(None),
    subject: str = Form(""),
    style: str = Form(""),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    width: int = Form(512),
    height: int = Form(512),
    maxSide: int = Form(768),
    backdrop: str = Form("neutral"),
):
    """逐帧动作生成：一次一帧，用户填的动作原样进指令。

    为什么不一次出整张精灵表：文生图/编辑底座都没有「第几格画什么」的定位
    能力，一张图排 4×4 时姿势大面积重复（实测 16 格几乎同姿势），动作循环
    根本不能用。改成一次一帧后，模型只需做「换个姿势」一件事，身份由同一张
    参考图钉住，拼板/对齐交给本地确定性管线。

    backdrop：生成时铺什么底色。默认中性灰——实测模型画不出纯色绿，还会把绿
    染到角色边缘（内圈偏底色像素绿幕 76.5% vs 中性灰 33.3%），详见
    pipelines/backdrop.py。
    """
    from .pipelines import sprite_loop

    if not (pose or "").strip():
        raise HTTPException(400, "动作不能为空")
    if not 0 <= index < max(1, total):
        raise HTTPException(400, f"帧序号越界：{index} / 共 {total} 帧")

    ref: Optional[Image.Image] = None
    if refAssetId:
        ref = _load_asset_image(refAssetId)
    elif file is not None:
        ref = (await _read_image(file)).convert("RGB")
    else:
        raise HTTPException(400, "逐帧生成需要一张角色参考图（refAssetId 或 file）")

    img, meta = await sprite_loop.run_frame(
        ref, pose.strip(), index, max(1, total),
        subject=subject, style=style, steps=steps, seed=seed,
        width=width, height=height, max_side=maxSide, backdrop=backdrop,
    )
    item = storage.save_asset(
        img, kind="generated",
        name=f"sprite-{index + 1:02d}-{pose.strip()[:8]}",
        meta={"pipeline": "sprite-frame", "pose": pose.strip(), **meta},
    )
    return {"image": item, "meta": {"pipeline": "sprite-frame", "pose": pose.strip(), **meta}}


class SpriteComposePayload(BaseModel):
    """逐帧结果拼板：按顺序给帧资产 id，cols 决定每行几帧。

    colors > 0：取全局色板再让每帧映射回它（帧间不偏色）；0 = 不锁色。
    """

    frameAssetIds: List[str]
    cols: int = 4
    size: int = 256
    anchor: str = "bottom"
    matte: bool = True
    tolerance: int = 80
    feather: int = 0
    colors: int = 0


@app.post("/api/pipeline/sprite/compose")
async def pipeline_sprite_compose(payload: SpriteComposePayload):
    """逐帧结果 → 规整（共享缩放、底部对齐、抠图）→ 精灵表 + 引擎索引。"""
    from .pipelines import sprite_loop

    ids = [i for i in (payload.frameAssetIds or []) if i]
    if not ids:
        raise HTTPException(400, "没有可合成的帧")
    if len(ids) > 64:
        raise HTTPException(400, f"一次最多合成 64 帧，收到 {len(ids)} 帧")

    imgs = [_load_asset_image(i) for i in ids]
    sheet, frames, index, meta = await run_in_threadpool(
        sprite_loop.compose, imgs, cols=payload.cols, size=payload.size, anchor=payload.anchor,
        matte=payload.matte, tolerance=payload.tolerance, feather=payload.feather,
        colors=payload.colors,
    )
    frame_items = [
        storage.save_asset(
            f, kind="frame", name=f"sprite-frame-{i + 1:02d}",
            meta={"pipeline": "sprite-compose", "index": i, "frameSize": index["frame_size"]},
        )
        for i, f in enumerate(frames)
    ]
    sheet_item = storage.save_asset(
        sheet, kind="sheet", name="sprite-sheet",
        meta={"pipeline": "sprite-compose", **meta, "index": index},
    )
    return {"asset": sheet_item, "frames": frame_items, "index": index,
            "meta": {"pipeline": "sprite-compose", **meta}}


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
    out, meta = await run_in_threadpool(
        pixelize_pipe.pixelize, img,
        colors=colors, pixel_size=pixel_size, dither=dither, palette=palette)
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
    out, meta = await run_in_threadpool(
        seamless_pipe.extend_horizontal, img, times=times, blend=blend, mirror=mirror)
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
    out, meta = await run_in_threadpool(tile_pipe.make_seamless, img, band=band, feather=feather)
    item = storage.save_asset(out, kind="tile", meta={"pipeline": "tile", **meta})
    return {"asset": item, "meta": meta}


@app.post("/api/providers/diffusers/unload")
async def unload_diffusers():
    from .providers.diffusers_local import DiffusersProvider

    await run_in_threadpool(DiffusersProvider.unload)
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
    out, meta = await run_in_threadpool(
        wm_pipe.remove_watermark, img, mask=mask_img, size=size, strength=strength)
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
    blend: int = Form(16),
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
    style: str = Form(...),
    # 输入两种来源可二选一：直接上传，或直接指资产库里的那张（素材库旧图换画风）
    file: Optional[UploadFile] = File(None),
    refAssetId: str = Form(""),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    maxSide: int = Form(1024),
    # 像素类画风默认自动接像素化；关掉可以先看 AI 重绘的原始结果
    pixelize: bool = Form(True),
):
    """图片画风转换：一张图 + 目标画风 → 编辑底座按画风重绘。

    与「参考图生成」的区别：那条是「按指令画一张新图」，这条是「把这张图
    重绘成另一种画风」——主体、构图、角色沿用原图，只换笔触/线条/上色。
    编辑前把长边压到 1024（编辑成本随像素数涨），出图后 Lanczos 放回原尺寸，
    所以转换前后画布尺寸一致，前端可以直接叠前后对比滑块。
    """
    from .pipelines import ai_edit

    # 画风先校验：未知 id 不该先读一遍图再报错
    if styles.get_style(style) is None:
        raise ValueError(f"未知画风：{style}")
    if file is not None:
        img = (await _read_image(file)).convert("RGB")
    elif refAssetId:
        img = _load_asset_image(refAssetId)
    else:
        raise ValueError("需要一张待转换的图片（上传 file 或指定 refAssetId）")

    out, meta = await ai_edit.restyle(
        img, style, steps=steps, seed=seed, max_side=maxSide, pixelize=pixelize)
    item = storage.save_asset(
        out, kind="restyled",
        name=f"换画风-{meta.get('styleName') or style}",
        meta={"pipeline": "restyle", "sourceAssetId": refAssetId or None, **meta},
    )
    return {"asset": item, "meta": {"pipeline": "restyle", **meta}}


# ---------- 生成底色（抠图质量的根因在生成侧，不在抠图参数） ----------
@app.get("/api/pipeline/backdrops")
async def pipeline_backdrops():
    """可选底色：中性灰（推荐）/ 绿幕 / 白。

    默认不是绿幕。实测（640×896、同提示词同 seed、只改背景色）模型声明
    #00FF00 实际画出薄荷绿 [90,199,126] + 径向渐变，还把绿染到角色边缘
    （内圈偏底色像素 76.5%，中性灰只有 33.3%）。详见 pipelines/backdrop.py。
    """
    from .pipelines import backdrop as backdrop_pipe

    return {"backdrops": backdrop_pipe.list_backdrops(),
            "default": backdrop_pipe.DEFAULT}


# ---------- 人物像素化（照片 → 像素角色，三档造型程度） ----------
@app.get("/api/pipeline/char-pixel/levels")
async def char_pixel_levels():
    """三档像素化程度：写实比例 / 半 Q 版 / Q 版。

    档位是「造型语言」而不是「颗粒大小」：头身比、眼睛大小、阴影层数一起变，
    只调 block 大小做不出 Q 版——那是 AI 重绘那一步的事。
    """
    from .pipelines import charpixel

    return {"levels": charpixel.list_levels(), "backgrounds": charpixel.BACKGROUNDS}


@app.post("/api/pipeline/char-pixel")
async def pipeline_char_pixel(
    level: str = Form("mid"),
    prompt: str = Form(""),
    file: Optional[UploadFile] = File(None),
    refAssetId: str = Form(""),
    steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    maxSide: int = Form(1024),
    # 不填则用档位预设；填了就是「在档位造型上微调像素密度」
    grid: Optional[int] = Form(None),
    colors: Optional[int] = Form(None),
    # 背景处理：keep 保留原图 / green 绿幕 / alpha 透明（绿幕→色键去背）
    background: str = Form("keep"),
):
    """人物照片 → 像素角色立绘：AI 按档位重绘，再确定性像素化收尾。

    返回两套资产：
    - asset：像素化成品（kind=charpixel）；
    - draft：AI 原稿（未像素化）。留着它，用户调像素密度/色数时可以只重跑
      毫秒级的像素化，而不是再等 2-4 分钟的 40 步编辑。
    """
    from .pipelines import charpixel

    lv = charpixel.get_level(level)   # 未知档位先报错，别白读一遍图
    if file is not None:
        img = (await _read_image(file)).convert("RGB")
    elif refAssetId:
        img = _load_asset_image(refAssetId)
    else:
        raise ValueError("需要一张人物照片（上传 file 或指定 refAssetId）")

    draft, out, meta = await charpixel.run(
        img, lv["id"], prompt, steps=steps, seed=seed,
        max_side=maxSide, grid=grid, colors=colors, background=background)

    draft_item = storage.save_asset(
        draft, kind="generated", name=f"人物像素化-{lv['short']}-AI原稿",
        meta={"pipeline": "char-pixel", "role": "draft", "level": lv["id"],
              "background": meta.get("background"),
              "instruction": meta.get("instruction"), "sourceAssetId": refAssetId or None},
        ai_label=False,
    )
    item = storage.save_asset(
        out, kind="charpixel", name=f"人物像素化-{lv['short']}",
        meta={**meta, "draftAssetId": draft_item["id"], "sourceAssetId": refAssetId or None},
    )
    return {"asset": item, "draft": draft_item,
            "meta": {**meta, "draftAssetId": draft_item["id"]}}


class CharRepixelPayload(BaseModel):
    """对 AI 原稿重新像素化：只改像素密度/色数/背景处理，不重新生成。

    背景可以任意切：原稿是「带背景的正常图」，绿幕/透明都是本地抠图合成的，
    不存在「换了背景就抠不干净」的问题。
    """

    draftAssetId: str
    grid: int = 120
    colors: int = 24
    dither: bool = False
    background: str = "keep"


@app.post("/api/pipeline/char-pixel/repixel")
async def pipeline_char_pixel_repixel(payload: CharRepixelPayload):
    from .pipelines import charpixel

    if not payload.draftAssetId:
        raise ValueError("需要 AI 原稿的资产 id（draftAssetId）")
    draft = _load_asset_image(payload.draftAssetId)
    out, meta = await run_in_threadpool(
        charpixel.repixel, draft, payload.grid, payload.colors,
        payload.dither, payload.background)
    item = storage.save_asset(
        out, kind="charpixel", name=f"人物像素化-{payload.grid}格",
        meta={**meta, "draftAssetId": payload.draftAssetId},
    )
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


# ---------- 三视图分步（流程画布用：一步一请求，单步可重跑） ----------
class ThreeViewStepPayload(BaseModel):
    """分步请求：view 决定这一步做什么，其余参数按需给。

    front   → prompt / style / 宽高 / 步数 / 种子
    side    → baseAssetId（正视图资产）+ 上面这些（编辑器不用宽高，但兜底要用）
    back    → 同上
    compose → viewAssetIds（正、侧、背顺序）+ 宽高（决定单格尺寸）
    """

    view: str = "front"
    prompt: str = ""
    style: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None
    width: int = 512
    height: int = 768
    maxSide: int = 768
    baseAssetId: Optional[str] = None
    viewAssetIds: Optional[List[str]] = None


def _load_asset_image(asset_id: str) -> Image.Image:
    item = storage.get_asset(asset_id)
    if not item:
        raise HTTPException(404, f"资产不存在：{asset_id}")
    path = storage.asset_path(item)
    if not path.exists():
        raise HTTPException(404, f"资产文件缺失：{asset_id}")
    return Image.open(path).convert("RGB")


@app.post("/api/pipeline/threeview/step")
async def pipeline_threeview_step(payload: ThreeViewStepPayload):
    """三视图分步生成：一次只做一步，任何一步都能单独重跑。

    为什么拆开：整套一次跑完要 5-9 分钟，中间任何一张不满意都得从头再来，
    而实际上「正面对了、侧面歪了」是最常见的情况。分步后前端按
    front → side → back → compose 串成流程，用户看完一张再决定下一步，
    不满意就换个种子重跑那一步，后面的步骤自动作废。

    每一步都落成资产，用 asset.id 串起下一步。
    """
    from .pipelines import threeview as tv_pipe

    view = (payload.view or "front").strip().lower()
    # 末步的规范名是 compose，但「拼板 / strip」是同一件事，别名直接兜住：
    # 前端画布内部节点键、早前版本的客户端、手写的 curl 都可能写成 strip，
    # 为此报「未知步骤」纯属自找麻烦。
    view = {"strip": "compose", "sheet": "compose", "composite": "compose"}.get(view, view)
    kw: Dict[str, Any] = dict(
        style=payload.style, steps=payload.steps, cfg=payload.cfg,
        seed=payload.seed, width=payload.width, height=payload.height,
    )

    if view == "compose":
        ids = [i for i in (payload.viewAssetIds or []) if i]
        if len(ids) != 3:
            raise ValueError(f"拼板需要按 正/侧/背 顺序给 3 张图，收到 {len(ids)} 张")
        imgs = [_load_asset_image(i) for i in ids]
        cw, ch = tv_pipe.cell(payload.width, payload.height)
        strip, cmeta = await run_in_threadpool(
            tv_pipe.compose_three_view, imgs, cell=(cw, ch))
        item = storage.save_asset(
            strip, kind="generated", name="threeview-三视图拼板",
            meta={"pipeline": "threeview", "view": "三视图拼板", "compose": cmeta},
            ai_label=False,
        )
        return {"images": [item], "step": "compose",
                "meta": {"pipeline": "threeview", "step": "compose",
                         "cell": [cw, ch], "compose": cmeta}}

    if view == "front":
        img, vmeta = await tv_pipe.run_front(payload.prompt, **kw)
    elif view in ("side", "back"):
        if not payload.baseAssetId:
            raise ValueError("生成侧/背视图需要先有正视图（baseAssetId）")
        base = _load_asset_image(payload.baseAssetId)
        img, vmeta = await tv_pipe.run_derived(
            view, base, payload.prompt, max_side=payload.maxSide, **kw)
    else:
        raise ValueError(
            f"未知步骤：{view}。合法值：front（正视图）/ side（侧视图）/ "
            f"back（背视图）/ compose（三视图拼板，别名 strip）"
        )

    item = storage.save_asset(
        img, kind="generated", name=f"threeview-{vmeta['name']}",
        meta={"pipeline": "threeview", "step": view, **vmeta},
        ai_label=False,
    )
    return {"images": [item], "step": view,
            "meta": {"pipeline": "threeview", "step": view, **vmeta}}


# ---------- 资产库 ----------
@app.get("/api/assets")
async def assets(kind: Optional[str] = None, pipeline: Optional[str] = None):
    """kind（是什么）与 pipeline（哪条管线产出的）两个维度，AND 关系。

    三视图的正/侧/背/拼板在 kind 上都属于 generated，只能靠 pipeline=threeview
    筛出来——只给 kind 一个维度的话，资产库里永远选不到三视图。
    """
    return {"assets": storage.list_assets(kind, pipeline)}


@app.get("/api/assets/facets")
async def asset_facets():
    """当前库里真实存在哪些 kind / pipeline 及各自数量，供筛选栏渲染。

    注意：必须排在 /api/assets/{asset_id} 类路由之前，否则 "facets" 会被
    当成资产 id 吃掉。
    """
    return storage.asset_facets()


@app.get("/api/assets/{asset_id}/thumb")
async def asset_thumb(asset_id: str, w: int = 160):
    """缩略图（等比缩到 w 宽，不裁切）。

    列图用原图会一次拉几百 MB，而且三视图拼板 1560×512 被当成方图裁切后
    只剩中间一团，完全看不出是拼板。这里保持宽高比，宽条自然就是宽条。
    资产 id 唯一且内容不可变，所以可以长期缓存。
    """
    item = storage.get_asset(asset_id)
    if not item:
        raise HTTPException(404, "资产不存在")
    try:
        # 缩略图首次生成要解码 + 重绘 + 编码 PNG，进线程池，别占事件循环
        p = await run_in_threadpool(storage.ensure_thumb, item, w)
    except FileNotFoundError:
        raise HTTPException(404, "文件缺失")
    return FileResponse(str(p), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/assets/{asset_id}/download")
async def download_asset(asset_id: str, label: Optional[int] = None):
    """单张下载。

    默认不补 AI 角标（存储阶段也没烧）；需要合规标识时显式传 ?label=1。
    AI 角标只在下载瞬间叠加，既保证合规可追溯，又不会污染后续抠图/合成。
    """
    if label is None:
        label = 1 if load_settings().get("aiLabel") else 0
    item = storage.get_asset(asset_id)
    if not item:
        raise HTTPException(404, "资产不存在")
    path = storage.asset_path(item)
    if not path.exists():
        raise HTTPException(404, "文件缺失")
    if not label:
        return FileResponse(str(path), media_type="image/png", filename=f"{item['name']}.png")

    def _stamp():
        img = Image.open(path).convert("RGBA")
        img = storage.stamp_ai_label(img)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        return buf

    buf = await run_in_threadpool(_stamp)
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

    def _zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            index = []
            for item in items:
                p = storage.asset_path(item)
                if not p.exists():      # 索引里有、磁盘上没了：跳过而不是整包 500
                    continue
                zf.write(p, arcname=item["file"])
                index.append({**{k: item[k] for k in ("id", "name", "file", "width", "height", "kind")},
                              # sha256 溯源（移植自 h3 的导出 manifest）：接收方
                              # 能校验文件完整性；老资产没有这个字段时为 null
                              "sha256": item.get("sha256")})
            zf.writestr("index.json", json.dumps(index, ensure_ascii=False, indent=2))
        buf.seek(0)
        return buf

    buf = await run_in_threadpool(_zip)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": 'attachment; filename="pixelforge-export.zip"'})


# ---------- 工作流 ----------
class RunPayload(BaseModel):
    input: str
    params: Optional[Dict[str, Any]] = None

    @field_validator("params")
    @classmethod
    def _bounded(cls, v):
        # 有界校验（移植自 FrameBaker shared/json.ts）：params 原样流进
        # run JSON 落盘，NaN / 无限 / 形状炸弹必须挡在入口
        if v is not None:
            validate_bounded(v)
        return v


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

# 人物像素化档位示例图：同样按需挂载
if CHAR_PIXEL_SAMPLES_DIR.is_dir():
    app.mount("/char-pixel-samples", StaticFiles(directory=str(CHAR_PIXEL_SAMPLES_DIR)),
              name="char-pixel-samples")

EXPORTS_DIR = DATA_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")
