"""工作流引擎：多段步骤、单步重跑、产物自动传递。

工作流 = 有序步骤序列；每步有明确输入（上一步产物 / 用户参数）与输出（资产 id）。
状态落盘在 data/runs/{run_id}.json，可暂停、可单步重跑。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import storage
from .config import DATA_DIR, load_settings
from .pipelines import resize as resize_pipe
from .pipelines import sheet as sheet_pipe
from .pipelines import sprite as sprite_pipe
from .providers import get_provider

RUNS_DIR = DATA_DIR / "runs"


# ---------------- 工作流注册 ----------------
WORKFLOWS: List[Dict[str, Any]] = [
    {
        "id": "animation",
        "name": "动画工作流",
        "tag": "ANIMATION",
        "desc": "三视图 → 动作帧 → 抽帧去重 → 插帧补帧 → 精灵表 + Godot 导出",
        "inputHint": "角色描述（如：灰白英短猫武士，红披风，武士刀）",
        "steps": [
            {
                "id": "three_view",
                "name": "三视图生成",
                "type": "generate",
                "desc": "生成正/侧/背三视图作为一致性基准",
                "prompt": "请生成一张角色三视图（正视图、侧视图、背视图横向排列）。角色设定：{input}。"
                          "二维日系Q版卡通风格，3头身，粗黑描边，平涂上色，暗绿色纯色背景。",
                "promptEn": "character turnaround reference sheet, exactly 3 views of the same character "
                            "(front, side, back) in a horizontal row, character: {input}, 2D chibi japanese "
                            "cartoon style, 3 heads tall, thick black outline, flat cel shading, "
                            "plain solid dark green background, game asset, no text",
                "n": 1,
            },
            {
                "id": "action_frames",
                "name": "动作帧生成",
                "type": "generate",
                "desc": "生成动作关键帧序列（真实模型下为视频，MVP 用关键帧模拟）",
                "prompt": "同一个角色（{input}），{action}动作，摄像机固定、角色原地不移动、始终全身，"
                          "纯绿色背景，每帧姿势明显不同。",
                "promptEn": "one single character only, exactly one figure in the frame, full body shot, "
                            "character: {input}, {action}, camera fixed, standing on the spot, "
                            "plain solid green background, 2D chibi japanese cartoon, thick black outline, "
                            "flat cel shading, game sprite, centered, no text, no turnaround sheet, "
                            "no multiple views, no duplicate characters",
                "n": 4,
            },
            {
                "id": "dedup",
                "name": "抽帧去重",
                "type": "dedup",
                "desc": "感知哈希去重，剔除近似帧，保证动作变化可见",
            },
            {
                "id": "inbetween",
                "name": "插帧补帧",
                "type": "blend",
                "desc": "相邻帧之间补一帧（可选）。纯像素素材建议保持 0，靠真实抽帧/视频模型补帧",
                "perGap": 0,
            },
            {
                "id": "normalize",
                "name": "抠图与规整",
                "type": "sprites",
                "desc": "连通域抠图 → 裁空白 → 最近邻缩放 → 底部居中对齐，消除逐帧漂移",
                "size": 128,
                "anchor": "bottom",
                "matte": True,
            },
            {
                "id": "sprite_sheet",
                "name": "精灵表导出",
                "type": "compose",
                "desc": "按列合成精灵表并生成引擎索引",
                "cols": 4,
            },
            {
                "id": "godot",
                "name": "Godot 工程导出",
                "type": "godot",
                "desc": "生成 SpriteFrames 资源 + AnimatedSprite2D 场景 + 导入说明",
            },
        ],
    },
    {
        "id": "parallax",
        "name": "视差图层工作流",
        "tag": "PARALLAX",
        "desc": "参考图 → 分层提取 → 同尺寸规整 → scroll_scale 参数表 + 打包",
        "inputHint": "场景描述（如：日式幻想山谷，城堡与远山）",
        "steps": [
            {
                "id": "ref",
                "name": "场景参考图",
                "type": "generate",
                "desc": "生成横向卷轴场景基准图",
                "prompt": "{input}，横向卷轴游戏场景，层次丰富，日式幻想美术，16:9 构图。",
                "n": 1,
            },
            {
                "id": "layers",
                "name": "图层提取",
                "type": "generate",
                "desc": "按层次提取 4 个图层（MVP 用独立生成模拟）",
                "prompt": "只保留{input}中的{layer}，其余元素移除，统一暗紫色纯色背景 #2A153F，同尺寸同坐标。",
                "promptEn": "parallax layer only: {layer} of {input}, remove all other elements, "
                            "plain flat dark purple background #2A153F, same canvas size and alignment, "
                            "2D game background art, flat colors, no text",
                "n": 4,
                "layerNames": ["远景天空与雪山", "中景丘陵与城堡", "近景地面与道路", "前景植物遮挡层"],
                "layerNamesEn": ["far background sky and snowy mountains",
                                  "mid ground hills and castle",
                                  "near ground terrain and road",
                                  "foreground plants and foliage"],
            },
            {
                "id": "normalize",
                "name": "同尺寸规整",
                "type": "normalize",
                "desc": "全部图层统一到基准尺寸，保证引擎内可重合",
            },
            {
                "id": "seamless",
                "name": "无缝延展",
                "type": "seamless",
                "desc": "每层横向镜像延展 ×2 并交叉淡化，消除循环接缝",
                "times": 2,
                "blend": 24,
            },
            {
                "id": "godot",
                "name": "视差工程导出",
                "type": "godot_parallax",
                "desc": "生成 scroll_scale 参数表与 Parallax2D 场景骨架",
            },
        ],
    },
    {
        "id": "scene",
        "name": "场景元素工作流",
        "tag": "SCENE",
        "desc": "参考图 → 元素批量提取 → 拼装 → 打包",
        "inputHint": "场景描述（如：日本现代城市街道）",
        "steps": [
            {
                "id": "ref",
                "name": "场景参考图",
                "type": "generate",
                "desc": "生成 45 度俯视实机场景参考图",
                "prompt": "{input}，正前方45度俯视视角，Q版日式RPG构图，实机场景而非概念图。",
                "promptEn": "{input}, 45 degree top-down view from front, chibi japanese RPG composition, "
                            "in-game scene not concept art, flat colors, game asset, no text",
                "n": 1,
            },
            {
                "id": "elements",
                "name": "元素批量提取",
                "type": "generate",
                "desc": "串行提取 4 个元素（真实模型下可并行多 agent）",
                "prompt": "只提取{input}中的{element}，完整补全遮挡，正视图不倾斜，统一暗紫底 #2A153F，同尺寸同坐标。",
                "promptEn": "isolated game asset: only {element} from {input}, fully complete occluded parts, "
                            "front view upright, plain flat dark purple background #2A153F, "
                            "same canvas size and alignment, 2D chibi japanese RPG style, flat colors, no text",
                "n": 4,
                "layerNames": ["建筑主体", "树木植被", "地面平面", "道具摆件"],
                "layerNamesEn": ["main building", "trees and vegetation",
                                  "ground plane", "props and furniture"],
            },
            {
                "id": "montage",
                "name": "拼装预览",
                "type": "montage",
                "desc": "把提取元素按网格拼成总览图",
                "cols": 2,
            },
        ],
    },
]


def list_workflows() -> List[Dict[str, Any]]:
    return WORKFLOWS


def get_workflow(wf_id: str) -> Dict[str, Any]:
    for wf in WORKFLOWS:
        if wf["id"] == wf_id:
            return wf
    raise KeyError(f"未知工作流: {wf_id}")


# ---------------- Run 持久化 ----------------
def _run_path(run_id: str):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / f"{run_id}.json"


def create_run(wf_id: str, input_text: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    wf = get_workflow(wf_id)
    run = {
        "id": uuid.uuid4().hex[:12],
        "workflowId": wf_id,
        "workflowName": wf["name"],
        "input": input_text,
        "params": params or {},
        "createdAt": int(time.time()),
        "steps": [
            {"id": s["id"], "name": s["name"], "status": "pending", "artifacts": [], "meta": {}}
            for s in wf["steps"]
        ],
    }
    _save_run(run)
    return run


def _save_run(run: Dict[str, Any]) -> None:
    _run_path(run["id"]).write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    p = _run_path(run_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RUNS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files[:limit]:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


# ---------------- 工具函数 ----------------
def _content_hash(img: Image.Image, size: int = 16) -> np.ndarray:
    """内容感知哈希：先按背景裁出主体包围盒，再 16×16 差值哈希。

    直接对整图做 8×8 哈希时，纯色背景会稀释掉角色姿态差异，
    导致不同姿势被判为重复帧。
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    bg = arr[0, 0, :3].astype(np.int16)
    diff = np.abs(arr[:, :, :3].astype(np.int16) - bg).sum(axis=2)
    mask = diff > 24
    if mask.any():
        ys, xs = np.where(mask)
        pad = 4
        box = (
            max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(rgba.width, int(xs.max()) + pad), min(rgba.height, int(ys.max()) + pad),
        )
        crop = rgba.crop(box)
    else:
        crop = rgba
    g = crop.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _load(asset_id: str) -> Image.Image:
    item = storage.get_asset(asset_id)
    if not item:
        raise RuntimeError(f"资产不存在: {asset_id}")
    return Image.open(storage.asset_path(item)).convert("RGBA")


def _prev_artifacts(run: Dict[str, Any], step_index: int) -> List[str]:
    for i in range(step_index - 1, -1, -1):
        arts = run["steps"][i].get("artifacts") or []
        if arts:
            return arts
    return []


def _save_asset(img: Image.Image, **kwargs) -> Dict[str, Any]:
    """工作流中间产物不烧 AI 角标。

    角标一旦画进像素就会被合成进精灵表 / Godot 贴图，整包素材作废。
    合规标识放到「导出」环节（Godot 包内 AI_GENERATED.txt + 导出角标）。
    """
    kwargs.setdefault("ai_label", False)
    return storage.save_asset(img, **kwargs)


def _template(step: Dict[str, Any], english: bool) -> str:
    """按 Provider 语言能力挑选提示词模板（英文模型用 promptEn）。"""
    if english and step.get("promptEn"):
        return step["promptEn"]
    return step["prompt"]


def _fill(template: str, run: Dict[str, Any], step: Dict[str, Any], idx: int,
          english: bool = False) -> str:
    cn = step.get("layerNames") or step.get("elementNames") or ["图层一", "图层二", "图层三", "图层四"]
    en = step.get("layerNamesEn") or step.get("elementNamesEn") or cn
    layers = en if english else cn
    subject = run.get("input", "")
    if english and run.get("params", {}).get("descEn"):
        subject = run["params"]["descEn"]
    action = run.get("params", {}).get("action") or "挥砍"
    action_en = run.get("params", {}).get("actionEn") or {
        "挥砍": "slashing with a sword",
        "奔跑": "running",
        "待机": "idle breathing",
        "行走": "walking",
        "攻击": "attacking",
        "跳跃": "jumping",
    }.get(action, "slashing with a sword")
    return (
        template.replace("{input}", subject)
        .replace("{action}", action_en if english else action)
        .replace("{actionEn}", action_en)
        .replace("{layer}", layers[idx % len(layers)])
        .replace("{element}", layers[idx % len(layers)])
    )


# ---------------- 步骤执行 ----------------
async def run_step(run_id: str, step_id: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise KeyError("run 不存在")
    wf = get_workflow(run["workflowId"])
    step_defs = wf["steps"]
    index = next((i for i, s in enumerate(step_defs) if s["id"] == step_id), None)
    if index is None:
        raise KeyError("步骤不存在")

    step_def = step_defs[index]
    step_state = run["steps"][index]
    overrides = overrides or {}

    # 依赖检查：前面的 generate/compose 步骤必须完成
    if index > 0:
        prev_status = run["steps"][index - 1]["status"]
        if prev_status != "done" and step_def["type"] not in ("generate",):
            raise RuntimeError("请先完成上一步")

    settings = load_settings()
    pid = settings.get("activeProvider") or "mock"
    provider = get_provider(pid, settings["providers"].get(pid, {}))

    step_state.update({"status": "running", "error": "", "startedAt": int(time.time())})
    _save_run(run)

    try:
        stype = step_def["type"]
        artifacts: List[str] = []
        meta: Dict[str, Any] = {}

        if stype == "generate":
            layers = step_def.get("layerNames")
            n = int(overrides.get("n") or step_def.get("n", 1))
            pw = int(run.get("params", {}).get("width") or 512)
            ph = int(run.get("params", {}).get("height") or 512)
            english = not getattr(provider, "supports_zh", False)
            tpl = _template(step_def, english)
            prompt = overrides.get("prompt") or _fill(tpl, run, step_def, 0, english)
            psteps = run.get("params", {}).get("steps")
            pcfg = run.get("params", {}).get("cfg")
            gen_kw = {}
            if psteps:
                gen_kw["steps"] = int(psteps)
            if pcfg is not None:
                gen_kw["cfg"] = float(pcfg)
            seed = run.get("params", {}).get("seed")

            if layers or n == 1:
                # 分层/元素提取：每项独立提示词，受限并发并行生成
                total = n if layers else 1
                conc = int(step_def.get("concurrency", 3))
                sem = asyncio.Semaphore(max(1, conc))

                async def _one(i: int):
                    async with sem:
                        p = overrides.get("prompt") or _fill(tpl, run, step_def, i, english)
                        res = await provider.generate(p, pw, ph, n=1, **gen_kw)
                        label = layers[i] if layers else step_def["name"]
                        ids = []
                        for img in res.images:
                            ids.append(_save_asset(
                                img, kind="workflow", name=f"{run['id']}-{step_id}-{i:02d}",
                                meta={"runId": run["id"], "step": step_id, "index": i, "label": label},
                            )["id"])
                        return i, ids, label

                results = await asyncio.gather(*[_one(i) for i in range(total)])
                results.sort(key=lambda r: r[0])
                for _, ids, label in results:
                    artifacts.extend(ids)
                    if layers:
                        meta.setdefault("labels", []).append(label)
                meta["prompt"] = prompt
                meta["concurrency"] = conc
                meta["promptLang"] = "en" if english else "zh"
            else:
                # 动作关键帧：一次请求 n 张，保证同角色不同姿态
                result = await provider.generate(prompt, pw, ph, n=n, seed=seed, **gen_kw)
                for i, img in enumerate(result.images):
                    artifacts.append(_save_asset(
                        img, kind="workflow", name=f"{run['id']}-{step_id}-{i:02d}",
                        meta={"runId": run["id"], "step": step_id, "index": i, "label": f"frame {i}"},
                    )["id"])
                meta = {"prompt": prompt, "frames": len(artifacts),
                        "provider": result.meta.get("provider"),
                        "promptLang": "en" if english else "zh"}

        elif stype == "dedup":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            threshold = int(step_def.get("threshold", 10))
            kept: List[str] = []
            hashes: List[np.ndarray] = []
            dropped = 0
            for aid in src:
                h = _content_hash(_load(aid))
                if any(_hamming(h, prev) <= threshold for prev in hashes):
                    dropped += 1
                    continue
                hashes.append(h)
                kept.append(aid)
            artifacts = kept
            meta = {"kept": len(kept), "dropped": dropped,
                    "method": "content-dhash-16x16", "threshold": threshold}

        elif stype == "blend":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            per_gap = int(step_def.get("perGap", 1))
            frames = [_load(a) for a in src]
            artifacts = []
            for i, f in enumerate(frames):
                artifacts.append(_save_asset(f, kind="workflow", name=f"{run['id']}-f{i:02d}",
                                                    meta={"runId": run["id"], "step": step_id, "index": i})["id"])
                if i < len(frames) - 1:
                    nxt = frames[i + 1]
                    if f.size != nxt.size:
                        nxt = nxt.resize(f.size, Image.NEAREST)
                    for g in range(per_gap):
                        alpha = (g + 1) / (per_gap + 1)
                        mid = Image.blend(f, nxt, alpha)
                        artifacts.append(_save_asset(
                            mid, kind="workflow", name=f"{run['id']}-f{i:02d}m{g}",
                            meta={"runId": run["id"], "step": step_id, "index": i, "inbetween": True},
                        )["id"])
            meta = {"generated": len(artifacts) - len(frames), "perGap": per_gap,
                    "note": ("未补帧：像素素材建议保持原生帧数（2-4 帧），补帧请用视频模型抽帧"
                             if per_gap == 0 else
                             "已用相邻帧混合模拟中间帧；接入视频/插帧模型后替换为真实生成")}

        elif stype == "compose":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            frames = [_load(a) for a in src]
            cols = int(step_def.get("cols", 4))
            composed, index_json = sheet_pipe.compose_sheet(frames, cols)
            item = _save_asset(composed, kind="sheet", name=f"{run['id']}-sheet",
                                      meta={"runId": run["id"], "step": step_id, "index": index_json})
            artifacts = [item["id"]]
            meta = {"cols": cols, "frames": len(frames), "frameSize": index_json["frame_size"]}

        elif stype == "seamless":
            from .pipelines import seamless as seamless_pipe

            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            times = int(step_def.get("times", 2))
            blend = int(step_def.get("blend", 24))
            size = None
            for i, aid in enumerate(src):
                img = _load(aid)
                out, m = seamless_pipe.extend_horizontal(img, times=times, blend=blend)
                size = m["size"]
                artifacts.append(_save_asset(
                    out, kind="seamless", name=f"{run['id']}-seamless-{i}",
                    meta={"runId": run["id"], "step": step_id, "index": i, **m},
                )["id"])
            meta = {"layers": len(artifacts), "times": times, "blend": blend, "size": size}

        elif stype == "sprites":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            if not src:
                raise RuntimeError("没有可规整的帧")
            imgs = [_load(a) for a in src]
            outs, smeta = sprite_pipe.normalize_frames(
                imgs,
                size=int(overrides.get("size") or step_def.get("size", 128)),
                anchor=str(overrides.get("anchor") or step_def.get("anchor", "bottom")),
                matte=bool(step_def.get("matte", True)),
                tolerance=int(step_def.get("tolerance", 80)),
                feather=int(step_def.get("feather", 4)),
            )
            for i, out in enumerate(outs):
                artifacts.append(_save_asset(
                    out, kind="workflow", name=f"{run['id']}-sprite{i:02d}",
                    meta={"runId": run["id"], "step": step_id, "index": i},
                )["id"])
            meta = smeta

        elif stype == "normalize":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            imgs = [_load(a) for a in src]
            if not imgs:
                raise RuntimeError("没有可规整的图层")
            w, h = imgs[0].size
            artifacts = []
            for i, img in enumerate(imgs):
                out = img if img.size == (w, h) else resize_pipe.fit_pad(img, (w, h))
                artifacts.append(_save_asset(out, kind="workflow", name=f"{run['id']}-layer{i}",
                                                    meta={"runId": run["id"], "step": step_id, "index": i})["id"])
            meta = {"size": [w, h], "layers": len(artifacts)}

        elif stype == "montage":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            frames = [_load(a) for a in src]
            cols = int(step_def.get("cols", 2))
            composed, index_json = sheet_pipe.compose_sheet(frames, cols)
            item = _save_asset(composed, kind="montage", name=f"{run['id']}-montage",
                                      meta={"runId": run["id"], "step": step_id})
            artifacts = [item["id"]]
            meta = {"cols": cols, "elements": len(frames)}

        elif stype in ("godot", "godot_parallax"):
            from .godot_export import build_godot_package, build_parallax_package
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            if not src:
                raise RuntimeError("没有可导出的产物")
            if stype == "godot":
                info = build_godot_package(src, run)
            else:
                info = build_parallax_package(src, run)
            artifacts = []
            meta = info

        else:
            raise RuntimeError(f"未知步骤类型: {stype}")

        step_state.update({
            "status": "done",
            "artifacts": artifacts,
            "meta": meta,
            "finishedAt": int(time.time()),
        })
        _save_run(run)
        return run

    except Exception as e:
        step_state.update({"status": "failed", "error": str(e), "finishedAt": int(time.time())})
        _save_run(run)
        raise


async def run_all(run_id: str) -> Dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise KeyError("run 不存在")
    wf = get_workflow(run["workflowId"])
    for i, step_def in enumerate(wf["steps"]):
        if run["steps"][i]["status"] == "done":
            continue
        run = await run_step(run_id, step_def["id"])
    return run
