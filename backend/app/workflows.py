"""工作流引擎：多段步骤、单步重跑、产物自动传递。

工作流 = 有序步骤序列；每步有明确输入（上一步产物 / 用户参数）与输出（资产 id）。
状态落盘在 data/runs/{run_id}.json，可暂停、可单步重跑。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import storage
from . import progress as progress_mod
from .config import DATA_DIR, load_settings, resolve_path
from .pipelines import matte as matte_pipe
from .pipelines import resize as resize_pipe
from .pipelines import sheet as sheet_pipe
from .pipelines import sprite as sprite_pipe
from .providers import get_provider
from .providers.diffusers_local import DiffusersProvider, detect_family

RUNS_DIR = DATA_DIR / "runs"

# ---------- 四方向行走（h3 的 recipe 思路：方向是角色的一等维度） ----------
FOUR_DIRS = ("down", "up", "left", "right")
DIR_DESC = {
    "down": "面向观众（正面视角）",
    "up": "背对观众（背面视角）",
    "left": "侧身朝左（左侧视角）",
    "right": "侧身朝右（右侧视角）",
}
# 行走循环的姿势序列：抬脚 → 并拢 → 换脚 → 并拢。凑不成 4 帧就按前者截断，
# 别用「均匀抽帧」——那样两帧同姿势，循环会卡顿。
WALK_POSES = ("左脚向前迈出、右臂前摆", "双脚并拢、身体微沉",
              "右脚向前迈出、左臂前摆", "双脚并拢、身体回正")


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
                "desc": "正/侧/背三张独立生成（帧 2+ 编辑底座接力保同一角色）+ 自动拼板",
                # 注意：这里只是「角色描述 + 画风描述」，不能自带「角色设定：」
                # 前缀——threeview 管线会补。工作流是「一句话跑完全程」的固定
                # 流程，画风钉在这里；Workbench 的单步生成才走画风选择器。
                # 单角色/视角约束同理由管线统一加（单图画齐三视角会重叠糊团）。
                "prompt": "{input}。二维日系Q版卡通风格，3头身，粗黑描边，平涂上色。",
                "n": 1,
                "threeView": True,
                "width": 512,
                "height": 768,
                "maxSide": 768,
            },
            {
                "id": "action_frames",
                "name": "动作帧生成",
                "type": "generate",
                "desc": "生成动作关键帧序列（真实模型下为视频，MVP 用关键帧模拟）",
                "prompt": "同一个角色（{input}），{action}动作，摄像机固定、角色原地不移动、始终全身，"
                          "纯绿色背景，每帧姿势明显不同。",
                "n": 4,
                # 一致性策略：第 1 帧文生图，其余帧拿第 1 帧做 img2img 参考
                # （三视图排版不可控，直接裁切容易裁到两个视图，改用自锚定）
                "refChain": True,
                # 双底座接力（当前模型是 qwen_edit 编辑器时生效）：
                # 帧 1 交给 txt2img 底座生成，帧 2+ 用编辑指令改姿势。
                # 编辑器没有自由创作能力，必须借生成底座出第一帧。
                "txt2imgModel": "models/z-image-turbo",
                "editPrompt": "同一个角色，保持角色外观、服装、配色完全不变，"
                              "将姿势改为：{action}，角色原地全身，纯色背景",
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
                "desc": "相邻帧之间补帧（可选）。默认光流插帧：按运动方向 warp 再融合，"
                        "比交叉淡化少一半重影；mode=blend 可退回淡化",
                "perGap": 0,
                "mode": "flow",
            },
            {
                "id": "normalize",
                "name": "抠图与规整",
                "type": "sprites",
                "desc": "连通域抠图 → 裁空白 → 最近邻缩放 → 底部居中对齐，消除逐帧漂移",
                "size": 128,
                "anchor": "bottom",
                "matte": True,
                "tolerance": 72,
                # 像素素材必须硬边：feather>0 会在浅色底显出白晕，还会啃掉细肢
                "feather": 0,
                "hard": True,
                # AI 出图常自带画框，向内切一圈再抠（切太多会切掉脚）
                "inset": 0.04,
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
        "id": "animation_video",
        "name": "动画工作流 · 视频生成",
        "tag": "I2V",
        "desc": "角色图 → 视频底座生成整段动作 → 抽帧去重 → 抠图规整 → 精灵表 + Godot 导出",
        "inputHint": "角色描述（如：灰白英短猫武士，红披风，武士刀）",
        "steps": [
            {
                "id": "anchor",
                "name": "角色锚点图",
                "type": "generate",
                "desc": "先出一张干净的正面全身图，作为视频底座的第一帧参考",
                "prompt": "{input}。游戏角色正面立绘，全身，双脚踩在同一水平线上，"
                          "纯绿色背景 #00FF00，无渐变、无纹理、无文字、无水印、无边框。",
                "n": 1,
                "refChain": True,
                "width": 480,
                "height": 832,
                "maxSide": 832,
            },
            {
                "id": "video",
                "name": "视频生成动作",
                "type": "video",
                "desc": "Wan2.2 TI2V-5B：锚点图 + 动作提示词 → 连续帧序列（肢体路径连贯，"
                        "不会像逐帧编辑那样抖）",
                "prompt": "画面中的角色在原地行走，双脚交替蹬地，手臂与腿反向摆动，"
                          "发饰和衣摆随步伐轻微晃动；角色位置、大小、朝向始终不变，"
                          "没有位移、没有镜头运动、没有起停；背景保持纯绿色。",
                "useRef": "anchor",
                "numFrames": 81,
                "fps": 24,
                "steps": 30,
                "width": 480,
                "height": 832,
                "maxSide": 832,
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
                "desc": "视频帧本身连续，默认不补；需要更慢的动作再开（光流插帧）",
                "perGap": 0,
                "mode": "flow",
            },
            {
                "id": "normalize",
                "name": "抠图与规整",
                "type": "sprites",
                "desc": "连通域抠图 → 裁空白 → 最近邻缩放 → 底部居中对齐，消除逐帧漂移",
                "size": 128,
                "anchor": "bottom",
                "matte": True,
                "tolerance": 72,
                "feather": 0,
                "hard": True,
                "inset": 0.04,
            },
            {
                "id": "sprite_sheet",
                "name": "精灵表导出",
                "type": "compose",
                "desc": "全局锁色后合成精灵表并生成引擎索引",
                "cols": 4,
                "colors": 24,
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
        "id": "walk_4dir",
        "name": "四方向行走工作流",
        "tag": "FOUR-DIR",
        "desc": "四方向立绘 → 每方向行走循环帧 → 横排精灵表 → Godot 四向工程",
        "inputHint": "角色描述（如：灰白英短猫武士，红披风，武士刀）",
        "steps": [
            {
                "id": "bases",
                "name": "四方向立绘",
                "type": "fourdir_bases",
                "desc": "下/上/左/右各一张全身立绘（同一角色描述与画风，{dir_desc} 由管线替换）",
                "prompt": "{input}。游戏角色{dir_desc}，全身，双脚踩在同一水平线上，"
                          "身体水平居中，纯色背景，无文字、无水印、无边框。",
                "width": 512,
                "height": 768,
                "maxSide": 768,
            },
            {
                "id": "walk",
                "name": "四方向行走帧",
                "type": "fourdir_walk",
                "desc": "每方向 4 帧行走循环（编辑底座接力保外观），自动拼横排精灵表 + 审查图",
                "frames": 4,
                "size": 128,
            },
            {
                "id": "godot",
                "name": "Godot 四向工程",
                "type": "godot_fourdir",
                "desc": "四方向 SpriteFrames + 场景 + 方向控制器脚本 + SHA256SUMS 清单",
            },
        ],
    },
    {
        "id": "parallax",
        "name": "视差图层工作流",
        "tag": "PARALLAX",
        "desc": "参考图 → 语义分层 → 同尺寸规整 → scroll_scale 参数表 + 打包",
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
                "name": "语义分层",
                "type": "scene_layers",
                "desc": "把基准图拆成 4 张同坐标 RGBA 图层（拆同一张图，不是各画一遍）",
                "layers": 4,
                "steps": 50,
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
                "n": 1,
            },
            {
                "id": "elements",
                "name": "元素批量提取",
                "type": "generate",
                "desc": "串行提取 4 个元素（真实模型下可并行多 agent）",
                "prompt": "只提取{input}中的{element}，完整补全遮挡，正视图不倾斜，统一暗紫底 #2A153F，同尺寸同坐标。",
                "n": 4,
                "layerNames": ["建筑主体", "树木植被", "地面平面", "道具摆件"],
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
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


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
    # 原子替换：run.json 是「跑了几分钟」的唯一进度记录，直接覆盖时
    # 中断会留下半截 JSON，后面 get_run 解析失败，整个 run 等于废掉。
    # allow_nan=False：指标里的 NaN 写坏不如写炸（同 storage._save_index）。
    path = _run_path(run["id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    # run_id 直接拼进文件名，必须限死为 12 位 hex；否则 Windows 下
    # "..\..\data\assets\index" 这类反斜杠路径能穿出 runs 目录。
    if not _RUN_ID_RE.match(run_id or ""):
        return None
    p = _run_path(run_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def recover_interrupted_runs() -> int:
    """启动时把上次进程遗留的 running 步骤标记为失败（移植自 FrameBaker
    queue.ts 的启动恢复：任务负载随进程消亡，queued/running 不可能再继续）。

    锁和线程池都随进程消失，唯一残留是 run JSON 里的 ``status=running``——
    不清理的话，重启后这些步骤在新进程里永远显示「执行中」，用户点重跑
    还会被并发检查挡住。这里一次性扫掉，detail 写明可重新执行。
    _run_step_locked 里的 10 分钟超时重置保留为兜底：那种是进程还活着、
    但某一步真卡死的情况。
    """
    recovered = 0
    for run in list_runs(limit=10_000):
        changed = False
        for s in run.get("steps", []):
            if s.get("status") == "running":
                s["status"] = "failed"
                s["error"] = "服务重启，步骤中断（可重新执行）"
                s["finishedAt"] = int(time.time())
                changed = True
        if changed:
            _save_run(run)
            recovered += 1
    return recovered


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
    标识只在「导出/下载」环节按设置开关叠加（默认关）。
    """
    kwargs.setdefault("ai_label", False)
    return storage.save_asset(img, **kwargs)


def _step_artifacts(run: Dict[str, Any], step_id: str) -> List[str]:
    for s in run.get("steps", []):
        if s["id"] == step_id:
            return list(s.get("artifacts") or [])
    return []


def _ref_from_sheet(img: Image.Image, view: str = "left",
                    size: Tuple[int, int] = (384, 384)) -> Image.Image:
    """从三视图里裁出单个视图，等比贴到方形画布上（底色调自原图），给 img2img 当参考。

    直接拉伸会让角色比例失真，所以必须「等比 + 留边」，
    留边颜色取原图左上角，避免黑边被模型当成画面内容。
    """
    w, h = img.size
    if view == "middle":
        part = img.crop((w // 3, 0, w * 2 // 3, h))
    elif view == "right":
        part = img.crop((w * 2 // 3, 0, w, h))
    else:
        part = img.crop((0, 0, w // 3, h))

    rgb = part.convert("RGB")
    bg = rgb.getpixel((2, 2))
    scale = min(size[0] / part.width, size[1] / part.height)
    nw = max(1, int(part.width * scale))
    nh = max(1, int(part.height * scale))
    canvas = Image.new("RGB", size, bg)
    canvas.paste(rgb.resize((nw, nh), Image.LANCZOS), ((size[0] - nw) // 2, (size[1] - nh) // 2))
    return canvas


def _inbetween(a: Image.Image, b: Image.Image, alpha: float, mode: str):
    """补一帧：默认光流（按运动方向 warp 再融合），失败/关闭时退回交叉淡化。

    交叉淡化的中间帧是「重影」——手臂同时出现在两个位置；光流先把两帧
    warp 到同一时刻再融合，肢体只有一条。移植自 FrameRonin 的
    opticalFlowInterpolate（双向 Farneback + 位移限幅 + 最近邻 remap）。
    """
    if mode == "flow":
        try:
            from .pipelines import inbetween as inbetween_pipe

            return inbetween_pipe.flow_mid(a, b, alpha, nearest=True)
        except Exception as e:  # 没装 cv2 或光流失败：退回淡化，不让整步崩
            meta = {"method": "blend", "reason": f"{type(e).__name__}: {e}"}
            return Image.blend(a, b, alpha), meta
    return Image.blend(a, b, alpha), {"method": "blend", "mode": mode}


def _fill(template: str, run: Dict[str, Any], step: Dict[str, Any], idx: int) -> str:
    """把工作流模板里的占位符替换成中文实参（两个底座都是中文原生）。"""
    layers = step.get("layerNames") or step.get("elementNames") or ["图层一", "图层二", "图层三", "图层四"]
    subject = run.get("input", "")
    action = run.get("params", {}).get("action") or "挥砍"
    return (
        template.replace("{input}", subject)
        .replace("{action}", action)
        .replace("{layer}", layers[idx % len(layers)])
        .replace("{element}", layers[idx % len(layers)])
    )


# ---------------- Run 并发保护 ----------------
# run 状态是「读-改-写」落盘：两个请求同时操作同一个 run（两个标签页各点
# 一次「执行此步」、或 run-all 与单步并行）会互相覆盖 artifacts，后写的赢。
# 每个 run 一把进程内锁，同一时刻只放一个步骤执行；已在执行中的步骤直接
# 拒绝，避免同一步被跑两遍。
# 进程崩溃会留下 status=running 的残留记录，锁已经随进程消亡，此时按
# startedAt 超时自动重置：单步最慢的是视频生成（约 90s）与 40 步编辑
# （约 4 分钟），10 分钟足够宽裕——真卡死时允许重跑比永远拒绝重跑重要。
_RUN_LOCKS: Dict[str, asyncio.Lock] = {}
_STALE_RUNNING_SECONDS = 600


def _run_lock(run_id: str) -> asyncio.Lock:
    # asyncio 单线程：get-or-create 之间没有 await，不需要额外加锁
    lock = _RUN_LOCKS.get(run_id)
    if lock is None:
        lock = asyncio.Lock()
        _RUN_LOCKS[run_id] = lock
    return lock


def _reset_stale_running(step_state: Dict[str, Any]) -> bool:
    """把上次进程崩溃遗留的 running 重置为 failed；返回是否重置过。"""
    started = int(step_state.get("startedAt") or 0)
    if int(time.time()) - started < _STALE_RUNNING_SECONDS:
        return False
    step_state.update({
        "status": "failed",
        "error": "服务中断导致步骤未完成（已自动重置，可重新执行）",
        "finishedAt": int(time.time()),
    })
    return True


# ---------------- 步骤执行 ----------------
async def run_step(run_id: str, step_id: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """执行单步（持 run 锁，理由见 _run_lock 说明）。"""
    async with _run_lock(run_id):
        return await _run_step_locked(run_id, step_id, overrides)


async def _run_step_locked(run_id: str, step_id: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

    # 并发保护：已在执行中的步骤直接拒绝（崩溃残留的 running 按超时重置）
    if step_state.get("status") == "running":
        if not _reset_stale_running(step_state):
            raise RuntimeError(f"步骤「{step_state['name']}」正在执行中，请勿重复触发")
        _save_run(run)

    # 依赖检查：前面的 generate/compose 步骤必须完成
    if index > 0:
        prev_status = run["steps"][index - 1]["status"]
        if prev_status != "done" and step_def["type"] not in ("generate",):
            raise RuntimeError("请先完成上一步")

    settings = load_settings()
    pid = settings.get("activeProvider") or "diffusers"
    provider = get_provider(pid, settings["providers"].get(pid, {}))

    step_state.update({"status": "running", "error": "", "startedAt": int(time.time())})
    _save_run(run)

    # 进度总线：前端订阅 /api/tasks/{id}/stream 看「第几步、哪一帧」，
    # 也可 POST /api/tasks/{id}/cancel 中止（provider 步进回调抛取消）
    task_id = progress_mod.register((overrides or {}).get("taskId"))
    progress_mod.report(task_id, 0, 1, f"步骤「{step_state['name']}」开始")

    try:
        stype = step_def["type"]
        artifacts: List[str] = []
        meta: Dict[str, Any] = {}

        if stype == "generate":
            layers = step_def.get("layerNames")
            n = int(overrides.get("n") or step_def.get("n", 1))
            pw = int(run.get("params", {}).get("width") or 512)
            ph = int(run.get("params", {}).get("height") or 512)
            tpl = step_def["prompt"]
            prompt = overrides.get("prompt") or _fill(tpl, run, step_def, 0)
            psteps = run.get("params", {}).get("steps")
            pcfg = run.get("params", {}).get("cfg")
            gen_kw = {}
            if psteps:
                gen_kw["steps"] = int(psteps)
            if pcfg is not None:
                gen_kw["cfg"] = float(pcfg)
            seed = run.get("params", {}).get("seed")

            # 参考图：给编辑器当输入（Z-Image 没有 img2img，ref 会被忽略）
            ref_img = None
            ref_step = step_def.get("useRef")
            if ref_step:
                ref_ids = _step_artifacts(run, ref_step)
                if ref_ids:
                    ref_img = _ref_from_sheet(
                        _load(ref_ids[-1]),
                        view=str(step_def.get("refCrop", "left")),
                        size=(pw // 16 * 16, ph // 16 * 16),
                    )
                    gen_kw["ref"] = ref_img

            if step_def.get("threeView"):
                # 三视图管线：三张独立生成 + 拼板（见 pipelines/threeview.py）。
                # prompt 此时是「角色设定+画风」文本，视角/单角色约束由管线加。
                from .pipelines import threeview as tv_pipe

                strip, views, tmeta = await tv_pipe.run_three_view(
                    prompt,
                    steps=psteps,
                    cfg=pcfg,
                    seed=seed,
                    width=int(step_def.get("width") or pw),
                    height=int(step_def.get("height") or ph),
                    max_side=int(step_def.get("maxSide") or 768),
                    task_id=task_id,
                )
                names = ["三视图拼板", "正视图", "侧视图", "背视图"]
                for i, img in enumerate([strip, *views]):
                    artifacts.append(_save_asset(
                        img, kind="workflow", name=f"{run['id']}-{step_id}-{i:02d}",
                        meta={"runId": run["id"], "step": step_id, "index": i,
                              "label": names[i]},
                    )["id"])
                meta = {"prompt": prompt, "views": 3, "mode": tmeta.get("mode"),
                        "cell": tmeta.get("cell"), "compose": tmeta.get("compose"),
                        "perView": tmeta.get("views"), "editModel": tmeta.get("editModel"),
                        "txt2imgModel": tmeta.get("txt2imgModel"),
                        "promptLang": "zh",
                        "seconds": tmeta.get("seconds")}
                if tmeta.get("editFallback"):
                    meta["editFallback"] = tmeta["editFallback"]
            elif layers or n == 1:
                # 分层/元素提取：每项独立提示词，受限并发并行生成
                total = n if layers else 1
                conc = int(step_def.get("concurrency", 3))
                sem = asyncio.Semaphore(max(1, conc))

                async def _one(i: int):
                    async with sem:
                        p = overrides.get("prompt") or _fill(tpl, run, step_def, i)
                        progress_mod.report(task_id, i, total, f"{layers[i] if layers else step_def['name']}")
                        res = await provider.generate(p, pw, ph, n=1, task_id=task_id, **gen_kw)
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
                meta["promptLang"] = "zh"
                if ref_img is not None:
                    meta["refStep"] = ref_step
            else:
                # 动作关键帧
                if step_def.get("refChain"):
                    # 自锚定：先文生图定角色，再拿它当 img2img 参考生成其余帧。
                    # 这样每帧都锚在同一张脸上，不会越跑越偏。
                    #
                    # 双底座接力：当前模型是编辑器（qwen_edit）时，它没有自由
                    # 创作能力——帧 1 借 txt2img 底座生成，帧 2+ 回到编辑器用
                    # 编辑指令改姿势。整条链对用户透明，不用手动切模型。
                    gen_provider = provider
                    edit_relay = (
                        getattr(provider, "id", "") == "diffusers"
                        and detect_family(provider._resolve_model()) == "qwen_edit"
                    )
                    if edit_relay:
                        # txt2imgModel 在工作流定义里是相对路径，按项目根解析
                        txt_model = resolve_path(step_def.get("txt2imgModel") or str(
                            Path(__file__).resolve().parents[2] / "models" / "z-image-turbo"))
                        if not os.path.isdir(txt_model):
                            # 没装生成底座：退回单模型链（提示词锚定），
                            # 并把原因写进 meta，别让用户以为是 Bug
                            edit_relay = False
                            relay_note = f"txt2img 底座不存在：{txt_model}"
                        else:
                            relay_note = None
                            gen_provider = DiffusersProvider({**provider.cfg, "model": txt_model})
                    else:
                        relay_note = None
                    edit_tpl = step_def.get("editPrompt") or step_def.get("prompt")
                    anchor: Optional[Image.Image] = None
                    for i in range(n):
                        kw = dict(gen_kw)
                        if anchor is not None:
                            # Z-Image 没有 img2img，ref 会被忽略；编辑器拿它当参考图
                            kw["ref"] = anchor
                        gen = gen_provider if anchor is None else provider
                        p = prompt if anchor is None else (
                            _fill(edit_tpl, run, step_def, i) if edit_relay else prompt)
                        progress_mod.report(task_id, i, n, f"动作帧 {i + 1}/{n}")
                        res = await gen.generate(p, pw, ph, n=1,
                                                 seed=None if seed is None else int(seed) + i,
                                                 task_id=task_id, **kw)
                        for img in res.images:
                            artifacts.append(_save_asset(
                                img, kind="workflow", name=f"{run['id']}-{step_id}-{i:02d}",
                                meta={"runId": run["id"], "step": step_id, "index": i,
                                      "label": f"frame {i}", "anchored": anchor is not None},
                            )["id"])
                        if anchor is None and artifacts:
                            # 锚点先抠成透明再铺纯绿底：否则模型会把锚点里的
                            # 画框 / 草地 / 地毯当成画面内容一路继承下去
                            base = _load(artifacts[0])
                            try:
                                cut, _ = matte_pipe.connected_matte(
                                    base, tolerance=70, feather=0, method="key"
                                )
                                flat = Image.new("RGB", cut.size, (27, 67, 50))
                                flat.paste(cut, (0, 0), cut)
                                anchor = flat
                            except Exception:
                                anchor = base.convert("RGB")
                    meta = {"prompt": prompt, "frames": len(artifacts),
                            "provider": "diffusers",
                            "mode": "txt2img→edit-chain" if edit_relay
                                    else "txt2img×N（提示词锚定）",
                            "promptLang": "zh"}
                    if edit_relay:
                        meta["editModel"] = provider._resolve_model()
                        meta["txt2imgModel"] = txt_model
                    if relay_note:
                        meta["relayNote"] = relay_note
                else:
                    # 一次请求 n 张（无参考图时由模型自行发散姿态）
                    result = await provider.generate(prompt, pw, ph, n=n, seed=seed,
                                                     task_id=task_id, **gen_kw)
                    for i, img in enumerate(result.images):
                        artifacts.append(_save_asset(
                            img, kind="workflow", name=f"{run['id']}-{step_id}-{i:02d}",
                            meta={"runId": run["id"], "step": step_id, "index": i, "label": f"frame {i}"},
                        )["id"])
                    meta = {"prompt": prompt, "frames": len(artifacts),
                            "provider": result.meta.get("provider"),
                            "mode": result.meta.get("mode"),
                            **({"refStep": ref_step} if ref_img is not None else {}),
                            "promptLang": "zh"}

        elif stype == "video":
            # 视频底座生成整段动作，再走既有的抽帧/抠图/拼表链。
            # 参考图来自上一步产物（锚点图）；没有视频底座时给明确错误，
            # 不静默退回逐帧生成——那会让用户以为视频链路是好的。
            ref_ids = _step_artifacts(run, index - 1)
            if not ref_ids:
                raise RuntimeError("视频生成需要上一步的产物作为参考图")
            ref_img = _load(ref_ids[-1])
            prompt = overrides.get("prompt") or _fill(step_def["prompt"], run, step_def, 0)
            psteps = run.get("params", {}).get("steps")
            seed = run.get("params", {}).get("seed")
            if not hasattr(provider, "generate_video"):
                raise RuntimeError(
                    f"当前 Provider（{pid}）不支持视频生成。"
                    "需要本地 Wan2.2 TI2V-5B：PYTHONPATH=. python backend/fetch_wan22.py"
                )
            res = await provider.generate_video(
                prompt,
                ref_img,
                width=int(step_def.get("width") or 480),
                height=int(step_def.get("height") or 832),
                steps=int(psteps) if psteps else None,
                seed=int(seed) if seed is not None else None,
                num_frames=int(step_def.get("numFrames", 81)),
                fps=int(step_def.get("fps", 24)),
                max_side=int(step_def.get("maxSide", 832)),
                task_id=task_id,
            )
            for i, img in enumerate(res.images):
                artifacts.append(_save_asset(
                    img, kind="workflow", name=f"{run['id']}-{step_id}-f{i:03d}",
                    meta={"runId": run["id"], "step": step_id, "index": i,
                          "label": f"frame {i}", "fps": res.meta.get("fps")},
                )["id"])
            meta = {"prompt": prompt, "frames": len(artifacts),
                    "model": res.meta.get("model"), "mode": res.meta.get("mode"),
                    "fps": res.meta.get("fps"), "seconds": res.meta.get("seconds"),
                    "size": res.meta.get("size"), "promptLang": "zh"}
            if res.meta.get("vramWarning"):
                meta["vramWarning"] = res.meta["vramWarning"]

        elif stype == "fourdir_bases":
            # 四方向立绘（h3 的 recipe 是「每方向一张 master + 各自 prompt」，
            # 这里用同一角色描述 + 方向描述词生成四张，随后各自派生行走帧）
            pw = int(step_def.get("width") or 512)
            ph = int(step_def.get("height") or 768)
            psteps = run.get("params", {}).get("steps")
            pcfg = run.get("params", {}).get("cfg")
            seed = run.get("params", {}).get("seed")
            gen_kw2: Dict[str, Any] = {}
            if psteps:
                gen_kw2["steps"] = int(psteps)
            if pcfg is not None:
                gen_kw2["cfg"] = float(pcfg)
            tpl = step_def["prompt"]
            dirs_meta = []
            for i, d in enumerate(FOUR_DIRS):
                prompt = _fill(tpl, run, step_def, 0).replace("{dir_desc}", DIR_DESC[d])
                progress_mod.report(task_id, i, len(FOUR_DIRS), f"{DIR_DESC[d]}立绘")
                res = await provider.generate(
                    prompt, pw, ph, n=1, seed=seed, task_id=task_id, **gen_kw2)
                aid = _save_asset(
                    res.images[0], kind="workflow", name=f"{run['id']}-{step_id}-{d}",
                    meta={"runId": run["id"], "step": step_id, "direction": d,
                          "label": f"{DIR_DESC[d]}立绘", "prompt": prompt},
                )["id"]
                artifacts.append(aid)
                dirs_meta.append({"id": d, "assetId": aid, "prompt": prompt})
            meta = {"directions": dirs_meta, "promptLang": "zh",
                    "mode": "txt2img×4"}

        elif stype == "fourdir_walk":
            # 每方向：以该方向立绘为参考，逐帧换姿势（编辑底座），再确定性拼排
            from .pipelines import sprite_loop

            prev_meta = (run["steps"][index - 1].get("meta") or {}) if index else {}
            src_dirs = prev_meta.get("directions") or []
            if not src_dirs:
                raise RuntimeError("四方向行走帧需要上一步的「四方向立绘」产物")
            total = max(2, min(int(step_def.get("frames", 4)), 8))
            poses = WALK_POSES[:total]
            size = int(step_def.get("size", 128))
            out_dirs = []
            done = 0
            for d in src_dirs:
                base = _load(d["assetId"])
                seq = []
                for i in range(total):
                    progress_mod.report(task_id, done, len(src_dirs) * total,
                                        f"{DIR_DESC.get(d['id'], d['id'])} 行走 {i + 1}/{total}")
                    img, _fm = await sprite_loop.run_frame(
                        base, poses[i], i, total,
                        subject=run.get("input", ""), style="",
                        width=int(step_def.get("width") or 512),
                        height=int(step_def.get("height") or 768),
                        max_side=int(step_def.get("maxSide") or 768),
                        task_id=task_id,
                    )
                    seq.append(img)
                    done += 1
                sheet, norm_frames, index_json, cmeta, review = sprite_loop.compose(
                    seq, cols=total, size=size, matte=True,
                    colors=int(step_def.get("colors", 0)),
                )
                sheet_id = _save_asset(
                    sheet, kind="sheet", name=f"{run['id']}-{step_id}-{d['id']}-sheet",
                    meta={"runId": run["id"], "step": step_id, "direction": d["id"],
                          "index": index_json, "metrics": cmeta.get("metrics")},
                )["id"]
                review_id = _save_asset(
                    review, kind="sheet", name=f"{run['id']}-{step_id}-{d['id']}-review",
                    meta={"runId": run["id"], "step": step_id, "direction": d["id"]},
                )["id"]
                frame_ids = [
                    _save_asset(f, kind="frame",
                                name=f"{run['id']}-{step_id}-{d['id']}-{i}",
                                meta={"runId": run["id"], "step": step_id,
                                      "direction": d["id"], "index": i})["id"]
                    for i, f in enumerate(norm_frames)
                ]
                artifacts.extend([sheet_id, *frame_ids, review_id])
                out_dirs.append({
                    "id": d["id"], "sheetAssetId": sheet_id, "frames": len(norm_frames),
                    "frameSize": [index_json["frame_size"]["w"], index_json["frame_size"]["h"]],
                    "reviewAssetId": review_id,
                })
            meta = {"directions": out_dirs, "poses": poses, "framesPerDir": total}

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
            mode = str(step_def.get("mode", "flow"))
            frames = [_load(a) for a in src]
            artifacts = []
            used_flow = 0
            for i, f in enumerate(frames):
                artifacts.append(_save_asset(f, kind="workflow", name=f"{run['id']}-f{i:02d}",
                                                    meta={"runId": run["id"], "step": step_id, "index": i})["id"])
                if i < len(frames) - 1:
                    nxt = frames[i + 1]
                    if f.size != nxt.size:
                        nxt = nxt.resize(f.size, Image.NEAREST)
                    for g in range(per_gap):
                        alpha = (g + 1) / (per_gap + 1)
                        mid, imeta = _inbetween(f, nxt, alpha, mode)
                        used_flow += 1 if imeta.get("method") == "optical-flow-farneback" else 0
                        artifacts.append(_save_asset(
                            mid, kind="workflow", name=f"{run['id']}-f{i:02d}m{g}",
                            meta={"runId": run["id"], "step": step_id, "index": i, "inbetween": True,
                                  "inbetweenMeta": imeta},
                        )["id"])
            if per_gap == 0:
                note = "未补帧：像素素材建议保持原生帧数（2-4 帧），补帧请用视频模型抽帧"
            elif used_flow:
                note = f"光流插帧（双向 Farneback，最近邻 remap）：{used_flow} 帧按运动方向 warp 后融合"
            else:
                note = "已用相邻帧混合模拟中间帧（cv2 不可用或该档关闭了光流）"
            meta = {"generated": len(artifacts) - len(frames), "perGap": per_gap,
                    "mode": mode, "note": note}

        elif stype == "compose":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            frames = [_load(a) for a in src]
            cols = int(step_def.get("cols", 4))
            composed, index_json = sheet_pipe.compose_sheet(frames, cols)
            item = _save_asset(composed, kind="sheet", name=f"{run['id']}-sheet",
                                      meta={"runId": run["id"], "step": step_id, "index": index_json})
            artifacts = [item["id"]]
            # 交付前自检：单元格与源帧逐位一致（移植自 h3 的 check_bundle）
            meta = {"cols": cols, "frames": len(frames), "frameSize": index_json["frame_size"],
                    "checks": sheet_pipe.verify_composed(composed, frames, cols)}

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
                # 动作序列必须共享缩放：下蹲/抬腿的帧包围盒更小，各自撑满画布
                # 会让角色在整段动作里忽大忽小，一播就露馅
                shared_scale=True,
                size=int(overrides.get("size") or step_def.get("size", 128)),
                anchor=str(overrides.get("anchor") or step_def.get("anchor", "bottom")),
                matte=bool(step_def.get("matte", True)),
                tolerance=int(step_def.get("tolerance", 80)),
                feather=int(step_def.get("feather", 0)),
                hard=bool(step_def.get("hard", True)),
                matte_method=str(step_def.get("matteMethod", "auto")),
                inset=float(step_def.get("inset", 0.0)),
            )
            for i, out in enumerate(outs):
                artifacts.append(_save_asset(
                    out, kind="workflow", name=f"{run['id']}-sprite{i:02d}",
                    meta={"runId": run["id"], "step": step_id, "index": i},
                )["id"])
            meta = smeta

        elif stype == "scene_layers":
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            if len(src) != 1:
                raise RuntimeError(
                    f"语义分层需要恰好一张基准场景图，上一步给了 {len(src)} 张。"
                    "请先用单图步骤生成场景，或在上一步把 n 改成 1。")
            from .pipelines import layers as layers_pipe

            base = _load(src[0])
            imgs, lmeta = await layers_pipe.decompose(
                base,
                layers=int(overrides.get("layers") or step_def.get("layers", 4)),
                steps=step_def.get("steps"),
                seed=run.get("params", {}).get("seed"),
                resolution=int(step_def.get("resolution", 640)),
                task_id=task_id,
            )
            # 落产物前按「远→近」重排：视差导出按产物顺序分配 scroll_scale，
            # 顺序反了整张地图的远近关系就颠倒了。模型原序不做保证，
            # 覆盖率降序是启发式（背景铺满、前景物件小），meta 里已写明。
            artifacts = []
            labels = []
            for rank, li in enumerate(lmeta["order"]):
                item = _save_asset(
                    imgs[li], kind="workflow",
                    name=f"{run['id']}-{step_id}-{rank:02d}",
                    meta={"runId": run["id"], "step": step_id, "index": rank,
                          "layerIndex": li, "coverage": lmeta["coverage"][li],
                          "label": f"layer {li}（覆盖 {lmeta['coverage'][li]:.0%}）"},
                )
                artifacts.append(item["id"])
                labels.append(f"layer {li}")
            meta = {**lmeta, "labels": labels, "sourceAssetId": src[0]}

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

        elif stype in ("godot", "godot_parallax", "godot_fourdir"):
            from .godot_export import (build_fourdir_package, build_godot_package,
                                       build_parallax_package)
            src = overrides.get("artifacts") or _prev_artifacts(run, index)
            if not src:
                raise RuntimeError("没有可导出的产物")
            if stype == "godot":
                info = build_godot_package(src, run)
            elif stype == "godot_parallax":
                info = build_parallax_package(src, run)
            else:
                # 四方向包按上一步 meta 里的方向表导出（不靠产物顺序猜）
                prev_meta = (run["steps"][index - 1].get("meta") or {}) if index else {}
                dirs = prev_meta.get("directions") or []
                if not dirs:
                    raise RuntimeError("四向导出需要上一步的「四方向行走帧」产物")
                info = build_fourdir_package(dirs, run)
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
        progress_mod.finish(task_id, ok=True)
        return run

    except progress_mod.CancelledError:
        step_state.update({"status": "failed", "error": "已取消", "finishedAt": int(time.time())})
        _save_run(run)
        progress_mod.finish(task_id, ok=False, error="已取消")
        raise
    except Exception as e:
        step_state.update({"status": "failed", "error": str(e), "finishedAt": int(time.time())})
        _save_run(run)
        progress_mod.finish(task_id, ok=False, error=str(e))
        raise


async def run_all(run_id: str) -> Dict[str, Any]:
    run = get_run(run_id)
    if not run:
        raise KeyError("run 不存在")
    # 全程持锁：边顺序跑边让单步请求插进来，两边会互相覆盖状态
    async with _run_lock(run_id):
        wf = get_workflow(run["workflowId"])
        for i, step_def in enumerate(wf["steps"]):
            # 锁内重读最新状态，反映此前单步请求的改动
            run = get_run(run_id) or run
            if run["steps"][i]["status"] == "done":
                continue
            run = await _run_step_locked(run_id, step_def["id"])
    return run
