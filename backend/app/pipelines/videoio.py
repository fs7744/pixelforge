"""视频 / GIF 导入拆帧（移植自 FrameBaker 的 extract 任务流程）。

FrameBaker 的素材入口是「任何来源」：GIF / MP4 / PNG / 外部 CLI 生成，
gif/mp4 统一走 ffmpeg 抽帧（fps 模式或时间戳定点），可选自动抠图。
本项目的对应缺口：生成工作台只能从零生成，用户手里的手绘动画扫描件、
实拍参考、现有游戏录像没有入口。这个模块补上「现有影像 → 帧资产」。

与 FrameBaker 的差异：GIF/WebP 用 PIL 原生拆（不依赖外部二进制，
动画 GIF 的帧本来就是逐帧存储的）；MP4 必须 ffmpeg，没装就给出
明确错误和安装提示，不静默失败。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageSequence

from . import matte as matte_pipe

MAX_FRAMES_CAP = 256


def sniff_media(data: bytes) -> str:
    """按魔数识别媒体类型：gif / webp / mp4 / 空串（不认识）。

    只看文件头，不看客户端声明的 content_type、也不看扩展名——声明错的
    文件交给 ffmpeg 只会得到一句听不懂的报错，不如在入口就给「格式无法
    识别」。合法的 GIF/WebP/MP4/MOV 都有可靠魔数，不需要兜底。
    """
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "mp4"
    return ""


def _sample_step(durations: List[int], fps: float) -> int:
    """GIF 抽帧步长：源帧率 / 目标帧率，至少 1（不跳帧）。

    源帧率 = 1000 / 平均帧时长(ms)，所以要每 round(1000 / (avg * fps))
    帧取一帧。注意别写成 avg * fps / 1000——那是倒数，会把「抽稀」
    算成「全取」（120ms 的 GIF 要 2fps 应该每 4 帧取 1）。
    """
    if fps <= 0 or not durations:
        return 1
    avg = sum(durations) / len(durations)
    if avg <= 0:
        return 1
    return max(1, round(1000.0 / (avg * fps)))


def _pil_frames(data: bytes, fps: float, max_frames: int) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """GIF / WebP 拆帧（PIL 原生）。"""
    import io

    img = Image.open(io.BytesIO(data))
    durations: List[int] = []
    all_frames: List[Image.Image] = []
    for frame in ImageSequence.Iterator(img):
        all_frames.append(frame.convert("RGBA"))
        durations.append(max(1, int(frame.info.get("duration", 100))))
    step = _sample_step(durations, fps)
    picked = all_frames[::step]
    meta = {
        "source": "gif" if sniff_media(data) == "gif" else "webp",
        "totalFrames": len(all_frames),
        "avgFrameMs": round(sum(durations) / len(durations), 1) if durations else 0,
        "sampleStep": step,
    }
    return picked[:max_frames], meta


def _ffmpeg_frames(data: bytes, fps: float, max_frames: int) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """MP4 拆帧（ffmpeg）。没装 ffmpeg 时抛带安装指引的错误。"""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "未找到 ffmpeg，MP4 拆帧需要它：Windows 用 winget install ffmpeg"
            "（或 https://ffmpeg.org/download.html）；macOS 用 brew install ffmpeg；"
            "Linux 用 apt install ffmpeg。GIF 不需要 ffmpeg，可直接导入。"
        )
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "input.mp4"
        src.write_bytes(data)
        out = Path(td) / "out"
        out.mkdir()
        vf = f"fps={fps}" if fps > 0 else None
        cmd = ["ffmpeg", "-y", "-i", str(src), "-start_number", "0"]
        if vf:
            cmd += ["-vf", vf]
        cmd += [str(out / "f_%05d.png")]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg 拆帧超时（>5 分钟）：视频可能损坏或过长")
        if proc.returncode != 0:
            tail = (proc.stderr or b"")[-400:].decode("utf-8", "ignore")
            raise RuntimeError(f"ffmpeg 拆帧失败：{tail}")
        files = sorted(out.glob("f_*.png"))
        frames = [Image.open(p).convert("RGBA") for p in files[:max_frames]]
        meta = {"source": "mp4", "totalFrames": len(files), "sampleStep": 1}
        return frames, meta


def extract(
    data: bytes,
    filename: str = "",
    *,
    fps: float = 0,
    max_frames: int = 64,
    matte: bool = False,
    tolerance: int = 80,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """拆帧入口。fps<=0 表示原样全取；max_frames 封顶 256。

    matte=True 时逐帧抠图（连通域 + 边缘去污）：绿幕/纯色背景的录像
    直接得到透明帧，可以接着进精灵表管线——这就是 FrameBaker
    autoMatting 的对应场景。
    """
    kind = sniff_media(data)
    if not kind:
        name = f"（{filename}）" if filename else ""
        raise ValueError(f"无法识别的媒体格式{name}：支持 GIF / WebP / MP4（按文件头判断）")
    max_frames = max(1, min(int(max_frames), MAX_FRAMES_CAP))

    if kind in ("gif", "webp"):
        frames, meta = _pil_frames(data, fps, max_frames)
    else:
        frames, meta = _ffmpeg_frames(data, fps, max_frames)

    if not frames:
        raise ValueError("没有拆出任何帧：文件可能是空的或损坏")
    if len(frames) >= max_frames and meta.get("totalFrames", 0) > max_frames:
        meta["truncated"] = f"源共 {meta['totalFrames']} 帧，已按上限取前 {max_frames} 帧"

    if matte:
        cleaned = []
        removed = []
        for f in frames:
            cut, m = matte_pipe.connected_matte(f, tolerance=tolerance, hard=True)
            cleaned.append(cut)
            removed.append(m.get("removedRatio", 0))
        frames = cleaned
        meta["matte"] = {"tolerance": tolerance,
                         "avgRemoved": round(sum(removed) / len(removed), 3)}

    meta["frames"] = len(frames)
    meta["fps"] = fps
    meta["maxFrames"] = max_frames
    return frames, meta
