"""逐路由覆盖测试：32 个路由里所有「不需要加载模型」的端点 + 错误路径。

分工：
- 本脚本：抠图 / 像素化 / 缩放 / 镜像延展 / 无缝平铺 / 分表 / 合表 /
  确定性去水印 / Provider 测试与卸载 / 资产删除 / 设置回写 / run 错误路径 /
  参考图生成的输入校验（无图、超 4 张、坏资产 id、非图片）。
- e2e_gpu_test.py：/api/generate、工作流步骤执行、/api/pipeline/restyle。
- e2e_gpu_gap_test.py：/api/generate/ref 的真实出图（单图 + 多图）。
- test_ai_pipelines.py：watermark/ai、seamless/ai、restyle 三个管线的真实出图。
- /api/pipeline/threeview：三视图端点（需 GPU，见 e2e_gpu_test 说明）。

用法：python backend/coverage_api_test.py（自动把仓库根加进 sys.path）
"""
import atexit
import inspect
import io
import json
import os
import signal
import sys
import zipfile
from pathlib import Path

# 按 `python backend/coverage_api_test.py` 直接跑时，sys.path[0] 是 backend/，
# 没有仓库根，`import backend.app.main` 会失败。与 smoke_test.py 同一做法。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.app.main import app
from backend.app import storage

client = TestClient(app)
ok = 0
fail = 0
created = []
# 基线快照：收尾时做差集，兜底清掉任何漏登记的资产。
# 只靠 created 列表会漏：多图响应（sheet/split 的 frames、sprite/compose 的
# frames、threeview 的 4 张）经常只记得第一张，库会被测试跑胖。
_baseline_ids = {a["id"] for a in storage.list_assets()}


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name} {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name} {detail}")


def png(img, fmt="PNG"):
    b = io.BytesIO()
    img.save(b, fmt)
    return b.getvalue()


def remember(item):
    if item and item.get("id"):
        created.append(item["id"])


_cleanup_done = False


def cleanup_assets(verbose: bool = False) -> None:
    """删测试资产：登记表 + 快照差集。

    正常收尾与异常退出共用——测试中途挂死（SSE 流、GPU 卡住）时也要清，
    否则库会被跑胖几百张。atexit 覆盖 SystemExit；SIGTERM 处理器把信号
    转成 SystemExit，让 atexit 有机会执行（Linux CI 有效；Windows 的
    强杀进程无法拦截，那种情况按时间戳手工清）。
    """
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    removed = 0
    for aid in created:
        try:
            if client.delete(f"/api/assets/{aid}").status_code == 200:
                removed += 1
        except Exception:
            pass
    if verbose:
        check("测试资产已清理", removed == len(created), f"{removed}/{len(created)}")
    try:
        leaked = {a["id"] for a in storage.list_assets()} - _baseline_ids - set(created)
    except Exception:
        leaked = set()
    leaked_removed = 0
    for aid in leaked:
        try:
            if client.delete(f"/api/assets/{aid}").status_code == 200:
                leaked_removed += 1
        except Exception:
            pass
    if verbose:
        check("快照兜底无泄漏", leaked_removed == len(leaked), f"{leaked_removed}/{len(leaked)}")


atexit.register(cleanup_assets)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


# ---------- 造图 ----------
# 抠图用：纯色底 + 中央白块，边缘留一圈混色
matte_src = Image.new("RGB", (96, 96), (36, 164, 85))
d = ImageDraw.Draw(matte_src)
d.rectangle([28, 24, 67, 71], fill=(250, 250, 245))

# 分表用：4x4 每格一个颜色
sheet_src = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
for r in range(4):
    for c in range(4):
        shade = 30 + (r * 4 + c) * 14
        ImageDraw.Draw(sheet_src).rectangle(
            [c * 16, r * 16, c * 16 + 15, r * 16 + 15], fill=(shade, shade // 2, 255 - shade, 255))

# 去水印用：水印区已叠加过白色 logo（a≈0.5），反向混合应还原成深色
wm_src = Image.new("RGBA", (96, 96), (20, 60, 120, 255))
ImageDraw.Draw(wm_src).rectangle([56, 56, 91, 91], fill=(157, 157, 157, 255))
wm_mask = Image.new("L", (96, 96), 0)
ImageDraw.Draw(wm_mask).rectangle([56, 56, 91, 91], fill=128)


print("== 健康检查 ==")
r = client.get("/api/health")
check("GET /api/health", r.status_code == 200, str(r.json())[:60])


print("== Provider ==")
r = client.post("/api/providers/diffusers/test")
check("POST /api/providers/diffusers/test", r.status_code == 200 and "ok" in r.json(),
      str(r.json())[:80])
r = client.post("/api/providers/diffusers/unload")
check("POST /api/providers/diffusers/unload", r.status_code == 200 and r.json()["ok"] is True)
r = client.post("/api/providers/nope/test")
check("未知 Provider → 404", r.status_code == 404, str(r.status_code))


print("== 抠图 matte ==")
r = client.post("/api/pipeline/matte",
                files={"file": ("m.png", png(matte_src), "image/png")},
                data={"tolerance": 60, "feather": 3, "sample": "topleft"})
check("matte 200", r.status_code == 200, str(r.json())[:100])
if r.status_code == 200:
    j = r.json()
    remember(j.get("asset"))
    out = Image.open(storage.asset_path(j["asset"]))
    check("matte 输出为 RGBA", out.mode == "RGBA", out.mode)
    alpha = out.getchannel("A")
    check("matte 背景被抠透", alpha.getpixel((2, 2)) == 0, str(alpha.getpixel((2, 2))))
    check("matte 主体保留", alpha.getpixel((48, 48)) == 255, str(alpha.getpixel((48, 48))))
    check("matte meta 含方法", "method" in j["meta"], str(j["meta"])[:80])


print("== 像素化 pixelize ==")
r = client.post("/api/pipeline/pixelize",
                files={"file": ("m.png", png(matte_src), "image/png")},
                data={"colors": 8, "pixel_size": 4, "dither": "true", "palette": "auto"})
check("pixelize 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    j = r.json()
    remember(j.get("asset"))
    check("pixelize 抖动已生效并记录", j["meta"]["dither"] is True, str(j["meta"]["dither"]))
    check("pixelize 网格正确", j["meta"]["grid"] == [24, 24], str(j["meta"]["grid"]))

r = client.post("/api/pipeline/pixelize",
                files={"file": ("m.png", png(matte_src), "image/png")},
                data={"colors": 16, "pixel_size": 4, "palette": "gameboy"})
check("pixelize 固定调色板", r.status_code == 200 and r.json()["meta"]["palette"] == "gameboy",
      str(r.json().get("meta", {}).get("palette")))


print("== 缩放 resize ==")
for mode in ("hard", "fit"):
    r = client.post("/api/pipeline/resize",
                    files={"file": ("m.png", png(matte_src), "image/png")},
                    data={"width": 48, "height": 48, "mode": mode})
    check(f"resize mode={mode} 200", r.status_code == 200, str(r.json())[:80])
    if r.status_code == 200:
        remember(r.json().get("asset"))
        w, h = r.json()["asset"]["width"], r.json()["asset"]["height"]
        check(f"resize mode={mode} 尺寸", (w, h) == (48, 48), f"{w}x{h}")

# rpgmaker 是固定输出尺寸，不吃 width/height
for variant, expect in (("v1", (144, 192)), ("v2", (144, 240)), ("v2_four_row", (144, 192))):
    r = client.post("/api/pipeline/resize",
                    files={"file": ("m.png", png(matte_src), "image/png")},
                    data={"mode": "rpgmaker", "variant": variant})
    check(f"resize rpgmaker {variant} 200", r.status_code == 200, str(r.json())[:60])
    if r.status_code == 200:
        remember(r.json().get("asset"))
        a = r.json()["asset"]
        check(f"rpgmaker {variant} 尺寸", (a["width"], a["height"]) == expect,
              f"{a['width']}x{a['height']}")
        check(f"rpgmaker {variant} meta", r.json()["meta"].get("variant") == variant,
              str(r.json()["meta"].get("size")))

# fit 模式对非 RGBA 输入不应崩（历史 bug）
rgb_only = Image.new("RGB", (37, 91), (10, 20, 30))
r = client.post("/api/pipeline/resize",
                files={"file": ("r.png", png(rgb_only), "image/png")},
                data={"width": 64, "height": 64, "mode": "fit"})
check("resize fit 非 RGBA 输入不崩", r.status_code == 200, str(r.json())[:80])
if r.status_code == 200:
    remember(r.json().get("asset"))


print("== 镜像延展 seamless ==")
r = client.post("/api/pipeline/seamless",
                files={"file": ("m.png", png(matte_src), "image/png")},
                data={"times": 2, "blend": 24, "mirror": "true"})
check("seamless 镜像 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    remember(r.json().get("asset"))
    check("seamless 宽度翻倍", r.json()["asset"]["width"] == 192,
          str(r.json()["asset"]["width"]))
r = client.post("/api/pipeline/seamless",
                files={"file": ("m.png", png(matte_src), "image/png")},
                data={"times": 2, "blend": 24, "mirror": "false"})
check("seamless 重复模式 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    remember(r.json().get("asset"))


print("== 无缝平铺 tile ==")
r = client.post("/api/pipeline/tile",
                files={"file": ("m.png", png(matte_src), "image/png")},
                data={"band": 12, "feather": 3})
check("tile 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    remember(r.json().get("asset"))
    check("tile 尺寸不变", r.json()["asset"]["width"] == 96, str(r.json()["asset"]["width"]))
    check("tile meta 有方法", "method" in r.json()["meta"], str(r.json()["meta"])[:80])


print("== 分表 / 合表 ==")
r = client.post("/api/pipeline/sheet/split",
                files={"file": ("s.png", png(sheet_src), "image/png")},
                data={"cols": 4, "rows": 4})
check("sheet/split 200", r.status_code == 200, str(r.json())[:80])
frames = []
if r.status_code == 200:
    frames = r.json()["frames"]
    for f in frames:
        remember(f)
    check("拆出 16 帧", len(frames) == 16, str(len(frames)))
    check("帧尺寸 16x16", (frames[0]["width"], frames[0]["height"]) == (16, 16),
          f"{frames[0]['width']}x{frames[0]['height']}")
    check("index 记录 16 项", len(r.json()["index"]["frames"]) == 16)

# 精灵表拆分默认只切不去背；matte=true 时逐帧清掉键色。
# 绿幕 + 深色主体：拆完四角应透明，主体完整。
matte_sheet = Image.new("RGB", (64, 64), (0, 255, 0))
_md = ImageDraw.Draw(matte_sheet)
for gy in range(2):
    for gx in range(2):
        _md.rectangle((gx * 32 + 8, gy * 32 + 8, gx * 32 + 23, gy * 32 + 23),
                      fill=(30, 30, 30))
r = client.post("/api/pipeline/sheet/split",
                files={"file": ("m.png", png(matte_sheet), "image/png")},
                data={"cols": 2, "rows": 2, "matte": "true", "tolerance": "80"})
check("sheet/split 支持逐帧去背", r.status_code == 200, str(r.json())[:80])
if r.status_code == 200:
    ms = r.json()
    for f in ms["frames"]:
        remember(f)
    check("去背结果写进 index（method / background）",
          bool(ms["index"]["frames"][0].get("matte", {}).get("method")),
          str(ms["index"]["frames"][0].get("matte")))
    _px = Image.open(storage.asset_path(ms["frames"][0])).convert("RGBA")
    check("去背后四角透明", _px.getpixel((0, 0))[3] == 0, str(_px.getpixel((0, 0))))
    check("去背后主体保留", _px.getpixel((16, 16))[3] == 255 and
          _px.getpixel((16, 16))[:3] == (30, 30, 30), str(_px.getpixel((16, 16))))

# 视频/GIF 导入拆帧（移植自 FrameBaker 的 extract 流程）
_vg = io.BytesIO()
_vfs = []
for _i in range(3):
    _f = Image.new("RGB", (32, 32), (20, 20, 28))
    ImageDraw.Draw(_f).rectangle((4 + _i * 8, 8, 20 + _i * 8, 24), fill=(0, 255, 136))
    _vfs.append(_f)
_vfs[0].save(_vg, format="GIF", save_all=True, append_images=_vfs[1:], duration=100, loop=0)
r = client.post("/api/pipeline/video/frames",
                files={"file": ("a.gif", _vg.getvalue(), "image/gif")},
                data={"fps": "0", "max_frames": "64", "matte": "false"})
check("video/frames GIF 200", r.status_code == 200, str(r.json())[:100])
if r.status_code == 200:
    vf = r.json()
    for f in vf["frames"]:
        remember(f)
    check("GIF 拆出 3 帧", len(vf["frames"]) == 3, str(len(vf["frames"])))
    check("meta 记录来源与帧数",
          vf["meta"].get("source") == "gif" and vf["meta"].get("frames") == 3, str(vf["meta"])[:80])

# 绿幕 GIF + matte=true：逐帧抠图后应记录移除率
_vg2 = io.BytesIO()
_g1 = Image.new("RGB", (32, 32), (0, 255, 0))
ImageDraw.Draw(_g1).rectangle((8, 8, 24, 24), fill=(30, 30, 30))
_g1.save(_vg2, format="GIF", save_all=True, append_images=[_g1.copy()], duration=100, loop=0)
r = client.post("/api/pipeline/video/frames",
                files={"file": ("g.gif", _vg2.getvalue(), "image/gif")},
                data={"fps": "0", "matte": "true", "tolerance": "80"})
check("video/frames matte 200", r.status_code == 200, str(r.json())[:100])
if r.status_code == 200:
    _mp = r.json()["meta"].get("matte") or {}
    check("逐帧抠图 meta 记录移除率", "avgRemoved" in _mp, str(_mp))

# 非媒体文件 → 400（按文件头判断，扩展名不算数）
r = client.post("/api/pipeline/video/frames",
                files={"file": ("x.mp4", b"definitely not a video", "video/mp4")},
                data={"fps": "0"})
check("video/frames 非媒体 → 400", r.status_code == 400, f"{r.status_code} {r.text[:60]}")

# MP4 且本机没有 ffmpeg → 明确错误 + 安装提示（有 ffmpeg 时跳过：
# 造不出一段合法 MP4，不值得为测试引依赖）
import shutil as _shutil  # noqa: E402
if not _shutil.which("ffmpeg"):
    _mp4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
    r = client.post("/api/pipeline/video/frames",
                    files={"file": ("a.mp4", _mp4, "video/mp4")},
                    data={"fps": "0"})
    check("MP4 无 ffmpeg → 400 带安装指引",
          r.status_code == 400 and "ffmpeg" in str(r.json().get("detail", "")),
          f"{r.status_code} {r.text[:80]}")

# 参考图规范化（h3 normalize-master）：主体偏右上 + 统一底色
_rn = Image.new("RGB", (300, 200), (25, 25, 25))
ImageDraw.Draw(_rn).rectangle((200, 40, 260, 160), fill=(220, 80, 60))
r = client.post("/api/pipeline/refnorm", files={"file": ("r.png", png(_rn), "image/png")})
check("refnorm 200", r.status_code == 200, str(r.json())[:100])
if r.status_code == 200:
    remember(r.json()["asset"])
    _m = r.json()["meta"]
    check("refnorm 输出方形 16 对齐",
          _m["canvas"][0] == _m["canvas"][1] and _m["canvas"][0] % 16 == 0,
          str(_m["canvas"]))
    check("refnorm 记录主体框与底色判定",
          "subjectBox" in _m and "backgroundDetected" in _m, str(_m)[:100])

# 四方向行走工作流注册与步骤类型
_wfs = {w["id"]: w for w in client.get("/api/workflows").json()["workflows"]}
check("四方向行走工作流已注册",
      "walk_4dir" in _wfs and len(_wfs.get("walk_4dir", {}).get("steps", [])) == 3,
      str(list(_wfs))[:100])
if "walk_4dir" in _wfs:
    check("四方向步骤类型齐全",
          [s["type"] for s in _wfs["walk_4dir"]["steps"]]
          == ["fourdir_bases", "fourdir_walk", "godot_fourdir"],
          str([s["type"] for s in _wfs["walk_4dir"]["steps"]]))
# 未完成立绘就点行走帧 → 明确 400（依赖检查先拦住，不碰 GPU）
_run4 = client.post("/api/workflows/walk_4dir/runs",
                    json={"input": "四向测试", "params": {}}).json()
_r = client.post(f"/api/runs/{_run4['id']}/steps/walk/run", json={})
check("四向：缺立绘点行走帧 → 400",
      _r.status_code == 400 and "上一步" in str(_r.json().get("detail", "")),
      f"{_r.status_code} {_r.text[:80]}")
Path("data/runs", f"{_run4['id']}.json").unlink(missing_ok=True)

# 四方向编排全链路（桩替 run_frame，不碰 GPU）：立绘 meta → 每方向 4 帧
# → 横排精灵表 + 审查图 → Godot 四向包
print("== 四方向行走编排（run_frame 打桩） ==")
import backend.app.pipelines.sprite_loop as _sl  # noqa: E402

_orig_rf = _sl.run_frame


async def _fake_rf(ref, pose, index, total, **kw):
    im = Image.new("RGB", (48, 64), (40, 40, 50))
    ImageDraw.Draw(im).rectangle((14, 8, 34, 62), fill=(200, 120, 90))
    return im, {"mode": "stub", "pose": pose}


_sl.run_frame = _fake_rf
try:
    _r4 = client.post("/api/workflows/walk_4dir/runs",
                      json={"input": "桩测试", "params": {}}).json()
    _p4 = Path("data/runs") / f"{_r4['id']}.json"
    _st4 = json.loads(_p4.read_text(encoding="utf-8"))
    _dirs4 = []
    for d in ("down", "up", "left", "right"):
        it = storage.save_asset(Image.new("RGB", (64, 96), (30, 30, 40)),
                                kind="generated", name=f"stub-base-{d}")
        remember(it)
        _dirs4.append({"id": d, "assetId": it["id"], "prompt": "stub"})
    _st4["steps"][0]["status"] = "done"
    _st4["steps"][0]["meta"] = {"directions": _dirs4}
    _p4.write_text(json.dumps(_st4), encoding="utf-8")

    _r = client.post(f"/api/runs/{_r4['id']}/steps/walk/run", json={})
    _body = _r.json() if _r.status_code == 200 else {}
    _steps = _body.get("steps") or []
    _wm = (_steps[1].get("meta") or {}) if len(_steps) > 1 else {}
    _outs = _wm.get("directions") or []
    check("fourdir_walk 200", _r.status_code == 200, str(_body)[:100])
    check("四方向各出精灵表", len(_outs) == 4
          and all(d.get("sheetAssetId") for d in _outs), str([d.get("id") for d in _outs]))
    check("每方向 4 帧、帧尺寸 128", all(d["frames"] == 4 and d["frameSize"] == [128, 128]
                                   for d in _outs), str(_outs)[:140])
    check("产物数 = 4 × (表+4帧+审查)", len(_steps[1].get("artifacts") or []) == 24,
          str(len(_steps[1].get("artifacts") or [])))
    created.extend(_steps[1].get("artifacts") or [])

    _r = client.post(f"/api/runs/{_r4['id']}/steps/godot/run", json={})
    _body = _r.json() if _r.status_code == 200 else {}
    _gm = ((_body.get("steps") or [{}, {}, {}])[2].get("meta") or {})
    check("Godot 四向包导出", _r.status_code == 200 and _gm.get("kind") == "godot_fourdir"
          and _gm.get("directions") == ["down", "up", "left", "right"], str(_gm)[:110])
    if _gm.get("zip"):
        (Path("data/exports") / _gm["zip"].rsplit("/", 1)[-1]).unlink(missing_ok=True)
finally:
    _sl.run_frame = _orig_rf
    Path("data/runs", f"{_r4['id']}.json").unlink(missing_ok=True)

# 硬 alpha 保留「描边 + 键色」的抗锯齿混色：绿幕底实测边缘环 51% 偏绿。
# clean_edge_spill 只改 RGB 不动 alpha，把它洗成内侧颜色。
from backend.app.pipelines import matte as matte_pipe  # noqa: E402
_edge = Image.new("RGB", (40, 40), (0, 255, 0))
_ed = ImageDraw.Draw(_edge)
_ed.rectangle((10, 10, 29, 29), fill=(20, 20, 20))
# 1px「黑 + 绿」混色边，模拟模型抗锯齿
_ed.rectangle((9, 9, 30, 30), outline=(10, 140, 10))
_arr = np.array(_edge.convert("RGBA"))
_arr[:, :, 3] = 0
_arr[9:31, 9:31, 3] = 255
_cut = Image.fromarray(_arr, "RGBA")


def _green_ring(im):
    a = np.asarray(im.convert("RGBA")).astype(np.int32)
    al = a[:, :, 3] > 8
    inner = np.zeros_like(al)
    inner[1:-1, 1:-1] = (al[1:-1, 1:-1] & al[:-2, 1:-1] & al[2:, 1:-1]
                         & al[1:-1, :-2] & al[1:-1, 2:])
    ring = al & ~inner
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return int(ring.sum()), int((ring & (g > r + 25) & (g > b + 25)).sum())


_n0, _g0 = _green_ring(_cut)
_clean, _npx = matte_pipe.clean_edge_spill(_cut, (0, 255, 0))
_n1, _g1 = _green_ring(_clean)
check("边缘清洗确实改到像素", _npx > 0, str(_npx))
check("绿幕边缘溢色大幅下降", _g1 <= max(1, _g0 // 5), f"{_g0} → {_g1}（环 {_n0}px）")
check("边缘清洗不动 alpha", np.array_equal(
    np.asarray(_cut)[:, :, 3], np.asarray(_clean)[:, :, 3]))
# 蓝幕同样适用（主通道判定不是写死绿）
assert _green_ring(_cut)[0] > 0
check("清洗认的是底色主通道，不写死绿",
      matte_pipe.clean_edge_spill(_cut, (0, 0, 255))[1] == 0)

files = []
for i, f in enumerate(frames[:6]):
    files.append(("files", (f"f{i}.png", png(Image.open(storage.asset_path(f))), "image/png")))
r = client.post("/api/pipeline/sheet/compose", files=files, data={"cols": 3})
check("sheet/compose 200", r.status_code == 200, str(r.json())[:80])
if r.status_code == 200:
    remember(r.json().get("asset"))
    a = r.json()["asset"]
    check("合成板 3 列 2 行", (a["width"], a["height"]) == (48, 32), f"{a['width']}x{a['height']}")


print("== 确定性去水印 ==")
r = client.post("/api/pipeline/watermark",
                files={"file": ("w.png", png(wm_src), "image/png")},
                data={"size": "48"})
check("无掩码 → 不破坏原图", r.status_code == 200 and r.json()["meta"]["applied"] is False
      and r.json()["asset"] is None, str(r.json()["meta"])[:80])

r = client.post("/api/pipeline/watermark",
                files={"file": ("w.png", png(wm_src), "image/png"),
                       "mask": ("k.png", png(wm_mask), "image/png")},
                data={"size": "48", "strength": "1.0"})
check("带掩码 → 应用反混合", r.status_code == 200 and r.json()["meta"]["applied"] is True,
      str(r.json().get("meta"))[:80])
if r.status_code == 200 and r.json().get("asset"):
    remember(r.json()["asset"])
    out = Image.open(storage.asset_path(r.json()["asset"])).convert("RGB")
    px = out.getpixel((75, 75))
    # 反混合：m=0.502、logo=255 → (157 - 0.502*255)/0.498 ≈ 58
    check("水印区被还原（反混合生效）", px[0] < 100, str(px))


print("== 资产 / 导出 ==")
r = client.get("/api/assets")
check("GET /api/assets", r.status_code == 200 and len(r.json()["assets"]) > 0)
r = client.get("/api/assets?kind=matte")
check("按 kind 过滤", r.status_code == 200 and all(a["kind"] == "matte" for a in r.json()["assets"]),
      str(len(r.json()["assets"])))
r = client.get("/api/assets/nonexistent/download")
check("下载不存在资产 → 404", r.status_code == 404, str(r.status_code))
z = client.post("/api/export/zip", json={"ids": created[:3]})
check("ZIP 导出", z.status_code == 200 and len(z.content) > 0, f"{len(z.content)} bytes")
if z.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        check("ZIP 含 index.json", "index.json" in zf.namelist(), str(zf.namelist()[:3]))


print("== 资产库：类型 / 来源两个维度 ==")
# 前端筛选栏曾经手写死 6 个 kind，而实际写入的有 12 种，于是「像素化 / 工作流 /
# 水印 / 换画风 / 拼图 / 三视图」这些资产在库里选不出来。facets 改成从索引派生后，
# 这里锁住两条不变量：出现过的 kind 一定有筛选项；三视图靠 pipeline 能被单独筛出。
before_total = len(client.get("/api/assets").json()["assets"])
facets = client.get("/api/assets/facets").json()
check("facets：同时给出 kinds 与 pipelines",
      isinstance(facets, dict) and isinstance(facets.get("kinds"), list)
      and isinstance(facets.get("pipelines"), list))
check("facets：total 与 /api/assets 条数一致", facets.get("total") == before_total,
      f"{facets.get('total')} vs {before_total}")

present = {a["kind"] for a in client.get("/api/assets").json()["assets"]}
missing = sorted(present - {k["id"] for k in facets["kinds"]})
check("每个实际存在的 kind 都有筛选项（不再有选不出来的类型）", not missing, str(missing))
check("facets 的 kind 数量都 ≥ 1", all(k["count"] >= 1 for k in facets["kinds"]))

tv_asset = storage.save_asset(Image.new("RGB", (32, 32), (10, 20, 30)), kind="generated",
                              name="facet-threeview", meta={"pipeline": "threeview"})
direct_asset = storage.save_asset(Image.new("RGB", (32, 32), (30, 20, 10)), kind="generated",
                                  name="facet-direct")
remember(tv_asset)
remember(direct_asset)

tv_ids = [a["id"] for a in client.get("/api/assets?pipeline=threeview").json()["assets"]]
check("pipeline=threeview 能单独筛出三视图",
      tv_asset["id"] in tv_ids and direct_asset["id"] not in tv_ids, f"{len(tv_ids)} 条")
check("pipeline=direct 兜住没有 pipeline 的记录",
      direct_asset["id"] in [a["id"] for a in client.get("/api/assets?pipeline=direct").json()["assets"]])
both = client.get("/api/assets?kind=generated&pipeline=threeview").json()["assets"]
check("kind + pipeline 取交集",
      tv_asset["id"] in [a["id"] for a in both] and direct_asset["id"] not in [a["id"] for a in both])
check("「全部」视同不过滤",
      len(client.get("/api/assets?kind=全部&pipeline=全部").json()["assets"]) == before_total + 2)
check("未知 pipeline 返回空列表而不是报错",
      client.get("/api/assets?pipeline=nope").json()["assets"] == [])
check("三视图的来源标签是中文",
      any(p["id"] == "threeview" and p["label"] == "三视图" for p in
          client.get("/api/assets/facets").json()["pipelines"]))

print("== 资产缩略图（素材库列图 / 图生图取图区） ==")
# 取图区原来直接铺原图：150+ 张 PNG 一次几百 MB，而且 object-fit: cover 会把
# 三视图拼板（3:1）裁成中间一团，看上去和单视图没区别。缩略图必须等比缩。
wide = storage.save_asset(Image.new("RGBA", (600, 200), (20, 60, 90)), kind="generated",
                          name="thumb-wide", meta={"pipeline": "threeview"})
r = client.get(f"/api/assets/{wide['id']}/thumb?w=160")
th = Image.open(io.BytesIO(r.content))
check("缩略图：200 且是 PNG", r.status_code == 200 and th.format == "PNG", r.status_code)
check("缩略图：等比缩放不裁切（宽条仍是宽条）", th.size == (160, 53), str(th.size))
check("缩略图：带长缓存头", "immutable" in r.headers.get("cache-control", ""),
      r.headers.get("cache-control", ""))
r2 = client.get(f"/api/assets/{wide['id']}/thumb?w=9999")
big = Image.open(io.BytesIO(r2.content))
check("缩略图：超大宽度收敛到最大档（320）",
      big.size[0] == 320 and abs(big.size[1] - 320 / 3) <= 1, str(big.size))
check("缩略图：不同宽度各自缓存，不互相覆盖",
      len(list(storage.THUMB_DIR.glob(f"{wide['id']}-*.png"))) == 2)
check("缩略图：不存在的资产 → 404",
      client.get("/api/assets/nonexistent01/thumb").status_code == 404)
client.delete(f"/api/assets/{wide['id']}")
check("删除资产时缩略图缓存一并清理",
      not list(storage.THUMB_DIR.glob(f"{wide['id']}-*.png")))

print("== 工作流 run 错误路径 ==")
run = client.post("/api/workflows/scene/runs", json={"input": "测试场景", "params": {}}).json()
run_id = run["id"]
check("GET run 详情", client.get(f"/api/runs/{run_id}").status_code == 200)
check("非法 run_id → 404", client.get("/api/runs/zzzzzzzzzzzz").status_code == 404)
check("路径穿越 run_id → 404",
      client.get("/api/runs/..%5C..%5Cdata%5Cassets%5Cindex").status_code == 404)
r = client.post(f"/api/runs/{run_id}/steps/nope/run", json={})
check("未知步骤 → 404", r.status_code == 404, str(r.status_code))
r = client.post("/api/runs/zzzzzzzzzzzz/run-all")
check("未知 run 的 run-all → 404", r.status_code == 404, str(r.status_code))


print("== 参考图生成（图生图）的输入校验 ==")
r = client.post("/api/generate/ref", data={"prompt": "测试"})
check("无参考图 → 400", r.status_code == 400, r.json().get("detail", "")[:40])
r = client.post("/api/generate/ref", data={"prompt": "测试"},
                files=[("refs", (f"r{i}.png", png(matte_src), "image/png")) for i in range(5)])
check("超过 4 张 → 400", r.status_code == 400, r.json().get("detail", "")[:40])
r = client.post("/api/generate/ref", data={"prompt": "测试", "refAssetIds": "notexist"},
                files=[("refs", ("a.png", png(matte_src), "image/png"))])
check("资产 id 不存在 → 400", r.status_code == 400, r.json().get("detail", "")[:40])
r = client.post("/api/generate/ref", data={"prompt": "测试"},
                files=[("refs", ("a.png", b"notanimage", "image/png"))])
check("参考图非图片 → 400", r.status_code == 400, r.json().get("detail", "")[:40])

print("== 错误输入 ==")
r = client.post("/api/pipeline/matte", files={"file": ("x.png", b"notanimage", "image/png")})
check("非图片输入 → 400", r.status_code == 400, str(r.status_code))
r = client.post("/api/pipeline/matte", files={"file": ("x.png", b"", "image/png")})
check("空文件 → 400", r.status_code == 400, str(r.status_code))
r = client.post("/api/pipeline/restyle",
                files={"file": ("x.png", png(matte_src), "image/png")},
                data={"style": "no_such_style"})
check("未知画风 → 4xx", 400 <= r.status_code < 500, str(r.status_code))


print("== 精灵表：逐帧动作生成（不加载模型的部分） ==")
# 造 4 帧：高度递增，用来验证「共享缩放」——逐帧各自撑满会让角色在循环里忽大忽小
_sp_ids = []
for i in range(4):
    im = Image.new("RGB", (96, 96), (0, 255, 0))
    dd = ImageDraw.Draw(im)
    hh = 36 + i * 14
    dd.rectangle((36, 92 - hh, 60, 92), fill=(30, 30, 30))
    rr = client.post("/api/pipeline/matte",
                     files={"file": ("f.png", png(im), "image/png")},
                     data={"tolerance": 80, "feather": 0})
    _id = rr.json()["asset"]["id"]
    _sp_ids.append(_id)
    remember(rr.json()["asset"])

_r = client.post("/api/pipeline/sprite/compose",
                 json={"frameAssetIds": _sp_ids, "cols": 2, "size": 64, "matte": True})
check("sprite/compose 合成成功", _r.status_code == 200, str(_r.json())[:120])
if _r.status_code == 200:
    _j = _r.json()
    remember(_j["asset"])
    for f in _j["frames"]:
        remember(f)
    check("合成按 cols 排成 2×2", _j["index"]["grid"] == {"cols": 2, "rows": 2}, str(_j["index"]["grid"]))
    check("单帧尺寸统一为 64", _j["index"]["frame_size"] == {"w": 64, "h": 64}, str(_j["index"]["frame_size"]))
    _sr = _j["meta"]["normalize"]["scaleRange"]
    check("共享缩放：所有帧用同一个系数（下蹲帧不被放大）",
          _j["meta"]["normalize"]["sharedScale"] and _sr and abs(_sr[0] - _sr[1]) < 1e-6, str(_sr))
    check("精灵表按帧数铺满", _j["asset"]["width"] == 128 and _j["asset"]["height"] == 128,
          f'{_j["asset"]["width"]}x{_j["asset"]["height"]}')
    # 审查图（移植自 h3 的 extraction.review）：4 帧 2 列 → 宽 2*(64+6)+6
    if _j.get("review"):
        remember(_j["review"])
        check("附带帧审查图", _j["review"]["width"] == 2 * (64 + 6) + 6 and _j["review"]["height"] == 2 * (64 + 26 + 6) + 6,
              f'{_j["review"]["width"]}x{_j["review"]["height"]}')
    else:
        check("附带帧审查图", False, "响应里没有 review")
    check("compose 报告含警告清单与自检",
          isinstance(_j["meta"].get("warnings"), list) and _j["meta"].get("checks", {}).get("cellsMatchFrames"),
          str(_j["meta"].get("checks")))

check("sprite/compose 空帧 → 400",
      client.post("/api/pipeline/sprite/compose", json={"frameAssetIds": []}).status_code == 400)

# 循环段选择（h3 select）：取 [1, 4) 三帧合成，meta 明确「未插帧/未跳帧」
_r = client.post("/api/pipeline/sprite/select",
                 json={"frameAssetIds": _sp_ids, "start": 1, "stop": 4,
                       "cols": 3, "size": 64, "matte": True, "colors": 0})
check("sprite/select 200", _r.status_code == 200, str(_r.json())[:120])
if _r.status_code == 200:
    _sj = _r.json()
    remember(_sj["asset"])
    for f in _sj["frames"]:
        remember(f)
    if _sj.get("review"):
        remember(_sj["review"])
    _sel = _sj["meta"]["selection"]
    check("选段区间与帧数正确",
          _sel["interval"] == [1, 4] and len(_sj["frames"]) == 3, str(_sel))
    check("选段声明未插帧/未跳帧/未改时长",
          _sel["frameSkipping"] is False and _sel["interpolation"] is False
          and _sel["retiming"] is False, str(_sel))
    check("选段合成 3 帧排 1×3", _sj["index"]["grid"] == {"cols": 3, "rows": 1},
          str(_sj["index"]["grid"]))
check("sprite/select 区间不足 2 帧 → 400",
      client.post("/api/pipeline/sprite/select",
                  json={"frameAssetIds": _sp_ids, "start": 2, "stop": 2}).status_code == 400)
check("sprite/select 帧数不足 → 400",
      client.post("/api/pipeline/sprite/select",
                  json={"frameAssetIds": ["one"], "start": 0, "stop": 1}).status_code == 400)
check("sprite/compose 超过 64 帧 → 400",
      client.post("/api/pipeline/sprite/compose",
                  json={"frameAssetIds": ["x"] * 65}).status_code == 400)
check("sprite/frame 动作留空 → 400",
      client.post("/api/pipeline/sprite/frame",
                  data={"pose": "  ", "index": 0, "total": 1,
                        "refAssetId": _sp_ids[0]}).status_code == 400)
check("sprite/frame 没有参考图 → 400",
      client.post("/api/pipeline/sprite/frame",
                  data={"pose": "站立", "index": 0, "total": 1}).status_code == 400)
check("sprite/frame 帧序号越界 → 400",
      client.post("/api/pipeline/sprite/frame",
                  data={"pose": "站立", "index": 3, "total": 3,
                        "refAssetId": _sp_ids[0]}).status_code == 400)
check("sprite/frame 参考图不存在 → 400/404",
      client.post("/api/pipeline/sprite/frame",
                  data={"pose": "站立", "index": 0, "total": 1,
                        "refAssetId": "nope"}).status_code in (400, 404))

# 指令写法是有实测依据的：带「保持…不变」的约束会把编辑底座压成「什么都别改」，
# 出来 16 格一个姿势。回归锁：不许再出现这类压制性措辞。
from backend.app.pipelines import sprite_loop as sl  # noqa: E402

_inst = sl.pose_instruction("迈步", 1, 4)
check("逐帧指令带上动作名", "迈步" in _inst, _inst[:40])
check("逐帧指令把两字动作扩成肢体描述", "蹬直" in _inst or "迈出" in _inst, _inst[:60])
check("逐帧指令不含「保持不变」这类压制措辞",
      not any(w in _inst for w in ("保持不变", "不要改", "完全不变")), _inst[:60])
check("逐帧指令要求全身入画", "全身" in _inst)
check("未知动作也给出可用指令", "旋转" in sl.pose_instruction("旋转", 0, 1))

# ---- 生成底色：默认不是绿幕 ----
# 实测（640×896、同提示词同 seed、只改背景色）：模型声明 #00FF00 实际画出薄荷绿
# [90,199,126] + 径向渐变，并把绿染到角色边缘（内圈偏底色像素绿幕 76.5% vs
# 中性灰 33.3%）。所以默认底色必须是中性灰——抠不干净的根因在生成侧。
from backend.app.pipelines import backdrop as bdp  # noqa: E402

check("默认底色是中性灰而非绿幕",
      bdp.DEFAULT == "neutral", bdp.DEFAULT)
check("底色选项至少含中性灰/绿幕/白",
      {"neutral", "green", "white"} <= {b["id"] for b in bdp.list_backdrops()},
      str([b["id"] for b in bdp.list_backdrops()]))
check("底色提示词钉死纯色、无渐变无接地阴影",
      all(k in bdp.backdrop_prompt("neutral") for k in ("均匀", "无渐变", "接地阴影")),
      bdp.backdrop_prompt("neutral")[:50])
check("绿幕提示词含 #00FF00（老用户按惯例选它时行为不变）",
      "#00FF00" in bdp.backdrop_prompt("green"))
check("底色 rgb 解析正确",
      bdp.backdrop_rgb("green") == (0, 255, 0)
      and bdp.backdrop_rgb("neutral") == (128, 128, 128),
      str(bdp.backdrop_rgb("neutral")))
_try_bd = None
try:
    bdp.get_backdrop("nope")
except ValueError as e:
    _try_bd = str(e)
check("未知底色报错并列出合法值",
      _try_bd is not None and "neutral" in _try_bd, str(_try_bd)[:60])
check("逐帧指令默认用中性底（不钉绿幕）",
      "#00FF00" not in sl.pose_instruction("迈步", 1, 4)
      and "#808080" in sl.pose_instruction("迈步", 1, 4),
      sl.pose_instruction("迈步", 1, 4)[-40:])
check("逐帧指令可切回绿幕",
      "#00FF00" in sl.pose_instruction("迈步", 1, 4, "green"))
check("兜底提示词同样跟随底色",
      "#808080" in sl.pose_prompt("迈步", 0, 4, "猫武士")
      and "#00FF00" in sl.pose_prompt("迈步", 0, 4, "猫武士", "", "green"))
_r_bd = client.get("/api/pipeline/backdrops")
_bd_json = _r_bd.json() if _r_bd.status_code == 200 else {}
check("底色列表接口可用且带默认项",
      _r_bd.status_code == 200 and _bd_json.get("default") == "neutral"
      and len(_bd_json.get("backdrops") or []) >= 3,
      str(_r_bd.status_code))
check("底色列表不含内部提示词（不想让前端拼提示词）",
      all("prompt" not in b for b in _bd_json.get("backdrops") or []))

print("== 模板库：提示词纯净度 ==")
from backend.app import templates as tpl_mod

r = client.get("/api/templates")
check("模板接口可用", r.status_code == 200)
api_tpls = {t["id"]: t for t in r.json()["templates"]}

# 混进 prompt 的「给人看的写作说明」——出现即污染（这些词只会误导模型）
DIRTY = ("替换", "示例", "无需", "管线", "勾选", "填写", "注意：", "如：",
         "花括号", "提示词里", "占位符")
bad_placeholder, dirty_hits, missing_notes, bad_kind = [], [], [], []
# 用接口返回的渲染版：占位符在服务端就填成默认值，前端拿到的能直接跑
for t in tpl_mod.api_items():
    p = t["prompt"]
    if "{" in p or "}" in p:
        bad_placeholder.append(t["id"])
    hit = [w for w in DIRTY if w in p]
    if hit:
        dirty_hits.append((t["id"], hit))
    if not t.get("notes"):
        missing_notes.append(t["id"])
    if t.get("kind") not in ("image", "text"):
        bad_kind.append(t["id"])

# 原始模板里的占位符必须都在 fields 里声明过，否则渲染会漏成 `{xxx}` 送给模型
undeclared = []
for t in tpl_mod.TEMPLATES:
    ids = {f["id"] for f in t.get("fields") or []}
    import re as _re
    used = set(_re.findall(r"\{([a-zA-Z_][\w]*)\}", t.get("prompt", "")))
    used |= set(_re.findall(r"\{([a-zA-Z_][\w]*)\}", (t.get("textMode") or {}).get("prompt", "")))
    if used - ids:
        undeclared.append((t["id"], sorted(used - ids)))
check("prompt 里的占位符都在 fields 里声明", not undeclared, str(undeclared))
check("字段都有 id/label/default", all(
    f.get("id") and f.get("label") and "default" in f
    for t in tpl_mod.TEMPLATES for f in t.get("fields") or []))
check("精灵表带「照参考图」的角色字段",
      any(f.get("fromRef") for f in api_tpls["char_sprite_sheet"].get("fields") or []))

check("占位符已填成可直接跑的示例值", not bad_placeholder, str(bad_placeholder))
check("prompt 不含写作说明", not dirty_hits, str(dirty_hits))
check("每条都带页面提示 notes", not missing_notes, str(missing_notes))
check("kind 合法（image/text）", not bad_kind, str(bad_kind))
check("notes 不进 prompt", all(
    note not in t["prompt"] for t in tpl_mod.TEMPLATES for note in t.get("notes", [])))
check("文本任务已标记 kind=text",
      {t["id"] for t in tpl_mod.TEMPLATES if t.get("kind") == "text"}
      == {"char_style_extract", "eng_spec_doc"})
check("接口透传 notes/kind", all(
    "notes" in api_tpls[t["id"]] and "kind" in api_tpls[t["id"]]
    for t in tpl_mod.TEMPLATES))

# 文本任务不做操作面板：只给「去别的对话式 AI 怎么做」的指引，
# 所以 guide 必须齐备——缺一项前端就只剩一句「本页不出图」的空说明。
text_tpls = [t for t in tpl_mod.TEMPLATES if t.get("kind") == "text"]
no_guide = [t["id"] for t in text_tpls if not isinstance(t.get("guide"), dict)]
guide_fields = ("needsImage", "targets", "steps", "edit", "after")
bad_guide = [
    (t["id"], [k for k in guide_fields if k not in t["guide"]])
    for t in text_tpls if isinstance(t.get("guide"), dict)
]
bad_guide = [x for x in bad_guide if x[1]]
guide_targets = [t["id"] for t in text_tpls
                 if isinstance(t.get("guide"), dict) and len(t["guide"].get("targets") or []) < 2]
guide_steps = [t["id"] for t in text_tpls
               if isinstance(t.get("guide"), dict) and len(t["guide"].get("steps") or []) < 2]
image_with_guide = [t["id"] for t in tpl_mod.TEMPLATES
                    if t.get("kind") != "text" and t.get("guide")]
check("文本任务都带 guide", not no_guide, str(no_guide))
check("guide 字段齐备", not bad_guide, str(bad_guide))
check("guide 至少列两个对话式 AI", not guide_targets, str(guide_targets))
check("guide 至少两步操作", not guide_steps, str(guide_steps))
check("只有文本任务带 guide", not image_with_guide, str(image_with_guide))
check("读图任务的 guide 标了 needsImage",
      api_tpls["char_style_extract"].get("guide", {}).get("needsImage") is True)
check("纯文本任务的 guide 不需要图",
      api_tpls["eng_spec_doc"].get("guide", {}).get("needsImage") is False)

# 双模式模板（目前只有角色精灵表）：同一个模板既能本地出图，也能复制提示词
# 交给外部出图 AI。两段 prompt 必须是两份独立话术，不能拿一段凑两家。
dual = [t for t in tpl_mod.TEMPLATES if t.get("textMode")]
check("双模式模板存在且是精灵表",
      {t["id"] for t in dual} == {"char_sprite_sheet"},
      str([t["id"] for t in dual]))
tm = api_tpls["char_sprite_sheet"].get("textMode") or {}
check("接口透传 textMode", bool(tm.get("prompt")), str(list(tm)))
check("textMode 与主 prompt 不是同一段话",
      tm.get("prompt") != tpl_mod.TEMPLATES[[t["id"] for t in tpl_mod.TEMPLATES]
                                           .index("char_sprite_sheet")]["prompt"])
check("textMode 带 hint 与 guide",
      bool(tm.get("hint")) and isinstance(tm.get("guide"), dict))
check("textMode 的 guide 字段齐备",
      set(("needsImage", "targets", "steps", "edit", "after")) - set((tm.get("guide") or {})) == set(),
      str(set(("needsImage", "targets", "steps", "edit", "after")) - set(tm.get("guide") or {})))
check("textMode 的 guide 至少两个 AI / 两步操作",
      len((tm.get("guide") or {}).get("targets") or []) >= 2
      and len((tm.get("guide") or {}).get("steps") or []) >= 2)
check("textMode 的 prompt 同样不含写作说明",
      not [w for w in DIRTY if w in (tm.get("prompt") or "")]
      and "{" not in (tm.get("prompt") or ""))

# 抠图底色：管线不是硬编码绿幕（边缘采样做色键），但同一功能两处写不同颜色
# 会让用户不知道信哪个。项目约定是绿幕，洋红是精灵表的历史孤例。
workflows_src = (Path(__file__).resolve().parent / "app" / "workflows.py").read_text(
    encoding="utf-8")
sprite = api_tpls["char_sprite_sheet"]
check("精灵表用绿幕键色", "#00FF00" in sprite["prompt"] and "洋红" not in sprite["prompt"],
      sprite["prompt"][-60:])
check("精灵表两条路线都用绿幕",
      "洋红" not in (tm.get("prompt") or "") and "#00FF00" in (tm.get("prompt") or ""))
check("模板与工作流的精灵表底色一致（都用绿）",
      "纯绿色背景" in sprite["prompt"] and "纯绿色背景" in workflows_src)
check("notes 里的底色说明跟着改（不再提洋红）",
      not [n for n in sprite.get("notes", []) if "洋红" in n])

print("== 画风推荐表 ==")
r = client.get("/api/styles")
check("画风接口可用", r.status_code == 200)
sty = r.json()
style_ids = {s["id"] for s in sty["styles"]}
rec = sty.get("recommended") or {}
check("接口返回 recommended", bool(rec), str(list(rec)))
check("推荐表覆盖全部分类",
      set(tpl_mod.CATEGORIES) - {"全部"} == set(rec), str(set(rec)))
check("推荐 id 都真实存在",
      all(i in style_ids for ids in rec.values() for i in ids),
      str([i for ids in rec.values() for i in ids if i not in style_ids]))
check("工程类不给画风（文本任务）", rec.get("工程") == [], str(rec.get("工程")))
check("每类推荐 4-6 个，不全量堆出来",
      all(4 <= len(ids) <= 6 for k, ids in rec.items() if k != "工程"),
      str({k: len(v) for k, v in rec.items()}))
check("推荐数少于画风总数（确实做了收敛）",
      all(len(ids) < len(style_ids) for k, ids in rec.items() if k != "工程"),
      f"{len(style_ids)} 种画风")

print("== 画风确实进提示词 ==")
# 线上反馈「选了画风没生效」。根因有三层，逐条守住：
# ① 风格词曾经拼在尾部，对整图语义影响最弱；
# ② 模板 prompt 自己写死了画风（「Q版日系卡通」），与新选画风直接打架；
# ③ 用户看不到最终提示词，无从判断到底加没加。
from backend.app import styles as sty_mod  # noqa: E402

sv = sty_mod.apply_style("灰白英短猫武士，红披风", "manga_bw")
check("风格词前置到开头", sv["prompt"].startswith(sty_mod.get_style("manga_bw")["prompt"]),
      sv["prompt"][:50])
check("返回 styleWords 供界面展示", bool(sv.get("styleWords")), sv.get("styleWords", "")[:30])
check("重复拼接是幂等的",
      sty_mod.apply_style(sv["prompt"], "manga_bw")["prompt"] == sv["prompt"])
check("不选画风时不改提示词",
      sty_mod.apply_style("主体", None)["prompt"] == "主体")
check("画风列表带 words（前端可展示）",
      all(s.get("words") for s in sty["styles"]),
      str([s["id"] for s in sty["styles"] if not s.get("words")]))

print("== HD-2D 画风与画风转换 ==")
# HD-2D 是「像素素材 + 3D 景深光照」，不是「更精细的像素」：
# 给它配 pixelize 会把虚化和光晕一起量化成色块，3D 感直接消失。
hd = next((s for s in sty["styles"] if s["id"] == "hd2d"), None)
check("HD-2D 画风存在", hd is not None)
check("HD-2D 有独立的风格词", bool(hd and hd.get("words")), (hd or {}).get("words", "")[:40])
check("HD-2D 风格词写清像素+3D 景深",
      all(k in (hd or {}).get("words", "") for k in ("像素", "3D", "景深", "泛光")),
      (hd or {}).get("words", "")[:60])
check("HD-2D 不做自动像素化（量化会抹掉景深）", not (hd or {}).get("pixelize"))
check("HD-2D 归到「像素 · HD-2D」分组", (hd or {}).get("group") == "像素 · HD-2D",
      str((hd or {}).get("group")))
check("每个画风都有分组", all(s.get("group") for s in sty["styles"]),
      str([s["id"] for s in sty["styles"] if not s.get("group")]))
check("分组都在 STYLE_GROUPS 里",
      all(s["group"] in sty_mod.STYLE_GROUPS for s in sty["styles"]),
      str({s["group"] for s in sty["styles"]} - set(sty_mod.STYLE_GROUPS)))
check("HD-2D 出现在角色/场景/视差的推荐里",
      all("hd2d" in rec.get(k, []) for k in ("角色", "场景", "视差")),
      str({k: rec.get(k) for k in ("角色", "场景", "视差")}))
check("场景类首选 HD-2D", (rec.get("场景") or [""])[0] == "hd2d", str(rec.get("场景")))
check("HD-2D 场景模板默认画风跟着换",
      api_tpls["scene_hd2d_ref"].get("defaultStyle") == "hd2d",
      str(api_tpls["scene_hd2d_ref"].get("defaultStyle")))

# 画风转换指令：必须以「重绘成…」开头。写成「保持…完全不变，只改变画风」时，
# 编辑底座会先读到「什么都别改」，整张图原样返回——观感就是转换没生效。
from backend.app.pipelines import ai_edit  # noqa: E402

_rp = ai_edit._restyle_prompt(sty_mod.get_style("hd2d"))
check("转换指令以重绘动作开头", _rp.startswith("把这张图整体重绘成"), _rp[:20])
check("转换指令带上目标画风名", "HD-2D 幻想" in _rp)
check("转换指令带上该画风的风格词", sty_mod.get_style("hd2d")["prompt"] in _rp)
check("转换指令用「沿用原图」做内容锚点", "沿用原图" in _rp)
check("转换指令不写压制性措辞（完全不变 / 保持不变 / 只改变）",
      not any(w in _rp for w in ("完全不变", "保持不变", "只改变")), _rp[-40:])
check("restyle 支持关掉自动像素化",
      "pixelize" in inspect.signature(ai_edit.restyle).parameters)
check("restyle 支持 max_side（转换后放回原尺寸）",
      "max_side" in inspect.signature(ai_edit.restyle).parameters)

check("转换：未知画风 → 400",
      client.post("/api/pipeline/restyle", data={"style": "nope"}).status_code == 400)
_r = client.post("/api/pipeline/restyle", data={"style": "hd2d"})
check("转换：没有输入图 → 400", _r.status_code == 400, _r.status_code)
check("转换：缺图报错说清两种输入方式",
      "refAssetId" in str(_r.json().get("detail", "")) or "上传" in str(_r.json().get("detail", "")),
      str(_r.json().get("detail", ""))[:60])
_r = client.post("/api/pipeline/restyle", data={"style": "hd2d", "refAssetId": "nope"})
check("转换：资产不存在 → 404", _r.status_code == 404, _r.status_code)

print("== 人物像素化（三档程度 + 背景处理） ==")
# 三档变的是「造型语言」（头身比/眼睛/简化程度），不是颗粒大小；
# 颗粒密度用「横向格数」描述，与输入分辨率无关。
from backend.app.pipelines import charpixel as cp  # noqa: E402

r = client.get("/api/pipeline/char-pixel/levels")
check("档位接口可用", r.status_code == 200)
lvj = r.json()
check("三档齐全且顺序固定",
      [x["id"] for x in lvj["levels"]] == ["fine", "mid", "chibi"],
      str([x["id"] for x in lvj["levels"]]))
check("档位密度递减（写实最细、Q 版最粗）",
      lvj["levels"][0]["grid"] > lvj["levels"][1]["grid"] > lvj["levels"][2]["grid"],
      str([x["grid"] for x in lvj["levels"]]))
check("档位带头身比与色数",
      all(x.get("heads") and x.get("colors") for x in lvj["levels"]))
check("背景三选一（保留/绿幕/透明）",
      {b["id"] for b in lvj["backgrounds"]} == {"keep", "green", "alpha"},
      str([b["id"] for b in lvj["backgrounds"]]))
check("档位提示词以重绘动作开头",
      all(x_id and cp.get_level(x_id)["prompt"].startswith("把这张照片重绘成")
          for x_id in ("fine", "mid", "chibi")))
_cp_chibi = cp.get_level("chibi")["prompt"]
check("Q 版档写明 2 头身与大眼",
      "2 头身" in _cp_chibi and "眼睛大而圆" in _cp_chibi)
check("三档都要求粗黑描边（像素画的辨识度一半在这里）",
      all("粗黑轮廓线" in cp.get_level(x)["prompt"] for x in ("fine", "mid", "chibi")))
check("三档都要求平涂、禁止渐变",
      all("平涂" in cp.get_level(x)["prompt"] and "渐变" in cp.get_level(x)["prompt"]
          for x in ("fine", "mid", "chibi")))

_p_keep = cp.build_prompt(cp.get_level("mid"), "戴红帽子")
check("生成指令不含背景色（绿幕/透明在出图后本地做）",
      "#00FF00" not in _p_keep and "绿幕" not in _p_keep, _p_keep[-50:])
check("生成指令锚定构图沿用原图", "构图沿用原图" in _p_keep)
check("用户补充要求拼在指令末尾", "补充要求：戴红帽子" in _p_keep)
check("留空不拼空补充段", "补充要求" not in cp.build_prompt(cp.get_level("mid"), ""))
try:
    cp.get_level("nope")
    _lv_err = ""
except ValueError as e:
    _lv_err = str(e)
check("未知档位报错列出合法值", "chibi" in _lv_err, _lv_err[:60])

# 横向格数 → 像素块：1024 宽 128 格 ≈ 8px 一格；分辨率无关是设计核心
check("格数换算：1024/128 → 8px", cp.block_for(1024, 128) == 8, cp.block_for(1024, 128))
check("格数换算：512/128 → 4px（同档位密度一致）", cp.block_for(512, 128) == 4,
      cp.block_for(512, 128))
check("格数换算：极小图不炸（至少 1px）", cp.block_for(16, 128) == 1, cp.block_for(16, 128))

_src = Image.new("RGB", (128, 128), (200, 120, 90))
_d = ImageDraw.Draw(_src)
_d.rectangle([40, 30, 90, 100], fill=(30, 60, 200))
_px_out, _px_meta = cp.apply_pixels(_src, 64, 8)
check("apply_pixels：横向格数命中请求值",
      abs(_px_meta["grid"][0] - 64) <= 1, str(_px_meta["grid"]))
check("apply_pixels：色数被限制", _px_meta["colors"] <= 8, str(_px_meta["colors"]))
check("apply_pixels：尺寸不变", _px_out.size == _src.size)

# 背景：AI 只画角色，绿幕/透明都由本地抠图 + 合成产出——
# 让 AI 直接画绿底会把角色染色（实测肤色变橄榄绿、白 T 变柠檬绿）。
_green = Image.new("RGB", (96, 96), (0, 255, 0))
_d = ImageDraw.Draw(_green)
_d.rectangle([24, 24, 71, 71], fill=(180, 40, 40))
_cut, _cut_meta = cp.cut_out(_green)
check("cut_out：走抠图链（u2net/grabcut/key 任一）",
      _cut_meta["method"] in ("u2net", "grabcut", "key"), _cut_meta["method"])
check("cut_out：背景被抠透", _cut.getpixel((2, 2))[3] == 0, str(_cut.getpixel((2, 2))))
check("cut_out：角色保留", _cut.getpixel((48, 48))[3] == 255, str(_cut.getpixel((48, 48))))
_comp = cp.compose_green(_cut)
check("compose_green：背景是纯 #00FF00",
      _comp.getpixel((2, 2)) == (0, 255, 0), str(_comp.getpixel((2, 2))))
check("compose_green：角色颜色不被绿幕污染",
      _comp.getpixel((48, 48)) == (180, 40, 40), str(_comp.getpixel((48, 48))))
check("compose_green：输出 RGB（绿幕素材不带 alpha）", _comp.mode == "RGB", _comp.mode)
_bgkeep, _bgkeep_meta = cp._apply_background(_green, "keep")
check("keep 背景不动原图", _bgkeep is _green and not _bgkeep_meta)

_rp_out, _rp_meta = cp.repixel(_px_out, 32, 4)
check("repixel：标记为复用原稿（没重跑 AI）", _rp_meta["reusedDraft"] is True)
check("repixel：新格数生效", _rp_meta["grid"] == 32, str(_rp_meta))

check("像素化：未知档位 → 400",
      client.post("/api/pipeline/char-pixel", data={"level": "nope"}).status_code == 400)
check("像素化：没有输入照片 → 400",
      client.post("/api/pipeline/char-pixel", data={"level": "mid"}).status_code == 400)
check("像素化：缺图报错说清两种输入方式",
      "refAssetId" in str(client.post("/api/pipeline/char-pixel",
                                      data={"level": "mid"}).json().get("detail", "")))
check("重新像素化：缺原稿 id → 400",
      client.post("/api/pipeline/char-pixel/repixel",
                  json={"draftAssetId": "", "grid": 64, "colors": 8}).status_code == 400)
check("重新像素化：原稿不存在 → 404",
      client.post("/api/pipeline/char-pixel/repixel",
                  json={"draftAssetId": "nope", "grid": 64, "colors": 8}).status_code == 404)

# 模板里写死画风 = 选择器失效。这几个是**编辑/提取类**指令，要求「参考原图、
# 画风不变」，它们本来就不参与画风切换，所以不计入、也不给默认画风。
NO_DEFAULT_STYLE = {
    "anim_action_video", "anim_inbetween", "scene_element_extract",
    "scene_parallax_layer", "tile_seamless",
}
STYLE_KW = ("像素", "卡通", "Q版", "Q 版", "日系", "赛璐璐", "写实", "厚涂",
            "水彩", "扁平", "矢量", "低多边形", "体素", "漫画", "速写", "3A")
dirty = []
for t in tpl_mod.TEMPLATES:
    if t.get("kind") == "text":
        continue
    hits = [k for k in STYLE_KW if k in t.get("prompt", "")]
    if hits and t["id"] not in NO_DEFAULT_STYLE:
        dirty.append((t["id"], hits))
check("图片模板的提示词不写死画风（画风交给选择器）", not dirty, str(dirty))

bad_ds = [t["id"] for t in tpl_mod.TEMPLATES
          if t.get("defaultStyle") and t["defaultStyle"] not in style_ids]
check("模板 defaultStyle 都是合法画风 id", not bad_ds, str(bad_ds))
no_ds = [t["id"] for t in tpl_mod.TEMPLATES
         if t.get("kind") != "text" and not t.get("defaultStyle")
         and t["id"] not in NO_DEFAULT_STYLE]
check("生成类模板都带默认画风", not no_ds, str(no_ds))

# 编辑/提取类模板（画风跟随原图）要在接口里显式标 styleLocked，前端据此隐藏画风区——
# 摆一个选了也不生效的选择器，就是「文本任务假面板」的翻版。
locked = {t["id"] for t in tpl_mod.TEMPLATES if t.get("styleLocked")}
check("styleLocked 覆盖全部编辑/提取类模板", locked == NO_DEFAULT_STYLE,
      f"locked={sorted(locked)}")
check("styleLocked 模板不带默认画风",
      not [t["id"] for t in tpl_mod.TEMPLATES
           if t.get("styleLocked") and t.get("defaultStyle")])

print("== 三视图提示词 ==")
# 背视图指令曾经把示例角色（灰白英短猫武士）的造型写死：后脑纯灰、灰耳廓、
# 红披风、手中兵器。结果任何角色转背面都会长出红披风。这里守住两条底线：
# 指令里不许出现具体外观词，外观只以参考图为准。
from backend.app.pipelines import threeview as tv  # noqa: E402

back_ins = tv._view_instruction("back")
side_ins = tv._view_instruction("side")
HARDCODED = ("红", "披风", "灰色", "后脑勺", "耳廓", "兵器", "条纹", "花纹",
             "武士刀", "猫", "尾巴", "鼻梁", "头发")
hit = [w for w in HARDCODED if w in back_ins]
check("背视图指令不写死具体造型", not hit, str(hit))
check("侧/背指令各不相同", back_ins != side_ins)
check("编辑指令都以参考图为准", all("参考图" in s for s in (back_ins, side_ins)))
check("背视图只改视角、不新增部件",
      "只改变观察角度" in back_ins and "不新增" in back_ins)
check("背视图禁止凭空添加",
      "没有的衣物" in back_ins and "一件都不要出现" in back_ins)

p = tv._view_prompt("角色设定：灰白英短猫武士，腰挎武士刀。", "back")
ps = tv._view_prompt("灰白英短猫武士，腰挎武士刀", "front",
                     sty_mod.style_words("chibi"))
check("三视图风格句在最前", ps.startswith("画风："), ps[:24])
check("三视图风格词只出现一次", ps.count("二维日系Q版卡通风格") == 1)
check("三视图无画风时不加空句", not tv._view_prompt("测试角色", "front").startswith("画风："))
check("重复的「角色设定：」前缀被去掉", p.count("角色设定：") == 1, p[:40])
check("结尾句号不重复", "。。" not in p)
check("三视角提示词各自带视角名",
      all(tv._VIEW_META[v]["cn"] in tv._view_prompt("测试角色", v) for v in tv.VIEW_ORDER))
check("三视角都带单角色约束",
      all("画面中只有一个角色" in tv._view_prompt("测试角色", v) for v in tv.VIEW_ORDER))

print("== 三视图分步（流程画布） ==")
r = client.post("/api/pipeline/threeview/step", json={"view": "side", "prompt": "x"})
check("分步：side 缺基准图 → 400", r.status_code == 400, r.status_code)
r = client.post("/api/pipeline/threeview/step", json={"view": "back", "prompt": "x"})
check("分步：back 缺基准图 → 400", r.status_code == 400, r.status_code)
r = client.post("/api/pipeline/threeview/step", json={"view": "nope"})
check("分步：未知步骤 → 400", r.status_code == 400, r.status_code)
r = client.post("/api/pipeline/threeview/step",
                json={"view": "compose", "viewAssetIds": ["a", "b"]})
check("分步：拼板张数不足 → 400", r.status_code == 400, r.status_code)
r = client.post("/api/pipeline/threeview/step", json={
    "view": "compose", "width": 240, "height": 360,
    "viewAssetIds": ["deadbeef0001", "deadbeef0002", "deadbeef0003"]})
check("分步：拼板引用不存在的资产 → 404", r.status_code == 404, r.status_code)

# 前端画布把末步叫 strip（节点键 + localStorage 键），后端规范名是 compose。
# 两边对不上就是「未知步骤：strip」——别名必须兜住，不能报未知。
r = client.post("/api/pipeline/threeview/step", json={
    "view": "strip", "width": 240, "height": 360,
    "viewAssetIds": ["deadbeef0001", "deadbeef0002", "deadbeef0003"]})
check("分步：strip 是 compose 的别名（不报未知步骤）",
      r.status_code == 404 and "未知步骤" not in str(r.json().get("detail", "")),
      f"{r.status_code} {r.text[:80]}")
r = client.post("/api/pipeline/threeview/step",
                json={"view": "strip", "viewAssetIds": ["a", "b"]})
check("分步：strip 别名仍校验张数",
      r.status_code == 400 and "3 张" in str(r.json().get("detail", "")), r.text[:80])
r = client.post("/api/pipeline/threeview/step", json={"view": "nope"})
check("分步：未知步骤的报错列出合法值",
      r.status_code == 400 and "compose" in str(r.json().get("detail", "")), r.text[:90])

check("分步：run_front / run_derived 是独立入口",
      callable(tv.run_front) and callable(tv.run_derived)
      and "engines" in inspect.signature(tv.run_front).parameters)

# 真拼一次（纯 CPU，不烧 GPU）：三张纯色块图 → 检查格数与尺寸
tv_ids = []
for i in range(3):
    im = Image.new("RGB", (240, 360), (27, 67, 50))
    d = ImageDraw.Draw(im)
    d.rectangle([80, 40, 160, 330], fill=(230, 230 - i * 40, 40))
    aid = storage.save_asset(im, kind="generated", name=f"tv-step-{i}")["id"]
    tv_ids.append(aid)
    created.append(aid)
r = client.post("/api/pipeline/threeview/step", json={
    "view": "compose", "width": 240, "height": 360, "viewAssetIds": tv_ids})
body = r.json() if r.status_code == 200 else {}
item = (body.get("images") or [{}])[0]
check("分步：三张 → 拼板生成", r.status_code == 200, r.status_code)
# 单格对齐到 16 → (240, 352)；拼板 = 3 格 + 2 条 12px 缝
check("分步：拼板尺寸 = 3 格 + 2 缝",
      item.get("width") == 744 and item.get("height") == 352,
      f"{item.get('width')}x{item.get('height')}")
check("分步：拼板 meta 含三视角",
      len(((body.get("meta") or {}).get("compose") or {}).get("views") or []) == 3)
if item.get("id"):
    created.append(item["id"])

print("== 移植自 D:/code/gen：全局色板 / 一致性指标 / 光流插帧 / 色相去污 ==")
# 来源：game-asset-video-sprites（sprite-postprocess.py：全局锁色 + BOX 缩放 + metrics）、
# h3-character-studio（key_pixels：按色相抓暗部溢色）、FrameRonin（双向光流插帧）。
from backend.app.pipelines import inbetween as ib_pipe  # noqa: E402
from backend.app.pipelines import metrics as met_pipe  # noqa: E402
from backend.app.pipelines import palette as pal_pipe  # noqa: E402
from backend.app.pipelines import pixelize as pixelize_pipe  # noqa: E402
from backend.app.pipelines import sprite_loop as sl_pipe  # noqa: E402

# ---- 全局色板：限量素材的 k 收敛到真实色数，且色板不重复 ----
_cp_imgs = []
for _col in [(200, 100, 50), (20, 40, 60), (240, 240, 240)]:
    _im = Image.new("RGB", (32, 32), _col)
    _d = ImageDraw.Draw(_im)
    _d.rectangle([8, 8, 24, 24], fill=(10, 10, 10))
    _cp_imgs.append(_im)
_pal = pal_pipe.extract_palette(_cp_imgs, k=6)
check("色板无重复色（k 收敛到真实颜色数）", len(set(_pal)) == len(_pal), str(_pal))
check("色板取到四种主色", len(_pal) == 4, str(_pal))
_locked = pal_pipe.apply_palette(_cp_imgs[0], _pal)
check("锁色后只剩色板内颜色",
      not (set(_locked.convert("RGB").getdata()) - set(_pal)))
_px_out, _px_meta = pixelize_pipe.pixelize(_cp_imgs[0], shared_palette=_pal, downsample="box")
check("pixelize 支持共享色板", _px_meta["palette"] == "shared"
      and _px_meta["sharedPalette"] == len(_pal), str(_px_meta.get("palette")))
check("pixelize 支持 BOX 降采样", _px_meta["downsample"] == "box")
check("pixelize 默认仍是最近邻（不改变既有行为）",
      pixelize_pipe.pixelize(_cp_imgs[0])[1]["downsample"] == "nearest")

# ---- 一致性指标 ----
_cp_m = met_pipe.sheet_metrics(_cp_imgs, _pal)
check("指标含首尾衔接与帧间差",
      all(k in _cp_m for k in ("seam_ratio", "step_mean", "step_max")), str(_cp_m)[:120])
check("锁色后色板外颜色为 0", _cp_m.get("off_palette_colors_max") == 0,
      str(_cp_m.get("off_palette_colors_max")))
check("单帧不崩（只返回基础信息）", "note" in met_pipe.sheet_metrics([_cp_imgs[0]]))
check("结论可读且可行动", isinstance(met_pipe.metrics_verdict(_cp_m), list))

# ---- compose：锁色 + 指标进 meta ----
_cp_frames = []
for _i, _col in enumerate([(200, 100, 50), (210, 110, 60), (190, 90, 40)]):
    _im = Image.new("RGB", (96, 96), (30, 30, 40))
    _d = ImageDraw.Draw(_im)
    _d.ellipse([24 + _i * 6, 20, 72 + _i * 6, 80], fill=_col)
    _cp_frames.append(_im)
_sheet, _frames, _index, _cmeta, _review = sl_pipe.compose(
    _cp_frames, cols=3, size=64, matte=False, colors=8)
check("compose 锁色后无调色板外颜色",
      _cmeta["metrics"]["off_palette_colors_max"] == 0,
      str(_cmeta["metrics"].get("off_palette_colors_max")))
check("compose 报告色板与指标",
      _cmeta.get("palette", {}).get("colors", 0) >= 2 and "metrics" in _cmeta
      and "metricsVerdict" in _cmeta, str(_cmeta.get("palette")))

# ---- 光流插帧：符号错了会静默变差，用平移真值钉住 ----
# 用仓库内的人像示例图（200²，随 char_pixel_samples 一起分发）做平移：
# 纯合成图没有亮度结构，Farneback 算不出位移（那是光流的正常前提，不是 bug）。
_src_path = Path(__file__).resolve().parents[1] / "data" / "char_pixel_samples" / "_source.png"
if _src_path.is_file():
    _base = np.asarray(Image.open(_src_path).convert("RGB").resize((200, 200)))
    _shift = 24
    _a = Image.fromarray(_base, "RGB")
    _b = Image.fromarray(
        np.concatenate([np.zeros_like(_base[:, :_shift]), _base[:, :-_shift]], axis=1), "RGB")
    _ref = np.concatenate([np.zeros_like(_base[:, :_shift // 2]), _base[:, :-_shift // 2]], axis=1)
    _mid, _fmeta = ib_pipe.flow_mid(_a, _b, 0.5)
    _mae_flow = float(np.abs(np.asarray(_mid).astype(float) - _ref.astype(float)).mean())
    _mae_blend = float(np.abs(np.asarray(Image.blend(_a, _b, 0.5)).astype(float)
                            - _ref.astype(float)).mean())
    check("光流算出位移量", abs(_fmeta["meanMotionPx"] - _shift) < 6, str(_fmeta["meanMotionPx"]))
    check("光流中间帧明显优于交叉淡化", _mae_flow < _mae_blend * 0.6,
          f"flow {_mae_flow:.1f} vs blend {_mae_blend:.1f}")
    check("remap 用最近邻（像素素材不产生插值色）", _fmeta["nearest"] is True)
else:
    check("光流中间帧优于交叉淡化（示例图缺失，跳过）", True, "no sample")

# ---- 色相去污：暗部溢出（RGB 距离抓不到的那类）----
_gr = Image.new("RGB", (64, 64), (0, 255, 0))
_d = ImageDraw.Draw(_gr)
_d.rectangle([16, 16, 47, 47], fill=(200, 160, 140))
_d.rectangle([45, 18, 46, 45], fill=(20, 80, 30))     # 贴边的深绿溢出
_cut2, _spill_meta = matte_pipe.connected_matte(
    _gr, tolerance=60, feather=0, sample="topleft", method="key", hard=True)
check("暗绿溢出被洗成角色色", _cut2.getpixel((45, 30))[:3] == (200, 160, 140),
      str(_cut2.getpixel((45, 30))))
check("洗后无残留键色像素", _spill_meta["residualKeyPixels"] == 0,
      str(_spill_meta["residualKeyPixels"]))
check("残留键色进了 meta（量化前先量污染）", "residualKeyRatio" in _spill_meta)
_wht = Image.new("RGB", (32, 32), (250, 250, 250))
_d = ImageDraw.Draw(_wht)
_d.rectangle([6, 6, 25, 25], fill=(200, 160, 140))
_cut3, _wmeta = matte_pipe.connected_matte(
    _wht, tolerance=60, feather=0, sample="topleft", method="key", hard=True)
check("白底不走色相判定（不当成溢色乱洗）",
      _cut3.getpixel((16, 16))[:3] == (200, 160, 140), str(_cut3.getpixel((16, 16))))

print("== 视频生成工作流（Wan2.2 TI2V-5B） ==")
from backend.app import workflows as wf_mod  # noqa: E402
from backend.app.providers import diffusers_local  # noqa: E402
from backend.app.providers.diffusers_local import DiffusersProvider, detect_family  # noqa: E402

r = client.get("/api/workflows")
check("视频工作流已注册", "animation_video" in [w["id"] for w in r.json()["workflows"]],
      str([w["id"] for w in r.json()["workflows"]]))
_vwf = wf_mod.get_workflow("animation_video")
check("视频工作流步骤链完整（锚点 → 视频 → 去重 → 插帧 → 规整 → 拼表 → 导出）",
      [s["type"] for s in _vwf["steps"]] == ["generate", "video", "dedup", "blend",
                                             "sprites", "compose", "godot"],
      str([s["type"] for s in _vwf["steps"]]))
_vstep = next(s for s in _vwf["steps"] if s["type"] == "video")
check("视频步骤默认 81 帧 / 24fps / 480×832",
      _vstep["numFrames"] == 81 and _vstep["fps"] == 24
      and _vstep["width"] == 480 and _vstep["height"] == 832,
      str({k: _vstep.get(k) for k in ("numFrames", "fps", "width", "height")}))
check("模型族识别：wan 目录名",
      detect_family("models/wan2.2-ti2v-5b") == "wan", detect_family("models/wan2.2-ti2v-5b"))
# _load 的分发：少了 wan 分支会在真机上报「不支持的模型」，这里用替身锁住路由
_sentinel = object()
_orig_load_wan = DiffusersProvider._load_wan
DiffusersProvider._load_wan = lambda self, model_id: _sentinel
try:
    _got = DiffusersProvider({"model": "models/wan2.2-ti2v-5b"})._load("models/wan2.2-ti2v-5b")
finally:
    DiffusersProvider._load_wan = _orig_load_wan
check("_load 把 wan 分到 _load_wan（不是报不支持的模型）", _got is _sentinel, str(type(_got)))

# 管线类必须是自己声明 image 入参的那支：仓库 model_index 写的是 WanPipeline
# （纯 T2V），照它加载会在真机上报 unexpected keyword argument 'image'。
from diffusers import WanImageToVideoPipeline as _WanI2V  # noqa: E402

check("Wan 用 I2V 管线（__call__ 接受 image）",
      "image" in inspect.signature(_WanI2V.__call__).parameters)
# transformer 48 输入通道（16 噪声 + 32 图像 latent）是 I2V 的硬指标
_wan_root = Path(__file__).resolve().parents[1] / "models" / "wan2.2-ti2v-5b"
if (_wan_root / "transformer" / "config.json").is_file():
    _wcfg = json.loads((_wan_root / "transformer" / "config.json").read_text())
    check("Wan transformer in_channels=48（图像 latents 已并入）",
          _wcfg.get("in_channels") == 48, _wcfg.get("in_channels"))
else:
    check("Wan transformer in_channels=48（未下载权重，跳过）", True)

# Wan I2V 默认 output_type="np"（返回 ndarray 而非 PIL），拿 [0] 当 PIL 用会报
# 'numpy.ndarray' object has no attribute 'convert'。归一化函数收全部结构。
from backend.app.providers.diffusers_local import _frames_to_pil  # noqa: E402

_a = Image.new("RGB", (8, 10), (10, 200, 30))
_b = Image.new("RGB", (8, 10), (30, 40, 250))
_np_hwc = np.stack([np.asarray(_a), np.asarray(_b)])          # [b][f][h][w][c]
_np_chw = np.transpose(_np_hwc, (0, 3, 1, 2))                # [b][f][c][h][w]
for _name, _src in (("PIL 列表", [[_a, _b]]), ("PIL 平铺", [_a, _b]),
                    ("numpy HWC", [_np_hwc]), ("numpy CHW", [_np_chw]),
                    ("裸 ndarray", _np_hwc)):
    _got = _frames_to_pil(_src)
    check(f"帧归一化 · {_name} → PIL RGBA 且尺寸/颜色正确",
          len(_got) > 0 and _got[0].mode == "RGBA"
          and _got[0].size == (8, 10)
          and _got[0].getpixel((0, 0)) == (10, 200, 30, 255),
          f"n={len(_got)} mode={_got[0].mode if _got else '-'} "
          f"size={_got[0].size if _got else '-'}")
# 小尺寸 HWC 图（8×10×3）不能被误当 CHW 转置：只用 shape[0] 判会转成 (3,8,10)
check("帧归一化 · 小尺寸 HWC 不被误转置",
      _frames_to_pil([np.asarray(_a)])[0].size == (8, 10),
      str(_frames_to_pil([np.asarray(_a)])[0].size))
# 灰度 2D 要真的复制成 3 通道；补一维 (h,w,1) PIL 不认
check("帧归一化 · 灰度图补成三通道",
      _frames_to_pil([np.asarray(_a.convert("L"))])[0].mode == "RGBA")

# _load_component_fp8 漏 import torch 时会在真机上退化成「fp8 快速路径失败
# （name 'torch' is not defined）」，静默走完整 bf16 加载（慢一倍、常驻翻倍）。
# 这里用不存在的目录触发早期异常：如果 torch 没导入，抛的是 NameError。
try:
    DiffusersProvider({})._load_component_fp8(
        "models/__no_such_dir__", "transformer", object, "test.tag")
    _fp8_exc = ""
except NameError as e:
    _fp8_exc = f"NameError:{e}"
except Exception as e:                       # 目录不存在 → OSError / ValueError 均可
    _fp8_exc = type(e).__name__
check("_load_component_fp8 用得到 torch（不抛 NameError）",
      not _fp8_exc.startswith("NameError"), _fp8_exc)

# ---- fp8 Linear 的 dtype 布局 ----
# diffusers 用 next(iter(module.parameters())).dtype 推断计算精度（Wan 有两处硬依赖：
# transformer_wan 的 time_embedder_dtype、pipeline 的 transformer_dtype）。
# 若 fp8 权重自己是 Parameter，时间步张量会被转成 fp8 —— e4m3 上限 448、
# 时间步 0-1000 直接溢出，报的是 "mul_cuda not implemented for Float8_e4m3fn"，
# 与真正的量化问题毫无关联，极难定位。布局必须是：fp8 放 buffer w_fp8，
# weight 置成 0 字节 bf16 Parameter。
import torch  # noqa: E402  （fp8 Linear 的 dtype 布局测试要用）
from backend.app.providers.diffusers_local import (  # noqa: E402
    _get_fp8_linear_cls as _fp8_cls, _wrap_fp8_weight as _wrap8)

if torch.cuda.is_available():

    def _mk_fp8_linear(bias_=True):
        torch.manual_seed(0)
        lin = torch.nn.Linear(128, 64, bias=bias_)
        ref = lin.weight.data.clone().cuda()
        w = lin.weight.data.cuda()
        s = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12) / 448.0
        lin.weight = torch.nn.Parameter(
            ((w / s).clamp_(-448, 448).to(torch.float8_e4m3fn)).cpu(), requires_grad=False)
        lin.register_buffer("w_scale", s.squeeze(1).float())   # 故意留在 CPU
        lin.__class__ = _fp8_cls()
        _wrap8(lin)
        return lin, ref

    _lin8, _ref8 = _mk_fp8_linear()
    check("fp8 权重放 buffer w_fp8（不是 Parameter）",
          _lin8.w_fp8.dtype == torch.float8_e4m3fn, str(_lin8.w_fp8.dtype))
    check("weight 是 0 字节 bf16 Parameter（dtype 探针看到 bf16）",
          _lin8.weight.dtype == torch.bfloat16 and _lin8.weight.numel() == 0,
          f"{_lin8.weight.dtype} numel={_lin8.weight.numel()}")
    check("diffusers 的 dtype 探针返回 bf16（不会把时间步转成 fp8）",
          next(iter(_lin8.parameters())).dtype == torch.bfloat16,
          str(next(iter(_lin8.parameters())).dtype))
    # 真实失败模式：时间步 0-1000 过 fp8 Linear
    _ts = torch.tensor([[0.0, 500.0, 999.9, 1000.0]], device="cuda")
    _out_ts = _lin8(torch.nn.functional.pad(_ts, (0, 124)))
    check("时间步 0-1000 能过 fp8 Linear 且有限（e4m3 上限 448，不溢出）",
          bool(torch.isfinite(_out_ts).all()), str(_out_ts.dtype))
    # 量化误差应在 fp8 预期范围内（实测 ~0.11）
    _x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    _err = float((_lin8(_x).float()
                  - torch.nn.functional.linear(_x.float(), _ref8)).abs().max())
    check("fp8 前向与 bf16 误差在预期内（<0.5）", _err < 0.5, round(_err, 3))
    # bias 为 None 与 fp32 输入都要能过（w_scale/bias 常驻 CPU 会撞 device mismatch）
    _lin8_nb, _ = _mk_fp8_linear(bias_=False)
    check("fp8 Linear 无 bias 可用",
          tuple(_lin8_nb(torch.randn(2, 128, device="cuda",
                                      dtype=torch.bfloat16)).shape) == (2, 64))
    check("fp8 Linear 吃 fp32 输入（时间步探针可能是 fp32）",
          tuple(_lin8(torch.zeros(2, 128, device="cuda")).shape) == (2, 64))
    # 缓存路径：state_dict 里 w_fp8 存 fp8、空 weight 存空张量，safetensors 能往返
    _sd = _lin8.state_dict()
    check("state_dict 键含 w_fp8 / w_scale / 空 weight",
          set(("w_fp8", "w_scale", "weight")) <= set(_sd)
          and _sd["w_fp8"].dtype == torch.float8_e4m3fn
          and _sd["weight"].numel() == 0,
          str(sorted(_sd)))
    import tempfile as _tf
    from safetensors.torch import load_file as _lf, save_file as _sf

    _dir = _tf.mkdtemp()
    _sf({k: (v.data if hasattr(v, "data") else v).detach().contiguous()
         for k, v in _sd.items()}, os.path.join(_dir, "m.safetensors"))
    _back = _lf(os.path.join(_dir, "m.safetensors"))
    check("fp8 缓存 safetensors 往返（fp8 原样存回）",
          _back["w_fp8"].dtype == torch.float8_e4m3fn, str(_back["w_fp8"].dtype))
else:
    check("fp8 Linear 布局（无 GPU，跳过）", True)
check("wan 与其他大族互斥驻留",
      "wan" in diffusers_local._BIG_FAMILIES, str(diffusers_local._BIG_FAMILIES))

# 视频步骤的依赖与底座校验（都不碰 GPU）
_run = wf_mod.create_run("animation_video", "灰白英短猫武士")
_r2 = client.post(f"/api/runs/{_run['id']}/steps/video/run", json={})
check("视频步骤：上一步未完成 → 400",
      _r2.status_code == 400 and "上一步" in str(_r2.json().get("detail", "")),
      f"{_r2.status_code} {_r2.text[:60]}")
check("视频步骤在 run 里可见",
      any(s["id"] == "video" for s in client.get(f"/api/runs/{_run['id']}").json()["steps"]))

import asyncio as _aio  # noqa: E402
_prov = DiffusersProvider({})
try:
    _aio.get_event_loop().run_until_complete(
        _prov.generate_video("测试", Image.new("RGB", (64, 64), (10, 10, 10))))
    _vmsg = ""
except RuntimeError as e:
    _vmsg = str(e)
except Exception as e:  # 事件循环关闭等：忽略，只看能否走到族校验
    _vmsg = ""
check("非视频底座调 generate_video → 明确报错并给出下载指引",
      "fetch_wan22" in _vmsg or "视频底座" in _vmsg, _vmsg[:80])

print("== 体检 / 有界 JSON / 启动恢复（移植自 FrameBaker 与 h3）==")
_d = client.get("/api/doctor")
_dbody = _d.json() if _d.status_code == 200 else {}
check("GET /api/doctor 200", _d.status_code == 200, _d.status_code)
check("doctor 返回结构化检查项",
      isinstance(_dbody.get("checks"), list) and len(_dbody["checks"]) >= 5
      and all({"id", "ok", "label", "detail"} <= set(c) for c in _dbody["checks"]),
      _dbody.get("summary", ""))

# 有界 JSON（FrameBaker shared/json.ts）：NaN / 无限 / 超深嵌套必须被拒，
# 不能流进 run JSON——json.dumps 会写出严格解析器拒绝的 NaN 字面量。
# 注意 httpx 的 json= 自己也用 allow_nan=False，发不出 NaN，所以这里用
# 原始 body 直抵服务端解析（模拟 curl 手写 body 的调用方）
from backend.app.jsonlimits import validate_bounded as _vb  # noqa: E402

for _name, _bad in (("NaN", float("nan")), ("Infinity", float("inf"))):
    try:
        _vb({"cfg": _bad})
        _unit_ok = False
    except ValueError:
        _unit_ok = True
    check(f"validate_bounded 拒绝 {_name}", _unit_ok)

_raw = b'{"input": "\xe6\x9c\x89\xe7\x95\x8c\xe6\xb5\x8b\xe8\xaf\x95", "params": {"cfg": NaN}}'
_r = client.post("/api/workflows/scene/runs", content=_raw,
                 headers={"Content-Type": "application/json"})
check("params 含 NaN → 400/422", _r.status_code in (400, 422), f"{_r.status_code}")
_deep = {"a": 1}
for _ in range(40):
    _deep = {"n": _deep}
_r = client.post("/api/workflows/scene/runs", json={"input": "有界测试", "params": _deep})
check("params 超深嵌套 → 400/422", _r.status_code in (400, 422), f"{_r.status_code}")

# 启动恢复（FrameBaker queue.ts）：running 步骤在「新进程」里应被标回 failed。
# 这里直接调 recover_interrupted_runs() 模拟启动那一刻
_r3 = wf_mod.create_run("animation_video", "恢复测试")
_p = Path("data/runs") / f"{_r3['id']}.json"
_st = json.loads(_p.read_text(encoding="utf-8"))
_st["steps"][1]["status"] = "running"
_p.write_text(json.dumps(_st), encoding="utf-8")
_n = wf_mod.recover_interrupted_runs()
_st2 = json.loads(_p.read_text(encoding="utf-8"))
check("启动恢复：running → failed", _n >= 1 and _st2["steps"][1]["status"] == "failed",
      f"recovered={_n}")
_p.unlink(missing_ok=True)

print("== 进度总线 / SSE / 取消（移植自 FrameBaker 的队列+广播）==")
from backend.app import progress as prog_mod  # noqa: E402

_tid = prog_mod.register()
prog_mod.report(_tid, 3, 40, "第 3/40 步")
_q = prog_mod.subscribe(_tid)
_first = _q.get_nowait()
check("订阅即补发当前进度", _first["type"] == "progress" and _first["step"] == 3
      and _first["total"] == 40, str(_first))
prog_mod.report(_tid, 40, 40, "第 40/40 步")
_second = _q.get_nowait()
check("report 推送增量事件", _second["step"] == 40, str(_second))
prog_mod.finish(_tid, ok=True)
_done_ev = _q.get_nowait()
check("finish 发 done 终态", _done_ev["type"] == "done", str(_done_ev))
prog_mod.unsubscribe(_tid, _q)

_tid2 = prog_mod.register()
check("cancel 已注册任务", prog_mod.cancel(_tid2) is True)
try:
    prog_mod.check_cancel(_tid2)
    _cancel_raised = False
except prog_mod.CancelledError:
    _cancel_raised = True
check("取消后 check_cancel 抛 CancelledError", _cancel_raised)
check("cancel 已终态任务返回 False", prog_mod.cancel(_tid) is False)
prog_mod.finish(_tid2, ok=False, error="已取消")  # 真实流程里由 guarded 收尾
check("sweep 清理终态任务", prog_mod.sweep(ttl=-1) >= 2)

# SSE 端点：register→finish 后拉流，应至少收到 done 事件
_tid3 = prog_mod.register()
prog_mod.report(_tid3, 1, 9, "第 1/9 步")
prog_mod.finish(_tid3, ok=True)
_r = client.get(f"/api/tasks/{_tid3}/stream")
_events = []
for _line in _r.text.splitlines():
    if _line.startswith("data: "):
        try:
            _events.append(json.loads(_line[6:]))
        except Exception:
            pass
check("SSE 流返回进度事件", _r.status_code == 200
      and any(e["type"] == "progress" for e in _events)
      and any(e["type"] == "done" for e in _events), f"{len(_events)} events")
check("SSE 未知任务 → 404", client.get("/api/tasks/nosuchtask999/stream").status_code == 404)
check("取消端点：不存在任务 ok=False",
      client.post("/api/tasks/nosuchtask999/cancel").json().get("ok") is False)
_tid4 = prog_mod.register()
_r = client.post(f"/api/tasks/{_tid4}/cancel")
check("取消端点：已注册任务 ok=True", _r.status_code == 200 and _r.json().get("ok") is True)

print("== 清理测试资产 ==")
cleanup_assets(verbose=True)

print(f"\n结果：{ok} 通过 / {fail} 失败")
raise SystemExit(1 if fail else 0)
