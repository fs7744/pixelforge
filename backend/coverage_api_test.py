"""逐路由覆盖测试：31 个路由里所有「不需要加载模型」的端点 + 错误路径。

分工：
- 本脚本：抠图 / 像素化 / 缩放 / 镜像延展 / 无缝平铺 / 分表 / 合表 /
  确定性去水印 / Provider 测试与卸载 / 资产删除 / 设置回写 / run 错误路径 /
  参考图生成的输入校验（无图、超 4 张、坏资产 id、非图片）。
- e2e_gpu_test.py：/api/generate、工作流步骤执行、/api/pipeline/restyle。
- e2e_gpu_gap_test.py：/api/generate/ref 的真实出图（单图 + 多图）。
- test_ai_pipelines.py：watermark/ai、seamless/ai、restyle 三个管线的真实出图。
- /api/pipeline/threeview：三视图端点（需 GPU，见 e2e_gpu_test 说明）。

用法：PYTHONPATH=. python backend/coverage_api_test.py
"""
import inspect
import io
import zipfile
from pathlib import Path

import numpy as np

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.app.main import app
from backend.app import storage

client = TestClient(app)
ok = 0
fail = 0
created = []


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


print("== 模板库：提示词纯净度 ==")
from backend.app import templates as tpl_mod

r = client.get("/api/templates")
check("模板接口可用", r.status_code == 200)
api_tpls = {t["id"]: t for t in r.json()["templates"]}

# 混进 prompt 的「给人看的写作说明」——出现即污染（这些词只会误导模型）
DIRTY = ("替换", "示例", "无需", "管线", "勾选", "填写", "注意：", "如：",
         "花括号", "提示词里", "占位符")
bad_placeholder, dirty_hits, missing_notes, bad_kind = [], [], [], []
for t in tpl_mod.TEMPLATES:
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

print("== 清理测试资产 ==")
removed = 0
for aid in created:
    if client.delete(f"/api/assets/{aid}").status_code == 200:
        removed += 1
check("测试资产已清理", removed == len(created), f"{removed}/{len(created)}")

print(f"\n结果：{ok} 通过 / {fail} 失败")
raise SystemExit(1 if fail else 0)
