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
import io
import zipfile

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
check("重复的「角色设定：」前缀被去掉", p.count("角色设定：") == 1, p[:40])
check("结尾句号不重复", "。。" not in p)
check("三视角提示词各自带视角名",
      all(tv._VIEW_META[v]["cn"] in tv._view_prompt("测试角色", v) for v in tv.VIEW_ORDER))
check("三视角都带单角色约束",
      all("画面中只有一个角色" in tv._view_prompt("测试角色", v) for v in tv.VIEW_ORDER))

print("== 清理测试资产 ==")
removed = 0
for aid in created:
    if client.delete(f"/api/assets/{aid}").status_code == 200:
        removed += 1
check("测试资产已清理", removed == len(created), f"{removed}/{len(created)}")

print(f"\n结果：{ok} 通过 / {fail} 失败")
raise SystemExit(1 if fail else 0)
