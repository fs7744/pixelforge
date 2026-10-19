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


print("== 清理测试资产 ==")
removed = 0
for aid in created:
    if client.delete(f"/api/assets/{aid}").status_code == 200:
        removed += 1
check("测试资产已清理", removed == len(created), f"{removed}/{len(created)}")

print(f"\n结果：{ok} 通过 / {fail} 失败")
raise SystemExit(1 if fail else 0)
