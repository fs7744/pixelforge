"""补齐最后 4 条未做 HTTP 实测的路由（需 GPU）。

覆盖：
1. POST /api/runs/{id}/run-all            场景工作流全部步骤（Z-Image）
2. POST /api/pipeline/threeview           三视图拼板（Z-Image + Qwen-Edit×2）
3. POST /api/pipeline/watermark/ai        指令式去水印（Qwen-Edit）
4. POST /api/pipeline/seamless/ai         指令式左右续画（Qwen-Edit）

顺序按模型族排：先跑只吃 Z-Image 的 run-all，再进三视图（中途换出到
Qwen-Edit），最后两个编辑端点复用已驻留的 Qwen-Edit，全程只换出一次。

用法：PYTHONPATH=. python backend/e2e_gpu_gap_test.py
"""
import io
import time

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.app.main import app
from backend.app import storage

client = TestClient(app)
ok = 0
fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name} {detail}", flush=True)
    else:
        fail += 1
        print(f"  FAIL  {name} {detail}", flush=True)


def img_bytes(path):
    b = io.BytesIO()
    Image.open(path).save(b, "PNG")
    b.seek(0)
    return b


print("== 1. run-all（场景工作流全部步骤）==", flush=True)
run = client.post("/api/workflows/scene/runs",
                  json={"input": "雪山脚下的日式小镇，石板路，木屋", "params": {}}).json()
run_id = run["id"]
t0 = time.time()
r = client.post(f"/api/runs/{run_id}/run-all")
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("run-all 200", r.status_code == 200, str(r.json())[:120] if r.status_code != 200 else "")
if r.status_code == 200:
    body = r.json()
    states = body.get("steps") or []
    done = [s for s in states if s.get("status") == "done"]
    check("所有步骤完成", len(done) == len(states) and len(states) > 0,
          f"{len(done)}/{len(states)}")
    arts = [a for s in states for a in (s.get("artifacts") or [])]
    check("run-all 产出工件", len(arts) >= 2, f"{len(arts)} 个")
    final = client.get(f"/api/runs/{run_id}").json()
    check("run 重读后步骤仍为 done",
          all(s["status"] == "done" for s in final["steps"]),
          str([(s["id"], s["status"]) for s in final["steps"]]))

print("== 2. threeview（正/侧/背 + 拼板）==", flush=True)
t0 = time.time()
r = client.post("/api/pipeline/threeview", json={
    "prompt": "灰白英短猫武士，红披风，腰挎武士刀",
    "style": "chibi",
    "seed": 4242,
    "width": 512,
    "height": 768,
    "maxSide": 768,
})
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("threeview 200", r.status_code == 200, str(r.json())[:150] if r.status_code != 200 else "")
three_ids = []
if r.status_code == 200:
    body = r.json()
    imgs = body["images"]
    three_ids = [i["id"] for i in imgs]
    check("返回 4 张（拼板 + 三视图）", len(imgs) == 4, str(len(imgs)))
    names = [i["name"] for i in imgs]
    check("资产命名含四个视角", all(k in " ".join(names) for k in ("拼板", "正视图", "侧视图", "背视图")),
          str(names))
    strip = Image.open(storage.asset_path(imgs[0]))
    v0 = Image.open(storage.asset_path(imgs[1]))
    check("拼板宽度约为单视图 3 倍", strip.width >= v0.width * 2.5,
          f"{strip.width} vs {v0.width}")
    check("三视图尺寸一致", len({Image.open(storage.asset_path(i)).size for i in imgs[1:]}) == 1,
          str([Image.open(storage.asset_path(i)).size for i in imgs[1:]]))
    check("meta 记录 mode", bool(body["meta"].get("mode")), str(body["meta"].get("mode")))

print("== 3. watermark/ai（指令式去水印）==", flush=True)
src_id = three_ids[1] if three_ids else None
all_assets = client.get("/api/assets").json()["assets"]
if src_id is None:
    src_id = all_assets[0]["id"]
src_item = next(a for a in all_assets if a["id"] == src_id)
# 造一张带文字水印的图，避免拿原图空跑
base = Image.open(storage.asset_path(src_item)).convert("RGB")
d = ImageDraw.Draw(base)
d.rectangle([base.width - 150, base.height - 50, base.width - 10, base.height - 10],
            fill=(255, 255, 255))
d.rectangle([base.width - 140, base.height - 40, base.width - 60, base.height - 30],
            fill=(200, 30, 30))
buf = io.BytesIO()
base.save(buf, "PNG")
t0 = time.time()
r = client.post("/api/pipeline/watermark/ai",
                files={"file": ("wm.png", buf.getvalue(), "image/png")},
                data={"seed": "11"})
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("watermark/ai 200", r.status_code == 200,
      str(r.json())[:150] if r.status_code != 200 else "")
if r.status_code == 200:
    check("返回资产", r.json().get("asset") is not None)
    check("meta 记录 watermark_ai", r.json()["meta"].get("pipeline") == "watermark_ai",
          str(r.json()["meta"].get("pipeline")))
    if r.json().get("asset"):
        out = Image.open(storage.asset_path(r.json()["asset"])).convert("RGB")
        check("输出尺寸与原图一致", out.size == base.size, f"{out.size} vs {base.size}")

print("== 4. seamless/ai（指令式续画）==", flush=True)
scene = Image.open("data/style_samples/watercolor.png").convert("RGB")
buf = io.BytesIO()
scene.save(buf, "PNG")
t0 = time.time()
r = client.post("/api/pipeline/seamless/ai",
                files={"file": ("s.png", buf.getvalue(), "image/png")},
                data={"times": "2", "seed": "42"})
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("seamless/ai 200", r.status_code == 200,
      str(r.json())[:150] if r.status_code != 200 else "")
if r.status_code == 200:
    a = r.json()["asset"]
    check("宽度翻倍", a["width"] == scene.width * 2, f"{a['width']}")
    m = r.json()["meta"]
    check("meta 记录 pad/blend", "pad" in m and "blend" in m, f"pad={m.get('pad')} blend={m.get('blend')}")
    check("meta 记录 sourceSize", m.get("sourceSize") == [scene.width, scene.height],
          str(m.get("sourceSize")))

print("== 5. generate/ref（参考图生成：2 张参考图）==", flush=True)
# 两张内容不同的参考图：角色帧 + 场景，模拟「角色 + 场景」的组合参考
ref1 = base.resize((512, 512), Image.LANCZOS)
ref2 = scene.resize((512, 512), Image.LANCZOS)


def _png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


t0 = time.time()
r = client.post("/api/generate/ref",
                files=[("refs", ("r1.png", _png(ref1), "image/png")),
                       ("refs", ("r2.png", _png(ref2), "image/png"))],
                data={"prompt": "以第一张图的角色为主角，保持造型与配色不变；"
                                "参考第二张图的场景色调，生成该角色的正面半身像，纯色背景",
                      "steps": "20", "seed": "42", "width": "512", "height": "512"})
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("generate/ref 200", r.status_code == 200,
      str(r.json())[:200] if r.status_code != 200 else "")
if r.status_code == 200:
    j = r.json()
    m = j["meta"]
    check("meta 标记 ref2img", m.get("mode") == "ref2img", str(m.get("mode")))
    check("meta 记录 2 张参考图", m.get("refCount") == 2, str(m.get("refCount")))
    check("自动改走编辑底座", m.get("autoRouted") is True, str(m.get("autoRouted")))
    check("使用了 qwen_edit 族", "qwen" in str(m.get("model", "")).lower(), str(m.get("model")))
    check("返回资产", bool(j.get("images")), str(len(j.get("images") or [])))
    if j.get("images"):
        a = j["images"][0]
        check("输出尺寸 512x512", (a["width"], a["height"]) == (512, 512), f"{a['width']}x{a['height']}")
        check("输出文件已落盘", storage.asset_path(a).exists(), storage.asset_path(a).name)

print(f"\n结果：{ok} 通过 / {fail} 失败", flush=True)
raise SystemExit(1 if fail else 0)
