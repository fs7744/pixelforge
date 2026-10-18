"""GPU 端到端测试：真实出图、AI 编辑、工作流步骤。

每一步都跑真实模型（Z-Image ~30s / Qwen-Edit ~190s），所以单独放一个
脚本，按"最便宜的验证覆盖最多代码路径"排序：
1. /api/generate          —— 核心出图（画风拼接、meta、存储不烧角标）
2. 场景工作流 ref 步骤     —— 工作流 _fill/_template 重写后的单帧路径
3. /api/pipeline/restyle  —— AI 编辑链路（provider.generate 签名已改）
4. /api/pipeline/watermark —— 确定性去水印（无模型）
"""
import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

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


print("== 1. /api/generate（Z-Image 真出图）==", flush=True)
t0 = time.time()
r = client.post("/api/generate", json={
    "prompt": "灰白英短猫武士，红披风，腰挎武士刀，3头身Q版",
    "style": "chibi",
    "width": 512,
    "height": 512,
    "seed": 777,
})
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("generate 200", r.status_code == 200, str(r.status_code))
body = r.json()
gen = body["images"][0]
meta = body["meta"]
check("返回 1 张 512x512", gen["width"] == 512 and gen["height"] == 512,
      f"{gen['width']}x{gen['height']}")
check("提示词为中文且拼了画风", "日系" in meta["prompt"] or "Q版" in meta["prompt"])
check("meta 无翻译字段", "translation" not in meta)
check("meta 画风 = chibi", (meta.get("style") or {}).get("id") == "chibi")
check("出图非纯色（真有内容）", True)
im = Image.open(f"data/assets/{gen['id']}.png").convert("RGB")
colors = len(set(im.getdata()))
check("图像颜色数 > 50", colors > 50, f"{colors} 色")
gen_asset_id = gen["id"]

print("== 2. 场景工作流 ref 步骤（工作流模板填充路径）==", flush=True)
run = client.post("/api/workflows/scene/runs",
                  json={"input": "雪山脚下的日式小镇，石板路，木屋", "params": {}}).json()
run_id = run["id"]
t0 = time.time()
r = client.post(f"/api/runs/{run_id}/steps/ref/run", json={"n": 1})
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("workflow ref 步骤 200", r.status_code == 200, str(r.json())[:120])
step_out = r.json()
step_state = next(x for x in step_out["steps"] if x["id"] == "ref")
arts = step_state.get("artifacts") or []
check("步骤产出工件", len(arts) >= 1, str(arts)[:60])
if arts:
    a = client.get("/api/assets").json()["assets"]
    got = next((x for x in a if x["id"] == arts[0]), None)
    check("工件已在资产库", got is not None, str(arts[0]))
    check("步骤状态 done", step_state["status"] == "done", step_state["status"])
    check("步骤 meta 记录 promptLang=zh",
          step_state.get("meta", {}).get("promptLang") == "zh", str(step_state.get("meta"))[:80])
    check("步骤提示词为中文",
          any("雪山" in str(step_state.get("meta", {}).get("prompt", "")) for _ in [0]))

print("== 3. /api/pipeline/restyle（Qwen-Edit AI 重绘）==", flush=True)
t0 = time.time()
r = client.post("/api/pipeline/restyle",
                files={"file": ("src.png", img_bytes(f"data/assets/{gen_asset_id}.png"), "image/png")},
                data={"style": "pixel_16bit", "seed": "777"})
print(f"  elapsed {time.time() - t0:.1f}s", flush=True)
check("restyle 200", r.status_code == 200, str(r.json())[:120])
rb = r.json()
check("返回资产", rb.get("asset") is not None)
if rb.get("asset"):
    check("输出尺寸有值", rb["asset"]["width"] > 0, f"{rb['asset']['width']}x{rb['asset']['height']}")
    check("meta 记录编辑模型", "model" in rb.get("meta", {}) or "seconds" in rb.get("meta", {}),
          str(rb.get("meta"))[:100])

print("== 4. /api/pipeline/watermark（确定性去水印）==", flush=True)
r = client.post("/api/pipeline/watermark",
                files={"file": ("src.png", img_bytes(f"data/assets/{gen_asset_id}.png"), "image/png")},
                data={"size": "48"})
check("watermark 200", r.status_code == 200)
check("watermark 返回 meta.applied 布尔", "applied" in r.json()["meta"], str(r.json()["meta"])[:80])

print(f"\n结果：{ok} 通过 / {fail} 失败", flush=True)
raise SystemExit(1 if fail else 0)
