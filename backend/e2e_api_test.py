"""API 端到端测试（不依赖外部启动的服务，走 in-process TestClient）。

覆盖：配置读取、模板/画风无英文残留、Provider 只有一个底座、下载角标
默认关闭/显式开启、ZIP 导出、资产删除、工作流建 run 与读取、已删除
端点确实 404。

GPU 相关（/api/generate、AI 编辑、工作流步骤）由 e2e_gpu_test.py 单独跑。
"""
import io
import json
import zipfile

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app import storage

client = TestClient(app)
ok = 0
fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name} {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name} {detail}")


print("== 基础配置 ==")
r = client.get("/api/health")
check("GET /api/health", r.status_code == 200, r.json())

tpls = client.get("/api/templates").json()["templates"]
check("模板无 promptEn", all("promptEn" not in t for t in tpls), f"共 {len(tpls)} 个")

styles = client.get("/api/styles").json()["styles"]
check("画风列表无 en 字段", all("en" not in s for s in styles), f"共 {len(styles)} 个")
from backend.app.styles import STYLES
check("内部画风仍有中文风格词", all(x.get("prompt") for x in STYLES))
check("内部画风已无 promptEn", all("promptEn" not in x and "negative" not in x for x in STYLES))

prov = client.get("/api/providers").json()["providers"]
check("Provider 只剩 diffusers", [p["id"] for p in prov] == ["diffusers"], str(prov))
check("diffusers 为默认", prov[0]["active"] and prov[0]["configured"])

s = client.get("/api/settings").json()
check("aiLabel 默认关", s.get("aiLabel") is False)
check("settings 只含 diffusers 配置", list(s["providers"]) == ["diffusers"])

# 写回原样，确认 PUT 不丢字段、不炸
r = client.put("/api/settings", json=s)
check("PUT /api/settings 回写", r.status_code == 200 and r.json().get("aiLabel") is False, str(r.json())[:80])


print("== 已删除端点 ==")
check("POST /api/translate 已移除",
      client.post("/api/translate", json={"text": "猫"}).status_code in (404, 405))
check("mock Provider 已移除",
      client.post("/api/providers/mock/test").status_code in (404, 400))
check("a1111 Provider 已移除",
      client.post("/api/providers/a1111/test").status_code in (404, 400))


print("== 资产 / 下载 / 导出 / 删除 ==")
assets = client.get("/api/assets").json()["assets"]
check("资产列表非空", len(assets) > 0, f"{len(assets)} 项")

target = next((a for a in assets if a["kind"] == "generated"), assets[0])
raw = client.get(f"/api/assets/{target['id']}/download")
check("默认下载 200", raw.status_code == 200)
img_default = Image.open(io.BytesIO(raw.content)).convert("RGB")

labeled = client.get(f"/api/assets/{target['id']}/download?label=1")
check("label=1 下载 200", labeled.status_code == 200)
img_label = Image.open(io.BytesIO(labeled.content)).convert("RGB")


def green_label_pixels(im):
    """左下角角标区域里 #00FF88 的像素数。"""
    w, h = im.size
    region = im.crop((0, h - 22, 40, h)).convert("RGB")
    px = region.load()
    n = 0
    for y in range(region.height):
        for x in range(region.width):
            r_, g_, b_ = px[x, y]
            if g_ > 170 and r_ < 110 and b_ < 190 and g_ - r_ > 70:
                n += 1
    return n


n_default = green_label_pixels(img_default)
n_label = green_label_pixels(img_label)
check("默认下载无角标像素", n_default == 0, f"{n_default}px")
check("显式 label=1 有角标像素", n_label > 0, f"{n_label}px")

# 存储层不烧角标（后续合成不会被污染）
stored = Image.open(storage.asset_path(target)).convert("RGB")
check("存储文件无角标", green_label_pixels(stored) == 0)

z = client.post("/api/export/zip", json={"ids": [a["id"] for a in assets[:3]]})
check("POST /api/export/zip", z.status_code == 200, f"{len(z.content)} bytes")
with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
    names = zf.namelist()
    index = json.loads(zf.read("index.json"))
check("ZIP 含 index.json + 3 张图", "index.json" in names and len(index) == 3, str(names[:4]))

# 造一个临时资产验证删除
tmp = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
tmp_item = storage.save_asset(tmp, kind="test", meta={"tag": "e2e"})
d = client.delete(f"/api/assets/{tmp_item['id']}")
check("DELETE 资产", d.status_code == 200 and d.json()["ok"] is True)
check("删除后 404", client.get(f"/api/assets/{tmp_item['id']}/download").status_code == 404)


print("== 工作流 ==")
wfs = client.get("/api/workflows").json()["workflows"]
check("工作流列表", len(wfs) >= 1, f"{[w['id'] for w in wfs]}")
steps = [st for w in wfs for st in w["steps"]]
check("工作流步骤无 promptEn/layerNamesEn",
      not any(k in st for st in steps for k in ("promptEn", "layerNamesEn", "elementNamesEn", "editPromptEn")))

run = client.post(f"/api/workflows/{wfs[0]['id']}/runs",
                  json={"input": "灰白英短猫武士", "params": {"action": "挥砍"}})
check("创建工作流 run", run.status_code == 200)
run_id = run.json()["id"]
check("GET /api/runs 含新 run", any(r["id"] == run_id for r in client.get("/api/runs").json()["runs"]))
detail = client.get(f"/api/runs/{run_id}")
check("GET run 详情", detail.status_code == 200 and detail.json()["id"] == run_id)

print(f"\n结果：{ok} 通过 / {fail} 失败")
raise SystemExit(1 if fail else 0)
