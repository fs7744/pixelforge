# PixelForge 像素工坊 · MVP

本地优先的 AI 游戏素材工作站。**阶段 0（MVP）**：跑通「一句话 → 透明底像素素材」全链路。

## 现在就能跑

```bash
# 1. 后端依赖（首次）
python -m venv %USERPROFILE%\.workbuddy\binaries\python\envs\pixelforge
"%USERPROFILE%\.workbuddy\binaries\python\envs\pixelforge\Scripts\python.exe" -m pip install -r backend/requirements.txt

# 2. 前端构建（首次）
cd frontend && npm install && npm run build && cd ..

# 3. 启动（Windows 双击 start.bat，或）
"%USERPROFILE%\.workbuddy\binaries\python\envs\pixelforge\Scripts\python.exe" -m uvicorn backend.app.main:app --port 8787
```

打开 http://127.0.0.1:8787

## 启用本地真模型（可选）

MVP 默认走 `mock` 引擎（无需模型即可跑通全链路）。要用真实模型：

```bash
# 1. CPU 版（快装，1-3 分钟，验证用）
pip install torch diffusers transformers accelerate safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或 NVIDIA GPU 版（约 2.5GB，速度提升 10-50 倍）
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate safetensors

# 2. 下载模型（默认走 hf-mirror，已自动禁用 xet）
python backend/fetch_model.py segmind/tiny-sd        # ~1.1GB，CPU 可跑
python backend/fetch_model.py stabilityai/sd-turbo   # ~2.5GB，1-4 步，6GB 显存友好

# 3. 在「模型设置 → 内置推理服务」把模型填成本地路径，如 models/tiny-sd
#    然后点「测试连接」→「设为默认」
```

Turbo 系列自动使用 1-8 步 / CFG 0；显存 < 8GB 自动开启 attention slicing / vae tiling。
设置页提供「卸载模型 / 释放显存」。

## MVP 已实现

| 模块 | 状态 | 说明 |
|---|---|---|
| Provider 抽象 | ✅ | `mock` / `openai_compat` / `a1111`，统一 `generate()` 接口 |
| 内置模拟引擎 | ✅ | 无需模型：单图出 4×4 精灵表；多图出同角色不同姿态关键帧 |
| 提示词模板库 | ✅ | 10 个模板 × 6 分类，来自教程资源提取 |
| 生成工作台 | ✅ | 模板选择、提示词编辑、参数、结果网格、历史、送后处理 |
| **工作流引擎** | ✅ | 3 条工作流 · 多段步骤 · 单步重跑 · 产物自动传递 |
| **动画工作流** | ✅ | 三视图 → 动作帧 → 抽帧去重(内容哈希) → 插帧 → 精灵表 → Godot 包 |
| **视差工作流** | ✅ | 参考图 → 4 层提取 → 同尺寸规整 → Parallax 场景 + scroll_scale 表 |
| **场景工作流** | ✅ | 参考图 → 4 元素提取 → 拼装总览 |
| **Godot 工程导出** | ✅ | SpriteFrames `.tres` + AnimatedSprite2D `.tscn` + 索引 + 说明 |
| **本地 diffusers 推理** | ✅ | 懒加载 · 单例缓存 · 6GB 显存自适应 · 国内走 hf-mirror |
| **无缝延展** | ✅ | 镜像拼接，接缝差值实测 0.0，循环播放无痕 |
| **瓦片接缝修补** | ✅ | 偏移-修复法，保留图案结构只修接缝 |
| 连通域抠图 | ✅ | 边缘连通 flood fill + 容差/羽化 |
| 尺寸规整 | ✅ | 最近邻硬缩放 / 等比留白 / RPGMaker V1·V2 三档 |
| Sprite Sheet 拆分 | ✅ | N×M 拆帧 + 引擎索引 JSON |
| 去水印 | ⚠️ Beta | 反向 Alpha 混合已实现，需自备 logo 掩码 |
| 素材库 | ✅ | 本地落盘 + 索引 + 批量 ZIP |
| AI 标识 | ✅ | 导出图左下角角标（可关） |

### 工作流引擎怎么工作

```
POST /api/workflows/{id}/runs        创建工作流（输入 + 参数）
POST /api/runs/{id}/steps/{step}/run 执行单步（可重跑）
POST /api/runs/{id}/run-all          顺序执行剩余步骤
GET  /api/runs/{id}                  查询步骤状态与产物
```

步骤类型：`generate`（生成）/ `dedup`（内容哈希去重）/ `blend`（插帧）/
`compose`（合成精灵表）/ `normalize`（同尺寸）/ `montage`（拼装）/
`godot`、`godot_parallax`（工程导出）。状态落盘在 `data/runs/{id}.json`。

## 目录

```
pixelforge/
├── backend/app/
│   ├── main.py            # FastAPI 路由
│   ├── config.py          # settings.json 持久化
│   ├── templates.py       # 10 个提示词模板
│   ├── storage.py         # 资产落盘 + index.json + AI 角标
│   ├── providers/         # mock / openai_compat / a1111
│   └── pipelines/         # matte / resize / sheet / watermark
├── frontend/src/          # React + Vite（像素风设计系统）
├── data/assets/           # 本地素材库（git 忽略）
└── start.bat / start.sh
```

## 下一步（阶段 1）

- 提示词工作流引擎（多段步骤 + 单步重跑）
- 动画工作流（三视图 → 动作视频 → 抽帧 → 插帧）
- 内置 diffusers 推理服务（替换 mock 为真实本地模型）
- Godot 工程导出（.tscn 模版 / scroll_scale 参数表）
- 视差 9+1 图层与无缝延展
