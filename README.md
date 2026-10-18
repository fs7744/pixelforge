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

# 3. 在「模型设置 → 内置推理服务」把模型填成本地路径，如 models/sd-turbo
#    然后点「测试连接」→「设为默认」

# 4.（可选）一致性增强：IP-Adapter + 像素 LoRA
python backend/fetch_support.py    # IP-Adapter 2.5GB + 像素 LoRA 26MB
# 服务启动后自动探测 models/ip-adapter-sd15 与 models/lora-pixel-redmond，
# 加载成功时「测试连接」会显示 IP-Adapter@0.7 · LoRA xxx@0.85
```

**一致性两条路，按硬件选：**

| 方案 | 原理 | 代价 | 效果 |
|---|---|---|---|
| `refMode=ip`（推荐） | IP-Adapter 用参考图控制身份，提示词自由控制姿势 | +2.5GB 权重、+约 40% 推理耗时 | 身份稳、姿势幅度大 |
| `refMode=img2img` | 参考图当初始图，`strength` 平衡改动幅度 | 无额外权重 | 身份稳，但姿势幅度被 `strength` 锁死 |

**⚠️ 底座必须选 SD1.5 系（这里踩过坑）**

`sd-turbo` 是 **SD2.1 架构**（cross-attention 维度 1024），而像素 LoRA 与 IP-Adapter 生态几乎全在 **SD1.5（768 维）**。硬装会报 `size mismatch for ... to_k.lora_A: 768 vs 1024`。

所以增强链路的标准组合是：

```
SD1.5 底座（Lykon/dreamshaper-8）
  + LCM LoRA（lcm-lora-sdv1-5，4 步出图，替代 turbo）
  + 像素 LoRA（26MB，压掉插画感）
  + IP-Adapter sd15（参考图控身份）
```

| 模型 | 架构 | 4 步出图 | 能否装 SD1.5 LoRA / IP-Adapter |
|---|---|---|---|
| `segmind/tiny-sd` | SD1.5（缩小 UNet） | ✅ | ❌ UNet 结构被裁过，形状不匹配 |
| `stabilityai/sd-turbo` | **SD2.1** | ✅ | ❌ 维度 1024 |
| `Lykon/dreamshaper-8` | **SD1.5** | 需配 LCM LoRA | ✅ 全部兼容 |

**实测参考（本机 GTX 1660 Ti，但当前装的是 CPU 版 torch）**：

| 模型 | 体积 | 512² 步数 | CPU 耗时 | 质量 |
|---|---|---|---|---|
| `segmind/tiny-sd` | 1.0GB | 4 步 | 35-49s | 粗糙，仅用于跑通链路 |
| `stabilityai/sd-turbo` | 5.1GB | 2 步 | ~34s | 可用，像素风精灵效果良好 |

装 GPU 版 torch 后同样参数通常 **1-3 秒**。sd-turbo 仓库有 12GB（含 fp16 变体与单文件），
`fetch_model.py` 只取 fp32 组件（约 5GB）；GPU 用户可自行加 `*.fp16.safetensors` 提速。

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
| **本地 diffusers 推理** | ✅ | 文生图 + 图生图（`from_pipe` 共享权重）· 懒加载 · 单例缓存 · 6GB 显存自适应 · 国内走 hf-mirror |
| **角色一致性（img2img 自锚定）** | ✅ | 第 1 帧文生图定角色，其余帧以「抠图后铺纯色底的第 1 帧」为参考做 img2img，四帧同角色 |
| **中文提示词自动翻译** | ✅ | 游戏素材领域词典（颜色/角色/装备/动作/画风/场景）+ 修饰语黏合，实测覆盖率 90-100% |
| **无缝延展** | ✅ | 镜像拼接，接缝差值实测 0.0，循环播放无痕 |
| **瓦片接缝修补** | ✅ | 偏移-修复法，保留图案结构只修接缝 |
| **像素化对齐** | ✅ | 分块单色 + 调色板量化（自动 8/16/32、GameBoy 4 色、NES 54 色）+ 可选抖动 |
| 连通域抠图 | ✅ | 四角取样定底色（多色板）+ 边缘连通洪泛 + 有限梯度生长 |
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
