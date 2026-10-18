# PixelForge 设计文档（Design & Progress）

> 版本：V1.2 ｜ 更新：2026-10-18（接入 Z-Image-Turbo 原生中文底座）
> 目的：让任何新接手的人**不看聊天记录、不翻外部资料**也能理解项目全貌并继续开发。
> 关联文档：仓库内 `README.md`（如何跑起来）、仓库外 `PixelForge-项目方案.md`（产品愿景 V1.0）。

---

## 一、项目是什么

**PixelForge（像素工坊）** —— 本地优先的 AI 游戏素材与工程工作流平台。

一句话定位：**从"一句话/一张参考图"到"可直接进 Godot 的工程资产"的流水线**，而不是单张出图工具。

- 用户：独立开发者 / 小团队 / 无代码游戏创作者
- 核心差异：卖的是「素材 → 动画 → 场景 → 引擎工程」完整链路，终点是 `.tscn` / 动画配置 / 视差参数，而非零散 PNG
- 技术底座：站点自持 diffusers 推理引擎（本地模型），兼容 A1111/Forge 接入，云端 OpenAI 兼容 API 兜底

---

## 二、当前真实进度（截至 2026-10-18）

> 注意：外部《项目方案》里的「下一步（阶段1）」已**大部分落地**，下面的状态以代码为准。

### 2.1 已实现（✅ 可运行）

| 模块 | 状态 | 关键实现 |
|---|---|---|
| Provider 抽象 | ✅ | `mock` / `diffusers`(本地) / `openai_compat`(云端) / `a1111`，统一 `generate()` |
| 内置 diffusers 推理 | ✅ | txt2img + img2img（`from_pipe` 共享权重）· 懒加载 · 单例缓存 · 6GB 显存自适应 · 国内 hf-mirror |
| **Z-Image 原生中文底座** | ✅ | `detect_family()` 模型族识别 · `ZImagePipeline` bf16 · 8 步蒸馏 · `supports_zh` 为真时全链路自动跳过翻译 |
| 角色一致性 | ✅ | 两条路：`refMode=ip`（IP-Adapter 控身份，推荐）/ `refMode=img2img`（参考图当初始图） |
| 中文提示词翻译 | ✅ | 领域词典优先（覆盖 90-100%）→ 漏译/长描述时本地 Marian 机翻兜底（`opus-mt-zh-en`），术语表校正 |
| 工作流引擎 | ✅ | 3 条工作流 · 多段步骤 · 单步重跑 · 产物自动传递 · 状态落盘 `data/runs/{id}.json` |
| 动画工作流 | ✅ | 三视图 → 动作帧 → 抽帧去重(内容哈希) → 插帧 → 精灵表 → Godot 包 |
| 视差工作流 | ✅ | 参考图 → 4 层提取 → 同尺寸规整 → Parallax 场景 + `scroll_scale` 表 |
| 场景工作流 | ✅ | 参考图 → 4 元素提取 → 拼装总览 |
| Godot 工程导出 | ✅ | SpriteFrames `.tres` + AnimatedSprite2D `.tscn` + 索引 + 说明 |
| 抠图（连通域） | ✅ | 四角取样定底色 + 边缘连通洪泛 + 有限梯度生长 |
| **深度抠图（三级降级）** | ✅ | `u2net → grabcut → key`；硬边二值 alpha + 边缘去污(despill) + 小洞填充 |
| 无缝延展 | ✅ | 镜像拼接，接缝差值 0.0 |
| 瓦片接缝修补 | ✅ | 偏移-修复法 |
| 像素化对齐 | ✅ | 分块单色 + 调色板量化（8/16/32、GameBoy 4、NES 54）+ 可选抖动 |
| 尺寸规整 | ✅ | 最近邻硬缩放 / 等比留白 / RPGMaker V1·V2 |
| Sprite Sheet 拆分 | ✅ | N×M 拆帧 + 引擎索引 JSON |
| 去水印 | ⚠️ Beta | 反向 Alpha 混合，需自备 logo 掩码 |
| 素材库 | ✅ | 本地落盘 + 索引 + 批量 ZIP |
| AI 标识 | ✅ | 导出图左下角角标（可关） |

### 2.2 未完成 / 待办

- 去水印仍 Beta（依赖 logo 掩码，未自动化）
- 本地视频生成 / 插帧尚未接真实视频 pipeline（当前插帧走"图生图低重绘"）
- 多 agent 并行批量 + 自动质检（方案里有，代码未做）
- 规范文档生成器 / 系统代码模板（M5，P2，未做）
- Docker 一键部署、用户文档（阶段3，未做）
- u2net 权重本机未就位（见 §5 模型说明），当前抠图实际走 GrabCut

### 2.3 已完成的关键提交

| commit | 内容 |
|---|---|
| `28e36b5` | 修复像素抠图白边/误抠白甲（硬边 alpha + 三级降级 + u2net 守卫） |
| `c1e0641` | 前端示例提示词改为中文 |
| `a9e1d1d` | 中文提示词翻译：词典优先 + 本地 Marian 机翻兜底 |
| `7a9714d` | 接入 IP-Adapter + 像素 LoRA + LCM LoRA，本地推理质量升级 |

---

## 三、技术架构

```
┌─────────────────────────────────────────────┐
│ 前端 React + Vite（像素风设计系统）          │
│  views/: Workbench(工作台) Workflow(工作流)  │
│          PostProcess(后处理) Library(素材库) │
│          Settings(模型设置)                  │
├─────────────────────────────────────────────┤
│ API 网关 FastAPI（backend/app/main.py）      │
│  · Provider 路由（diffusers/a1111/云端/mock）│
│  · 工作流引擎（步骤状态机，data/runs 落盘）   │
│  · 资产服务（storage.py + index.json）       │
├─────────────────────────────────────────────┤
│ 推理层（providers/diffusers_local.py）       │
│  Z-Image-Turbo（默认，中文直喂，8 步蒸馏）    │
│  SD1.5 底座 + LCM LoRA(提速) + 像素 LoRA     │
│  + IP-Adapter(一致性，仅 SD 族)              │
├─────────────────────────────────────────────┤
│ 后处理层 pipelines/                          │
│  matte(抠图) pixelize(像素化) resize sheet   │
│  seamless(无缝) tile(瓦片) watermark sprite  │
├─────────────────────────────────────────────┤
│ 提示词翻译 zh2en.py + mt.py(Marian 机翻)     │
│  （Z-Image 族自动跳过，仅 SD 族走此链路）     │
└─────────────────────────────────────────────┘
```

**关键设计决策（踩坑结论，勿回退）：**

1. **底座必须 SD1.5 系**：`sd-turbo` 是 SD2.1（cross-attn 1024 维）、`tiny-sd` UNet 被裁过，都装不了 SD1.5 的像素 LoRA / IP-Adapter。标准组合 = `dreamshaper-8 + LCM LoRA + 像素 LoRA + IP-Adapter sd15`。
2. **抠图硬边优先**：像素素材 alpha 只能全不透明/全透明。旧版对 alpha 做高斯羽化会产生半透明白晕（实测均值 15.4%），已改为二值 alpha + despill。
3. **IP-Adapter 挂载顺序**：必须先挂 IP（会改 UNet `encoder_hid_dim_type`）再挂 LoRA；失败要回滚，否则留下脏 UNet 报 `NoneType not iterable`。
4. **`_CAPS` 要先 `_load` 再读**：先读会拿到空字典，导致 IP 挂上却不传参考图。
5. **中文翻译策略**：词典覆盖率 ≥0.70 走词典（更快更准，机翻会把「像素风/Q版」译成错词）；只有丢过半语义才切 Marian 机翻兜底。

---

## 四、目录与运行

### 4.1 目录结构

```
pixelforge/
├── backend/
│   ├── requirements.txt     # Python 依赖
│   ├── app/
│   │   ├── main.py          # FastAPI 路由入口
│   │   ├── config.py        # settings.json 持久化
│   │   ├── templates.py     # 提示词模板库
│   │   ├── storage.py       # 资产落盘 + 索引 + AI 角标
│   │   ├── zh2en.py         # 中文→英文（领域词典 + 最长匹配）
│   │   ├── mt.py            # 本地 Marian 机翻兜底
│   │   ├── workflows.py     # 3 条工作流定义 + 步骤编排
│   │   ├── godot_export.py  # Godot 工程导出
│   │   ├── providers/       # mock/diffusers_local/openai_compat/a1111
│   │   └── pipelines/       # matte/pixelize/resize/sheet/seamless/tile/watermark/sprite
│   ├── fetch_model.py       # 下载底座模型
│   ├── fetch_support.py     # 下载 IP-Adapter + 像素 LoRA
│   ├── parallel_dl.py       # curl 分块并发下载
│   └── smoke_test.py
├── frontend/src/            # React + TS + Vite
│   ├── App.tsx / api.ts / main.tsx / styles.css
│   └── views/  Workbench/Workflow/PostProcess/Library/Settings
├── models/                  # 本地权重（git 忽略）
├── data/                    # 资产/运行状态（git 忽略）
├── start.bat / start.sh     # 一键启动（uvicorn 8787）
└── README.md / docs/DESIGN.md
```

### 4.2 快速开始

```bash
# 1. 依赖
python -m venv %USERPROFILE%\.workbuddy\binaries\python\envs\pixelforge
"%USERPROFILE%\.workbuddy\binaries\python\envs\pixelforge\Scripts\python.exe" -m pip install -r backend/requirements.txt

# 2. 前端构建
cd frontend && npm install && npm run build && cd ..

# 3. 启动
"%USERPROFILE%\.workbuddy\binaries\python\envs\pixelforge\Scripts\python.exe" -m uvicorn backend.app.main:app --port 8787
```

打开 http://127.0.0.1:8787。改 `backend/` 需重启服务；改 `frontend/src` 需重新 `npm run build`。

### 4.3 当前配置状态

`data/settings.json` 默认 `activeProvider = "mock"`（在设置页切换到 `diffusers` 即用真模型）。
diffusers provider 默认模型为 `models/z-image-turbo`（Z-Image 族，中文直喂）；
`ipAdapter` / `lora` / `lcmLora` 留空时自动探测 `models/` 下默认目录，**仅 SD1.5 族生效**。
一键部署：`bash backend/install_gpu_deps.sh && bash backend/fetch_zimage.sh`，
冒烟：`python backend/smoke_zimage.py`。

---

## 五、模型选型分析：中文提示词更优的开源本地模型

> 2026-10-18 重写：本机已确认是 **RTX 4090 24GB**（早期文档按 6GB 显卡论证，选型空间被严重低估），
> 且 Z-Image-Turbo 已于 2025-11 开源并成为中文开源生图的新标杆。本章以 4090 24GB 为约束重排。

### 5.1 现状问题

早期底座是 **SD1.5（dreamshaper-8）+ CLIP 文本编码器**。CLIP 训练语料以英文为主，直接喂中文会丢语义，因此必须走「中文 → 词典/机翻 → 英文 → CLIP」的间接链路（`zh2en.py` + `mt.py`）。这套方案在短语式游戏素材描述上覆盖率高、够用，但语义有天然损耗。

### 5.2 2026-10 中文友好开源模型横评（按 4090 24GB 适配排序）

| 排序 | 模型 | 文本编码器 | 中文能力 | 架构/体积 | 4090 24GB 适配 | 许可 | 结论 |
|---|---|---|---|---|---|---|---|
| **1** | **Z-Image-Turbo**（阿里通义，2025-11） | Qwen3-4B | **原生中文直喂**，指令理解率 92% | 6B S3-DiT 单流，bf16 约 20GB | ✅ 全载，**8 步约 2.5s/张** | Apache 2.0 | ⭐ **已落地为本项目默认底座**（见 §4.3） |
| 2 | Qwen-Image 2.0（阿里，2026-02） | Qwen2.5-VL 7B | 中文文字渲染顶尖，语义编辑强 | 20.4B DiT | ⚠️ FP16 42GB 超了；FP8 约 22GB 勉强进，需 offload | Apache 2.0 | 备选：图内中文字样 / 指令式角色编辑 |
| 3 | GLM-Image（智谱） | Glyph-byT5 | 双语文字渲染 #1（CVTG-2K 91.16%） | 16B（9B AR + 7B DiT） | ⚠️ 需 offload | MIT | 观察：游戏招牌/标题嵌字 |
| 4 | Kolors 2.1（快手） | GLM | 中文美学/写实强 | SDXL 架构，8-12GB | ✅ 宽松 | 商用需登记 | 生态备选：唯一能复用 SDXL 像素 LoRA |
| 5 | HunyuanImage 3.0（腾讯） | — | 中文尚可 | 80B MoE | ❌ 单卡跑不动 | 开源 | 排除 |
| 6 | FLUX.2 [dev] | T5（偏英文） | 中文弱 | 32B | ⚠️ 24GB 紧张 | 商用另收费 | 不推荐（中文弱 + 许可） |

**为什么 Z-Image-Turbo 是首选**：原生中文（翻译链路整个绕过）+ 8 步蒸馏（4090 约 2.5s/张，
与项目「快速迭代素材」的工作流匹配）+ bf16 全载进 24GB + Apache 2.0 可商用 +
diffusers 有原生 `ZImagePipeline`（ComfyUI day-0）。开源榜第一（Artificial Analysis
Elo 1025，超过 GPT-Image-1 / Seedream 3.0）。

### 5.3 落地方式：Z-Image 已接入（2026-10-18）

代码已落地，新增/改动：

1. `providers/diffusers_local.py`：新增模型族识别 `detect_family()`（读 `model_index.json`
   的 `_class_name`，目录名兜底，支持 zimage/sd/qwen_edit 三族）；`_load_zimage()` 走
   `ZImagePipeline`（bf16；`low_cpu_mem_usage` 保持默认 True——实测 False 会让推理慢
   10 倍以上且加载完整性无差异）；`_run_zimage()` 固定 `guidance_scale=0.0`、
   `num_inference_steps=9`（调度器把 step 0 当热身，实际 8 次 DiT 前向）、分辨率按 16 对齐；
   `supports_zh` 属性为真时，`main.py` / `workflows.py` 自动跳过翻译链路。
2. `config.py`：默认模型改为 `models/z-image-turbo`。
3. 新增脚本：`backend/install_gpu_deps.sh`（一键装 torch cu124 + diffusers）、
   `backend/fetch_zimage.sh` / `backend/fetch_qwen_edit.sh`（ModelScope 下载，断点续传）、
   `backend/smoke_zimage.py` / `backend/smoke_qwen_edit.py`（冒烟测试）、
   `backend/bench_zimage.py`（稳态测速）。
4. `fetch_model.py` 的 ALLOW 增加 Z-Image 布局（`transformer/` 分片 + `text_encoder/` 分片）。

**接入注意点（代码已处理）**：

- Z-Image 是 bf16 单流 DiT，**CPU 推理不可行**，无 GPU 时需切回 SD 族；
- SD1.5 的像素 LoRA / IP-Adapter / LCM 均不适用（UNet 结构不同），caps.notes 会说明；
- 参考图（ref）在 Z-Image 分支被忽略（txt2img 无编辑管线）：一致性走双底座——动画帧 2+
  切 Qwen-Image-Edit-2511 指令式编辑（`qwen_edit` 分支已落地，见 §5.5），或短期用 SD 族
  `refMode=ip/img2img`；
- diffusers 需较新版本（含 `ZImagePipeline`），`install_gpu_deps.sh` 装的是最新 stable。

### 5.4 像素风专精：不换底座的轻量路线（对 Z-Image 同样适用）

像素风格一致性主要靠后处理而非 LoRA：

- **1024² 出图 + 8× 最近邻降采样**（绝不 bilinear/bicubic）+ 调色板量化——
  本项目 `pixelize.py` 已实现；大图降采样比 512² 直出细腻一个量级；
- **cuDNN benchmark 必开**（`torch.backends.cudnn.benchmark = True`，已写进
  `_load_zimage`/`_load_sd`）：cuDNN 卷积算法启发式在显存紧张时会选中非 Tensor Core
  慢速路径，实测 4090 上 VAE 解码从 5s 劣化到 4 分钟以上且**非确定性复现**（py-spy
  栈定位卡在 `vae.decode → conv_forward`）；开 benchmark 后确定性走最快算法，
  稳态 1024² 九步 5.1s/张、峰值显存 21.0GB；
- 若后续要风格 LoRA：Z-Image 生态的像素 LoRA 尚未成熟，可训练（DiffSynth-Studio 支持
  Z-Image/Qwen-Image LoRA），或短期切 Kolors 2.1 复用 SDXL 的 Pixel Art XL。

### 5.5 角色一致性设计：双底座架构（Turbo 生成 + Edit 编辑）

**结论：Z-Image-Turbo 与 Qwen-Image-Edit-2511 是分工关系，不是二选一。Turbo 负责
「从无到有」，Edit-2511 负责「从有到优」，两者都留。**

IP-Adapter 路线对 Z-Image 是**关闭的**——IP-Adapter 只有 SD1.5/SDXL 版，S3-DiT 的
`encoder_hid_dim_type` 挂载点不存在，别再尝试。

#### 5.5.1 为什么 Edit 模型替代不了 Turbo（回答"能不能只留 Edit"）

1. **Edit 是编辑器不是生成器**：`QwenImageEditPlusPipeline` 必须吃输入图（支持多图
   list），无参考自由创作不是它的能力域（官方与社区口径一致）。动画工作流的第一帧
   （一句话 → 角色）只能靠 txt2img。
2. **成本差一个量级**：Turbo 6B/8 步/约 2.5s 每张；Edit-2511 20B/40 步/约 25s 每张
   （Lightning 4 步蒸馏版约 3.5s，可用于迭代）。概念探索、道具/场景批量出图用 Edit
   做纯属浪费。
3. **显存互斥但无冲突**：两者各约 20GB，不能同时驻留 24GB；Provider 单例缓存切换时
   自动卸载旧管线（`_load` 已实现），对用户透明。

#### 5.5.2 双底座分工（推荐架构）

| 阶段 | 模型 | 输入 → 输出 | 用途 |
|---|---|---|---|
| 帧 1 / 道具 / 场景 / 概念探索 | **Z-Image-Turbo** | 中文提示词 → 图 | 默认 txt2img，8 步 2.5s |
| 帧 2+（保身份改姿势） | **Qwen-Image-Edit-2511** | 上一帧 + 中文指令 → 图 | 指令式编辑，2511 版专项优化了角色一致性与抗图像漂移 |
| 低配/重 IP-Adapter 场景 | SD1.5（dreamshaper-8） | 现有链路 | 兜底，不改动 |

动画工作流因此变成：**帧 1 用 Turbo 定角色 → 帧 2+ 全部走 Edit-2511**，
指令模板：「保持角色外观、服装、配色完全不变，将姿势改为：{action}，纯色背景」。
Turbo 在动画里只出场一次，但在道具/场景/概念探索上是主力，**不退役**。

Edit-2511 的白捡能力：内置多视角生成 LoRA（角色三视图直接出）、内置光照增强 LoRA，
对游戏素材管线直接可用。

#### 5.5.3 已核实事实（2026-10-18 实测）

- `Qwen/Qwen-Image-Edit-2511` 官方仓库存在，**ModelScope 有镜像**（bf16 共 53.8GB：
  transformer 5 分片 37GB + text_encoder 4 分片 15.5GB + VAE），Apache 2.0；
- diffusers 管线类名 `QwenImageEditPlusPipeline`（需较新 diffusers，安装脚本已装
  git 最新版）；调用参数：`true_cfg_scale=4.0`、`guidance_scale=1.0`、40 步；
- FP8 量化版约 20GB（`drbaph/Qwen-Image-Edit-2511-FP8`，HF only，需代理）；
- Z-Image-Edit 官方仍是 "To be released"（ModelScope/HF 均无仓库），**不作为依赖**；
- 2511 版核心升级正是本项目要的：角色一致性增强 + 缓解图像漂移 + 社区 LoRA 内置。

#### 5.5.4 接入落地（2026-10-18 已完成代码，模型下载中）

- `backend/fetch_qwen_edit.sh`：ModelScope 下载 Edit-2511（bf16 53.8GB，断点续传）；
- provider 已加 `qwen_edit` family 分支：`detect_family` 按 `model_index.json`
  （`QwenImageEditPlusPipeline`）或目录名（`qwen-image-edit`）识别；
  `_load_qwen_edit()` 走 `QwenImageEditPlusPipeline`（bf16 + cpu offload，
  42GB 全量超 24GB，按层换入换出慢约 30% 但可跑；有 FP8 量化版可去 offload）；
  `_run_qwen_edit()` 必传 ref、40 步、`true_cfg_scale=4.0`、`guidance_scale=1.0`；
- **动画工作流零改动**：`useRef` 步骤本来就在传 ref，把模型切到 Edit-2511 即接力；
- **自动接力已内置**（2026-10-18）：`workflows.py` 的 `action_frames` 步骤检测到当前
  模型族为 `qwen_edit` 时，帧 1 自动借 txt2img 底座（`txt2imgModel`，默认
  `models/z-image-turbo`）生成，帧 2+ 回到编辑器走 `editPrompt` 指令模板
  （「保持角色外观、服装、配色完全不变，将姿势改为：{action}…」——**刻意不带角色描述**，
  外观由参考图承载，带文字描述反而会让模型重释外观）；meta 记录
  `mode=txt2img→edit-chain`；底座缺失时退回单模型链并在 meta.relayNote 说明；
- `supports_zh` 对 qwen_edit 同为真——中文指令直喂不过翻译；
- 冒烟脚本 `backend/smoke_qwen_edit.py`：以 Z-Image 首图为参考图，验证
  「同角色换姿势」并拼对比图；
- 备选同家族单栈方案：Qwen-Image base（txt2img）+ Edit-2511，text_encoder 可共享
  去重，但 base 20B FP8 更慢更紧，除非出现跨家族风格漂移问题再考虑。

### 5.6 美术画风库（14 种游戏行业典型风格）

`backend/app/styles.py`：生成时可选画风，风格词拼接在提示词尾部。

**语言分流（关键设计）**：

| Provider | 拼接内容 | 原因 |
|---|---|---|
| 中文模型（Z-Image / Qwen-Image-Edit，`supports_zh=True`） | 中文风格词 | 直喂，无翻译损耗 |
| SD 系（CLIP，`supports_zh=False`） | 先译中文（`add_style=style is None`）再拼 `promptEn` | 机翻会毁掉风格词；选了画风就要关掉 zh2en 的 `DEFAULT_STYLE` 兜底，否则两段风格词打架 |

**踩坑记录**（2026-10-18）：最初把英文风格词拼进中文 prompt 再整体送 `zh2en`——
`zh2en.translate()` 只处理中文 chunk，ASCII 片段被静默丢弃，译后还补了它自己的
`DEFAULT_STYLE`，用户选的画风完全失效。且英文 prompt（无中文、passthrough 分支）
会被 `apply_style` 和 main.py 各拼一次风格词。最终形态：SD 分支不复用
`apply_style` 合并的 prompt，只用它的 negative/style/suggestPost 字段。

**像素化联动**：`pixel_16bit`（8×/16 色）与 `pixel_hd`（4×/32 色）带
`suggestPost` 参数，随 asset meta 落库；PostProcess 页收到该资产时自动切
pixelize 管线并预填参数。

**负向词**：SD 系把画风的 `negative`（如 blurry/3d render）追加进 provider
cfg 的 negativePrompt；Z-Image/Qwen-Edit 蒸馏模型不吃负向，不加。

**覆盖范围**：像素×2、2D 动漫/卡通×4（Q 版/赛璐璐/黑白漫画/美式）、UI×2
（扁平矢量/粗描边图标）、写实 3D×3（厚涂/写实 3A/低多边形/体素）、手绘×2
（水彩/概念速写）。动画工作流不挂画风——帧 2+ 的编辑指令是锁外观的，
风格词会干扰身份保持。

### 5.7 结论

1. **默认底座已切 Z-Image-Turbo**：中文直喂绕过翻译链路，8 步 2.5s/张，4090 全载；
2. SD1.5 链路保留为低配/一致性兜底（IP-Adapter 仍在 SD 族有效）；
3. **一致性走双底座**：帧 1 生成用 Turbo，帧 2+ 编辑用 Qwen-Image-Edit-2511
   （provider `qwen_edit` 分支已落地，模型下载中，见 §5.5.4）——Turbo 不退役，
   两者分工不互斥；
4. 图内中文字样是 Qwen-Image 2.0（FP8）的活，作为第三底座观察；
5. 像素风格靠「大图出图 + 最近邻降采样 + 调色板量化」，不必依赖换底座。

---

## 六、参考资料

- 外部产品方案：仓库外 `PixelForge-项目方案.md`（V1.0，含教程资源提取方法论 + 六类模板库结构 + 路线图）
- 上游灵感项目：`FrameRonin`（Gemini Gem 外链生成 + Canvas 后处理管线，已分析，见 `D:/code/gen/FrameRonin-项目分析与Skill化评估.md`）
- 教程资源：`D:\video\教程资源`（0代码 AI 游戏编程 + AI 游戏美术五大工作流）
- 模型参考：
  - Kolors：`github.com/Kwai-Kolors/Kolors`，HuggingFace `Kwai-Kolors/Kolors`
  - Qwen-Image：阿里通义，Apache 2.0，2025-08 开源
  - 像素 LoRA：Pixel Art XL（nerijs）/ Pixel Party XL / FLUX.2-klein Sprite Sheet LoRA
  - 抠图：rembg/u2net、OpenCV GrabCut

---

## 七、给接手者的上手路径

1. 读 `README.md` → 装依赖、跑起来（mock 引擎无需模型即可验证全链路）
2. 读本文 §2（进度）、§3（架构与踩坑）、§5（模型选型）
3. 看 `backend/app/workflows.py` 理解 3 条工作流如何编排；`pipelines/` 是纯函数后处理，可独立测试
4. 改代码后：后端改完重启 uvicorn，前端改完 `npm run build`
5. 提交前确认 `.gitignore` 已排除 `models/`、`data/`（权重与产物不入库）
