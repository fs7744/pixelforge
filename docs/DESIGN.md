# PixelForge 设计文档（Design & Progress）

> 版本：V1.1 ｜ 更新：2026-10-18
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
│  SD1.5 底座 + LCM LoRA(提速) + 像素 LoRA     │
│  + IP-Adapter(一致性)                        │
├─────────────────────────────────────────────┤
│ 后处理层 pipelines/                          │
│  matte(抠图) pixelize(像素化) resize sheet   │
│  seamless(无缝) tile(瓦片) watermark sprite  │
├─────────────────────────────────────────────┤
│ 提示词翻译 zh2en.py + mt.py(Marian 机翻)     │
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

`data/settings.json` 默认 `activeProvider = "diffusers"`，模型指向 `models/dreamshaper-8`，`ipAdapterScale 0.75` / `loraScale 0.8` / `refMode ip`。

---

## 五、模型选型分析：中文提示词更优的开源本地模型

### 5.1 现状问题

当前底座是 **SD1.5（dreamshaper-8）+ CLIP 文本编码器**。CLIP 训练语料以英文为主，直接喂中文会丢语义，因此必须走「中文 → 词典/机翻 → 英文 → CLIP」的**间接链路**（`zh2en.py`）。这套词典 + Marian 机翻方案在短语式游戏素材描述上覆盖率高、够用，但**语义有天然损耗**，遇到长句、生僻词、文化意象时仍会打折。

### 5.2 根本解法：换「原生中文文本编码器」的底座

开源社区已出现一批用**中文 LLM 做文本编码器**（而非 CLIP）的生图模型，能**直接吃中文提示词**，从根上绕开翻译损耗。按本项目「像素素材 + 本地优先 + 消费级硬件」的约束排序：

| 排序 | 模型 | 文本编码器 | 中文支持 | 架构/体积 | 显存 | 许可 | 适配像素素材 | 结论 |
|---|---|---|---|---|---|---|---|---|
| **1** | **Kolors（快手可图）** | ChatGLM3-6B | **原生中文（最强）** | SDXL 系 UNet，2.6B | 8–12GB（量化可更低） | Apache 2.0（商用需登记） | **好**：兼容 SDXL 像素 LoRA 生态 | ⭐ **首选** |
| 2 | Qwen-Image（阿里） | Qwen2.5-VL | 原生中文，文字渲染顶尖 | 20B MMDiT | 12GB+ 起步 | Apache 2.0 | 一般：偏海报/文字，对像素素材偏重 | 备选（重） |
| 3 | SD3.5 Medium/Large | T5+CLIP | 原生中文 | DiT | 6–12GB | 社区许可 | 中：LoRA 生态比 SDXL 薄 | 备选 |
| 4 | FLUX.1 schnell/dev | T5（偏英文） | 中文弱 | 12B | 12–16GB | schnell Apache2.0 / dev 非商用 | 薄：像素 LoRA 少 | 不推荐（中文弱） |
| 5 | HunyuanDiT（腾讯）/ Z-Image（阿里） | 中文 LLM | 中文尚可 | DiT | 12GB+ | 开源 | 中 | 观察 |

### 5.3 首选：Kolors 的接入评估

**为什么是 Kolors：**
- **中文理解最强**：ChatGLM3-6B 是全自回归 LLM（非 CLIP 对比学习），中文语义、成语、文化意象理解远超 CLIP，256 token 提示词预算（SDXL 仅 77）。
- **生态最顺**：SDXL 同构 UNet，可直接复用 SDXL 的像素 LoRA 生态（如 Pixel Art XL 等），迁移成本最低。
- **显存友好**：fp16 UNet 仅 5.16GB；文本编码器 ChatGLM 是主要开销（12.5GB），社区 8-bit/4-bit 量化可压到 6.78GB/3.92GB，整管线可落在 10GB 内。

**接入成本与注意点（需新写一个 provider 或适配层）：**
1. ChatGLM 编码器大 → 走 `Kijai/ChatGLM3-safetensors` 量化；SDXL VAE 有 fp16 溢出黑图 bug，需 fp32 或换 fp16-fix VAE。
2. 官方 repo 是 pickle 分片（非 safetensors），diffusers 加载需注意；提示词超 256 token 静默截断。
3. 原官方环境 pin Python 3.8（本项目是 3.13，需验证兼容）。
4. **保留现有像素 LoRA 的复用**：需要把 SD1.5 像素 LoRA 换成 SDXL 版（如 Pixel Art XL），IP-Adapter 也需 SDXL 版本。

**建议落地方式：** 在 Provider 抽象下新增 `kolors` provider（或给 `diffusers_local` 加模型族分支），保留现有 SD1.5 链路作为低配兜底，实现「Kolors 原生中文（推荐）/ SD1.5 翻译链路（兜底）」双底座。

### 5.4 纯像素风专精的补充方向（不换底座，轻量）

若暂不换底座，可只加/换**像素风 LoRA/checkpoint**提升风格一致性（不改中文翻译链路）：

- **Pixel Art XL（nerijs）**：SDXL 像素 LoRA，等距/非等距，无触发词，配 LCM 8 步。
- **Pixel Party XL**：整 UNet 替换的纯像素 checkpoint，128×128 出图。
- 关键后处理铁律：**生图后 8× 最近邻降采样**（绝不 bilinear/bicubic）+ 调色板量化，本项目 `pixelize.py` 已实现这部分。

### 5.5 结论

1. **当前 SD1.5 + 词典/机翻方案对"短语式像素素材描述"已经够用**，短句覆盖率 90-100%。
2. **若要显著提升中文长句/复杂语义**，首选换 **Kolors**（原生中文 + SDXL 生态 + 显存可控），在 Provider 层新增适配，保留 SD1.5 兜底。
3. Qwen-Image 中文文字渲染最强但更重、更偏海报场景，作为后续观察项。
4. 像素风格一致性可独立用「像素 LoRA + 最近邻降采样」增强，不必依赖换底座。

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
