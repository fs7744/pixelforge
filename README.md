# PixelForge 像素工坊

本地优先的 AI 游戏素材工作站：一句话生成角色，自动抠图、像素化、拼精灵表、导出引擎工程。
所有数据落在本机 `data/`，不上传云端。

---

## 快速开始

只需要 **Python 3.10 – 3.13**（推荐 3.12 或 3.13）。构建界面还需要 **Node.js 18+**（没有也能跑 API）。

> 不要用 Python 3.14/3.15：PyYAML 等依赖还没有预编译 wheel，pip 会退回源码编译，
> 而 Windows 上默认没有 MSVC 编译器，安装会以 `Microsoft Visual C++ 14.0 or greater is required` 失败。
> 脚本会优先挑 3.10–3.13；万一只装得过新的版本，会自动去掉 `uvicorn[standard]` 的 extras 兜底。

```bash
# Windows：双击 start.bat
# macOS / Linux：
./start.sh
```

首次运行会自动完成：建 `.venv` → 装 Python 依赖 → 构建前端 → 启动服务 → 打开浏览器。
之后再运行会跳过已完成步骤，通常 1-2 秒起来。

如果之前用 3.14/3.15 建过 `.venv`，脚本会识别出版本越界，自动把旧目录改名成 `.venv.bak-<时间戳>` 并重建，不用手删。

打开 <http://127.0.0.1:8787>。端口被占用时会自动顺延到下一个空闲端口，启动信息里会打印实际地址。

### 常用参数

| 命令 | 作用 |
|---|---|
| `./start.sh` | 启动并打开浏览器 |
| `./start.sh --no-browser` | 启动但不开浏览器 |
| `./start.sh --port 9000` | 指定端口（占用时仍会自动顺延） |
| `./start.sh --reload` | 开发模式，改后端代码自动重载 |
| `./start.sh --setup` | 只准备环境，不启动 |
| `./start.sh --check` | 自检：打印 Python / 依赖 / torch / 前端 / 模型 / 端口状态 |
| `./start.sh --gpu` | 额外安装 AI 推理依赖（torch + diffusers，约 2.9GB） |

服务跑起来后，设置页还有「环境体检」按钮（`GET /api/doctor`）：逐项检查
依赖 / GPU / 模型底座，缺什么直接给修复命令，不用回文档翻排错表。

Windows 下把 `./start.sh` 换成 `start.bat`，参数相同。

### 环境变量

| 变量 | 说明 |
|---|---|
| `PF_PYTHON` | 指定解释器路径，跳过 `.venv` 探测 |
| `PF_VENV` | 虚拟环境目录（默认项目内 `.venv`） |
| `PF_PIP_INDEX` | 自定义 pip 源；默认先用本机配置，失败自动切阿里云 |
| `TORCH_INDEX` | torch wheel 源，默认 CUDA 12.4；纯 CPU 环境可设 `.../whl/cpu` |
| `HOST` / `PORT` | 监听地址，默认 `127.0.0.1:8787` |

### 没有模型时能用什么

AI 生成需要本地模型，**其余功能全部可用**：模板库、素材库、抠图、像素化、
尺寸规整、无缝延展（镜像模式）、瓦片修补、分表合表、去水印（掩码模式）、
ZIP 批量导出。启动脚本会明确提示模型状态。

---

## 让 AI 生成跑起来（可选）

需要 NVIDIA 显卡。**Z-Image 约 21GB 显存，Qwen-Edit 用 fp8 量化后约 20GB**，两者不会同时驻留。

```bash
# 1. GPU 推理依赖（torch cu124 + diffusers + transformers，约 2.9GB）
./start.sh --gpu          # Windows：start.bat --gpu
# 等价于直接跑：bash backend/install_gpu_deps.sh

# 2. 生成底座：Z-Image-Turbo（约 20GB，文生图，中文提示词直喂）
bash backend/fetch_zimage.sh

# 3. 编辑底座：Qwen-Image-Edit-2511（约 54GB，可选；角色一致性、AI 去水印/延展/画风重绘要用）
bash backend/fetch_qwen_edit.sh

# 4. 视频底座：Wan2.2 TI2V-5B（约 32GB，可选；「动画工作流 · 视频生成」要用）
#    走 ModelScope（huggingface.co 在本机代理下不通），支持断点续传
PYTHONPATH=. python backend/fetch_wan22.py

# 5. 分层底座：Qwen-Image-Layered（约 57.7GB，可选；场景语义分层、视差工作流要用）
PYTHONPATH=. python backend/fetch_qwen_layered.py
```

> **为什么 torch 不随 `start.sh` 一起装**：它单独就占 2.9GB，而抠图、像素化、延展
> 这些确定性功能根本不需要它。所以默认不装，只在检测到缺失时提示。
> 如果没装就点「生成」，会看到 `No module named 'torch'` —— 跑一次 `--gpu` 即可。

下载都支持断点续传，中断后重跑即可接着下。装完在「设置 → 内置推理服务」里确认模型路径，
点「测试连接」。模型目录换了位置就改这里的路径，不用改代码。

| | Z-Image-Turbo | Qwen-Image-Edit-2511 | Wan2.2 TI2V-5B | Qwen-Image-Layered |
|---|---|---|---|---|
| 分工 | 文生图（一句话 → 角色/场景） | 参考图生成 / 指令式编辑 | 图生视频（角色图 + 动作 → 连续帧） | 单图拆语义层（一张场景 → N 张 RGBA） |
| 规模 | 6B · 8 步蒸馏 · 约 5s/张 | 20B · 40 步 · 约 3-4 分钟/张 | 5B · 30 步 · 33 帧约 60-90 秒 | 20B · 50 步 · 4 层约 1-2 分钟 |
| 显存 | bf16 全载约 21GB | fp8 权重约 20GB | transformer fp8 9.3GB + umT5 fp8 5.3GB + VAE 2.6GB | fp8 权重约 20.5GB |

视频底座解决的是「逐帧编辑帧间不连续」：Qwen-Edit 一帧一帧改姿势，每帧独立推理，
快速动作会抖；Wan 一次生成整段动作，肢体路径天然连贯，抽帧后就是现成的循环素材
（h3-character-studio 用 MiniMax H3 是同一思路）。第一次加载会把 transformer 量化
成 fp8 并落盘到 `models/wan2.2-ti2v-5b/transformer-fp8/`，之后冷启动直接命中，
比重新量化快约 30 秒。

两者不能同时驻留，Provider 会自动换出另一族，日志里能看到「显存换出」。

**图生图（参考图生成）**：工作台点「参考图」加图，可以直接上传（最多 4 张），
也可以「从资产库选」——刚生成的角色不用下载再上传。带了参考图就会自动改走
Qwen-Image-Edit（Z-Image 是纯文生图，没有参考图输入），输出尺寸沿用工作台的宽/高。
多张用于组合语义，例如「角色正面 + 侧面」保持同一角色、「角色 + 场景」定风格氛围。
对应接口 `POST /api/generate/ref`（`refs` 文件 + `refAssetIds` 资产 id 可混用；
宽高传 0 则跟随第一张参考图）。

**可选：深度抠图权重**（176MB，不装则退到 GrabCut，效果略差）

```bash
mkdir -p ~/.u2net && curl -L -o ~/.u2net/u2net.onnx \
  "https://modelscope.cn/models/AI-ModelScope/u2net/resolve/master/u2net.onnx"
```

---

## 功能

| 模块 | 说明 |
|---|---|
| 生成工作台 | 10 个提示词模板 × 6 分类，15 种游戏美术画风（含 HD-2D），参数面板，结果网格，一键送后处理 |
| 画风转换 | 图片 → 指定画风重绘：画风分组画廊（带示例图）、前后拖动对比、会话内多次转换留档、结果可接着再转 |
| 人物像素化 | 照片 → 像素角色：三档造型程度（写实比例 / 半 Q 版 / Q 版）+ 补充提示词 + 背景三选一（保留 / 绿幕 / 透明），像素密度与色数可随时微调（不重跑 AI） |
| 角色三视图 | 无限画布式流程：正面 → 侧面 → 背面 → 拼板，四步分开跑，任何一步不满意单独重跑 |
| 素材库 | 本地落盘 + 索引 + 筛选 + 批量 ZIP |
| 视频/GIF 导入 | 录像/GIF 拆成帧资产（GIF 原生 / MP4 走 ffmpeg），可选逐帧抠图，绿幕录像直接得到透明帧 |
| 循环段选择 | 拆帧结果里取 [start, stop) 连续段合成精灵表：不插帧、不跳帧、不改时长，附审查图 |
| 参考图规范化 | 裁主体 → 方形画布 → 底部对齐统一基线，参考图几何一致后重跑结果才可比 |
| 场景语义分层 | 一张场景图拆成 N 张**同坐标** RGBA 图层（拆同一张图，不是各画一遍），每层带 alpha；给出覆盖率与「远→近」启发式排序，可在导图前肉眼确认 |
| 帧审查图 | 逐帧合成时附帧网格图：序号 + 逐帧差异 + 循环接缝标色，和指标数字配成一套 |
| 动画编辑器 | 骨骼（父子级联）+ **IK 目标（FABRIK 2-4 节链、可逐帧混合权重、pole 定肘向）** + 逐轨关键帧（位移/旋转/缩放/透明度）+ **贝塞尔曲线编辑器（可过冲回弹）** + 多轨合成（正常/相加/正片叠底）+ **3×3 自由变形（可打关键帧）** + **骨骼权重蒙皮（自动权重 + 权重笔刷）** + 5 种攻击特效笔刷（slash/bristle/dry/spark/echo）+ 洋葱皮 + 播放与 ZIP 导出 |
| 法线 / 材质掩码 | 逐帧几何法线（内缩距离 78% + 亮度 22% 反推高度场）+ 金属高光近似掩码；OpenGL 切空间，可批量导出整段；编辑器导出可一并带上 |
| 生成进度 / 取消 | 所有 AI 生成实时显示「第 x/y 步 + 已用时长」，可随时取消（在下一个推理步生效；模型加载与文本编码阶段不可中断） |
| 图片预览 | 点击任意图片全屏查看，滚轮缩放、拖拽平移，放大 ≥2 倍用最近邻渲染（像素不糊） |
| 抠图 | u2net 深度抠图 → GrabCut → 颜色键三级降级；硬边 alpha + 边缘去污，不产生半透明白边 |
| 像素化 | 分块单色 + 调色板量化（自动 8/16/32、GameBoy、NES）+ 可选抖动 |
| 尺寸规整 | 最近邻硬缩放 / 等比留白 / RPGMaker V1·V2 三档 |
| 无缝延展 | 镜像拼接（接缝数学连续）+ AI 续画（Qwen-Edit 顺着场景往外画） |
| 瓦片修补 | 偏移-修复法，保留图案结构只修接缝 |
| 分表 / 合表 | N×M 拆帧 + 引擎索引 JSON |
| 去水印 | 反向 Alpha 混合（需掩码）+ AI 修补（无需掩码，按周围内容补全） |
| 工作流 | 场景（3 步）/ 动画（7 步）/ 动画 · 视频生成（7 步）/ 四方向行走（3 步）/ 视差（5 步：基准图 → 语义分层 → 规整 → 无缝 → 工程导出），单步重跑、产物自动传递、状态落盘；插帧默认走双向光流，精灵表合成默认全局锁色并输出一致性指标 |
| Godot 导出 | 精灵：`SpriteFrames` + `AnimatedSprite2D` 场景 + 索引 + 说明 + SHA256SUMS 清单；四方向工作流另出四向包（4 组动画 + 方向控制器脚本）。动画编辑器另出**骨架包**：`anim.json`（逐帧骨骼世界矩阵 + 刚性轨道仿射 / 蒙皮轨道烘焙顶点）+ 纹理 + `anim_player.gd` 参考运行时 |
| AI 标识 | 默认关闭；下载时可用 `?label=1` 临时补角标 |

### 动画编辑器的边界（写清楚，免得踩空）

- **骨骼是 2D FK + 可选 IK**：FK 下父转子跟；加 IK 目标后，链上骨骼改由 FABRIK 解算，
  拖红叉手柄即摆姿势（目标点本身可以打关键帧）。没有约束、没有蒙皮权重、没有极向目标
  （pole target），链长最多 4 节。
- **关键帧是逐轨 + 逐骨骼 + 逐 IK 目标**：位移/旋转/缩放/透明度；缓动可以是命名档
  （线性/缓入/缓出），也可以是曲线编辑器里拖出来的 cubic-bezier（y 允许出框，用来做过冲与回弹）。
  没有曲线**叠加**（多条曲线相加）与循环修改器。
- **自由变形可打关键帧**：3×3 控制点双线性变形，`warpKeys` 之间做插值（也吃贝塞尔）。
  3×3 是粗网格，做大变形会看出折痕——要更贴合就用下面的蒙皮。
- **骨骼蒙皮**：默认按「顶点到骨段的距离」自动算权重（取最近两根归一化）；
  「权重」模式下可手刷——选骨骼、刷入/刷出、调半径与强度，热力图实时显示
  （红=完全不跟、蓝=完全跟），刷过的顶点数与清除按钮都在面板上。
  网格 4–16 可选，渲染用三角形分片仿射（不是 PIL 的 MESH，那个只会贴到包围盒）。
  **网格密度改了必须重刷**——手刷数据按顶点下标存，改密度会错位（面板里有提示）。
  没有扭转骨骼（twist bone），四足那种长躯干要拆成多节骨骼才不扭。
- **IK 权重可以逐帧打点**（0=跟 FK，1=跟目标），过渡自然；pole（肘/膝朝哪边）是**静态**的，
  不能逐帧变——同一个 IK 想在两个镜头里换肘向，得建两个 IK 目标。
- **特效笔迹是全帧可见的**：笔迹带 `from/to` 帧区间，界面默认铺满整段；同一份笔迹任何尺寸下重渲染一致（随机走固定种子）。
- **导出走服务端合成**：预览和导出用同一套渲染代码，所见即所得；单帧是几十毫秒的 CPU 操作。

**法线不是 h3 的标注传播**：h3 需要人工标注关键姿态的形体再双向传播；本项目从像素反推
（内缩距离 + 亮度），薄片、遮挡边界、正对镜头的平面都会有偏差，够做 2D 光照，不够做精确形体。

**生成底色（抠图质量的根因在生成侧，不在抠图参数）**

默认底色是**中性灰**而不是绿幕。实测（640×896、同一提示词同一种子，只改背景色）：

| 底色 | 模型实际画出 | 内圈偏底色像素 | 角色染色 |
|---|---|---|---|
| 绿幕 #00FF00 | **[90,199,126] 薄荷绿 + 径向渐变** | **76.5%** | 披风/衣领被染绿 |
| 中性灰 #808080 | [178,178,177] 极均匀 | **33.3%** | 无 |

两个结论：
1. **模型画不出纯色底**——声明 #00FF00 给的是薄荷绿，还带中心亮四周暗的径向渐变。
   色键的固定容差在这种底上必然留脏边，这不是容差调小了的问题。
2. **绿幕脏的主因不是「难抠」，是模型主动把绿染进角色**——绿色高饱和、与角色配色
   冲突，模型为了画面协调会把它带到人物身上。同位置绿底图均值色 `[91,121,80]`（偏绿），
   中性灰 `[156,125,124]`（正常）。色键调参对此无能为力，那些像素本身就是错的。

所以：抠图走语义分割（u2net），它根本不看背景色；生成侧铺中性底，避免污染。
绿幕保留为可选（行业惯例，方便在引擎里自己键控），想要绝对纯的绿幕用「人物像素化」——
它的绿幕是本地抠完人物后自己合成的纯 #00FF00，模型从不参与。
接地阴影（比底色暗，色键必判成前景）两种底色都有，靠语义分割忽略。

工作流 API：
```
POST /api/workflows/{id}/runs          创建工作流（输入 + 参数）
POST /api/runs/{id}/steps/{step}/run   执行单步（可重跑）
POST /api/runs/{id}/run-all            顺序执行剩余步骤
GET  /api/runs/{id}                    查询步骤状态与产物
```

步骤类型：`generate` / `dedup` / `blend` / `compose` / `normalize` / `montage` / `godot` / `godot_parallax`。

角色三视图 API（工作台的流程画布走这条，一步一请求，单步可重跑）：

```
POST /api/pipeline/threeview/step
     { view: "front", prompt, style, width, height, steps, cfg, seed }
     { view: "side" | "back", baseAssetId, ... }      # baseAssetId = 正视图资产 id
     { view: "compose", viewAssetIds: [正, 侧, 背], width, height }
→ { images: [asset], step, meta }

POST /api/pipeline/threeview      # 一次性跑完四步（工作流 / 脚本用）
```

分工：正视图是唯一需要文生图的一步（编辑器不能从零创作），侧/背都由它推导，
所以重跑正视图会让后面三张自动作废；重跑侧/背只作废拼板。

---

## 排错

| 现象 | 处理 |
|---|---|
| `未找到 Python 3.10+` | 装 Python 并勾选「Add python.exe to PATH」；或 `PF_PYTHON=/path/to/python ./start.sh` |
| `Microsoft Visual C++ 14.0 or greater is required` | 解释器版本过新（3.14/3.15），部分依赖没有 wheel。装 Python 3.12/3.13 后重跑即可，脚本会自动重建 `.venv` |
| 创建虚拟环境失败（Linux） | `sudo apt install python3-venv` |
| 依赖装不上 / 很慢 | `PF_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple ./start.sh` |
| 界面空白 / 打不开 | 装 Node.js 18+ 后重跑 `./start.sh`；也可 `cd frontend && npm install && npm run build` |
| 端口被占用 | 默认自动顺延；也可 `--port 9000` |
| 生成报错「模型不存在」 | 按上面下载模型，并在设置页确认路径 |
| 分层报错「未找到 Qwen-Image-Layered 底座」 | 这张图要用分层能力，先跑 `PYTHONPATH=. python backend/fetch_qwen_layered.py`（约 57.7GB）；其它功能不受影响 |
| 生成报错 `No module named 'torch'` | 只装了 Web 依赖，没装推理依赖。跑 `./start.sh --gpu`（Windows `start.bat --gpu`） |
| MP4 导入提示找不到 ffmpeg | `winget install ffmpeg`（或 https://ffmpeg.org/download.html）；GIF 不需要 ffmpeg |
| 抠图边缘有杂色 | 确认 `~/.u2net/u2net.onnx` 存在（深度抠图）；否则是 GrabCut 兜底 |
| 显存不足 | 同一时间只能驻留一个底座；设置页点「卸载模型」再切另一个 |

---

## 测试

```bash
# 确定性管线冒烟（无需模型，约 10s）
python backend/smoke_test.py

# API 端到端（无需模型）
python backend/e2e_api_test.py

# 逐路由覆盖（无需模型）
python backend/coverage_api_test.py

# GPU 端到端（需模型，较慢）
python backend/e2e_gpu_test.py
python backend/e2e_gpu_gap_test.py
```

测试资产会自动清理，不会污染素材库。

---

## 目录

```
pixelforge/
├── start.sh / start.bat     # 一键启动（自动建环境、装依赖、构建前端）
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI 路由
│   │   ├── config.py        # settings.json 持久化
│   │   ├── workflows.py     # 工作流引擎
│   │   ├── providers/       # 本地双底座推理
│   │   └── pipelines/       # matte / pixelize / resize / sheet / seamless / tile / watermark / ai_edit
│   ├── requirements.txt     # 核心依赖
│   ├── requirements-optional.txt  # 可选：rembg 深度抠图
│   └── install_gpu_deps.sh  # GPU 推理依赖
├── frontend/src/            # React + Vite
├── models/                  # 本地模型权重（不入库）
└── data/                    # 素材库、设置、工作流状态、导出包（不入库）
```

---

## 更多

架构、模型选型、踩坑结论与待办见 [`docs/DESIGN.md`](docs/DESIGN.md)。
API 交互文档在服务启动后访问 <http://127.0.0.1:8787/docs>。
