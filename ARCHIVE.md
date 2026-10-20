# 归档说明

**归档日期**：2026-10-20 ｜ **代码版本**：`101a479`

---

## 一、这是什么

这是 PixelForge 像素工坊的**归档版本**。项目功能已冻结，不再在此开发。
后续开发在新的项目里进行，起步方向见 [docs/新项目起点.md](docs/新项目起点.md)。

想了解项目全貌（有什么功能、每个干什么、哪里有缺陷、什么还没做、
参考了哪些外部项目），读 **[docs/项目总览.md](docs/项目总览.md)**。

---

## 二、归档后的目录布局

代码和数据被**有意分开**：

```
D:\code\archive\pixelforge-2026-10\    ← 代码（23 MB，含完整 git 历史）
├── backend/          FastAPI 后端
├── frontend/         React + TS 前端（node_modules 已删）
├── docs/             全部文档
├── data/             两样静态示例图 + settings.json.example（配置留档）
├── start.sh / .bat   一键启动
└── .git/             完整历史，可随时回看任何一版

D:\code\pixelforge\                     ← 只剩模型权重
└── models\            215 GB 本地模型权重（后面还要用）
```

**为什么分开**：模型权重 215 GB，移动代价高且后续新项目仍要用，所以留在原位。

**为什么删掉 `data/`**（248 MB）：里面全是可再生的生成产物 ——
`tmp/` 189M（验证与冒烟的临时渲染）、`smoke/` 35M、`assets/` 21M（393 个生成的素材）、
`runs/` 631K（工作流运行状态）、`exports/` 8K。里面**没有不可再生的东西**：
唯一的配置 `settings.json` 里 `apiKey` 本来就是空的，已留档为
`data/settings.json.example`（记录模型路径约定）。

---

## 三、把归档项目跑起来

```sh
cd D:\code\archive\pixelforge-2026-10
./start.sh --setup          # 建 .venv、装依赖、构建前端（已删的会自动重装）
./start.sh                  # 启动，默认 http://127.0.0.1:8787
```

Windows 上双击 `start.bat` 也行。

### 需要手动处理的一处路径

**模型路径**：`backend/app/config.py` 按**项目根**解析 `models/`，
所以从归档目录启动会找不到 `D:\code\pixelforge\models`。三选一：
- 在「模型设置」页把 `model` 字段填绝对路径 `D:/code/pixelforge/models/<底座>`
- 建目录联接：`mklink /J D:\code\archive\pixelforge-2026-10\models D:\code\pixelforge\models`
- 用环境变量覆盖（见 README「环境变量」节）

数据目录不需要处理——原来的 393 个资产已按要求删除，新项目会建自己的目录
（或用 `PIXELFORGE_DATA_DIR` 指到别处）。

---

## 四、代码可读性检查

`.venv` 与 `frontend/node_modules` 已按要求删除（可再生，合计 5.5 GB）。
首次运行 `./start.sh --setup` 会自动重装。

若只想读代码 / 看文档，不需要装任何依赖。

---

## 五、git 历史

完整保留，可回看任何一版：

```sh
cd D:\code\archive\pixelforge-2026-10
git log --oneline              # 全部提交
git log --oneline -- docs/     # 只看文档演进
git show 101a479:docs/项目总览.md   # 看某一版的总览文档
```

主要里程碑（近期）：

| 提交 | 内容 |
|---|---|
| `101a479` | 项目总览 + 新起点文档（归档） |
| `a125fd5` | 序列帧循环段 + 循环接缝诊断 |
| `ecc93f6` | 序列帧轨道 + 编辑器内插帧（打通生成→编辑） |
| `4925e61` | GIF 导出（自写编码器） |
| `f8eb3a4` | 帧审查图接进编辑器 |

---

## 六、交接清单

从这份归档里**值得带走**的（详见 `docs/项目总览.md` 第三节）：

1. **「数学只有一处实现」** —— 预览与导出走同一套代码，所见即所得
2. **确定性优先，模型可选** —— 22 个管线里只有 6 个碰模型，其余纯 PIL/numpy
3. **失败要给可行动的下一步** —— 数字和提示都要能翻译成「所以你现在该做什么」
4. **验证要独立于被验证对象** —— 用注入错误确认检查本身有效
5. **测试数据隔离** —— `PIXELFORGE_DATA_DIR` 指向临时目录，不污染真实数据

**已知技术债**（新项目别继承）：`Editor.tsx` 2242 行、`PostProcess.tsx` 1156 行、
管线定义前后端双份、5 处绕过 `api.ts` 直接 fetch、无 UI 框架。