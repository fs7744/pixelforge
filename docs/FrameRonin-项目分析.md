# FrameRonin 项目分析与 Skill 化评估

> 仓库：https://github.com/systemchester/FrameRonin （本地浅克隆：`D:/code/gen/FrameRonin`）
> 分析日期：2026-10-18 ｜ 代码版本：main @ V3

---

## 一、项目定位（一句话）

FrameRonin 是一个**像素游戏素材工作站**：前端（纯 TypeScript + Canvas）做全部确定性图像处理（视频拆帧、GIF 互转、抠图、Sprite Sheet、像素化、去水印），AI 生成部分则**不自建模型**，而是把几十个预调好提示词的 **Gemini Gem（nanobanana）链接当"生成器按钮"用**——它本质是「Gemini 生成入口的导航站 + 本地后处理管线」。

```
用户 → [Gemini Gem 出图] → [FrameRonin 本地管线后处理] → [可用素材]
         （提示词在 Gem 里）      （去水印/抠图/缩放/切 sheet）
```

---

## 二、架构

| 层 | 技术 | 职责 |
|---|---|---|
| 前端 | React + TS + Canvas API，gifuct-js / gifenc | 所有像素处理、GIF、Sprite Sheet、色度键抠图 |
| 后端 | FastAPI + Redis + rq Worker + FFmpeg + rembg(U2Net) | 视频任务管线（可选，服务端抠图） |
| 生成 | 外链 `gemini.google.com/gem/<id>` | 全部 AI 生图（零自建） |
| 部署 | GitHub Pages 仅前端；后端需 Docker 自建 | — |

**重要事实：源码中 grep 不到任何 `prompt` 字样。所有生成功能的提示词都不在仓库里，而是烤进了每个 Gemini Gem 的指令中。** 仓库只存了 Gem 链接（`frontend/src/lib/gemPixelUrls.ts`）+ 预览图 + 按钮文案。

---

## 三、生成类功能逐项：怎么做到的 / 提示词是什么

### 3.1 实现机制（所有生成功能统一）

每个"生成器"就是一个指向共享 Gemini Gem 的链接，例如：

```ts
// frontend/src/lib/gemPixelUrls.ts
export const GEM_V4TX3_URL = 'https://gemini.google.com/gem/1zerS4eXHUGNj2tj-63omHyFRo_4K5S7p?usp=sharing'
```

用户点击 → 打开 Gemini 对话（Gem 已内置该系统提示词）→ 用户在对话里描述需求/上传参考图 → 出图 → 回 FrameRonin 用本地管线后处理。**Gem 的指令内容对非成员不可见**（Google 不公开共享 Gem 的 system instruction），所以「提示词是什么」只能从命名和约定反推，无法逐字还原。

### 3.2 生成功能清单（按代码中的按钮枚举）

**A. RPG Maker 角色素材系列**（GEM_RPGMAKER_URL_V1 / V1_1 / V3 等 3 个 Gem）
- 用途：生成 RPG Maker 规格（3×4 格、每格 144×144 类）角色行走图
- 提示词约定（从 README 管线反推）：Gem 内约定输出固定网格排布，出图后交给本地「RPGMAKER 一键处理」收尾

**B. 像素角色生成器 V2 / V2.2 / V3**（GEM_V2_URL ×2、GEM_V3_URL）
- V2.3OT 版注明「支持更大的像素和 2:1 角色」

**C. 僵尸 B1 / B2**（GEM_MONSTER_ZOMBIE_B1/B2）——怪物素材

**D. 像素场景系列**（GEM_SCENE_URL ×4：正视角 / 45度 / 泰拉 / 街机）
- 像素风游戏场景图

**E. 立绘生成**（GEM_ILLUST_URL）——立绘风格角色

**F. 全人物动作系列**（nanobananaFullChar，6 个 Gem）：
| 按钮 | Gem | 说明 |
|---|---|---|
| 连生动作 V4Tx3 | GEM_V4TX3_URL | **代码注明「A0、A1、A2 关键字触发三套动作」**——Gem 内提示词约定用 A0/A1/A2 关键字切换动作集 |
| 横版人物生成 | GEM_HORIZONTAL_CHAR_URL | 横版动作帧 |
| 八方向 TopDown | GEM_8DIR_TOPDOWN_URL | 8 方向俯视角色 |
| 骑马动作生成 | GEM_HORSE_RIDING_URL | 骑马帧 |
| 一图全动作 / 2 | GEM_ONE_IMAGE_ALL_ACTIONS_URL(_2) | 单图内含全部动作 |

**G. AI 像素动物**（狗 / 鸟怪 / 只因，AiPixelAnimalsHub）——动物素材

**H. 生成大杂烩**（GEM_PIXEL_POTPOURRI）：吞食天地(S)、RPG骑马(潜墨千羽)、RPGMakerXP、RPGMakerVX、横版剑士一套(小苏早睡) 等社区作者共建的 Gem

**结论：生成侧没有任何可复用的提示词资产。** 唯一可提取的"提示词工程知识"是这些约定：
1. **关键字触发多动作集**（A0/A1/A2）
2. **输出格式烤进 Gem 指令**（几行几列、每格尺寸、透明底、无描边）
3. **一图全动作** vs **连生多图** 两种产出范式
4. 每个 Gem 只干一件事（单一职责），用「Gem 矩阵」而非一个万能 prompt

---

## 四、后处理管线逐项：怎么做到的（这部分提示词无关，全是确定性算法）

这是仓库的真正技术含量，也是唯一可以被完整复用的部分。

### 4.1 Gemini 水印去除（`frontend/src/lib/geminiWatermark.ts`）
- **算法：反向 Alpha 混合**。源自 allenk/GeminiWatermarkTool。`watermarked = α·logo + (1-α)·original` ⇒ `original = (watermarked - α·logo)/(1-α)`
- 内置 48/96px 两种水印的 logo alpha 图（base64 硬编码）
- 定位规则：右下角，margin = size/2×2（48→32，96→64）；图片 >1024px 用 96 否则 48

### 4.2 连通域色度键抠图（`ParamsStep/utils.ts` `applyChromaKeyContiguousFromTopLeft`）
- 取左上角 (0,0) 像素为背景色 → **4-邻域 flood fill** 只删与角点连通的同色区域（不会误删画面中部同色像素）
- 容差 80（欧氏距离 RGB）、羽化 5px 边缘过渡
- 这正是「AI 像素图去纯色背景」的标准解法

### 4.3 RPG Maker 一键处理（`ImageResizeStroke.tsx` handleOneClickProcess 等）
固定流水线（代码写死）：
```
去 Gemini 水印 → 取左上角色 → 连通域抠图(容差80/羽化5)
→ 144×144 硬缩放(最近邻) → 底部扩 48px 透明 → RPGMAKER 生成
```
- RPGMAKER 生成规则：3 行切分、第 2 行水平镜像复制、第 3 行下移 48px（补 RPG Maker 身位偏移）
- V2 变体：256² 缩放 → 右裁 64 / 下扩 64 → 第 3/4 行下移 → 5×3 格 → 144×240 / 裁 48 → 144×192
- 「一图全动作」：去水印 → 256² → 左上连通域抠图 → 右/下各裁 4px → 252×252

### 4.4 视频 → 序列帧 → Sprite Sheet
- 前端：色度键实时抠图（浏览器内完成）
- 后端（可选）：FFmpeg 抽帧 → rembg u2net 抠图 → 合成 sheet，RQ 队列 + 索引 JSON（`{i,x,y,w,h,t}` 帧坐标+时间戳）

### 4.5 其他处理能力
- **像素化**：算法对齐开源 Proper Pixel Art（Canny→闭运算→霍夫线→间距中值补网格→格内众色→先透明再放大），另移植了 theamusing/perfectPixel 的 FFT 周期估计 + Sobel 网格精修，专治「AI 像素图网格不齐」
- **GIF ↔ 序列帧**：gifuct-js 拆帧 / gifenc 合成，支持逐帧 delay
- **Sprite Sheet 调整**：分割预览、勾选帧、A/D 键切帧、按偏移重排后重组合导出
- **扩缩图**：N×M 格子裁切合并；**双背景抠图**（doubleBackgroundMatte，两张背景图差分抠像 + 8-连通微尘清理）
- **RSEPRITE**：内置一个网页像素画编辑器（图层/时间轴/调色板）
- **Seedance/即梦视频去水印**：后端 ffmpeg 处理（需自建）

---

## 五、Skill 化可行性评估

### 5.1 结论

**可以做成 skill，但要拆开做、只做对的一半。**

| 部分 | 能否 skill 化 | 判断 |
|---|---|---|
| 后处理管线（去水印/连通域抠图/RPGMaker 管线/sprite sheet/像素化对齐） | ✅ **非常适合** | 纯确定性算法，输入输出明确，可用 Python(PIL/numpy) 忠实复刻，agent 一键调用 |
| 生成功能（Gemini Gems） | ❌ **不可原样复刻** | 提示词不在仓库、烤在 Google 服务器里；skill 只能「参考其约定重建提示词」，且生成应走宿主自身的 ImageGen 能力，而非跳转外链 |
| 视频去水印 / RSEPRITE 编辑器 | ⚠️ 边缘 | ffmpeg 任务可做；编辑器与 agent 场景不匹配，跳过 |

### 5.2 与你本地已有技能的重叠（重要）

本地已装 `pixel-asset-forge`（v4.1.0，像素游戏素材生成器）——它的定位是「宿主模型直出像素素材 + 脚本工程规整（透明底/精确尺寸/精灵图/GIF/引擎 JSON）」。FrameRonin 的后处理管线与它**互补但不重叠**：

- pixel-asset-forge 缺的：**Gemini 水印去除**、**左上连通域抠图**、**RPG Maker 专属网格管线**、**perfectPixel 网格对齐**、**视频/GIF 拆帧**
- FrameRonin 缺的：模型直出能力（它靠外链）

→ **最优方案不是新开一个 skill，而是把 FrameRonin 的管线作为「后处理技能」并入/挂到现有像素素材工作流末端。**

### 5.3 建议的 Skill 蓝图（如需落地）

```
skill: frame-ronin-postprocess（像素素材后处理管线）
├── SKILL.md                     # 触发词：去Gemini水印/连通域抠图/RPGMaker处理/spritesheet/像素化对齐
├── scripts/
│   ├── gemini_watermark.py      # 反向alpha混合，内置48/96 logo掩码
│   ├── cc_chromakey.py          # 左上角flood fill去背(容差/羽化可调)
│   ├── rpgmaker_pipeline.py     # 一键/全动作/V2 三条写死管线
│   ├── sprite_sheet.py          # 拆分/合成/偏移重排/导出JSON索引
│   ├── pixel_align.py           # perfectPixel移植(FFT+Sobel网格对齐)
│   └── video_frames.py          # ffmpeg抽帧→抠图→sheet
└── references/
    └── pipelines.md             # 各管线参数表(144/192/240、48px偏移等)
```

**落地要点：**
1. 水印去除需要 48/96 两张 logo 掩码图——可从上游 GeminiWatermarkTool 项目的 `assets/embedded_assets.hpp` 提取，或对一张带水印图 + 无水印图差分自建
2. 连通域抠图在 numpy 里用迭代 flood fill（大图需注意栈深，建议 BFS + 行扫描优化）
3. 所有「硬缩放」必须是最近邻，否则像素素材会糊
4. 提示词重建：生成侧若要覆盖，按 FrameRonin 的约定写（A0/A1/A2 关键字、固定网格、透明底），但交给宿主 ImageGen 直出，不跳转外链

### 5.4 风险提示

- 仓库含大量二进制素材（mp4/png/gif/ogg），skill 化只取算法代码，**不要打包素材**
- Gemini 水印算法针对当前水印样式，Google 改版即失效，skill 需保留容错分支
- Gem 链接随时可能失效/被改权限，不能作为 skill 的硬依赖

---

## 六、一页总结

1. FrameRonin = **Gemini Gem 导航站 + 像素素材后处理管线**，生成侧提示词全部烤在 Google 的 Gem 里，仓库零 prompt 资产
2. 真正有价值的是 6 条确定性管线：反 Alpha 混合去水印、左上连通域抠图、RPGMaker 一键管线、Sprite Sheet、perfectPixel 网格对齐、视频拆帧
3. Skill 化**可行且价值高**，但应定位为「像素素材后处理技能」，与你已装的 pixel-asset-forge 衔接（它负责生成，本技能负责收尾），而不是复刻它的外链生成器
