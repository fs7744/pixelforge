export type AssetFacet = { id: string; label: string; count: number }

/** 筛选栏数据：kind=是什么（生成/抠图/帧…），pipeline=哪条管线产出的（三视图…） */
export type AssetFacets = {
  total: number
  kinds: AssetFacet[]
  pipelines: AssetFacet[]
}

export type Asset = {
  id: string
  name: string
  kind: string
  file: string
  url: string
  width: number
  height: number
  createdAt: number
  meta?: Record<string, any>
}

export type TextGuide = {
  needsImage: boolean
  targets: string[]
  steps: string[]
  edit: string
  after: string
}

export type TemplateField = {
  id: string
  label: string
  default: string
  /** 选了参考图之后该字段该变成什么（精灵表：角色 → 照参考图） */
  fromRef?: string
  hint?: string
  presets?: { label: string; value: string }[]
}

export type Template = {
  id: string
  category: string
  name: string
  hint: string
  prompt: string
  /** 页面单独展示的提示，非提示词内容（不进模型） */
  notes?: string[]
  /** image = 本地出图；text = 文本任务，本页不出图 */
  kind?: 'image' | 'text'
  /** 模板默认画风 id；画风不写进 prompt，避免和选择器打架 */
  defaultStyle?: string
  /** 参考原图的编辑/提取类模板：画风跟随原图，选画风无效，前端不显示画风区 */
  styleLocked?: boolean
  /** 有参考图时画风跟随原图（编辑底座保画风），不前置画风词，也不显示画风区 */
  styleFollowsRef?: boolean
  /**
   * 需要用户单独填的内容（角色、动作序列…）。
   * prompt 里用 `{id}` 占位，接口返回时已用 default 填好；
   * 用户改字段 → 前端把提示词里那段旧值替换成新值。
   */
  fields?: TemplateField[]
  /** 仅 text：引导用户去别的对话式 AI 完成，不在本页做操作面板 */
  guide?: TextGuide
  /**
   * image 模板的第二条路：复制提示词交给外部出图 AI。
   * prompt 与主 prompt 分开写——本地底座吃短句，外部模型能接受结构化约束。
   */
  textMode?: {
    hint: string
    prompt: string
    guide: TextGuide
  }
  params: Record<string, number>
}

export type StyleInfo = {
  id: string
  name: string
  /** 画廊分区（像素 · HD-2D / 动漫卡通 / 写实 · 3D …） */
  group?: string
  tags: string[]
  hint: string
  /** 实际拼进提示词的风格词（前置到开头），用于界面展示「加的是什么」 */
  words?: string
  pixelize?: { block: number; colors: number }
  /** 画风示例图 URL（gen_style_samples.py 生成过才有） */
  sample?: string | null
}

/** 人物像素化的一档：造型语言（头身比）+ 像素密度 + 色数 */
/** 生成底色（抠图质量的根因在生成侧） */
export type Backdrop = {
  id: string
  name: string
  hint: string
  hex: string
  /** 底色是否适合做色键（灰/白不适合，色键要底色够纯且角色不同色） */
  chromaKeyable: boolean
}

export type CharPixelLevel = {
  id: string
  name: string
  short: string
  hint: string
  heads: string
  /** 成品横向格数（分辨率无关的像素密度） */
  grid: number
  colors: number
  /** 档位示例图（gen_char_pixel_samples.py 生成过才有） */
  sample?: string | null
}

export type ProviderInfo = {
  id: string
  label: string
  active: boolean
  configured: boolean
  hasKey: boolean
  model: string
}

export type ProviderCfg = {
  label: string
  baseUrl: string
  apiKey: string
  model: string
  /** 负向提示词（仅 Qwen-Image-Edit 编辑底座生效） */
  negativePrompt?: string
  enabled: boolean
  hasKey?: boolean
}

export type Settings = {
  activeProvider: string
  providers: Record<string, ProviderCfg>
  defaults: { steps: number; cfg: number; width: number; height: number }
  aiLabel: boolean
}

/** 体检项（移植自 FrameBaker doctor.ts）：detail 里带可执行的修复命令 */
export type DoctorCheck = { id: string; ok: boolean; label: string; detail: string }
export type DoctorReport = {
  ok: boolean
  summary: string
  checks: DoctorCheck[]
  activeProvider: string
  torch: boolean
}

const json = async (res: Response) => {
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try {
      const data = await res.json()
      msg = data.detail || msg
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  health: () => fetch('/api/health').then(json),
  /** 运行环境体检：依赖 / GPU / 模型底座逐项检查，detail 带修复命令 */
  doctor: (): Promise<DoctorReport> => fetch('/api/doctor').then(json),
  /** 取消任务：provider 下一步回调抛 CancelledError 中止推理 */
  cancelTask: (id: string) =>
    fetch(`/api/tasks/${id}/cancel`, { method: 'POST' }).then(json),
  templates: () => fetch('/api/templates').then(json),
  styles: () => fetch('/api/styles').then(json),
  providers: () => fetch('/api/providers').then(json),
  settings: () => fetch('/api/settings').then(json),
  saveSettings: (payload: Partial<Settings>) =>
    fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  testProvider: (id: string) => fetch(`/api/providers/${id}/test`, { method: 'POST' }).then(json),
  generate: (payload: {
    prompt: string
    provider?: string
    width: number
    height: number
    steps: number
    cfg: number
    seed?: number | null
    n: number
    templateId?: string
    style?: string
    /** 进度总线任务 id：传了才能订阅 SSE 进度与取消 */
    taskId?: string
  }) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  // 参考图生成（图生图）：1–4 张参考图 + 指令 → Qwen-Image-Edit。
  // 后端会自动改用编辑底座（Z-Image 是纯文生图，没有参考图输入）。
  generateRef: (
    payload: {
      prompt: string
      style?: string
      width?: number
      height?: number
      steps?: number
      cfg?: number
      seed?: number | null
      n?: number
      refAssetIds?: string[]
      taskId?: string
    },
    files: File[] = [],
  ) => {
    const fd = new FormData()
    fd.append('prompt', payload.prompt)
    if (payload.style) fd.append('style', payload.style)
    // 0 表示「跟随参考图尺寸」，交给后端判断
    fd.append('width', String(payload.width ?? 0))
    fd.append('height', String(payload.height ?? 0))
    fd.append('steps', String(payload.steps ?? 0))
    fd.append('cfg', String(payload.cfg ?? 0))
    fd.append('n', String(payload.n ?? 1))
    if (payload.seed != null) fd.append('seed', String(payload.seed))
    if (payload.refAssetIds?.length) fd.append('refAssetIds', payload.refAssetIds.join(','))
    if (payload.taskId) fd.append('taskId', payload.taskId)
    files.forEach((f) => fd.append('refs', f))
    return fetch('/api/generate/ref', { method: 'POST', body: fd }).then(json)
  },
  // 角色三视图：正/侧/背三张独立生成 + 拼板（images[0]=拼板，1-3=正/侧/背单视图）
  threeview: (payload: {
    prompt: string
    style?: string
    steps?: number
    cfg?: number
    seed?: number | null
    width: number
    height: number
    maxSide?: number
    taskId?: string
  }) =>
    fetch('/api/pipeline/threeview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  // 分步版：view = front | side | back | compose，一次只做一步，可单独重跑
  threeviewStep: (payload: {
    view: 'front' | 'side' | 'back' | 'compose'
    prompt?: string
    style?: string
    steps?: number
    cfg?: number
    seed?: number | null
    width: number
    height: number
    maxSide?: number
    baseAssetId?: string
    viewAssetIds?: string[]
    taskId?: string
  }) =>
    fetch('/api/pipeline/threeview/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  /**
   * 资产列表。kind 与 pipeline 都是「是什么 / 从哪来」，两个都传时取交集。
   * 三视图属于 kind=generated，只能靠 pipeline=threeview 单独筛出来。
   */
  assets: (kind?: string, pipeline?: string) => {
    const qs = new URLSearchParams()
    if (kind && kind !== '全部') qs.set('kind', kind)
    if (pipeline && pipeline !== '全部') qs.set('pipeline', pipeline)
    const q = qs.toString()
    return fetch(`/api/assets${q ? `?${q}` : ''}`).then(json)
  },
  assetFacets: (): Promise<AssetFacets> => fetch('/api/assets/facets').then(json),
  /**
   * 缩略图地址：等比缩放、不裁切，所以三视图那种 3:1 的宽条在列图里也是宽条。
   * 直接用 <img src>，不要 fetch——后端带 long immutable 缓存，浏览器会自己复用。
   */
  assetThumb: (id: string, w: 96 | 160 | 320 = 160) => `/api/assets/${id}/thumb?w=${w}`,
  delAsset: (id: string) => fetch(`/api/assets/${id}`, { method: 'DELETE' }).then(json),
  matte: (file: File, tolerance: number, feather: number) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('tolerance', String(tolerance))
    fd.append('feather', String(feather))
    return fetch('/api/pipeline/matte', { method: 'POST', body: fd }).then(json)
  },
  resize: (file: File, width: number, height: number, mode: string, variant: string) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('width', String(width))
    fd.append('height', String(height))
    fd.append('mode', mode)
    fd.append('variant', variant)
    return fetch('/api/pipeline/resize', { method: 'POST', body: fd }).then(json)
  },
  /** 参考图规范化（h3 normalize-master）：裁主体 → 方形画布 → 底部对齐基线 */
  refNorm: (
    file: File,
    payload: { size?: number; padRatio?: number; baseline?: number; tolerance?: number } = {},
  ) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('size', String(payload.size ?? 0))
    fd.append('padRatio', String(payload.padRatio ?? 0.06))
    fd.append('baseline', String(payload.baseline ?? 0.94))
    fd.append('tolerance', String(payload.tolerance ?? 24))
    return fetch('/api/pipeline/refnorm', { method: 'POST', body: fd }).then(json)
  },
  /** 场景语义分层：一张场景图 → N 张同坐标 RGBA 图层（Qwen-Image-Layered） */
  sceneLayers: (
    payload: {
      layers?: number
      steps?: number
      seed?: number | null
      resolution?: number
      taskId?: string
      assetId?: string
    } = {},
    file?: File | null,
  ) => {
    const fd = new FormData()
    if (file) fd.append('file', file)
    if (payload.assetId) fd.append('refAssetId', payload.assetId)
    fd.append('layers', String(payload.layers ?? 4))
    fd.append('resolution', String(payload.resolution ?? 640))
    if (payload.steps) fd.append('steps', String(payload.steps))
    if (payload.seed != null) fd.append('seed', String(payload.seed))
    if (payload.taskId) fd.append('taskId', payload.taskId)
    return fetch('/api/pipeline/scene/layers', { method: 'POST', body: fd }).then(json)
  },
  /** 循环段选择（h3 select）：取 [start, stop) 连续帧合成精灵表 + 审查图 */
  spriteSelect: (payload: {
    frameAssetIds: string[]
    start: number
    stop: number
    cols?: number
    size?: number
    matte?: boolean
    colors?: number
  }) =>
    fetch('/api/pipeline/sprite/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  splitSheet: (file: File, cols: number, rows: number, matte = false, tolerance = 80) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('cols', String(cols))
    fd.append('rows', String(rows))
    fd.append('matte', String(matte))
    fd.append('tolerance', String(tolerance))
    return fetch('/api/pipeline/sheet/split', { method: 'POST', body: fd }).then(json)
  },
  /**
   * 逐帧动作生成：一次一帧（用户填的动作原样进编辑指令）。
   * 参考图两种来源：资产库 id，或直接传本地文件。
   */
  spriteFrame: (payload: {
    pose: string
    index: number
    total: number
    refAssetId?: string
    subject?: string
    style?: string
    steps?: number
    seed?: number | null
    width?: number
    height?: number
    /** 生成时铺什么底色；不传则后端默认「中性灰」（比绿幕干净，见 /backdrops） */
    backdrop?: string
    taskId?: string
  }, file?: File | null) => {
    const fd = new FormData()
    fd.append('pose', payload.pose)
    fd.append('index', String(payload.index))
    fd.append('total', String(payload.total))
    if (payload.refAssetId) fd.append('refAssetId', payload.refAssetId)
    if (payload.subject) fd.append('subject', payload.subject)
    if (payload.style) fd.append('style', payload.style)
    if (payload.steps) fd.append('steps', String(payload.steps))
    if (payload.seed != null) fd.append('seed', String(payload.seed))
    if (payload.width) fd.append('width', String(payload.width))
    if (payload.height) fd.append('height', String(payload.height))
    if (payload.backdrop) fd.append('backdrop', payload.backdrop)
    if (payload.taskId) fd.append('taskId', payload.taskId)
    if (file) fd.append('file', file)
    return fetch('/api/pipeline/sprite/frame', { method: 'POST', body: fd }).then(json)
  },
  /** 逐帧结果 → 共享缩放/基线对齐/抠图 → 精灵表 + 引擎索引。
   *  colors > 0：取全局色板再让每帧映射回它（帧间不偏色）；0 = 每帧各自量化 */
  spriteCompose: (payload: {
    frameAssetIds: string[]
    cols: number
    size?: number
    matte?: boolean
    colors?: number
  }) =>
    fetch('/api/pipeline/sprite/compose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  /**
   * 图片画风转换：把一张图重绘成目标画风（主体/构图沿用原图，只换笔触与上色）。
   * 输入二选一：assetId（素材库里的图，不用先下载再上传）或 file（本地文件）。
   * 单次 40 步约 2-4 分钟；像素类画风默认在 AI 重绘后自动像素化。
   */
  restyle: (
    payload: {
      style: string
      assetId?: string
      steps?: number
      seed?: number | null
      maxSide?: number
      pixelize?: boolean
      taskId?: string
    },
    file?: File | null,
  ) => {
    const fd = new FormData()
    fd.append('style', payload.style)
    if (payload.assetId) fd.append('refAssetId', payload.assetId)
    if (payload.steps) fd.append('steps', String(payload.steps))
    if (payload.seed != null) fd.append('seed', String(payload.seed))
    if (payload.maxSide) fd.append('maxSide', String(payload.maxSide))
    fd.append('pixelize', String(payload.pixelize ?? true))
    if (payload.taskId) fd.append('taskId', payload.taskId)
    if (file) fd.append('file', file)
    return fetch('/api/pipeline/restyle', { method: 'POST', body: fd }).then(json)
  },
  /**
   * 生成底色选项：中性灰（默认）/ 绿幕 / 白。
   * 默认不是绿幕——模型画不出纯色绿，还会把绿染到角色边缘（实测内圈污染
   * 76.5% vs 中性灰 33.3%）。
   */
  backdrops: (): Promise<{
    backdrops: Backdrop[]
    default: string
  }> => fetch('/api/pipeline/backdrops').then(json),

  /** 人物像素化三档（写实比例 / 半 Q 版 / Q 版）+ 背景处理选项 */
  charPixelLevels: (): Promise<{
    levels: CharPixelLevel[]
    backgrounds: { id: string; name: string; hint: string }[]
  }> => fetch('/api/pipeline/char-pixel/levels').then(json),
  /**
   * 人物照片 → 像素角色：AI 按档位重绘 + 确定性像素化。
   * 返回 { asset: 像素化成品, draft: AI 原稿 }；原稿留着给「重新像素化」用。
   */
  charPixel: (
    payload: {
      level: string
      prompt?: string
      assetId?: string
      steps?: number
      seed?: number | null
      maxSide?: number
      grid?: number
      colors?: number
      /** keep 保留原图 / green 绿幕 / alpha 透明（生成时就要定，改了得重生成） */
      background?: string
      taskId?: string
    },
    file?: File | null,
  ) => {
    const fd = new FormData()
    fd.append('level', payload.level)
    if (payload.prompt) fd.append('prompt', payload.prompt)
    if (payload.assetId) fd.append('refAssetId', payload.assetId)
    if (payload.steps) fd.append('steps', String(payload.steps))
    if (payload.seed != null) fd.append('seed', String(payload.seed))
    if (payload.maxSide) fd.append('maxSide', String(payload.maxSide))
    if (payload.grid) fd.append('grid', String(payload.grid))
    if (payload.colors) fd.append('colors', String(payload.colors))
    fd.append('background', payload.background || 'keep')
    if (payload.taskId) fd.append('taskId', payload.taskId)
    if (file) fd.append('file', file)
    return fetch('/api/pipeline/char-pixel', { method: 'POST', body: fd }).then(json)
  },
  /** 只对 AI 原稿重新像素化（毫秒级，不占 GPU） */
  charRepixel: (payload: {
    draftAssetId: string
    grid: number
    colors: number
    dither?: boolean
    background?: string
  }) =>
    fetch('/api/pipeline/char-pixel/repixel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  pixelize: (file: File, colors: number, pixelSize: number, dither: boolean, palette: string) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('colors', String(colors))
    fd.append('pixel_size', String(pixelSize))
    fd.append('dither', String(dither))
    fd.append('palette', palette)
    return fetch('/api/pipeline/pixelize', { method: 'POST', body: fd }).then(json)
  },
  seamless: (file: File, times: number, blend: number, mirror: boolean) => {    const fd = new FormData()
    fd.append('file', file)
    fd.append('times', String(times))
    fd.append('blend', String(blend))
    fd.append('mirror', String(mirror))
    return fetch('/api/pipeline/seamless', { method: 'POST', body: fd }).then(json)
  },
  tileRepair: (file: File, band: number, feather: number) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('band', String(band))
    fd.append('feather', String(feather))
    return fetch('/api/pipeline/tile', { method: 'POST', body: fd }).then(json)
  },
  // 视频/GIF 导入拆帧（移植自 FrameBaker 的 extract 流程）：fps<=0 原样全取，
  // matte=true 逐帧抠图（绿幕录像直接得到透明帧，可接精灵表管线）
  videoFrames: (
    payload: { fps?: number; maxFrames?: number; matte?: boolean; tolerance?: number },
    file: File,
  ) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('fps', String(payload.fps ?? 0))
    fd.append('max_frames', String(payload.maxFrames ?? 64))
    fd.append('matte', String(payload.matte ?? false))
    fd.append('tolerance', String(payload.tolerance ?? 80))
    return fetch('/api/pipeline/video/frames', { method: 'POST', body: fd }).then(json)
  },
  exportZip: (ids: string[]) =>
    fetch('/api/export/zip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    }).then(async (res) => {
      if (!res.ok) throw new Error('导出失败')
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'pixelforge-export.zip'
      a.click()
      // 不能马上 revoke：部分浏览器（Firefox）在 click() 返回时还没开始
      // 读取 blob，立即撤销会让下载失败。延后一小会儿再回收。
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }),

  // ---- 工作流 ----
  workflows: () => fetch('/api/workflows').then(json),
  createRun: (wfId: string, input: string, params: Record<string, any>) =>
    fetch(`/api/workflows/${wfId}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input, params }),
    }).then(json),
  runs: () => fetch('/api/runs').then(json),
  getRun: (id: string) => fetch(`/api/runs/${id}`).then(json),
  runStep: (runId: string, stepId: string, payload: Record<string, any>) =>
    fetch(`/api/runs/${runId}/steps/${stepId}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  runAll: (runId: string) => fetch(`/api/runs/${runId}/run-all`, { method: 'POST' }).then(json),

  // ---- 动画编辑器 ----
  animList: (): Promise<{ projects: any[] }> => fetch('/api/anim/projects').then(json),
  animCreate: (payload: { name?: string; canvas?: [number, number]; frames?: number }) =>
    fetch('/api/anim/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  animGet: (id: string) => fetch(`/api/anim/projects/${id}`).then(json),
  animSave: (id: string, proj: any) =>
    fetch(`/api/anim/projects/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(proj),
    }).then(json),
  animDelete: (id: string) =>
    fetch(`/api/anim/projects/${id}`, { method: 'DELETE' }).then(json),
  /** 单帧 PNG；bust 用于编辑后强制刷新（后端已带 no-store，双保险） */
  animFrameUrl: (id: string, index: number, scale = 1, bust = 0) =>
    `/api/anim/projects/${id}/frame/${index}.png?scale=${scale}&t=${bust}`,
  animRender: async (id: string, payload: {
    frames?: string; scale?: number; cols?: number; normals?: boolean; name?: string
  }) => {
    const res = await fetch(`/api/anim/projects/${id}/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!res.ok) {
      const d = await res.json().catch(() => ({}))
      throw new Error(d.detail || '导出失败')
    }
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${payload.name || 'anim'}.zip`
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  },

  /** 角色部件拆分：角色图 + 色标图 → 各部位 RGBA */
  parts: (art: File | Blob, tag: Blob, parts: { id: string; name: string; color: string }[],
          tolerance = 40): Promise<any> => {
    const fd = new FormData()
    fd.append('file', art as File)
    fd.append('tag', tag, 'tag.png')
    fd.append('parts', JSON.stringify(parts))
    fd.append('tolerance', String(tolerance))
    return fetch('/api/pipeline/parts', { method: 'POST', body: fd }).then(json)
  },

  /** 拆分件分组（按 meta.groupId 归组） */
  spriteGroups: (pipeline = 'sheet-split'): Promise<any> =>
    fetch(`/api/sprite/groups?pipeline=${encodeURIComponent(pipeline)}`).then(json),

  /** 权重可视化：顶点静止位 + 每骨骼最终权重（数学只在后端算一份） */
  animWeights: (id: string, trackId: string, frame: number): Promise<any> => {
    const q = new URLSearchParams({ track: trackId, frame: String(frame) })
    return fetch(`/api/anim/projects/${id}/weights?${q.toString()}`).then(json)
  },

  /** 导出 Godot 包：anim.json + 纹理 + 参考运行时（骨骼与蒙皮已烘焙） */
  animGodot: async (id: string, name = 'anim') => {
    const res = await fetch(`/api/anim/projects/${id}/export/godot`, { method: 'POST' })
    if (!res.ok) {
      const d = await res.json().catch(() => ({}))
      throw new Error(d.detail || '导出失败')
    }
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}-godot.zip`
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  },

  // ---- 法线 / 材质掩码 ----
  normals: (
    payload: {
      strength?: number; blur?: number; metalMode?: string
      metalThreshold?: number; flipGreen?: boolean; preview?: boolean; assetId?: string
    } = {},
    file?: File | null,
  ) => {
    const fd = new FormData()
    if (file) fd.append('file', file)
    if (payload.assetId) fd.append('refAssetId', payload.assetId)
    fd.append('strength', String(payload.strength ?? 1.3))
    fd.append('blur', String(payload.blur ?? 2))
    fd.append('metalMode', payload.metalMode || 'highlight')
    fd.append('metalThreshold', String(payload.metalThreshold ?? 0.6))
    fd.append('flipGreen', String(payload.flipGreen ?? false))
    fd.append('preview', String(payload.preview ?? true))
    return fetch('/api/pipeline/normals', { method: 'POST', body: fd }).then(json)
  },
  normalsBatch: async (frameAssetIds: string[], opts: Record<string, any> = {}) => {
    const res = await fetch('/api/pipeline/normals/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frameAssetIds, ...opts }),
    })
    if (!res.ok) throw new Error('批量法线失败')
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'normals.zip'
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  },
}
