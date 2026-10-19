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
  tags: string[]
  hint: string
  /** 实际拼进提示词的风格词（前置到开头），用于界面展示「加的是什么」 */
  words?: string
  pixelize?: { block: number; colors: number }
  /** 画风示例图 URL（gen_style_samples.py 生成过才有） */
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
  splitSheet: (file: File, cols: number, rows: number, matte = false, tolerance = 80) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('cols', String(cols))
    fd.append('rows', String(rows))
    fd.append('matte', String(matte))
    fd.append('tolerance', String(tolerance))
    return fetch('/api/pipeline/sheet/split', { method: 'POST', body: fd }).then(json)
  },
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
      URL.revokeObjectURL(url)
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
}
