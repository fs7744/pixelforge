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

export type Template = {
  id: string
  category: string
  name: string
  hint: string
  prompt: string
  params: Record<string, number>
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
  }) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
  assets: (kind?: string) =>
    fetch(`/api/assets${kind && kind !== '全部' ? `?kind=${kind}` : ''}`).then(json),
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
  splitSheet: (file: File, cols: number, rows: number) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('cols', String(cols))
    fd.append('rows', String(rows))
    return fetch('/api/pipeline/sheet/split', { method: 'POST', body: fd }).then(json)
  },
  seamless: (file: File, times: number, blend: number, mirror: boolean) => {
    const fd = new FormData()
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
