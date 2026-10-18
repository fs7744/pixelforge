import { useEffect, useMemo, useState } from 'react'
import { api, Asset, ProviderInfo, Template } from '../api'

type Props = { onSendToPost: (asset: Asset) => void }

export default function Workbench({ onSendToPost }: Props) {
  const [templates, setTemplates] = useState<Template[]>([])
  const [categories, setCategories] = useState<string[]>(['全部'])
  const [cat, setCat] = useState('全部')
  const [tplId, setTplId] = useState<string>('')
  const [prompt, setPrompt] = useState('')
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [provider, setProvider] = useState('')
  const [width, setWidth] = useState(512)
  const [height, setHeight] = useState(512)
  const [steps, setSteps] = useState(28)
  const [cfg, setCfg] = useState(7.5)
  const [n, setN] = useState(1)
  const [seed, setSeed] = useState<string>('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState<Asset[]>([])
  const [selected, setSelected] = useState<string>('')
  const [meta, setMeta] = useState<any>(null)

  useEffect(() => {
    api.templates().then((d) => {
      setTemplates(d.templates)
      setCategories(d.categories)
      if (d.templates[0]) {
        setTplId(d.templates[0].id)
        setPrompt(d.templates[0].prompt)
      }
    })
    api.providers().then((d) => {
      setProviders(d.providers)
      const act = d.providers.find((p: ProviderInfo) => p.active)
      setProvider(act ? act.id : d.providers[0]?.id ?? 'mock')
    })
    api.settings().then((s) => {
      setSteps(s.defaults.steps)
      setCfg(s.defaults.cfg)
      setWidth(s.defaults.width)
      setHeight(s.defaults.height)
    })
  }, [])

  const filtered = useMemo(
    () => (cat === '全部' ? templates : templates.filter((t) => t.category === cat)),
    [templates, cat],
  )

  const pickTemplate = (t: Template) => {
    setTplId(t.id)
    setPrompt(t.prompt)
    if (t.params?.width) setWidth(t.params.width)
    if (t.params?.height) setHeight(t.params.height)
  }

  const run = async () => {
    setBusy(true)
    setError('')
    try {
      const data = await api.generate({
        prompt,
        provider,
        width,
        height,
        steps,
        cfg,
        n,
        seed: seed.trim() ? Number(seed) : null,
        templateId: tplId,
      })
      setResults((prev) => [...data.images, ...prev])
      setMeta(data.meta)
      if (data.images[0]) setSelected(data.images[0].id)
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  const current = results.find((r) => r.id === selected)

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          生成参数 <span className="tag">PARAMS</span>
        </div>

        <div>
          <div className="label">模型服务</div>
          <select className="select" value={provider} onChange={(e) => setProvider(e.target.value)}>
            {providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
                {p.active ? '（默认）' : ''}
              </option>
            ))}
          </select>
        </div>

        <div>
          <div className="label">模板库 · 单图模式</div>
          <div className="chips mt8">
            {categories.map((c) => (
              <button key={c} className={`chip ${cat === c ? 'on' : ''}`} onClick={() => setCat(c)}>
                {c}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10, maxHeight: 210, overflowY: 'auto' }}>
            {filtered.map((t) => (
              <button key={t.id} className={`tpl ${tplId === t.id ? 'on' : ''}`} onClick={() => pickTemplate(t)}>
                <span className="tpl-name">{t.name}</span>
                <span className="tpl-hint">{t.hint}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="label">提示词（可编辑，花括号为占位符）</div>
          <textarea className="textarea" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </div>

        <div>
          <div className="label">高级参数</div>
          <div className="row" style={{ marginTop: 6 }}>
            <div className="grow">
              <div className="label">宽</div>
              <input className="input" type="number" value={width} onChange={(e) => setWidth(+e.target.value)} />
            </div>
            <div className="grow">
              <div className="label">高</div>
              <input className="input" type="number" value={height} onChange={(e) => setHeight(+e.target.value)} />
            </div>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <div className="grow">
              <div className="label">Steps</div>
              <input className="input" type="number" value={steps} onChange={(e) => setSteps(+e.target.value)} />
            </div>
            <div className="grow">
              <div className="label">CFG</div>
              <input className="input" type="number" step="0.1" value={cfg} onChange={(e) => setCfg(+e.target.value)} />
            </div>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <div className="grow">
              <div className="label">数量</div>
              <input className="input" type="number" min={1} max={4} value={n} onChange={(e) => setN(+e.target.value)} />
            </div>
            <div className="grow">
              <div className="label">种子（空=随机）</div>
              <input className="input" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" />
            </div>
          </div>
        </div>

        <button className="btn btn-primary btn-block" disabled={busy || !prompt.trim()} onClick={run}>
          {busy ? '生成中' : '生成'}
        </button>
        {meta?.note && <div className="hint">{meta.note}</div>}
        {error && <div className="err">{error}</div>}
      </aside>

      <main className="stage">
        <div className="row between">
          <div className="section-title">生成结果</div>
          <div className="row">
            {current && (
              <>
                <button className="btn btn-ghost" onClick={() => onSendToPost(current)}>
                  送后处理
                </button>
                <button className="btn" onClick={() => api.exportZip([current.id])}>
                  下载 PNG ZIP
                </button>
              </>
            )}
          </div>
        </div>

        {results.length === 0 ? (
          <div className="empty">
            <div>
              <span className="pixel">READY</span>
              选择模板 → 填写提示词 → 点击生成。
              <br />
              当前 Provider 为「内置模拟引擎」时会产出 4×4 像素精灵表，
              <br />
              可直接送后处理验证抠图与尺寸规整管线。
            </div>
          </div>
        ) : (
          <div className="grid">
            {results.map((r) => (
              <div
                key={r.id}
                className={`thumb ${selected === r.id ? 'on' : ''}`}
                style={{ aspectRatio: '1 / 1' }}
                onClick={() => setSelected(r.id)}
              >
                <img src={r.url} alt={r.name} />
                <div className="cap">
                  {r.width}×{r.height} · {r.meta?.provider ?? r.kind}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      <aside className="panel right">
        <div className="section-title">
          历史记录 <span className="tag">HISTORY</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {results.slice(0, 8).map((r) => (
            <div
              key={r.id}
              className="row"
              style={{ cursor: 'pointer', border: selected === r.id ? '1px solid #00FF88' : '1px solid transparent', padding: 4 }}
              onClick={() => setSelected(r.id)}
            >
              <div className="checker" style={{ width: 40, height: 40, border: '2px solid #0a0a0a' }}>
                <img src={r.url} style={{ width: '100%', height: '100%', objectFit: 'contain', imageRendering: 'pixelated' }} />
              </div>
              <div>
                <div style={{ fontSize: 11 }}>{r.meta?.templateId || r.kind}</div>
                <div className="hint">{new Date(r.createdAt * 1000).toLocaleTimeString()}</div>
              </div>
            </div>
          ))}
          {results.length === 0 && <div className="hint">暂无记录</div>}
        </div>

        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 9, marginBottom: 4 }}>EXPORT</div>
          <div style={{ fontSize: 11, lineHeight: 1.6 }}>
            选中结果可下载 PNG ZIP；导出自动附带 AI 生成标识。
          </div>
        </div>
      </aside>
    </div>
  )
}
