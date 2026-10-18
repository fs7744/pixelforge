import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, Asset, ProviderInfo, StyleInfo, Template } from '../api'

type Props = { onSendToPost: (asset: Asset) => void }

// 画风 chip：悬浮（120ms 延迟）/点击弹出示例图 tooltip。
// 左侧 .panel 有 overflow-y:auto，组件内绝对定位会被裁剪，所以走 portal + fixed 定位，
// 打开期间监听 scroll/resize 重定位；触屏没有 hover，用点击打开、点外部/Esc 关闭。
// PostProcess 的「AI 画风重统一」复用同一组件（选目标画风时同样想看示例）。
export function StyleChip({ style, active, onSelect }: { style: StyleInfo; active: boolean; onSelect: () => void }) {
  const btnRef = useRef<HTMLButtonElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const showTimer = useRef<number | undefined>(undefined)
  const hideTimer = useRef<number | undefined>(undefined)

  const clearTimers = useCallback(() => {
    window.clearTimeout(showTimer.current)
    window.clearTimeout(hideTimer.current)
  }, [])

  const place = useCallback(() => {
    const el = btnRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const W = 252
    const H = 400 // 预估高度：252 方图 + 文字区
    let left = Math.min(r.left, window.innerWidth - W - 12)
    left = Math.max(8, left)
    let top = r.bottom + 8
    if (top + H > window.innerHeight - 8) top = Math.max(8, r.top - H - 8) // 下方不够则翻到上方
    setPos({ top, left })
  }, [])

  const show = useCallback(() => {
    window.clearTimeout(hideTimer.current)
    window.clearTimeout(showTimer.current)
    showTimer.current = window.setTimeout(() => {
      place()
      setOpen(true)
    }, 120)
  }, [place])

  const hide = useCallback(() => {
    window.clearTimeout(showTimer.current)
    window.clearTimeout(hideTimer.current)
    hideTimer.current = window.setTimeout(() => setOpen(false), 160) // 留出移入 tooltip 的缓冲
  }, [])

  const cancelHide = useCallback(() => window.clearTimeout(hideTimer.current), [])

  useEffect(() => clearTimers, [clearTimers])

  // 打开期间：滚动/缩放跟随、Esc 关闭、点外部关闭（点 chip 由 onClick 重新打开，保证全局只有一个 tooltip）
  useEffect(() => {
    if (!open) return
    const follow = () => place()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as HTMLElement | null
      if (t?.closest('.style-tip')) return
      setOpen(false)
    }
    window.addEventListener('scroll', follow, true)
    window.addEventListener('resize', follow)
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onPointerDown, true)
    return () => {
      window.removeEventListener('scroll', follow, true)
      window.removeEventListener('resize', follow)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onPointerDown, true)
    }
  }, [open, place])

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className={`chip ${active ? 'on' : ''}`}
        onClick={() => {
          onSelect()
          place()
          setOpen(true)
        }}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        aria-describedby={open ? `style-tip-${style.id}` : undefined}
      >
        {style.name}
      </button>
      {open &&
        pos &&
        createPortal(
          <div
            id={`style-tip-${style.id}`}
            role="tooltip"
            className="style-tip"
            style={{ top: pos.top, left: pos.left }}
            onMouseEnter={cancelHide}
            onMouseLeave={hide}
          >
            <div className="checker style-tip-img">
              {style.sample ? (
                <img
                  src={style.sample}
                  alt={`${style.name}画风示例`}
                  loading="lazy"
                  style={{ imageRendering: style.pixelize ? 'pixelated' : 'auto' }}
                />
              ) : (
                <div className="style-tip-none">
                  示例图未生成
                  <br />
                  跑 backend/gen_style_samples.py
                </div>
              )}
            </div>
            <div className="style-tip-body">
              <div className="style-tip-name">{style.name}</div>
              <div className="style-tip-en">{style.en}</div>
              <div className="chips" style={{ marginTop: 4 }}>
                {style.tags.map((t) => (
                  <span key={t} className="chip sm">
                    {t}
                  </span>
                ))}
              </div>
              <div className="style-tip-hint">{style.hint}</div>
              {style.pixelize && (
                <span className="tag" style={{ marginTop: 6, alignSelf: 'flex-start' }}>
                  建议像素化 {style.pixelize.block}× · {style.pixelize.colors} 色
                </span>
              )}
            </div>
          </div>,
          document.body,
        )}
    </>
  )
}

export default function Workbench({ onSendToPost }: Props) {
  const [templates, setTemplates] = useState<Template[]>([])
  const [categories, setCategories] = useState<string[]>(['全部'])
  const [cat, setCat] = useState('全部')
  const [tplId, setTplId] = useState<string>('')
  const [prompt, setPrompt] = useState('')
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [provider, setProvider] = useState('')
  const [styles, setStyles] = useState<StyleInfo[]>([])
  const [styleId, setStyleId] = useState('')
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
  const [translated, setTranslated] = useState('')

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
    api.styles().then((d) => setStyles(d.styles))
  }, [])

  const filtered = useMemo(
    () => (cat === '全部' ? templates : templates.filter((t) => t.category === cat)),
    [templates, cat],
  )

  // 本地 SD 系模型（diffusers / A1111）用 CLIP，只认英文，但中文已由词典+机翻兜底
  const englishOnly = provider === 'diffusers' || provider === 'a1111'
  const hasCjk = /[\u4e00-\u9fa5]/.test(prompt)

  // 示例提示词默认展示中文模板（更可读）；底层英文由翻译链路自动处理，
  // 想看英文译法点「预览英文译法」即可。
  const promptFor = (t: Template) => t.prompt

  // Provider 切换时，若提示词仍是模板原文（用户没改过），自动换成对应语言版本
  useEffect(() => {
    const t = templates.find((x) => x.id === tplId)
    if (!t) return
    const alt = englishOnly ? t.prompt : t.promptEn
    if (prompt === '' || prompt === alt) setPrompt(promptFor(t))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, templates, tplId])

  const pickTemplate = (t: Template) => {
    setTplId(t.id)
    setPrompt(promptFor(t))
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
        style: styleId || undefined,
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
          <div className="label">美术画风 · 游戏行业典型风格</div>
          <div className="chips mt8">
            <button className={`chip ${styleId === '' ? 'on' : ''}`} onClick={() => setStyleId('')}>
              不限定
            </button>
            {styles.map((s) => (
              <StyleChip key={s.id} style={s} active={styleId === s.id} onSelect={() => setStyleId(s.id)} />
            ))}
          </div>
          {styleId && (() => {
            const s = styles.find((x) => x.id === styleId)
            if (!s) return null
            return (
              <div className="hint" style={{ marginTop: 8 }}>
                {s.hint}
                {s.pixelize && (
                  <>
                    {' '}
                    <span className="tag" style={{ color: 'var(--green)' }}>
                      建议像素化 {s.pixelize.block}× · {s.pixelize.colors} 色
                    </span>
                  </>
                )}
              </div>
            )
          })()}
        </div>

        <div>
          <div className="label">
            提示词（可编辑，花括号为占位符）
            {englishOnly && <span style={{ color: 'var(--green)' }}> · 本地模型用英文</span>}
          </div>
          <textarea className="textarea" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          {englishOnly && (
            <div className="row" style={{ marginTop: 6, alignItems: 'center', gap: 8 }}>
              <button
                className="btn btn-ghost"
                style={{ height: 28 }}
                onClick={async () => {
                  try {
                    const r = await api.translate(prompt)
                    setTranslated(r.english)
                  } catch {
                    setTranslated('')
                  }
                }}
              >
                预览英文译法
              </button>
              {translated && (
                <span className="hint" style={{ color: 'var(--green)', flex: 1 }}>
                  {translated}
                </span>
              )}
            </div>
          )}
          {englishOnly && hasCjk && (
            <div className="hint" style={{ color: 'var(--yellow)', marginTop: 6 }}>
              本地模型只认英文：中文会先用领域词典自动翻译（颜色/角色/装备/动作/画风），生僻描写建议自己补英文
            </div>
          )}
          {meta?.translation?.english && (
            <div className="hint" style={{ marginTop: 6 }}>
              本次实际送入模型：{meta.translation.english}
            </div>
          )}
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
                  {r.width}×{r.height} · {r.meta?.style?.name ? `${r.meta.style.name} · ` : ''}
                  {r.meta?.provider ?? r.kind}
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
