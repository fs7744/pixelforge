import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, Asset, ProviderInfo, StyleInfo, Template } from '../api'
import { openPreview } from '../Preview'
import { Splitter } from '../split'

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
  // 参考图（图生图）：本地文件 + objectURL 缩略图，移除时立即释放
  const [refItems, setRefItems] = useState<{ file: File; url: string }[]>([])
  // 参考图另一种来源：直接从资产库挑（不用先下载再上传）
  const [refAssetIds, setRefAssetIds] = useState<string[]>([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const [library, setLibrary] = useState<Asset[]>([])

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
      setProvider(act ? act.id : d.providers[0]?.id ?? 'diffusers')
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

  // 底座是中文原生（Z-Image / Qwen-Edit），模板只有中文一版
  const promptFor = (t: Template) => t.prompt

  // 切模板时若提示词没被改过，自动换成模板原文
  useEffect(() => {
    const t = templates.find((x) => x.id === tplId)
    if (t && (prompt === '' || templates.some((x) => x.prompt === prompt))) {
      setPrompt(promptFor(t))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, templates, tplId])

  const pickTemplate = (t: Template) => {
    setTplId(t.id)
    setPrompt(promptFor(t))
    if (t.params?.width) setWidth(t.params.width)
    if (t.params?.height) setHeight(t.params.height)
  }

  // 参考图最多 4 张：再多也分不清主次，编码时间还线性上涨
  const MAX_REFS = 4
  const totalRefs = refItems.length + refAssetIds.length

  const addRefs = (files: FileList | null) => {
    if (!files || !files.length) return
    const picked = Array.from(files).filter((f) => f.type.startsWith('image/'))
    setRefItems((prev) => {
      const room = Math.max(0, MAX_REFS - prev.length - refAssetIds.length)
      const next = picked.slice(0, room).map((f) => ({ file: f, url: URL.createObjectURL(f) }))
      return [...prev, ...next]
    })
  }

  const removeRef = (idx: number) => {
    setRefItems((prev) => {
      const item = prev[idx]
      if (item) URL.revokeObjectURL(item.url)
      return prev.filter((_, i) => i !== idx)
    })
  }

  const openPicker = () => {
    const next = !pickerOpen
    setPickerOpen(next)
    if (next && !library.length) {
      api
        .assets()
        .then((d) => setLibrary(d.assets.slice(0, 24)))
        .catch(() => setLibrary([]))
    }
  }

  const toggleAsset = (id: string) => {
    setRefAssetIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      if (prev.length + refItems.length >= MAX_REFS) return prev
      return [...prev, id]
    })
  }

  const run = async () => {
    setBusy(true)
    setError('')
    try {
      // 有参考图 = 参考图生成（图生图）：Z-Image 没有参考图输入，后端会自动改走
      // Qwen-Image-Edit。这条路径优先于三视图模板——三视图是「三张独立文生图」，
      // 与参考图语义冲突。
      const data = refItems.length || refAssetIds.length
        ? await api.generateRef(
            {
              prompt,
              style: styleId || undefined,
              width,
              height,
              steps,
              cfg,
              n,
              seed: seed.trim() ? Number(seed) : null,
              refAssetIds,
            },
            refItems.map((r) => r.file),
          )
        : tplId === 'char_three_view'
          ? await api.threeview({
              prompt,
              style: styleId || undefined,
              steps,
              cfg,
              seed: seed.trim() ? Number(seed) : null,
              width,
              height,
            })
          : await api.generate({
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
          <div className="card" style={{ padding: '8px 10px' }}>
            <div style={{ fontSize: 14, fontWeight: 600 }}>
              {providers.find((p) => p.active)?.label || '本地双底座'}
            </div>
            <div className="hint" style={{ marginTop: 4 }}>
              {providers.find((p) => p.active)?.model || 'models/z-image-turbo'}
            </div>
          </div>
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
          {tplId === 'char_three_view' && (
            <div className="hint" style={{ marginTop: 8 }}>
              三视图管线：正/侧/背三张独立生成（帧 2+ 走编辑底座转视角，保持同一角色）后自动拼板，
              不重叠、不缺视图。宽高为单格尺寸。预计 5-8 分钟（编辑底座 40 步 × 2 + 拼板）。
            </div>
          )}
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
          <div className="label">提示词（可编辑，花括号为占位符）</div>
          <textarea className="textarea" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="hint" style={{ marginTop: 6 }}>
            中文直喂本地底座（Z-Image / Qwen-Edit），全程不做英文翻译。
          </div>
        </div>

        <div>
          <div className="row between" style={{ alignItems: 'center' }}>
            <div className="label" style={{ marginBottom: 0 }}>
              参考图 · 图生图（可选，最多 {MAX_REFS} 张）
            </div>
            <button className="btn btn-ghost" style={{ padding: '2px 8px', fontSize: 13 }} onClick={openPicker}>
              {pickerOpen ? '收起资产库' : '从资产库选'}
            </button>
          </div>
          <div className="ref-grid">
            {refItems.map((r, i) => (
              <div className="ref-thumb" key={r.url}>
                <img src={r.url} alt={`参考图 ${i + 1}`} />
                <span className="ref-idx">{i + 1}</span>
                <button className="ref-del" title="移除" onClick={() => removeRef(i)}>
                  ×
                </button>
              </div>
            ))}
            {refAssetIds.map((id, i) => {
              const a = library.find((x) => x.id === id)
              return (
                <div className="ref-thumb" key={id}>
                  {a ? <img src={a.url} alt={a.name} /> : <div className="ref-miss">?</div>}
                  <span className="ref-idx">{refItems.length + i + 1}</span>
                  <button className="ref-del" title="移除" onClick={() => toggleAsset(id)}>
                    ×
                  </button>
                </div>
              )
            })}
            {totalRefs < MAX_REFS && (
              <label className="ref-add" title="上传参考图">
                <span>+</span>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  hidden
                  onChange={(e) => {
                    addRefs(e.target.files)
                    e.currentTarget.value = ''
                  }}
                />
              </label>
            )}
          </div>

          {pickerOpen && (
            <div className="ref-picker">
              {library.length === 0 && <div className="hint">资产库还是空的</div>}
              {library.map((a) => {
                const on = refAssetIds.includes(a.id)
                const full = !on && totalRefs >= MAX_REFS
                return (
                  <button
                    key={a.id}
                    className={`ref-pick ${on ? 'on' : ''}`}
                    disabled={full}
                    title={a.name}
                    onClick={() => toggleAsset(a.id)}
                  >
                    <img src={a.url} alt={a.name} />
                    {on && <span className="ref-idx">{refAssetIds.indexOf(a.id) + 1}</span>}
                  </button>
                )
              })}
            </div>
          )}

          <div className="hint" style={{ marginTop: 6 }}>
            {totalRefs
              ? `已选 ${totalRefs} 张：走 Qwen-Image-Edit 指令式生成，输出尺寸用上方「宽/高」。第一张是主参考，其余作补充（如正面+侧面、角色+场景）。`
              : '不选 = 文生图（Z-Image）。需要按参考图生成时加图，会自动改用编辑底座；首图是主参考。'}
          </div>
          {totalRefs > 0 && tplId === 'char_three_view' && (
            <div className="hint" style={{ marginTop: 4, color: 'var(--yellow)' }}>
              当前模板是三视图（三张独立文生图），已选参考图时按单张参考图生成处理。
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
        {meta?.mode === 'ref2img' && (
          <div className="hint">
            参考图生成 · {meta.refCount} 张参考图 · {(meta.model || '').split('/').pop()}
            {meta.autoRouted ? '（已自动改用编辑底座）' : ''}
          </div>
        )}
        {meta?.pipeline === 'threeview' && (
          <div className="hint">
            三视图完成（{meta.mode}）· 拼板 {meta.compose?.size?.join('×')} · 总耗时 {meta.seconds}s
            {meta.editFallback ? ` · 编辑底座不可用，已退回同种子三连发：${meta.editFallback}` : ''}
          </div>
        )}
        {error && <div className="err">{error}</div>}
      </aside>

      <Splitter side="left" />

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
              选择模板 → 填写提示词（中文）→ 点击生成。
              <br />
              出图后点图片可放大预览，可一键送后处理。
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
                <img
                  src={r.url}
                  alt={r.name}
                  style={{ cursor: 'zoom-in' }}
                  onClick={(e) => {
                    e.stopPropagation()
                    openPreview(r.url, r.name)
                  }}
                />
                <div className="cap">
                  {r.width}×{r.height} · {r.meta?.style?.name ? `${r.meta.style.name} · ` : ''}
                  {r.meta?.provider ?? r.kind}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      <Splitter side="right" />

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
              <div
                className="checker"
                style={{ width: 40, height: 40, border: '2px solid #0a0a0a', cursor: 'zoom-in' }}
                onClick={(e) => {
                  e.stopPropagation()
                  openPreview(r.url, r.name)
                }}
              >
                <img src={r.url} style={{ width: '100%', height: '100%', objectFit: 'contain', imageRendering: 'pixelated' }} />
              </div>
              <div>
                <div style={{ fontSize: 13 }}>{r.meta?.templateId || r.kind}</div>
                <div className="hint">{new Date(r.createdAt * 1000).toLocaleTimeString()}</div>
              </div>
            </div>
          ))}
          {results.length === 0 && <div className="hint">暂无记录</div>}
        </div>

        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 10, marginBottom: 4 }}>EXPORT</div>
          <div style={{ fontSize: 13, lineHeight: 1.6 }}>
            选中结果可下载 PNG ZIP。默认不烧 AI 角标（角标会污染后续抠图/合成）；
            需要合规标识时到「模型设置」打开「下载时补 AI 标识」。
          </div>
        </div>
      </aside>
    </div>
  )
}
