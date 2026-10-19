import { useEffect, useMemo, useRef, useState } from 'react'
import { api, Asset, AssetFacets, CharPixelLevel } from '../api'
import { RefPicker } from '../RefPicker'
import { openPreview } from '../Preview'
import { Splitter } from '../split'

type Props = {
  incoming?: Asset | null
}

type Src = { url: string; name: string; assetId?: string }
type Run = { asset: Asset; levelId: string; levelName: string; background: string }

const STEPS_DEFAULT = 40
const COLOR_CHOICES = [12, 16, 20, 24, 28, 32, 48]

/**
 * 人物像素化：一张人物照片 → 像素角色立绘。
 *
 * 三档「程度」变的不是颗粒大小，而是造型语言：写实比例 → 半 Q 版（约 4 头身）
 * → Q 版（约 2 头身），头身比、眼睛大小、阴影层数一起变——那只能靠 AI 重绘；
 * 像素语法（每格单色、限定调色板、硬边）由确定性像素化收尾，两步缺一不可。
 *
 * 生成返回 AI 原稿 + 像素化成品两套资产：调像素密度/色数/背景处理是毫秒级的
 * CPU 操作，不必再等 2-4 分钟的编辑。
 */
export default function CharPixel({ incoming }: Props) {
  const [src, setSrc] = useState<Src | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [levels, setLevels] = useState<CharPixelLevel[]>([])
  const [backgrounds, setBackgrounds] = useState<{ id: string; name: string; hint: string }[]>([])
  const [levelId, setLevelId] = useState('mid')
  const [bg, setBg] = useState('keep')
  const [prompt, setPrompt] = useState('')
  const [steps, setSteps] = useState(STEPS_DEFAULT)
  const [seed, setSeed] = useState('')
  const [maxSide, setMaxSide] = useState(1024)
  const [grid, setGrid] = useState(120)
  const [colors, setColors] = useState(24)
  const [out, setOut] = useState<Asset | null>(null)
  // 原稿只用到 id/name（重新像素化按 id 取），不必摆整套 Asset
  const [draft, setDraft] = useState<{ id: string; name: string } | null>(null)
  const [outMeta, setOutMeta] = useState<Record<string, any> | null>(null)
  const [shots, setShots] = useState<Record<string, Asset>>({})
  const [history, setHistory] = useState<Run[]>([])
  const [busy, setBusy] = useState(false)
  const [repixelBusy, setRepixelBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [split, setSplit] = useState(50)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [over, setOver] = useState(false)

  const [libOpen, setLibOpen] = useState(false)
  const [library, setLibrary] = useState<Asset[]>([])
  const [libFacets, setLibFacets] = useState<AssetFacets | null>(null)
  const [libPick, setLibPick] = useState<string[]>([])

  const inputRef = useRef<HTMLInputElement>(null)
  const objectUrl = useRef('')
  const boxRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  useEffect(() => {
    api.charPixelLevels()
      .then((r) => {
        setLevels(r.levels || [])
        setBackgrounds(r.backgrounds || [])
      })
      .catch(() => setLevels([]))
  }, [])

  useEffect(() => () => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
  }, [])

  const clearResult = () => {
    setOut(null)
    setDraft(null)
    setOutMeta(null)
    setSplit(50)
    setError('')
    setNote('')
  }

  const acceptLocal = (f: File) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    const url = URL.createObjectURL(f)
    objectUrl.current = url
    setFile(f)
    setSrc({ url, name: f.name })
    clearResult()
  }

  const acceptAsset = (a: Asset) => {
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = ''
    }
    setFile(null)
    setSrc({ url: a.url, name: a.name, assetId: a.id })
    clearResult()
  }

  useEffect(() => {
    if (incoming) acceptAsset(incoming)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incoming])

  const openLib = () => {
    const next = !libOpen
    setLibOpen(next)
    if (!next) return
    api.assets().then((d) => setLibrary(d.assets)).catch(() => setLibrary([]))
    api.assetFacets().then(setLibFacets).catch(() => setLibFacets(null))
  }

  const pickFromLibrary = (id: string) => {
    const a = library.find((x) => x.id === id)
    if (!a) return
    setLibPick([id])
    acceptAsset(a)
    setLibOpen(false)
    setLibPick([])
  }

  const cur = useMemo(() => levels.find((l) => l.id === levelId) || null, [levels, levelId])

  // 换档位 = 换造型语言，像素密度/色数跟着回到该档预设（用户可再手动微调）
  const pickLevel = (id: string) => {
    setLevelId(id)
    const lv = levels.find((l) => l.id === id)
    if (lv) {
      setGrid(lv.grid)
      setColors(lv.colors)
    }
  }

  useEffect(() => {
    if (!busy) return
    setElapsed(0)
    const t = window.setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => window.clearInterval(t)
  }, [busy])

  const apply = async () => {
    if (!src || !levelId) return
    const seedNum = seed.trim() === '' ? null : Number(seed)
    if (seedNum != null && !Number.isFinite(seedNum)) {
      setError('种子要填整数，留空则每次不同')
      return
    }
    setBusy(true)
    clearResult()
    const t0 = performance.now()
    try {
      const res = await api.charPixel(
        {
          level: levelId,
          prompt,
          assetId: src.assetId,
          steps,
          seed: seedNum,
          maxSide,
          grid,
          colors,
          background: bg,
        },
        src.assetId ? null : file,
      )
      if (!res.asset) throw new Error(res.meta?.reason || '生成失败')
      const item: Asset = res.asset
      setOut(item)
      setDraft(res.draft ? { id: res.draft.id, name: res.draft.name } : null)
      setOutMeta(res.meta || null)
      setSplit(50)
      if (res.meta?.grid) setGrid(res.meta.grid)
      if (res.meta?.colors) setColors(res.meta.colors)
      const secs = Math.round((performance.now() - t0) / 1000)
      const bgName = backgrounds.find((b) => b.id === (res.meta?.background || 'keep'))?.name || ''
      setNote(
        `${res.meta?.levelName || ''} · ${secs}s · seed ${res.meta?.seed ?? '随机'} · ` +
        `${res.meta?.grid} 格 / ${res.meta?.pixelize?.colors ?? res.meta?.colors} 色 · ${bgName}`,
      )
      setShots((s) => ({ ...s, [levelId]: item }))
      setHistory((h) => [
        { asset: item, levelId, levelName: res.meta?.levelName || levelId, background: res.meta?.background || bg },
        ...h.filter((x) => x.asset.id !== item.id),
      ].slice(0, 8))
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  // 背景是生成时就写进 AI 指令的：改背景必须重新生成；
  // 唯一的例外是「绿幕 → 透明」，那一步是纯色键，不占 GPU。
  const canRepixelAs = (target: string) => {
    const made = outMeta?.background || 'keep'
    return made === target || (made === 'green' && target === 'alpha')
  }
  const repixelAllowed = !!draft && canRepixelAs(bg)

  const applyRepixel = async () => {
    if (!draft) return
    setRepixelBusy(true)
    setError('')
    setNote('')
    try {
      const res = await api.charRepixel({
        draftAssetId: draft.id,
        grid,
        colors,
        background: bg,
      })
      if (!res.asset) throw new Error('重新像素化失败')
      setOut(res.asset)
      setOutMeta((m) => ({ ...(m || {}), ...res.meta, level: outMeta?.level, levelName: outMeta?.levelName }))
      setSplit(50)
      setShots((s) => ({ ...s, [levelId]: res.asset }))
      setHistory((h) => [
        { asset: res.asset, levelId, levelName: outMeta?.levelName || levelId, background: bg },
        ...h.filter((x) => x.asset.id !== res.asset.id),
      ].slice(0, 8))
      setNote(`重新像素化完成 · ${res.meta?.grid} 格 / ${res.meta?.colors} 色 · 未重跑 AI`)
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setRepixelBusy(false)
    }
  }

  const at = (clientX: number) => {
    const el = boxRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setSplit(Math.max(0, Math.min(100, ((clientX - r.left) / Math.max(1, r.width)) * 100)))
  }

  const bgName = (id: string) => backgrounds.find((b) => b.id === id)?.name || id

  return (
    <div className="body">
      <aside className="panel">
        <div className="row between" style={{ alignItems: 'center' }}>
          <div className="section-title">
            人物照片 <span className="tag">PHOTO</span>
          </div>
          <button
            className="btn btn-ghost"
            style={{ padding: '2px 8px', fontSize: 13 }}
            onClick={openLib}
          >
            {libOpen ? '收起资产库' : '从资产库选'}
          </button>
        </div>

        <div
          className={`drop ${over ? 'over' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            const f = e.dataTransfer.files?.[0]
            if (f) acceptLocal(f)
          }}
        >
          {src ? (
            <div>
              <div className="mono">{src.name}</div>
              <div className="hint">{src.assetId ? '素材库资产 · 点击替换' : '本地文件 · 点击替换'}</div>
            </div>
          ) : (
            <div>
              拖入或点击上传人物照片
              <div className="hint" style={{ marginTop: 6 }}>
                半身 / 全身都行，构图会沿用原图
              </div>
            </div>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) acceptLocal(f)
            e.currentTarget.value = ''
          }}
        />

        {libOpen && (
          <RefPicker
            library={library}
            facets={libFacets}
            selected={libPick}
            usedSlots={0}
            maxRefs={1}
            single
            onToggle={pickFromLibrary}
          />
        )}

        <div>
          <div className="label">像素化程度（{levels.length} 档）</div>
          <div className="lv-cards">
            {levels.map((lv) => (
              <button
                key={lv.id}
                type="button"
                className={`lv-card ${levelId === lv.id ? 'on' : ''}`}
                onClick={() => pickLevel(lv.id)}
              >
                <span className="lv-thumb checker">
                  {shots[lv.id] ? (
                    <img src={api.assetThumb(shots[lv.id].id, 160)} alt={`${lv.name}结果`} loading="lazy" />
                  ) : lv.sample ? (
                    <img src={lv.sample} alt={`${lv.name}示例`} loading="lazy" />
                  ) : (
                    <span className="lv-ph">{lv.heads}</span>
                  )}
                </span>
                <span className="lv-name">{lv.name}</span>
                <span className="lv-spec">{lv.grid} 格 · {lv.colors} 色</span>
              </button>
            ))}
          </div>
          {cur && <div className="hint" style={{ marginTop: 6 }}>{cur.hint}</div>}
        </div>

        <div>
          <div className="label">人物背景</div>
          <div className="chips mt8">
            {backgrounds.map((b) => (
              <button
                key={b.id}
                className={`chip ${bg === b.id ? 'on' : ''}`}
                title={b.hint}
                onClick={() => setBg(b.id)}
              >
                {b.name}
              </button>
            ))}
          </div>
          <div className="hint" style={{ marginTop: 6 }}>
            {backgrounds.find((b) => b.id === bg)?.hint}
            {out && !canRepixelAs(bg) && (
              <> 当前结果是「{bgName(outMeta?.background || 'keep')}」，换背景要重新生成。</>
            )}
          </div>
        </div>

        <div>
          <div className="label">补充要求（可留空）</div>
          <textarea
            className="textarea"
            style={{ minHeight: 72 }}
            placeholder="例如：加一顶红色贝雷帽；背景换成夜晚的街道"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <button className="btn btn-primary btn-block" disabled={!src || busy} onClick={apply}>
          {busy ? `生成中 ${elapsed}s（约 2-4 分钟）` : out ? '再生成一次' : '生成像素角色'}
        </button>
        <div className="hint">AI 重绘（40 步）+ 本地像素化 · 结果自动存入素材库</div>
        {note && <div className="ok">{note}</div>}
        {error && <div className="err">{error}</div>}
      </aside>

      <Splitter side="left" />

      <main className="stage">
        <div className="row between">
          <div className="section-title">
            人物像素化 · {cur ? cur.name : '未选档位'} <span className="tag pink">PIXEL-CHAR</span>
          </div>
          <div className="row">
            {out && (
              <>
                <button className="btn btn-ghost" onClick={() => openPreview(out.url, out.name)}>
                  放大
                </button>
                <a className="btn" href={`/api/assets/${out.id}/download`} download={`${out.name}.png`}>
                  下载 PNG
                </a>
                <button className="btn" onClick={() => api.exportZip([out.id])}>
                  导出 ZIP
                </button>
              </>
            )}
          </div>
        </div>

        {out && src ? (
          <>
            <div
              className="cmp"
              ref={boxRef}
              onPointerDown={(e) => {
                dragging.current = true
                e.currentTarget.setPointerCapture(e.pointerId)
                at(e.clientX)
              }}
              onPointerMove={(e) => {
                if (dragging.current) at(e.clientX)
              }}
              onPointerUp={() => {
                dragging.current = false
              }}
              onPointerCancel={() => {
                dragging.current = false
              }}
            >
              <img className="cmp-img" src={src.url} alt="照片" draggable={false} />
              <div className="cmp-after" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
                <img
                  className="cmp-img"
                  src={out.url}
                  alt="像素角色"
                  draggable={false}
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>
              <div className="cmp-line" style={{ left: `${split}%` }}>
                <i />
              </div>
              <span className="cmp-tag left">照片</span>
              <span className="cmp-tag right">{outMeta?.levelName || '像素角色'}</span>
            </div>

            <div className="card">
              <div className="row between">
                <div className="label">像素微调（改这些不用重跑 AI）</div>
                {draft && (
                  <span className="hint">AI 原稿已存：{draft.name}</span>
                )}
              </div>
              <div className="row" style={{ marginTop: 6 }}>
                <div className="grow">
                  <div className="label">横向格数 {grid}</div>
                  <input
                    type="range"
                    min={64}
                    max={200}
                    step={4}
                    value={grid}
                    onChange={(e) => setGrid(+e.target.value)}
                    style={{ width: '100%' }}
                  />
                </div>
                <div style={{ flex: '0 0 auto' }}>
                  <div className="label">色数</div>
                  <div className="chips">
                    {COLOR_CHOICES.map((n) => (
                      <button
                        key={n}
                        className={`chip sm ${colors === n ? 'on' : ''}`}
                        onClick={() => setColors(n)}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="row" style={{ marginTop: 8 }}>
                <button
                  className="btn btn-primary"
                  style={{ height: 36, boxShadow: 'none' }}
                  disabled={!repixelAllowed || repixelBusy}
                  onClick={applyRepixel}
                >
                  {repixelBusy ? '像素化中…' : '按当前参数重新像素化'}
                </button>
                <span className="hint">
                  {repixelAllowed
                    ? '毫秒级完成，不占 GPU。'
                    : draft
                      ? `当前原稿是「${bgName(outMeta?.background || 'keep')}」背景，改背景要重新生成。`
                      : '生成一次后可用。'}
                </span>
              </div>
            </div>
          </>
        ) : (
          <div className="cmp-wrap">
            {src ? (
              <div className="thumb checker" style={{ height: 420, borderColor: 'rgba(255,255,255,0.35)' }}>
                <img src={src.url} alt="待转换" style={{ cursor: 'zoom-in' }} onClick={() => openPreview(src.url, src.name)} />
              </div>
            ) : (
              <div className="empty" style={{ minHeight: 420 }}>
                <span className="pixel">NO PHOTO</span>
                左侧上传一张人物照片，选一档像素化程度，点「生成像素角色」。
                <div className="hint mt8">
                  三档差别是造型语言（头身比 / 眼睛 / 简化程度），不是颗粒大小——
                  颗粒在生成后还能随时微调。
                </div>
              </div>
            )}
            {busy && (
              <div className="warn">
                AI 正在重绘第 {elapsed}s：40 步单次约 2-4 分钟。期间如驻留着另一族模型，
                会先换出显存再加载编辑底座，首次会多花几十秒。
              </div>
            )}
          </div>
        )}

        {history.length > 0 && (
          <div className="card">
            <div className="label">本次会话的结果（点一下回到对比视图）</div>
            <div className="hist-strip">
              {history.map((h) => (
                <button
                  key={h.asset.id}
                  className={`hist-item ${out?.id === h.asset.id ? 'on' : ''}`}
                  onClick={() => {
                    const m = (h.asset.meta || {}) as Record<string, any>
                    setOut(h.asset)
                    setOutMeta(m)
                    setDraft(m.draftAssetId ? { id: m.draftAssetId, name: m.draftAssetId } : null)
                    setSplit(50)
                  }}
                  title={`${h.levelName} · ${bgName(h.background)}`}
                >
                  <span className="hist-img checker">
                    <img src={api.assetThumb(h.asset.id, 160)} alt={h.levelName} loading="lazy" />
                  </span>
                  <span className="hist-name">
                    {h.levelName} · {bgName(h.background)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      <Splitter side="right" />

      <aside className="panel right">
        <div className="section-title">
          生成参数 <span className="tag pink">PIPELINE</span>
        </div>

        {cur ? (
          <div className="card">
            <div className="label">当前档位</div>
            <div style={{ fontSize: 15, fontWeight: 600 }}>{cur.name}</div>
            <div className="chips">
              <span className="chip sm">{cur.heads}</span>
              <span className="chip sm">{cur.grid} 格</span>
              <span className="chip sm">{cur.colors} 色</span>
            </div>
            <div className="hint">{cur.hint}</div>
          </div>
        ) : (
          <div className="card">
            <div className="hint">左侧选一档像素化程度。</div>
          </div>
        )}

        <div className="card">
          <div className="row">
            <div className="grow">
              <div className="label">步数</div>
              <input
                className="input"
                type="number"
                min={10}
                max={60}
                value={steps}
                onChange={(e) => setSteps(Math.max(10, Math.min(60, +e.target.value || STEPS_DEFAULT)))}
              />
            </div>
            <div className="grow">
              <div className="label">长边</div>
              <select className="select" value={maxSide} onChange={(e) => setMaxSide(+e.target.value)}>
                {[768, 1024, 1280].map((n) => (
                  <option key={n} value={n}>
                    {n}px
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div>
            <div className="label">随机种子（可留空）</div>
            <input
              className="input"
              type="number"
              placeholder="留空则每次不同"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
          </div>
          <div className="hint">同种子 + 同参数 + 同照片可复现同一张，方便对比档位与补充要求。</div>
        </div>

        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 10, marginBottom: 4 }}>HOW IT WORKS</div>
          <div style={{ fontSize: 13, lineHeight: 1.7 }}>
            · 第 1 步 AI 重绘：Qwen-Image-Edit 按档位改造型语言（头身比/眼睛/简化）
            <br />· 第 2 步确定性像素化：按横向格数分块取单色 + 限定调色板
            <br />· 透明背景：生成时铺纯绿幕，出图后按色键抠成 alpha（硬边无白晕）
            <br />· AI 出不了真像素，最后一道必须是本地像素化
          </div>
        </div>

        {outMeta?.instruction && (
          <div className="card">
            <div className="label">本次实际指令（可复现）</div>
            <pre
              className="mono"
              style={{ margin: 0, fontSize: 14, lineHeight: 1.7, color: 'var(--muted)', whiteSpace: 'pre-wrap', maxHeight: 220, overflow: 'auto' }}
            >
              {outMeta.instruction}
            </pre>
          </div>
        )}
      </aside>
    </div>
  )
}
