import { useEffect, useMemo, useRef, useState } from 'react'
import { api, Asset } from '../api'

type Props = { incoming?: Asset | null }

const PIPELINES = [
  { id: 'matte', name: '连通域抠图', tag: 'FLOOD-FILL', desc: '只删与边缘连通的同色背景，不误删内部同色像素' },
  { id: 'resize', name: '尺寸规整', tag: 'NEAREST', desc: '最近邻硬缩放 / 等比留白 / RPGMaker 一键' },
  { id: 'sheet', name: 'Sprite Sheet 拆分', tag: 'N×M', desc: '按行列拆帧并生成引擎索引 JSON' },
  { id: 'seamless', name: '左右无缝延展', tag: 'SEAMLESS', desc: '镜像拼接 ×N，接缝数学连续，用于视差图层循环' },
  { id: 'tile', name: '瓦片接缝修补', tag: 'OFFSET-HEAL', desc: '偏移-修复法把贴图变成可平铺单元' },
  { id: 'watermark', name: '去水印（Beta）', tag: 'REV-ALPHA', desc: '需水印掩码；无掩码时不做破坏性处理' },
]

export default function PostProcess({ incoming }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [before, setBefore] = useState('')
  const [pipeline, setPipeline] = useState('matte')
  const [tolerance, setTolerance] = useState(80)
  const [feather, setFeather] = useState(5)
  const [rw, setRw] = useState(144)
  const [rh, setRh] = useState(144)
  const [mode, setMode] = useState('hard')
  const [variant, setVariant] = useState('v1')
  const [cols, setCols] = useState(4)
  const [rows, setRows] = useState(4)
  const [wmSize, setWmSize] = useState(48)
  const [mask, setMask] = useState<File | null>(null)
  const [seamTimes, setSeamTimes] = useState(2)
  const [seamMirror, setSeamMirror] = useState(true)
  const [tileBand, setTileBand] = useState(12)
  const [tileFeather, setTileFeather] = useState(3)
  const [out, setOut] = useState<Asset | null>(null)
  const [frames, setFrames] = useState<Asset[]>([])
  const [index, setIndex] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [over, setOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const accept = (f: File) => {
    setFile(f)
    setBefore(URL.createObjectURL(f))
    setOut(null)
    setFrames([])
    setIndex(null)
    setError('')
    setNote('')
  }

  // 从生成工作台传入资产
  useEffect(() => {
    if (!incoming) return
    fetch(incoming.url)
      .then((r) => r.blob())
      .then((b) => accept(new File([b], `${incoming.name}.png`, { type: 'image/png' })))
      .catch(() => setError('载入生成结果失败'))
  }, [incoming])

  const active = useMemo(() => PIPELINES.find((p) => p.id === pipeline)!, [pipeline])

  const apply = async () => {
    if (!file) return
    setBusy(true)
    setError('')
    setNote('')
    setOut(null)
    setFrames([])
    try {
      if (pipeline === 'matte') {
        const res = await api.matte(file, tolerance, feather)
        setOut(res.asset)
        setNote(`背景色 ${res.meta.background} · 移除 ${(res.meta.removedRatio * 100).toFixed(1)}% 像素`)
      } else if (pipeline === 'resize') {
        const res = await api.resize(file, rw, rh, mode, variant)
        setOut(res.asset)
        setNote(`${res.meta.size[0]}×${res.meta.size[1]} · ${res.meta.mode || res.meta.variant}`)
      } else if (pipeline === 'sheet') {
        const res = await api.splitSheet(file, cols, rows)
        setFrames(res.frames)
        setIndex(res.index)
        setNote(`拆出 ${res.frames.length} 帧 · 每帧 ${res.index.frame_size.w}×${res.index.frame_size.h}`)
      } else if (pipeline === 'seamless') {
        const res = await api.seamless(file, seamTimes, 24, seamMirror)
        setOut(res.asset)
        setNote(`${res.meta.size[0]}×${res.meta.size[1]} · ${res.meta.mode} · ${res.meta.note}`)
      } else if (pipeline === 'tile') {
        const res = await api.tileRepair(file, tileBand, tileFeather)
        setOut(res.asset)
        setNote(`${res.meta.method} · band ${res.meta.band} · 已可平铺`)
      } else {
        if (!mask) {
          setNote('请上传水印掩码（logo alpha 图），否则不做破坏性处理。')
          setBusy(false)
          return
        }
        const fd = new FormData()
        fd.append('file', file)
        fd.append('mask', mask)
        fd.append('size', String(wmSize))
        const res = await fetch('/api/pipeline/watermark', { method: 'POST', body: fd }).then((r) => r.json())
        if (res.asset) {
          setOut(res.asset)
          setNote('已按掩码执行反向 Alpha 混合')
        } else {
          setNote(res.meta?.reason || '未处理')
        }
      }
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          图片输入 <span className="tag">INPUT</span>
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
            if (f) accept(f)
          }}
        >
          {file ? (
            <div>
              <div className="mono">{file.name}</div>
              <div className="hint">点击替换</div>
            </div>
          ) : (
            <div>
              拖入或点击上传 PNG
              <div className="hint" style={{ marginTop: 6 }}>支持生成结果 / 外部素材</div>
            </div>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && accept(e.target.files[0])}
        />

        <div>
          <div className="label">处理管线</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
            {PIPELINES.map((p) => (
              <button
                key={p.id}
                className={`tpl ${pipeline === p.id ? 'on' : ''}`}
                onClick={() => setPipeline(p.id)}
              >
                <span className="row between">
                  <span className="tpl-name">{p.name}</span>
                  <span className="hint">{p.tag}</span>
                </span>
                <span className="tpl-hint">{p.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {pipeline === 'matte' && (
          <div>
            <div className="label">容差 {tolerance} · 羽化 {feather}</div>
            <input
              type="range"
              min={0}
              max={160}
              value={tolerance}
              onChange={(e) => setTolerance(+e.target.value)}
              style={{ width: '100%' }}
            />
            <input
              type="range"
              min={0}
              max={16}
              value={feather}
              onChange={(e) => setFeather(+e.target.value)}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {pipeline === 'resize' && (
          <div>
            <div className="label">模式</div>
            <div className="chips mt8">
              {[
                ['hard', '硬缩放'],
                ['fit', '等比留白'],
                ['rpgmaker', 'RPGMaker'],
              ].map(([v, l]) => (
                <button key={v} className={`chip ${mode === v ? 'on' : ''}`} onClick={() => setMode(v)}>
                  {l}
                </button>
              ))}
            </div>
            {mode === 'rpgmaker' ? (
              <div className="chips mt8">
                {[
                  ['v1', 'V1 144+48'],
                  ['v2', 'V2 144×240'],
                  ['v2_four_row', 'V2 四行 144×192'],
                ].map(([v, l]) => (
                  <button key={v} className={`chip sm ${variant === v ? 'on' : ''}`} onClick={() => setVariant(v)}>
                    {l}
                  </button>
                ))}
              </div>
            ) : (
              <div className="row" style={{ marginTop: 8 }}>
                <div className="grow">
                  <div className="label">宽</div>
                  <input className="input" type="number" value={rw} onChange={(e) => setRw(+e.target.value)} />
                </div>
                <div className="grow">
                  <div className="label">高</div>
                  <input className="input" type="number" value={rh} onChange={(e) => setRh(+e.target.value)} />
                </div>
              </div>
            )}
          </div>
        )}

        {pipeline === 'sheet' && (
          <div className="row">
            <div className="grow">
              <div className="label">列（每行帧数）</div>
              <input className="input" type="number" value={cols} onChange={(e) => setCols(+e.target.value)} />
            </div>
            <div className="grow">
              <div className="label">行（动作组）</div>
              <input className="input" type="number" value={rows} onChange={(e) => setRows(+e.target.value)} />
            </div>
          </div>
        )}

        {pipeline === 'seamless' && (
          <div>
            <div className="label">延展倍数</div>
            <div className="chips mt8">
              {[2, 3, 4].map((n) => (
                <button key={n} className={`chip ${seamTimes === n ? 'on' : ''}`} onClick={() => setSeamTimes(n)}>
                  {n}×
                </button>
              ))}
            </div>
            <div className="row between" style={{ marginTop: 10 }}>
              <div className="label">镜像拼接（推荐，接缝为 0）</div>
              <div className={`toggle ${seamMirror ? 'on' : 'off'}`} onClick={() => setSeamMirror(!seamMirror)}>
                <span />
              </div>
            </div>
            <div className="hint" style={{ marginTop: 6 }}>
              镜像模式下第 2 份水平翻转，接缝两侧像素完全相同，引擎循环播放无痕。
            </div>
          </div>
        )}

        {pipeline === 'tile' && (
          <div>
            <div className="label">
              接缝带宽 {tileBand} · 柔化 {tileFeather}
            </div>
            <input
              type="range"
              min={4}
              max={48}
              value={tileBand}
              onChange={(e) => setTileBand(+e.target.value)}
              style={{ width: '100%' }}
            />
            <input
              type="range"
              min={0}
              max={8}
              value={tileFeather}
              onChange={(e) => setTileFeather(+e.target.value)}
              style={{ width: '100%' }}
            />
            <div className="hint" style={{ marginTop: 6 }}>
              偏移-修复法：先把四边移到画面中央暴露接缝，再局部修复，保留图案结构。
            </div>
          </div>
        )}

        {pipeline === 'watermark' && (
          <div>
            <div className="label">水印掩码（logo alpha PNG）</div>
            <input
              type="file"
              accept="image/png"
              style={{ marginTop: 6, fontSize: 11, color: 'var(--muted)' }}
              onChange={(e) => setMask(e.target.files?.[0] ?? null)}
            />
            <div className="label" style={{ marginTop: 8 }}>水印尺寸 {wmSize}px</div>
            <div className="chips mt8">
              {[48, 96].map((s) => (
                <button key={s} className={`chip sm ${wmSize === s ? 'on' : ''}`} onClick={() => setWmSize(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <button className="btn btn-primary btn-block" disabled={!file || busy} onClick={apply}>
          {busy ? '处理中' : '应用管线'}
        </button>
        <div className="hint">全部处理在本机完成 · 原图不上传云端</div>
        {note && <div className="ok">{note}</div>}
        {error && <div className="err">{error}</div>}
      </aside>

      <main className="stage">
        <div className="row between">
          <div className="section-title">处理预览 · {active.name}</div>
          <div className="row">
            {out && (
              <>
                <a className="btn" href={out.url} download={`${out.name}.png`}>
                  下载 PNG
                </a>
                <button className="btn" onClick={() => api.exportZip([out.id])}>
                  导出 ZIP
                </button>
              </>
            )}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: out || frames.length ? '1fr 32px 1fr' : '1fr', gap: 16, alignItems: 'center' }}>
          <div>
            <div className="hint" style={{ marginBottom: 6 }}>BEFORE</div>
            <div className="thumb checker" style={{ height: 380, borderColor: 'rgba(255,255,255,0.35)' }}>
              {before ? <img src={before} alt="before" /> : <span className="hint">等待上传</span>}
            </div>
          </div>
          {(out || frames.length > 0) && (
            <>
              <div style={{ textAlign: 'center', color: '#00FF88', fontSize: 20 }}>→</div>
              <div>
                <div className="hint" style={{ marginBottom: 6 }}>AFTER</div>
                {out ? (
                  <div className="thumb checker" style={{ height: 380, borderColor: '#00FF88' }}>
                    <img src={out.url} alt="after" />
                  </div>
                ) : (
                  <div className="grid" style={{ maxHeight: 380, overflowY: 'auto', padding: 4 }}>
                    {frames.map((f) => (
                      <div key={f.id} className="thumb checker" style={{ aspectRatio: '1 / 1', borderColor: 'rgba(0,255,136,0.5)' }}>
                        <img src={f.url} alt={f.name} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {index && (
          <div className="card">
            <div className="row between">
              <div className="label">引擎索引 JSON（Godot / 通用）</div>
              <button className="btn btn-ghost" onClick={() => navigator.clipboard.writeText(JSON.stringify(index, null, 2))}>
                复制
              </button>
            </div>
            <pre className="mono" style={{ margin: 0, maxHeight: 160, overflow: 'auto', fontSize: 13, color: 'var(--muted)' }}>
              {JSON.stringify(index, null, 2)}
            </pre>
          </div>
        )}
      </main>

      <aside className="panel right">
        <div className="section-title">
          管线参数 <span className="tag pink">PIPELINE</span>
        </div>
        <div className="card">
          <div className="label">当前管线</div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>{active.name}</div>
          <div className="hint">{active.desc}</div>
        </div>
        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 9, marginBottom: 4 }}>SPEC</div>
          <div style={{ fontSize: 11, lineHeight: 1.7 }}>
            · 抠图：边缘连通 flood fill，容差 80 / 羽化 5
            <br />· 硬缩放：最近邻，禁止插值
            <br />· RPGMaker：144² + 底部扩 48px
            <br />· 拆分：同尺寸切帧 + 时间戳索引
          </div>
        </div>
        {(out || frames[0]) && (
          <div className="card">
            <div className="label">产物</div>
            <div className="mono">{out ? `${out.width}×${out.height}` : `${frames.length} frames`}</div>
            <div className="hint">已自动存入素材库</div>
          </div>
        )}
      </aside>
    </div>
  )
}
