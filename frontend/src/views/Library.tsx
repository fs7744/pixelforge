import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, Asset, AssetFacet, AssetFacets } from '../api'
import { openPreview } from '../Preview'
import { Splitter } from '../split'

const ALL = '全部'
const DIRECT = 'direct'

/** 资产的来源管线；没有 meta.pipeline 的直接生成，后端同样归到 direct */
const pipelineOf = (a: Asset): string => (a.meta?.pipeline as string) || DIRECT

type ChipRowProps = {
  title: string
  hint: string
  active: string
  options: AssetFacet[]
  total: number
  onPick: (id: string) => void
}

/** 一行筛选项。数量随「另一维」的选择实时联动，0 的项标灰但可点。 */
function ChipRow({ title, hint, active, options, total, onPick }: ChipRowProps) {
  return (
    <div className="facet-group">
      <div className="label">
        {title} <span className="facet-hint">{hint}</span>
      </div>
      <div className="chips">
        <button className={`chip ${active === ALL ? 'on' : ''}`} onClick={() => onPick(ALL)}>
          全部
          <span className="facet-n">{total}</span>
        </button>
        {options.map((o) => (
          <button
            key={o.id}
            className={`chip ${active === o.id ? 'on' : ''} ${o.count === 0 ? 'chip-zero' : ''}`}
            onClick={() => onPick(o.id)}
            title={`${o.label} · ${o.count}`}
          >
            {o.label}
            <span className="facet-n">{o.count}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default function Library() {
  const [all, setAll] = useState<Asset[]>([])
  const [facets, setFacets] = useState<AssetFacets | null>(null)
  const [kind, setKind] = useState(ALL)
  // 三视图的正/侧/背/拼板在 kind 上都是 generated，只有 pipeline 能区分子类。
  // 筛选栏按「类型（是什么）+ 来源（哪条管线）」两个维度，任选其一是「全部」。
  const [pipeline, setPipeline] = useState(ALL)
  const [selected, setSelected] = useState<string[]>([])
  const [detail, setDetail] = useState<Asset | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(() => {
    setLoading(true)
    // 一次取全量再在本地切：库是本地文件，量级小，换来筛选 chip 的数量能
    // 跟着另一维实时联动（否则点完一个维度，另一维的数量还是旧的）。
    Promise.all([api.assets(), api.assetFacets()])
      .then(([d, f]) => {
        setAll(d.assets)
        setFacets(f)
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(load, [load])

  const kindLabels = useMemo(() => {
    const m: Record<string, string> = {}
    ;(facets?.kinds || []).forEach((k) => (m[k.id] = k.label))
    return m
  }, [facets])

  const pipeLabels = useMemo(() => {
    const m: Record<string, string> = {}
    ;(facets?.pipelines || []).forEach((p) => (m[p.id] = p.label))
    return m
  }, [facets])

  const matchKind = (a: Asset) => kind === ALL || a.kind === kind
  const matchPipe = (a: Asset) => pipeline === ALL || pipelineOf(a) === pipeline

  const assets = useMemo(
    () => all.filter((a) => matchKind(a) && matchPipe(a)),
    // matchKind/matchPipe 只依赖 kind / pipeline
    [all, kind, pipeline],
  )

  // chip 上的数字：把「另一维」的筛选也算进去，避免显示 32 结果点了是 0
  const kindOptions = useMemo<AssetFacet[]>(
    () =>
      (facets?.kinds || []).map((k) => ({
        ...k,
        count: all.filter((a) => matchPipe(a) && a.kind === k.id).length,
      })),
    [facets, all, pipeline],
  )
  const pipeOptions = useMemo<AssetFacet[]>(
    () =>
      (facets?.pipelines || []).map((p) => ({
        ...p,
        count: all.filter((a) => matchKind(a) && pipelineOf(a) === p.id).length,
      })),
    [facets, all, kind],
  )

  const filtered = kind !== ALL || pipeline !== ALL
  const toggle = (id: string) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }

  /** 卡片角标：kind 是「是什么」，pipeline 补上「哪来的」，重复时只留一个 */
  const badgeOf = (a: Asset) => {
    const k = kindLabels[a.kind] || a.kind
    const p = pipeLabels[pipelineOf(a)] || ''
    return p && p !== k && p !== pipeLabels[DIRECT] ? `${k} · ${p}` : k
  }

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          筛选{' '}
          <span className="tag">
            {filtered ? `${assets.length} / ${all.length}` : `${all.length} ASSETS`}
          </span>
        </div>

        <ChipRow
          title="类型"
          hint="是什么"
          active={kind}
          options={kindOptions}
          total={all.filter(matchPipe).length}
          onPick={setKind}
        />
        <ChipRow
          title="来源"
          hint="哪条管线产出的"
          active={pipeline}
          options={pipeOptions}
          total={all.filter(matchKind).length}
          onPick={setPipeline}
        />

        {filtered && (
          <button
            className="btn btn-ghost btn-block"
            onClick={() => {
              setKind(ALL)
              setPipeline(ALL)
            }}
          >
            清除筛选
          </button>
        )}

        <div className="card">
          <div className="label">批量导出</div>
          <div className="mono">{selected.length} 个已选</div>
          <button
            className="btn btn-primary btn-block mt8"
            disabled={selected.length === 0}
            onClick={() => api.exportZip(selected)}
          >
            导出 ZIP
          </button>
          <button className="btn btn-ghost btn-block" onClick={() => setSelected([])}>
            清空选择
          </button>
        </div>
        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 10, marginBottom: 4 }}>LOCAL FIRST</div>
          <div style={{ fontSize: 13, lineHeight: 1.7 }}>
            所有资产保存在本机 data/assets，导出 ZIP 内含 index.json 索引。
          </div>
        </div>
      </aside>

      <Splitter side="left" />

      <main className="stage">
        <div className="row between">
          <div className="section-title">素材库</div>
          <button className="btn btn-ghost" onClick={load}>
            刷新
          </button>
        </div>

        {loading ? (
          <div className="empty">读取中…</div>
        ) : assets.length === 0 ? (
          <div className="empty">
            <div>
              <span className="pixel">{filtered ? 'NO MATCH' : 'EMPTY'}</span>
              {filtered ? (
                <>
                  当前筛选下没有资产（
                  {kind !== ALL ? (kindLabels[kind] || kind) : '全部类型'} ·{' '}
                  {pipeline !== ALL ? (pipeLabels[pipeline] || pipeline) : '全部来源'}）。
                  <div className="mt8">
                    <button
                      className="btn btn-ghost"
                      onClick={() => {
                        setKind(ALL)
                        setPipeline(ALL)
                      }}
                    >
                      清除筛选
                    </button>
                  </div>
                </>
              ) : (
                '还没有资产。去「生成工作台」出图，或用「后处理」处理一张图。'
              )}
            </div>
          </div>
        ) : (
          <div className="grid">
            {assets.map((a) => (
              <div key={a.id} className="lib-card">
                <div
                  className={`thumb checker ${selected.includes(a.id) ? 'on' : ''}`}
                  style={{ aspectRatio: '1 / 1' }}
                  onClick={() => toggle(a.id)}
                  onDoubleClick={() => setDetail(a)}
                >
                  <img src={a.url} alt={a.name} />
                  <div className="cap">{badgeOf(a)}</div>
                </div>
                <div className="lib-name" title={a.name}>
                  {a.name}
                </div>
                <div className="lib-meta">
                  <span>
                    {a.width}×{a.height}
                  </span>
                  <span>{new Date(a.createdAt * 1000).toLocaleDateString()}</span>
                </div>
                <div className="row">
                  <button
                    className="btn btn-ghost"
                    style={{ height: 32, fontSize: 13 }}
                    onClick={() => openPreview(a.url, a.name)}
                  >
                    放大
                  </button>
                  <a className="btn btn-ghost" href={`/api/assets/${a.id}/download`} download={`${a.name}.png`} style={{ height: 32, fontSize: 13 }}>
                    下载
                  </a>
                  <button
                    className="btn btn-ghost"
                    style={{ height: 32, fontSize: 13 }}
                    onClick={() => {
                      if (confirm('删除该资产？')) api.delAsset(a.id).then(load)
                    }}
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      <Splitter side="right" />

      <aside className="panel right">
        <div className="section-title">
          详情 <span className="tag">DETAIL</span>
        </div>
        {detail ? (
          <>
            <div
              className="thumb checker"
              style={{ aspectRatio: '1 / 1', cursor: 'zoom-in' }}
              onClick={() => openPreview(detail.url, detail.name)}
            >
              <img src={detail.url} alt={detail.name} />
            </div>
            <div className="card">
              <div className="mono">{detail.name}</div>
              <div className="label">
                {detail.width}×{detail.height} · {badgeOf(detail)}
              </div>
              <div className="hint">ID {detail.id}</div>
            </div>
            {detail.meta && Object.keys(detail.meta).length > 0 && (
              <div className="card">
                <div className="label">元数据</div>
                <pre className="mono" style={{ margin: 0, fontSize: 14, color: 'var(--muted)', whiteSpace: 'pre-wrap', maxHeight: 240, overflow: 'auto' }}>
                  {JSON.stringify(detail.meta, null, 2)}
                </pre>
              </div>
            )}
          </>
        ) : (
          <div className="hint">双击素材查看详情</div>
        )}
      </aside>
    </div>
  )
}
