import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, Asset } from '../api'

const KINDS: [string, string][] = [
  ['全部', '全部'],
  ['generated', '生成'],
  ['matte', '抠图'],
  ['resized', '规整'],
  ['frame', '帧'],
  ['sheet', '精灵表'],
]

export default function Library() {
  const [assets, setAssets] = useState<Asset[]>([])
  const [kind, setKind] = useState('全部')
  const [selected, setSelected] = useState<string[]>([])
  const [detail, setDetail] = useState<Asset | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(() => {
    setLoading(true)
    api
      .assets(kind)
      .then((d) => setAssets(d.assets))
      .finally(() => setLoading(false))
  }, [kind])

  useEffect(load, [load])

  const toggle = (id: string) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }

  const stats = useMemo(() => `${assets.length} ASSETS`, [assets])

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          筛选 <span className="tag">{stats}</span>
        </div>
        <div className="chips">
          {KINDS.map(([id, label]) => (
            <button key={id} className={`chip ${kind === id ? 'on' : ''}`} onClick={() => setKind(id)}>
              {label}
            </button>
          ))}
        </div>
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
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 9, marginBottom: 4 }}>LOCAL FIRST</div>
          <div style={{ fontSize: 11, lineHeight: 1.7 }}>
            所有资产保存在本机 data/assets，导出 ZIP 内含 index.json 索引。
          </div>
        </div>
      </aside>

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
              <span className="pixel">EMPTY</span>
              还没有资产。去「生成工作台」出图，或用「后处理」处理一张图。
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
                  <div className="cap">{a.kind}</div>
                </div>
                <div className="lib-meta">
                  <span>{a.width}×{a.height}</span>
                  <span>{new Date(a.createdAt * 1000).toLocaleDateString()}</span>
                </div>
                <div className="row">
                  <a className="btn btn-ghost" href={`/api/assets/${a.id}/download?label=1`} download={`${a.name}.png`} style={{ height: 26, fontSize: 11 }}>
                    下载
                  </a>
                  <button
                    className="btn btn-ghost"
                    style={{ height: 26, fontSize: 11 }}
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

      <aside className="panel right">
        <div className="section-title">
          详情 <span className="tag">DETAIL</span>
        </div>
        {detail ? (
          <>
            <div className="thumb checker" style={{ aspectRatio: '1 / 1' }}>
              <img src={detail.url} alt={detail.name} />
            </div>
            <div className="card">
              <div className="mono">{detail.name}</div>
              <div className="label">
                {detail.width}×{detail.height} · {detail.kind}
              </div>
              <div className="hint">ID {detail.id}</div>
            </div>
            {detail.meta && Object.keys(detail.meta).length > 0 && (
              <div className="card">
                <div className="label">元数据</div>
                <pre className="mono" style={{ margin: 0, fontSize: 12, color: 'var(--muted)', whiteSpace: 'pre-wrap', maxHeight: 240, overflow: 'auto' }}>
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
