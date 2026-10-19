import { useMemo, useState } from 'react'
import { api, Asset, AssetFacets } from './api'

const ALL = '全部'
const DIRECT = 'direct'
/** 一屏先铺这么多，剩下的按需展开——原图 PNG 一张几百 KB，全量铺开没有意义 */
const PAGE = 36
const POP_W = 320

const pipelineOf = (a: Asset): string => (a.meta?.pipeline as string) || DIRECT

type Hover = { a: Asset; x: number; y: number }

type Props = {
  library: Asset[]
  facets: AssetFacets | null
  selected: string[]
  /** 已被本地文件占掉的槽位，决定还剩几张可选 */
  usedSlots: number
  maxRefs: number
  onToggle: (id: string) => void
}

/**
 * 图生图的资产库取图区。
 *
 * 原来这里只是 24 张方图 + 浏览器原生 title 提示，三个问题叠在一起：
 * 1. 缩略图用 object-fit: cover，三视图拼板（1560×512）被裁成中间一团，
 *    看上去和单视图没区别；
 * 2. 只有原生 tooltip，内容就一个 name，看不了图；
 * 3. 只取最新 24 张，库里有 150+ 张时大部分根本够不到。
 *
 * 现在：缩略图走后端等比缩略图（宽条仍是宽条）、悬停出大图预览卡、
 * 可以按「来源」筛（三视图/精灵表拆分/直接生成…）、分页展开。
 */
export function RefPicker({ library, facets, selected, usedSlots, maxRefs, onToggle }: Props) {
  const [pipe, setPipe] = useState(ALL)
  const [limit, setLimit] = useState(PAGE)
  const [hover, setHover] = useState<Hover | null>(null)

  const labels = useMemo(() => {
    const m: Record<string, string> = {}
    ;(facets?.pipelines || []).forEach((p) => (m[p.id] = p.label))
    return m
  }, [facets])

  const options = useMemo(
    () => (facets?.pipelines || []).filter((p) => p.count > 0),
    [facets],
  )

  const items = useMemo(
    () => library.filter((a) => pipe === ALL || pipelineOf(a) === pipe),
    [library, pipe],
  )

  const shown = items.slice(0, limit)

  // 预览卡优先放右边（左侧栏本身就在左边，右边空间更大）；
  // 右边放不下就翻到左边，纵向再夹一次，避免贴出视口。
  const showPop = (a: Asset, el: HTMLElement) => {
    const r = el.getBoundingClientRect()
    const x = r.right + 12 + POP_W > window.innerWidth
      ? Math.max(8, r.left - POP_W - 12)
      : r.right + 12
    const y = Math.min(Math.max(8, r.top - 30), Math.max(8, window.innerHeight - 320))
    setHover({ a, x, y })
  }

  const pick = (id: string) => {
    onToggle(id)
    setHover(null)
  }

  const badge = (a: Asset) => labels[pipelineOf(a)] || ''

  return (
    <div className="ref-picker-wrap" onMouseLeave={() => setHover(null)} onScroll={() => setHover(null)}>
      <div className="ref-picker-head">
        <div className="chips ref-chips">
          <button
            className={`chip ${pipe === ALL ? 'on' : ''}`}
            onClick={() => {
              setPipe(ALL)
              setLimit(PAGE)
            }}
          >
            全部来源
            <span className="facet-n">{library.length}</span>
          </button>
          {options.map((p) => (
            <button
              key={p.id}
              className={`chip ${pipe === p.id ? 'on' : ''}`}
              onClick={() => {
                setPipe(p.id)
                setLimit(PAGE)
              }}
            >
              {p.label}
              <span className="facet-n">{p.count}</span>
            </button>
          ))}
        </div>
      </div>

      {library.length === 0 ? (
        <div className="hint">资产库还是空的，先在「生成工作台」出几张图。</div>
      ) : items.length === 0 ? (
        <div className="hint">这个来源下还没有资产。</div>
      ) : (
        <>
          <div className="ref-picker" onScroll={() => setHover(null)}>
            {shown.map((a) => {
              const on = selected.includes(a.id)
              const full = !on && usedSlots + selected.length >= maxRefs
              return (
                <button
                  key={a.id}
                  className={`ref-pick ${on ? 'on' : ''}`}
                  disabled={full}
                  onMouseEnter={(e) => showPop(a, e.currentTarget)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={(e) => showPop(a, e.currentTarget)}
                  onBlur={() => setHover(null)}
                  onClick={() => pick(a.id)}
                >
                  <span className="ref-pick-box">
                    <img src={api.assetThumb(a.id, 160)} alt={a.name} loading="lazy" />
                  </span>
                  <span className="ref-pick-cap">{a.name}</span>
                  {on && <span className="ref-idx">{selected.indexOf(a.id) + 1}</span>}
                </button>
              )
            })}
          </div>
          {items.length > limit && (
            <button className="btn btn-ghost btn-block mt8" onClick={() => setLimit((n) => n + PAGE)}>
              再显示 {Math.min(PAGE, items.length - limit)} 张（还有 {items.length - limit} 张）
            </button>
          )}
        </>
      )}

      {hover && (
        <div className="ref-pop" style={{ left: hover.x, top: hover.y }}>
          <div className="ref-pop-img checker">
            <img src={hover.a.url} alt={hover.a.name} />
          </div>
          <div className="ref-pop-name">{hover.a.name}</div>
          <div className="ref-pop-meta">
            <span>
              {hover.a.width}×{hover.a.height}
            </span>
            {badge(hover.a) && <span className="tag">{badge(hover.a)}</span>}
            {hover.a.width >= hover.a.height * 2 && <span className="tag pink">宽幅</span>}
          </div>
        </div>
      )}
    </div>
  )
}
