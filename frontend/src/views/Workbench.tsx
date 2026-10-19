import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, Asset, AssetFacets, Backdrop, ProviderInfo, StyleInfo, Template, TemplateField } from '../api'
import { openPreview } from '../Preview'
import { RefPicker } from '../RefPicker'
import { Splitter } from '../split'
import TextTaskPanel from '../TextTaskPanel'
import ThreeViewBoard, { type BoardHandle } from '../ThreeViewBoard'
import { newTaskId, TaskProgressBar, useTaskProgress } from '../useTaskProgress'

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
          // 第一次点击 = 选中（只亮 chip，不弹浮层）；已选中再点才看示例图。
          // 原来点击同时选中+弹浮层，浮层盖住下方提示，用户看不到自己选了什么，
          // 观感上就像「点了没反应」。
          if (active) {
            place()
            setOpen(true)
            return
          }
          onSelect()
          setOpen(false)
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
                  暂无示例图
                  <br />
                  按下面的画风要点判断
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

/**
 * 模板字段：把提示词里那段旧值换成新值。
 *
 * 提示词是用户可编辑的唯一事实来源，字段只负责「改那一段」：
 * 找不到旧值就静默不动（用户可能已经手改过，硬塞会推翻他的手写内容）。
 */
function swapText(text: string, from: string, to: string): string {
  if (!from || !to || from === to || !text.includes(from)) return text
  return text.split(from).join(to)
}

/** 动作序列的拆分：顿号 / 逗号 / 换行都算分隔符 */
const splitPoses = (s: string) =>
  s.split(/[、,，\n]+/).map((x) => x.trim()).filter(Boolean)

/** 模板字段的默认值；有参考图时带 fromRef 的字段默认走「照参考图」 */
function defaultsOf(t: Template, useRef: boolean): Record<string, string> {
  const v: Record<string, string> = {}
  ;(t.fields || []).forEach((f) => {
    v[f.id] = useRef && f.fromRef ? f.fromRef : f.default
  })
  return v
}

/** 接口返回的 prompt 里已经填了默认值，这里换成字段当前值 */
function applyFields(text: string, fields: TemplateField[] | undefined, values: Record<string, string>): string {
  let out = text
  ;(fields || []).forEach((f) => {
    out = swapText(out, f.default, values[f.id] || f.default)
  })
  return out
}

export default function Workbench({ onSendToPost }: Props) {
  const [templates, setTemplates] = useState<Template[]>([])
  const [categories, setCategories] = useState<string[]>(['全部'])
  const [cat, setCat] = useState('全部')
  const [tplId, setTplId] = useState<string>('')
  const [prompt, setPrompt] = useState('')
  // 同一个模板的第二条路：本地出图 / 复制提示词交给外部 AI。
  // 只对声明了 textMode 的模板有意义，其余模板固定按 kind 走。
  const [runMode, setRunMode] = useState<'image' | 'text'>('image')
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [provider, setProvider] = useState('')
  const [styles, setStyles] = useState<StyleInfo[]>([])
  const [styleRecs, setStyleRecs] = useState<Record<string, string[]>>({})
  const [showAllStyles, setShowAllStyles] = useState(false)
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
  // 切 tab 卸载时统一回收 object URL：单张移除时 removeRef 已各自 revoke
  // （重复 revoke 是无害操作），没有这段的话每次切走工作台都漏一批 blob
  const refItemsRef = useRef(refItems)
  useEffect(() => {
    refItemsRef.current = refItems
  }, [refItems])
  useEffect(
    () => () => {
      for (const it of refItemsRef.current) URL.revokeObjectURL(it.url)
    },
    [],
  )
  // 参考图另一种来源：直接从资产库挑（不用先下载再上传）
  const [refAssetIds, setRefAssetIds] = useState<string[]>([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const [library, setLibrary] = useState<Asset[]>([])
  // 取图区的「来源」筛选要中文标签和数量，跟资产库同源（后端 facets 派生）
  const [libFacets, setLibFacets] = useState<AssetFacets | null>(null)
  // 三视图流程画布：四步分开跑，单步可重跑。hasFront/boardBusy 只用来同步左栏按钮
  const boardRef = useRef<BoardHandle>(null)
  const [boardState, setBoardState] = useState({ hasFront: false, busy: false })
  const isThreeView = tplId === 'char_three_view'
  // 模板字段：角色 / 动作序列这类要用户单独填的内容，改动直接落到提示词里
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({})
  // 逐帧动作生成（精灵表）：一次一帧，满足「不满意单独重出」
  const [spriteCols, setSpriteCols] = useState(4)
  const [spriteCell, setSpriteCell] = useState(256)
  // 逐帧走编辑底座，步数越多越听话也越慢；20 步是「能改姿势」的实测下限
  const [spriteSteps, setSpriteSteps] = useState(20)
  // 全局锁色：0 = 每帧各自量化（同一个肤色会逐帧偏色，动画一播就闪）；
  // >0 = 先取一块共享色板再让每帧映射回它。默认 24。
  const [spriteColors, setSpriteColors] = useState(24)
  const [spriteBusy, setSpriteBusy] = useState(false)
  const [spriteWriting, setSpriteWriting] = useState(false)
  const [spriteProg, setSpriteProg] = useState<{ i: number; total: number; pose: string } | null>(null)
  const [spriteFrames, setSpriteFrames] = useState<Record<number, Asset>>({})
  const spriteCancel = useRef(false)
  // 生成进度（SSE）+ 取消：单次 40 步 2-4 分钟，转圈不如给步数
  const taskProgress = useTaskProgress()
  // 生成底色。默认「中性灰」而不是绿幕：实测模型声明 #00FF00 会画成薄荷绿
  // [90,199,126] + 径向渐变，还会把绿染到角色边缘（内圈偏底色像素 76.5% vs
  // 中性灰 33.3%）。数据见 pipelines/backdrop.py。
  const [backdrops, setBackdrops] = useState<Backdrop[]>([])
  const [backdrop, setBackdrop] = useState('neutral')

  useEffect(() => {
    api.backdrops()
      .then((d) => {
        setBackdrops(d.backdrops)
        setBackdrop(d.default)
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    api.templates().then((d) => {
      setTemplates(d.templates)
      setCategories(d.categories)
      if (d.templates[0]) {
        setTplId(d.templates[0].id)
        setPrompt(d.templates[0].prompt)
        setStyleId(d.templates[0].defaultStyle || '')
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
    api.styles().then((d) => {
      setStyles(d.styles)
      setStyleRecs(d.recommended || {})
    })
  }, [])

  const filtered = useMemo(
    () => (cat === '全部' ? templates : templates.filter((t) => t.category === cat)),
    [templates, cat],
  )

  // 底座是中文原生（Z-Image / Qwen-Edit），模板只有中文一版
  const promptFor = (t: Template) => t.prompt

  // 切模板时若提示词没被改过，自动换成模板原文（字段按当前参考图状态取默认）
  useEffect(() => {
    const t = templates.find((x) => x.id === tplId)
    if (!t) return
    const untouched =
      prompt === '' ||
      templates.some((x) => applyFields(x.prompt, x.fields, defaultsOf(x, false)) === prompt)
    if (untouched) {
      const fv = defaultsOf(t, refItems.length + refAssetIds.length > 0)
      setFieldValues(fv)
      setPrompt(applyFields(promptFor(t), t.fields, fv))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, templates, tplId])

  const pickTemplate = (t: Template) => {
    setTplId(t.id)
    // 有参考图时，角色字段直接落成「照参考图」，不再写死某个示例角色
    const fv = defaultsOf(t, totalRefs > 0)
    setFieldValues(fv)
    setPrompt(t.fields?.length ? applyFields(promptFor(t), t.fields, fv) : promptFor(t))
    // 切模板回到它自己的默认路线：文本模板只能是 text，双模式模板默认本地出图
    setRunMode(t.kind === 'text' ? 'text' : 'image')
    if (t.params?.width) setWidth(t.params.width)
    if (t.params?.height) setHeight(t.params.height)
    // 画风不写在提示词里（写死就会和选择器打架），模板自带默认画风，这里点亮它；
    // 文本任务不出图、编辑提取类模板画风跟随原图，都留空
    setStyleId(t.kind === 'text' || t.styleLocked ? '' : t.defaultStyle || '')
  }

  /**
   * 改模板字段 = 改提示词里那一段。
   * 参考图从无到有（或反之）时，带 fromRef 的字段跟着切换成「照参考图」/默认描述。
   */
  const setField = (id: string, val: string) => {
    const prev = fieldValues[id] || ''
    setPrompt((p) => swapText(p, prev, val))
    setFieldValues((v) => ({ ...v, [id]: val }))
    // 帧是按动作序号存的：动作数量一变，旧帧就对不上号了；只改措辞则保留
    if (id === 'actions' && splitPoses(prev).length !== splitPoses(val).length) {
      setSpriteFrames({})
    }
  }

  // 参考图最多 4 张：再多也分不清主次，编码时间还线性上涨
  const MAX_REFS = 4
  const totalRefs = refItems.length + refAssetIds.length

  // 参考图数量 0 ↔ >0：把「角色」字段在默认描述与「照参考图」之间对调
  useEffect(() => {
    const t = templates.find((x) => x.id === tplId)
    const f = t?.fields?.find((x) => x.fromRef)
    if (!f?.fromRef) return
    const want = totalRefs > 0 ? f.fromRef : f.default
    const cur = fieldValues[f.id]
    if (!cur || cur === want) return
    const other = totalRefs > 0 ? f.default : f.fromRef
    // 用户自己改写过角色描述（既不是默认值也不是参考值）就不要覆盖
    if (cur !== other) return
    setFieldValues((v) => ({ ...v, [f.id]: want }))
    setPrompt((p) => swapText(p, cur, want))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [totalRefs, tplId, templates])


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
    if (!next) return
    // 每次展开都重新拉：只拉一次的话，刚生成的图在取图区里永远看不到，
    // 必须整页刷新才行（原来就是这么写的）。
    api
      .assets()
      .then((d) => setLibrary(d.assets))
      .catch(() => setLibrary([]))
    api
      .assetFacets()
      .then(setLibFacets)
      .catch(() => setLibFacets(null))
  }

  const toggleAsset = (id: string) => {
    setRefAssetIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      if (prev.length + refItems.length >= MAX_REFS) return prev
      return [...prev, id]
    })
  }

  const run = async () => {
    // 三视图走流程画布：一次只推进一张，任何一步都能单独重跑
    if (isThreeView) {
      boardRef.current?.start()
      return
    }
    setBusy(true)
    setError('')
    const taskId = newTaskId()
    taskProgress.start(taskId)
    try {
      // 有参考图 = 参考图生成（图生图）：Z-Image 没有参考图输入，后端会自动改走
      // Qwen-Image-Edit。首图是主参考，其余作补充。
      const data = refItems.length || refAssetIds.length
        ? await api.generateRef(
            {
              prompt,
              style: effectiveStyle,
              width,
              height,
              steps,
              cfg,
              n,
              seed: seed.trim() ? Number(seed) : null,
              refAssetIds,
              taskId,
            },
            refItems.map((r) => r.file),
          )
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
            style: effectiveStyle,
            taskId,
          })
      setResults((prev) => [...data.images, ...prev])
      setMeta(data.meta)
      if (data.images[0]) setSelected(data.images[0].id)
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
      taskProgress.stop()
    }
  }

  // 画布每出一张就并进历史记录，右栏和结果区都能看到
  const handleBoardAsset = useCallback((a: Asset) => {
    setResults((prev) => [a, ...prev])
    setSelected(a.id)
  }, [])

  // ---------- 精灵表：逐帧动作生成 ----------
  // 动作列表就是「动作序列」字段本身（顿号/逗号/换行都能分隔），
  // 用户在字段里改一次，提示词和逐帧生成同时生效，不会出现两处分叉。
  const poseList = () => splitPoses(fieldValues.actions || '')

  const runSpriteFrames = async (only?: number) => {
    const list = poseList()
    if (!list.length) {
      setError('先填「动作序列」，一行一个动作')
      return
    }
    if (!totalRefs) {
      setError('逐帧生成需要一张角色参考图：先在下面「参考图」里选一张')
      return
    }
    const refId = refAssetIds[0]
    const refFile = refId ? null : refItems[0]?.file || null
    setSpriteBusy(true)
    setError('')
    spriteCancel.current = false
    const taskId = newTaskId()
    taskProgress.start(taskId)
    try {
      const targets = only == null ? list.map((_, i) => i) : [only]
      for (const i of targets) {
        if (spriteCancel.current) break
        setSpriteProg({ i, total: list.length, pose: list[i] })
        const d = await api.spriteFrame(
          {
            pose: list[i],
            index: i,
            total: list.length,
            refAssetId: refId || undefined,
            subject: fieldValues.subject,
            style: effectiveStyle,
            steps: spriteSteps,
            seed: seed.trim() ? Number(seed) : null,
            width,
            height,
            backdrop,
            taskId,
          },
          refFile,
        )
        if (!d.image) throw new Error(d.detail || '这一帧没出图')
        setSpriteFrames((prev) => ({ ...prev, [i]: d.image }))
        setResults((prev) => [d.image, ...prev])
      }
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setSpriteProg(null)
      setSpriteBusy(false)
      taskProgress.stop()
    }
  }

  const composeSprite = async () => {
    const list = poseList()
    const ids = list.map((_, i) => spriteFrames[i]?.id).filter(Boolean) as string[]
    if (!ids.length) {
      setError('还没有帧可以合成')
      return
    }
    if (ids.length !== list.length) {
      setError(`还有 ${list.length - ids.length} 帧没生成（或重出后没补上）`)
      return
    }
    setSpriteWriting(true)
    setError('')
    try {
      const d = await api.spriteCompose({
        frameAssetIds: ids,
        cols: Math.max(1, spriteCols),
        size: Math.max(64, spriteCell),
        matte: true,
        colors: spriteColors,
      })
      if (!d.asset) throw new Error(d.detail || '合成失败')
      // review = 帧审查图（移植自 h3 的 extraction.review）：和精灵表一起
      // 进结果区，验收时先看它——哪帧跑飞、接缝跳不跳一眼可见
      setResults((prev) => [d.asset, ...(d.review ? [d.review] : []), ...prev])
      setSelected(d.asset.id)
      const g = d.index?.grid
      const verdict = (d.meta?.metricsVerdict || []) as string[]
      const pal = d.meta?.palette?.colors
      const warns = (d.meta?.warnings || []) as string[]
      setMeta({
        ...d.meta,
        note: `精灵表 ${g?.cols}×${g?.rows} · 单帧 ${d.index?.frame_size?.w}×${d.index?.frame_size?.h} · 已共享缩放并对齐基线`
          + (pal ? ` · 全局锁色 ${pal} 色` : '')
          + (verdict.length ? ` · ${verdict.join('；')}` : '')
          + (warns.length ? ` · 警告：${warns.join('；')}` : '')
          + (d.review ? ` · 附帧审查图（序号+逐帧差+接缝标色）` : ''),
      })
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setSpriteWriting(false)
    }
  }

  const current = results.find((r) => r.id === selected)
  const currentTpl = templates.find((t) => t.id === tplId)
  // 纯文本模板（kind=text）根本没得选；带 textMode 的模板由用户切路线
  const isPureText = currentTpl?.kind === 'text'
  const canSwitchMode = !!currentTpl?.textMode
  // 文本路线上本页不出图：整块操作面板换成「去别的 AI 做」指引
  const isTextTask = isPureText || (runMode === 'text' && canSwitchMode)
  const textPrompt = isPureText
    ? applyFields(currentTpl!.prompt, currentTpl!.fields, fieldValues)
    : applyFields(currentTpl?.textMode?.prompt || '', currentTpl?.fields, fieldValues)
  const textGuide = isPureText ? currentTpl?.guide : currentTpl?.textMode?.guide
  // 当前模板适用的画风：后端按分类给出推荐集（tags 太粗，按它过滤等于没过滤）。
  // 文本路线不出图、编辑/提取类模板画风跟随原图——这两种都不显示画风区，
  // 摆一个选了也不起作用的选择器，比不显示更糟。
  const recStyleIds = styleRecs[currentTpl?.category || '角色'] || []
  // 有参考图时，「画风跟随参考图」的模板也不再让用户选画风：
  // 前置一句「写实 3A」会和参考图的画风打架，帧与帧之间还会各画各的。
  const refsPresent = totalRefs > 0
  const styleFollowsRef = refsPresent && !!currentTpl?.styleFollowsRef
  const styleLockedNow = !!currentTpl?.styleLocked || styleFollowsRef
  const showStyleBlock = !isTextTask && !styleLockedNow && recStyleIds.length > 0
  const visibleStyles = showAllStyles
    ? styles
    : styles.filter((s) => recStyleIds.includes(s.id) || s.id === styleId)
  // 选了但不在本类推荐里的画风：仍然显示（否则用户看不到自己选了什么），标注一下
  const styleOutOfRec = !!styleId && !recStyleIds.includes(styleId)
  // 当前画风实际会拼到提示词里的那句话；给「最终提示词」预览用
  const styleWords = styleLockedNow ? '' : styles.find((s) => s.id === styleId)?.words || ''
  // 后端 apply_style 是幂等的（已含就不再拼），预览也要一致，否则手动粘过会显示两遍
  const finalPrompt = styleWords && !prompt.startsWith(styleWords)
    ? `${styleWords}。${prompt}`
    : prompt
  const effectiveStyle = styleLockedNow ? undefined : styleId || undefined

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          {isTextTask ? '文本生成' : '生成参数'}{' '}
          {isTextTask ? <span className="tag pink">TEXT</span> : <span className="tag">PARAMS</span>}
        </div>

        {!isTextTask && (
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
        )}

        <div>
          <div className="label">模板库</div>
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

        {canSwitchMode && (
          <div>
            <div className="label">运行方式</div>
            <div className="chips mt8">
              <button
                className={`chip ${runMode === 'image' ? 'on' : ''}`}
                onClick={() => setRunMode('image')}
              >
                图片生成 · 本地
              </button>
              <button
                className={`chip ${runMode === 'text' ? 'on' : ''}`}
                onClick={() => setRunMode('text')}
              >
                文本生成 · 外部 AI
              </button>
            </div>
            <div className="hint" style={{ marginTop: 8 }}>
              {runMode === 'text'
                ? currentTpl?.textMode?.hint
                : '本地底座直接出图，快、免费；多格排布和帧间一致性看运气，排不满就换外部 AI。'}
            </div>
          </div>
        )}

        {isTextTask && (
          <TextTaskPanel
            prompt={textPrompt}
            guide={textGuide}
            why={
              canSwitchMode
                ? '你选了「文本生成 · 外部 AI」：这一条不在本地跑，复制提示词交给能出图的对话式 AI。'
                : undefined
            }
          />
        )}

        {showStyleBlock && (
          <div>
            <div className="row between" style={{ alignItems: 'center' }}>
              <div className="label" style={{ marginBottom: 0 }}>
                美术画风 ·{' '}
                {showAllStyles
                  ? `全部 ${styles.length} 种`
                  : `适合「${currentTpl?.category || '角色'}」的 ${recStyleIds.length} 种`}
              </div>
              <button
                className="btn btn-ghost"
                style={{ padding: '2px 8px', fontSize: 13 }}
                onClick={() => setShowAllStyles((v) => !v)}
              >
                {showAllStyles ? '只看推荐' : `全部 ${styles.length} 种`}
              </button>
            </div>
            <div className="chips mt8">
              <button className={`chip ${styleId === '' ? 'on' : ''}`} onClick={() => setStyleId('')}>
                不限定
              </button>
              {visibleStyles.map((s) => (
                <StyleChip key={s.id} style={s} active={styleId === s.id} onSelect={() => setStyleId(s.id)} />
              ))}
            </div>
            {styleId && (() => {
              const s = styles.find((x) => x.id === styleId)
              if (!s) return null
              return (
                <div className="hint" style={{ marginTop: 8 }}>
                  {s.hint}
                  {styleOutOfRec && (
                    <>
                      {' '}
                      <span className="tag pink">非本类推荐</span>
                    </>
                  )}
                  {s.pixelize && (
                    <>
                      {' '}
                      <span className="tag" style={{ color: 'var(--green)' }}>
                        建议像素化 {s.pixelize.block}× · {s.pixelize.colors} 色
                      </span>
                    </>
                  )}
                  {s.words && (
                    <div className="style-words">
                      <span className="tag">前置到提示词开头</span> {s.words}
                    </div>
                  )}
                </div>
              )
            })()}
          </div>
        )}

        {!isTextTask && (currentTpl?.fields?.length ?? 0) > 0 && (
          <div>
            <div className="label">模板字段（改动会写进下面的提示词）</div>
            <div className="tpl-fields">
              {(currentTpl!.fields || []).map((f) => (
                <div key={f.id}>
                  <div className="row between" style={{ alignItems: 'center' }}>
                    <span className="tpl-field-label">{f.label}</span>
                    {f.presets?.length ? (
                      <span className="chips" style={{ gap: 4 }}>
                        {f.presets.map((p) => (
                          <button
                            key={p.label}
                            className={`chip sm ${fieldValues[f.id] === p.value ? 'on' : ''}`}
                            onClick={() => setField(f.id, p.value)}
                          >
                            {p.label}
                          </button>
                        ))}
                      </span>
                    ) : null}
                  </div>
                  <input
                    className="input"
                    value={fieldValues[f.id] ?? ''}
                    onChange={(e) => setField(f.id, e.target.value)}
                  />
                  {f.hint && <div className="hint" style={{ marginTop: 4 }}>{f.hint}</div>}
                </div>
              ))}
            </div>
          </div>
        )}

        {!isTextTask && (
          <div>
            <div className="label">提示词（可编辑）</div>
            <textarea className="textarea" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            {styleWords && (
              <details className="sent-preview">
                <summary>最终提示词（画风已加在最前面）</summary>
                <pre>{finalPrompt}</pre>
              </details>
            )}
            {currentTpl?.notes?.length ? (
              <ul className="tpl-notes">
                {currentTpl.notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            ) : null}
            <div className="hint" style={{ marginTop: 6 }}>
              {styleFollowsRef
                ? '有参考图：画风与角色外观都跟随参考图（编辑底座只改要求改的部分），所以不再前置画风词。要换画风请去「后处理 · AI 画风重统一」。'
                : currentTpl?.styleLocked
                  ? '这类任务是对参考图做编辑/提取，画风跟随原图，不参与画风切换。'
                  : '中文直喂本地底座（Z-Image / Qwen-Edit），全程不做英文翻译。'}
            </div>
          </div>
        )}

        {!isTextTask && !isThreeView && (
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
                <div className="ref-thumb" key={id} title={a?.name}>
                  {a ? (
                    <img src={api.assetThumb(a.id, 160)} alt={a.name} />
                  ) : (
                    <div className="ref-miss">?</div>
                  )}
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
            <RefPicker
              library={library}
              facets={libFacets}
              selected={refAssetIds}
              usedSlots={refItems.length}
              maxRefs={MAX_REFS}
              onToggle={toggleAsset}
            />
          )}

          <div className="hint" style={{ marginTop: 6 }}>
            {totalRefs
              ? `已选 ${totalRefs} 张：走 Qwen-Image-Edit 指令式生成，输出尺寸用上方「宽/高」。第一张是主参考，其余作补充（如正面+侧面、角色+场景）。`
              : '不选 = 文生图（Z-Image）。需要按参考图生成时加图，会自动改用编辑底座；首图是主参考。'}
          </div>
        </div>
        )}

        {!isTextTask && !isThreeView && tplId === 'char_sprite_sheet' && (
          <div className="card sprite-loop">
            <div className="row between" style={{ alignItems: 'center' }}>
              <div className="label" style={{ marginBottom: 0 }}>
                逐帧动作生成 <span className="tag pink">RECOMMENDED</span>
              </div>
              <span className="hint">{poseList().length} 帧</span>
            </div>
            <div className="hint" style={{ marginTop: 6 }}>
              一张图排 4×4 时模型只会把同一个姿势贴 16 遍，动作循环用不了。这里改成
              一帧一次编辑：外观/画风由参考图钉住，每帧只改姿势，最后本地拼板并对齐基线。
              单帧 1–3 分钟（编辑底座），4 帧约 5–12 分钟；动作写得越具体，改得越准
              （「迈步」会自动补成「一腿前迈屈膝、另一腿后蹬、手臂摆动」这种四肢描述）。
            </div>

            <div className="label" style={{ marginTop: 10 }}>逐帧进度</div>
            <div className="sprite-frames">
              {poseList().map((p, i) => {
                const f = spriteFrames[i]
                const running = spriteProg?.i === i
                return (
                  <div className={`sprite-cell ${f ? 'done' : ''} ${running ? 'run' : ''}`} key={`${i}-${p}`}>
                    {f ? (
                      <img src={f.url} alt={p} onClick={() => openPreview(f.url, p)} title="点击看大图" />
                    ) : (
                      <span className="sprite-empty">{running ? '生成中' : '—'}</span>
                    )}
                    <span className="sprite-pose" title={p}>{p}</span>
                    <button
                      className="btn btn-ghost sprite-redo"
                      disabled={spriteBusy || !totalRefs}
                      onClick={() => runSpriteFrames(i)}
                    >
                      {f ? '重出' : '出这一帧'}
                    </button>
                  </div>
                )
              })}
            </div>

            <div className="row" style={{ marginTop: 8 }}>
              <div className="grow">
                <div className="label">每行帧数</div>
                <input
                  className="input"
                  type="number"
                  min={1}
                  max={8}
                  value={spriteCols}
                  onChange={(e) => setSpriteCols(+e.target.value)}
                />
              </div>
              <div className="grow">
                <div className="label">单帧尺寸</div>
                <input
                  className="input"
                  type="number"
                  min={64}
                  max={512}
                  value={spriteCell}
                  onChange={(e) => setSpriteCell(+e.target.value)}
                />
              </div>
              <div className="grow">
                <div className="label">步数</div>
                <input
                  className="input"
                  type="number"
                  min={8}
                  max={40}
                  value={spriteSteps}
                  onChange={(e) => setSpriteSteps(+e.target.value)}
                />
              </div>
            </div>

            {backdrops.length > 0 && (
              <>
                <div className="label" style={{ marginTop: 10 }}>生成底色（决定抠图干不干净）</div>
                <div className="chips mt8">
                  {backdrops.map((b) => (
                    <button
                      key={b.id}
                      className={`chip ${backdrop === b.id ? 'on' : ''}`}
                      onClick={() => setBackdrop(b.id)}
                    >
                      <span className="bd-dot" style={{ background: b.hex }} />
                      {b.name}
                    </button>
                  ))}
                </div>
                <div className="hint" style={{ marginTop: 6 }}>
                  {backdrops.find((b) => b.id === backdrop)?.hint}
                  {' '}合成用语义分割（u2net），它不看背景色，所以中性底不会被误抠。
                </div>
              </>
            )}

            <div className="label" style={{ marginTop: 10 }}>全局锁色（帧间不偏色）</div>
            <div className="chips mt8">
              {[
                [0, '关闭'],
                [16, '16 色'],
                [24, '24 色'],
                [32, '32 色'],
              ].map(([v, l]) => (
                <button
                  key={String(v)}
                  className={`chip ${spriteColors === v ? 'on' : ''}`}
                  onClick={() => setSpriteColors(v as number)}
                >
                  {l as string}
                </button>
              ))}
            </div>
            <div className="hint" style={{ marginTop: 6 }}>
              每帧各自量化时，同一个肤色第 1 帧偏橙、第 3 帧偏红，动画一播就闪。
              开启后先取一块共享色板，再让每一帧映射回它；合成完会在结果说明里给出
              首尾衔接、帧间差、色板外颜色等一致性指标。
            </div>

            <div className="row" style={{ marginTop: 8 }}>
              <button
                className="btn btn-primary grow"
                disabled={spriteBusy || !totalRefs || poseList().length === 0}
                onClick={() => runSpriteFrames()}
              >
                {spriteBusy
                  ? `生成中 ${spriteProg ? `${spriteProg.i + 1}/${spriteProg.total}` : ''}`
                  : `全部生成（${poseList().length} 帧）`}
              </button>
              {spriteBusy ? (
                <button className="btn" onClick={() => (spriteCancel.current = true)}>
                  停止
                </button>
              ) : (
                <button
                  className="btn"
                  disabled={spriteWriting || !Object.keys(spriteFrames).length}
                  onClick={composeSprite}
                >
                  {spriteWriting ? '合成中…' : '合成精灵表'}
                </button>
              )}
            </div>
            {!totalRefs && (
              <div className="warn" style={{ marginTop: 8 }}>
                逐帧生成需要一张角色参考图：先在下面「参考图 · 图生图」里选一张（角色三视图或立绘都行），
                否则每帧的角色会各画各的。
              </div>
            )}
            {spriteProg && (
              <div className="hint" style={{ marginTop: 6 }}>
                正在生成第 {spriteProg.i + 1}/{spriteProg.total} 帧 · {spriteProg.pose}
              </div>
            )}
            <div className="hint" style={{ marginTop: 6 }}>
              「动作序列」在上面「模板字段」里改，这里和提示词共用同一份。合成时按顺序铺格子，
              尺寸统一、底部对齐、绿底自动抠掉，并按上面的锁色设置统一帧间配色。
            </div>
          </div>
        )}

        {!isTextTask && (
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
              <input
                className="input"
                type="number"
                min={1}
                max={4}
                value={n}
                disabled={isThreeView}
                title={isThreeView ? '三视图固定一张一张出，每步单独重跑' : undefined}
                onChange={(e) => setN(+e.target.value)}
              />
            </div>
            <div className="grow">
              <div className="label">种子（空=随机）</div>
              <input className="input" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" />
            </div>
          </div>
        </div>
        )}

        {!isTextTask && (
          <button
            className="btn btn-primary btn-block"
            disabled={busy || boardState.busy || !prompt.trim()}
            onClick={run}
          >
            {busy || boardState.busy
              ? '生成中'
              : isThreeView
                ? boardState.hasFront
                  ? '重生成正视图'
                  : '生成正视图'
                : '生成'}
          </button>
        )}
        {(busy || boardState.busy) && (
          <TaskProgressBar progress={taskProgress.progress} onCancel={taskProgress.cancel} />
        )}
        {isThreeView && (
          <div className="hint">
            三视图按「正 → 侧 → 背 → 拼板」分四步跑，每张出来都能先看一眼；
            不满意就单独重跑那一步，不用整套重来。
          </div>
        )}
        {meta?.vramWarning && <div className="warn">{meta.vramWarning}</div>}
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

      <main className={`stage ${isThreeView ? 'canvas' : ''}`}>
        {isThreeView ? (
          <ThreeViewBoard
            ref={boardRef}
            params={{ prompt, styleId, width, height, steps, cfg, seed }}
            onAsset={handleBoardAsset}
            onChange={setBoardState}
            onSendToPost={onSendToPost}
          />
        ) : (
        <>
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
        </>
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
