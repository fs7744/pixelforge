import { useEffect, useMemo, useState } from 'react'
import { api, Asset } from '../api'
import { openPreview } from '../Preview'

type StepDef = {
  id: string
  name: string
  type: string
  desc: string
  prompt?: string
  layerNames?: string[]
  layerNamesEn?: string[]
  n?: number
}
type Workflow = { id: string; name: string; tag: string; desc: string; inputHint: string; steps: StepDef[] }
type RunStep = { id: string; name: string; status: string; artifacts: string[]; meta: any; error?: string }

export default function WorkflowView() {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [wfId, setWfId] = useState('animation')
  const [input, setInput] = useState('灰白英短猫武士，红披风，武士刀')
  const [action, setAction] = useState('挥砍')
  const [run, setRun] = useState<any>(null)
  const [active, setActive] = useState<string>('')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [assets, setAssets] = useState<Record<string, Asset>>({})
  const [promptDraft, setPromptDraft] = useState('')
  const [size, setSize] = useState(512)

  useEffect(() => {
    api.workflows().then((d) => {
      setWorkflows(d.workflows)
      if (d.workflows[0]) setActive(d.workflows[0].steps[0].id)
    })
  }, [])

  const wf = useMemo(() => workflows.find((w) => w.id === wfId), [workflows, wfId])
  const stepDef = useMemo(() => wf?.steps.find((s) => s.id === active), [wf, active])
  const runStep: RunStep | undefined = useMemo(
    () => run?.steps?.find((s: RunStep) => s.id === active),
    [run, active],
  )

  // 切换步骤时刷新提示词草稿
  useEffect(() => {
    if (stepDef) setPromptDraft(promptOf(stepDef, input, action, run))
  }, [stepDef?.id, wfId, run?.id])

  // 拉取产物详情
  useEffect(() => {
    const ids: string[] = []
    run?.steps?.forEach((s: RunStep) => ids.push(...(s.artifacts || [])))
    const missing = ids.filter((id) => !assets[id])
    if (missing.length === 0) return
    Promise.all(missing.map((id) => api.assets().then((d) => d.assets.find((a: Asset) => a.id === id))))
      .then((list) => {
        const next = { ...assets }
        list.forEach((a) => a && (next[a.id] = a))
        setAssets(next)
      })
      .catch(() => void 0)
  }, [run])

  const create = async () => {
    setBusy('create')
    setError('')
    try {
      const r = await api.createRun(wfId, input, {
        action,
        width: size,
        height: size,
        steps: 4,
      })
      setRun(r)
      setActive(r.steps[0].id)
      setAssets({})
    } catch (e: any) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }

  const runOne = async (stepId: string, payload: Record<string, any> = {}) => {
    if (!run) return
    setBusy(stepId)
    setError('')
    try {
      const r = await api.runStep(run.id, stepId, payload)
      setRun(r)
      const idx = r.steps.findIndex((s: RunStep) => s.id === stepId)
      const next = r.steps.slice(idx + 1).find((s: RunStep) => s.status !== 'done')
      if (next) setActive(next.id)
    } catch (e: any) {
      setError(e.message)
      const r = await api.getRun(run.id)
      setRun(r)
    } finally {
      setBusy('')
    }
  }

  const runAll = async () => {
    if (!run) return
    setBusy('all')
    setError('')
    try {
      const r = await api.runAll(run.id)
      setRun(r)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setBusy('')
    }
  }

  const statusOf = (s: RunStep) => {
    if (s.status === 'done') return { label: '已完成', cls: 'ok' }
    if (s.status === 'running') return { label: '进行中', cls: 'warn' }
    if (s.status === 'failed') return { label: '失败', cls: 'off' }
    return { label: '待处理', cls: 'off' }
  }

  const currentStepMeta = runStep?.meta || {}
  const godotZip: string | undefined = currentStepMeta.zip

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          工作流 <span className="tag pink">{wf?.tag || 'WORKFLOW'}</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {workflows.map((w) => (
            <button
              key={w.id}
              className={`tpl ${wfId === w.id ? 'on' : ''}`}
              onClick={() => {
                setWfId(w.id)
                setRun(null)
                setActive(w.steps[0].id)
                setAssets({})
              }}
            >
              <span className="tpl-name">{w.name}</span>
              <span className="tpl-hint">{w.desc}</span>
            </button>
          ))}
        </div>

        <div>
          <div className="label">输入</div>
          <textarea className="textarea" style={{ minHeight: 76 }} value={input} onChange={(e) => setInput(e.target.value)} placeholder={wf?.inputHint} />
          <div className="row" style={{ marginTop: 6 }}>
            {wfId === 'animation' && (
              <div className="grow">
                <div className="label">动作</div>
                <input className="input" value={action} onChange={(e) => setAction(e.target.value)} />
              </div>
            )}
            <div>
              <div className="label">尺寸</div>
              <div className="row" style={{ gap: 6 }}>
                {[384, 512].map((s) => (
                  <button
                    key={s}
                    className={`chip ${size === s ? 'on' : ''}`}
                    onClick={() => setSize(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <button className="btn btn-primary btn-block" disabled={!!busy} onClick={create}>
          {busy === 'create' ? '创建中' : run ? '重新开始工作流' : '开始工作流'}
        </button>

        <div>
          <div className="label">步骤</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
            {wf?.steps.map((s, i) => {
              const st = run?.steps?.[i]
              const info = st ? statusOf(st) : { label: '未开始', cls: 'off' }
              return (
                <button
                  key={s.id}
                  className={`tpl ${active === s.id ? 'on' : ''}`}
                  onClick={() => setActive(s.id)}
                  style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}
                >
                  <span className="hint" style={{ minWidth: 18 }}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span className="tpl-name" style={{ flex: 1 }}>
                    {s.name}
                  </span>
                  <span className={`status ${info.cls}`}>{info.label}</span>
                </button>
              )
            })}
          </div>
        </div>

        <button className="btn btn-block" disabled={!run || !!busy} onClick={runAll}>
          {busy === 'all' ? '执行中' : '一键跑完剩余步骤'}
        </button>
        {error && <div className="err">{error}</div>}
      </aside>

      <main className="stage">
        <div className="row between">
          <div className="section-title">
            步骤 {stepDef ? wf?.steps.findIndex((s) => s.id === stepDef.id)! + 1 : '-'} / {wf?.steps.length} · {stepDef?.name || '选择步骤'}
          </div>
          <div className="row">
            {stepDef && run && (
              <button
                className="btn btn-primary"
                disabled={!!busy || runStep?.status === 'done'}
                onClick={() => runOne(stepDef.id, stepDef.type === 'generate' ? { prompt: promptDraft } : {})}
              >
                {busy === stepDef.id ? '执行中' : runStep?.status === 'done' ? '已完成' : '执行此步'}
              </button>
            )}
          </div>
        </div>

        {!run ? (
          <div className="empty">
            <div>
              <span className="pixel">WORKFLOW</span>
              左侧选择工作流并填写输入，点击「开始工作流」。
              <br />
              每步可单独执行或一键跑完；产物自动传递到下一步。
            </div>
          </div>
        ) : (
          <>
            <div className="card-hard">
              <div className="row between">
                <div style={{ fontSize: 15, fontWeight: 600 }}>{stepDef?.name}</div>
                <span className="hint">{stepDef?.type?.toUpperCase()}</span>
              </div>
              <div className="label">{stepDef?.desc}</div>

              {stepDef?.type === 'generate' && (
                <div>
                  <div className="label">提示词（可编辑，执行时以此为准）</div>
                  <textarea
                    className="textarea"
                    value={promptDraft}
                    onChange={(e) => setPromptDraft(e.target.value)}
                  />
                </div>
              )}

              {stepDef?.type === 'dedup' && (
                <div className="row" style={{ gap: 16 }}>
                  <div>
                    <div className="label">保留帧</div>
                    <div className="mono">{currentStepMeta.kept ?? '-'}</div>
                  </div>
                  <div>
                    <div className="label">剔除近似帧</div>
                    <div className="mono">{currentStepMeta.dropped ?? '-'}</div>
                  </div>
                  <div>
                    <div className="label">方法</div>
                    <div className="hint">{currentStepMeta.method || '-'}</div>
                  </div>
                </div>
              )}

              {stepDef?.type === 'blend' && (
                <div className="hint">
                  生成 {currentStepMeta.generated ?? '-'} 帧 · {currentStepMeta.note || ''}
                </div>
              )}

              {(stepDef?.type === 'godot' || stepDef?.type === 'godot_parallax') && godotZip && (
                <div className="card-white">
                  <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 9, marginBottom: 4 }}>GODOT PACKAGE</div>
                  <div style={{ fontSize: 12, lineHeight: 1.7 }}>
                    {currentStepMeta.kind === 'godot_parallax'
                      ? `图层 ${currentStepMeta.layers} · 画布 ${currentStepMeta.canvas?.join('×')} · scroll_scale ${JSON.stringify(currentStepMeta.scrollScale)}`
                      : `帧数 ${currentStepMeta.frames} · 帧尺寸 ${currentStepMeta.frameSize?.join('×')} · ${currentStepMeta.fps}fps`}
                    <br />
                    {currentStepMeta.note}
                  </div>
                  <a className="btn btn-primary btn-block mt8" href={godotZip} download>
                    下载 Godot 工程 ZIP
                  </a>
                </div>
              )}

              {runStep?.error && <div className="err">{runStep.error}</div>}
            </div>

            <div className="section-title">本步产物</div>
            {runStep?.artifacts?.length ? (
              <div className="grid">
                {runStep.artifacts.map((id, i) => {
                  const a = assets[id]
                  return (
                    <div key={id} className="lib-card">
                      <div
                        className="thumb checker"
                        style={{ aspectRatio: '1 / 1' }}
                        onClick={() => a && openPreview(a.url, a.name)}
                      >
                        {a ? (
                          <img src={a.url} alt={a.name} style={{ cursor: 'zoom-in' }} />
                        ) : (
                          <span className="hint">载入中</span>
                        )}
                        <div className="cap">{a ? `${a.width}×${a.height}` : id}</div>
                      </div>
                      <div className="lib-meta">
                        <span>#{i + 1}</span>
                        {a && (
                          <a href={a.url} download={`${a.name}.png`} style={{ color: 'var(--green)', fontSize: 10 }}>
                            下载
                          </a>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="empty" style={{ minHeight: 140 }}>
                <div>该步骤还没有产物</div>
              </div>
            )}
          </>
        )}
      </main>

      <aside className="panel right">
        <div className="section-title">
          运行信息 <span className="tag">RUN</span>
        </div>
        {run ? (
          <>
            <div className="card">
              <div className="label">Run ID</div>
              <div className="mono">{run.id}</div>
              <div className="label" style={{ marginTop: 6 }}>输入</div>
              <div style={{ fontSize: 12 }}>{run.input}</div>
            </div>
            <div className="card">
              <div className="label">步骤状态</div>
              {run.steps.map((s: RunStep, i: number) => {
                const info = statusOf(s)
                return (
                  <div key={s.id} className="row between" style={{ marginTop: 4 }}>
                    <span style={{ fontSize: 11 }}>
                      {i + 1}. {s.name}
                    </span>
                    <span className={`status ${info.cls}`}>{info.label}</span>
                  </div>
                )
              })}
            </div>
            <div className="card-white">
              <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 9, marginBottom: 4 }}>TIPS</div>
              <div style={{ fontSize: 11, lineHeight: 1.7 }}>
                单步失败可单独重跑，不影响已完成步骤的产物。
                <br />
                导出 ZIP 直接解压进 Godot 项目即可播放动画。
              </div>
            </div>
          </>
        ) : (
          <div className="hint">尚未创建工作流</div>
        )}
      </aside>
    </div>
  )
}

function promptOf(step: StepDef, input: string, action: string, run: any) {
  const tpl = step.prompt || ''
  const layers = step.layerNames || []
  const label = run?.steps?.[0]?.meta?.labels?.[0] as string | undefined
  const layer = layers[0] || label || '目标图层'
  return tpl
    .replace(/\{input\}/g, input)
    .replace(/\{action\}/g, action)
    .replace(/\{layer\}/g, layer)
    .replace(/\{element\}/g, '目标元素')
}
