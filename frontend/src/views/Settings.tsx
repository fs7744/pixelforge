import { useEffect, useState } from 'react'
import { api, Settings } from '../api'
import { Splitter } from '../split'

/** 只有本地双底座，配置页也从「多 Provider 卡片网格」收敛成一张卡。 */
export default function SettingsView() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const [testMsg, setTestMsg] = useState('')

  useEffect(() => {
    api.settings().then(setSettings)
  }, [])

  if (!settings) return <div className="empty">读取设置中…</div>

  const cfg = settings.providers?.diffusers || ({} as any)

  const update = (key: string, value: any) => {
    setSettings({
      ...settings,
      providers: { ...settings.providers, diffusers: { ...cfg, [key]: value } },
    })
  }

  const save = async () => {
    setSaving(true)
    setMsg('')
    try {
      const saved = await api.saveSettings(settings)
      setSettings(saved)
      setMsg('已保存')
    } catch (e: any) {
      setMsg(`保存失败：${e.message}`)
    } finally {
      setSaving(false)
      setTimeout(() => setMsg(''), 2500)
    }
  }

  const test = async () => {
    setTestMsg('测试中…')
    try {
      const r = await api.testProvider('diffusers')
      setTestMsg(`${r.ok ? 'OK' : 'FAIL'} · ${r.message}`)
    } catch (e: any) {
      setTestMsg(`FAIL · ${e.message}`)
    }
  }

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          模型服务 <span className="tag pink">LOCAL</span>
        </div>
        <div className="card">
          <div style={{ fontSize: 15, fontWeight: 600 }}>{cfg.label || '本地双底座'}</div>
          <div className="hint" style={{ marginTop: 6 }}>
            生成底座：Z-Image-Turbo（文生图）
            <br />
            编辑底座：Qwen-Image-Edit-2511（指令式编辑）
            <br />
            两个底座都是中文原生，提示词不需要翻译。
          </div>
        </div>
        <button className="btn btn-primary btn-block" disabled={saving} onClick={save}>
          {saving ? '保存中' : '保存设置'}
        </button>
        {msg && <div className={msg.startsWith('保存失败') ? 'err' : 'ok'}>{msg}</div>}
        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 10, marginBottom: 4 }}>LOCAL FIRST</div>
          <div style={{ fontSize: 13, lineHeight: 1.7 }}>
            全部推理在本机完成，素材存在 data/assets，不上传云端。
          </div>
        </div>
      </aside>

      <Splitter side="left" />

      <main className="stage">
        <div className="section-title">本地底座</div>
        <div className="provider-card on">
          <div className="row between">
            <div style={{ fontSize: 16, fontWeight: 600 }}>{cfg.label || '本地双底座'}</div>
            <span className="status ok">默认</span>
          </div>

          <div>
            <div className="label">生成底座目录（Z-Image-Turbo）</div>
            <input
              className="input"
              placeholder="models/z-image-turbo"
              value={cfg.model || ''}
              onChange={(e) => update('model', e.target.value)}
            />
            <div className="hint">
              切到 Qwen-Image-Edit 目录会变成「编辑器」模式：它不能从零出图，
              三视图 / 动画工作流会自动把帧 1 交给上面的生成底座。
            </div>
          </div>
          <div>
            <div className="label">负向提示词（仅编辑底座生效）</div>
            <input
              className="input"
              value={cfg.negativePrompt || ''}
              onChange={(e) => update('negativePrompt', e.target.value)}
            />
          </div>
          <div className="row">
            <button className="btn" onClick={test}>
              测试连接
            </button>
            <button
              className="btn btn-ghost"
              onClick={() =>
                fetch('/api/providers/diffusers/unload', { method: 'POST' })
                  .then((r) => r.json())
                  .then((d) => setTestMsg(d.message))
              }
            >
              卸载模型 / 释放显存
            </button>
          </div>
          {testMsg && (
            <div
              className={testMsg.startsWith('OK') ? 'ok' : 'err'}
              style={{ fontSize: 13 }}
            >
              {testMsg}
            </div>
          )}
        </div>

        <div className="section-title" style={{ marginTop: 8 }}>
          生成默认值
        </div>
        <div className="card">
          <div className="row">
            <div className="grow">
              <div className="label">Steps</div>
              <input
                className="input"
                type="number"
                value={settings.defaults.steps}
                onChange={(e) =>
                  setSettings({ ...settings, defaults: { ...settings.defaults, steps: +e.target.value } })
                }
              />
            </div>
            <div className="grow">
              <div className="label">CFG</div>
              <input
                className="input"
                type="number"
                step="0.1"
                value={settings.defaults.cfg}
                onChange={(e) =>
                  setSettings({ ...settings, defaults: { ...settings.defaults, cfg: +e.target.value } })
                }
              />
            </div>
            <div className="grow">
              <div className="label">宽</div>
              <input
                className="input"
                type="number"
                value={settings.defaults.width}
                onChange={(e) =>
                  setSettings({ ...settings, defaults: { ...settings.defaults, width: +e.target.value } })
                }
              />
            </div>
            <div className="grow">
              <div className="label">高</div>
              <input
                className="input"
                type="number"
                value={settings.defaults.height}
                onChange={(e) =>
                  setSettings({ ...settings, defaults: { ...settings.defaults, height: +e.target.value } })
                }
              />
            </div>
          </div>
        </div>

        <div className="section-title" style={{ marginTop: 8 }}>
          系统
        </div>
        <div className="card">
          <div className="row between">
            <div>
              <div style={{ fontSize: 15 }}>下载时补 AI 标识</div>
              <div className="label">
                默认关闭：角标画进像素后，后续抠图/合成会把它当画面内容带走。
                存储与预览都不烧角标，只在下载瞬间按此开关叠加。
              </div>
            </div>
            <div
              className={`toggle ${settings.aiLabel ? 'on' : 'off'}`}
              onClick={() => setSettings({ ...settings, aiLabel: !settings.aiLabel })}
            >
              <span />
            </div>
          </div>
          <div className="row between" style={{ marginTop: 6 }}>
            <div>
              <div style={{ fontSize: 15 }}>数据目录</div>
              <div className="label">pixelforge/data/assets · 不上传云端</div>
            </div>
            <div className="hint">LOCAL</div>
          </div>
        </div>
      </main>
    </div>
  )
}
