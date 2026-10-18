import { useEffect, useState } from 'react'
import { api, Settings } from '../api'

export default function SettingsView() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const [testMsg, setTestMsg] = useState<Record<string, string>>({})

  useEffect(() => {
    api.settings().then(setSettings)
  }, [])

  if (!settings) return <div className="empty">读取设置中…</div>

  const update = (pid: string, key: string, value: any) => {
    setSettings({
      ...settings,
      providers: {
        ...settings.providers,
        [pid]: { ...settings.providers[pid], [key]: value },
      },
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

  const test = async (pid: string) => {
    setTestMsg((m) => ({ ...m, [pid]: '测试中…' }))
    try {
      const r = await api.testProvider(pid)
      setTestMsg((m) => ({ ...m, [pid]: `${r.ok ? 'OK' : 'FAIL'} · ${r.message}` }))
    } catch (e: any) {
      setTestMsg((m) => ({ ...m, [pid]: `FAIL · ${e.message}` }))
    }
  }

  return (
    <div className="body">
      <aside className="panel">
        <div className="section-title">
          模型服务 <span className="tag pink">PROVIDERS</span>
        </div>
        <div className="card">
          <div className="label">当前默认 Provider</div>
          <select
            className="select"
            value={settings.activeProvider}
            onChange={(e) => setSettings({ ...settings, activeProvider: e.target.value })}
          >
            {Object.entries(settings.providers).map(([pid, cfg]) => (
              <option key={pid} value={pid}>
                {cfg.label}
              </option>
            ))}
          </select>
        </div>
        <button className="btn btn-primary btn-block" disabled={saving} onClick={save}>
          {saving ? '保存中' : '保存设置'}
        </button>
        {msg && <div className={msg.startsWith('保存失败') ? 'err' : 'ok'}>{msg}</div>}
        <div className="card-white">
          <div style={{ fontFamily: 'Silkscreen, monospace', fontSize: 9, marginBottom: 4 }}>LOCAL FIRST</div>
          <div style={{ fontSize: 11, lineHeight: 1.7 }}>
            内置模拟引擎无需任何模型即可跑通全链路；配置本地 A1111 或云端 API 后即可产出真实素材。
          </div>
        </div>
      </aside>

      <main className="stage">
        <div className="section-title">服务接入</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
          {Object.entries(settings.providers).map(([pid, cfg]) => {
            const on = settings.activeProvider === pid
            return (
              <div key={pid} className={`provider-card ${on ? 'on' : ''}`}>
                <div className="row between">
                  <div style={{ fontSize: 14, fontWeight: 600 }}>{cfg.label}</div>
                  <div className="row">
                    {pid === 'mock' && <span className="status ok">内置</span>}
                    <span className={`status ${on ? 'ok' : 'off'}`}>{on ? '默认' : '待用'}</span>
                  </div>
                </div>

                {pid !== 'mock' && pid !== 'diffusers' && (
                  <>
                    <div>
                      <div className="label">Base URL</div>
                      <input className="input" value={cfg.baseUrl} onChange={(e) => update(pid, 'baseUrl', e.target.value)} />
                    </div>
                    <div>
                      <div className="label">API Key {pid === 'a1111' ? '（通常留空）' : ''}</div>
                      <input
                        className="input"
                        type="password"
                        placeholder={cfg.hasKey ? '已保存（留空不改）' : ''}
                        value={cfg.apiKey?.startsWith('•') ? '' : cfg.apiKey}
                        onChange={(e) => update(pid, 'apiKey', e.target.value)}
                      />
                    </div>
                    <div>
                      <div className="label">模型</div>
                      <input
                        className="input"
                        placeholder={pid === 'a1111' ? 'checkpoint 名称（可空）' : '模型名'}
                        value={cfg.model}
                        onChange={(e) => update(pid, 'model', e.target.value)}
                      />
                    </div>
                  </>
                )}

                {pid === 'mock' && (
                  <div className="hint">
                    产出 4×4 像素精灵表 + 纯色背景，用于验证抠图 / 硬缩放 / 拆帧全链路。
                  </div>
                )}

                {pid === 'diffusers' && (
                  <>
                    <div>
                      <div className="label">模型（HF 仓库名或本地路径）</div>
                      <input
                        className="input"
                        placeholder="stabilityai/sd-turbo"
                        value={cfg.model}
                        onChange={(e) => update(pid, 'model', e.target.value)}
                      />
                    </div>
                    <div>
                      <div className="label">负向提示词</div>
                      <input
                        className="input"
                        value={(cfg as any).negativePrompt || ''}
                        onChange={(e) => update(pid, 'negativePrompt', e.target.value)}
                      />
                    </div>
                    <div className="hint">
                      Turbo 模型自动使用 1-8 步 / CFG 0；6GB 显存会自动开启 attention slicing。
                      国内网络可设环境变量 HF_ENDPOINT=https://hf-mirror.com 后重启服务。
                    </div>
                    <div className="row">
                      <button
                        className="btn"
                        onClick={() =>
                          fetch('/api/providers/diffusers/unload', { method: 'POST' })
                            .then((r) => r.json())
                            .then((d) => setTestMsg((m) => ({ ...m, diffusers: d.message })))
                        }
                      >
                        卸载模型 / 释放显存
                      </button>
                    </div>
                  </>
                )}

                <div className="row">
                  <button className="btn" onClick={() => test(pid)}>
                    测试连接
                  </button>
                  <button
                    className="btn btn-ghost"
                    onClick={() => setSettings({ ...settings, activeProvider: pid })}
                  >
                    设为默认
                  </button>
                </div>
                {testMsg[pid] && (
                  <div className={testMsg[pid].startsWith('OK') ? 'ok' : 'err'} style={{ fontSize: 11 }}>
                    {testMsg[pid]}
                  </div>
                )}
              </div>
            )
          })}
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
              <div style={{ fontSize: 13 }}>AI 生成内容标识</div>
              <div className="label">导出图片左下角附加 AI 角标（合规要求，建议开启）</div>
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
              <div style={{ fontSize: 13 }}>数据目录</div>
              <div className="label">pixelforge/data/assets · 不上传云端</div>
            </div>
            <div className="hint">LOCAL</div>
          </div>
        </div>
      </main>
    </div>
  )
}
