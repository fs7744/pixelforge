import { useEffect, useState } from 'react'
import { api } from './api'
import Workbench from './views/Workbench'
import WorkflowView from './views/Workflow'
import PostProcess from './views/PostProcess'
import Library from './views/Library'
import SettingsView from './views/Settings'
import PreviewOverlay from './Preview'

type Tab = 'workbench' | 'workflow' | 'post' | 'library' | 'settings'

const TABS: { id: Tab; label: string }[] = [
  { id: 'workbench', label: '生成工作台' },
  { id: 'workflow', label: '工作流' },
  { id: 'post', label: '后处理' },
  { id: 'library', label: '素材库' },
  { id: 'settings', label: '模型设置' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('workbench')
  const [activeLabel, setActiveLabel] = useState('连接中…')
  const [pending, setPending] = useState<any>(null)

  useEffect(() => {
    Promise.all([api.health(), api.providers()])
      .then(([h, p]) => {
        const act = (p.providers || []).find((x: any) => x.id === h.activeProvider)
        setActiveLabel(act ? act.label : h.activeProvider)
      })
      .catch(() => setActiveLabel('后端未连接'))
  }, [tab])

  return (
    <div className="app">
      <header className="nav">
        <div className="logo">
          <div className="logo-mark">
            <i style={{ background: '#00FF88' }} />
            <i style={{ background: '#FF3366' }} />
            <i style={{ background: '#FFD500' }} />
            <i style={{ background: '#5B57EB' }} />
          </div>
          <span className="logo-en">PIXELFORGE</span>
          <span className="logo-cn">像素工坊</span>
        </div>
        <nav className="nav-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`nav-tab ${tab === t.id ? 'on' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="nav-spacer" />
        <div className="badge">
          <span className="dot" />
          {activeLabel}
        </div>
        <div className="avatar">DEV</div>
      </header>

      {tab === 'workbench' && (
        <Workbench
          onSendToPost={(asset) => {
            setPending(asset)
            setTab('post')
          }}
        />
      )}
      {tab === 'workflow' && <WorkflowView />}
      {tab === 'post' && <PostProcess incoming={pending} />}
      {tab === 'library' && <Library />}
      {tab === 'settings' && <SettingsView />}
      <PreviewOverlay />
    </div>
  )
}
