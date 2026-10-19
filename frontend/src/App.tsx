import { useEffect, useState, type CSSProperties } from 'react'
import { api, type Asset } from './api'
import { SplitProvider, useSplit } from './split'
import Workbench from './views/Workbench'
import WorkflowView from './views/Workflow'
import PostProcess from './views/PostProcess'
import Restyle from './views/Restyle'
import Library from './views/Library'
import SettingsView from './views/Settings'
import PreviewOverlay from './Preview'

type Tab = 'workbench' | 'restyle' | 'workflow' | 'post' | 'library' | 'settings'

const TABS: { id: Tab; label: string }[] = [
  { id: 'workbench', label: '生成工作台' },
  { id: 'restyle', label: '画风转换' },
  { id: 'workflow', label: '工作流' },
  { id: 'post', label: '后处理' },
  { id: 'library', label: '素材库' },
  { id: 'settings', label: '模型设置' },
]

/** 当前页写进 hash：刷新后还停在同一页，也能直接把某页链接发给别人 */
const tabFromHash = (): Tab => {
  const h = window.location.hash.replace('#', '')
  return TABS.find((t) => t.id === h)?.id || 'workbench'
}

export default function App() {
  return (
    <SplitProvider>
      <Shell />
    </SplitProvider>
  )
}

function Shell() {
  const [tab, setTabState] = useState<Tab>(tabFromHash)
  const [activeLabel, setActiveLabel] = useState('连接中…')
  const [pending, setPending] = useState<Asset | null>(null)
  // 素材库「转画风」：把资产带到画风转换页，不用先下载再上传
  const [pendingRestyle, setPendingRestyle] = useState<Asset | null>(null)
  const { left, right, dragging } = useSplit()

  const setTab = (t: Tab) => {
    setTabState(t)
    if (window.location.hash !== `#${t}`) {
      window.history.replaceState(null, '', `#${t}`)
    }
  }

  useEffect(() => {
    Promise.all([api.health(), api.providers()])
      .then(([h, p]) => {
        const act = (p.providers || []).find((x: any) => x.id === h.activeProvider)
        setActiveLabel(act ? act.label : h.activeProvider)
      })
      .catch(() => setActiveLabel('后端未连接'))
  }, [tab])

  // 分栏宽度走 CSS 变量，五个视图共用同一套，切换 tab 不跳变
  const style = {
    '--panel-left': `${left}px`,
    '--panel-right': `${right}px`,
  } as CSSProperties

  return (
    <div className={`app ${dragging ? 'dragging-col' : ''}`} style={style}>
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
      {tab === 'restyle' && (
        <Restyle
          incoming={pendingRestyle}
          onSendToPost={(asset) => {
            setPending(asset)
            setTab('post')
          }}
        />
      )}
      {tab === 'post' && <PostProcess incoming={pending} />}
      {tab === 'library' && (
        <Library
          onRestyle={(asset) => {
            setPendingRestyle(asset)
            setTab('restyle')
          }}
        />
      )}
      {tab === 'settings' && <SettingsView />}
      <PreviewOverlay />
    </div>
  )
}
