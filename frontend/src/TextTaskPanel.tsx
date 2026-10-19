import { useState } from 'react'
import { Template } from './api'

/**
 * 文本任务面板。
 *
 * 本地底座（Z-Image / Qwen-Image-Edit）只有图像能力，没有任何文本模型，
 * 所以这类模板在本页做不了事。与其摆一个「按钮变灰」的假操作面板，
 * 不如直接告诉用户：复制这段、去哪个 AI、拿到什么、之后怎么用。
 */
export default function TextTaskPanel({ tpl }: { tpl: Template }) {
  const [copied, setCopied] = useState(false)
  const guide = tpl.guide

  const copy = async () => {
    const text = tpl.prompt
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch {
      // 非 https / 无剪贴板权限时的兜底：临时 textarea + execCommand
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      const ok = document.execCommand('copy')
      document.body.removeChild(ta)
      setCopied(ok)
    }
    window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <div className="text-task">
      <div className="tt-why">
        本地两个底座（Z-Image / Qwen-Image-Edit）只出图，没有文本模型，
        这条在这里跑不了。复制下面的提示词，交给对话式 AI 即可。
      </div>

      {guide ? (
        <>
          <ol className="tt-steps">
            {guide.steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>

          <div className="tt-where">
            <span className="label" style={{ marginBottom: 0 }}>可用</span>
            <span className="tt-targets">{guide.targets.join(' · ')}</span>
            {guide.needsImage && <span className="tt-flag">要选能传图的</span>}
          </div>
        </>
      ) : null}

      <div className="tt-prompt-head">
        <span className="label" style={{ marginBottom: 0 }}>提示词 · 整段复制</span>
        <button className="btn" style={{ height: 30, padding: '0 10px' }} onClick={copy}>
          {copied ? '已复制' : '复制'}
        </button>
      </div>
      <pre className="tt-prompt">{tpl.prompt}</pre>

      {guide?.edit && (
        <div className="tt-note">
          <b>怎么改</b>
          {guide.edit}
        </div>
      )}
      {guide?.after && (
        <div className="tt-note">
          <b>结果怎么用</b>
          {guide.after}
        </div>
      )}
    </div>
  )
}
