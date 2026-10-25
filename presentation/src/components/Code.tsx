import { useRef } from 'react'
import { useBuild } from '../deck/SlideFrame'
import { build } from './layouts'

// one-pass highlighter: comments, JSON keys, strings, numbers, keywords (Python / JS / JSON)
const TOKENS = /(?<c>#.*|\/\/.*)|(?<k>"(?:[^"\\]|\\.)*"(?=\s*:))|(?<s>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(?<n>\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)|(?<w>\b(?:def|return|import|from|for|in|if|else|elif|while|class|with|as|lambda|None|True|False|and|or|not|const|let|function|export|true|false|null)\b)/g

function highlight(line: string) {
  const out: (string | { cls: string; text: string })[] = []
  let last = 0
  for (const m of line.matchAll(TOKENS)) {
    if (m.index > last) out.push(line.slice(last, m.index))
    out.push({ cls: Object.entries(m.groups!).find(([, v]) => v !== undefined)![0], text: m[0] })
    last = m.index + m[0].length
  }
  out.push(line.slice(last))
  return out.map((t, i) => (typeof t === 'string' ? t : <span key={i} className={t.cls}>{t.text}</span>))
}

/**
 * Code block. focus = { step: [1-based line numbers] }: at that build step those lines light up and the
 * rest dim; an empty list restores all lines.
 */
export function Code({ code, focus, step, size, className = '' }: {
  code: string; focus?: Record<number, number[]>; step?: number; size?: number; className?: string
}) {
  const ref = useRef<HTMLPreElement>(null)
  useBuild((k, tl, at) => {
    const lines = focus?.[k]
    if (!lines || !ref.current) return
    const els = [...ref.current.querySelectorAll('.ln')]
    const on = (i: number) => !lines.length || lines.includes(i + 1)
    tl.to(els, {
      opacity: i => (on(i) ? 1 : 0.3),
      backgroundColor: i => (on(i) && lines.length ? 'rgba(127, 212, 255, 0.07)' : 'rgba(127, 212, 255, 0)'),
      borderLeftColor: i => (on(i) && lines.length ? 'rgba(127, 212, 255, 0.9)' : 'rgba(127, 212, 255, 0)'),
      duration: 0.5,
    }, at)
  })
  return (
    <pre ref={ref} className={`code glass ${className}`} style={size ? { fontSize: size } : undefined} {...build(step)}>
      {code.replace(/^\n|\n\s*$/g, '').split('\n').map((l, i) => <span key={i} className="ln">{highlight(l) }{'\n'}</span>)}
    </pre>
  )
}
