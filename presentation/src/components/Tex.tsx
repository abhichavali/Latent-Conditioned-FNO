import { useMemo } from 'react'
import katex from 'katex'
import { build } from './layouts'

/** KaTeX math. `<Tex>{String.raw`\partial_t u = \nabla^2 u`}</Tex>`; `block` for display style. */
export function Tex({ children, block, step, className = '' }: { children: string; block?: boolean; step?: number; className?: string }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: !!block, throwOnError: false }), [children, block])
  const Tag = block ? 'div' : 'span'
  return <Tag className={`tex ${className}`} {...build(step)} dangerouslySetInnerHTML={{ __html: html }} />
}
