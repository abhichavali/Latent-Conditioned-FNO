import { Component, StrictMode, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/newsreader/opsz.css'
import '@fontsource/ibm-plex-sans/300.css'
import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/500.css'
import '@fontsource/ibm-plex-mono/400.css'
import 'katex/dist/katex.min.css'
import './theme.css'
import './deck/motion' // registers GSAP plugins
import { Deck } from './deck/Deck'
import type { SlideDef } from './deck/SlideFrame'

// Every file in src/slides/ is a slide, ordered by file name. Adding a slide = adding a file.
const modules = import.meta.glob<{ default: SlideDef }>('./slides/*.tsx', { eager: true })
const slides = Object.keys(modules).sort().map(k => modules[k].default)

/** Shows render errors on screen instead of a blank page. */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div style={{ padding: 48, font: '16px/1.5 ui-monospace, Menlo, monospace', color: '#ff8a80', overflow: 'auto', height: '100%' }}>
        <h1 style={{ font: '500 22px system-ui', color: '#e9eef5', marginBottom: 16 }}>The deck hit an error</h1>
        <pre style={{ whiteSpace: 'pre-wrap' }}>{error.message}{'\n\n'}{error.stack}</pre>
        <button onClick={() => this.setState({ error: null })} style={{ marginTop: 16, padding: '8px 16px' }}>Try again</button>
      </div>
    )
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      {slides.length ? <Deck slides={slides} /> : <p style={{ padding: 48 }}>No slides yet: add a file to src/slides/.</p>}
    </ErrorBoundary>
  </StrictMode>,
)
