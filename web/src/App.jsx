import { useEffect, useState, useCallback, Fragment } from 'react'

const ROW = 3 // cards per row on desktop

const SearchIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" />
  </svg>
)
const SparkIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2c1.2 4.2 3.6 6.6 7.8 7.8-4.2 1.2-6.6 3.6-7.8 7.8-1.2-4.2-3.6-6.6-7.8-7.8C8.4 8.6 10.8 6.2 12 2z"/></svg>
)
const LayersIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg>
)
const SunIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
)
const MoonIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>
)

const COMPONENT_ORDER = ['global', 'region', 'coverage', 'style', 'context']

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`)
  return r.json()
}

function ScoreBars({ components }) {
  const rows = COMPONENT_ORDER.filter((k) => (components[k] ?? 0) > 0.001)
  if (rows.length === 0) rows.push('global')
  return (
    <div className="bars">
      {rows.map((k) => (
        <div className="bar-row" key={k}>
          <span className="n">{k}</span>
          <span className="v">{(components[k] || 0).toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

function ResultCard({ r, onSimilar, onZoom, index }) {
  return (
    <div className="card" style={{ animationDelay: `${index * 45}ms` }}>
      <div className="card-img" onClick={() => onZoom(r.image_url)}>
        <img src={r.image_url} alt={r.image_id} loading="lazy" />
        <span className="rank-badge">RANK {r.rank}</span>
        <span className="score-badge">{r.final_score.toFixed(2)}</span>
      </div>
      <div className="card-body">
        <div className="matched">
          {r.matched_clauses.map((m) => <span className="m" key={m}>✓ {m}</span>)}
        </div>
        <ScoreBars components={r.components} />
        <div className="card-actions">
          <button className="btn-ghost" onClick={() => onSimilar(r.image_id, index)}>
            <LayersIcon /> Find similar looks
          </button>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [examples, setExamples] = useState({ assignment: [], more: [] })
  const [config, setConfig] = useState(null)

  const [backbone, setBackbone] = useState('marqo')
  const [colorGate, setColorGate] = useState(true)
  const [showWeights, setShowWeights] = useState(false)
  const [weights, setWeights] = useState(null)

  const [similar, setSimilar] = useState(null)
  const [similarLoading, setSimilarLoading] = useState(false)
  const [lightbox, setLightbox] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    fetch('/api/examples').then((r) => r.json()).then(setExamples).catch(() => {})
    fetch('/api/config').then((r) => r.json()).then((c) => {
      setConfig(c)
      setBackbone(c.default_backbone)
      setWeights(c.default_weights)
    }).catch(() => {})
  }, [])

  const runSearch = useCallback(async (q) => {
    const text = (q ?? query).trim()
    if (!text) return
    setQuery(text)
    setLoading(true); setError(null); setSimilar(null)
    try {
      const body = { query: text, top_k: 6, backbone, color_gate: colorGate }
      if (showWeights && weights) body.weights = weights
      setData(await postJSON('/api/search', body))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [query, backbone, colorGate, showWeights, weights])

  const findSimilar = useCallback(async (imageId, index) => {
    setSimilarLoading(true); setSimilar({ image_id: imageId, results: [], sourceIndex: index })
    try {
      const res = await postJSON('/api/similar', { image_id: imageId, top_k: 6, backbone })
      setSimilar({ ...res, sourceIndex: index })
    } catch { setSimilar(null) }
    finally { setSimilarLoading(false) }
  }, [backbone])

  // re-run when ablation toggles change and we already have results
  useEffect(() => {
    if (data) runSearch(data.query)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backbone, colorGate])

  return (
    <>
      <div className="aurora" />
      <div className="shell">
        {/* top bar */}
        <div className="topbar">
          <div className="brand">
            <img src="/favicon.svg" alt="" />
            <span className="brand-name">Glance</span>
            <span className="brand-badge">Fashion Retrieval</span>
          </div>
          <div className="topbar-right">
            {config && <span className="backbone-note">Backbone · <b style={{ color: 'var(--text-dim)' }}>{backbone === 'marqo' ? 'Marqo-FashionSigLIP' : 'FashionCLIP'}</b></span>}
            <button className="theme-toggle" onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              aria-label="Toggle colour theme">
              {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
              {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
          </div>
        </div>

        {/* hero */}
        <div className="hero">
          <div className="eyebrow"><SparkIcon /> Multimodal · Zero-shot · Region-grounded</div>
          <h1>Search the look,<br /><span className="grad">not just the words.</span></h1>
          <p>
            Describe a garment, colour, style and setting in plain language. The engine
            grounds each constraint in garment-region evidence and scene context — then
            shows you exactly why every result matched.
          </p>

          <div className="search-wrap">
            <div className="search-bar">
              <SearchIcon />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && runSearch()}
                placeholder="e.g. a red tie and a white shirt in a formal setting"
              />
              <button className="btn-primary" onClick={() => runSearch()} disabled={loading}>
                {loading ? 'Searching…' : 'Discover'}
              </button>
            </div>

            {examples.assignment.length > 0 && (
              <>
                <div className="chips-label">Try a search</div>
                <div className="chips">
                  {examples.assignment.map((e) => (
                    <button className="chip" key={e.query} onClick={() => runSearch(e.query)}>{e.label}</button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        {/* controls */}
        <div className="controls">
          <div className="control-group">
            <span className="control-label">Backbone</span>
            <div className="seg">
              {(config?.backbones || ['marqo', 'fashionclip']).map((b) => (
                <button key={b} className={backbone === b ? 'active' : ''} onClick={() => setBackbone(b)}>
                  {b === 'marqo' ? 'Marqo-SigLIP' : 'FashionCLIP'}
                </button>
              ))}
            </div>
          </div>
          <div className="control-group">
            <span className="control-label">Colour verification</span>
            <div className="toggle" onClick={() => setColorGate((v) => !v)}>
              <span className={`switch ${colorGate ? 'on' : ''}`} />
              <span>{colorGate ? 'On' : 'Off'}</span>
            </div>
          </div>
          <button className="link-btn" onClick={() => setShowWeights((v) => !v)}>
            {showWeights ? 'Hide' : 'Tune'} fusion weights ▾
          </button>
        </div>

        {showWeights && weights && config && (
          <div className="weights">
            {config.components.filter((c) => c !== 'conjunction').map((c) => (
              <div className="weight-item" key={c}>
                <label><span style={{ textTransform: 'capitalize' }}>{c}</span> <b>{(weights[c] ?? 0).toFixed(2)}</b></label>
                <input type="range" min="0" max="0.6" step="0.05"
                  value={weights[c] ?? 0}
                  onChange={(e) => setWeights({ ...weights, [c]: parseFloat(e.target.value) })}
                  onMouseUp={() => data && runSearch(data.query)}
                  onTouchEnd={() => data && runSearch(data.query)}
                />
              </div>
            ))}
          </div>
        )}

        {/* states */}
        {loading && <div className="loading"><div className="spinner" />Grounding garment & scene evidence…</div>}
        {error && <div className="loading warn">⚠ {error}<br /><span style={{ fontSize: '0.85rem' }}>Is the API running on :8000?</span></div>}

        {/* results — the "similar" strip is injected right after the row of
            the card whose "Find similar" was clicked (spans the full width). */}
        {!loading && data && (() => {
          const src = similar?.sourceIndex ?? -1
          const insertAfter = src < 0 ? -1 : Math.min((Math.floor(src / ROW) + 1) * ROW - 1, data.results.length - 1)
          const panel = similar && (
            <div className="similar in-grid">
              <div className="similar-head">
                <h3>Visually similar looks <span style={{ color: 'var(--text-faint)', fontWeight: 400 }}>· image-to-image</span></h3>
                <button className="close" onClick={() => setSimilar(null)}>✕</button>
              </div>
              {similarLoading ? <div className="loading" style={{ padding: 20 }}><div className="spinner" /></div> : (
                <div className="similar-row">
                  {similar.results.map((s) => (
                    <div className="s-item" key={s.image_id} onClick={() => setLightbox(s.image_url)}>
                      <img src={s.image_url} alt={s.image_id} loading="lazy" />
                      <span className="sim">{s.similarity.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
          return (
            <div className="grid">
              {data.results.map((r, i) => (
                <Fragment key={r.image_id}>
                  <ResultCard r={r} index={i} onSimilar={findSimilar} onZoom={setLightbox} />
                  {i === insertAfter && panel}
                </Fragment>
              ))}
            </div>
          )
        })()}

        {!loading && !data && !error && (
          <div className="empty">Describe a look above — or try one of the searches — to see grounded, explainable results.</div>
        )}

        <div className="foot">
          <b>Fashion-Aware Context Retrieval</b> — Marqo-FashionSigLIP + region evidence + pixel-HSV colour verification.<br />
          Built for the Glance ML internship assignment · Not an official Glance product.
        </div>
      </div>

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <button className="lb-close">✕</button>
          <img src={lightbox} alt="" onClick={(e) => e.stopPropagation()} />
        </div>
      )}
    </>
  )
}
