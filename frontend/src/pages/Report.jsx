import { useState, useMemo, useRef, useEffect, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import styles from './Report.module.css'
import { TRIANGLES, ACCENTS } from '../components/polyMesh'
import SeverityChart from '../components/SeverityChart'

const PARTICLE_COUNT = 15

const SEVERITY_CONFIG = {
  stuttering:         { historyKey: 'stuttering',         label: 'Stuttering',         color: '#ef4444' },
  dysathria:          { historyKey: 'slurring',           label: 'Slurring',           color: '#f97316' },
  filler_words:       { historyKey: 'filler_words',       label: 'Filler Words',       color: '#eab308' },
  fast_speaking_rate: { historyKey: 'fast_speaking_rate', label: 'Fast Speaking Rate', color: '#a855f7' },
}

/** Compute the flagging threshold for a given problem from the current report. */
function getThreshold(problemKey, report) {
  switch (problemKey) {
    case 'stuttering': {
      const chunks = report.disorder_result?.total_chunks_analyzed || 0
      if (chunks <= 5) return 30
      if (chunks <= 20) return 20
      if (chunks <= 40) return 15
      return 10
    }
    case 'dysathria': // slurring — fixed 60 %
      return 60
    case 'filler_words':
      return report.bad_habit_result?.filler_detection?.threshold_pct ?? 25
    case 'fast_speaking_rate':
      return report.bad_habit_result?.pace_analysis?.threshold_pct ?? 30
    default:
      return null
  }
}

const EMOTION_PROBLEM_LABELS = {
  nervouseness:          'NERVOUSNESS',
  negative_emotion:      'NEGATIVE EMOTION',
  emotion_inconsistency: 'EMOTION INCONSISTENCY',
  monotone_voice:        'MONOTONE VOICE',
}

const BAD_HABIT_LABELS = {
  filler_words:       'FILLER WORDS',
  fast_speaking_rate: 'FAST SPEAKING RATE',
}

const EMOTION_COLORS = {
  neutral:   '#94a3b8',
  calm:      '#06b6d4',
  happy:     '#eab308',
  sad:       '#3b82f6',
  angry:     '#ef4444',
  fearful:   '#a855f7',
  disgust:   '#10b981',
  surprised: '#f97316',
}

export default function Report() {
  const { state } = useLocation()
  const navigate  = useNavigate()
  const report    = state?.report
  const audioUrl  = state?.audioUrl

  // Audio player state
  const audioRef = useRef(null)
  const progressRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  // Revoke blob URL on unmount to free memory (skip for data URLs)
  useEffect(() => {
    return () => {
      if (audioUrl && audioUrl.startsWith('blob:')) URL.revokeObjectURL(audioUrl)
    }
  }, [audioUrl])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTime = () => setCurrentTime(audio.currentTime)
    const onMeta = () => setDuration(audio.duration)
    const onEnd = () => setIsPlaying(false)
    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('loadedmetadata', onMeta)
    audio.addEventListener('ended', onEnd)
    return () => {
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('loadedmetadata', onMeta)
      audio.removeEventListener('ended', onEnd)
    }
  }, [audioUrl])

  const togglePlay = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (isPlaying) { audio.pause() } else { audio.play() }
    setIsPlaying(!isPlaying)
  }, [isPlaying])

  const handleSeek = useCallback((e) => {
    const audio = audioRef.current
    const bar = progressRef.current
    if (!audio || !bar) return
    const rect = bar.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    audio.currentTime = ratio * duration
  }, [duration])

  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  const particles = useMemo(() =>
    Array.from({ length: PARTICLE_COUNT }, () => ({
      left:              `${Math.random() * 100}%`,
      animationDuration: `${8 + Math.random() * 12}s`,
      animationDelay:    `${Math.random() * 10}s`,
      width:             `${1.5 + Math.random() * 2}px`,
      height:            `${1.5 + Math.random() * 2}px`,
      opacity:           0.2 + Math.random() * 0.4,
    })), [])

  const problems = useMemo(() => {
    if (!report) return []
    const list = []

    const stutter = report.disorder_result?.stuttering
    if (stutter?.flagged) {
      list.push({
        key:      'stuttering',
        label:    'STUTTERING',
        info:     `${stutter.severity_percentage}% of speech affected`,
        feedback: stutter.rag_feedback,
        severityData: {
          actual:    stutter.severity_percentage,
          threshold: getThreshold('stuttering', report),
          unit:      '% of speech affected',
        },
      })
    }

    const slur = report.disorder_result?.slurring
    if (slur?.flagged) {
      list.push({
        key:      'dysathria',
        label:    'SLURRING',
        info:     `${slur.dysarthric_ratio}% of speech affected`,
        feedback: slur.rag_feedback,
        severityData: {
          actual:    slur.dysarthric_ratio,
          threshold: getThreshold('dysathria', report),
          unit:      '% of speech affected',
        },
      })
    }

    const emotionFeedback = report.emotion_result?.rag_feedback || {}
    const dominant = report.emotion_result?.dominant_emotion
    const confidence = report.emotion_result?.confidence
    const top3 = report.emotion_result?.top_3_emotions || []
    const emotionInfo = dominant
      ? `Dominant emotion: ${dominant} (${confidence}%)`
      : null

    for (const [key, label] of Object.entries(EMOTION_PROBLEM_LABELS)) {
      if (emotionFeedback[key]) {
        list.push({
          key,
          label,
          info: emotionInfo,
          feedback: emotionFeedback[key],
          emotionTop3: top3.length > 0 ? top3 : null,
        })
      }
    }

    // Bad-habit problems
    const bh = report.bad_habit_result
    if (bh) {
      const filler = bh.filler_detection
      if (filler?.flagged) {
        list.push({
          key:      'filler_words',
          label:    BAD_HABIT_LABELS.filler_words,
          info:     `${filler.filler_rate_pct}% of speech affected`,
          feedback: filler.rag_feedback,
          severityData: {
            actual:    filler.filler_rate_pct,
            threshold: getThreshold('filler_words', report),
            unit:      '% of speech affected',
          },
        })
      }

      const pace = bh.pace_analysis
      if (pace?.flagged) {
        list.push({
          key:      'fast_speaking_rate',
          label:    BAD_HABIT_LABELS.fast_speaking_rate,
          info:     `${pace.instability_rate_pct}% of speech unstable`,
          feedback: pace.rag_feedback,
          severityData: {
            actual:    pace.instability_rate_pct,
            threshold: getThreshold('fast_speaking_rate', report),
            unit:      '% of speech unstable',
          },
        })
      }
    }

    return list
  }, [report])

  const [selectedIdx, setSelectedIdx] = useState(0)

  // ── Severity history for trend charts ──
  const [severityHistory, setSeverityHistory] = useState(null)

  useEffect(() => {
    if (!report?.user_id) return
    fetch(`http://localhost:5001/results/${report.user_id}/severity-history`)
      .then(r => r.json())
      .then(d => setSeverityHistory(d.history || []))
      .catch(() => {})
  }, [report?.user_id])

  const severityCharts = useMemo(() => {
    if (!severityHistory || !report) return []

    const reportTime = new Date(report.timestamp).getTime()
    const currentSid = String(report.session_id)

    // Only include sessions at or before the current report's time
    const filtered = severityHistory
      .filter(h => new Date(h.timestamp).getTime() <= reportTime)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())

    const charts = []

    for (const problem of problems) {
      const cfg = SEVERITY_CONFIG[problem.key]
      if (!cfg) continue // skip emotion problems — no numeric severity

      const data = filtered
        .filter(h => h.severity[cfg.historyKey] !== undefined)
        .map(h => ({
          session_id: String(h.session_id),
          timestamp:  h.timestamp,
          value:      h.severity[cfg.historyKey],
        }))

      // Need at least 4 *previous* data points (excluding current)
      const previousCount = data.filter(d => d.session_id !== currentSid).length
      if (previousCount >= 4) {
        const threshold = getThreshold(problem.key, report)
        charts.push({ key: problem.key, label: cfg.label, color: cfg.color, data, threshold })
      }
    }

    return charts
  }, [severityHistory, report, problems])

  // Build timeline highlights from chunk-level data
  const timelineHighlights = useMemo(() => {
    if (!report) return []
    const highlights = []
    const DISORDER_CHUNK_SEC = 3
    const BAD_HABIT_CHUNK_SEC = 5

    const HIGHLIGHT_COLORS = {
      stuttering:         'rgba(239, 68, 68, 0.7)',
      slurring:           'rgba(249, 115, 22, 0.7)',
      filler_words:       'rgba(234, 179, 8, 0.7)',
      fast_speaking_rate: 'rgba(168, 85, 247, 0.7)',
    }

    // Disorder: stuttering flagged chunks (3s each)
    const stutter = report.disorder_result?.stuttering
    if (stutter?.flagged && stutter.flagged_chunks) {
      for (const idx of stutter.flagged_chunks) {
        highlights.push({
          startSec: idx * DISORDER_CHUNK_SEC,
          endSec:   (idx + 1) * DISORDER_CHUNK_SEC,
          color:    HIGHLIGHT_COLORS.stuttering,
          label:    'Stuttering',
        })
      }
    }

    // Disorder: slurring flagged chunks (3s each)
    const slur = report.disorder_result?.slurring
    if (slur?.flagged && slur.flagged_chunks) {
      for (const idx of slur.flagged_chunks) {
        highlights.push({
          startSec: idx * DISORDER_CHUNK_SEC,
          endSec:   (idx + 1) * DISORDER_CHUNK_SEC,
          color:    HIGHLIGHT_COLORS.slurring,
          label:    'Slurring',
        })
      }
    }

    // Bad habit: filler chunks (5s each, have time_start_s / time_end_s)
    const bh = report.bad_habit_result
    if (bh?.filler_detection?.flagged && bh.filler_detection.chunks) {
      for (const chunk of bh.filler_detection.chunks) {
        if (chunk.label === 'Filler') {
          highlights.push({
            startSec: chunk.time_start_s,
            endSec:   chunk.time_end_s,
            color:    HIGHLIGHT_COLORS.filler_words,
            label:    'Filler Words',
          })
        }
      }
    }

    // Bad habit: pace unstable chunks (5s each)
    if (bh?.pace_analysis?.flagged && bh.pace_analysis.chunks) {
      for (const chunk of bh.pace_analysis.chunks) {
        if (chunk.label?.toLowerCase() === 'unstable') {
          highlights.push({
            startSec: chunk.time_start_s,
            endSec:   chunk.time_end_s,
            color:    HIGHLIGHT_COLORS.fast_speaking_rate,
            label:    'Fast Pace',
          })
        }
      }
    }

    return highlights
  }, [report])

  /** Split markdown feedback into section groups for separate boxes */
  const parseFeedbackSections = (feedback) => {
    if (!feedback) return null

    // Split by h3 headers (### ...)
    const sectionRegex = /###\s+(.+)/g
    const parts = []
    let lastIndex = 0
    let match

    const matches = []
    while ((match = sectionRegex.exec(feedback)) !== null) {
      matches.push({ title: match[1].trim(), index: match.index })
    }

    for (let i = 0; i < matches.length; i++) {
      const start = matches[i].index + feedback.slice(matches[i].index).indexOf('\n') + 1
      const end = i + 1 < matches.length ? matches[i + 1].index : feedback.length
      let body = feedback.slice(start, end).replace(/^---\s*$/gm, '').trim()
      parts.push({ title: matches[i].title, body })
    }

    // Group: "What This Means" + "Why This May Be Happening" → box 1
    const meaningSection = parts.find(p => /what this means/i.test(p.title))
    const whySection = parts.find(p => /why this may be happening/i.test(p.title))
    const stepsSection = parts.find(p => /actionable improvement/i.test(p.title))
    const practiceSection = parts.find(p => /quick practice/i.test(p.title))
    const noteSection = parts.find(p => /quick note/i.test(p.title))

    return { meaningSection, whySection, stepsSection, practiceSection, noteSection }
  }

  const background = (
    <>
      <svg
        className={styles.polyBg}
        viewBox="0 0 1280 1440"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="reportBg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"   stopColor="#1a3a70" />
            <stop offset="50%"  stopColor="#0e2248" />
            <stop offset="100%" stopColor="#060e20" />
          </linearGradient>
        </defs>
        <rect width="1280" height="1440" fill="url(#reportBg)" />
        {TRIANGLES.map((t, i) => (
          <polygon key={i} points={t.pts} fill={t.fill} stroke="rgba(255,255,255,0.04)" strokeWidth="0.5"
            style={{ animationDelay: `${t.delay.toFixed(2)}s` }} />
        ))}
        {ACCENTS.map((t, i) => (
          <polygon key={`a${i}`} points={t.pts} fill={t.fill} stroke="none" />
        ))}
      </svg>
      <div className={`${styles.glowOrb} ${styles.glowOrb1}`} />
      <div className={`${styles.glowOrb} ${styles.glowOrb2}`} />
      <div className={styles.particles}>
        {particles.map((p, i) => (
          <div key={i} className={styles.particle} style={p} />
        ))}
      </div>
    </>
  )

  if (!report || problems.length === 0) {
    return (
      <div className={styles.page}>
        {background}
        <div className={styles.noData}>
          <img src="/LOGO.png" alt="Vocal Insight" style={{ width: '280px', maxWidth: '70vw', height: 'auto' }} />
          <p>No problems were detected in your speech.</p>
          <button className={styles.noDataBtn} onClick={() => navigate('/record', { state: { clearAudio: true } })}>
            RECORD AGAIN
          </button>
        </div>
      </div>
    )
  }

  const selected = problems[selectedIdx]

  return (
    <div className={styles.page}>
      {background}

      <div className={styles.content}>
        {/* Header */}
        <div className={styles.header}>
          <img src="/LOGO.png" alt="Vocal Insight" className={styles.logo} />
          <button className={styles.backBtn} onClick={() => navigate('/record', { state: { clearAudio: true } })}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            RECORD AGAIN
          </button>
        </div>

        {/* Title */}
        <div className={styles.titleArea}>
          <h1 className={styles.title}>YOUR FLUENCY REPORT</h1>
          <p className={styles.subtitle}>{problems.length} {problems.length === 1 ? 'issue' : 'issues'} detected</p>
        </div>

        {/* Problem tabs */}
        <div className={styles.tabs}>
          {problems.map((p, i) => {
            const cfg = SEVERITY_CONFIG[p.key]
            const color = cfg?.color || '#38bdf8'
            const sd = p.severityData
            return (
              <button
                key={p.key}
                className={`${styles.tab} ${i === selectedIdx ? styles.tabActive : ''}`}
                onClick={() => setSelectedIdx(i)}
              >
                <span className={styles.tabLabel}>Problem {i + 1}</span>
                <span className={styles.tabProblem}>{p.label}</span>
                {p.emotionTop3 ? (() => {
                  const total = p.emotionTop3.reduce((s, e) => s + e.confidence, 0) || 1
                  return (
                    <div className={styles.tabEmotionWrap}>
                      <div className={styles.tabEmotionLabels}>
                        {p.emotionTop3.map((e) => {
                          const pct = (e.confidence / total) * 100
                          return (
                            <span
                              key={e.emotion}
                              className={styles.tabEmotionLabelItem}
                              style={{ width: `${pct}%` }}
                            >
                              {e.emotion}
                            </span>
                          )
                        })}
                      </div>
                      <div className={styles.tabEmotionBar}>
                        {p.emotionTop3.map((e) => {
                          const ec = EMOTION_COLORS[e.emotion] || '#a855f7'
                          const pct = (e.confidence / total) * 100
                          return (
                            <div
                              key={e.emotion}
                              className={styles.tabEmotionSegment}
                              style={{ width: `${pct}%`, background: ec }}
                              title={`${e.emotion} ${e.confidence}%`}
                            />
                          )
                        })}
                      </div>
                    </div>
                  )
                })() : sd && sd.threshold != null ? (
                  <div className={styles.tabGauge}>
                    <div className={styles.tabGaugeTrack}>
                      <div
                        className={styles.tabGaugeFill}
                        style={{
                          width: `${Math.min(100, sd.actual)}%`,
                          background: color,
                          boxShadow: `0 0 10px ${color}80`,
                        }}
                      />
                      <div
                        className={styles.tabGaugeMarker}
                        style={{ left: `${Math.min(100, sd.threshold)}%` }}
                        title={`Threshold ${sd.threshold}%`}
                      />
                    </div>
                    <div className={styles.tabGaugeLabel}>
                      <span className={styles.tabGaugeActual} style={{ color }}>
                        {sd.actual}%
                      </span>
                      <span className={styles.tabGaugeThresh}>
                        threshold {sd.threshold}%
                      </span>
                    </div>
                  </div>
                ) : (
                  p.info && <span className={styles.tabInfo}>{p.info}</span>
                )}
              </button>
            )
          })}
        </div>

        {/* Feedback sections */}
        {(() => {
          const sections = parseFeedbackSections(selected.feedback)
          if (!sections) {
            return (
              <div className={styles.feedbackCard} key={selectedIdx}>
                <div className={styles.markdownBody}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {'*No feedback available for this problem.*'}
                  </ReactMarkdown>
                </div>
              </div>
            )
          }

          return (
            <div className={styles.sectionGrid} key={selectedIdx}>
              {/* Box 1: What This Means + Why This May Happen */}
              {(sections.meaningSection || sections.whySection) && (
                <div className={`${styles.feedbackCard} ${styles.insightBox}`}>
                  <div className={styles.markdownBody}>
                    {sections.meaningSection && (
                      <>
                        <div className={styles.boxHeader}>
                          <div className={styles.boxIcon}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <circle cx="12" cy="12" r="10" />
                              <line x1="12" y1="16" x2="12" y2="12" />
                              <line x1="12" y1="8" x2="12.01" y2="8" />
                            </svg>
                          </div>
                          <h3>What This Means</h3>
                        </div>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {sections.meaningSection.body}
                        </ReactMarkdown>
                      </>
                    )}
                    {sections.whySection && (
                      <>
                        <div className={styles.sectionDivider} />
                        <div className={styles.boxHeader}>
                          <div className={styles.boxIcon}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <circle cx="12" cy="12" r="10" />
                              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                              <line x1="12" y1="17" x2="12.01" y2="17" />
                            </svg>
                          </div>
                          <h3>Why This May Be Happening</h3>
                        </div>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {sections.whySection.body}
                        </ReactMarkdown>
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Box 2: Actionable Improvement Steps */}
              {sections.stepsSection && (
                <div className={`${styles.feedbackCard} ${styles.stepsBox}`}>
                  <div className={styles.markdownBody}>
                    <div className={styles.boxHeader}>
                      <div className={styles.boxIcon}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="9 11 12 14 22 4" />
                          <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
                        </svg>
                      </div>
                      <h3>Actionable Improvement Steps</h3>
                    </div>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {sections.stepsSection.body}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {/* Box 3: Quick Practice Exercise */}
              {sections.practiceSection && (
                <div className={`${styles.feedbackCard} ${styles.practiceBox}`}>
                  <div className={styles.markdownBody}>
                    <div className={styles.boxHeader}>
                      <div className={styles.boxIcon}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polygon points="5 3 19 12 5 21 5 3" />
                        </svg>
                      </div>
                      <h3>Quick Practice Exercise</h3>
                    </div>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {sections.practiceSection.body}
                    </ReactMarkdown>
                  </div>
                </div>
              )}

              {/* Box 4: Quick Note */}
              {sections.noteSection && (
                <div className={`${styles.feedbackCard} ${styles.noteBox}`}>
                  <div className={styles.markdownBody}>
                    <div className={styles.boxHeader}>
                      <div className={styles.boxIcon}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                        </svg>
                      </div>
                      <h3>Quick Note</h3>
                    </div>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {sections.noteSection.body}
                    </ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          )
        })()}

        {/* Audio Player */}
        {audioUrl && (
          <>
            <div className={styles.playerTitleArea}>
              <h2 className={styles.playerTitle}>YOUR RECORDING</h2>
              {timelineHighlights.length > 0 && (
                <div className={styles.legend}>
                  {[...new Map(timelineHighlights.map(h => [h.label, h.color])).entries()].map(([label, color]) => (
                    <div key={label} className={styles.legendItem}>
                      <span className={styles.legendDot} style={{ background: color }} />
                      <span>{label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className={styles.audioPlayer}>
            <audio ref={audioRef} src={audioUrl} preload="metadata" />

            <div className={styles.playerRow}>
              <button className={styles.playBtn} onClick={togglePlay}>
                {isPlaying ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16" rx="1" />
                    <rect x="14" y="4" width="4" height="16" rx="1" />
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="6,4 20,12 6,20" />
                  </svg>
                )}
              </button>

              <div className={styles.timelineWrap}>
                {/* Highlight rows — one per problem type, stacked */}
                {duration > 0 && timelineHighlights.length > 0 && (
                  [...new Map(timelineHighlights.map(h => [h.label, h.color])).keys()].map(label => (
                    <div key={label} className={styles.highlightBar}>
                      {timelineHighlights.filter(h => h.label === label).map((h, i) => (
                        <div
                          key={i}
                          className={styles.highlight}
                          style={{
                            left:  `${(h.startSec / duration) * 100}%`,
                            width: `${((h.endSec - h.startSec) / duration) * 100}%`,
                            background: h.color,
                          }}
                          title={`${h.label} (${h.startSec}s–${h.endSec}s)`}
                        />
                      ))}
                    </div>
                  ))
                )}

                {/* Playback timeline */}
                <div
                  className={styles.progressBar}
                  ref={progressRef}
                  onClick={handleSeek}
                >
                  <div
                    className={styles.progressFill}
                    style={{ width: duration ? `${(currentTime / duration) * 100}%` : '0%' }}
                  />
                </div>
              </div>
            </div>

            <div className={styles.playerTimes}>
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration || 0)}</span>
            </div>
            </div>
          </>
        )}

        {/* Emotion Breakdown */}
        {report.emotion_result?.top_3_emotions?.length > 0 && (
          <>
            <div className={styles.emotionTitleArea}>
              <h2 className={styles.emotionTitle}>EMOTION BREAKDOWN</h2>
              <p className={styles.subtitle}>How your voice sounded across the recording</p>
            </div>
            <div className={styles.emotionCard}>
              <div className={styles.emotionHeader}>
                <div
                  className={styles.dominantBadge}
                  style={{
                    background: `${EMOTION_COLORS[report.emotion_result.dominant_emotion] || '#a855f7'}1F`,
                    borderColor: `${EMOTION_COLORS[report.emotion_result.dominant_emotion] || '#a855f7'}55`,
                  }}
                >
                  <span className={styles.dominantLabel}>Dominant</span>
                  <span className={styles.dominantEmotion}>
                    {report.emotion_result.dominant_emotion}
                  </span>
                  <span
                    className={styles.dominantConf}
                    style={{ color: EMOTION_COLORS[report.emotion_result.dominant_emotion] || '#c4b5fd' }}
                  >
                    {report.emotion_result.confidence}%
                  </span>
                </div>
                {(() => {
                  const analyzed = report.emotion_result.frame_count || 0
                  const skipped  = report.emotion_result.skipped_frames || 0
                  const total    = analyzed + skipped
                  if (total === 0 || skipped === 0) return null
                  const pct = Math.round((analyzed / total) * 100)
                  return (
                    <div className={styles.frameStats}>
                      {pct}% of audio confidently analyzed
                    </div>
                  )
                })()}
              </div>
              <div className={styles.emotionBars}>
                {report.emotion_result.top_3_emotions.map((e) => {
                  const color = EMOTION_COLORS[e.emotion] || '#a855f7'
                  return (
                    <div key={e.emotion} className={styles.emotionBarRow}>
                      <span className={styles.emotionName}>{e.emotion}</span>
                      <div className={styles.emotionBarTrack}>
                        <div
                          className={styles.emotionBarFill}
                          style={{
                            width: `${e.confidence}%`,
                            background: color,
                            boxShadow: `0 0 12px ${color}66`,
                          }}
                        />
                      </div>
                      <span className={styles.emotionPct}>{e.confidence}%</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </>
        )}

        {/* Severity Charts */}
        {severityCharts.length > 0 && (
          <>
            <div className={styles.severityTitleArea}>
              <h2 className={styles.severityTitle}>SEVERITY CHARTS</h2>
              <p className={styles.subtitle}>Tracking detected issues over your previous recordings</p>
            </div>
            <div className={styles.severityGrid}>
              {severityCharts.map(chart => (
                <div key={chart.key} className={styles.severityCard}>
                  <div className={styles.boxHeader}>
                    <div className={styles.boxIcon}
                      style={{ background: `${chart.color}1A`, color: chart.color }}>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                      </svg>
                    </div>
                    <h3 style={{ color: chart.color, margin: 0, fontWeight: 800,
                      fontSize: '0.9rem', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                      {chart.label}
                    </h3>
                  </div>
                  <SeverityChart
                    data={chart.data}
                    currentSessionId={report.session_id}
                    color={chart.color}
                    threshold={chart.threshold}
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
