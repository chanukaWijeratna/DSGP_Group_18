import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import styles from './Record.module.css'
import { TRIANGLES, ACCENTS } from '../components/polyMesh'

const BAR_HEIGHTS = [0.45, 0.7, 0.9, 0.6, 1, 0.75, 0.55, 0.85, 0.65, 0.5, 0.8, 0.95, 0.6, 0.7, 0.4]
const PARTICLE_COUNT = 15
const WAVEFORM_BARS = 80
const TARGET_SAMPLE_RATE = 16000

const STAGES = [
  { id: 1, lines: ['ANALYZING SPEECH', 'QUALITY'] },
  { id: 2, lines: ['DETECTING PROBLEMS'] },
  { id: 3, lines: ['GENERATING REPORT'] },
]

function IconWaveform({ dark }) {
  const fill = dark ? '#0f2247' : 'currentColor'
  return (
    <svg width="30" height="30" viewBox="0 0 30 30" fill={fill}>
      <rect x="1"  y="11" width="4" height="8"  rx="2" />
      <rect x="7"  y="7"  width="4" height="16" rx="2" />
      <rect x="13" y="3"  width="4" height="24" rx="2" />
      <rect x="19" y="7"  width="4" height="16" rx="2" />
      <rect x="25" y="11" width="4" height="8"  rx="2" />
    </svg>
  )
}

function IconGears({ dark }) {
  const fill = dark ? '#0f2247' : 'currentColor'
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill={fill}>
      <path d="M19.14 12.94c.04-.3.06-.61.06-.94s-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
    </svg>
  )
}

function IconReport({ dark }) {
  const fill = dark ? '#0f2247' : 'currentColor'
  return (
    <svg width="30" height="30" viewBox="0 0 24 24" fill={fill}>
      <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 7V3.5L18.5 9H13zM8 13h8v1.5H8V13zm0 3h8v1.5H8V16zm0-6h4v1.5H8V10z"/>
    </svg>
  )
}

const STAGE_ICONS = [IconWaveform, IconGears, IconReport]

// ── WAV helpers ──

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
}

function float32ToWavBlob(samples, sampleRate) {
  const numChannels = 1
  const bytesPerSample = 2
  const dataLength = samples.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataLength)
  const view = new DataView(buffer)

  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataLength, true)
  writeString(view, 8, 'WAVE')
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true)
  view.setUint16(32, numChannels * bytesPerSample, true)
  view.setUint16(34, 16, true)
  writeString(view, 36, 'data')
  view.setUint32(40, dataLength, true)

  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

function computeWaveformPeaks(samples) {
  const blockSize = Math.floor(samples.length / WAVEFORM_BARS)
  if (blockSize === 0) return null
  const peaks = []
  for (let i = 0; i < WAVEFORM_BARS; i++) {
    let sum = 0
    for (let j = 0; j < blockSize; j++) sum += Math.abs(samples[i * blockSize + j])
    peaks.push(sum / blockSize)
  }
  const max = Math.max(...peaks, 0.01)
  return peaks.map(p => p / max)
}

// ── Quality check (runs on submit) ──

async function checkAudioQuality(blob) {
  const arrayBuffer = await blob.arrayBuffer()
  const audioCtx = new AudioContext()
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0))
  audioCtx.close()

  const duration = audioBuffer.duration
  const data = audioBuffer.getChannelData(0)

  if (duration < 3) return { ok: false, reason: 'Audio is too short. Please provide at least 3 seconds of speech.' }
  if (duration > 180) return { ok: false, reason: 'Audio is too long. Please keep it under 3 minutes.' }

  let totalSum = 0
  for (let i = 0; i < data.length; i++) totalSum += data[i] * data[i]
  const overallRms = Math.sqrt(totalSum / data.length)
  if (overallRms < 0.005) return { ok: false, reason: 'Audio appears to be silent. Please provide a recording with audible speech.' }

  let clippedSamples = 0
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i]) >= 0.99) clippedSamples++
  }
  if (clippedSamples / data.length > 0.01) return { ok: false, reason: 'Audio is distorted (clipping detected). Please lower the volume and re-record.' }

  const frameSize = Math.floor(audioBuffer.sampleRate * 0.025)
  const frameCount = Math.floor(data.length / frameSize)
  const energies = []
  for (let i = 0; i < frameCount; i++) {
    let sum = 0
    for (let j = 0; j < frameSize; j++) {
      const s = data[i * frameSize + j]
      sum += s * s
    }
    energies.push(Math.sqrt(sum / frameSize))
  }
  energies.sort((a, b) => a - b)

  const noiseEnd = Math.floor(energies.length * 0.2)
  const speechStart = Math.floor(energies.length * 0.7)
  let noiseAvg = 0
  for (let i = 0; i < noiseEnd; i++) noiseAvg += energies[i]
  noiseAvg /= noiseEnd || 1
  let speechAvg = 0
  for (let i = speechStart; i < energies.length; i++) speechAvg += energies[i]
  speechAvg /= (energies.length - speechStart) || 1

  if (noiseAvg > 0) {
    const snr = 20 * Math.log10(speechAvg / noiseAvg)
    if (snr < 10) return { ok: false, reason: 'Audio is too noisy. Please record in a quieter environment.' }
  }

  return { ok: true }
}

// ══════════════════════════════════════
// Component
// ══════════════════════════════════════

export default function Record() {
  const navigate = useNavigate()
  const { state: navState } = useLocation()

  // Recording state
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [scenario, setScenario] = useState('')
  const [processingStage, setProcessingStage] = useState(0)

  // Audio state — one source of truth
  const [audioBlob, setAudioBlob] = useState(null)       // WAV blob (from recording or upload-conversion)
  const [audioLabel, setAudioLabel] = useState('')        // label shown to user
  const [waveformPeaks, setWaveformPeaks] = useState(null)

  // Playback state
  const audioRef = useRef(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackTime, setPlaybackTime] = useState(0)
  const [audioDuration, setAudioDuration] = useState(0)

  // Recording internals
  const audioCtxRef = useRef(null)
  const sourceRef = useRef(null)
  const processorRef = useRef(null)
  const samplesRef = useRef([])
  const streamRef = useRef(null)
  const timerRef = useRef(null)

  // UI refs
  const fileInputRef = useRef(null)
  const wrapperRef = useRef(null)
  const historyGridRef = useRef(null)

  // History
  const user = useMemo(() => {
    try { return JSON.parse(localStorage.getItem('vi_user')) } catch { return null }
  }, [])
  const [history, setHistory] = useState([])

  const PROBLEM_DISPLAY = {
    stuttering:            'Stuttering',
    slurring:              'Slurring',
    nervouseness:          'Nervousness',
    negative_emotion:      'Negative Emotion',
    emotion_inconsistency: 'Emotion Inconsistency',
    monotone_voice:        'Monotone Voice',
    filler_words:          'Filler Words',
    fast_speaking_rate:    'Fast Speaking Rate',
  }

  const particles = useMemo(() =>
    Array.from({ length: PARTICLE_COUNT }, () => ({
      left: `${Math.random() * 100}%`,
      animationDuration: `${8 + Math.random() * 12}s`,
      animationDelay: `${Math.random() * 10}s`,
      width: `${1.5 + Math.random() * 2}px`,
      height: `${1.5 + Math.random() * 2}px`,
      opacity: 0.2 + Math.random() * 0.4,
    })), [])

  // ── Cleanup ──
  useEffect(() => {
    return () => clearInterval(timerRef.current)
  }, [])

  // ── Load history ──
  useEffect(() => {
    const userId = String(user?.id ?? 'guest')
    fetch(`http://localhost:5001/results/${userId}`)
      .then(r => r.json())
      .then(data => setHistory(data.results || []))
      .catch(() => {})
  }, [user])

  // ── Animate history grid on scroll into view ──
  useEffect(() => {
    const grid = historyGridRef.current
    if (!grid) return
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { grid.dataset.visible = 'true'; observer.disconnect() } },
      { threshold: 0.1 }
    )
    observer.observe(grid)
    return () => observer.disconnect()
  }, [history])

  // ── Object URL lifecycle ──
  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob)
      setAudioUrl(url)
      setIsPlaying(false)
      setPlaybackTime(0)
      setAudioDuration(0)
      return () => URL.revokeObjectURL(url)
    } else {
      setAudioUrl(null)
      setWaveformPeaks(null)
    }
  }, [audioBlob])

  // ── Audio element event binding ──
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTime = () => setPlaybackTime(audio.currentTime)
    const onMeta = () => setAudioDuration(audio.duration)
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

  // ── Helpers ──
  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  const formatPlaybackTime = (t) => {
    if (!t || isNaN(t)) return '0:00'
    const m = Math.floor(t / 60)
    const s = Math.floor(t % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  const formatHistoryDate = (iso) => new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })
  const formatHistoryTime = (iso) => new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })

  // ══════════════════════════════════════
  // RECORDING — direct PCM capture to WAV
  // ══════════════════════════════════════

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Clear previous audio
      setAudioBlob(null)
      setAudioLabel('')
      setWaveformPeaks(null)
      samplesRef.current = []

      const audioCtx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
      audioCtxRef.current = audioCtx
      const source = audioCtx.createMediaStreamSource(stream)
      sourceRef.current = source

      // ScriptProcessorNode to capture raw PCM
      const processor = audioCtx.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor
      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0)
        samplesRef.current.push(new Float32Array(input))
      }
      source.connect(processor)
      processor.connect(audioCtx.destination)

      setRecording(true)
      setSeconds(0)
      timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)
    } catch {
      alert('Microphone access denied.')
    }
  }, [])

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current)
    setRecording(false)

    // Disconnect audio nodes
    processorRef.current?.disconnect()
    sourceRef.current?.disconnect()
    audioCtxRef.current?.close()
    streamRef.current?.getTracks().forEach(t => t.stop())

    // Merge all captured chunks into a single Float32Array
    const chunks = samplesRef.current
    const totalLength = chunks.reduce((sum, c) => sum + c.length, 0)
    const merged = new Float32Array(totalLength)
    let offset = 0
    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }
    samplesRef.current = []

    // Build WAV blob and waveform
    const wavBlob = float32ToWavBlob(merged, TARGET_SAMPLE_RATE)
    const peaks = computeWaveformPeaks(merged)

    setWaveformPeaks(peaks)
    setAudioLabel('RECORDING READY')
    setAudioBlob(wavBlob)
  }, [])

  const toggleRecording = () => {
    if (recording) stopRecording()
    else startRecording()
  }

  // ══════════════════════════════════════
  // UPLOAD — decode any format, show waveform
  // ══════════════════════════════════════

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setAudioLabel(file.name.toUpperCase().slice(0, 28))

    try {
      // Decode to get raw PCM for waveform
      const arrayBuffer = await file.arrayBuffer()
      const audioCtx = new AudioContext()
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0))
      audioCtx.close()

      const rawData = audioBuffer.getChannelData(0)
      setWaveformPeaks(computeWaveformPeaks(rawData))

      // Convert to WAV so backend always gets WAV
      const wavBlob = float32ToWavBlob(rawData, audioBuffer.sampleRate)
      setAudioBlob(wavBlob)
    } catch {
      // Fallback: send raw file, skip waveform
      setWaveformPeaks(null)
      setAudioBlob(file)
    }

    // Reset file input so same file can be re-selected
    e.target.value = ''
  }

  // ══════════════════════════════════════
  // PLAYBACK
  // ══════════════════════════════════════

  const togglePlayback = () => {
    const audio = audioRef.current
    if (!audio) return
    if (isPlaying) { audio.pause(); setIsPlaying(false) }
    else { audio.play(); setIsPlaying(true) }
  }

  const handleSeek = (e) => {
    const audio = audioRef.current
    if (!audio || !audioDuration) return
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    audio.currentTime = ratio * audioDuration
    setPlaybackTime(audio.currentTime)
  }

  // ══════════════════════════════════════
  // SUBMIT
  // ══════════════════════════════════════

  const handleSubmit = async () => {
    if (!audioBlob) {
      alert('Please record or upload an audio file first.')
      return
    }

    setProcessingStage(1)

    try {
      const qualityResult = await checkAudioQuality(audioBlob)
      if (!qualityResult.ok) {
        setProcessingStage(0)
        alert(qualityResult.reason)
        return
      }

      setProcessingStage(2)

      const fileName = 'recording.wav'
      const emotionForm = new FormData()
      emotionForm.append('file', audioBlob, fileName)
      const disorderForm = new FormData()
      disorderForm.append('audio', audioBlob, fileName)
      const badHabitForm = new FormData()
      badHabitForm.append('file', audioBlob, fileName)

      const [emotionRes, disorderRes, badHabitRes] = await Promise.all([
        fetch('http://localhost:8000/predict', { method: 'POST', body: emotionForm }),
        fetch('http://localhost:5000/api/analyze/disorder', { method: 'POST', body: disorderForm }),
        fetch('http://localhost:8001/analyze', { method: 'POST', body: badHabitForm }),
      ])

      if (!emotionRes.ok) {
        const err = await emotionRes.json()
        throw new Error(`Emotion model error: ${err.detail || 'Server error'}`)
      }
      if (!disorderRes.ok) {
        const err = await disorderRes.json()
        throw new Error(`Disorder model error: ${err.error || 'Server error'}`)
      }
      if (!badHabitRes.ok) {
        const err = await badHabitRes.json()
        throw new Error(`Bad habit model error: ${err.detail || 'Server error'}`)
      }

      const emotionJson = await emotionRes.json()
      const disorderJson = await disorderRes.json()
      const badHabitJson = await badHabitRes.json()

      setProcessingStage(3)

      const sessionId = `${Date.now()}`
      const userId = String(user?.id ?? 'guest')

      const feedbackRes = await fetch('http://localhost:5001/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id:          userId,
          session_id:       sessionId,
          scenario:         scenario,
          emotion_result:   emotionJson,
          disorder_result:  disorderJson.data,
          bad_habit_result: badHabitJson,
        }),
      })

      if (!feedbackRes.ok) {
        const err = await feedbackRes.json()
        const detail = Array.isArray(err.detail)
          ? err.detail.map(e => `${e.loc?.join('.')} — ${e.msg}`).join('; ')
          : (err.detail || 'Server error')
        throw new Error(`Feedback error: ${detail}`)
      }

      const feedbackJson = await feedbackRes.json()

      // Convert blob to a data URL so it survives navigation (blob URLs get revoked on unmount)
      const audioDataUrl = await new Promise((resolve) => {
        const reader = new FileReader()
        reader.onloadend = () => resolve(reader.result)
        reader.readAsDataURL(audioBlob)
      })

      navigate('/report', { state: { report: feedbackJson, audioUrl: audioDataUrl } })
    } catch (err) {
      console.error('[Submit] Error:', err)
      setProcessingStage(0)
      alert('Could not process the audio file.')
    }
  }

  const loadHistoricalResult = async (sessionId) => {
    const userId = String(user?.id ?? 'guest')
    try {
      const res = await fetch(`http://localhost:5001/results/${userId}/${sessionId}`)
      if (!res.ok) throw new Error('Not found')
      navigate('/report', { state: { report: await res.json() } })
    } catch (err) {
      console.error('Failed to load result:', err)
    }
  }

  const deleteResult = async (sessionId) => {
    const userId = String(user?.id ?? 'guest')
    try {
      const res = await fetch(`http://localhost:5001/results/${userId}/${sessionId}`, {
        method: 'DELETE',
      })
      if (!res.ok) throw new Error('Delete failed')
      setHistory(prev => prev.filter(h => h.session_id !== sessionId))
    } catch (err) {
      console.error('Failed to delete result:', err)
      alert('Failed to delete result.')
    }
  }

  // Clear audio when returning from report page
  useEffect(() => {
    if (navState?.clearAudio) {
      setAudioBlob(null)
      setAudioLabel('')
      setWaveformPeaks(null)
      setIsPlaying(false)
      setPlaybackTime(0)
      setAudioDuration(0)
      // Clear the navigation state so it doesn't re-trigger
      window.history.replaceState({}, '')
    }
  }, [navState])

  const clearAudio = () => {
    setAudioBlob(null)
    setAudioLabel('')
    setWaveformPeaks(null)
    setIsPlaying(false)
    setPlaybackTime(0)
    setAudioDuration(0)
  }

  // ══════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════

  const background = (
    <>
      <svg
        className={styles.polyBg}
        viewBox="0 0 1280 1440"
        preserveAspectRatio="xMidYMin slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="baseBg2" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"   stopColor="#1a3a70" />
            <stop offset="50%"  stopColor="#0e2248" />
            <stop offset="100%" stopColor="#060e20" />
          </linearGradient>
        </defs>
        <rect width="1280" height="1440" fill="url(#baseBg2)" />
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

  // ── Processing screen ──
  if (processingStage > 0) {
    return (
      <div className={styles.bg}>
        {background}
        <div className={styles.processingContainer}>
          <div className={styles.logo}>
            <img src="/LOGO.png" alt="Vocal Insight Logo" style={{ width: '420px', maxWidth: '85vw', height: 'auto' }} />
          </div>

          <div className={styles.stages}>
            {STAGES.map((stage, i) => {
              const stageNum = i + 1
              const isDone   = processingStage > stageNum
              const isActive = processingStage === stageNum
              const Icon     = STAGE_ICONS[i]
              const nodeClass = isDone
                ? styles.nodeDone
                : isActive ? styles.nodeActive : styles.nodePending

              return (
                <div key={stage.id} className={styles.stageGroup}>
                  <div className={styles.stageItem}>
                    <div className={`${styles.nodeCircle} ${nodeClass}`}>
                      {isActive && <div className={styles.nodePulse} />}
                      <Icon dark={isDone || isActive} />
                    </div>
                    <div className={styles.stageLabel}>
                      {stage.lines.map((line, j) => (
                        <span key={j} className={isActive ? styles.stageLabelActive : ''}>{line}</span>
                      ))}
                    </div>
                  </div>

                  {i < STAGES.length - 1 && (
                    <div className={styles.connectorWrap}>
                      <div className={`${styles.connector} ${processingStage > stageNum ? styles.connectorDone : processingStage === stageNum ? styles.connectorActive : ''}`}>
                        <div className={styles.connectorFill} />
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          <div className={styles.spinner} />

          <p className={styles.processingStatus}>
            {processingStage === 1 && 'Checking audio quality...'}
            {processingStage === 2 && 'Running speech analysis models...'}
            {processingStage === 3 && 'Generating personalised feedback...'}
          </p>
        </div>
      </div>
    )
  }

  // ── Main UI ──
  const hasAudio = !!audioBlob && !recording

  return (
    <div className={history.length > 0 ? styles.pageWrapperScrollable : styles.pageWrapper} ref={wrapperRef}>

      {/* ── Section 1: Record UI ── */}
      <section className={styles.section1}>
        {background}
        <div className={styles.fadeOverlay} />

        <button
          onClick={() => { localStorage.removeItem('vi_user'); navigate('/login') }}
          style={{
            position: 'absolute',
            top: '20px',
            right: '24px',
            zIndex: 10,
            background: 'none',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: '8px',
            color: 'rgba(255,255,255,0.4)',
            fontSize: '0.62rem',
            fontFamily: 'inherit',
            fontWeight: 600,
            letterSpacing: '0.12em',
            padding: '6px 12px',
            cursor: 'pointer',
            transition: 'color 0.2s, border-color 0.2s',
          }}
          onMouseEnter={e => { e.currentTarget.style.color = 'rgba(255,255,255,0.75)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.35)' }}
          onMouseLeave={e => { e.currentTarget.style.color = 'rgba(255,255,255,0.4)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.15)' }}
        >
          LOG OUT
        </button>

        <div className={styles.container}>
          <div className={styles.logo}>
            <img src="/LOGO.png" alt="Vocal Insight Logo" style={{ width: '380px', maxWidth: '85vw', height: 'auto' }} />
          </div>

          <div className={styles.waveform}>
            {BAR_HEIGHTS.map((h, i) => (
              <div
                key={i}
                className={`${styles.bar} ${recording ? styles.barActive : ''}`}
                style={{ '--bar-h': h, animationDelay: `${(i * 0.08).toFixed(2)}s` }}
              />
            ))}
          </div>

          <div className={styles.timer}>{formatTime(seconds)}</div>

          <button
            className={`${styles.micBtn} ${recording ? styles.micBtnRecording : ''}`}
            onClick={toggleRecording}
            aria-label={recording ? 'Stop recording' : 'Start recording'}
          >
            <svg width="36" height="36" viewBox="0 0 24 24" fill="currentColor">
              <rect x="9" y="2" width="6" height="11" rx="3" />
              <path d="M5 11a7 7 0 0 0 14 0" strokeWidth="2" stroke="currentColor" fill="none" strokeLinecap="round"/>
              <line x1="12" y1="18" x2="12" y2="22" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <line x1="8" y1="22" x2="16" y2="22" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>

          {recording && <p className={styles.recordingLabel}>RECORDING...</p>}

          {/* ── Audio Player ── */}
          {hasAudio && (
            <div className={styles.playerCard}>
              <audio ref={audioRef} src={audioUrl} preload="metadata" />

              <div className={styles.playerHeader}>
                <span className={styles.playerLabel}>{audioLabel || 'RECORDING READY'}</span>
                <button className={styles.clearBtn} onClick={clearAudio} aria-label="Remove audio">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>

              <div className={styles.playerRow}>
                <button className={styles.playBtn} onClick={togglePlayback} aria-label={isPlaying ? 'Pause' : 'Play'}>
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

                <div className={styles.playerTimeline} onClick={handleSeek}>
                  <div className={styles.waveformBars}>
                    {(waveformPeaks || Array(WAVEFORM_BARS).fill(0.15)).map((h, i) => {
                      const progress = audioDuration ? playbackTime / audioDuration : 0
                      const isPast = (i / WAVEFORM_BARS) < progress
                      return (
                        <div
                          key={i}
                          className={`${styles.waveBar} ${isPast ? styles.waveBarPlayed : ''}`}
                          style={{ height: `${Math.max(8, h * 100)}%` }}
                        />
                      )
                    })}
                  </div>
                  {audioDuration > 0 && (
                    <div className={styles.playhead} style={{ left: `${(playbackTime / audioDuration) * 100}%` }} />
                  )}
                </div>
              </div>

              <div className={styles.playerTimes}>
                <span>{formatPlaybackTime(playbackTime)}</span>
                <span>{formatPlaybackTime(audioDuration)}</span>
              </div>
            </div>
          )}

          <div className={styles.inputWrapper}>
            <input
              className={styles.input}
              type="text"
              placeholder="EXPLAIN SCENARIO IN BRIEF"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
            />
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            style={{ display: 'none' }}
            onChange={handleUpload}
          />

          <button className={styles.btnSubmit} onClick={handleSubmit}>
            SUBMIT FOR ANALYSIS
          </button>

          <button
            onClick={() => fileInputRef.current.click()}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'rgba(255,255,255,0.3)',
              fontSize: '0.65rem',
              fontFamily: 'inherit',
              fontWeight: 600,
              letterSpacing: '0.12em',
              padding: '4px 0',
              transition: 'color 0.2s',
            }}
            onMouseEnter={e => e.currentTarget.style.color = 'rgba(255,255,255,0.55)'}
            onMouseLeave={e => e.currentTarget.style.color = 'rgba(255,255,255,0.3)'}
          >
            OR UPLOAD AN AUDIO FILE <span style={{ color: '#ef4444' }}>[NOT RECOMMENDED]</span>
          </button>
        </div>

        {history.length > 0 && (
          <div className={styles.scrollHint}>
            <span>PREVIOUS RESULTS</span>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        )}
      </section>

      {/* ── Section 2: History ── */}
      {history.length > 0 && (
        <section className={styles.section2}>
          <h2 className={styles.historyTitle}>PREVIOUS RESULTS</h2>
          <div ref={historyGridRef} className={styles.historyGrid}>
            {history.map((item) => (
              <div key={item.session_id} className={styles.historyCard}>
                <button
                  className={styles.historyCardBody}
                  onClick={() => loadHistoricalResult(item.session_id)}
                >
                  <div className={styles.historyCardLeft}>
                    <span className={styles.historyDate}>{formatHistoryDate(item.timestamp)}</span>
                    <span className={styles.historyTime}>{formatHistoryTime(item.timestamp)}</span>
                    {item.scenario && (
                      <span className={styles.historyScenario}>{item.scenario}</span>
                    )}
                    {item.problems_detected?.length > 0 && (
                      <div className={styles.historyTags}>
                        {item.problems_detected.map(p => (
                          <span key={p} className={styles.historyTag}>
                            {PROBLEM_DISPLAY[p] || p}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <svg className={styles.historyArrow} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                </button>
                <button
                  className={styles.historyDeleteBtn}
                  title="Delete result"
                  onClick={() => {
                    if (window.confirm('Permanently delete this result?')) {
                      deleteResult(item.session_id)
                    }
                  }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
