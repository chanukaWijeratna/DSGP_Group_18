import { useMemo } from 'react'

const W = 560, H = 240
const PAD = { top: 15, right: 58, bottom: 45, left: 48 }
const PLOT_W = W - PAD.left - PAD.right
const PLOT_H = H - PAD.top - PAD.bottom
const Y_TICKS = [0, 25, 50, 75, 100]
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

export default function SeverityChart({ data, currentSessionId, color, threshold }) {
  const gradientId = `sev-area-${color.replace('#', '')}`

  const yScale = (v) => PAD.top + PLOT_H - (v / 100) * PLOT_H
  const xScale = (i) =>
    PAD.left + (data.length === 1 ? PLOT_W / 2 : (i / (data.length - 1)) * PLOT_W)

  const linePath = useMemo(() =>
    data.map((d, i) =>
      `${i === 0 ? 'M' : 'L'} ${xScale(i).toFixed(1)} ${yScale(d.value).toFixed(1)}`
    ).join(' '),
  [data])

  const areaPath = useMemo(() => {
    if (data.length < 2) return ''
    const top = data.map((d, i) =>
      `${xScale(i).toFixed(1)} ${yScale(d.value).toFixed(1)}`
    ).join(' L ')
    return `M ${xScale(0).toFixed(1)} ${yScale(0).toFixed(1)} L ${top} L ${xScale(data.length - 1).toFixed(1)} ${yScale(0).toFixed(1)} Z`
  }, [data])

  const formatDate = (ts) => {
    const d = new Date(ts)
    return `${MONTHS[d.getMonth()]} ${d.getDate()}`
  }

  // Show max 6 x-axis labels, always include first and last
  const maxLabels = 6
  const step = Math.max(1, Math.ceil(data.length / maxLabels))
  const showLabel = (i) =>
    i === 0 || i === data.length - 1 || (i % step === 0 && data.length - 1 - i >= step / 2)

  const thresholdY = threshold != null ? yScale(threshold) : null

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {/* Horizontal grid lines */}
      {Y_TICKS.map(t => (
        <line key={t}
          x1={PAD.left} y1={yScale(t)} x2={W - PAD.right} y2={yScale(t)}
          stroke="rgba(255,255,255,0.06)" strokeWidth="1"
        />
      ))}

      {/* Threshold baseline */}
      {threshold != null && (
        <>
          <line
            x1={PAD.left} y1={thresholdY} x2={W - PAD.right} y2={thresholdY}
            stroke="rgba(255, 255, 255, 0.45)" strokeWidth="1.5"
            strokeDasharray="8 5"
          />
          <text
            x={W - PAD.right + 2} y={thresholdY + 4}
            fill="rgba(255, 255, 255, 0.5)"
            fontSize="9" fontFamily="'Inter', sans-serif" fontWeight="700"
            textAnchor="start"
          >FLAGGED</text>
        </>
      )}

      {/* Y-axis labels */}
      {Y_TICKS.map(t => (
        <text key={t} x={PAD.left - 8} y={yScale(t) + 4}
          textAnchor="end" fill="rgba(255,255,255,0.35)"
          fontSize="11" fontFamily="'Inter', sans-serif" fontWeight="600"
        >{t}%</text>
      ))}

      {/* Area fill */}
      {areaPath && <path d={areaPath} fill={`url(#${gradientId})`} />}

      {/* Line */}
      {data.length > 1 && (
        <path d={linePath} fill="none" stroke={color} strokeWidth="2.5"
          strokeLinecap="round" strokeLinejoin="round" />
      )}

      {/* Data points */}
      {data.map((d, i) => {
        const isCurrent = String(d.session_id) === String(currentSessionId)
        const cx = xScale(i), cy = yScale(d.value)
        return (
          <g key={i}>
            {isCurrent && (
              <circle cx={cx} cy={cy} r="9" fill={color} opacity="0.12" />
            )}
            <circle cx={cx} cy={cy} r={isCurrent ? 5 : 3.5}
              fill={isCurrent ? '#ffffff' : color}
              stroke={isCurrent ? color : 'none'} strokeWidth={isCurrent ? 2.5 : 0}
            />
            <title>{`${formatDate(d.timestamp)}: ${d.value.toFixed(1)}%`}</title>
          </g>
        )
      })}

      {/* X-axis labels */}
      {data.map((d, i) => showLabel(i) ? (
        <text key={i} x={xScale(i)} y={H - PAD.bottom + 20}
          textAnchor="middle" fill="rgba(255,255,255,0.35)"
          fontSize="10" fontFamily="'Inter', sans-serif" fontWeight="500"
        >{formatDate(d.timestamp)}</text>
      ) : null)}
    </svg>
  )
}
