import { useMemo } from 'react'
import styles from './AuthLayout.module.css'
import { TRIANGLES, ACCENTS } from './polyMesh'

const PARTICLE_COUNT = 20

export default function AuthLayout({ children }) {
  const particles = useMemo(() =>
    Array.from({ length: PARTICLE_COUNT }, () => ({
      left: `${Math.random() * 100}%`,
      animationDuration: `${8 + Math.random() * 12}s`,
      animationDelay: `${Math.random() * 10}s`,
      width: `${1.5 + Math.random() * 2}px`,
      height: `${1.5 + Math.random() * 2}px`,
      opacity: 0.2 + Math.random() * 0.4,
    })), [])

  return (
    <div className={styles.bg}>
      <svg
        className={styles.polyBg}
        viewBox="0 0 1280 720"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="baseBg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"   stopColor="#1a3a70" />
            <stop offset="50%"  stopColor="#0e2248" />
            <stop offset="100%" stopColor="#060e20" />
          </linearGradient>
        </defs>
        <rect width="1280" height="720" fill="url(#baseBg)" />
        {TRIANGLES.map((t, i) => (
          <polygon
            key={i}
            points={t.pts}
            fill={t.fill}
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="0.5"
            style={{ animationDelay: `${t.delay.toFixed(2)}s` }}
          />
        ))}
        {ACCENTS.map((t, i) => (
          <polygon key={`a${i}`} points={t.pts} fill={t.fill} stroke="none" />
        ))}
      </svg>

      {/* Ambient glow orbs */}
      <div className={`${styles.glowOrb} ${styles.glowOrb1}`} />
      <div className={`${styles.glowOrb} ${styles.glowOrb2}`} />
      <div className={`${styles.glowOrb} ${styles.glowOrb3}`} />

      {/* Floating particles */}
      <div className={styles.particles}>
        {particles.map((p, i) => (
          <div key={i} className={styles.particle} style={p} />
        ))}
      </div>

      <div className={styles.container}>
        <div className={styles.logo}>
          <img src="/LOGO.png" alt="Vocal Insight Logo" style={{ width: '520px', maxWidth: '90vw', height: 'auto' }} />
        </div>
        <div className={styles.card}>
          {children}
        </div>
      </div>
    </div>
  )
}
