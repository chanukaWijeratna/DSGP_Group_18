// Generate a dense low-poly triangle mesh across a 1280x1440 canvas
// The mesh is taller than the viewport so triangles continue as the user scrolls.
// Animation delays flow as a wave from top-left to bottom-right.

const COLS = 14
const ROWS = 18
const W = 1280
const H = 1440

// Color palette — lifted the darkest shades for a more even look
const PALETTE = [
  '#142848', '#162c50', '#172e52', '#1a3460', '#1a3560',
  '#1d3968', '#1d3a68', '#1e3a6a', '#1e3c6c', '#1f3d6c',
  '#1f3e6e', '#20406e', '#214070', '#254878', '#1a3a70',
  '#1c3e72', '#204476', '#244a7c', '#264e80', '#2a5488',
]

// Seeded random for deterministic jitter
function seededRandom(seed) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

function generateMesh() {
  const rand = seededRandom(42)
  const cellW = W / COLS
  const cellH = H / ROWS

  // Generate grid vertices with jitter
  const vertices = []
  for (let r = 0; r <= ROWS; r++) {
    for (let c = 0; c <= COLS; c++) {
      let x = c * cellW
      let y = r * cellH

      // Add jitter except at edges to maintain full coverage
      const isEdgeX = c === 0 || c === COLS
      const isEdgeY = r === 0 || r === ROWS
      if (!isEdgeX) x += (rand() - 0.5) * cellW * 0.55
      if (!isEdgeY) y += (rand() - 0.5) * cellH * 0.55

      // Clamp to canvas
      x = Math.max(0, Math.min(W, Math.round(x)))
      y = Math.max(0, Math.min(H, Math.round(y)))

      vertices.push({ x, y })
    }
  }

  const getVert = (r, c) => vertices[r * (COLS + 1) + c]

  const triangles = []
  const accents = []

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const tl = getVert(r, c)
      const tr = getVert(r, c + 1)
      const bl = getVert(r + 1, c)
      const br = getVert(r + 1, c + 1)

      // Two triangles per cell — alternate split direction for variety
      const splitDir = (r + c) % 2 === 0

      const makeTri = (a, b, cc) => {
        const pts = `${a.x},${a.y} ${b.x},${b.y} ${cc.x},${cc.y}`
        const cx = (a.x + b.x + cc.x) / 3
        const cy = (a.y + b.y + cc.y) / 3

        // Fully random palette pick — no position bias
        const colorIdx = Math.floor(rand() * PALETTE.length)

        // Wave delay: top-left (0,0) starts first, bottom-right last
        const waveDelay = ((cx / W) * 0.6 + (cy / H) * 0.4) * 3.5

        return { pts, fill: PALETTE[colorIdx], delay: waveDelay, cx, cy }
      }

      if (splitDir) {
        triangles.push(makeTri(tl, tr, bl))
        triangles.push(makeTri(tr, br, bl))
      } else {
        triangles.push(makeTri(tl, tr, br))
        triangles.push(makeTri(tl, br, bl))
      }
    }
  }

  // Pick a few center-ish triangles as accents (within top half — the visible hero area)
  const centerTris = triangles
    .filter(t => {
      const dx = t.cx / W - 0.5
      const dy = t.cy / (H * 0.5) - 0.5  // relative to top half
      return dy >= -0.5 && dy <= 0.5 && Math.sqrt(dx * dx + dy * dy) < 0.3
    })
    .slice(0, 6)

  centerTris.forEach((t, i) => {
    const opacity = 0.10 + (i % 3) * 0.04
    accents.push({
      pts: t.pts,
      fill: `rgba(80,140,220,${opacity})`,
    })
  })

  return { triangles, accents }
}

const mesh = generateMesh()
export const TRIANGLES = mesh.triangles
export const ACCENTS = mesh.accents
