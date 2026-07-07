export type AccentKey = 'green' | 'yellow' | 'drizzle' | 'purple' | 'pink' | 'red'

/** Cycle order for successive detections. 'red' is reserved for unrecognized items and never appears here. */
export const ACCENT_CYCLE: AccentKey[] = ['green', 'yellow', 'drizzle', 'purple', 'pink']

export interface AccentStyle {
  /** 2px bounding-box border */
  border: string
  /** matching 18px box-shadow glow */
  glow: string
  /** badge/label text color */
  text: string
  /** badge/label tinted background */
  bgTint: string
  /** badge/label tinted border */
  borderTint: string
  /** solid dot background, for cart line status dots */
  dot: string
}

// Every value below is a literal Tailwind class string so the JIT content
// scanner picks it up — class names built from a runtime accent variable
// would not survive a production build.
export const ACCENT_STYLES: Record<AccentKey, AccentStyle> = {
  green: {
    border: 'border-accent-green',
    glow: 'shadow-bbox-green',
    text: 'text-accent-green',
    bgTint: 'bg-accent-green/10',
    borderTint: 'border-accent-green/30',
    dot: 'bg-accent-green shadow-bbox-green',
  },
  yellow: {
    border: 'border-accent-yellow',
    glow: 'shadow-bbox-yellow',
    text: 'text-accent-yellow',
    bgTint: 'bg-accent-yellow/10',
    borderTint: 'border-accent-yellow/30',
    dot: 'bg-accent-yellow shadow-bbox-yellow',
  },
  drizzle: {
    border: 'border-accent-drizzle',
    glow: 'shadow-bbox-drizzle',
    text: 'text-accent-drizzle',
    bgTint: 'bg-accent-drizzle/10',
    borderTint: 'border-accent-drizzle/30',
    dot: 'bg-accent-drizzle shadow-bbox-drizzle',
  },
  purple: {
    border: 'border-accent-purple',
    glow: 'shadow-bbox-purple',
    text: 'text-accent-purple',
    bgTint: 'bg-accent-purple/10',
    borderTint: 'border-accent-purple/30',
    dot: 'bg-accent-purple shadow-bbox-purple',
  },
  pink: {
    border: 'border-accent-pink',
    glow: 'shadow-bbox-pink',
    text: 'text-accent-pink',
    bgTint: 'bg-accent-pink/10',
    borderTint: 'border-accent-pink/30',
    dot: 'bg-accent-pink shadow-bbox-pink',
  },
  red: {
    border: 'border-accent-red',
    glow: 'shadow-bbox-red',
    text: 'text-accent-red',
    bgTint: 'bg-accent-red/10',
    borderTint: 'border-accent-red/30',
    dot: 'bg-accent-red shadow-bbox-red',
  },
}

export function accentForIndex(i: number): AccentKey {
  return ACCENT_CYCLE[i % ACCENT_CYCLE.length]
}

export function isAccentKey(value: string): value is AccentKey {
  return value in ACCENT_STYLES
}
