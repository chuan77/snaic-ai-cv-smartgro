import type { MatchTier } from '@/lib/matchCatalog'
import { useSmartCart } from '@/hooks/useSmartCart'

const TIER_STYLE: Record<MatchTier, string> = {
  'exact-segment': 'text-accent-green border-accent-green/30 bg-accent-green/10',
  'whole-word': 'text-accent-drizzle border-accent-drizzle/30 bg-accent-drizzle/10',
  'weak-substring': 'text-accent-yellow border-accent-yellow/30 bg-accent-yellow/10',
  none: 'text-accent-red border-accent-red/30 bg-accent-red/10',
}

const TIER_LABEL: Record<MatchTier, string> = {
  'exact-segment': 'EXACT',
  'whole-word': 'WORD',
  'weak-substring': 'WEAK',
  none: 'NO MATCH',
}

export function DebugPanel() {
  const debugLog = useSmartCart((state) => state.debugLog)

  return (
    <div className="mt-4 rounded-panel border border-white/10 bg-surface-raised p-3">
      <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.16em] text-text-lo">
        Debug · filename → catalog match
      </p>
      {debugLog.length === 0 ? (
        <p className="font-mono text-xs text-text-lo">No scans yet. Drop a file or capture a frame to see match reasoning here.</p>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {debugLog.map((entry, i) => (
            <li key={`${entry.filename}-${i}`} className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
              <span className="rounded border border-white/15 bg-canvas px-1.5 py-0.5 text-text-mid">
                {entry.filename}
              </span>
              <span className="text-text-lo">→ needle "{entry.needle || '(empty)'}"</span>
              <span className={`rounded-full border px-1.5 py-0.5 ${TIER_STYLE[entry.tier]}`}>
                {TIER_LABEL[entry.tier]}
              </span>
              {entry.match ? (
                <span className="text-text-hi">{entry.match.sku}</span>
              ) : (
                <span className="text-text-lo">unrecognized</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
