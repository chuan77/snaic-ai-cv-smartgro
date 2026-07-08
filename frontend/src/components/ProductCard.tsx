import { motion } from 'framer-motion'
import { ACCENT_STYLES, isAccentKey } from '@/lib/accents'
import { cn } from '@/lib/utils'
import { useSmartCart, type CartLineRow } from '@/hooks/useSmartCart'

interface ProductCardProps {
  line: CartLineRow
  interactive?: boolean
}

export function ProductCard({ line, interactive = true }: ProductCardProps) {
  const toggleLine = useSmartCart((state) => state.toggleLine)
  const accent = ACCENT_STYLES[isAccentKey(line.accent) ? line.accent : 'green']

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, height: 0 }}
      onClick={interactive ? () => toggleLine(line.lineId) : undefined}
      className={cn('flex items-center gap-3 border-b border-white/10 px-1 py-3 last:border-b-0', interactive && 'cursor-pointer')}
    >
      <span className={cn('h-2.5 w-2.5 shrink-0 rounded-full', line.active ? accent.dot : 'bg-white/20')} />
      <div className={cn('min-w-0 flex-1 transition-opacity', !line.active && 'opacity-55 line-through')}>
        <p className="truncate font-sans text-sm text-text-hi">{line.name}</p>
        <p className="truncate font-mono text-[11px] text-text-lo">
          {line.sku} · {(line.confidence * 100).toFixed(0)}%
        </p>
      </div>
      <span
        className={cn(
          'shrink-0 font-mono text-sm tabular-nums text-text-hi transition-opacity',
          !line.active && 'opacity-55 line-through',
        )}
      >
        USD {line.priceUsd.toFixed(2)}
      </span>
    </motion.div>
  )
}
