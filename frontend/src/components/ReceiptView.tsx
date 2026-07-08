import { AnimatePresence } from 'framer-motion'
import { ProductCard } from '@/components/ProductCard'
import type { CartLineRow } from '@/hooks/useSmartCart'

interface ReceiptViewProps {
  lines: CartLineRow[]
}

export function ReceiptView({ lines }: ReceiptViewProps) {
  if (lines.length === 0) {
    return <p className="py-8 text-center font-mono text-xs text-text-lo">No items to display.</p>
  }

  return (
    <AnimatePresence initial={false}>
      {lines.map((line) => (
        <ProductCard key={line.lineId} line={line} interactive={false} />
      ))}
    </AnimatePresence>
  )
}
