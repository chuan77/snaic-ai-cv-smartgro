import { AnimatePresence, motion } from 'framer-motion'
import { SectionHeader } from '@/components/SectionHeader'
import { ProductCard } from '@/components/ProductCard'
import { ReceiptView } from '@/components/ReceiptView'
import { Button } from '@/components/ui/Button'
import { useCountUp } from '@/hooks/useCountUp'
import { selectActiveCount, selectTotal, useSmartCart } from '@/hooks/useSmartCart'

export function CartSidebar() {
  const cartLines = useSmartCart((state) => state.cartLines)
  const activeCount = useSmartCart(selectActiveCount)
  const total = useSmartCart(selectTotal)
  const animatedTotal = useCountUp(total)
  const isCheckedOut = useSmartCart((state) => state.isCheckedOut)
  const checkout = useSmartCart((state) => state.checkout)
  const closeReceipt = useSmartCart((state) => state.closeReceipt)

  const receiptLines = cartLines.filter((line) => line.active)

  return (
    <aside className="flex flex-col rounded-card border border-white/10 bg-surface p-5 shadow-float lg:w-[400px]">
      <SectionHeader index="02" title={isCheckedOut ? 'Receipt' : 'Cart'} />
      <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.16em] text-text-lo">
        {activeCount} item{activeCount === 1 ? '' : 's'} {isCheckedOut ? 'on receipt' : 'detected'}
      </p>

      <div className="min-h-[120px] flex-1 overflow-y-auto">
        <AnimatePresence mode="wait" initial={false}>
          {isCheckedOut ? (
            <motion.div key="receipt" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <ReceiptView lines={receiptLines} />
            </motion.div>
          ) : (
            <motion.div key="cart" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {cartLines.length === 0 ? (
                <p className="py-8 text-center font-mono text-xs text-text-lo">Basket is empty.</p>
              ) : (
                <AnimatePresence initial={false}>
                  {cartLines.map((line) => (
                    <ProductCard key={line.lineId} line={line} />
                  ))}
                </AnimatePresence>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="mt-4 border-t border-white/10 pt-4">
        <div className="flex items-baseline justify-between">
          <span className="font-display text-lg uppercase text-text-hi">Total</span>
          <motion.span
            className="font-display text-3xl tabular-nums text-accent-green"
            style={{ textShadow: '0 0 20px rgba(47,232,160,0.4)' }}
          >
            USD {animatedTotal.toFixed(2)}
          </motion.span>
        </div>

        {isCheckedOut ? (
          <Button variant="primary" onClick={closeReceipt} className="mt-4 w-full bg-gradient-to-r from-accent-green to-accent-cyan">
            Acknowledge &amp; Close
          </Button>
        ) : (
          <Button
            variant="primary"
            onClick={checkout}
            disabled={activeCount === 0}
            className="mt-4 w-full bg-gradient-to-r from-accent-green to-accent-cyan"
          >
            Checkout →
          </Button>
        )}

        <p className="mt-3 text-center font-mono text-[11px] uppercase tracking-[0.16em] text-text-lo">
          {isCheckedOut ? 'Receipt is final — acknowledge to start a new scan.' : 'Tap a line to remove / restore'}
        </p>
      </div>
    </aside>
  )
}
