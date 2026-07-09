import { motion } from 'framer-motion'
import { ScanLine } from 'lucide-react'
import { useSmartCart } from '@/hooks/useSmartCart'
import { cn } from '@/lib/utils'

export function HeaderBar() {
  const isConnected = useSmartCart((state) => state.backendStatus === 'connected')

  return (
    <header>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-panel bg-gradient-to-br from-accent-green to-accent-cyan shadow-glow-green">
            <ScanLine className="h-6 w-6 text-canvas" strokeWidth={2.5} />
          </div>
          <div className="min-w-0">
            <h1 className="truncate font-display text-xl uppercase leading-none tracking-wide text-text-hi sm:text-2xl">
              SmartCart Vision
            </h1>
            <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.18em] text-text-lo">
              ONNX Detection Demo
            </p>
          </div>
        </div>

        <div
          className={cn(
            'flex shrink-0 items-center gap-2 whitespace-nowrap rounded-full border px-3 py-1.5',
            isConnected ? 'border-accent-green/30 bg-accent-green/[0.08]' : 'border-accent-red/30 bg-accent-red/[0.08]',
          )}
        >
          <motion.span
            className={cn('h-2 w-2 shrink-0 rounded-full', isConnected ? 'bg-accent-green' : 'bg-accent-red')}
            animate={{ opacity: [1, 0.35, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
          <span
            className={cn(
              'font-mono text-[11px] uppercase tracking-[0.16em]',
              isConnected ? 'text-accent-green' : 'text-accent-red',
            )}
          >
            {isConnected ? 'Model Ready' : 'Disconnected'}
          </span>
        </div>
      </div>
      <div className="mt-6 h-px bg-white/10" />
    </header>
  )
}
