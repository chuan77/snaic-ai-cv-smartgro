import { AnimatePresence, motion } from 'framer-motion'
import { ACCENT_STYLES } from '@/lib/accents'
import { useSmartCart } from '@/hooks/useSmartCart'

export function DetectionOverlay() {
  const detections = useSmartCart((state) => state.detections)
  const detectionAccents = useSmartCart((state) => state.detectionAccents)

  return (
    <div className="pointer-events-none absolute inset-0">
      <AnimatePresence>
        {detections.map((detection) => {
          const accentKey = detectionAccents[detection.id] ?? 'green'
          const style = ACCENT_STYLES[accentKey]
          const isUnrecognized = accentKey === 'red'
          const [x, y, w, h] = detection.bbox
          const label = isUnrecognized ? 'UNRECOGNIZED' : (detection.label.split('/').pop() ?? detection.label)

          return (
            <motion.div
              key={detection.id}
              initial={{ opacity: 0, scale: 0.92 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className={`absolute rounded-control border-2 ${style.border} ${style.glow}`}
              style={{
                left: `${x * 100}%`,
                top: `${y * 100}%`,
                width: `${w * 100}%`,
                height: `${h * 100}%`,
              }}
            >
              <span
                className={`absolute -top-6 left-0 whitespace-nowrap rounded-full border px-2 py-0.5 font-mono text-[11px] ${style.bgTint} ${style.borderTint} ${style.text}`}
              >
                {label.replace(/-/g, ' ')} · {(detection.confidence * 100).toFixed(0)}%
              </span>
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}
