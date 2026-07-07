import { motion } from 'framer-motion'
import { UploadCloud } from 'lucide-react'
import { type DragEvent, useEffect, useRef, useState } from 'react'
import Webcam from 'react-webcam'
import { DebugPanel } from '@/components/DebugPanel'
import { DetectionOverlay } from '@/components/DetectionOverlay'
import { Button } from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import { useSmartCart } from '@/hooks/useSmartCart'

const RESUME_DELAY_MS = 1000

const GRID_BACKGROUND = {
  backgroundImage:
    'linear-gradient(rgba(56,198,244,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(56,198,244,0.06) 1px, transparent 1px)',
  backgroundSize: '32px 32px',
}

function CornerTicks() {
  const positions = [
    'left-3 top-3 border-l-2 border-t-2',
    'right-3 top-3 border-r-2 border-t-2',
    'left-3 bottom-3 border-l-2 border-b-2',
    'right-3 bottom-3 border-r-2 border-b-2',
  ]
  return (
    <>
      {positions.map((pos) => (
        <div key={pos} className={`pointer-events-none absolute h-4 w-4 border-accent-cyan/50 ${pos}`} />
      ))}
    </>
  )
}

export function CameraFeed() {
  const feedImage = useSmartCart((state) => state.feedImage)
  const isProcessing = useSmartCart((state) => state.isProcessing)
  const cartLineCount = useSmartCart((state) => state.cartLines.length)
  const setFeedImage = useSmartCart((state) => state.setFeedImage)
  const runDetection = useSmartCart((state) => state.runDetection)
  const clearCart = useSmartCart((state) => state.clearCart)
  const resumeLive = useSmartCart((state) => state.resumeLive)
  const debugMode = useSmartCart((state) => state.debugMode)
  const toggleDebugMode = useSmartCart((state) => state.toggleDebugMode)

  const webcamRef = useRef<Webcam>(null)
  const resumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [webcamReady, setWebcamReady] = useState(false)

  useEffect(() => {
    return () => {
      if (resumeTimerRef.current) clearTimeout(resumeTimerRef.current)
    }
  }, [])

  const detectAndScheduleResume = async (files: File[]) => {
    await runDetection(files)
    if (resumeTimerRef.current) clearTimeout(resumeTimerRef.current)
    resumeTimerRef.current = setTimeout(() => {
      resumeLive()
      resumeTimerRef.current = null
    }, RESUME_DELAY_MS)
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragOver(false)
    const files = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith('image/'))
    if (!files.length) return
    setFeedImage(URL.createObjectURL(files[files.length - 1]))
    void detectAndScheduleResume(files)
  }

  const handleFreezeAndDetect = async () => {
    const screenshot = webcamRef.current?.getScreenshot()
    if (!screenshot) return
    const blob = await fetch(screenshot).then((res) => res.blob())
    const capture = new File([blob], 'capture.jpg', { type: blob.type })
    setFeedImage(screenshot)
    await detectAndScheduleResume([capture])
  }

  const handleReset = () => {
    if (resumeTimerRef.current) {
      clearTimeout(resumeTimerRef.current)
      resumeTimerRef.current = null
    }
    clearCart()
  }

  const showEmptyState = !feedImage && !webcamReady

  return (
    <section>
      <div className="relative aspect-video overflow-hidden rounded-card border border-white/10 bg-surface shadow-float">
        <div className="absolute inset-0" style={GRID_BACKGROUND} />

        {feedImage ? (
          <img src={feedImage} alt="Detection feed" className="absolute inset-0 h-full w-full rounded-panel object-cover" />
        ) : (
          <Webcam
            ref={webcamRef}
            audio={false}
            mirrored
            onUserMedia={() => setWebcamReady(true)}
            onUserMediaError={() => setWebcamReady(false)}
            className="absolute inset-0 h-full w-full rounded-panel object-cover"
          />
        )}

        <DetectionOverlay />
        <CornerTicks />

        <motion.div
          className="pointer-events-none absolute left-0 right-0 h-px bg-accent-cyan shadow-glow-cyan"
          animate={{ top: ['0%', '100%'] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        />

        <div
          data-testid="detection-dropzone"
          onDragOver={(event) => {
            event.preventDefault()
            setIsDragOver(true)
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          className={`absolute inset-0 transition-colors ${isDragOver ? 'bg-accent-cyan/10' : ''}`}
        >
          {showEmptyState && (
            <div className="flex h-full flex-col items-center justify-center gap-2 rounded-panel border-2 border-dashed border-white/15 bg-canvas/40">
              <UploadCloud className="h-6 w-6 text-text-lo" />
              <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-text-lo">
                Drag Item Images Here
              </p>
            </div>
          )}
          {isProcessing && (
            <div className="absolute right-3 top-3 rounded-full border border-accent-yellow/30 bg-accent-yellow/10 px-2 py-0.5 font-mono text-[11px] uppercase tracking-[0.16em] text-accent-yellow">
              Analyzing…
            </div>
          )}
        </div>
      </div>

      <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.16em] text-text-lo">
        Cam 01 · 30 FPS · 1280×720 · {cartLineCount} item{cartLineCount === 1 ? '' : 's'} scanned
      </p>

      <div className="mt-4 flex items-center gap-3">
        <Button variant="primary" onClick={handleFreezeAndDetect} disabled={!webcamReady || isProcessing}>
          Freeze &amp; Detect
        </Button>
        <Button variant="secondary" onClick={clearCart}>
          Reset
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={toggleDebugMode}
          className={cn('ml-auto', debugMode && 'border-accent-cyan text-accent-cyan')}
        >
          Debug {debugMode ? 'On' : 'Off'}
        </Button>
      </div>

      {debugMode && <DebugPanel />}
    </section>
  )
}
