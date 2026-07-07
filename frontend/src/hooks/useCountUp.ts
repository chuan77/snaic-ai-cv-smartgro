import { animate } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

/** Tweens a displayed number toward `value` whenever it changes. */
export function useCountUp(value: number, duration = 0.4): number {
  const [display, setDisplay] = useState(value)
  const displayRef = useRef(value)

  useEffect(() => {
    const controls = animate(displayRef.current, value, {
      duration,
      ease: 'easeOut',
      onUpdate: (v) => {
        displayRef.current = v
        setDisplay(v)
      },
    })
    return () => controls.stop()
  }, [value, duration])

  return display
}
