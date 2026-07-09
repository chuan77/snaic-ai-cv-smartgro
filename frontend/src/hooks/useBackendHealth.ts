import { useEffect } from 'react'
import { useSmartCart } from '@/hooks/useSmartCart'

const POLL_INTERVAL_MS = 5000
const HEALTH_CHECK_TIMEOUT_MS = 3000

export function useBackendHealth(): void {
  const checkBackendHealth = useSmartCart((state) => state.checkBackendHealth)

  useEffect(() => {
    let inFlight = false
    let currentController: AbortController | null = null

    const poll = () => {
      if (inFlight) return
      inFlight = true
      const controller = new AbortController()
      currentController = controller
      const timeoutId = window.setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT_MS)

      void checkBackendHealth(controller.signal).finally(() => {
        window.clearTimeout(timeoutId)
        inFlight = false
      })
    }

    poll()
    const intervalId = window.setInterval(poll, POLL_INTERVAL_MS)

    return () => {
      window.clearInterval(intervalId)
      currentController?.abort()
    }
  }, [checkBackendHealth])
}
