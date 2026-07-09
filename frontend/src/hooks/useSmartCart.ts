import { create } from 'zustand'
import { accentForIndex, type AccentKey } from '@/lib/accents'
import { api } from '@/lib/api'
import { explainMatch, type MatchExplanation } from '@/lib/matchCatalog'
import type { CartLine, CatalogItem, Detection } from '@/types'

export interface CartLineRow extends CartLine {
  lineId: string
}

const MAX_DEBUG_LOG = 20

export type BackendStatus = 'connected' | 'disconnected'

interface SmartCartState {
  catalog: CatalogItem[]
  detections: Detection[]
  detectionAccents: Record<string, AccentKey>
  cartLines: CartLineRow[]
  feedImage: string | null
  isProcessing: boolean
  error: string | null
  debugMode: boolean
  debugLog: MatchExplanation[]
  isCheckedOut: boolean
  backendStatus: BackendStatus

  loadCatalog: () => Promise<void>
  checkBackendHealth: (signal?: AbortSignal) => Promise<void>
  setFeedImage: (src: string | null) => void
  runDetection: (files: File[]) => Promise<void>
  toggleLine: (lineId: string) => void
  clearCart: () => void
  resumeLive: () => void
  toggleDebugMode: () => void
  checkout: () => void
  closeReceipt: () => void
}

export const useSmartCart = create<SmartCartState>((set, get) => ({
  catalog: [],
  detections: [],
  detectionAccents: {},
  cartLines: [],
  feedImage: null,
  isProcessing: false,
  error: null,
  debugMode: false,
  debugLog: [],
  isCheckedOut: false,
  backendStatus: 'connected',

  loadCatalog: async () => {
    const catalog = await api.getCatalog()
    set({ catalog })
  },

  checkBackendHealth: async (signal) => {
    try {
      await api.getHealth(signal)
      set({ backendStatus: 'connected' })
    } catch {
      set({ backendStatus: 'disconnected' })
    }
  },

  setFeedImage: (src) => set({ feedImage: src }),

  runDetection: async (files) => {
    if (!files.length) return
    set({ isProcessing: true, error: null })
    try {
      const catalog = get().catalog
      const explanations = files.map((file) => explainMatch(file.name, catalog))
      const results = await Promise.all(files.map((file) => api.predict(file)))
      const flatDetections = results.flat()

      const detectionAccents: Record<string, AccentKey> = {}
      const newRows: CartLineRow[] = []
      let nextAccentIndex = get().cartLines.length

      for (const detection of flatDetections) {
        const match = catalog.find((item) => item.sku === detection.label)
        if (!match) {
          detectionAccents[detection.id] = 'red'
          continue
        }
        const accent = accentForIndex(nextAccentIndex)
        nextAccentIndex += 1
        detectionAccents[detection.id] = accent
        newRows.push({
          ...match,
          lineId: detection.id,
          active: true,
          accent,
          confidence: detection.confidence,
        })
      }

      set((state) => {
        if (state.isCheckedOut) return state
        return {
          detections: flatDetections,
          detectionAccents,
          cartLines: [...state.cartLines, ...newRows],
          debugLog: [...explanations.reverse(), ...state.debugLog].slice(0, MAX_DEBUG_LOG),
        }
      })
    } catch {
      set({ error: 'Detection failed' })
    } finally {
      set({ isProcessing: false })
    }
  },

  toggleLine: (lineId) =>
    set((state) => ({
      cartLines: state.cartLines.map((line) => (line.lineId === lineId ? { ...line, active: !line.active } : line)),
    })),

  clearCart: () =>
    set({
      cartLines: [],
      detections: [],
      detectionAccents: {},
      feedImage: null,
    }),

  resumeLive: () =>
    set({
      feedImage: null,
      detections: [],
      detectionAccents: {},
    }),

  toggleDebugMode: () => set((state) => ({ debugMode: !state.debugMode })),

  checkout: () => {
    if (selectActiveCount(get()) === 0) return
    set({ isCheckedOut: true })
  },

  closeReceipt: () => {
    get().clearCart()
    set({ isCheckedOut: false })
  },
}))

export const selectTotal = (state: SmartCartState): number =>
  state.cartLines.filter((line) => line.active).reduce((sum, line) => sum + line.priceUsd, 0)

export const selectActiveCount = (state: SmartCartState): number =>
  state.cartLines.filter((line) => line.active).length
