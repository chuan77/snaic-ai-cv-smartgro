import type { CatalogItem, Detection } from '@/types'

export interface ApiClient {
  predict(input: File): Promise<Detection[]>
  getCatalog(): Promise<CatalogItem[]>
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export const api: ApiClient = {
  async predict(input) {
    const formData = new FormData()
    formData.append('file', input)
    const response = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: formData })
    if (!response.ok) throw new Error(`predict failed: ${response.status}`)
    return response.json() as Promise<Detection[]>
  },

  async getCatalog() {
    const response = await fetch(`${API_BASE_URL}/catalog`)
    if (!response.ok) throw new Error(`getCatalog failed: ${response.status}`)
    return response.json() as Promise<CatalogItem[]>
  },
}
