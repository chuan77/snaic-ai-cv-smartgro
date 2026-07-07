export interface Detection {
  id: string
  label: string
  confidence: number
  /** [x, y, w, h] as 0-1 fractions of the feed's width/height */
  bbox: [number, number, number, number]
}

export interface CatalogItem {
  sku: string
  name: string
  priceUsd: number
}

export interface CartLine extends CatalogItem {
  active: boolean
  accent: string
  confidence: number
}
