import type { CatalogItem } from '@/types'

export type MatchTier = 'exact-segment' | 'whole-word' | 'weak-substring' | 'none'

export interface MatchExplanation {
  filename: string
  needle: string
  tier: MatchTier
  match: CatalogItem | null
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/\.[a-z0-9]+$/, '')
    .replace(/[-_]/g, ' ')
    .replace(/\s+\d+$/, '')
    .trim()
}

function wordBoundaryIncludes(haystack: string, needle: string): boolean {
  return new RegExp(`(?:^|\\s)${needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:\\s|$)`).test(haystack)
}

/**
 * Fuzzy-matches a dropped filename against the catalog. Checks every path
 * segment (not just the leaf) so a category-level word like "apple" resolves
 * to Fruit/Apple/* rather than any product whose *name* merely contains the
 * word (e.g. "God-Morgon-Apple-Juice"). Returns 'none' — never a fallback
 * substitution — so unmatched drops can be flagged as "unrecognized".
 */
export function explainMatch(filename: string, catalog: CatalogItem[]): MatchExplanation {
  const needle = normalize(filename)
  if (!needle) return { filename, needle, tier: 'none', match: null }

  const items = catalog.map((item) => ({
    item,
    segments: item.sku.split('/').map(normalize),
    leaf: normalize(item.sku.split('/').pop() ?? ''),
  }))

  const exactSegment = items.find(({ segments }) => segments.includes(needle))
  if (exactSegment) return { filename, needle, tier: 'exact-segment', match: exactSegment.item }

  const wholeWordInLeaf = items.find(({ leaf }) => wordBoundaryIncludes(leaf, needle))
  if (wholeWordInLeaf) return { filename, needle, tier: 'whole-word', match: wholeWordInLeaf.item }

  const weakSubstring = items.find(({ leaf }) => leaf.length > 0 && (leaf.includes(needle) || needle.includes(leaf)))
  if (weakSubstring) return { filename, needle, tier: 'weak-substring', match: weakSubstring.item }

  return { filename, needle, tier: 'none', match: null }
}

export function matchDroppedFileToSku(filename: string, catalog: CatalogItem[]): CatalogItem | null {
  return explainMatch(filename, catalog).match
}
