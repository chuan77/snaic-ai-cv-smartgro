interface SectionHeaderProps {
  index: string
  title: string
}

export function SectionHeader({ index, title }: SectionHeaderProps) {
  return (
    <div className="mb-4 flex items-center gap-3">
      <span className="font-mono text-xs uppercase tracking-[0.16em] text-text-lo">{index}</span>
      <h2 className="font-display text-lg uppercase text-text-hi">{title}</h2>
      <div className="h-px flex-1 bg-white/10" />
    </div>
  )
}
