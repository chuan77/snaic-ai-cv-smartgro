import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Tailwind classes for text/background/border tint, e.g. from ACCENT_STYLES */
  bgTint: string
  borderTint: string
  text: string
}

export function Badge({ bgTint, borderTint, text, className, children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[11px] tracking-wide',
        bgTint,
        borderTint,
        text,
        className,
      )}
      {...props}
    >
      {children}
    </span>
  )
}
