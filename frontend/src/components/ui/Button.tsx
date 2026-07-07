import { cva, type VariantProps } from 'class-variance-authority'
import { type ButtonHTMLAttributes, forwardRef } from 'react'
import { cn } from '@/lib/utils'

// Buttons use `rounded-control` (not the design doc's literal `rounded-lg`) so every
// radius in the app resolves to one of the three bespoke tokens (control/panel/card).
const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 rounded-control font-sans text-sm font-semibold transition-colors disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        primary: 'bg-accent-green text-canvas shadow-glow-green hover:bg-accent-green/90',
        secondary: 'border border-white/30 bg-transparent text-text-hi hover:border-accent-cyan hover:text-accent-cyan',
        danger: 'border border-accent-red/40 bg-accent-red/10 text-accent-red hover:bg-accent-red/20',
      },
      size: {
        default: 'h-10 px-4',
        sm: 'h-8 px-3 text-xs',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'default',
    },
  },
)

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...props }, ref) => {
  return <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
})
Button.displayName = 'Button'
