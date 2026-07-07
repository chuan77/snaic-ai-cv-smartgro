import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#05070D',
        surface: { DEFAULT: '#0B1019', raised: '#0E1422' },
        text: { hi: '#EDF1F8', mid: '#93A0B4', lo: '#5A6477' },
        accent: {
          green: '#2FE8A0',
          cyan: '#38C6F4',
          purple: '#9B5CF6',
          pink: '#E5379A',
          yellow: '#FFC53D',
          red: '#FF4D5E',
          blue: '#3F7DF6',
          drizzle: '#57C9FF',
        },
      },
      fontFamily: {
        display: ['"Saira Condensed"', 'sans-serif'],
        sans: ['"IBM Plex Sans"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      borderRadius: {
        control: '6px',
        panel: '12px',
        card: '18px',
      },
      boxShadow: {
        float: '0 8px 24px rgba(0,0,0,0.5)',
        'glow-cyan': '0 0 22px rgba(56,198,244,0.4)',
        'glow-green': '0 0 20px rgba(47,232,160,0.4)',
        // Dedicated 18px glows for detection-box accents (cycle + unrecognized flag),
        // kept separate from the 20-22px UI glows above.
        'bbox-green': '0 0 18px rgba(47,232,160,0.4)',
        'bbox-yellow': '0 0 18px rgba(255,197,61,0.4)',
        'bbox-drizzle': '0 0 18px rgba(87,201,255,0.4)',
        'bbox-purple': '0 0 18px rgba(155,92,246,0.4)',
        'bbox-pink': '0 0 18px rgba(229,55,154,0.4)',
        'bbox-red': '0 0 18px rgba(255,77,94,0.4)',
      },
    },
  },
  plugins: [],
} satisfies Config
