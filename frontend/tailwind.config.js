/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Dark medical theme palette
        void:    '#080c14',
        deep:    '#0d1422',
        panel:   '#111827',
        surface: '#1a2236',
        border:  '#1e2d45',
        // Accent colors
        cyan:    { DEFAULT: '#00d4ff', dim: '#0099cc', glow: 'rgba(0,212,255,0.15)' },
        emerald: { DEFAULT: '#00ff88', dim: '#00c060', glow: 'rgba(0,255,136,0.12)' },
        amber:   { DEFAULT: '#ffb020', dim: '#cc8800', glow: 'rgba(255,176,32,0.12)' },
        rose:    { DEFAULT: '#ff4466', dim: '#cc2244', glow: 'rgba(255,68,102,0.12)' },
        violet:  { DEFAULT: '#8855ff', dim: '#6633cc', glow: 'rgba(136,85,255,0.12)' },
        // Text
        muted:   '#64748b',
        subtle:  '#94a3b8',
        primary: '#e2e8f0',
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
        mono:    ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        sans:    ['"DM Sans"', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)',
        'radial-glow': 'radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.08) 0%, transparent 70%)',
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
      boxShadow: {
        'cyan-glow':   '0 0 20px rgba(0,212,255,0.2), 0 0 60px rgba(0,212,255,0.05)',
        'emerald-glow':'0 0 20px rgba(0,255,136,0.2), 0 0 60px rgba(0,255,136,0.05)',
        'rose-glow':   '0 0 20px rgba(255,68,102,0.2)',
        'amber-glow':  '0 0 20px rgba(255,176,32,0.2)',
        'panel':       '0 4px 24px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.04) inset',
      },
      animation: {
        'scan-line': 'scanLine 3s ease-in-out infinite',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        scanLine: {
          '0%, 100%': { transform: 'translateY(-100%)' },
          '50%': { transform: 'translateY(400%)' },
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.6' },
          '50%': { opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
      },
    },
  },
  plugins: [],
}
