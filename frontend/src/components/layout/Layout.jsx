import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Upload, History, Shield, Menu, X, Activity } from 'lucide-react'

const navItems = [
  { path: '/',        label: 'Home',    icon: Brain },
  { path: '/upload',  label: 'Analyze', icon: Upload },
  { path: '/history', label: 'History', icon: History },
  { path: '/safety',  label: 'AI Safety', icon: Shield },
]

export default function Layout({ children }) {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="min-h-screen bg-void bg-grid-pattern bg-grid">
      {/* Background radial glow */}
      <div className="fixed inset-0 bg-radial-glow pointer-events-none" />

      {/* ── Navigation ─────────────────────────────────────── */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-border/40 bg-void/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-cyan/10 border border-cyan/30 flex items-center justify-center group-hover:bg-cyan/20 transition-all">
              <Brain className="w-4 h-4 text-cyan" />
            </div>
            <span className="font-display font-semibold text-primary tracking-tight">
              NeuroScan <span className="text-cyan">AI</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map(({ path, label, icon: Icon }) => {
              const active = location.pathname === path
              return (
                <Link
                  key={path}
                  to={path}
                  className={`relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    active
                      ? 'text-cyan bg-cyan/10'
                      : 'text-muted hover:text-primary hover:bg-surface'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                  {active && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute inset-0 rounded-lg border border-cyan/30"
                      transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
                    />
                  )}
                </Link>
              )
            })}
          </div>

          {/* Status indicator */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface border border-border/50">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald animate-pulse" />
            <span className="text-xs text-muted font-mono">SYSTEM ONLINE</span>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-lg text-muted hover:text-primary"
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile menu */}
        <AnimatePresence>
          {mobileOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="md:hidden border-t border-border/40 bg-void/95 backdrop-blur-xl overflow-hidden"
            >
              <div className="px-4 py-3 space-y-1">
                {navItems.map(({ path, label, icon: Icon }) => (
                  <Link
                    key={path}
                    to={path}
                    onClick={() => setMobileOpen(false)}
                    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                      location.pathname === path
                        ? 'text-cyan bg-cyan/10'
                        : 'text-muted hover:text-primary'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </Link>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* ── Main Content ───────────────────────────────────── */}
      <main className="pt-16 min-h-screen">
        {children}
      </main>

      {/* ── Medical Disclaimer Footer ───────────────────────── */}
      <footer className="border-t border-border/40 bg-deep/80 py-6 px-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-xs text-muted">
            <Activity className="w-3.5 h-3.5 text-amber" />
            <span className="text-amber font-medium">Research Use Only</span>
            <span>—</span>
            <span>Not a substitute for professional medical diagnosis</span>
          </div>
          <div className="text-xs text-muted font-mono">
            NeuroScan AI v1.0 • EfficientNet-B0 + MobileNetV3
          </div>
        </div>
      </footer>
    </div>
  )
}
