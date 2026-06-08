import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Brain, ShieldCheck, Zap, Eye, ArrowRight, Activity, AlertTriangle } from 'lucide-react'

const features = [
  {
    icon: Brain,
    color: 'cyan',
    title: 'EfficientNet-B0 Classifier',
    desc: 'Transfer learning on ImageNet for accurate glioma, meningioma, pituitary, and normal classification.',
  },
  {
    icon: ShieldCheck,
    color: 'emerald',
    title: 'Multi-Layer Validation',
    desc: 'MobileNetV3 image validator rejects non-MRI uploads, low-quality scans, and out-of-distribution images.',
  },
  {
    icon: Eye,
    color: 'violet',
    title: 'GradCAM Explainability',
    desc: 'Gradient-weighted heatmaps show exactly which brain regions drove the model\'s prediction.',
  },
  {
    icon: Zap,
    color: 'amber',
    title: 'Uncertainty Estimation',
    desc: 'Monte Carlo Dropout runs 20 stochastic passes to quantify epistemic uncertainty and flag uncertain cases.',
  },
]

const stats = [
  { label: 'Tumor Classes', value: '4' },
  { label: 'Validation Layers', value: '5' },
  { label: 'MC Dropout Passes', value: '20' },
  { label: 'Apple Silicon', value: 'MPS' },
]

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
}
const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
}

const colorMap = {
  cyan: 'text-cyan border-cyan/20 bg-cyan/5',
  emerald: 'text-emerald border-emerald/20 bg-emerald/5',
  violet: 'text-violet border-violet/20 bg-violet/5',
  amber: 'text-amber border-amber/20 bg-amber/5',
}

export default function LandingPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-16">
      {/* ── Hero ─────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="text-center mb-20"
      >
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan/20 bg-cyan/5 text-cyan text-xs font-medium font-mono mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan animate-pulse" />
          AI-POWERED NEUROIMAGING RESEARCH PLATFORM
        </div>

        {/* Main heading */}
        <h1 className="font-display text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-none">
          <span className="text-primary">Brain MRI</span>
          <br />
          <span className="gradient-text">Tumor Detection</span>
        </h1>

        <p className="text-subtle text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed">
          Research-grade AI analysis combining explainability, uncertainty estimation,
          and multi-layer safety validation. Built for academic exploration, not clinical diagnosis.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link to="/upload">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex items-center gap-2 px-8 py-4 rounded-xl bg-cyan text-void font-display font-semibold text-sm hover:bg-cyan/90 transition-all"
              style={{ boxShadow: '0 0 30px rgba(0,212,255,0.3)' }}
            >
              <Brain className="w-4 h-4" />
              Analyze MRI Scan
              <ArrowRight className="w-4 h-4" />
            </motion.button>
          </Link>
          <Link to="/safety">
            <button className="flex items-center gap-2 px-8 py-4 rounded-xl border border-border/60 text-subtle hover:text-primary hover:border-border transition-all text-sm font-medium">
              <ShieldCheck className="w-4 h-4" />
              View Safety Framework
            </button>
          </Link>
        </div>
      </motion.div>

      {/* ── Disclaimer Banner ────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="flex items-start gap-3 p-4 rounded-xl border border-amber/20 bg-amber/5 mb-16"
      >
        <AlertTriangle className="w-5 h-5 text-amber shrink-0 mt-0.5" />
        <div>
          <p className="text-amber text-sm font-semibold mb-0.5">Research & Educational Use Only</p>
          <p className="text-amber/70 text-xs leading-relaxed">
            This system is NOT a medical device and does NOT provide medical advice.
            All predictions are for research purposes only. Always consult a qualified
            radiologist or neurologist for medical decisions.
          </p>
        </div>
      </motion.div>

      {/* ── Stats ───────────────────────────────────────── */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16"
      >
        {stats.map(({ label, value }) => (
          <motion.div key={label} variants={item} className="glass-card p-5 text-center">
            <div className="font-display text-3xl font-bold text-cyan mb-1">{value}</div>
            <div className="text-xs text-muted">{label}</div>
          </motion.div>
        ))}
      </motion.div>

      {/* ── Features ────────────────────────────────────── */}
      <motion.div
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true }}
        className="grid md:grid-cols-2 gap-5"
      >
        {features.map(({ icon: Icon, color, title, desc }) => (
          <motion.div key={title} variants={item} className="glass-card-hover p-6">
            <div className={`w-10 h-10 rounded-xl border flex items-center justify-center mb-4 ${colorMap[color]}`}>
              <Icon className="w-5 h-5" />
            </div>
            <h3 className="font-display font-semibold text-primary mb-2">{title}</h3>
            <p className="text-subtle text-sm leading-relaxed">{desc}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* ── Pipeline Flow ───────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="mt-16 glass-card p-8"
      >
        <h2 className="font-display font-semibold text-primary text-xl mb-6 text-center">
          Analysis Pipeline
        </h2>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {[
            'Upload MRI', 'Format Check', 'Quality Assessment',
            'MRI Validation', 'OOD Detection', 'Tumor Classification',
            'GradCAM', 'Uncertainty Est.', 'Final Report'
          ].map((step, i) => (
            <div key={step} className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-border/50 text-xs text-subtle font-mono">
                <span className="text-cyan">{String(i + 1).padStart(2, '0')}</span>
                {step}
              </div>
              {i < 8 && <Activity className="w-3 h-3 text-border shrink-0" />}
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
