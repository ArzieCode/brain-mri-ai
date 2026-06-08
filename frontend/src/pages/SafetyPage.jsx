import { motion } from 'framer-motion'
import { Shield, ShieldCheck, AlertTriangle, Eye, Activity, Brain, Scale, Layers } from 'lucide-react'

const sections = [
  {
    icon: ShieldCheck,
    color: 'emerald',
    title: 'Multi-Layer Image Validation',
    points: [
      'MobileNetV3-Small classifier detects non-MRI images before they reach the tumor classifier',
      'Supported types: Brain MRI only. X-rays, blood cells, natural photos are explicitly rejected',
      'Image quality checks: blur detection (Laplacian variance), darkness threshold, dimension limits',
      'Aspect ratio validation rejects suspicious image formats',
    ],
  },
  {
    icon: Activity,
    color: 'cyan',
    title: 'Uncertainty Estimation',
    points: [
      'Monte Carlo Dropout: 20 stochastic forward passes with dropout enabled at inference',
      'Epistemic uncertainty quantified as standard deviation across MC passes',
      '95% confidence intervals reported for every prediction',
      'High-uncertainty predictions are explicitly flagged — never reported as confident',
    ],
  },
  {
    icon: Eye,
    color: 'violet',
    title: 'Out-of-Distribution Detection',
    points: [
      'Shannon entropy of softmax distribution measures prediction confidence',
      'High entropy (>1.2) signals OOD — image differs from training distribution',
      'Maximum softmax confidence threshold (60%) prevents low-confidence predictions',
      'OOD images trigger a clear warning instead of a potentially misleading classification',
    ],
  },
  {
    icon: Scale,
    color: 'amber',
    title: 'Bias Reduction Strategies',
    points: [
      'Inverse-frequency class weighting in loss function prevents majority-class dominance',
      'WeightedRandomSampler oversamples minority classes during training',
      'Stratified train/val splits maintain class distribution across folds',
      'Per-class accuracy and F1 tracked separately to detect per-class bias',
    ],
  },
  {
    icon: Brain,
    color: 'rose',
    title: 'Explainable AI (GradCAM)',
    points: [
      'Gradient-weighted Class Activation Maps highlight tumor-relevant regions',
      'Heatmap overlays let clinicians verify AI attention matches expected anatomy',
      'Attention region estimated via center-of-mass of activation map',
      'Three views provided: original MRI, heatmap only, and blended overlay',
    ],
  },
  {
    icon: Layers,
    color: 'cyan',
    title: 'Medical Safety Guarantees',
    points: [
      'System NEVER claims medical certainty — all outputs include uncertainty',
      'Prominent disclaimers on every result: "AI-Assisted Only, Not for Clinical Use"',
      'Predictions below 60% confidence are flagged as uncertain regardless of class',
      'PDF reports include mandatory disclaimer and model version for auditability',
    ],
  },
]

const colorMap = {
  emerald: 'text-emerald bg-emerald/5 border-emerald/20',
  cyan:    'text-cyan bg-cyan/5 border-cyan/20',
  violet:  'text-violet bg-violet/5 border-violet/20',
  amber:   'text-amber bg-amber/5 border-amber/20',
  rose:    'text-rose bg-rose/5 border-rose/20',
}

const container = { hidden: {}, show: { transition: { staggerChildren: 0.07 } } }
const item = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0 } }

export default function SafetyPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-emerald/20 bg-emerald/5 text-emerald text-xs font-mono mb-6">
          <Shield className="w-3 h-3" />
          AI SAFETY FRAMEWORK
        </div>
        <h1 className="font-display text-4xl font-bold text-primary mb-4">
          Responsible AI Design
        </h1>
        <p className="text-subtle text-lg max-w-2xl mx-auto leading-relaxed">
          NeuroScan AI is built around safety-first principles. Every component
          is designed to prevent overconfident or misleading predictions.
        </p>
      </motion.div>

      {/* Primary disclaimer */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="flex items-start gap-4 p-5 rounded-2xl border border-amber/20 bg-amber/5 mb-10">
        <AlertTriangle className="w-6 h-6 text-amber shrink-0 mt-0.5" />
        <div>
          <h3 className="text-amber font-semibold mb-1">Fundamental Limitation</h3>
          <p className="text-amber/80 text-sm leading-relaxed">
            This system is a research prototype, not a medical device. It has not been
            clinically validated, is not FDA/CE approved, and must NEVER be used for actual
            patient diagnosis or treatment decisions. Always consult a board-certified radiologist
            or neurologist for medical interpretation of MRI scans.
          </p>
        </div>
      </motion.div>

      {/* Safety sections */}
      <motion.div variants={container} initial="hidden" animate="show"
        className="grid md:grid-cols-2 gap-5">
        {sections.map(({ icon: Icon, color, title, points }) => (
          <motion.div key={title} variants={item} className="glass-card p-6">
            <div className={`w-10 h-10 rounded-xl border flex items-center justify-center mb-4 ${colorMap[color]}`}>
              <Icon className="w-5 h-5" />
            </div>
            <h3 className="font-display font-semibold text-primary mb-3">{title}</h3>
            <ul className="space-y-2">
              {points.map((p) => (
                <li key={p} className="flex items-start gap-2 text-sm text-subtle">
                  <span className="mt-1.5 w-1 h-1 rounded-full bg-muted shrink-0" />
                  {p}
                </li>
              ))}
            </ul>
          </motion.div>
        ))}
      </motion.div>

      {/* Model info */}
      <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
        className="mt-8 glass-card p-6">
        <h3 className="font-display font-semibold text-primary mb-4">Model Architecture</h3>
        <div className="grid sm:grid-cols-2 gap-4 font-mono text-sm">
          {[
            ['Tumor Classifier',  'EfficientNet-B0 (ImageNet pretrained)'],
            ['Image Validator',   'MobileNetV3-Small (ImageNet pretrained)'],
            ['GradCAM Target',    'EfficientNet features[-1] (last conv block)'],
            ['Uncertainty',       'Monte Carlo Dropout (20 passes)'],
            ['Augmentation',      'Rotation ±10°, H-flip, Brightness/Contrast'],
            ['Optimizer',         'AdamW + Cosine Annealing Warm Restarts'],
            ['Bias Reduction',    'Weighted loss + WeightedRandomSampler'],
            ['Device Support',    'MPS (Apple Silicon) / CUDA / CPU'],
          ].map(([k, v]) => (
            <div key={k} className="data-row">
              <span className="data-label text-xs">{k}</span>
              <span className="data-value text-xs text-subtle">{v}</span>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}
