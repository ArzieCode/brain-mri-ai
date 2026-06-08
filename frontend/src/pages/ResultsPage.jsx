import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { RadialBarChart, RadialBar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'
import { Brain, Download, ArrowLeft, AlertTriangle, CheckCircle, XCircle, Activity, Eye, Shield } from 'lucide-react'
import { getReport, downloadReportPdf } from '../utils/api'

const CLASS_COLORS = {
  glioma: '#ff4466',
  meningioma: '#ffb020',
  pituitary: '#8855ff',
  normal: '#00ff88',
}

const RISK_STYLES = {
  low:      { text: 'text-emerald', bg: 'bg-emerald/10', border: 'border-emerald/30', label: 'LOW RISK' },
  moderate: { text: 'text-amber',   bg: 'bg-amber/10',   border: 'border-amber/30',   label: 'MODERATE' },
  high:     { text: 'text-rose',    bg: 'bg-rose/10',    border: 'border-rose/30',    label: 'HIGH RISK' },
}

const UNC_STYLES = {
  low:      { text: 'text-emerald', label: 'LOW' },
  moderate: { text: 'text-amber',   label: 'MODERATE' },
  high:     { text: 'text-rose',    label: 'HIGH' },
}

const fadeUp = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0 } }

export default function ResultsPage() {
  const { reportId } = useParams()
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeImage, setActiveImage] = useState('overlay')

  useEffect(() => {
    getReport(reportId)
      .then(setReport)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [reportId])

  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-cyan/30 border-t-cyan rounded-full animate-spin mx-auto mb-4" />
        <p className="text-muted text-sm">Loading report...</p>
      </div>
    </div>
  )

  if (error || !report) return (
    <div className="max-w-xl mx-auto px-4 py-20 text-center">
      <XCircle className="w-12 h-12 text-rose mx-auto mb-4" />
      <h2 className="font-display text-xl font-semibold text-primary mb-2">Report Not Found</h2>
      <p className="text-muted text-sm mb-6">{error || 'The report could not be loaded.'}</p>
      <Link to="/upload" className="btn-primary">New Analysis</Link>
    </div>
  )

  const { validation, prediction, gradcam } = report

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
      {/* Header */}
      <motion.div initial="hidden" animate="show" variants={fadeUp} className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <Link to="/upload" className="w-8 h-8 rounded-lg border border-border/50 flex items-center justify-center text-muted hover:text-primary transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-xs font-mono text-muted">REPORT</span>
              <span className="text-xs font-mono text-cyan">{report.report_id}</span>
            </div>
            <h1 className="font-display font-bold text-xl text-primary">Analysis Results</h1>
          </div>
        </div>
        <button
          onClick={() => downloadReportPdf(report.report_id)}
          className="btn-primary flex items-center gap-2 text-xs"
        >
          <Download className="w-3.5 h-3.5" />
          Export PDF
        </button>
      </motion.div>

      {/* ── Validation rejected ─────────────────────────── */}
      {!validation.is_valid_brain_mri && (
        <motion.div initial="hidden" animate="show" variants={fadeUp}
          className="glass-card p-8 text-center border-rose/20">
          <XCircle className="w-14 h-14 text-rose mx-auto mb-4" />
          <h2 className="font-display text-2xl font-bold text-primary mb-3">Image Rejected</h2>
          <p className="text-rose/80 mb-4">{validation.rejection_message}</p>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-rose/10 border border-rose/20 text-rose text-sm">
            Status: {validation.status.replace(/_/g, ' ').toUpperCase()}
          </div>
          <div className="mt-6">
            <Link to="/upload" className="btn-primary">Upload Different Scan</Link>
          </div>
        </motion.div>
      )}

      {/* ── Full results ────────────────────────────────── */}
      {validation.is_valid_brain_mri && prediction && (
        <motion.div initial="hidden" animate="show" variants={{ hidden: {}, show: { transition: { staggerChildren: 0.08 } } }}>

          {/* ── Top row: Prediction + GradCAM ──────────── */}
          <div className="grid lg:grid-cols-5 gap-5 mb-5">

            {/* Prediction Card */}
            <motion.div variants={fadeUp} className="lg:col-span-2 glass-card p-6 flex flex-col gap-5">
              {/* Risk badge */}
              {(() => {
                const s = RISK_STYLES[prediction.risk_level]
                return (
                  <div className={`self-start flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold font-mono border ${s.text} ${s.bg} ${s.border}`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${s.text === 'text-emerald' ? 'bg-emerald' : s.text === 'text-amber' ? 'bg-amber' : 'bg-rose'} animate-pulse`} />
                    {s.label}
                  </div>
                )
              })()}

              {/* Main prediction */}
              <div>
                <p className="text-xs text-muted font-mono mb-1">PREDICTION</p>
                <h2 className="font-display text-4xl font-bold capitalize mb-1"
                  style={{ color: CLASS_COLORS[prediction.prediction] }}>
                  {prediction.prediction}
                </h2>
                <p className="text-muted text-sm">Brain tumor classification</p>
              </div>

              {/* Confidence gauge */}
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-muted font-mono">CONFIDENCE</span>
                  <span className="text-primary font-semibold">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-border rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${prediction.confidence * 100}%` }}
                    transition={{ duration: 1, ease: 'easeOut' }}
                    className="h-full rounded-full"
                    style={{ background: CLASS_COLORS[prediction.prediction] }}
                  />
                </div>
              </div>

              {/* Uncertainty */}
              {(() => {
                const u = prediction.uncertainty
                const s = UNC_STYLES[u.uncertainty_level]
                return (
                  <div className="space-y-2">
                    <div className="data-row">
                      <span className="data-label">Uncertainty</span>
                      <span className={`text-sm font-semibold ${s.text}`}>{s.label}</span>
                    </div>
                    <div className="data-row">
                      <span className="data-label">Std Dev (±)</span>
                      <span className="data-value font-mono">{(u.std_confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="data-row">
                      <span className="data-label">95% CI</span>
                      <span className="data-value font-mono text-xs">
                        [{(u.confidence_interval_low * 100).toFixed(0)}% – {(u.confidence_interval_high * 100).toFixed(0)}%]
                      </span>
                    </div>
                    <div className="data-row">
                      <span className="data-label">MC Passes</span>
                      <span className="data-value font-mono">{u.mc_passes}</span>
                    </div>
                    <div className="data-row">
                      <span className="data-label">Inference</span>
                      <span className="data-value font-mono">{prediction.inference_time_ms.toFixed(0)} ms</span>
                    </div>
                  </div>
                )
              })()}

              {/* OOD */}
              {!prediction.ood_result.is_in_distribution && (
                <div className="flex items-start gap-2 p-3 rounded-xl bg-amber/5 border border-amber/20">
                  <AlertTriangle className="w-4 h-4 text-amber shrink-0 mt-0.5" />
                  <p className="text-amber/80 text-xs">{prediction.ood_result.ood_message}</p>
                </div>
              )}
            </motion.div>

            {/* GradCAM Viewer */}
            {gradcam && (
              <motion.div variants={fadeUp} className="lg:col-span-3 glass-card overflow-hidden">
                {/* Image tabs */}
                <div className="flex gap-1 p-3 border-b border-border/40">
                  {[
                    { key: 'overlay', label: 'Overlay' },
                    { key: 'heatmap', label: 'Heatmap' },
                    { key: 'original', label: 'Original' },
                  ].map(({ key, label }) => (
                    <button
                      key={key}
                      onClick={() => setActiveImage(key)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        activeImage === key ? 'bg-cyan/10 text-cyan border border-cyan/30' : 'text-muted hover:text-primary'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                  <div className="ml-auto flex items-center gap-1.5 text-xs text-muted">
                    <Eye className="w-3 h-3" />
                    <span>{gradcam.attention_region}</span>
                  </div>
                </div>

                {/* Image display */}
                <div className="bg-black relative h-64">
                  <img
                    src={activeImage === 'overlay' ? gradcam.overlay_url :
                         activeImage === 'heatmap' ? gradcam.heatmap_url :
                         gradcam.original_image_url}
                    alt={activeImage}
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-void/80 p-3">
                    <span className="text-xs font-mono text-cyan/80">GradCAM — {activeImage.toUpperCase()}</span>
                  </div>
                </div>

                {/* Explanation */}
                <div className="p-4">
                  <p className="text-subtle text-xs leading-relaxed">{gradcam.explanation}</p>
                </div>
              </motion.div>
            )}
          </div>

          {/* ── Class Probabilities Chart ────────────────── */}
          <motion.div variants={fadeUp} className="glass-card p-6 mb-5">
            <h3 className="font-display font-semibold text-primary mb-5 flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan" />
              Class Probability Distribution
            </h3>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={prediction.class_probabilities} layout="vertical" margin={{ left: 20, right: 40 }}>
                <XAxis type="number" domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                  tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="class_name" tick={{ fontSize: 11, fill: '#94a3b8', textTransform: 'capitalize' }}
                  axisLine={false} tickLine={false} width={80} />
                <Tooltip
                  formatter={(v) => [`${(v * 100).toFixed(2)}%`, 'Probability']}
                  contentStyle={{ background: '#111827', border: '1px solid #1e2d45', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
                  {prediction.class_probabilities.map((entry) => (
                    <Cell key={entry.class_name} fill={CLASS_COLORS[entry.class_name]} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* ── Clinical Notes ───────────────────────────── */}
          <motion.div variants={fadeUp} className="glass-card p-6 mb-5">
            <h3 className="font-display font-semibold text-primary mb-3 flex items-center gap-2">
              <Brain className="w-4 h-4 text-violet" />
              Clinical Context
              <span className="text-xs text-muted font-normal ml-1">(Informational only)</span>
            </h3>
            <p className="text-subtle text-sm leading-relaxed">{prediction.clinical_notes}</p>
          </motion.div>

          {/* ── Disclaimer ───────────────────────────────── */}
          <motion.div variants={fadeUp}
            className="flex items-start gap-3 p-4 rounded-xl border border-amber/20 bg-amber/5">
            <Shield className="w-5 h-5 text-amber shrink-0 mt-0.5" />
            <div>
              <p className="text-amber text-sm font-semibold mb-0.5">Medical Disclaimer</p>
              <p className="text-amber/70 text-xs leading-relaxed">{report.disclaimer}</p>
            </div>
          </motion.div>

        </motion.div>
      )}
    </div>
  )
}
