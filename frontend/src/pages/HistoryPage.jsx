import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Clock, Brain, Trash2, ExternalLink, Download, AlertTriangle } from 'lucide-react'
import { getReports, deleteReport, downloadReportPdf } from '../utils/api'

const CLASS_COLORS = {
  glioma: '#ff4466', meningioma: '#ffb020', pituitary: '#8855ff', normal: '#00ff88',
}
const RISK_COLORS = { low: 'text-emerald', moderate: 'text-amber', high: 'text-rose' }

export default function HistoryPage() {
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)

  const load = () => getReports().then(setReports).catch(() => setReports([])).finally(() => setLoading(false))

  useEffect(() => { load() }, [])

  const handleDelete = async (id) => {
    await deleteReport(id).catch(() => {})
    setReports(prev => prev.filter(r => r.report_id !== id))
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-display text-2xl font-bold text-primary mb-1">Analysis History</h1>
            <p className="text-muted text-sm">{reports.length} report{reports.length !== 1 ? 's' : ''} stored in this session</p>
          </div>
          <Link to="/upload" className="btn-primary text-xs flex items-center gap-2">
            <Brain className="w-3.5 h-3.5" /> New Analysis
          </Link>
        </div>
      </motion.div>

      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => <div key={i} className="skeleton h-20 rounded-xl" />)}
        </div>
      ) : reports.length === 0 ? (
        <div className="glass-card p-16 text-center">
          <Clock className="w-12 h-12 text-muted/30 mx-auto mb-4" />
          <p className="text-muted">No reports yet. Upload an MRI scan to get started.</p>
          <Link to="/upload" className="mt-4 inline-block btn-primary text-xs">Upload MRI</Link>
        </div>
      ) : (
        <div className="space-y-3">
          <AnimatePresence>
            {reports.map((r, i) => (
              <motion.div
                key={r.report_id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ delay: i * 0.05 }}
                className="glass-card p-4 flex items-center gap-4"
              >
                {/* Color indicator */}
                <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                  style={{ background: r.prediction ? `${CLASS_COLORS[r.prediction.prediction]}15` : '#1a2236' }}>
                  <Brain className="w-4 h-4" style={{ color: r.prediction ? CLASS_COLORS[r.prediction.prediction] : '#64748b' }} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-xs font-mono text-cyan">{r.report_id}</span>
                    {r.prediction ? (
                      <span className="text-xs font-semibold capitalize" style={{ color: CLASS_COLORS[r.prediction.prediction] }}>
                        {r.prediction.prediction}
                      </span>
                    ) : (
                      <span className="text-xs text-rose">Rejected</span>
                    )}
                    {r.prediction && (
                      <span className={`text-xs ${RISK_COLORS[r.prediction.risk_level]}`}>
                        {r.prediction.risk_level} risk
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted truncate">{r.image_filename}</p>
                  <p className="text-[10px] text-muted/50 font-mono mt-0.5">
                    {new Date(r.timestamp).toLocaleString()}
                  </p>
                </div>

                {r.prediction && (
                  <div className="text-right shrink-0">
                    <p className="text-sm font-semibold text-primary">{(r.prediction.confidence * 100).toFixed(1)}%</p>
                    <p className="text-[10px] text-muted">confidence</p>
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center gap-1 shrink-0">
                  <Link to={`/results/${r.report_id}`}
                    className="w-7 h-7 rounded-lg border border-border/40 flex items-center justify-center text-muted hover:text-cyan transition-colors">
                    <ExternalLink className="w-3 h-3" />
                  </Link>
                  <button onClick={() => downloadReportPdf(r.report_id)}
                    className="w-7 h-7 rounded-lg border border-border/40 flex items-center justify-center text-muted hover:text-emerald transition-colors">
                    <Download className="w-3 h-3" />
                  </button>
                  <button onClick={() => handleDelete(r.report_id)}
                    className="w-7 h-7 rounded-lg border border-border/40 flex items-center justify-center text-muted hover:text-rose transition-colors">
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
