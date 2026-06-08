import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Brain, AlertTriangle, CheckCircle, Loader2, FileImage, X } from 'lucide-react'
import { uploadImage, analyzeImage } from '../utils/api'

const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/bmp': ['.bmp'],
  'image/tiff': ['.tiff', '.tif'],
}

const STEPS = [
  { id: 'upload',    label: 'Uploading image...' },
  { id: 'validate',  label: 'Validating MRI...' },
  { id: 'classify',  label: 'Running classifier...' },
  { id: 'gradcam',   label: 'Generating GradCAM...' },
  { id: 'report',    label: 'Building report...' },
]

export default function UploadPage() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [status, setStatus] = useState('idle') // idle | processing | error | done
  const [currentStep, setCurrentStep] = useState(0)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState(null)

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return
    const f = accepted[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setError(null)
    setStatus('idle')
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
    onDropRejected: (files) => {
      const err = files[0]?.errors[0]
      setError(err?.code === 'file-too-large'
        ? 'File too large. Maximum size is 10MB.'
        : 'Invalid file type. Accepted: JPG, PNG, BMP, TIFF'
      )
    },
  })

  const handleAnalyze = async () => {
    if (!file) return
    setStatus('processing')
    setError(null)
    setCurrentStep(0)

    try {
      // Step 1: Upload
      const { file_id } = await uploadImage(file, (pct) => setUploadProgress(pct))
      setCurrentStep(1)

      // Steps 2-5 happen server-side during analysis
      // We simulate step advances while waiting
      const timer = setInterval(() => {
        setCurrentStep(prev => Math.min(prev + 1, STEPS.length - 1))
      }, 1200)

      const report = await analyzeImage(file_id)
      clearInterval(timer)
      setCurrentStep(STEPS.length - 1)
      setStatus('done')

      // Navigate to results
      setTimeout(() => navigate(`/results/${report.report_id}`), 400)
    } catch (err) {
      setStatus('error')
      setError(err.message || 'Analysis failed. Please try again.')
    }
  }

  const clearFile = () => {
    setFile(null)
    setPreview(null)
    setStatus('idle')
    setError(null)
    setCurrentStep(0)
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan/20 bg-cyan/5 text-cyan text-xs font-mono mb-4">
          <Brain className="w-3 h-3" />
          MRI ANALYSIS
        </div>
        <h1 className="font-display text-3xl font-bold text-primary mb-2">Upload Brain MRI</h1>
        <p className="text-muted text-sm">
          Upload an axial, sagittal, or coronal brain MRI scan for AI analysis.
        </p>
      </motion.div>

      <div className="grid md:grid-cols-5 gap-6">
        {/* ── Drop Zone ─────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="md:col-span-3"
        >
          {!file ? (
            <div
              {...getRootProps()}
              className={`glass-card border-2 border-dashed cursor-pointer h-72 flex flex-col items-center justify-center gap-4 transition-all duration-300 ${
                isDragActive ? 'dropzone-active' : 'border-border/40 hover:border-cyan/30'
              }`}
            >
              <input {...getInputProps()} />
              <div className="w-16 h-16 rounded-2xl bg-cyan/5 border border-cyan/20 flex items-center justify-center">
                <Upload className={`w-7 h-7 ${isDragActive ? 'text-cyan' : 'text-muted'}`} />
              </div>
              <div className="text-center">
                <p className="text-primary font-medium mb-1">
                  {isDragActive ? 'Drop MRI scan here' : 'Drag & drop MRI scan'}
                </p>
                <p className="text-muted text-sm">or click to browse</p>
                <p className="text-muted/60 text-xs mt-2">JPG, PNG, BMP, TIFF • Max 10MB</p>
              </div>
            </div>
          ) : (
            <div className="glass-card overflow-hidden">
              {/* Preview */}
              <div className="relative bg-black">
                <img
                  src={preview}
                  alt="MRI preview"
                  className="w-full h-64 object-contain"
                />
                {/* Scan line effect */}
                <div className="absolute inset-0 scan-container pointer-events-none" />
                <button
                  onClick={clearFile}
                  className="absolute top-3 right-3 w-7 h-7 rounded-lg bg-void/80 border border-border/50 flex items-center justify-center text-muted hover:text-rose transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
              {/* File info */}
              <div className="p-4 flex items-center gap-3">
                <FileImage className="w-4 h-4 text-cyan shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-primary truncate">{file.name}</p>
                  <p className="text-xs text-muted">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <span className="badge-valid">Ready</span>
              </div>
            </div>
          )}

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="mt-3 flex items-start gap-2.5 p-3 rounded-xl border border-rose/20 bg-rose/5"
              >
                <AlertTriangle className="w-4 h-4 text-rose shrink-0 mt-0.5" />
                <p className="text-rose text-sm">{error}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ── Analysis Status + CTA ──────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15 }}
          className="md:col-span-2 flex flex-col gap-4"
        >
          {/* Processing Steps */}
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-primary mb-4 font-display">Analysis Pipeline</h3>
            <div className="space-y-3">
              {STEPS.map((step, idx) => {
                const isDone = status === 'processing' && idx < currentStep
                const isActive = status === 'processing' && idx === currentStep
                const isPending = status !== 'processing' || idx > currentStep

                return (
                  <div key={step.id} className="flex items-center gap-3">
                    <div className={`w-5 h-5 rounded-full border flex items-center justify-center shrink-0 transition-all ${
                      isDone ? 'bg-emerald/20 border-emerald/40 text-emerald' :
                      isActive ? 'bg-cyan/20 border-cyan/40 text-cyan' :
                      'border-border/40 text-muted/30'
                    }`}>
                      {isDone ? (
                        <CheckCircle className="w-3 h-3" />
                      ) : isActive ? (
                        <Loader2 className="w-2.5 h-2.5 animate-spin" />
                      ) : (
                        <span className="text-[9px] font-mono">{idx + 1}</span>
                      )}
                    </div>
                    <span className={`text-xs transition-colors ${
                      isDone ? 'text-emerald' :
                      isActive ? 'text-cyan' :
                      'text-muted/50'
                    }`}>
                      {step.label}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Upload progress bar */}
          {status === 'processing' && currentStep === 0 && (
            <div className="glass-card p-4">
              <div className="flex justify-between text-xs text-muted mb-2">
                <span>Uploading</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="h-1 bg-border rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-cyan rounded-full"
                  animate={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Analyze Button */}
          <motion.button
            onClick={handleAnalyze}
            disabled={!file || status === 'processing'}
            whileHover={file && status !== 'processing' ? { scale: 1.02 } : {}}
            whileTap={file && status !== 'processing' ? { scale: 0.98 } : {}}
            className={`w-full py-4 rounded-xl font-display font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
              !file || status === 'processing'
                ? 'bg-surface border border-border/30 text-muted cursor-not-allowed'
                : 'bg-cyan text-void hover:bg-cyan/90'
            }`}
            style={file && status !== 'processing' ? { boxShadow: '0 0 25px rgba(0,212,255,0.25)' } : {}}
          >
            {status === 'processing' ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
            ) : (
              <><Brain className="w-4 h-4" /> Run Analysis</>
            )}
          </motion.button>

          {/* Disclaimer */}
          <div className="p-3 rounded-xl border border-amber/20 bg-amber/5">
            <p className="text-amber/80 text-xs leading-relaxed">
              <span className="font-semibold text-amber">⚠ Disclaimer: </span>
              AI predictions are for research use only. Not for clinical decision-making.
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
