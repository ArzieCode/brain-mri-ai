/**
 * API Service
 * Centralized axios client for all backend communication.
 */

import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 2 min for inference
})

// Request interceptor for logging
api.interceptors.request.use(config => {
  console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
  return config
})

// Response error handler
api.interceptors.response.use(
  res => res,
  err => {
    const msg = err.response?.data?.detail || err.message || 'Unknown error'
    console.error(`[API Error] ${msg}`)
    return Promise.reject(new Error(msg))
  }
)

// ── Upload ─────────────────────────────────────────────────
export const uploadImage = async (file, onProgress) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/upload/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress?.(Math.round((e.loaded * 100) / e.total)),
  })
  return res.data
}

// ── Full Analysis ──────────────────────────────────────────
export const analyzeImage = async (fileId) => {
  const res = await api.post(`/predict/${fileId}`)
  return res.data
}

// ── Validate Only ──────────────────────────────────────────
export const validateImage = async (fileId) => {
  const res = await api.post(`/validate/${fileId}`)
  return res.data
}

// ── GradCAM ────────────────────────────────────────────────
export const getGradCAM = async (fileId, classIdx = 0) => {
  const res = await api.post(`/gradcam/${fileId}?class_idx=${classIdx}`)
  return res.data
}

// ── Reports ────────────────────────────────────────────────
export const getReports = async () => {
  const res = await api.get('/reports/')
  return res.data
}

export const getReport = async (reportId) => {
  const res = await api.get(`/reports/${reportId}`)
  return res.data
}

export const deleteReport = async (reportId) => {
  await api.delete(`/reports/${reportId}`)
}

export const downloadReportPdf = async (reportId) => {
  const res = await api.get(`/reports/${reportId}/pdf`, { responseType: 'blob' })
  const url = window.URL.createObjectURL(new Blob([res.data]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', `mri_report_${reportId}.pdf`)
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
}

// ── Health ─────────────────────────────────────────────────
export const checkHealth = async () => {
  const res = await api.get('/health')
  return res.data
}

export default api
