import { Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Layout from './components/layout/Layout'
import LandingPage from './pages/LandingPage'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'
import HistoryPage from './pages/HistoryPage'
import SafetyPage from './pages/SafetyPage'

export default function App() {
  return (
    <Layout>
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/"          element={<LandingPage />} />
          <Route path="/upload"    element={<UploadPage />} />
          <Route path="/results/:reportId" element={<ResultsPage />} />
          <Route path="/history"   element={<HistoryPage />} />
          <Route path="/safety"    element={<SafetyPage />} />
        </Routes>
      </AnimatePresence>
    </Layout>
  )
}
