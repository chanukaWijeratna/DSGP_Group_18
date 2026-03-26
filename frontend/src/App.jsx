import { Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import SignUp from './pages/SignUp'
import Record from './pages/Record'
import Report from './pages/Report'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/login" replace />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<SignUp />} />
      <Route path="/record" element={<Record />} />
      <Route path="/report" element={<Report />} />
    </Routes>
  )
}

export default App
