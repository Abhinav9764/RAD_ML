import { useState, useCallback, useEffect, createContext, useContext } from 'react'

const API = '/api'
export const AuthContext = createContext(null)

export function useAuthProvider() {
  const [user, setUser]       = useState(null)
  const [loading, setLoading] = useState(true)   // checking existing session

  // Check existing JWT on mount
  useEffect(() => {
    const token = localStorage.getItem('radml_token')
    if (!token) { setLoading(false); return }
    fetch(`${API}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => setUser(data.user))
      .catch(() => localStorage.removeItem('radml_token'))
      .finally(() => setLoading(false))
  }, [])

  const _storeToken = (token, userData) => {
    localStorage.setItem('radml_token', token)
    setUser(userData)
  }

  const register = useCallback(async (username, password, email = '') => {
    const res  = await fetch(`${API}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, email }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Registration failed')
    _storeToken(data.token, data.user)
    return data.user
  }, [])

  const login = useCallback(async (username, password) => {
    const res  = await fetch(`${API}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'Login failed')
    _storeToken(data.token, data.user)
    return data.user
  }, [])

  const logout = useCallback(async () => {
    localStorage.removeItem('radml_token')
    setUser(null)
    await fetch(`${API}/auth/logout`, { method: 'POST' }).catch(() => {})
  }, [])

  const getToken = useCallback(() => localStorage.getItem('radml_token'), [])

  return { user, loading, register, login, logout, getToken }
}

export function useAuth() {
  return useContext(AuthContext)
}
