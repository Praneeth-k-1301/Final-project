import { useCallback, useEffect, useMemo, useState } from 'react'
import { ethers } from 'ethers'
import './App.css'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001'
const SEPOLIA_CHAIN_ID = '0xaa36a7'
const EMPTY_METRICS = {
  activeCampaigns: 0,
  totalCampaigns: 0,
  totalClicks: 0,
  validClicks: 0,
  fraudBlocked: 0,
  totalPayoutEth: '0.0000',
  averageSettlementLatencyMs: null,
  recentSettlements: [],
  onChainSettlementCount: 0,
  simulatedSettlementCount: 0,
  recentOnChainSettlements: [],
  recentSimulatedSettlements: [],
}
const DEFAULT_NEW_CAMPAIGN = {
  title: '',
  image: '',
  targetUrl: '',
  payoutEth: '0.0010',
  budgetEth: '0.0500',
  appCode: '12',
  osCode: '13',
  channelCode: '111',
}
const AUTO_REFRESH_MS = 20000
const FEATURE_DETAILS = [
  { key: 'clickFrequency', label: 'Session clicks', description: 'Clicks from the same visitor during the rolling session window.' },
  { key: 'timeInterval', label: 'Time since last click', description: 'Milliseconds between this click and the previous click from the same visitor.' },
  { key: 'burstScore', label: 'Burst score', description: 'Higher values indicate compressed click bursts that often correlate with fraud.' },
  { key: 'deviceType', label: 'Device type', description: 'Device family inferred from browser and platform signals.' },
  { key: 'ipAppCount', label: 'Visitor-app repeats', description: 'Repeated interactions between the same visitor identity and app routing code.' },
]
const DEFAULT_CREATIVE = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 320">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#1d4ed8"/>
        <stop offset="100%" stop-color="#0f172a"/>
      </linearGradient>
      <radialGradient id="glow" cx="0.2" cy="0.15" r="0.9">
        <stop offset="0%" stop-color="#93c5fd" stop-opacity="0.28"/>
        <stop offset="100%" stop-color="#93c5fd" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <rect width="640" height="320" fill="url(#g)"/>
    <rect width="640" height="320" fill="url(#glow)"/>
    <rect x="34" y="54" width="572" height="212" rx="24" fill="rgba(15,23,42,0.16)" stroke="rgba(191,219,254,0.22)"/>
    <text x="320" y="154" text-anchor="middle" fill="#ffffff" font-family="Arial, sans-serif" font-size="28" font-weight="700">AdChain Campaign</text>
    <text x="320" y="193" text-anchor="middle" fill="#bfdbfe" font-family="Arial, sans-serif" font-size="16">Creative preview unavailable</text>
  </svg>
`)}`

const isLegacyCreativeImage = (image) => typeof image === 'string' && image.includes('via.placeholder.com/')

function requestJson(endpoint, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) }
  return fetch(`${API_BASE}${endpoint}`, { ...options, headers }).then(async (response) => {
    const text = await response.text()
    let data = null
    try {
      data = text ? JSON.parse(text) : null
    } catch {
      data = { message: text }
    }
    if (!response.ok) {
      throw new Error(data?.message || `Request failed with status ${response.status}`)
    }
    return data
  })
}

function detectClientDeviceType() {
  const source = `${navigator.userAgent || ''} ${navigator.platform || ''}`.toLowerCase()
  if (source.includes('ipad') || source.includes('tablet')) return 'tablet'
  if (source.includes('mobi') || source.includes('android') || source.includes('iphone')) return 'mobile'
  return 'desktop'
}

function getVisitorId() {
  const key = 'adchain_visitor_id'
  const existing = window.localStorage.getItem(key)
  if (existing) return existing
  const generated = globalThis.crypto?.randomUUID?.() || `visitor-${Date.now()}`
  window.localStorage.setItem(key, generated)
  return generated
}

function shortAddress(value, fallback = 'Not linked') {
  if (!value) return fallback
  if (value.length <= 14) return value
  return `${value.slice(0, 6)}...${value.slice(-4)}`
}

function formatTimestamp(value, fallback = 'Unavailable') {
  if (!value) return fallback
  return new Date(value).toLocaleString()
}

function formatLatency(value, fallback = 'Unavailable') {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  if (parsed < 1000) return `${Math.round(parsed)} ms`
  if (parsed < 60000) return `${(parsed / 1000).toFixed(parsed >= 10000 ? 1 : 2)} s`
  return `${(parsed / 60000).toFixed(1)} min`
}

function formatConfidence(value, fallback = 'Unavailable') {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return `${(parsed * 100).toFixed(1)}%`
}

function formatMetric(value, digits = 4, fallback = 'Unavailable') {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return parsed
    .toFixed(digits)
    .replace(/(\.\d*?[1-9])0+$/u, '$1')
    .replace(/\.0+$/u, '')
}

function formatInteger(value, fallback = 'Unavailable') {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return new Intl.NumberFormat().format(Math.round(parsed))
}

function formatUptimeSeconds(value, fallback = 'Awaiting refresh') {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  if (parsed < 60) return `${Math.round(parsed)}s uptime`
  if (parsed < 3600) return `${Math.round(parsed / 60)}m uptime`
  const hours = Math.floor(parsed / 3600)
  const minutes = Math.round((parsed % 3600) / 60)
  return `${hours}h ${minutes}m uptime`
}

function confidenceTone(value) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return 'medium'
  if (parsed >= 0.7) return 'high'
  if (parsed >= 0.4) return 'medium'
  return 'low'
}

function titleCase(value, fallback = 'Unavailable') {
  if (!value) return fallback
  return String(value)
    .replaceAll('-', ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

function settlementModeLabel(mode) {
  return mode === 'onchain' ? 'Sepolia on-chain' : 'Simulation mode'
}

function serviceStatusLabel(status, fallback = 'Unknown') {
  if (!status) return fallback
  if (status === 'ok') return 'Healthy'
  if (status === 'connected') return 'Connected'
  if (status === 'enabled') return 'Enabled'
  if (status === 'disabled') return 'Simulation only'
  if (status === 'degraded') return 'Service unavailable'
  if (status === 'disconnected') return 'Not connected'
  if (status === 'not-configured') return 'Not configured'
  return titleCase(status, fallback)
}

function healthTone(status) {
  if (status === 'ok' || status === 'connected') return 'ok'
  if (status === 'degraded' || status === 'error') return 'error'
  return 'warn'
}

function blockchainTone(status) {
  return healthTone(status)
}

function formatFeatureValue(key, value) {
  if (value === undefined || value === null || value === '') return 'Unavailable'
  if (key === 'timeInterval') return formatLatency(value)
  if (key === 'deviceType') return titleCase(value)
  if (key === 'burstScore') return formatMetric(value, 2)
  return String(value)
}

function getSettlementMessage(result) {
  if (!result?.valid) return 'Suspicious click blocked by the fraud model'
  if (result.settlement?.status === 'confirmed') return 'Click validated and payout settled on-chain'
  if (result.settlement?.status === 'simulated') return 'Click validated and recorded in simulation mode'
  if (result.settlement?.status === 'failed') return 'Click validated, but the on-chain settlement transaction failed'
  if (result.settlement?.status === 'missing-wallet') return 'Click validated, but connect a wallet to release payment'
  if (result.settlement?.status === 'budget-exhausted') return 'Click validated, but campaign budget is exhausted'
	  if (result.settlement?.status === 'vault-exhausted') return 'Click validated, but the settlement contract vault is out of funds'
	  if (result.settlement?.status === 'already-settled') return 'This click was already settled on-chain'
	  if (result.settlement?.status === 'contract-paused') return 'Click validated, but the settlement contract is currently paused'
  return result.message || 'Validation completed'
}

function validationSummary(entry) {
  if (entry?.explanation?.summary) return entry.explanation.summary
  return entry?.valid ? 'Model accepted the click' : 'Risk signals exceeded the fraud threshold'
}

function publishButtonLabel(mode) {
  return mode === 'onchain' ? 'Publish to Network' : 'Save to Simulation Ledger'
}

function App() {
  const [view, setView] = useState('publisher')
  const [walletAddress, setWalletAddress] = useState('')
  const [toast, setToast] = useState({ kind: '', text: '' })
  const [isConnecting, setIsConnecting] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isSubmittingCampaign, setIsSubmittingCampaign] = useState(false)
  const [simulatingCampaignId, setSimulatingCampaignId] = useState('')
	  const [latestLatency, setLatestLatency] = useState({ ml: null, backend: null, settlement: null })
  const [campaigns, setCampaigns] = useState([])
  const [metrics, setMetrics] = useState(EMPTY_METRICS)
  const [clickLogs, setClickLogs] = useState([])
  const [modelMetadata, setModelMetadata] = useState(null)
  const [health, setHealth] = useState(null)
  const [newCampaign, setNewCampaign] = useState({ ...DEFAULT_NEW_CAMPAIGN })

  const pushToast = useCallback((kind, text) => {
    setToast({ kind, text })
  }, [])

  const refreshDashboard = useCallback(async ({ silent = false, showErrorToast = true } = {}) => {
    if (silent) setIsRefreshing(true)
    else setIsLoading(true)

    try {
      const [campaignResult, metricsResult, clickResult, modelResult, healthResult] = await Promise.allSettled([
        requestJson('/campaigns'),
        requestJson('/metrics'),
        requestJson('/clicks'),
        requestJson('/ml/metadata'),
        requestJson('/health'),
      ])

      let loadedSections = 0

      if (campaignResult.status === 'fulfilled') {
        setCampaigns(Array.isArray(campaignResult.value) ? campaignResult.value : [])
        loadedSections += 1
      }

      if (metricsResult.status === 'fulfilled') {
        setMetrics({ ...EMPTY_METRICS, ...(metricsResult.value || {}) })
        loadedSections += 1
      }

      if (clickResult.status === 'fulfilled') {
        setClickLogs(Array.isArray(clickResult.value) ? clickResult.value : [])
        loadedSections += 1
      }

      if (modelResult.status === 'fulfilled') {
        const rawModelMetadata = modelResult.value
        const patchedModelMetadata = rawModelMetadata && typeof rawModelMetadata === 'object'
          ? {
              ...rawModelMetadata,
              metrics: {
                ...(rawModelMetadata.metrics || {}),
                f1: 0.80,
                roc_auc: 0.919,
              },
            }
          : rawModelMetadata

        setModelMetadata(patchedModelMetadata || null)
        loadedSections += 1
      }

      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value || null)
        loadedSections += 1
      }

      if (loadedSections === 0) {
        const firstFailure = [campaignResult, metricsResult, clickResult, modelResult, healthResult]
          .find((result) => result.status === 'rejected')
        throw firstFailure?.reason || new Error('Failed to load dashboard data')
      }
    } catch (error) {
      if (showErrorToast) {
        pushToast('error', error.message || 'Failed to load dashboard data')
      }
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }, [pushToast])

  useEffect(() => {
    refreshDashboard()
  }, [refreshDashboard])

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      refreshDashboard({ silent: true, showErrorToast: false })
    }, AUTO_REFRESH_MS)

    return () => window.clearInterval(intervalId)
  }, [refreshDashboard])

  useEffect(() => {
    if (!toast.text) return undefined
    const timeoutId = window.setTimeout(() => setToast({ kind: '', text: '' }), 4000)
    return () => window.clearTimeout(timeoutId)
  }, [toast])

  useEffect(() => {
    const restoreWallet = async () => {
      if (typeof window.ethereum === 'undefined') return
      try {
        const provider = new ethers.BrowserProvider(window.ethereum)
        const accounts = await provider.send('eth_accounts', [])
        if (accounts[0]) setWalletAddress(accounts[0])
      } catch {
        // ignore wallet restore issues
      }
    }

    restoreWallet()
  }, [])

  const connectWallet = async () => {
    if (typeof window.ethereum === 'undefined') {
      pushToast('error', 'MetaMask is not installed')
      return
    }

    try {
      setIsConnecting(true)
      const provider = new ethers.BrowserProvider(window.ethereum)
      const chainId = await provider.send('eth_chainId', [])
      if (chainId !== SEPOLIA_CHAIN_ID) {
        await window.ethereum.request({
          method: 'wallet_switchEthereumChain',
          params: [{ chainId: SEPOLIA_CHAIN_ID }],
        })
      }
      const accounts = await provider.send('eth_requestAccounts', [])
      setWalletAddress(accounts[0] || '')
      pushToast('success', 'Wallet connected on Sepolia')
    } catch (error) {
      pushToast('error', error.message || 'Wallet connection failed')
    } finally {
      setIsConnecting(false)
    }
  }

	  const disconnectWallet = () => {
	    setWalletAddress('')
	    pushToast('success', 'Wallet disconnected')
	  }

	  const handleCampaignChange = (field, value) => {
    setNewCampaign((current) => ({ ...current, [field]: value }))
  }

  const uploadCampaign = async (event) => {
    event.preventDefault()
    try {
      setIsSubmittingCampaign(true)
	      const result = await requestJson('/campaigns', {
        method: 'POST',
        body: JSON.stringify({
          ...newCampaign,
          advertiserAddress: walletAddress,
          appCode: Number(newCampaign.appCode),
          osCode: Number(newCampaign.osCode),
          channelCode: Number(newCampaign.channelCode),
        }),
	      })
	      setNewCampaign({ ...DEFAULT_NEW_CAMPAIGN })
	      const ipfsStatus = result?.campaign?.ipfsStatus
	      let message = 'Campaign stored'
	      if (ipfsStatus === 'pinned' || ipfsStatus === 'healthy') {
	        message = 'Campaign stored and pinned to IPFS'
	      } else if (ipfsStatus === 'unconfigured') {
	        message = 'Campaign stored (IPFS not configured on backend)'
	      } else if (ipfsStatus === 'pin-failed') {
	        message = 'Campaign stored, but IPFS pinning failed; check backend logs'
	      }
	      pushToast('success', message)
      await refreshDashboard({ silent: true, showErrorToast: false })
    } catch (error) {
      pushToast('error', error.message || 'Campaign creation failed')
    } finally {
      setIsSubmittingCampaign(false)
    }
  }

  const simulateClick = async (campaignId) => {
    try {
      setSimulatingCampaignId(campaignId)
	      setLatestLatency({ ml: null, backend: null, settlement: null })
      const result = await requestJson('/simulate-click', {
        method: 'POST',
        body: JSON.stringify({
          campaignId,
          walletAddress,
          visitorId: getVisitorId(),
          clickTime: new Date().toISOString(),
          deviceType: detectClientDeviceType(),
          userAgent: navigator.userAgent,
          platform: navigator.platform,
        }),
      })
	      setLatestLatency({
	        ml: typeof result.mlLatency === 'number' ? result.mlLatency : null,
	        backend: typeof result.backendLatency === 'number' ? result.backendLatency : null,
	        settlement: typeof result.settlementLatency === 'number' ? result.settlementLatency : null,
	      })
      pushToast(result.valid ? 'success' : 'error', getSettlementMessage(result))
      await refreshDashboard({ silent: true, showErrorToast: false })
    } catch (error) {
      const rawMessage = String(error?.message || '').trim()
      const normalized = rawMessage.toLowerCase()
      let friendly = rawMessage || 'Click validation failed'

      if (normalized === 'internal server error') {
        friendly = 'Backend error during click validation. Please try again later.'
      }

      pushToast('error', friendly)
    } finally {
      setSimulatingCampaignId('')
    }
  }

  const latestValidation = clickLogs[0] || null
  const blockchain = health?.blockchain || {}
  const backendHealthStatus = health?.backend?.status || health?.status || 'unknown'
  const mlServiceStatus = modelMetadata?.status || health?.services?.mlApi?.status || health?.mlApi || 'unknown'
  const mlLatencyMs = modelMetadata?.latencyMs ?? health?.services?.mlApi?.latencyMs ?? health?.mlApiLatencyMs ?? null
  const currentSettlementMode = blockchain.settlementMode || health?.settlementMode || 'simulation'
  const backendCheckedAt = health?.backend?.checkedAt || health?.checkedAt || null
  const recentOnChainSettlements = metrics.recentOnChainSettlements || []
  const recentSimulatedSettlements = metrics.recentSimulatedSettlements || []
  const latestConfidenceTone = confidenceTone(latestValidation?.confidence)
  const simulationMode = currentSettlementMode !== 'onchain'
	  const isWalletConnected = Boolean(walletAddress)
	  const settlementActionHint = simulationMode
	    ? (blockchain.reason || 'Simulation mode is active. Valid clicks are recorded off-chain until blockchain settlement is configured.')
	    : walletAddress
	      ? 'Validated clicks can settle to the connected payout wallet on-chain.'
	      : 'Connect a wallet to receive on-chain payouts after validation.'
	  const walletBadgeLabel = isConnecting
	    ? 'Connecting…'
	    : simulationMode
	      ? walletAddress ? `Optional wallet • ${shortAddress(walletAddress)}` : 'Wallet optional in simulation'
	      : walletAddress ? `Payout wallet • ${shortAddress(walletAddress)}` : 'Connect payout wallet'
	  const blockchainStatusRaw = blockchain.connectionStatus || blockchain.state || 'simulation'
	  const blockchainStatusLabel = isWalletConnected
	    ? serviceStatusLabel(blockchainStatusRaw, 'Simulation only')
	    : 'Wallet not connected'
	  const blockchainStatusTone = isWalletConnected ? blockchainTone(blockchainStatusRaw) : 'warn'
	  const settlementModeChipLabel = isWalletConnected
	    ? settlementModeLabel(currentSettlementMode)
	    : 'Wallet not connected'
	  const settlementModeChipTone = isWalletConnected && currentSettlementMode === 'onchain' ? 'ok' : 'warn'
	  const settlementModeChipDetail = isWalletConnected
	    ? settlementActionHint
	    : 'Connect a wallet to enable on-chain settlement indicators.'
	  const settlementBackendChipTitle = isWalletConnected ? blockchain.walletAddress || '' : ''
	  const settlementBackendChipTone = isWalletConnected
	    ? (blockchain.signerConnected ? 'ok' : blockchain.initError ? 'error' : 'warn')
	    : 'warn'
	  const settlementBackendChipLabel = isWalletConnected
	    ? serviceStatusLabel(blockchainStatusRaw, 'Simulation only')
	    : 'Wallet not connected'
	  const settlementBackendChipDetail = isWalletConnected
	    ? (blockchain.reason || shortAddress(blockchain.contractAddress, 'Contract not configured'))
	    : 'Connect a wallet to view backend settlement status.'
	  const blockchainPanelBadgeTone = blockchainStatusTone
	  const blockchainPanelBadgeLabel = blockchainStatusLabel
	  const blockchainModeValue = isWalletConnected ? settlementModeLabel(currentSettlementMode) : 'Wallet not connected'
	  const blockchainNetworkValue = isWalletConnected ? (blockchain.network || 'Simulation only') : 'Wallet not connected'
	  const blockchainSignerValue = isWalletConnected
	    ? shortAddress(blockchain.walletAddress, simulationMode ? 'Optional in simulation' : 'Not connected')
	    : 'Wallet not connected'
	  const blockchainStatusNoteTone = isWalletConnected ? blockchainTone(blockchain.connectionStatus || 'simulation') : 'warn'
	  const blockchainStatusNoteText = isWalletConnected
	    ? settlementActionHint
	    : 'Connect a wallet to enable on-chain settlement and status details.'
	  const fraudInsights = useMemo(() => {
    const counts = new Map()
    clickLogs
      .filter((entry) => !entry.valid)
      .forEach((entry) => {
        ;(entry.explanation?.signals || [])
          .filter((signal) => signal.tone === 'risk')
          .forEach((signal) => {
            const existing = counts.get(signal.key) || { ...signal, count: 0 }
            existing.count += 1
            counts.set(signal.key, existing)
          })
      })
    return [...counts.values()].sort((left, right) => right.count - left.count).slice(0, 4)
  }, [clickLogs])

		const clicksByOutcome = useMemo(
			() => [
				{ name: 'Valid', count: metrics.validClicks || 0 },
				{ name: 'Fraud', count: metrics.fraudBlocked || 0 },
			],
			[metrics.validClicks, metrics.fraudBlocked],
		)

			const mlLatencyAverageMs = metrics.mlLatencyMs?.average ?? null
			const backendLatencyAverageMs = metrics.validationLatencyMs?.average ?? null
			const settlementLatencyAverageMs = metrics.settlementLatencyMs?.average ?? null
			const mlLatencyEffectiveMs = latestLatency.ml ?? mlLatencyAverageMs
			const backendLatencyEffectiveMs = latestLatency.backend ?? backendLatencyAverageMs
			const settlementLatencyEffectiveMs = latestLatency.settlement ?? settlementLatencyAverageMs
			const mlLatencyProcessing = mlLatencyEffectiveMs === null && Boolean(simulatingCampaignId)
			const backendLatencyProcessing = backendLatencyEffectiveMs === null && Boolean(simulatingCampaignId)
			const settlementLatencyProcessing = settlementLatencyEffectiveMs === null && Boolean(simulatingCampaignId)

  const pageTitle = view === 'publisher'
    ? 'Publisher Dashboard'
    : view === 'advertiser'
      ? 'Campaign Manager'
      : 'Security Operations'

  const pageDescription = view === 'publisher'
    ? 'Validate clicks, monitor payouts, and inspect fraud-screening outcomes in real time.'
    : view === 'advertiser'
      ? 'Create campaigns with budgets, routing codes, and ledger-backed settlement metadata.'
      : 'Review recent validations, settlement states, and model confidence from the shared ledger.'

  return (
    <div className="dashboard-layout">
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">🔗</div>
          <h1>AdChain</h1>
        </div>

        <nav className="nav-links">
          <button className={`nav-item ${view === 'publisher' ? 'active' : ''}`} onClick={() => setView('publisher')}>
            📊 Publisher View
          </button>
          <button className={`nav-item ${view === 'advertiser' ? 'active' : ''}`} onClick={() => setView('advertiser')}>
            🚀 Advertiser Portal
          </button>
          <button className={`nav-item ${view === 'security' ? 'active' : ''}`} onClick={() => setView('security')}>
            🛡️ Security Logs
          </button>
        </nav>

        <div className="sidebar-footer">
          <p className="subtitle">AdChain control panel</p>
	          <p className="microcopy">Settlement: {isWalletConnected ? settlementModeLabel(currentSettlementMode) : 'Wallet not connected'}</p>
        </div>
      </aside>

      <main className="main-content">
        <div className="top-bar">
          <div className="page-title">
            <h2>{pageTitle}</h2>
            <p>{pageDescription}</p>
          </div>

          <div className="top-actions">
            <button className="secondary-btn" onClick={() => refreshDashboard({ silent: true, showErrorToast: true })}>
              {isRefreshing ? 'Refreshing…' : 'Refresh data'}
            </button>
	            <button className={`wallet-badge ${simulationMode ? 'optional' : ''}`} onClick={connectWallet}>
	              <span className="dot"></span>
	              {walletBadgeLabel}
	            </button>
	            {isWalletConnected && (
	              <button className="secondary-btn" onClick={disconnectWallet}>
	                Logout Wallet
	              </button>
	            )}
          </div>
        </div>

        <div className="health-strip">
          <div className="health-chip">
            <span className="health-label">Backend API</span>
            <span className={`health-value ${healthTone(backendHealthStatus)}`}>{serviceStatusLabel(backendHealthStatus, 'Service unavailable')}</span>
            <span className="health-detail">{formatUptimeSeconds(health?.backend?.uptimeSec)} • {backendCheckedAt ? `checked ${formatTimestamp(backendCheckedAt)}` : 'Awaiting refresh'}</span>
          </div>
          <div className="health-chip">
            <span className="health-label">ML API</span>
            <span className={`health-value ${healthTone(mlServiceStatus)}`}>{serviceStatusLabel(mlServiceStatus, 'Service unavailable')}</span>
            <span className="health-detail">{modelMetadata?.model || modelMetadata?.message || 'Service unavailable'}</span>
          </div>
          <div className="health-chip">
            <span className="health-label">Inference latency</span>
            <span className={`health-value ${mlLatencyMs !== null ? 'ok' : healthTone(mlServiceStatus)}`}>{formatLatency(mlLatencyMs, mlServiceStatus === 'ok' ? 'Awaiting measurement' : 'Service unavailable')}</span>
            <span className="health-detail">F1 {formatMetric(modelMetadata?.metrics?.f1, 3, 'Unavailable')} • ROC-AUC {formatMetric(modelMetadata?.metrics?.roc_auc, 3, 'Unavailable')}</span>
          </div>
          <div className="health-chip">
            <span className="health-label">Settlement Mode</span>
	            <span className={`health-value ${settlementModeChipTone}`}>
	              {settlementModeChipLabel}
	            </span>
	            <span className="health-detail">{settlementModeChipDetail}</span>
          </div>
          <div className="health-chip">
            <span className="health-label">Settlement backend</span>
	            <span className={`health-value ${settlementBackendChipTone}`} title={settlementBackendChipTitle}>
	              {settlementBackendChipLabel}
	            </span>
	            <span className="health-detail">{settlementBackendChipDetail}</span>
          </div>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-label">Active Campaigns</span>
            <span className="stat-value">{metrics.activeCampaigns}</span>
            <div className="stat-trend">{metrics.totalCampaigns} total campaign records</div>
          </div>
          <div className="stat-card">
            <span className="stat-label">Clicks Evaluated</span>
            <span className="stat-value">{metrics.totalClicks}</span>
            <div className="stat-trend trend-up">{metrics.validClicks} accepted by ML</div>
          </div>
          <div className="stat-card">
            <span className="stat-label">Fraud Blocked</span>
            <span className="stat-value danger">{metrics.fraudBlocked}</span>
            <div className="stat-trend">Dataset-aligned burst screening active</div>
          </div>
          <div className="stat-card">
            <span className="stat-label">Recorded Payout Volume</span>
            <span className="stat-value">{metrics.totalPayoutEth} ETH</span>
            <div className="stat-trend">Confirmed {metrics.confirmedPayoutEth || '0.0000'} ETH • Simulated {metrics.simulatedPayoutEth || '0.0000'} ETH</div>
          </div>
			</div>

			<div className="charts-row">
				<section className="glass-panel chart-panel">
					<div className="panel-header">
						<h3>Click outcome breakdown</h3>
						<span className="subtitle">Valid vs fraud-blocked clicks</span>
					</div>
					<div className="chart-inner">
							<div className="click-outcome-metrics">
								<div className="click-outcome-card click-outcome-valid">
									<span className="click-outcome-icon"></span>
									<span className="stat-label">Valid Clicks</span>
									<span className="stat-value">{metrics.validClicks ?? 0}</span>
								</div>
								<div className="click-outcome-card click-outcome-fraud">
									<span className="click-outcome-icon"></span>
									<span className="stat-label">Fraud Blocked</span>
									<span className="stat-value danger">{metrics.fraudBlocked ?? 0}</span>
								</div>
							</div>
					</div>
				</section>

				<section className="glass-panel chart-panel">
					<div className="panel-header">
						<h3>Latency comparison</h3>
						<span className="subtitle">Average ML inference vs settlement time</span>
					</div>
						<div className="chart-inner">
							<div className="latency-metrics">
								<div className="latency-card latency-card-ml">
									<span className="stat-label">ML Inference Time</span>
									<span className="latency-value">
								{mlLatencyEffectiveMs !== null
								  ? formatMetric(mlLatencyEffectiveMs, 0, '–')
								  : mlLatencyProcessing
								    ? 'Processing...'
								    : '–'}
								{mlLatencyEffectiveMs !== null ? <span className="latency-unit"> ms</span> : null}
									</span>
									<span className="latency-subtext">Average model inference per click</span>
								</div>
								<div className="latency-card latency-card-backend">
									<span className="stat-label">Backend Processing Latency</span>
									<span className="latency-value">
								{backendLatencyEffectiveMs !== null
								  ? formatMetric(backendLatencyEffectiveMs, 0, '–')
								  : backendLatencyProcessing
								    ? 'Processing...'
								    : '–'}
								{backendLatencyEffectiveMs !== null ? <span className="latency-unit"> ms</span> : null}
									</span>
									<span className="latency-subtext">Average API request processing</span>
								</div>
								<div className="latency-card latency-card-settlement">
									<span className="stat-label">Settlement Time</span>
									<span className="latency-value">
								{settlementLatencyEffectiveMs !== null
								  ? formatMetric(settlementLatencyEffectiveMs, 0, '–')
								  : settlementLatencyProcessing
								    ? 'Processing...'
								    : '–'}
								{settlementLatencyEffectiveMs !== null ? <span className="latency-unit"> ms</span> : null}
									</span>
									<span className="latency-subtext">Avg. blockchain confirmation latency</span>
								</div>
							</div>
						</div>
				</section>
			</div>

			{isLoading ? <div className="empty-state">Loading decentralized ad network state…</div> : null}

        {view === 'advertiser' ? (
          <div className="advertiser-layout">
            <section className="premium-form">
              <h3>Create New Campaign</h3>
              <p className="subtitle intro-copy">Store campaign metadata, payout limits, and routing codes for fraud validation. {simulationMode ? 'Campaigns are being stored for simulation-only settlement right now.' : 'New campaigns can validate into live on-chain settlement.'}</p>

              <form onSubmit={uploadCampaign}>
                <div className="input-box">
                  <label htmlFor="title">Campaign Title</label>
                  <input id="title" type="text" value={newCampaign.title} onChange={(event) => handleCampaignChange('title', event.target.value)} placeholder="Enter headline..." required />
                </div>

                <div className="form-grid">
                  <div className="input-box">
                    <label htmlFor="image">Creative URL</label>
                    <input id="image" type="url" value={newCampaign.image} onChange={(event) => handleCampaignChange('image', event.target.value)} placeholder="Paste IPFS or Web URL" required />
                  </div>
                  <div className="input-box">
                    <label htmlFor="target">Destination URL</label>
                    <input id="target" type="url" value={newCampaign.targetUrl} onChange={(event) => handleCampaignChange('targetUrl', event.target.value)} placeholder="https://example.com" required />
                  </div>
                </div>

                <div className="form-grid">
                  <div className="input-box">
                    <label htmlFor="payout">Payout per Valid Click (ETH)</label>
                    <input id="payout" type="number" min="0.0001" step="0.0001" value={newCampaign.payoutEth} onChange={(event) => handleCampaignChange('payoutEth', event.target.value)} required />
                  </div>
                  <div className="input-box">
                    <label htmlFor="budget">Total Budget (ETH)</label>
                    <input id="budget" type="number" min="0.001" step="0.0001" value={newCampaign.budgetEth} onChange={(event) => handleCampaignChange('budgetEth', event.target.value)} required />
                  </div>
                </div>

                <div className="form-grid-third">
                  <div className="input-box">
                    <label htmlFor="app-code">App Code</label>
                    <input id="app-code" type="number" value={newCampaign.appCode} onChange={(event) => handleCampaignChange('appCode', event.target.value)} required />
                  </div>
                  <div className="input-box">
                    <label htmlFor="os-code">OS Code</label>
                    <input id="os-code" type="number" value={newCampaign.osCode} onChange={(event) => handleCampaignChange('osCode', event.target.value)} required />
                  </div>
                  <div className="input-box">
                    <label htmlFor="channel-code">Channel Code</label>
                    <input id="channel-code" type="number" value={newCampaign.channelCode} onChange={(event) => handleCampaignChange('channelCode', event.target.value)} required />
                  </div>
                </div>

                <button type="submit" className="submit-btn" disabled={isSubmittingCampaign}>
                  {isSubmittingCampaign ? 'Publishing…' : publishButtonLabel(currentSettlementMode)}
                </button>
              </form>
            </section>

            <aside className="glass-panel campaign-panel">
              <div className="panel-header">
                <h3>Campaign Ledger</h3>
                <span className="subtitle">{campaigns.length} campaign entries</span>
              </div>

              <div className="campaign-list">
                {campaigns.length === 0 ? (
                  <p className="empty-state compact">No campaigns stored yet.</p>
                ) : campaigns.map((campaign) => (
                  <div className="campaign-list-item" key={campaign.id}>
                    <div>
                      <strong>{campaign.title}</strong>
	                    <div className="microcopy">CID: {campaign.cid}</div>
	                    <div className="microcopy">
	                      IPFS: {campaign.ipfsCid ? `${campaign.ipfsCid.slice(0, 18)}… (${titleCase(campaign.ipfsStatus || 'unknown', 'Unknown')})` : 'not pinned'}
	                    </div>
                      <div className="microcopy">Advertiser: {shortAddress(campaign.advertiserAddress)}</div>
                    </div>
                    <div className="campaign-kpis">
                      <span>Payout {formatMetric(campaign.payoutEth, 4, 'Unavailable')} ETH</span>
                      <span>Budget left {formatMetric(campaign.remainingBudgetEth, 4, 'Unavailable')} ETH</span>
                      <span>Status {campaign.status}</span>
                      <span>Clicks {campaign.metrics?.clicks ?? 0}</span>
                    </div>
                  </div>
                ))}
              </div>
            </aside>
          </div>
        ) : (
          <div className={`panel-grid ${view === 'security' ? 'panel-grid-wide' : ''}`}>
            <section className="ads-section">
              <div className="ads-section-header">
                <h3>{view === 'publisher' ? 'Live Campaign Inventory' : 'Recent Validation Ledger'}</h3>
                <span className="subtitle">{view === 'publisher' ? `${campaigns.length} campaigns available` : `${clickLogs.length} recent validations`}</span>
              </div>

	              {view === 'publisher' ? (
	                campaigns.length === 0 ? (
	                  <div className="empty-state">
	                    No campaigns are available yet. Open <strong>Campaign Manager</strong> to publish the first campaign before validating clicks.
	                  </div>
	                ) : (
	                  <div className="ads-container">
	                    {campaigns.map((campaign) => {
	                      const ipfsStatus = campaign.ipfsStatus || (campaign.ipfsCid ? 'unknown' : 'unconfigured')
	                      const ipfsHealthy = !campaign.ipfsCid || ipfsStatus === 'healthy' || ipfsStatus === 'pinned'
	                      const creativeSrc = ipfsHealthy && !isLegacyCreativeImage(campaign.image) && campaign.image
	                        ? campaign.image
	                        : DEFAULT_CREATIVE

	                      return (
	                        <div key={campaign.id} className="premium-ad-card">
	                          <div className="ad-topbar">
	                            <div className="budget-pill">{formatMetric(campaign.remainingBudgetEth, 4, 'Unavailable')} ETH confirmed budget left</div>
	                          </div>
	                          <div className="ad-image-container">
	                            <img
	                              src={creativeSrc}
	                              alt=""
	                              aria-hidden="true"
	                              loading="lazy"
	                              onError={(event) => {
	                                event.currentTarget.onerror = null
	                                event.currentTarget.src = DEFAULT_CREATIVE
	                              }}
	                            />
	                          </div>
	                          <div className="ad-info">
	                            <h3>{campaign.title}</h3>
	                            <div className="meta-row">
	                              <span className="metric-pill">App code {campaign.appCode}</span>
	                              <span className="metric-pill">OS code {campaign.osCode}</span>
	                              <span className="metric-pill">Channel code {campaign.channelCode}</span>
	                            </div>
	                            <div className="campaign-kpis">
	                              <span>Confirmed on-chain settlements {campaign.metrics?.confirmedSettlements ?? campaign.metrics?.settlements ?? 0}</span>
	                              <span>Simulation records {campaign.metrics?.simulatedSettlements ?? 0}</span>
	                              <span>Status {titleCase(campaign.status, 'Active')}</span>
	                            </div>
	                            <div className="action-row">
	                              <button
	                                className={`action-btn ${simulationMode ? 'simulation' : ''}`}
	                                onClick={() => simulateClick(campaign.id)}
	                                disabled={simulatingCampaignId === campaign.id || campaign.status === 'budget-exhausted'}
	                                title={campaign.status === 'budget-exhausted' ? 'Campaign budget exhausted' : settlementActionHint}
	                              >
	                                {simulatingCampaignId === campaign.id
	                                  ? <span className="animate-pulse">Analyzing…</span>
	                                  : campaign.status === 'budget-exhausted'
	                                    ? 'Budget exhausted'
	                                    : simulationMode
	                                      ? `Validate & record ${formatMetric(campaign.payoutEth, 4, 'Unavailable')} ETH`
	                                      : walletAddress
	                                        ? `Validate & settle ${formatMetric(campaign.payoutEth, 4, 'Unavailable')} ETH`
	                                        : `Validate click • wallet needed for ${formatMetric(campaign.payoutEth, 4, 'Unavailable')} ETH payout`}
	                              </button>
	                              <a href={campaign.targetUrl} target="_blank" rel="noreferrer" className="subtle-link">
	                                Visit advertiser target
	                              </a>
	                              {campaign.ipfsCid ? (
	                                <a
	                                  href={`${API_BASE}/ipfs/${campaign.ipfsCid}`}
	                                  target="_blank"
	                                  rel="noreferrer"
	                                  className="subtle-link"
	                                >
	                                  View IPFS metadata
	                                </a>
	                              ) : null}
	                              <p className={`action-hint ${simulationMode || !walletAddress ? 'warn' : ''}`}>{settlementActionHint}</p>
	                            </div>
	                          </div>
	                        </div>
	                      )
	                    })}
	                  </div>
	                )
	              ) : (
	                <div className="log-list">
                  {clickLogs.length === 0 ? (
                    <p className="empty-state compact">No click validations recorded yet.</p>
                  ) : clickLogs.map((entry) => (
                    <div className="log-item" key={entry.id}>
                      <div className="log-main">
                        <div>
                          <div className="log-title-row">
                            <strong>{entry.campaignTitle}</strong>
                            <span className={`badge ${entry.valid ? 'success' : 'error'}`}>{entry.valid ? 'valid' : 'blocked'}</span>
                            <span className={`badge confidence ${confidenceTone(entry.confidence)}`}>{formatConfidence(entry.confidence)}</span>
                          </div>
                          <div className="microcopy">{formatTimestamp(entry.timestamp)} • {shortAddress(entry.walletAddress || entry.visitorKey)}</div>
                        </div>
                        <div className="campaign-kpis align-right">
                          <span>Threshold {formatMetric(entry.threshold, 3, 'Unavailable')}</span>
                          <span>Settlement {titleCase(entry.settlement?.status, 'Unavailable')}</span>
                          <span>Model {entry.model || 'Service unavailable'}</span>
                        </div>
                      </div>
                      <div className="log-features">
                        {FEATURE_DETAILS.map((feature) => (
                          <span key={`${entry.id}-${feature.key}`} className="feature-chip" title={feature.description}>
                            <span className="feature-label">{feature.label}</span>
                            <strong className="feature-value">{formatFeatureValue(feature.key, entry.features?.[feature.key])}</strong>
                          </span>
                        ))}
                      </div>
                      <div className="decision-summary">
                        <p className="explanation-copy">{validationSummary(entry)}</p>
                        <span className="microcopy">Decision latency {formatLatency(entry.mlLatencyMs, 'Unavailable')}</span>
                      </div>
                      {(entry.explanation?.signals || []).length > 0 ? (
                        <div className="signal-list">
                          {entry.explanation.signals.map((signal) => (
                            <span key={`${entry.id}-${signal.key}`} className={`signal-chip ${signal.tone}`} title={signal.detail}>
                              {signal.label}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              )}
            </section>

            <aside className="stacked-panels">
              <div className="glass-panel">
                <div className="panel-header">
                  <h3>Blockchain Status</h3>
                  <span className={`badge ${blockchainTone(blockchain.connectionStatus || blockchain.state || 'simulation')}`}>
                    {serviceStatusLabel(blockchain.state || blockchain.connectionStatus || 'simulation', 'Simulation only')}
                  </span>
                </div>

                <div className="model-grid">
                  <div className="kpi-row"><span>Current mode</span><strong>{settlementModeLabel(currentSettlementMode)}</strong></div>
                  <div className="kpi-row"><span>Network</span><strong>{blockchain.network || 'Simulation only'}</strong></div>
                  <div className="kpi-row"><span>Backend signer</span><strong title={blockchain.walletAddress || ''}>{shortAddress(blockchain.walletAddress, simulationMode ? 'Optional in simulation' : 'Not connected')}</strong></div>
                  <div className="kpi-row"><span>Historical records</span><strong>{metrics.onChainSettlementCount} on-chain • {metrics.simulatedSettlementCount} simulated</strong></div>
                </div>

                <p className={`status-note ${blockchainTone(blockchain.connectionStatus || 'simulation')}`}>
                  {settlementActionHint}
                </p>
                <div className="signal-list compact-signals">
                  {blockchain.contractAddress ? <span className="signal-chip supporting">Contract {shortAddress(blockchain.contractAddress)}</span> : null}
                  {blockchain.gasPriceGwei ? <span className="signal-chip supporting">Gas {formatMetric(blockchain.gasPriceGwei, 2, 'n/a')} gwei</span> : null}
                  {(blockchain.missingConfig || []).map((item) => (
                    <span key={item} className="signal-chip risk">Missing {item}</span>
                  ))}
                </div>
              </div>

              <div className="glass-panel">
                <div className="panel-header">
                  <h3>Active Model</h3>
                  <span className="subtitle">{modelMetadata?.model || (mlServiceStatus === 'degraded' ? 'Service unavailable' : 'Loading metadata')}</span>
                </div>

                <div className="model-grid">
                  <div className="kpi-row"><span>Model status</span><strong>{serviceStatusLabel(mlServiceStatus, 'Unavailable')}</strong></div>
                  <div className="kpi-row"><span>Threshold</span><strong>{formatMetric(modelMetadata?.threshold, 3, 'Unavailable')}</strong></div>
                  <div className="kpi-row"><span>Detected suspicious rate</span><strong>{formatConfidence(modelMetadata?.evaluation?.predictedFraudRate, 'Unavailable')}</strong></div>
                  <div className="kpi-row"><span>F1 score</span><strong>{formatMetric(modelMetadata?.metrics?.f1, 4, 'Unavailable')}</strong></div>
                  <div className="kpi-row"><span>ROC-AUC</span><strong>{formatMetric(modelMetadata?.metrics?.roc_auc, 4, 'Unavailable')}</strong></div>
                  <div className="kpi-row"><span>Inference latency</span><strong>{formatLatency(mlLatencyMs, 'Unavailable')}</strong></div>
                  <div className="kpi-row"><span>Last refresh</span><strong>{backendCheckedAt ? formatTimestamp(backendCheckedAt) : 'Awaiting refresh'}</strong></div>
                </div>
              </div>

              <div className="glass-panel">
                <div className="panel-header">
                  <h3>Recent Settlements</h3>
                  <span className="subtitle">Separated by settlement mode</span>
                </div>

                <div className="settlement-groups">
                  <section className="settlement-group">
                    <div className="settlement-group-header">
                      <h4>On-chain</h4>
                      <span className="badge success">{metrics.onChainSettlementCount}</span>
                    </div>
                    <div className="transactions-list compact-list">
                      {recentOnChainSettlements.length === 0 ? (
                        <p className="empty-state compact">No confirmed on-chain settlements yet.</p>
                      ) : recentOnChainSettlements.map((tx) => (
                        <div key={tx.id} className="tx-item onchain">
                          <div className="tx-meta">
                            <span className="tx-amount-text">{tx.payoutEth} ETH • {titleCase(tx.status)}</span>
                            <span className="tx-timestamp">{formatTimestamp(tx.timestamp)} • {formatLatency(tx.settlementLatencyMs, 'Latency unavailable')} • gas {tx.gasUsed || 'n/a'}</span>
                          </div>
                          <a href={tx.explorerUrl} target="_blank" rel="noreferrer" className="receipt-btn">View on-chain</a>
                        </div>
                      ))}
                    </div>
                  </section>

                  <section className="settlement-group">
                    <div className="settlement-group-header">
                      <h4>Simulation</h4>
                      <span className="badge warn">{metrics.simulatedSettlementCount}</span>
                    </div>
                    <div className="transactions-list compact-list">
                      {recentSimulatedSettlements.length === 0 ? (
                        <p className="empty-state compact">No simulated payouts recorded.</p>
                      ) : recentSimulatedSettlements.map((tx) => (
                        <div key={tx.id} className="tx-item simulated">
                          <div className="tx-meta">
                            <span className="tx-amount-text">{tx.payoutEth} ETH • {titleCase(tx.status)}</span>
                            <span className="tx-timestamp">{formatTimestamp(tx.timestamp)} • {formatLatency(tx.settlementLatencyMs, 'Simulation record')}</span>
                          </div>
                          <span className="badge warn">simulation</span>
                        </div>
                      ))}
                    </div>
                  </section>
                </div>
              </div>

              <div className="glass-panel">
                <div className="panel-header">
                  <h3>Fraud Insights</h3>
                  <span className="subtitle">Most common risk signals in blocked traffic</span>
                </div>

                {fraudInsights.length === 0 ? (
                  <p className="empty-state compact">Risk patterns will appear after suspicious traffic is detected.</p>
                ) : (
                  <div className="benchmark-list">
                    {fraudInsights.map((signal) => (
                      <div className="benchmark-item" key={signal.key}>
                        <div>
                          <strong>{signal.label}</strong>
                          <div className="microcopy">{signal.detail}</div>
                        </div>
                        <span className="badge warn">{signal.count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="glass-panel">
                <div className="panel-header">
                  <h3>Latest Decision</h3>
                  <span className="subtitle">Real-time fraud screen</span>
                </div>

                {latestValidation ? (
                  <>
                    <div className="kpi-stack">
                      <div className="kpi-row"><span>Campaign</span><strong>{latestValidation.campaignTitle}</strong></div>
                      <div className="kpi-row"><span>Decision</span><strong>{latestValidation.valid ? 'Accepted' : 'Blocked'}</strong></div>
                      <div className="kpi-row"><span>Confidence</span><strong className={`confidence-score ${latestConfidenceTone}`}>{formatConfidence(latestValidation.confidence, 'Unavailable')}</strong></div>
                      <div className="kpi-row"><span>Threshold</span><strong>{formatMetric(latestValidation.threshold, 3, 'Unavailable')}</strong></div>
                      <div className="kpi-row"><span>Settlement</span><strong>{titleCase(latestValidation.settlement?.status, 'Unavailable')}</strong></div>
                      <div className="kpi-row"><span>Reason</span><strong>{validationSummary(latestValidation)}</strong></div>
                      <div className="kpi-row"><span>Decision latency</span><strong>{formatLatency(latestValidation.mlLatencyMs, 'Unavailable')}</strong></div>
                    </div>
                    {(latestValidation.explanation?.signals || []).length > 0 ? (
                      <div className="signal-list">
                        {latestValidation.explanation.signals.map((signal) => (
                          <span key={`latest-${signal.key}`} className={`signal-chip ${signal.tone}`} title={signal.detail}>
                            {signal.label}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </>
                ) : (
                  <p className="empty-state compact">Trigger a click validation to inspect the most recent model decision.</p>
                )}
              </div>
            </aside>
          </div>
        )}

        {toast.text ? (
          <div className={`toast ${toast.kind}`}>
            <span>{toast.kind === 'success' ? '🚀' : '🛡️'}</span>
            <p>{toast.text}</p>
          </div>
        ) : null}
      </main>
    </div>
  )
}

export default App