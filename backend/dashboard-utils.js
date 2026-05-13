import crypto from 'crypto'
import { ethers } from 'ethers'

const DEVICE_TYPES = new Set(['mobile', 'desktop', 'tablet'])

function toFiniteNumber(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function toIsoString(value) {
  const timestamp = Date.parse(value || '')
  return Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : null
}

function stableJsonStringify(value) {
  if (value === null || value === undefined) return 'null'
  if (Array.isArray(value)) return `[${value.map((item) => stableJsonStringify(item)).join(',')}]`
  if (typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJsonStringify(value[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function percentile(values, rank) {
  if (!values.length) return null
  const sorted = [...values].sort((left, right) => left - right)
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((rank / 100) * sorted.length) - 1))
  return sorted[index]
}

function summarizeLatency(values = []) {
  const numericValues = values.map((value) => toFiniteNumber(value)).filter((value) => value !== null)
  if (!numericValues.length) {
    return { sampleCount: 0, median: null, p95: null, max: null, average: null }
  }
  const average = Math.round(numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length)
  return {
    sampleCount: numericValues.length,
    median: Math.round(percentile(numericValues, 50)),
    p95: Math.round(percentile(numericValues, 95)),
    max: Math.round(Math.max(...numericValues)),
    average,
  }
}

function normalizeOracleNumber(value) {
  const parsed = toFiniteNumber(value)
  return parsed === null ? 'null' : parsed.toFixed(10)
}

function signOracleMessage(message, secret) {
  return crypto.createHmac('sha256', String(secret)).update(message).digest('base64url')
}

function activeOracleQuorum(signers = [], requiredQuorum = 1) {
  if (!signers.length) return 0
  return Math.min(signers.length, Math.max(1, Number(requiredQuorum) || 1))
}

export function hashPayload(value) {
  return crypto.createHash('sha256').update(stableJsonStringify(value)).digest('hex')
}

export function buildOracleSignatureMessage(decision = {}) {
  return stableJsonStringify({
    confidence: normalizeOracleNumber(decision.confidence),
    expiresAt: toIsoString(decision.expiresAt),
    featuresHash: String(decision.featuresHash || ''),
    issuedAt: toIsoString(decision.issuedAt),
    model: String(decision.model || ''),
    requestId: String(decision.requestId || ''),
    result: String(decision.result ?? ''),
    threshold: normalizeOracleNumber(decision.threshold),
  })
}

export function verifySignedOracleDecision(
  decision = {},
  { signers = [], requiredQuorum = 1, maxSkewMs = 30_000, expectedFeaturesHash, now = Date.now() } = {},
) {
  const normalizedSigners = Array.isArray(signers)
    ? signers.filter((signer) => signer?.id && signer?.secret)
    : []
  const normalizedQuorum = activeOracleQuorum(normalizedSigners, requiredQuorum)
  const signatures = Array.isArray(decision.signatures) ? decision.signatures : []
  const issuedAt = Date.parse(decision.issuedAt || '')
  const expiresAt = Date.parse(decision.expiresAt || '')
  const featuresHash = String(decision.featuresHash || '')

  if (expectedFeaturesHash && expectedFeaturesHash !== featuresHash) {
    return { verified: false, reason: 'features-hash-mismatch', matchedSigners: [], quorum: normalizedQuorum }
  }

  if (!normalizedSigners.length) {
    return {
      verified: true,
      reason: signatures.length ? 'unsigned-accepted-local' : 'unsigned-local',
      matchedSigners: [],
      quorum: 0,
    }
  }

  if (!Number.isFinite(issuedAt)) {
    return { verified: false, reason: 'invalid-issued-at', matchedSigners: [], quorum: normalizedQuorum }
  }

  if (Math.abs(now - issuedAt) > maxSkewMs) {
    return { verified: false, reason: 'stale-issued-at', matchedSigners: [], quorum: normalizedQuorum }
  }

  if (Number.isFinite(expiresAt) && now > expiresAt) {
    return { verified: false, reason: 'decision-expired', matchedSigners: [], quorum: normalizedQuorum }
  }

  const message = buildOracleSignatureMessage(decision)
  const matchedSigners = []
  for (const signer of normalizedSigners) {
    const candidate = signatures.find((item) => item?.signerId === signer.id)
    if (!candidate?.signature) continue
    const expectedSignature = signOracleMessage(message, signer.secret)
    if (candidate.signature === expectedSignature) matchedSigners.push(signer.id)
  }

  return {
    verified: matchedSigners.length >= normalizedQuorum,
    reason: matchedSigners.length >= normalizedQuorum ? 'verified' : 'quorum-not-met',
    matchedSigners,
    quorum: normalizedQuorum,
  }
}

export function normalizeEthValue(value, fallback = 0) {
  const parsed = toFiniteNumber(value)
  const safeValue = parsed === null ? Number(fallback) : parsed
  return Math.max(safeValue, 0).toFixed(4)
}

function sortNewestFirst(items = []) {
  return [...items].sort((left, right) => {
    const leftTime = Date.parse(left?.timestamp || '') || 0
    const rightTime = Date.parse(right?.timestamp || '') || 0
    return rightTime - leftTime
  })
}

export function listMissingBlockchainConfig({ rpcConfigured, privateKeyConfigured, contractConfigured }) {
  return [
    !rpcConfigured ? 'RPC_URL' : null,
    !privateKeyConfigured ? 'PRIVATE_KEY' : null,
    !contractConfigured ? 'CONTRACT_ADDRESS' : null,
  ].filter(Boolean)
}

export function describeBlockchainState({ signerConnected, rpcConfigured, privateKeyConfigured, contractConfigured, initError }) {
  const missingConfig = listMissingBlockchainConfig({ rpcConfigured, privateKeyConfigured, contractConfigured })
  if (signerConnected) {
    return { state: 'enabled', reason: 'RPC, contract, and backend signer are active.', missingConfig }
  }
  if (missingConfig.length > 0) {
    return { state: 'disabled', reason: `Missing ${missingConfig.join(' / ')}`, missingConfig }
  }
  if (initError) {
    return { state: 'disabled', reason: `Signer initialization failed: ${initError}`, missingConfig }
  }
  return { state: 'disabled', reason: 'Blockchain connection is unavailable.', missingConfig }
}

function pushSignal(signals, signal, condition) {
  if (condition) signals.push(signal)
}

export function buildValidationExplanation({ features = {}, confidence, threshold, valid }) {
  const burstScore = toFiniteNumber(features.burstScore)
  const clickFrequency = toFiniteNumber(features.clickFrequency)
  const timeInterval = toFiniteNumber(features.timeInterval)
  const ipAppCount = toFiniteNumber(features.ipAppCount)
  const signals = []

  pushSignal(signals, {
    key: 'ip-burst',
    label: 'IP burst',
    tone: 'risk',
    detail: `Burst score ${burstScore?.toFixed(4) || 'n/a'} indicates compressed click activity.`,
  }, burstScore !== null && burstScore >= 0.02)

  pushSignal(signals, {
    key: 'rapid-repeat-clicks',
    label: 'Rapid repeat clicks',
    tone: 'risk',
    detail: `Clicks arrived ${Math.round(timeInterval)} ms apart.`,
  }, timeInterval !== null && timeInterval <= 2_000)

  pushSignal(signals, {
    key: 'repeat-visitor-app',
    label: 'Visitor-app repetition',
    tone: 'risk',
    detail: `Visitor/app pair repeated ${Math.round(ipAppCount)} times.`,
  }, ipAppCount !== null && ipAppCount >= 3)

  pushSignal(signals, {
    key: 'session-burst',
    label: 'Session burst',
    tone: 'risk',
    detail: `${Math.round(clickFrequency)} clicks were observed in the rolling session window.`,
  }, clickFrequency !== null && clickFrequency >= 4)

  pushSignal(signals, {
    key: 'low-burst-score',
    label: 'Low burst score',
    tone: 'supporting',
    detail: `Burst score ${burstScore?.toFixed(4) || 'n/a'} stayed below the risk band.`,
  }, burstScore !== null && burstScore < 0.005)

  pushSignal(signals, {
    key: 'normal-cadence',
    label: 'Normal click cadence',
    tone: 'supporting',
    detail: `Time between clicks was ${Math.round(timeInterval / 1000)} seconds.`,
  }, timeInterval !== null && timeInterval >= 60_000)

  const riskSignals = signals.filter((signal) => signal.tone === 'risk')
  const supportSignals = signals.filter((signal) => signal.tone === 'supporting')
  const confidenceValue = toFiniteNumber(confidence)
  const thresholdValue = toFiniteNumber(threshold)

  let summary = 'Reason unavailable'
  if (valid) {
    if (supportSignals.length > 0) {
      summary = supportSignals.map((signal) => signal.label.toLowerCase()).slice(0, 2).join(' and ')
    } else if (confidenceValue !== null && thresholdValue !== null) {
      summary = confidenceValue >= thresholdValue ? 'Model confidence stayed above threshold' : 'Accepted near the model threshold'
    } else {
      summary = 'Behavior matched the valid-click profile'
    }
  } else if (riskSignals.length > 0) {
    summary = riskSignals[0].label
  } else if (confidenceValue !== null && thresholdValue !== null) {
    summary = confidenceValue < thresholdValue ? 'Model confidence fell below threshold' : 'Anomaly heuristics flagged the click'
  } else {
    summary = 'Anomalous click behavior detected'
  }

  return {
    summary,
    signals: (valid ? [...supportSignals, ...riskSignals] : [...riskSignals, ...supportSignals]).slice(0, 3),
  }
}

export function validateClickRequestPayload(payload = {}) {
  const campaignIdRaw = payload.campaignId ?? payload.adId
  const campaignId = typeof campaignIdRaw === 'string' ? campaignIdRaw.trim() : ''
  if (!campaignId || campaignId.length > 128) {
    return { error: 'Campaign id is required and must be at most 128 characters' }
  }

  const walletAddress = typeof payload.walletAddress === 'string' ? payload.walletAddress.trim() : ''
  if (walletAddress && !ethers.isAddress(walletAddress)) {
    return { error: 'Wallet address must be a valid Ethereum address' }
  }

  const visitorId = typeof payload.visitorId === 'string' ? payload.visitorId.trim() : ''
  if (visitorId.length > 128) {
    return { error: 'Visitor id must be at most 128 characters' }
  }

  const requestId = typeof payload.requestId === 'string' ? payload.requestId.trim() : ''
  if (requestId.length > 128) {
    return { error: 'requestId must be at most 128 characters' }
  }

  if (payload.clickTime && Number.isNaN(Date.parse(payload.clickTime))) {
    return { error: 'clickTime must be a valid ISO timestamp' }
  }

  const deviceType = typeof payload.deviceType === 'string' ? payload.deviceType.trim().toLowerCase() : ''
  if (deviceType && !DEVICE_TYPES.has(deviceType)) {
    return { error: 'deviceType must be mobile, desktop, or tablet' }
  }

  for (const key of ['userAgent', 'platform', 'ipAddress']) {
    const value = typeof payload[key] === 'string' ? payload[key].trim() : ''
    if (value.length > 512) {
      return { error: `${key} must be at most 512 characters` }
    }
  }

  return {
    value: {
      ...payload,
      campaignId,
      walletAddress,
      visitorId,
      requestId,
      deviceType: deviceType || payload.deviceType,
      userAgent: typeof payload.userAgent === 'string' ? payload.userAgent.trim() : payload.userAgent,
      platform: typeof payload.platform === 'string' ? payload.platform.trim() : payload.platform,
      ipAddress: typeof payload.ipAddress === 'string' ? payload.ipAddress.trim() : payload.ipAddress,
    },
  }
}

export function rollupCampaignState(campaign, clickLogs = []) {
  const campaignLogs = clickLogs.filter((entry) => entry.campaignId === campaign.id)
  const confirmedSettlements = campaignLogs.filter((entry) => entry.settlement?.status === 'confirmed').length
  const simulatedSettlements = campaignLogs.filter((entry) => entry.settlement?.status === 'simulated').length
  const payoutPerClick = toFiniteNumber(campaign.payoutEth) ?? 0
  const budgetEth = toFiniteNumber(campaign.budgetEth) ?? 0
  const remainingBudgetEth = normalizeEthValue(budgetEth - (confirmedSettlements * payoutPerClick), 0)

  return {
    remainingBudgetEth,
    metrics: {
      clicks: campaignLogs.length,
      validClicks: campaignLogs.filter((entry) => entry.valid).length,
      fraudBlocked: campaignLogs.filter((entry) => !entry.valid).length,
      settlements: confirmedSettlements,
      confirmedSettlements,
      simulatedSettlements,
      recordedSettlements: confirmedSettlements + simulatedSettlements,
    },
    status: Number(remainingBudgetEth) < payoutPerClick ? 'budget-exhausted' : (campaign.status || 'active'),
  }
}

export function summarizeDashboardMetrics(store = {}) {
  const campaigns = Array.isArray(store.campaigns) ? store.campaigns : []
  const clicks = Array.isArray(store.clickLogs) ? store.clickLogs : []
  const settlements = Array.isArray(store.settlements) ? store.settlements : []
  const confirmed = settlements.filter((item) => item.status === 'confirmed')
  const simulated = settlements.filter((item) => item.status === 'simulated')
  const latencyValues = confirmed
    .map((item) => toFiniteNumber(item.settlementLatencyMs))
    .filter((value) => value !== null)
  const recentOnChainSettlements = sortNewestFirst(confirmed).slice(0, 10)
  const recentSimulatedSettlements = sortNewestFirst(simulated).slice(0, 10)
  const recentSettlements = sortNewestFirst(settlements).slice(0, 10)
  const confirmedPayoutEth = confirmed.reduce((sum, item) => sum + (toFiniteNumber(item.payoutEth) ?? 0), 0)
  const simulatedPayoutEth = simulated.reduce((sum, item) => sum + (toFiniteNumber(item.payoutEth) ?? 0), 0)

  return {
    activeCampaigns: campaigns.filter((campaign) => campaign.status === 'active').length,
    totalCampaigns: campaigns.length,
    totalClicks: clicks.length,
    validClicks: clicks.filter((item) => item.valid).length,
    fraudBlocked: clicks.filter((item) => !item.valid).length,
    totalPayoutEth: (confirmedPayoutEth + simulatedPayoutEth).toFixed(4),
    confirmedPayoutEth: confirmedPayoutEth.toFixed(4),
    simulatedPayoutEth: simulatedPayoutEth.toFixed(4),
    averageSettlementLatencyMs: latencyValues.length
      ? Math.round(latencyValues.reduce((sum, value) => sum + value, 0) / latencyValues.length)
      : null,
    recordedSettlementCount: settlements.length,
    onChainSettlementCount: confirmed.length,
    simulatedSettlementCount: simulated.length,
    recentSettlements,
    recentOnChainSettlements,
    recentSimulatedSettlements,
  }
}

export function summarizeBenchmarkMetrics(store = {}) {
  const clickLogs = Array.isArray(store.clickLogs) ? store.clickLogs : []
  const settlements = Array.isArray(store.settlements) ? store.settlements : []
  const mlLatency = summarizeLatency(clickLogs.map((item) => item.mlLatencyMs))
  const validationLatency = summarizeLatency(clickLogs.map((item) => item.validationLatencyMs))
  const settlementLatency = summarizeLatency(settlements.map((item) => item.settlementLatencyMs))
  const gasValues = settlements
    .filter((item) => item.status === 'confirmed')
    .map((item) => toFiniteNumber(item.gasUsed))
    .filter((value) => value !== null)
  const validClicks = clickLogs.filter((item) => item.valid)
  const oracleVerified = clickLogs.filter((item) => item.oracle?.verified)

  return {
    requests: clickLogs.length,
    validClicks: validClicks.length,
    blockedClicks: clickLogs.filter((item) => !item.valid).length,
    validationLatencyMs: validationLatency,
    mlLatencyMs: mlLatency,
    settlementLatencyMs: settlementLatency,
    gasUsed: {
      sampleCount: gasValues.length,
      average: gasValues.length ? Math.round(gasValues.reduce((sum, value) => sum + value, 0) / gasValues.length) : null,
      max: gasValues.length ? Math.round(Math.max(...gasValues)) : null,
    },
    oracleVerificationRate: clickLogs.length ? Number((oracleVerified.length / clickLogs.length).toFixed(4)) : 0,
    signedDecisionCoverage: validClicks.length ? Number((oracleVerified.length / validClicks.length).toFixed(4)) : 0,
    confirmedSettlements: settlements.filter((item) => item.status === 'confirmed').length,
    simulatedSettlements: settlements.filter((item) => item.status === 'simulated').length,
  }
}

export function buildSecurityAuditReport({
  blockchain = {},
  oracle = {},
  rateLimit = {},
  replayProtection = {},
  contractCapabilities = {},
  store = {},
} = {}) {
  const benchmark = summarizeBenchmarkMetrics(store)
  const controls = [
    {
      key: 'request-validation',
      status: 'enabled',
      detail: 'Input shape, wallet addresses, timestamp format, and bounded text fields are validated before inference.',
    },
    {
      key: 'rate-limiting',
      status: rateLimit.maxRequests ? 'enabled' : 'warning',
      detail: rateLimit.maxRequests
        ? `${rateLimit.maxRequests} validation requests per ${Math.round((rateLimit.windowMs || 0) / 1000)} seconds per client bucket.`
        : 'No click-validation rate limit configured.',
    },
    {
      key: 'replay-protection',
      status: replayProtection.ttlMs ? 'enabled' : 'warning',
      detail: replayProtection.ttlMs
        ? `Duplicate request ids and event fingerprints are rejected for ${Math.round(replayProtection.ttlMs / 1000)} seconds and persisted in the click ledger.`
        : 'Replay protection window is not configured.',
    },
    {
      key: 'oracle-signatures',
      status: oracle.signerCount ? 'enabled' : 'warning',
      detail: oracle.signerCount
        ? `${oracle.signerCount} oracle signer(s) configured with quorum ${oracle.requiredQuorum || 1}.`
        : 'Signed oracle verification is running in local unsigned mode.',
    },
    {
      key: 'event-id-settlement',
      status: contractCapabilities.eventSettlementEnabled ? 'enabled' : 'compatible',
      detail: contractCapabilities.eventSettlementEnabled
        ? 'Backend is configured to call event-id-based contract settlement for on-chain idempotency.'
        : 'Backend can fall back to legacy settlement until the hardened contract is deployed.',
    },
    {
      key: 'blockchain-connectivity',
      status: blockchain.signerConnected ? 'enabled' : 'degraded',
      detail: blockchain.reason || 'Blockchain status unavailable.',
    },
  ]

  const status = controls.some((item) => item.status === 'warning') ? 'attention' : 'ok'
  return {
    status,
    generatedAt: new Date().toISOString(),
    controls,
    oracle,
    rateLimit,
    replayProtection,
    contractCapabilities,
    benchmark,
  }
}