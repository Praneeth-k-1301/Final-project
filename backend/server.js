import cors from 'cors'
import dotenv from 'dotenv'
import express from 'express'
import axios from 'axios'
import { ethers } from 'ethers'
import { promises as fs } from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

import { buildValidationFeatures, stableCid } from './click-utils.js'
import {
		  buildSecurityAuditReport,
		  buildValidationExplanation,
		  describeBlockchainState,
		  hashPayload,
		  normalizeEthValue,
		  rollupCampaignState,
		  summarizeBenchmarkMetrics,
		  summarizeDashboardMetrics,
		  validateClickRequestPayload,
		  verifySignedOracleDecision,
		} from './dashboard-utils.js'
import {
		  checkCidAvailabilityAcrossGateways,
		  getConfiguredGateways,
		  uploadAndPinJsonToIpfs,
		} from './ipfs-utils.js'
import { TOPICS, publishEvent, subscribeToTopic } from './message-queue.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
dotenv.config({ path: path.join(__dirname, '.env') })
dotenv.config({ path: path.join(__dirname, '..', '.env') })

const app = express()
const PORT = Number(process.env.PORT || 3001)
const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5000'
const STORE_PATH = path.join(__dirname, 'data', 'store.json')
const CAMPAIGN_LIMITS = Object.freeze({
  titleMaxLength: 120,
  urlMaxLength: 2048,
  payoutMin: 0.0001,
  payoutMax: 1,
  budgetMin: 0.001,
  budgetMax: 100,
  routingCodeMin: 1,
  routingCodeMax: 1000000,
})
const NETWORK_NAMES = Object.freeze({
  11155111: 'Sepolia',
})
const DEFAULT_CREATIVE_IMAGE = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`
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
const CONTRACT_ABI = [
  'function releasePaymentForEvent(bytes32 eventId, address publisher, uint256 amountWei) public returns (bool)',
  'function releasePayment(address publisher, uint256 amountWei) public returns (bool)',
  'function getBalance() public view returns (uint256)',
  'function isSettled(bytes32 eventId) public view returns (bool)',
  'function owner() public view returns (address)',
]
const RATE_LIMIT_WINDOW_MS = Math.max(5_000, Number(process.env.CLICK_RATE_LIMIT_WINDOW_MS || 60_000))
const RATE_LIMIT_MAX_REQUESTS = Math.max(1, Number(process.env.CLICK_RATE_LIMIT_MAX_REQUESTS || 30))
const REPLAY_TTL_MS = Math.max(30_000, Number(process.env.CLICK_REPLAY_TTL_MS || 15 * 60 * 1000))
const ORACLE_MAX_SKEW_MS = Math.max(5_000, Number(process.env.ORACLE_MAX_SKEW_MS || 30_000))
const EVENT_SETTLEMENT_ENABLED = process.env.CONTRACT_EVENT_SETTLEMENT === 'true'
const FORCE_SIMULATION_MODE = String(process.env.SETTLEMENT_MODE || '').toLowerCase() === 'simulation'
const IPFS_MONITOR_INTERVAL_MS = Math.max(60_000, Number(process.env.IPFS_MONITOR_INTERVAL_MS || 300_000))

let provider = null
let wallet = null
let contract = null
let ethereumInitError = null
const rateLimitBuckets = new Map()
const replayCache = new Map()

// Simple in-process queue so we never try to send multiple settlement
// transactions concurrently from the same backend signer. This reduces
// nonce/"replacement underpriced" issues when traffic is bursty.
let settlementQueueTail = Promise.resolve()

async function runInSettlementQueue(operation) {
	const previous = settlementQueueTail
	settlementQueueTail = (async () => {
		try {
			// Ensure previous settlement (success or failure) completes first.
			await previous
		} catch (error) {
			console.warn('Previous settlement operation failed:', error?.message || error)
		}
		return operation()
	})()
	return settlementQueueTail
}

const ORACLE_SIGNERS = (() => {
  const configured = []
  const raw = String(process.env.ORACLE_SIGNERS || '').trim()
  if (raw) {
    for (const entry of raw.split(',')) {
      const [id, secret] = entry.split(':').map((value) => value?.trim())
      if (id && secret) configured.push({ id, secret })
    }
  } else if (process.env.ORACLE_SHARED_SECRET) {
    configured.push({ id: 'oracle-1', secret: process.env.ORACLE_SHARED_SECRET })
  }
  return configured
})()
const ORACLE_REQUIRED_QUORUM = ORACLE_SIGNERS.length
  ? Math.min(ORACLE_SIGNERS.length, Math.max(1, Number(process.env.ORACLE_QUORUM || 1)))
  : 0

app.use(cors())
app.use(express.json({ limit: '100kb' }))

	const seedCampaign = {
	  id: 'cmp-seeded-demo', cid: 'cid-seeded-demo', title: 'TalkingData Quality Screen Demo',
	  image: DEFAULT_CREATIVE_IMAGE, targetUrl: 'https://example.com',
	  payoutEth: '0.0010', budgetEth: '0.0500', remainingBudgetEth: '0.0500', appCode: 12, osCode: 13,
	  channelCode: 111, advertiserAddress: '', status: 'active', createdAt: new Date().toISOString(),
	  updatedAt: new Date().toISOString(), metrics: { clicks: 0, validClicks: 0, fraudBlocked: 0, settlements: 0 },
	  ipfsCid: null, ipfsStatus: 'unconfigured', ipfsLastCheckedAt: null,
	}

function isLegacyCreativeImage(image) {
  return typeof image === 'string' && image.includes('via.placeholder.com/640x320?text=Decentralized+Ad+Campaign')
}

function isOutdatedDefaultCreative(image) {
  return typeof image === 'string'
    && image.startsWith('data:image/svg+xml')
    && image.includes('AdChain%20Campaign')
    && image.includes('Creative%20preview%20unavailable')
    && !image.includes('text-anchor%3D%22middle%22')
}

function rpcUrl() {
  if (process.env.RPC_URL) return process.env.RPC_URL
  if (process.env.INFURA_API_KEY) return `https://sepolia.infura.io/v3/${process.env.INFURA_API_KEY}`
  return null
}

function normalizeEth(value, fallback) {
  return normalizeEthValue(value ?? fallback, fallback)
}

function sanitizeText(value, maxLength) {
  return String(value ?? '')
    .replace(/[\u0000-\u001F\u007F]/g, '')
    .trim()
    .slice(0, maxLength)
}

function parseFiniteNumber(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function normalizeNetworkName(chainId, networkName) {
  if (NETWORK_NAMES[chainId]) return NETWORK_NAMES[chainId]
  if (networkName && networkName !== 'unknown') return networkName
  return chainId ? `Chain ${chainId}` : null
}

	function extractOnchainErrorDetails(error) {
	  const rawMessage = error?.info?.error?.message
	    || error?.shortMessage
	    || error?.reason
	    || error?.message
	    || 'Unknown on-chain error'
	  const message = String(rawMessage)
	  const lower = message.toLowerCase()
	  let classification = 'unknown'
	  if (lower.includes('insufficientbalance') || lower.includes('insufficient balance')) {
	    classification = 'insufficient-balance'
	  } else if (lower.includes('alreadysettled') || lower.includes('already settled')) {
	    classification = 'already-settled'
	  } else if (lower.includes('nonce') || lower.includes('replacement transaction underpriced')) {
	    classification = 'nonce-conflict'
	  } else if (lower.includes('paused')) {
	    classification = 'paused'
	  } else if (lower.includes('rate limit') || lower.includes('too many requests')) {
	    classification = 'rpc-rate-limit'
	  }
	  const code = error?.code || error?.info?.error?.code || null
	  const hash = error?.transaction?.hash || error?.info?.transactionHash || null
	  return { message, code, classification, transactionHash: hash }
	}

function parseHttpUrl(value) {
  try {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol)) return null
    if (url.username || url.password) return null
    return url.toString()
  } catch {
    return null
  }
}

function parseIntegerInRange(value, fallback, { min, max, label }) {
  const parsed = value === undefined || value === null || value === '' ? Number(fallback) : Number(value)
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    return { error: `${label} must be an integer between ${min} and ${max}` }
  }
  return { value: parsed }
}

function oracleRuntimeSummary(remoteOracle = {}) {
  return {
    mode: remoteOracle.mode || (ORACLE_SIGNERS.length ? 'hmac-quorum' : 'unsigned-local'),
    signerCount: remoteOracle.signerCount ?? ORACLE_SIGNERS.length,
    requiredQuorum: remoteOracle.requiredQuorum ?? ORACLE_REQUIRED_QUORUM,
    maxSkewMs: ORACLE_MAX_SKEW_MS,
  }
}

function rateLimitKey(req, safePayload, walletAddress) {
  return [
    req.ip || req.headers['x-forwarded-for'] || 'unknown',
    safePayload.visitorId || walletAddress || safePayload.ipAddress || 'anonymous',
    safePayload.campaignId,
  ].join(':')
}

function applyClickRateLimit(key, now = Date.now()) {
  const bucket = (rateLimitBuckets.get(key) || []).filter((value) => now - value < RATE_LIMIT_WINDOW_MS)
  if (bucket.length >= RATE_LIMIT_MAX_REQUESTS) {
    const retryAfterMs = Math.max(0, RATE_LIMIT_WINDOW_MS - (now - bucket[0]))
    rateLimitBuckets.set(key, bucket)
    return { allowed: false, retryAfterMs, limit: RATE_LIMIT_MAX_REQUESTS }
  }
  bucket.push(now)
  rateLimitBuckets.set(key, bucket)
  return { allowed: true, remaining: Math.max(RATE_LIMIT_MAX_REQUESTS - bucket.length, 0), limit: RATE_LIMIT_MAX_REQUESTS }
}

function buildRequestId(payload, walletAddress) {
  if (payload.requestId) return payload.requestId
  return stableCid(JSON.stringify({
    campaignId: payload.campaignId,
    clickTime: payload.clickTime,
    visitorId: payload.visitorId,
    walletAddress,
    ipAddress: payload.ipAddress,
  }))
}

function buildEventFingerprint(payload, walletAddress) {
  return stableCid(JSON.stringify({
    campaignId: payload.campaignId,
    clickTime: payload.clickTime,
    visitorId: payload.visitorId,
    walletAddress,
    ipAddress: payload.ipAddress,
    userAgent: payload.userAgent,
  }))
}

function replayCacheKeys(requestId, eventFingerprint) {
  return [`request:${requestId}`, `fingerprint:${eventFingerprint}`]
}

function reserveReplayProtection(requestId, eventFingerprint, now = Date.now()) {
  const keys = replayCacheKeys(requestId, eventFingerprint)
  for (const [key, value] of replayCache.entries()) {
    if ((value?.expiresAt || 0) <= now) replayCache.delete(key)
  }
  const existing = keys.map((key) => replayCache.get(key)).find((value) => value && value.expiresAt > now)
  if (existing) return { blocked: true, existing }
  for (const key of keys) {
    replayCache.set(key, { requestId, eventFingerprint, expiresAt: now + REPLAY_TTL_MS, state: 'pending' })
  }
  return { blocked: false, keys }
}

function releaseReplayProtection(keys = []) {
  for (const key of keys) {
    if (replayCache.get(key)?.state === 'pending') replayCache.delete(key)
  }
}

function commitReplayProtection(keys = [], details = {}) {
  const expiresAt = Date.now() + REPLAY_TTL_MS
  for (const key of keys) {
    replayCache.set(key, { ...details, expiresAt, state: 'complete' })
  }
}

function mapCampaignToLegacyAd(campaign) {
  return {
    id: campaign.id,
    cid: campaign.cid,
    title: campaign.title,
    image: campaign.image,
    targetUrl: campaign.targetUrl,
    budgetEth: campaign.budgetEth,
    payoutEth: campaign.payoutEth,
    remainingBudgetEth: campaign.remainingBudgetEth,
    appCode: campaign.appCode,
    osCode: campaign.osCode,
    channelCode: campaign.channelCode,
    advertiserAddress: campaign.advertiserAddress,
    status: campaign.status,
    createdAt: campaign.createdAt,
    updatedAt: campaign.updatedAt,
    metrics: campaign.metrics,
	    ipfsCid: campaign.ipfsCid || null,
	    ipfsStatus: campaign.ipfsStatus || (campaign.ipfsCid ? 'unknown' : 'unconfigured'),
	    ipfsLastCheckedAt: campaign.ipfsLastCheckedAt || null,
  }
}

async function ensureStore() {
  await fs.mkdir(path.dirname(STORE_PATH), { recursive: true })
  try {
    await fs.access(STORE_PATH)
  } catch {
    await fs.writeFile(STORE_PATH, JSON.stringify({ campaigns: [seedCampaign], clickLogs: [], settlements: [] }, null, 2))
  }
}

async function readStore() {
  await ensureStore()
  const contents = await fs.readFile(STORE_PATH, 'utf-8')
  const parsed = JSON.parse(contents)
  const rawClickLogs = parsed.clickLogs || []
  const clickLogs = rawClickLogs.map((entry) => ({
    ...entry,
    explanation: buildValidationExplanation({
      features: entry.features,
      confidence: entry.confidence,
      threshold: entry.threshold,
      valid: entry.valid,
    }),
  }))
	  const campaigns = (parsed.campaigns || []).map((campaign) => {
	    const rollup = rollupCampaignState(campaign, clickLogs)
	    return {
	      ...campaign,
	      image: isLegacyCreativeImage(campaign.image) || isOutdatedDefaultCreative(campaign.image)
	        ? DEFAULT_CREATIVE_IMAGE
	        : campaign.image,
	      remainingBudgetEth: rollup.remainingBudgetEth,
	      metrics: rollup.metrics,
	      status: rollup.status,
	      ipfsCid: campaign.ipfsCid || null,
	      ipfsStatus: campaign.ipfsStatus || (campaign.ipfsCid ? 'unknown' : 'unconfigured'),
	      ipfsLastCheckedAt: campaign.ipfsLastCheckedAt || null,
	    }
	  })
  const settlements = clickLogs
    .filter((entry) => ['confirmed', 'simulated'].includes(entry.settlement?.status))
    .map((entry) => ({
      id: entry.id,
      campaignId: entry.campaignId,
      timestamp: entry.timestamp,
      ...entry.settlement,
    }))
  const normalizedStore = {
    campaigns,
    clickLogs,
    settlements,
  }
  if (JSON.stringify(normalizedStore) !== JSON.stringify({
    campaigns: parsed.campaigns || [],
    clickLogs: parsed.clickLogs || [],
    settlements: parsed.settlements || [],
  })) {
    await writeStore(normalizedStore)
  }
  return normalizedStore
}

async function writeStore(store) {
  await fs.writeFile(STORE_PATH, JSON.stringify(store, null, 2))
}

function createCampaign(payload) {
  const title = sanitizeText(payload.title, CAMPAIGN_LIMITS.titleMaxLength)
  const imageInput = sanitizeText(payload.image, CAMPAIGN_LIMITS.urlMaxLength)
  const targetInput = sanitizeText(payload.targetUrl, CAMPAIGN_LIMITS.urlMaxLength)
  const advertiserAddress = sanitizeText(payload.advertiserAddress, 128)
  const errors = []

  if (title.length < 3) errors.push('Campaign title must be between 3 and 120 characters')

  const image = parseHttpUrl(imageInput)
  if (!image) errors.push('Creative URL must be a valid http or https URL')

  const targetUrl = parseHttpUrl(targetInput)
  if (!targetUrl) errors.push('Destination URL must be a valid http or https URL')

  const payoutValue = parseFiniteNumber(payload.payoutEth)
  if (payoutValue === null || payoutValue < CAMPAIGN_LIMITS.payoutMin || payoutValue > CAMPAIGN_LIMITS.payoutMax) {
    errors.push(`Payout must be between ${CAMPAIGN_LIMITS.payoutMin} and ${CAMPAIGN_LIMITS.payoutMax} ETH`)
  }

  const budgetValue = parseFiniteNumber(payload.budgetEth)
  if (budgetValue === null || budgetValue < CAMPAIGN_LIMITS.budgetMin || budgetValue > CAMPAIGN_LIMITS.budgetMax) {
    errors.push(`Budget must be between ${CAMPAIGN_LIMITS.budgetMin} and ${CAMPAIGN_LIMITS.budgetMax} ETH`)
  }

  if (budgetValue !== null && payoutValue !== null && budgetValue < payoutValue) {
    errors.push('Budget must be greater than or equal to payout per valid click')
  }

  const appCode = parseIntegerInRange(payload.appCode, 12, {
    min: CAMPAIGN_LIMITS.routingCodeMin,
    max: CAMPAIGN_LIMITS.routingCodeMax,
    label: 'App code',
  })
  if (appCode.error) errors.push(appCode.error)

  const osCode = parseIntegerInRange(payload.osCode, 13, {
    min: CAMPAIGN_LIMITS.routingCodeMin,
    max: CAMPAIGN_LIMITS.routingCodeMax,
    label: 'OS code',
  })
  if (osCode.error) errors.push(osCode.error)

  const channelCode = parseIntegerInRange(payload.channelCode, 111, {
    min: CAMPAIGN_LIMITS.routingCodeMin,
    max: CAMPAIGN_LIMITS.routingCodeMax,
    label: 'Channel code',
  })
  if (channelCode.error) errors.push(channelCode.error)

  if (advertiserAddress && !ethers.isAddress(advertiserAddress)) {
    errors.push('Advertiser address must be a valid Ethereum address')
  }

  if (errors.length > 0) return { error: errors[0], details: errors }

	  const createdAt = new Date().toISOString()
	  return {
	    value: {
	      id: `cmp-${stableCid(`${title}-${createdAt}`).slice(-16)}`,
	      cid: stableCid(JSON.stringify({ title, image, targetUrl, payoutEth: payoutValue, budgetEth: budgetValue })),
	      title,
	      image,
	      targetUrl,
	      payoutEth: normalizeEth(payoutValue, 0.001),
	      budgetEth: normalizeEth(budgetValue, 0.05),
	      remainingBudgetEth: normalizeEth(budgetValue, 0.05),
	      appCode: appCode.value,
	      osCode: osCode.value,
	      channelCode: channelCode.value,
	      advertiserAddress: advertiserAddress ? ethers.getAddress(advertiserAddress) : '',
	      status: 'active',
	      createdAt,
	      updatedAt: createdAt,
	      metrics: { clicks: 0, validClicks: 0, fraudBlocked: 0, settlements: 0 },
	      ipfsCid: null,
	      ipfsStatus: 'pending',
	      ipfsLastCheckedAt: null,
	    },
	  }
}

async function fetchModelMetadata() {
  const startedAt = Date.now()
  try {
    const response = await axios.get(`${ML_API_URL}/model-metadata`, { timeout: 4000 })
    return {
      ...response.data,
      status: response.data?.status || 'ok',
      latencyMs: Date.now() - startedAt,
      checkedAt: new Date().toISOString(),
    }
  } catch (error) {
    return {
      status: 'degraded',
      message: 'ML API unavailable',
      model: null,
      threshold: null,
      metrics: { f1: null },
      dataset: { rows: null },
      availableModels: {},
      oracle: oracleRuntimeSummary(),
      latencyMs: null,
      checkedAt: new Date().toISOString(),
      error: error.message,
    }
  }
}

async function initializeEthereum() {
  ethereumInitError = null
  provider = null
  wallet = null
  contract = null
  try {
    if (!rpcUrl() || !process.env.PRIVATE_KEY || !process.env.CONTRACT_ADDRESS) return
    provider = new ethers.JsonRpcProvider(rpcUrl())
    wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider)
    contract = new ethers.Contract(ethers.getAddress(process.env.CONTRACT_ADDRESS), CONTRACT_ABI, wallet)
  } catch (error) {
    provider = null
    wallet = null
    contract = null
    ethereumInitError = error.message
    console.error('Failed to initialize Ethereum connection:', error.message)
  }
}

async function getBlockchainStatus() {
  const contractAddress = process.env.CONTRACT_ADDRESS && ethers.isAddress(process.env.CONTRACT_ADDRESS)
    ? ethers.getAddress(process.env.CONTRACT_ADDRESS)
    : null
  const signerConfigured = Boolean(rpcUrl() && process.env.PRIVATE_KEY && contractAddress)
  const signerConnected = !FORCE_SIMULATION_MODE && Boolean(provider && wallet && contract)
  const status = {
    settlementMode: signerConnected ? 'onchain' : 'simulation',
    connectionStatus: signerConnected ? 'connected' : signerConfigured ? 'disconnected' : 'not-configured',
    network: null,
    chainId: null,
    contractAddress,
    walletAddress: wallet?.address || null,
    gasPriceGwei: null,
    signerConnected,
    rpcConfigured: Boolean(rpcUrl()),
    privateKeyConfigured: Boolean(process.env.PRIVATE_KEY),
    contractConfigured: Boolean(contractAddress),
    signerConfigured,
    eventSettlementEnabled: EVENT_SETTLEMENT_ENABLED,
    initError: ethereumInitError,
  }

  Object.assign(status, describeBlockchainState(status))

  if (!signerConnected) return status

  try {
    const [network, feeData] = await Promise.all([provider.getNetwork(), provider.getFeeData()])
    const chainId = Number(network.chainId)
    const gasPrice = feeData?.gasPrice ?? feeData?.maxFeePerGas ?? null
    status.chainId = chainId
    status.network = normalizeNetworkName(chainId, network.name)
    status.gasPriceGwei = gasPrice ? Number(ethers.formatUnits(gasPrice, 'gwei')).toFixed(2) : null
  } catch (error) {
    status.connectionStatus = 'degraded'
    status.initError = error.message
  }

  return status
}

	async function runIpfsHealthSweep() {
	  const gateways = getConfiguredGateways()
	  if (!gateways.length) return
	  try {
	    const store = await readStore()
	    const campaigns = Array.isArray(store.campaigns) ? store.campaigns : []
	    let changed = false
	    for (const campaign of campaigns) {
	      if (!campaign.ipfsCid) continue
	      const result = await checkCidAvailabilityAcrossGateways(campaign.ipfsCid, { timeoutMs: 5000 })
	      campaign.ipfsLastCheckedAt = result.checkedAt
	      if (!result.gateways?.length && result.status === 'unconfigured') {
	        campaign.ipfsStatus = campaign.ipfsStatus || 'pinned'
	      } else if (result.available) {
	        campaign.ipfsStatus = 'healthy'
	      } else {
	        campaign.ipfsStatus = 'missing'
	      }
	      changed = true
	    }
	    if (changed) await writeStore(store)
	  } catch (error) {
	    console.error('IPFS health sweep failed:', error.message)
	  }
	}

	function startIpfsMonitor() {
	  const gateways = getConfiguredGateways()
	  if (!gateways.length) return
	  runIpfsHealthSweep()
	  setInterval(runIpfsHealthSweep, IPFS_MONITOR_INTERVAL_MS)
	}

async function attemptSettlement(campaign, walletAddress, eventId) {
  if (Number(campaign.remainingBudgetEth) < Number(campaign.payoutEth)) {
    console.warn('Settlement skipped: budget exhausted', {
      campaignId: campaign.id,
      walletAddress,
      payoutEth: campaign.payoutEth,
      remainingBudgetEth: campaign.remainingBudgetEth,
      eventId,
    })
    return { status: 'budget-exhausted', payoutEth: campaign.payoutEth, eventId }
  }
  if (FORCE_SIMULATION_MODE || !contract || !provider) {
    console.warn('Settlement in simulation mode (no contract/provider configured)', {
      campaignId: campaign.id,
      walletAddress,
      payoutEth: campaign.payoutEth,
      eventId,
    })
    return { status: 'simulated', simulationMode: true, payoutEth: campaign.payoutEth, eventId }
  }
  if (!walletAddress) {
    console.warn('Settlement blocked: missing payout wallet', {
      campaignId: campaign.id,
      payoutEth: campaign.payoutEth,
      eventId,
    })
    return { status: 'missing-wallet', payoutEth: campaign.payoutEth, eventId }
  }
	  const start = Date.now()
	  const amountWei = ethers.parseEther(String(campaign.payoutEth))
	  const settlementMethod = EVENT_SETTLEMENT_ENABLED ? 'event-id' : 'legacy'

	  // Quick idempotency check against the contract if event-based settlement is active.
	  if (EVENT_SETTLEMENT_ENABLED && typeof contract.isSettled === 'function') {
	    try {
	      const alreadySettled = await contract.isSettled(eventId)
	      if (alreadySettled) {
        console.warn('Settlement skipped: event already settled on-chain', {
          campaignId: campaign.id,
          walletAddress,
          payoutEth: campaign.payoutEth,
          eventId,
          settlementMethod,
        })
	        return {
	          status: 'already-settled',
	          simulationMode: false,
	          payoutEth: campaign.payoutEth,
	          eventId,
	          settlementMethod,
	        }
	      }
	    } catch (error) {
	      console.error('Settlement isSettled check failed:', extractOnchainErrorDetails(error))
	      // Continue; a failure here should not crash the request, the tx path
	      // below will surface any real on-chain issues.
	    }
	  }

	  try {
	    const result = await runInSettlementQueue(async () => {
	      // 1) Static-call the contract first so we can see *why* a tx would revert
	      //    (e.g., vault out of funds, duplicate event, paused) without sending
	      //    an on-chain transaction that is doomed to fail.
	      try {
	        if (EVENT_SETTLEMENT_ENABLED) {
	          await contract.releasePaymentForEvent.staticCall(eventId, walletAddress, amountWei)
	        } else {
	          await contract.releasePayment.staticCall(walletAddress, amountWei)
	        }
	      } catch (error) {
	        const details = extractOnchainErrorDetails(error)
	        console.error('Settlement static call failed:', {
          stage: 'static-call',
	          eventId,
	          walletAddress,
	          payoutEth: campaign.payoutEth,
	          details,
	        })
	        return { kind: 'static-error', details }
	      }

	      // 2) Static call passed; now send the real transaction.
	      try {
	        const tx = EVENT_SETTLEMENT_ENABLED
	          ? await contract.releasePaymentForEvent(eventId, walletAddress, amountWei)
	          : await contract.releasePayment(walletAddress, amountWei)
	        const receipt = await tx.wait()
	        return { kind: 'tx', tx, receipt }
	      } catch (error) {
	        const details = extractOnchainErrorDetails(error)
	        console.error('Settlement transaction failed:', {
          stage: 'send-tx',
	          eventId,
	          walletAddress,
	          payoutEth: campaign.payoutEth,
	          details,
	        })
	        return { kind: 'tx-error', details }
	      }
	    })

	    if (result.kind === 'static-error') {
	      // Classify well-known static-call failures into user-facing statuses
	      const { classification, message, code } = result.details
	      if (classification === 'insufficient-balance') {
	        // Contract vault has run out of free balance to back new payouts.
	        return {
	          status: 'vault-exhausted',
	          simulationMode: false,
	          payoutEth: campaign.payoutEth,
	          eventId,
	          settlementMethod,
	          errorCode: code,
	          errorMessage: message,
	        }
	      }
	      if (classification === 'already-settled') {
	        return {
	          status: 'already-settled',
	          simulationMode: false,
	          payoutEth: campaign.payoutEth,
	          eventId,
	          settlementMethod,
	          errorCode: code,
	          errorMessage: message,
	        }
	      }
	      if (classification === 'paused') {
	        return {
	          status: 'contract-paused',
	          simulationMode: false,
	          payoutEth: campaign.payoutEth,
	          eventId,
	          settlementMethod,
	          errorCode: code,
	          errorMessage: message,
	        }
	      }
	      // Fallback classification for other static-call failures.
	      return {
	        status: 'failed',
	        simulationMode: false,
	        payoutEth: campaign.payoutEth,
	        eventId,
	        settlementMethod,
	        errorCode: result.details.code,
	        errorMessage: result.details.message,
	      }
	    }

	    if (result.kind === 'tx-error') {
	      return {
	        status: 'failed',
	        simulationMode: false,
	        payoutEth: campaign.payoutEth,
	        eventId,
	        settlementMethod,
	        errorCode: result.details.code,
	        errorMessage: result.details.message,
	        transactionHash: result.details.transactionHash || null,
	      }
	    }

	    if (result.kind !== 'tx') {
	      return {
	        status: 'failed',
	        simulationMode: false,
	        payoutEth: campaign.payoutEth,
	        eventId,
	        settlementMethod,
	        errorMessage: 'Unknown settlement result',
	      }
	    }

	    const { tx, receipt } = result
	    if (!receipt || Number(receipt.status) !== 1) {
	      return {
	        status: 'failed',
	        simulationMode: false,
	        payoutEth: campaign.payoutEth,
	        eventId,
	        settlementMethod,
	        transactionHash: tx?.hash || null,
	        errorMessage: 'Transaction mined with unsuccessful status',
	      }
	    }

	    return {
	      status: 'confirmed',
	      simulationMode: false,
	      payoutEth: campaign.payoutEth,
	      eventId,
	      settlementMethod,
	      transactionHash: tx.hash,
	      blockNumber: receipt.blockNumber,
	      gasUsed: receipt.gasUsed?.toString() || null,
	      settlementLatencyMs: Date.now() - start,
	      explorerUrl: `https://sepolia.etherscan.io/tx/${tx.hash}`,
	    }
	  } catch (error) {
	    const details = extractOnchainErrorDetails(error)
	    console.error('Unexpected settlement error:', {
	      eventId,
	      walletAddress,
	      payoutEth: campaign.payoutEth,
	      details,
	    })
	    return {
	      status: 'failed',
	      simulationMode: false,
	      payoutEth: campaign.payoutEth,
	      eventId,
	      settlementMethod,
	      errorCode: details.code,
	      errorMessage: details.message,
	      transactionHash: details.transactionHash || null,
	    }
	  }
}

function settlementUserMessage(settlement, isValid) {
  if (!settlement) {
    return isValid
      ? 'Click validated; awaiting settlement result'
      : 'Suspicious click blocked';
  }

  const status = settlement.status || 'unknown';

  if (status === 'budget-exhausted') {
    return 'Campaign budget exhausted for this payout.';
  }
  if (status === 'already-settled') {
    return 'Click already processed; duplicate settlement skipped.';
  }
  if (status === 'missing-wallet') {
    return 'Connect a payout wallet to settle this click on-chain.';
  }
  if (status === 'vault-exhausted') {
    return 'Settlement vault has insufficient balance to fund this payout.';
  }
  if (status === 'contract-paused') {
    return 'Settlement contract is currently paused; settlements are temporarily disabled.';
  }
  if (status === 'simulated') {
    return isValid
      ? 'Click validated and recorded in simulation mode.'
      : 'Suspicious click blocked in simulation mode.';
  }
  if (status === 'confirmed') {
    return 'Click validated and settled on-chain.';
  }
  if (status === 'blocked') {
    return 'Suspicious click blocked by the fraud model.';
  }
  if (status === 'failed') {
    return 'On-chain settlement failed; please check logs or retry later.';
  }

  return isValid
    ? 'Click validated; settlement status unknown.'
    : 'Suspicious click blocked.';
}

function startClickPipelineWorkers() {
	  // Stage 1: raw click ingestion -> feature extraction
	  subscribeToTopic(TOPICS.RAW_CLICKS, async (event) => {
	    let keys = event.reservedReplayKeys || []
	    try {
	      const store = await readStore()
	      const campaign = store.campaigns.find((item) => item.id === event.campaignId || item.cid === event.campaignId)
	      if (!campaign) {
	        console.warn('Async click worker: campaign not found for event', event.campaignId)
	        releaseReplayProtection(keys)
	        return
	      }
	      const features = buildValidationFeatures({
	        campaign,
	        payload: { ...event.payload, walletAddress: event.walletAddress },
	        history: store.clickLogs,
	      })
	      const enriched = {
	        ...event,
	        campaignId: campaign.id,
	        campaignTitle: campaign.title,
	        features,
	        eventId: event.eventId || ethers.keccak256(ethers.toUtf8Bytes(event.requestId)),
	      }
	      await publishEvent(TOPICS.FEATURES, enriched)
	    } catch (error) {
	      console.error('Async click worker (features) failed:', error)
	      releaseReplayProtection(keys)
	    }
	  })

	  // Stage 2: features -> ML validation
	  subscribeToTopic(TOPICS.FEATURES, async (message) => {
	    let keys = message.reservedReplayKeys || []
	    try {
	      const { requestId, features } = message
	      const mlStartedAt = Date.now()
	      const mlResponse = await axios.post(`${ML_API_URL}/predict-signed`, { requestId, features }, { timeout: 8000 })
	      const mlLatencyMs = Date.now() - mlStartedAt
	      const oracleVerification = verifySignedOracleDecision(mlResponse.data, {
	        signers: ORACLE_SIGNERS,
	        requiredQuorum: ORACLE_REQUIRED_QUORUM,
	        maxSkewMs: ORACLE_MAX_SKEW_MS,
	        expectedFeaturesHash: hashPayload(features),
	      })
	      if (!oracleVerification.verified) {
	        console.warn('Async ML oracle verification failed; continuing with unsigned decision', {
	          requestId,
	          reason: oracleVerification.reason,
	          matchedSigners: oracleVerification.matchedSigners,
	          quorum: oracleVerification.quorum,
	        })
	      }
	      const isValid = Number(mlResponse.data.result) === 1
	      const mlDecision = {
	        result: Number(mlResponse.data.result),
	        confidence: mlResponse.data.confidence,
	        threshold: mlResponse.data.threshold,
	        model: mlResponse.data.model,
	        oracle: mlResponse.data.oracle,
	        signatures: mlResponse.data.signatures,
	      }
	      await publishEvent(TOPICS.VALIDATED, {
	        ...message,
	        isValid,
	        mlLatencyMs,
	        mlDecision,
	        oracleVerification,
	      })
	    } catch (error) {
	      console.error('Async click worker (ML validation) failed:', error)
	      releaseReplayProtection(keys)
	    }
	  })

	  // Stage 3: validated decision -> settlement + store write
	  subscribeToTopic(TOPICS.VALIDATED, async (message) => {
	    let keys = message.reservedReplayKeys || []
	    try {
	      const store = await readStore()
	      const campaign = store.campaigns.find((item) => item.id === message.campaignId || item.cid === message.campaignId)
	      if (!campaign) {
	        console.warn('Async click worker: campaign not found at settlement stage for event', message.campaignId)
	        releaseReplayProtection(keys)
	        return
	      }
	      const eventId = message.eventId || ethers.keccak256(ethers.toUtf8Bytes(message.requestId))
	      const settlement = message.isValid
	        ? await attemptSettlement(campaign, message.walletAddress, eventId)
	        : { status: 'blocked', payoutEth: campaign.payoutEth, eventId }
	      const timestamp = new Date().toISOString()
	      const explanation = buildValidationExplanation({
	        features: message.features,
	        confidence: message.mlDecision.confidence,
	        threshold: message.mlDecision.threshold,
	        valid: message.isValid,
	      })
	      const validation = {
	        id: stableCid(`${campaign.id}-${message.requestId}`),
	        campaignId: campaign.id,
	        campaignTitle: campaign.title,
	        requestId: message.requestId,
	        eventFingerprint: message.eventFingerprint,
	        eventId,
	        visitorKey: message.features.visitorKey,
	        walletAddress: message.walletAddress,
	        timestamp: message.features.timestamp,
	        processedAt: timestamp,
	        app: message.features.app,
	        valid: message.isValid,
	        confidence: message.mlDecision.confidence,
	        threshold: message.mlDecision.threshold,
	        model: message.mlDecision.model,
	        mlLatencyMs: message.mlLatencyMs,
	        validationLatencyMs: Date.now() - (message.requestStartedAt || Date.now()),
	        features: message.features,
	        explanation,
	        oracle: {
	          ...oracleRuntimeSummary(message.mlDecision.oracle),
	          verified: message.oracleVerification.verified,
	          matchedSigners: message.oracleVerification.matchedSigners,
	          signatureCount: Array.isArray(message.mlDecision.signatures) ? message.mlDecision.signatures.length : 0,
	        },
	        replayProtection: {
	          ttlMs: REPLAY_TTL_MS,
	        },
	        settlement,
	      }

	      store.clickLogs.unshift(validation)
	      if (['confirmed', 'simulated'].includes(settlement.status)) {
	        store.settlements.unshift({
	          id: validation.id,
	          campaignId: campaign.id,
	          timestamp: validation.timestamp,
	          ...settlement,
	        })
	      }
	      const rollup = rollupCampaignState(campaign, store.clickLogs)
	      campaign.metrics = rollup.metrics
	      campaign.remainingBudgetEth = rollup.remainingBudgetEth
	      campaign.status = rollup.status
	      campaign.updatedAt = timestamp
	      await writeStore(store)
	      commitReplayProtection(keys, {
	        requestId: message.requestId,
	        eventFingerprint: message.eventFingerprint,
	        validationId: validation.id,
	      })
	      keys = []
	    } catch (error) {
	      console.error('Async click worker (settlement) failed:', error)
	      releaseReplayProtection(keys)
	    }
	  })
}

app.get('/health', async (req, res) => {
  const [modelMetadata, blockchain] = await Promise.all([fetchModelMetadata(), getBlockchainStatus()])
  const checkedAt = new Date().toISOString()
  res.json({
    status: 'ok',
    checkedAt,
    contractConfigured: blockchain.contractConfigured,
    walletAddress: blockchain.walletAddress,
    rpcConfigured: blockchain.rpcConfigured,
    mlApi: modelMetadata.status || 'unknown',
    mlApiLatencyMs: modelMetadata.latencyMs ?? null,
    settlementMode: blockchain.settlementMode,
    backend: {
      status: 'ok',
      uptimeSec: Math.round(process.uptime()),
      checkedAt,
    },
    services: {
      mlApi: {
        status: modelMetadata.status || 'unknown',
        latencyMs: modelMetadata.latencyMs ?? null,
        message: modelMetadata.message || null,
        checkedAt: modelMetadata.checkedAt || checkedAt,
        oracle: modelMetadata.oracle || oracleRuntimeSummary(),
      },
    },
    security: {
      oracle: oracleRuntimeSummary(modelMetadata.oracle),
      rateLimit: {
        windowMs: RATE_LIMIT_WINDOW_MS,
        maxRequests: RATE_LIMIT_MAX_REQUESTS,
      },
      replayProtection: {
        ttlMs: REPLAY_TTL_MS,
      },
    },
    blockchain,
  })
})

	app.get('/ipfs/health', async (req, res) => {
	  const store = await readStore()
	  const campaigns = Array.isArray(store.campaigns) ? store.campaigns : []
	  const withIpfs = campaigns.filter((campaign) => campaign.ipfsCid)
	  let healthy = 0
	  let missing = 0
	  let other = 0
	  let lastCheckedAt = null
	  for (const campaign of withIpfs) {
	    if (campaign.ipfsLastCheckedAt && (!lastCheckedAt || campaign.ipfsLastCheckedAt > lastCheckedAt)) {
	      lastCheckedAt = campaign.ipfsLastCheckedAt
	    }
	    const status = campaign.ipfsStatus || 'unknown'
	    if (status === 'healthy' || status === 'pinned') healthy += 1
	    else if (status === 'missing') missing += 1
	    else other += 1
	  }
	  res.json({
	    totalCampaigns: campaigns.length,
	    trackedCampaigns: withIpfs.length,
	    healthy,
	    missing,
	    other,
	    lastCheckedAt,
	  })
	})

	app.get(['/campaigns', '/get-ads'], async (req, res) => {
	  const store = await readStore()
	  res.json(store.campaigns.map(mapCampaignToLegacyAd))
	})

app.get('/clicks', async (req, res) => {
  const store = await readStore()
  res.json(store.clickLogs.slice(0, 50))
})

app.get('/metrics', async (req, res) => {
		const store = await readStore()
		const dashboard = summarizeDashboardMetrics(store)
		const benchmark = summarizeBenchmarkMetrics(store)
		
		res.json({
			...dashboard,
			mlLatencyMs: benchmark.mlLatencyMs,
			validationLatencyMs: benchmark.validationLatencyMs,
			settlementLatencyMs: benchmark.settlementLatencyMs,
		})
	})

app.get('/ml/metadata', async (req, res) => {
  res.json(await fetchModelMetadata())
})

app.get('/benchmarks/report', async (req, res) => {
  const store = await readStore()
  const blockchain = await getBlockchainStatus()
  res.json({
    status: 'ok',
    generatedAt: new Date().toISOString(),
    runtime: summarizeBenchmarkMetrics(store),
    blockchain,
  })
})

app.get('/security/report', async (req, res) => {
  const [store, blockchain, modelMetadata] = await Promise.all([readStore(), getBlockchainStatus(), fetchModelMetadata()])
  res.json(buildSecurityAuditReport({
    store,
    blockchain,
    oracle: oracleRuntimeSummary(modelMetadata.oracle),
    rateLimit: { windowMs: RATE_LIMIT_WINDOW_MS, maxRequests: RATE_LIMIT_MAX_REQUESTS },
    replayProtection: { ttlMs: REPLAY_TTL_MS },
    contractCapabilities: {
      eventSettlementEnabled: EVENT_SETTLEMENT_ENABLED,
      contractConfigured: blockchain.contractConfigured,
    },
  }))
})

	app.get('/ipfs/:cid', async (req, res) => {
	  const gateways = getConfiguredGateways()
	  if (!gateways.length) {
	    return res.status(503).json({ message: 'No IPFS gateways configured on backend' })
	  }
	  const cid = req.params.cid
	  let lastError = null
	  for (const gateway of gateways) {
	    const url = `${gateway.replace(/\/+$/u, '')}/${encodeURIComponent(cid)}`
	    try {
	      const response = await axios.get(url, { responseType: 'stream', timeout: 10_000 })
	      res.status(response.status)
	      for (const [header, value] of Object.entries(response.headers || {})) {
	        if (typeof value !== 'undefined') res.setHeader(header, value)
	      }
	      response.data.pipe(res)
	      return
	    } catch (error) {
	      lastError = error
	    }
	  }
	  return res.status(502).json({
	    message: 'Unable to retrieve CID from any configured gateway',
	    error: lastError ? lastError.message : 'Unknown error',
	  })
	})

app.post('/ml/drift/report', async (req, res) => {
  const store = await readStore()
  const samples = Array.isArray(req.body?.samples) && req.body.samples.length
    ? req.body.samples
    : store.clickLogs.slice(0, Math.max(1, Number(req.body?.sampleSize || 25))).map((entry) => entry.features)
  if (!samples.length) return res.status(404).json({ message: 'No feature samples available' })
  try {
    const response = await axios.post(`${ML_API_URL}/drift/report`, { samples }, { timeout: 8000 })
    return res.json(response.data)
  } catch (error) {
    return res.status(error.response?.status || 500).json({
      message: error.response?.data?.message || 'Unable to fetch drift report',
      error: error.message,
    })
  }
})

app.post('/ml/adversarial-evaluation', async (req, res) => {
  const store = await readStore()
  const sample = req.body?.sample || req.body?.features || store.clickLogs[0]?.features
  if (!sample) return res.status(404).json({ message: 'No feature sample available' })
  try {
    const response = await axios.post(`${ML_API_URL}/adversarial-evaluation`, sample, { timeout: 8000 })
    return res.json(response.data)
  } catch (error) {
    return res.status(error.response?.status || 500).json({
      message: error.response?.data?.message || 'Unable to fetch adversarial evaluation',
      error: error.message,
    })
  }
})

app.post(['/campaigns', '/upload-ad'], async (req, res) => {
	  const campaignResult = createCampaign(req.body)
	  if (campaignResult.error) {
	    return res.status(400).json({ message: campaignResult.error, details: campaignResult.details })
	  }
	  const store = await readStore()
	  const campaign = campaignResult.value
	  let ipfsSummary = null
	  try {
	    const metadata = {
	      id: campaign.id,
	      cid: campaign.cid,
	      title: campaign.title,
	      image: campaign.image,
	      targetUrl: campaign.targetUrl,
	      payoutEth: campaign.payoutEth,
	      budgetEth: campaign.budgetEth,
	      createdAt: campaign.createdAt,
	    }
	    ipfsSummary = await uploadAndPinJsonToIpfs(metadata)
	    if (ipfsSummary?.cid) {
	      campaign.ipfsCid = ipfsSummary.cid
	      campaign.ipfsStatus = ipfsSummary.status || 'pinned'
	      campaign.ipfsLastCheckedAt = ipfsSummary.checkedAt || new Date().toISOString()
	    } else if (ipfsSummary?.status === 'unconfigured') {
	      campaign.ipfsStatus = 'unconfigured'
	    } else if (ipfsSummary?.status === 'pin-failed') {
	      campaign.ipfsStatus = 'pin-failed'
	      campaign.ipfsLastCheckedAt = ipfsSummary.checkedAt || new Date().toISOString()
	    }
	  } catch (error) {
	    console.error('IPFS upload failed for campaign', campaign.id, error.message)
	    if (!campaign.ipfsStatus) {
	      campaign.ipfsStatus = 'pin-failed'
	      campaign.ipfsLastCheckedAt = new Date().toISOString()
	    }
	  }
	  store.campaigns.unshift(campaign)
	  await writeStore(store)
	  const message = campaign.ipfsCid
	    ? 'Campaign stored and pinned to IPFS'
	    : ipfsSummary?.status === 'unconfigured'
	      ? 'Campaign stored; IPFS not configured on backend'
	      : 'Campaign stored; IPFS pinning unavailable'
	  return res.status(201).json({ message, campaign })
	})

app.post(['/clicks/validate', '/simulate-click'], async (req, res) => {
  let reservedReplayKeys = []
  try {
    const requestStartedAt = Date.now()
    const requestValidation = validateClickRequestPayload(req.body)
    if (requestValidation.error) {
      return res.status(400).json({ message: requestValidation.error })
    }

    const safePayload = requestValidation.value
    const store = await readStore()
    const campaignId = safePayload.campaignId
    const campaign = store.campaigns.find((item) => item.id === campaignId || item.cid === campaignId)
    if (!campaign) return res.status(404).json({ message: 'Campaign not found' })

    const walletAddress = typeof safePayload.walletAddress === 'string' && ethers.isAddress(safePayload.walletAddress)
      ? ethers.getAddress(safePayload.walletAddress)
      : ''
    const requestId = buildRequestId(safePayload, walletAddress)
    const eventFingerprint = buildEventFingerprint(safePayload, walletAddress)
    const existingValidation = store.clickLogs.find((entry) => entry.requestId === requestId || entry.eventFingerprint === eventFingerprint)
    if (existingValidation) {
      return res.status(409).json({
        message: 'Click already processed',
        duplicateOf: existingValidation.id,
        validation: existingValidation,
      })
    }

    const rateLimit = applyClickRateLimit(rateLimitKey(req, safePayload, walletAddress))
    if (!rateLimit.allowed) {
      res.set('Retry-After', String(Math.ceil(rateLimit.retryAfterMs / 1000)))
      return res.status(429).json({
        message: 'Rate limit exceeded for click validation',
        retryAfterMs: rateLimit.retryAfterMs,
        limit: rateLimit.limit,
      })
    }

    const replayReservation = reserveReplayProtection(requestId, eventFingerprint)
    if (replayReservation.blocked) {
      const existing = replayReservation.existing || {}
      const state = existing.state || 'unknown'
      const duplicateId = existing.validationId || existing.requestId || null
      const friendlyMessage = state === 'pending'
        ? 'Transaction pending, please wait'
        : 'Click already processed'
      return res.status(409).json({
        message: friendlyMessage,
        duplicateOf: duplicateId,
        state,
      })
    }
    reservedReplayKeys = replayReservation.keys

    const features = buildValidationFeatures({ campaign, payload: { ...safePayload, walletAddress }, history: store.clickLogs })
    const eventId = ethers.keccak256(ethers.toUtf8Bytes(requestId))
    console.info('Click validation started', {
      requestId,
      eventId,
      campaignId: campaign.id,
      walletAddress,
    })
    const mlStartedAt = Date.now()
    const mlResponse = await axios.post(`${ML_API_URL}/predict-signed`, { requestId, features }, { timeout: 8000 })
    const mlLatencyMs = Date.now() - mlStartedAt
    const oracleVerification = verifySignedOracleDecision(mlResponse.data, {
      signers: ORACLE_SIGNERS,
      requiredQuorum: ORACLE_REQUIRED_QUORUM,
      maxSkewMs: ORACLE_MAX_SKEW_MS,
      expectedFeaturesHash: hashPayload(features),
    })
    if (!oracleVerification.verified) {
      console.warn('Synchronous ML oracle verification failed; continuing with unsigned decision', {
        requestId,
        eventId,
        campaignId: campaign.id,
        walletAddress,
        reason: oracleVerification.reason,
        matchedSigners: oracleVerification.matchedSigners,
        quorum: oracleVerification.quorum,
      })
    }
    console.info('ML inference completed', {
      requestId,
      eventId,
      campaignId: campaign.id,
      walletAddress,
      latencyMs: mlLatencyMs,
      result: mlResponse.data?.result,
      confidence: mlResponse.data?.confidence,
      threshold: mlResponse.data?.threshold,
      model: mlResponse.data?.model,
      oracleVerified: oracleVerification.verified,
    })
    const isValid = Number(mlResponse.data.result) === 1
    const settlement = isValid
      ? await attemptSettlement(campaign, walletAddress, eventId)
      : { status: 'blocked', payoutEth: campaign.payoutEth, eventId }
    console.info('Settlement attempt completed', {
      requestId,
      eventId,
      campaignId: campaign.id,
      walletAddress,
      status: settlement.status,
      simulationMode: settlement.simulationMode ?? false,
      transactionHash: settlement.transactionHash || null,
      errorCode: settlement.errorCode || null,
    })
    const explanation = buildValidationExplanation({
      features,
      confidence: mlResponse.data.confidence,
      threshold: mlResponse.data.threshold,
      valid: isValid,
    })
    const timestamp = new Date().toISOString()

    const validation = {
      id: stableCid(`${campaign.id}-${requestId}`),
      campaignId: campaign.id,
      campaignTitle: campaign.title,
      requestId,
      eventFingerprint,
      eventId,
      visitorKey: features.visitorKey,
      walletAddress,
      timestamp: features.timestamp,
      processedAt: timestamp,
      app: features.app,
      valid: isValid,
      confidence: mlResponse.data.confidence,
      threshold: mlResponse.data.threshold,
      model: mlResponse.data.model,
      mlLatencyMs,
      validationLatencyMs: Date.now() - requestStartedAt,
      features,
      explanation,
      oracle: {
        ...oracleRuntimeSummary(mlResponse.data.oracle),
        verified: oracleVerification.verified,
        matchedSigners: oracleVerification.matchedSigners,
        signatureCount: Array.isArray(mlResponse.data.signatures) ? mlResponse.data.signatures.length : 0,
      },
      replayProtection: {
        ttlMs: REPLAY_TTL_MS,
      },
      settlement,
    }

    store.clickLogs.unshift(validation)
    if (['confirmed', 'simulated'].includes(settlement.status)) {
      store.settlements.unshift({ id: validation.id, campaignId: campaign.id, timestamp: validation.timestamp, ...settlement })
    }
    const rollup = rollupCampaignState(campaign, store.clickLogs)
    campaign.metrics = rollup.metrics
    campaign.remainingBudgetEth = rollup.remainingBudgetEth
    campaign.status = rollup.status
    campaign.updatedAt = timestamp
    await writeStore(store)
    commitReplayProtection(reservedReplayKeys, {
      requestId,
      eventFingerprint,
      validationId: validation.id,
    })
    reservedReplayKeys = []

    return res.json({
      message: settlementUserMessage(settlement, isValid),
      valid: isValid,
      validation,
      settlement,
      campaign,
	      metrics: summarizeDashboardMetrics(store),
	      // Per-request latency metrics in milliseconds for the latest validation
	      mlLatency: mlLatencyMs,
	      backendLatency: validation.validationLatencyMs,
	      settlementLatency: settlement && typeof settlement.settlementLatencyMs === 'number'
	        ? settlement.settlementLatencyMs
	        : null,
    })
  } catch (error) {
    releaseReplayProtection(reservedReplayKeys)
    const statusCode = error.response?.status || 500
    const axiosCode = error.code || error.response?.code || null
    const isTimeout = axiosCode === 'ECONNABORTED' || String(error.message || '').toLowerCase().includes('timeout')
    const isMlRequest = Boolean(error.config?.url && String(error.config.url).includes('/predict-signed'))

    console.error('Click validation failed:', {
      statusCode,
      axiosCode,
      isTimeout,
      isMlRequest,
      message: error.message,
      stack: error.stack,
      responseStatus: error.response?.status,
      responseData: error.response?.data,
    })

    // Friendly error messages for common failure modes
    if (error.response?.status === 409 && error.response?.data?.message) {
      // Duplicate / replay scenarios already mapped server-side.
      return res.status(409).json({
        message: error.response.data.message,
        error: error.message,
        details: error.response.data,
      })
    }

    if (error.response?.status === 429) {
      return res.status(429).json({
        message: 'Transaction rate limit exceeded, please wait before retrying.',
        error: error.message,
        details: error.response?.data || null,
      })
    }

    if (isTimeout && isMlRequest) {
      return res.status(504).json({
        message: 'Fraud-screening service took too long to respond. Please try again in a moment.',
        error: error.message,
        details: null,
      })
    }

    return res.status(statusCode).json({
      message: error.response?.data?.message || 'Backend error during click validation. Please try again later.',
      error: error.message,
      details: error.response?.data || null,
    })
  }
})

app.post('/clicks/ingest-async', async (req, res) => {
	  let reservedReplayKeys = []
	  try {
	    const requestStartedAt = Date.now()
	    const requestValidation = validateClickRequestPayload(req.body)
	    if (requestValidation.error) {
	      return res.status(400).json({ message: requestValidation.error })
	    }

	    const safePayload = requestValidation.value
	    const store = await readStore()
	    const campaignId = safePayload.campaignId
	    const campaign = store.campaigns.find((item) => item.id === campaignId || item.cid === campaignId)
	    if (!campaign) return res.status(404).json({ message: 'Campaign not found' })

	    const walletAddress = typeof safePayload.walletAddress === 'string' && ethers.isAddress(safePayload.walletAddress)
	      ? ethers.getAddress(safePayload.walletAddress)
	      : ''
	    const requestId = buildRequestId(safePayload, walletAddress)
	    const eventFingerprint = buildEventFingerprint(safePayload, walletAddress)
	    const existingValidation = store.clickLogs.find((entry) =>
	      entry.requestId === requestId || entry.eventFingerprint === eventFingerprint,
	    )
	    if (existingValidation) {
	      return res.status(409).json({
	        message: 'Duplicate click validation request blocked',
	        duplicateOf: existingValidation.id,
	        validation: existingValidation,
	      })
	    }

	    const rateLimit = applyClickRateLimit(rateLimitKey(req, safePayload, walletAddress))
	    if (!rateLimit.allowed) {
	      res.set('Retry-After', String(Math.ceil(rateLimit.retryAfterMs / 1000)))
	      return res.status(429).json({
	        message: 'Rate limit exceeded for click validation',
	        retryAfterMs: rateLimit.retryAfterMs,
	        limit: rateLimit.limit,
	      })
	    }

	    const replayReservation = reserveReplayProtection(requestId, eventFingerprint)
	    if (replayReservation.blocked) {
	      return res.status(409).json({
	        message: 'Replay protection blocked a duplicate validation request',
	        duplicateOf: replayReservation.existing?.validationId || replayReservation.existing?.requestId || null,
	      })
	    }
	    reservedReplayKeys = replayReservation.keys

	    const eventId = ethers.keccak256(ethers.toUtf8Bytes(requestId))
	    await publishEvent(TOPICS.RAW_CLICKS, {
	      requestId,
	      eventFingerprint,
	      eventId,
	      campaignId: campaign.id,
	      walletAddress,
	      payload: safePayload,
	      reservedReplayKeys,
	      requestStartedAt,
	    })

	    return res.status(202).json({
	      message: 'Click accepted for asynchronous validation',
	      requestId,
	      eventId,
	      campaignId: campaign.id,
	    })
	  } catch (error) {
	    releaseReplayProtection(reservedReplayKeys)
	    const statusCode = error.response?.status || 500
	    console.error('Async click ingestion failed:', {
	      statusCode,
	      message: error.message,
	      stack: error.stack,
	      responseStatus: error.response?.status,
	      responseData: error.response?.data,
	    })
	    return res.status(statusCode).json({
	      message: error.response?.data?.message || 'Backend error during async click ingestion. Please try again later.',
	      error: error.message,
	      details: error.response?.data || null,
	    })
	  }
})

await ensureStore()
await initializeEthereum()
startIpfsMonitor()
startClickPipelineWorkers()

app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`)
  console.log(`Health check: http://localhost:${PORT}/health`)
})
