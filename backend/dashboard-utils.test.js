import test from 'node:test'
import assert from 'node:assert/strict'
import crypto from 'crypto'

import {
  buildSecurityAuditReport,
  buildOracleSignatureMessage,
  buildValidationExplanation,
  describeBlockchainState,
  rollupCampaignState,
  summarizeBenchmarkMetrics,
  summarizeDashboardMetrics,
  validateClickRequestPayload,
  verifySignedOracleDecision,
} from './dashboard-utils.js'

test('validateClickRequestPayload trims normalized fields', () => {
  const result = validateClickRequestPayload({
    campaignId: ' cmp-1 ',
    walletAddress: '0x000000000000000000000000000000000000dEaD',
    visitorId: ' visitor-1 ',
    deviceType: ' Mobile ',
  })

  assert.equal(result.error, undefined)
  assert.equal(result.value.campaignId, 'cmp-1')
  assert.equal(result.value.visitorId, 'visitor-1')
  assert.equal(result.value.requestId, '')
  assert.equal(result.value.deviceType, 'mobile')
})

test('validateClickRequestPayload trims request ids', () => {
  const result = validateClickRequestPayload({
    campaignId: 'cmp-1',
    requestId: ' req-123 ',
  })

  assert.equal(result.error, undefined)
  assert.equal(result.value.requestId, 'req-123')
})

test('validateClickRequestPayload rejects invalid wallet addresses', () => {
  const result = validateClickRequestPayload({ campaignId: 'cmp-1', walletAddress: 'not-a-wallet' })
  assert.equal(result.error, 'Wallet address must be a valid Ethereum address')
})

test('rollupCampaignState counts confirmed settlements separately from simulated ones', () => {
  const campaign = { id: 'cmp-1', payoutEth: '0.0010', budgetEth: '0.0020', status: 'active' }
  const clickLogs = [
    { campaignId: 'cmp-1', valid: true, settlement: { status: 'confirmed' } },
    { campaignId: 'cmp-1', valid: true, settlement: { status: 'confirmed' } },
    { campaignId: 'cmp-1', valid: true, settlement: { status: 'simulated' } },
    { campaignId: 'cmp-1', valid: false, settlement: { status: 'blocked' } },
  ]

  const result = rollupCampaignState(campaign, clickLogs)

  assert.equal(result.remainingBudgetEth, '0.0000')
  assert.equal(result.metrics.settlements, 2)
  assert.equal(result.metrics.confirmedSettlements, 2)
  assert.equal(result.metrics.simulatedSettlements, 1)
  assert.equal(result.metrics.recordedSettlements, 3)
  assert.equal(result.status, 'budget-exhausted')
})

test('buildValidationExplanation returns risk reasons and blockchain state exposes missing config', () => {
  const explanation = buildValidationExplanation({
    valid: false,
    confidence: 0.21,
    threshold: 0.6,
    features: { burstScore: 0.031, timeInterval: 900, ipAppCount: 4, clickFrequency: 5 },
  })
  const blockchainState = describeBlockchainState({
    signerConnected: false,
    rpcConfigured: false,
    privateKeyConfigured: true,
    contractConfigured: false,
    initError: null,
  })

  assert.equal(explanation.summary, 'IP burst')
  assert.ok(explanation.signals.some((signal) => signal.key === 'ip-burst'))
  assert.deepEqual(blockchainState.missingConfig, ['RPC_URL', 'CONTRACT_ADDRESS'])
  assert.equal(blockchainState.state, 'disabled')
})

test('summarizeDashboardMetrics combines settlement history consistently across modes', () => {
  const result = summarizeDashboardMetrics({
    campaigns: [{ id: 'cmp-1', status: 'active' }, { id: 'cmp-2', status: 'budget-exhausted' }],
    clickLogs: [{ valid: true }, { valid: false }, { valid: true }],
    settlements: [
      { id: 'confirmed-old', status: 'confirmed', payoutEth: '0.0010', timestamp: '2024-01-01T00:00:00.000Z', settlementLatencyMs: 1200 },
      { id: 'simulated-newer', status: 'simulated', payoutEth: '0.0020', timestamp: '2024-01-02T00:00:00.000Z' },
      { id: 'confirmed-newest', status: 'confirmed', payoutEth: '0.0030', timestamp: '2024-01-03T00:00:00.000Z', settlementLatencyMs: 800 },
    ],
  })

  assert.equal(result.activeCampaigns, 1)
  assert.equal(result.totalCampaigns, 2)
  assert.equal(result.totalClicks, 3)
  assert.equal(result.validClicks, 2)
  assert.equal(result.fraudBlocked, 1)
  assert.equal(result.confirmedPayoutEth, '0.0040')
  assert.equal(result.simulatedPayoutEth, '0.0020')
  assert.equal(result.totalPayoutEth, '0.0060')
  assert.equal(result.averageSettlementLatencyMs, 1000)
  assert.equal(result.recordedSettlementCount, 3)
  assert.deepEqual(result.recentSettlements.map((item) => item.id), ['confirmed-newest', 'simulated-newer', 'confirmed-old'])
  assert.deepEqual(result.recentOnChainSettlements.map((item) => item.id), ['confirmed-newest', 'confirmed-old'])
  assert.deepEqual(result.recentSimulatedSettlements.map((item) => item.id), ['simulated-newer'])
})

test('verifySignedOracleDecision accepts quorum-satisfying signatures', () => {
  const signers = [{ id: 'oracle-1', secret: 'shared-secret' }]
  const decision = {
    result: 1,
    confidence: 0.8123,
    threshold: 0.5,
    model: 'test-model',
    requestId: 'req-1',
    issuedAt: new Date('2024-01-01T00:00:00.000Z').toISOString(),
    expiresAt: new Date('2024-01-01T00:00:20.000Z').toISOString(),
    featuresHash: 'abc123',
  }
  const message = buildOracleSignatureMessage(decision)
  decision.signatures = [{
    signerId: 'oracle-1',
    signature: crypto.createHmac('sha256', 'shared-secret').update(message).digest('base64url'),
  }]

  const result = verifySignedOracleDecision(decision, {
    signers,
    requiredQuorum: 1,
    maxSkewMs: 30_000,
    expectedFeaturesHash: 'abc123',
    now: Date.parse('2024-01-01T00:00:05.000Z'),
  })

  assert.equal(result.verified, true)
  assert.deepEqual(result.matchedSigners, ['oracle-1'])
  assert.equal(result.reason, 'verified')
})

test('verifySignedOracleDecision rejects feature hash mismatches', () => {
  const decision = {
    requestId: 'req-2',
    issuedAt: new Date('2024-01-01T00:00:00.000Z').toISOString(),
    expiresAt: new Date('2024-01-01T00:00:20.000Z').toISOString(),
    featuresHash: 'different',
    signatures: [],
  }

  const result = verifySignedOracleDecision(decision, {
    signers: [],
    expectedFeaturesHash: 'expected',
    now: Date.parse('2024-01-01T00:00:05.000Z'),
  })

  assert.equal(result.verified, false)
  assert.equal(result.reason, 'features-hash-mismatch')
})

test('summarizeBenchmarkMetrics and buildSecurityAuditReport expose hardening metrics', () => {
  const store = {
    clickLogs: [
      { valid: true, mlLatencyMs: 120, validationLatencyMs: 220, oracle: { verified: true } },
      { valid: false, mlLatencyMs: 80, validationLatencyMs: 180, oracle: { verified: false } },
    ],
    settlements: [
      { status: 'confirmed', gasUsed: '45000', settlementLatencyMs: 900 },
      { status: 'simulated', settlementLatencyMs: 300 },
    ],
  }

  const benchmark = summarizeBenchmarkMetrics(store)
  const audit = buildSecurityAuditReport({
    store,
    blockchain: { signerConnected: true, reason: 'connected' },
    oracle: { signerCount: 2, requiredQuorum: 1 },
    rateLimit: { windowMs: 60000, maxRequests: 30 },
    replayProtection: { ttlMs: 900000 },
    contractCapabilities: { eventSettlementEnabled: true },
  })

  assert.equal(benchmark.requests, 2)
  assert.equal(benchmark.validClicks, 1)
  assert.equal(benchmark.confirmedSettlements, 1)
  assert.equal(benchmark.gasUsed.average, 45000)
  assert.equal(audit.status, 'ok')
  assert.ok(audit.controls.some((control) => control.key === 'oracle-signatures' && control.status === 'enabled'))
})