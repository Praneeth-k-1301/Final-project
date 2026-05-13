import test from 'node:test'
import assert from 'node:assert/strict'

import { buildValidationFeatures, stableCid } from './click-utils.js'

test('stableCid returns deterministic ids', () => {
  assert.equal(stableCid('campaign-1'), stableCid('campaign-1'))
  assert.notEqual(stableCid('campaign-1'), stableCid('campaign-2'))
})

test('buildValidationFeatures derives rolling click metrics', () => {
  const history = [{ visitorKey: 'wallet-1', app: 12, timestamp: '2024-01-01T00:00:00.000Z' }]
  const campaign = { id: 'cmp-1', appCode: 12, osCode: 13, channelCode: 111 }
  const payload = { walletAddress: 'wallet-1', deviceType: 'mobile', clickTime: '2024-01-01T00:00:02.000Z' }
  const features = buildValidationFeatures({ campaign, payload, history })
  assert.equal(features.clickFrequency, 2)
  assert.equal(features.ipClickCount, 2)
  assert.equal(features.deviceCode, 1)
  assert.equal(features.app, 12)
})

test('buildValidationFeatures includes extended behavioral features', () => {
  const now = new Date('2024-01-01T00:00:10.000Z').getTime()
  const history = [
    { visitorKey: 'w1', app: 12, deviceCode: 1, timestamp: '2024-01-01T00:00:02.000Z' },
    { visitorKey: 'w1', app: 12, deviceCode: 1, timestamp: '2024-01-01T00:00:05.000Z' },
    { visitorKey: 'w1', app: 15, deviceCode: 1, timestamp: '2024-01-01T00:00:07.000Z' },
  ]
  const campaign = { id: 'cmp-1', appCode: 12, osCode: 13, channelCode: 111 }
  const payload = { walletAddress: 'w1', deviceType: 'mobile', clickTime: '2024-01-01T00:00:10.000Z' }
  const f = buildValidationFeatures({ campaign, payload, history, now })

  // Rolling window counts (all within 10s)
  assert.equal(f.clickCountLast10Seconds, 4)
  assert.equal(f.clickCountLast60Seconds, 4)
  assert.equal(f.clickCountLast10Minutes, 4)

  // Extended fields exist and are numeric
  assert.equal(typeof f.timeSinceLastClickPerDevice, 'number')
  assert.equal(typeof f.deviceAppEntropy, 'number')
  assert.equal(typeof f.deviceIpRatio, 'number')
  assert.equal(typeof f.uniqueAppsPerDevice, 'number')
  assert.equal(typeof f.uniqueDevicesPerIp, 'number')
  assert.equal(typeof f.burstClickScore, 'number')

  // uniqueAppsPerDevice: device 1 used apps 12, 15, and current 12 => Set{12,15} => 2
  assert.equal(f.uniqueAppsPerDevice, 2)

  // deviceAppEntropy should be > 0 since device has used 2 different apps
  assert.ok(f.deviceAppEntropy > 0, 'entropy should be positive for diverse app usage')
})
