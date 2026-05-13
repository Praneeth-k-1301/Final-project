import test from 'node:test'
import assert from 'node:assert/strict'

import { getConfiguredGateways, getConfiguredIpfsApis, uploadAndPinJsonToIpfs } from './ipfs-utils.js'

test('getConfiguredIpfsApis returns empty array when unset', () => {
  const originalUrl = process.env.IPFS_API_URL
  const originalUrls = process.env.IPFS_API_URLS
  delete process.env.IPFS_API_URL
  delete process.env.IPFS_API_URLS
  const apis = getConfiguredIpfsApis()
  assert.deepEqual(apis, [])
  if (originalUrl !== undefined) process.env.IPFS_API_URL = originalUrl
  if (originalUrls !== undefined) process.env.IPFS_API_URLS = originalUrls
})

test('uploadAndPinJsonToIpfs reports unconfigured when no APIs are set', async () => {
  const originalUrl = process.env.IPFS_API_URL
  const originalUrls = process.env.IPFS_API_URLS
  delete process.env.IPFS_API_URL
  delete process.env.IPFS_API_URLS
  const result = await uploadAndPinJsonToIpfs({ demo: true })
  assert.equal(result.status, 'unconfigured')
  assert.equal(result.cid, null)
  assert.deepEqual(result.pins, [])
  if (originalUrl !== undefined) process.env.IPFS_API_URL = originalUrl
  if (originalUrls !== undefined) process.env.IPFS_API_URLS = originalUrls
})

test('getConfiguredGateways splits comma-separated list', () => {
  const originalGateways = process.env.IPFS_GATEWAYS
  process.env.IPFS_GATEWAYS = 'https://a.example/ipfs, https://b.example/ipfs '
  const gateways = getConfiguredGateways()
  assert.deepEqual(gateways, ['https://a.example/ipfs', 'https://b.example/ipfs'])
  if (originalGateways !== undefined) process.env.IPFS_GATEWAYS = originalGateways
  else delete process.env.IPFS_GATEWAYS
})

