import axios from 'axios'
import FormData from 'form-data'

function splitAndTrim(value) {
  return String(value || '')
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
}

function normalizeBase(url) {
  return String(url || '').replace(/\/+$/u, '')
}

export function getConfiguredIpfsApis() {
  const fromList = splitAndTrim(process.env.IPFS_API_URLS)
  const primary = (process.env.IPFS_API_URL || '').trim()
  const all = []
  if (primary) all.push(primary)
  for (const url of fromList) {
    if (!all.includes(url)) all.push(url)
  }
  return all
}

export function getConfiguredGateways() {
  return splitAndTrim(process.env.IPFS_GATEWAYS)
}

async function addJsonToApi(apiUrl, json, timeoutMs) {
  const body = JSON.stringify(json)
  const form = new FormData()
  form.append('file', Buffer.from(body), {
    filename: 'campaign.json',
    contentType: 'application/json',
  })
  const url = `${normalizeBase(apiUrl)}/add?pin=true`
  const response = await axios.post(url, form, {
    timeout: timeoutMs,
    maxBodyLength: Infinity,
    headers: form.getHeaders(),
  })
  const data = response.data || {}
  const cid = data.Hash || data.Cid?.['/'] || data.cid || null
  if (!cid) throw new Error('IPFS add response did not include a CID')
  return cid
}

async function pinCidAtApi(apiUrl, cid, timeoutMs) {
  const url = `${normalizeBase(apiUrl)}/pin/add?arg=${encodeURIComponent(cid)}&recursive=true`
  const startedAt = Date.now()
  try {
    const response = await axios.post(url, null, {
      timeout: timeoutMs,
      maxBodyLength: 0,
    })
    return {
      apiUrl,
      ok: true,
      status: response.status,
      latencyMs: Date.now() - startedAt,
    }
  } catch (error) {
    return {
      apiUrl,
      ok: false,
      status: error.response?.status || null,
      latencyMs: Date.now() - startedAt,
      error: error.message,
    }
  }
}

export async function checkCidAvailabilityAcrossGateways(cid, options = {}) {
  const gateways = getConfiguredGateways()
  const checkedAt = new Date().toISOString()
  if (!gateways.length) {
    return { cid, available: false, status: 'unconfigured', gateways: [], checkedAt }
  }
  const timeoutMs = options.timeoutMs || 5000
  const results = []
  let available = false
  for (const gateway of gateways) {
    const url = `${normalizeBase(gateway)}/${encodeURIComponent(cid)}`
    const startedAt = Date.now()
    try {
      const response = await axios.head(url, {
        timeout: timeoutMs,
        validateStatus: () => true,
      })
      const ok = response.status >= 200 && response.status < 400
      if (ok) available = true
      results.push({
        gateway,
        ok,
        status: response.status,
        latencyMs: Date.now() - startedAt,
      })
    } catch (error) {
      results.push({
        gateway,
        ok: false,
        status: null,
        latencyMs: Date.now() - startedAt,
        error: error.message,
      })
    }
  }
  const status = available ? 'available' : 'missing'
  return { cid, available, status, gateways: results, checkedAt }
}

export async function uploadAndPinJsonToIpfs(json, options = {}) {
  const apis = getConfiguredIpfsApis()
  const checkedAt = new Date().toISOString()
  if (!apis.length) {
    return {
      cid: null,
      pins: [],
      status: 'unconfigured',
      checkedAt,
      error: 'No IPFS_API_URL or IPFS_API_URLS configured',
    }
  }
  const timeoutMs = options.timeoutMs || 10000
  const primary = apis[0]
  let cid
  const pins = []
  try {
    cid = await addJsonToApi(primary, json, timeoutMs)
    pins.push(await pinCidAtApi(primary, cid, timeoutMs))
  } catch (error) {
    return {
      cid: null,
      pins,
      status: 'pin-failed',
      checkedAt,
      error: error.message,
    }
  }
  for (const apiUrl of apis.slice(1)) {
    pins.push(await pinCidAtApi(apiUrl, cid, timeoutMs))
  }
  let availability
  try {
    availability = await checkCidAvailabilityAcrossGateways(cid, { timeoutMs })
  } catch {
    availability = null
  }
  const status = availability?.available ? 'pinned' : 'pinned-unverified'
  return {
    cid,
    pins,
    status,
    checkedAt: availability?.checkedAt || checkedAt,
    gateways: availability?.gateways || [],
  }
}

export async function replicateCid(cid, options = {}) {
  const apis = getConfiguredIpfsApis()
  const timeoutMs = options.timeoutMs || 10000
  const pins = []
  for (const apiUrl of apis) {
    pins.push(await pinCidAtApi(apiUrl, cid, timeoutMs))
  }
  return { cid, pins }
}

