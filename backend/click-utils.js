import crypto from 'crypto'

export const DEVICE_CODE_MAP = { mobile: 1, desktop: 2, tablet: 3 }

export function stableCid(value) {
  return `cid-${crypto.createHash('sha256').update(String(value)).digest('hex').slice(0, 24)}`
}

export function detectDeviceType(payload = {}) {
  if (payload.deviceType && DEVICE_CODE_MAP[payload.deviceType]) return payload.deviceType
  const source = `${payload.userAgent || ''}`.toLowerCase()
  if (source.includes('ipad') || source.includes('tablet')) return 'tablet'
  if (source.includes('mobi') || source.includes('android') || source.includes('iphone')) return 'mobile'
  return 'desktop'
}

export function inferOsCode(payload = {}, fallback = 13) {
  if (payload.osCode !== undefined && payload.osCode !== null) return Number(payload.osCode)
  const source = `${payload.platform || payload.userAgent || ''}`.toLowerCase()
  if (source.includes('android')) return 13
  if (source.includes('iphone') || source.includes('ios')) return 17
  if (source.includes('windows')) return 19
  if (source.includes('mac')) return 27
  if (source.includes('linux')) return 20
  return Number(fallback)
}

export function buildValidationFeatures({ campaign, payload, history, now = Date.now() }) {
  const timestamp = new Date(payload.clickTime || now).getTime()
  const visitorKey = payload.visitorId || payload.walletAddress || payload.ipAddress || 'anonymous'
  const app = Number(payload.app ?? campaign.appCode ?? 12)
  const channel = Number(payload.channel ?? campaign.channelCode ?? 111)
  const deviceType = detectDeviceType(payload)
  const deviceCode = DEVICE_CODE_MAP[deviceType] || 2

  const sameVisitor = history.filter((entry) => entry.visitorKey === visitorKey)
  const sameApp = history.filter((entry) => entry.app === app)
  const sameVisitorApp = sameVisitor.filter((entry) => entry.app === app)
  const sameDevice = history.filter((entry) => (entry.deviceCode || DEVICE_CODE_MAP[entry.deviceType]) === deviceCode)

  const sessionStart = timestamp - (30 * 60 * 1000)
  const clickFrequency = sameVisitor.filter((entry) => new Date(entry.timestamp).getTime() >= sessionStart).length + 1
  const lastClickTime = sameVisitor.length
    ? Math.max(...sameVisitor.map((entry) => new Date(entry.timestamp).getTime()))
    : timestamp - 30000
  const timeInterval = Math.max(250, timestamp - lastClickTime)

  const lastDeviceClickTime = sameDevice.length
    ? Math.max(...sameDevice.map((entry) => new Date(entry.timestamp).getTime()))
    : timestamp - 30000
  const timeSinceLastClickPerDevice = Math.max(250, timestamp - lastDeviceClickTime)

  // Rolling window counts
  const clickCountLast10Seconds = sameVisitor.filter((e) => timestamp - new Date(e.timestamp).getTime() <= 10000).length + 1
  const clickCountLast60Seconds = sameVisitor.filter((e) => timestamp - new Date(e.timestamp).getTime() <= 60000).length + 1
  const clickCountLast10Minutes = sameVisitor.filter((e) => timestamp - new Date(e.timestamp).getTime() <= 600000).length + 1

  // Unique counts
  const uniqueAppsPerDevice = new Set(sameDevice.map((e) => e.app)).add(app).size
  const uniqueDevicesPerIp = new Set(sameVisitor.map((e) => e.deviceCode || DEVICE_CODE_MAP[e.deviceType] || 2)).add(deviceCode).size

  // Device-app entropy (approximation from history)
  const deviceApps = sameDevice.map((e) => e.app).concat([app])
  const appCounts = {}
  for (const a of deviceApps) appCounts[a] = (appCounts[a] || 0) + 1
  const total = deviceApps.length
  let deviceAppEntropy = 0
  if (total > 1) {
    for (const count of Object.values(appCounts)) {
      const p = count / total
      if (p > 0) deviceAppEntropy -= p * Math.log(p)
    }
  }

  const deviceClickCount = sameDevice.length + 1
  const ipClickCount = sameVisitor.length + 1
  const deviceIpRatio = Number((deviceClickCount / Math.max(ipClickCount, 1)).toFixed(4))
  const gapSeconds = Math.max(timeInterval / 1000, 1)
  const burstScore = Number((clickFrequency / gapSeconds).toFixed(4))
  const burstClickScore = Number(((clickCountLast10Seconds * 6 + clickCountLast60Seconds + clickCountLast10Minutes / 10) / gapSeconds).toFixed(4))

  return {
    visitorKey,
    timestamp: new Date(timestamp).toISOString(),
    clickFrequency,
    timeInterval,
    deviceType,
    deviceCode,
    app,
    osCode: inferOsCode(payload, campaign.osCode),
    channel,
    hourOfDay: new Date(timestamp).getUTCHours(),
    ipClickCount,
    appClickCount: sameApp.length + 1,
    ipAppCount: sameVisitorApp.length + 1,
    burstScore,
    timeSinceLastClickPerDevice,
    clickCountLast10Seconds,
    clickCountLast60Seconds,
    clickCountLast10Minutes,
    deviceAppEntropy: Number(deviceAppEntropy.toFixed(4)),
    deviceIpRatio,
    uniqueAppsPerDevice,
    uniqueDevicesPerIp,
    burstClickScore,
  }
}
