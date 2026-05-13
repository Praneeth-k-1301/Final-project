// Lightweight in-process message queue abstraction.
//
// This can later be swapped for Kafka/RabbitMQ by re-implementing
// publishEvent/subscribeToTopic with a real broker client. For now it
// gives you an event-driven architecture inside a single Node process
// without adding heavy broker dependencies.

const subscribers = new Map()

export const TOPICS = {
  RAW_CLICKS: 'clicks.raw',
  FEATURES: 'clicks.features',
  VALIDATED: 'clicks.validated',
  SETTLEMENT_REQUESTS: 'settlements.requests',
  SETTLEMENT_OUTCOMES: 'settlements.outcomes',
}

export function subscribeToTopic(topic, handler) {
  if (typeof handler !== 'function') {
    throw new TypeError('subscribeToTopic handler must be a function')
  }
  const set = subscribers.get(topic) || new Set()
  set.add(handler)
  subscribers.set(topic, set)
  return () => {
    set.delete(handler)
    if (set.size === 0) subscribers.delete(topic)
  }
}

export async function publishEvent(topic, message) {
  const set = subscribers.get(topic)
  if (!set || set.size === 0) return

  // Fire-and-forget fan-out within this process.
  for (const handler of set) {
    queueMicrotask(() => {
      Promise.resolve()
        .then(() => handler(message))
        .catch((error) => {
          // Do not let one subscriber crash the process.
          console.error(`Error in subscriber for topic ${topic}:`, error)
        })
    })
  }
}

