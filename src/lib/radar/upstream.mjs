export const UPSTREAM_URL = 'https://prodclub.xyz/api/v1/radar/codex'
export const UPSTREAM_PAGE = 'https://prodclub.xyz/radar'

function timestamp(value) {
	if (typeof value !== 'string' || !/(Z|[+-]\d{2}:\d{2})$/.test(value) || !Number.isFinite(Date.parse(value))) throw new Error('Invalid upstream timestamp')
	return new Date(value).toISOString()
}

function normalizeEvent(record, scheduled = false) {
	if (!record || typeof record.id !== 'string' || !['regular', 'banked'].includes(record.type)) throw new Error('Unknown upstream event')
	const source = new URL(record.url)
	if (
		!['x.com', 'twitter.com'].includes(source.hostname) ||
		source.protocol !== 'https:' ||
		!/^\/[A-Za-z0-9_]{1,15}\/status\/\d+\/?$/.test(source.pathname) ||
		source.username ||
		source.password
	)
		throw new Error('Invalid upstream post URL')
	const kind = record.type === 'banked' ? 'banked' : 'reset'
	const label = kind === 'banked' ? '重置卡发放' : '额度重置'
	const announcedAt = timestamp(record.announced_at)
	return {
		id: `prodclub-${record.id}-${kind}`,
		title: `Codex ${label}${scheduled ? '预告' : ''}`,
		summary: scheduled
			? '参考站解析的重置预告。预计时间已换算为北京时间，最终生效与适用范围以原公告为准。'
			: record.observed
				? `参考站观察到的${label}记录，尚未由本站独立核对；时间采用来源的观察时间。`
				: `参考站收录的${label}公告；记录时间采用公告发布时间，适用范围以原帖为准。`,
		kind,
		status: scheduled ? 'scheduled' : 'confirmed',
		announcedAt,
		eventAt: scheduled ? timestamp(record.scheduled_for) : announcedAt,
		sourceName: `@${source.pathname.split('/')[1]} · 经 ProdClub 整理`,
		sourceUrl: `https://x.com${source.pathname.replace(/\/$/, '')}`,
		evidence: record.observed ? 'source-observed' : 'source-record',
		timeBasis: scheduled ? 'scheduled' : 'announcement'
	}
}

export function normalizeProdclubFeed(payload, checkedAt = new Date().toISOString()) {
	const source = payload?.data
	if (!source || !Array.isArray(source.resets) || !source.resets.length) throw new Error('Missing upstream history')
	const events = source.resets.map(record => normalizeEvent(record))
	if (source.scheduled_reset) events.push(normalizeEvent(source.scheduled_reset, true))
	return {
		updatedAt: events.reduce((latest, event) => (event.announcedAt > latest ? event.announcedAt : latest), events[0].announcedAt),
		events,
		sync: { mode: 'upstream', checkedAt: timestamp(source.fetched_at ?? checkedAt), stale: false }
	}
}

export function mergeRadarFeeds(previous, incoming) {
	const events = new Map()
	const sourceKeys = new Map()
	const currentSources = new Set(incoming.events.map(event => `${event.sourceUrl}|${event.kind}`))
	const retained = previous.events.filter(
		event =>
			incoming.sync?.mode !== 'upstream' ||
			event.status !== 'scheduled' ||
			event.evidence === 'direct-post' ||
			currentSources.has(`${event.sourceUrl}|${event.kind}`)
	)
	for (const event of [...retained, ...incoming.events]) {
		const sourceKey = `${event.sourceUrl}|${event.kind}`
		const id = sourceKeys.get(sourceKey) ?? event.id
		const existing = events.get(id)
		// A later provider snapshot must not downgrade a direct confirmation or cancellation.
		if (existing?.evidence === 'direct-post' && event.evidence !== 'direct-post') continue
		if (existing?.status === 'confirmed' && event.status === 'scheduled') continue
		events.set(id, { ...event, id })
		sourceKeys.set(sourceKey, id)
	}
	const list = [...events.values()].sort((a, b) => Date.parse(b.announcedAt) - Date.parse(a.announcedAt))
	return { updatedAt: list[0]?.announcedAt ?? null, events: list, ...(incoming.sync ? { sync: incoming.sync } : previous.sync ? { sync: previous.sync } : {}) }
}

export async function fetchProdclubFeed(fetcher = fetch) {
	const response = await fetcher(UPSTREAM_URL, {
		headers: { Accept: 'application/json' },
		signal: AbortSignal.timeout(12000),
		cache: 'no-store',
		redirect: 'error'
	})
	if (!response.ok) throw new Error(`Radar upstream returned HTTP ${response.status}`)
	return normalizeProdclubFeed(await response.json())
}
