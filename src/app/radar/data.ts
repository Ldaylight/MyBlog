export type ResetKind = 'reset' | 'banked'

export interface RadarEvent {
	id: string
	title: string
	summary: string
	kind: ResetKind
	status: 'scheduled' | 'confirmed' | 'cancelled'
	announcedAt: string
	eventAt: string | null
	sourceName: string
	sourceUrl: string
	evidence?: 'source-record' | 'source-observed' | 'direct-post'
	timeBasis?: 'announcement' | 'scheduled' | 'effective'
}

export interface RadarFeed {
	updatedAt: string | null
	events: RadarEvent[]
	sync?: {
		mode: 'upstream' | 'x' | 'snapshot'
		checkedAt: string
		stale: boolean
	}
}

const dateFormatter = new Intl.DateTimeFormat('en-CA', {
	timeZone: 'Asia/Shanghai',
	year: 'numeric',
	month: '2-digit',
	day: '2-digit'
})

export function dayKey(date: Date | number | string) {
	const parts = dateFormatter.formatToParts(new Date(date))
	return ['year', 'month', 'day'].map(type => parts.find(part => part.type === type)?.value).join('-')
}

function isTimestamp(value: unknown): value is string {
	return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(value) && Number.isFinite(Date.parse(value))
}

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function parseRadarFeed(value: unknown): RadarFeed {
	if (!isObject(value) || !Array.isArray(value.events) || (value.updatedAt !== null && !isTimestamp(value.updatedAt))) {
		throw new Error('消息记录格式有误，请稍后重试。')
	}
	if (
		value.sync !== undefined &&
		(!isObject(value.sync) ||
			!['upstream', 'x', 'snapshot'].includes(String(value.sync.mode)) ||
			!isTimestamp(value.sync.checkedAt) ||
			typeof value.sync.stale !== 'boolean')
	) {
		throw new Error('同步状态格式有误。')
	}

	const ids = new Set<string>()
	for (const event of value.events) {
		if (
			!isObject(event) ||
			!['id', 'title', 'summary', 'sourceName', 'sourceUrl'].every(key => typeof event[key] === 'string' && event[key].trim().length > 0) ||
			(event.kind !== 'reset' && event.kind !== 'banked') ||
			(event.status !== 'scheduled' && event.status !== 'confirmed' && event.status !== 'cancelled') ||
			(event.evidence !== undefined && !['source-record', 'source-observed', 'direct-post'].includes(String(event.evidence))) ||
			(event.timeBasis !== undefined && !['announcement', 'scheduled', 'effective'].includes(String(event.timeBasis))) ||
			!isTimestamp(event.announcedAt) ||
			(event.eventAt !== null && !isTimestamp(event.eventAt)) ||
			(event.status === 'confirmed' && event.eventAt === null)
		) {
			throw new Error('部分消息记录不完整，请稍后重试。')
		}
		const url = new URL(event.sourceUrl as string)
		if (url.protocol !== 'https:' || url.username || url.password || ids.has(event.id as string)) {
			throw new Error('消息来源或记录编号有误，请稍后重试。')
		}
		ids.add(event.id as string)
	}
	return value as unknown as RadarFeed
}

export function getRadarStats(events: RadarEvent[], now: number) {
	const confirmed = events
		.filter(event => event.status === 'confirmed' && event.eventAt && Date.parse(event.eventAt) <= now)
		.sort((a, b) => Date.parse(b.eventAt!) - Date.parse(a.eventAt!))
	const resets = confirmed.filter(event => event.kind === 'reset')
	const upcoming = events
		.filter(event => event.status === 'scheduled' && event.eventAt && Date.parse(event.eventAt) > now)
		.sort((a, b) => Date.parse(a.eventAt!) - Date.parse(b.eventAt!))[0]
	const awaitingConfirmation = events.some(
		event =>
			event.status === 'scheduled' &&
			event.eventAt &&
			Date.parse(event.eventAt) <= now &&
			!confirmed.some(record => record.kind === event.kind && Date.parse(record.eventAt!) >= Date.parse(event.eventAt!))
	)
	const lastReset = resets[0]
	const averageDays =
		resets.length > 1 ? (Date.parse(resets[0].eventAt!) - Date.parse(resets[resets.length - 1].eventAt!)) / (resets.length - 1) / 86400000 : null
	return {
		confirmed,
		upcoming,
		awaitingConfirmation,
		lastReset,
		averageDays,
		daysSinceReset: lastReset ? (now - Date.parse(lastReset.eventAt!)) / 86400000 : null,
		last30Days: confirmed.filter(event => Date.parse(event.eventAt!) >= now - 30 * 86400000).length,
		resetCount: resets.length,
		bankedCount: confirmed.length - resets.length
	}
}

export function buildCalendar(events: RadarEvent[], today: string) {
	const counts = new Map<string, { reset: number; banked: number }>()
	for (const event of events) {
		if (event.status === 'confirmed' && event.eventAt && dayKey(event.eventAt) <= today) {
			const key = dayKey(event.eventAt)
			const count = counts.get(key) ?? { reset: 0, banked: 0 }
			count[event.kind] += 1
			counts.set(key, count)
		}
	}
	// UTC arithmetic on date-only strings keeps the grid independent of the visitor's timezone.
	const end = new Date(`${today}T00:00:00Z`)
	const start = new Date(end)
	start.setUTCDate(start.getUTCDate() - ((start.getUTCDay() + 6) % 7) - 25 * 7)
	return Array.from({ length: 26 }, (_, week) =>
		Array.from({ length: 7 }, (_, day) => {
			const date = new Date(start)
			date.setUTCDate(date.getUTCDate() + week * 7 + day)
			const key = date.toISOString().slice(0, 10)
			const count = counts.get(key) ?? { reset: 0, banked: 0 }
			return { key, count: count.reset + count.banked, resets: count.reset, banked: count.banked, future: key > today }
		})
	)
}
