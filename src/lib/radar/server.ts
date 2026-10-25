import { unstable_cache } from 'next/cache'
import snapshot from '@/../public/radar/events.json'
import { parseRadarFeed } from '@/app/radar/data'
import { fetchProdclubFeed, mergeRadarFeeds } from './upstream.mjs'

const readUpstream = unstable_cache(async () => parseRadarFeed(await fetchProdclubFeed()), ['codex-radar-upstream-v1'], { revalidate: 60 })

export async function getRadarFeed() {
	const fallback = parseRadarFeed(snapshot)
	try {
		const live = await readUpstream()
		const feed = parseRadarFeed(mergeRadarFeeds(fallback, live))
		if (feed.sync && Date.now() - Date.parse(feed.sync.checkedAt) > 180000) feed.sync.stale = true
		return feed
	} catch {
		return {
			...fallback,
			sync: { mode: 'snapshot' as const, checkedAt: fallback.sync?.checkedAt ?? fallback.updatedAt ?? new Date(0).toISOString(), stale: true }
		}
	}
}
