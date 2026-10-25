import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'
import { parseRadarFeed, getRadarStats, buildCalendar, dayKey } from '../src/app/radar/data.ts'
import { normalizeProdclubFeed, mergeRadarFeeds } from '../src/lib/radar/upstream.mjs'
import { getAccounts, fetchAccountPosts, classifyPost, validateClassifications, monitorX } from './radar/x-monitor.mjs'

const now = Date.parse('2026-10-25T04:00:00Z')
const empty = { updatedAt: null, events: [] }
const record = { id: '123', type: 'regular', announced_at: '2026-10-23T17:00:00Z', url: 'https://x.com/thsottiaux/status/123', observed: false }
const payload = {
	data: {
		fetched_at: '2026-10-25T04:00:00Z',
		resets: [record, { ...record, id: '124', type: 'banked', url: 'https://x.com/thsottiaux/status/124' }],
		scheduled_reset: {
			...record,
			id: '125',
			url: 'https://x.com/thsottiaux/status/125',
			announced_at: '2026-10-25T03:40:04Z',
			scheduled_for: '2026-10-25T07:00:00Z'
		}
	}
}
const post = { id: '500', username: 'thsottiaux', created_at: '2026-10-25T03:00:00Z', text: 'Codex usage limits have been reset for everyone.', context: [] }
const classification = {
	kind: 'reset',
	status: 'confirmed',
	scope: 'shared',
	eventAt: null,
	relatedEventId: null,
	title: 'Codex 额度重置',
	summary: '面向用户的重置公告。',
	evidence: 'limits have been reset',
	confidence: 0.99
}

test('upstream history imports both kinds and an absolute-time forecast', () => {
	const feed = parseRadarFeed(normalizeProdclubFeed(payload))
	assert.equal(feed.events.length, 3)
	assert.equal(feed.events.filter(event => event.kind === 'banked').length, 1)
	assert.equal(getRadarStats(feed.events, now).upcoming.eventAt, '2026-10-25T07:00:00.000Z')
	assert.equal(dayKey(feed.events[0].eventAt), '2026-10-24')
	assert.throws(() => normalizeProdclubFeed({ data: { resets: [] } }))
	assert.throws(() => normalizeProdclubFeed({ data: { ...payload.data, resets: [{ ...record, url: 'https://evil.example/status/123' }] } }))
})

test('snapshot contains imported real history with provenance, no copied post bodies', () => {
	const snapshot = parseRadarFeed(JSON.parse(fs.readFileSync(new URL('../public/radar/events.json', import.meta.url), 'utf8')))
	assert.ok(snapshot.events.filter(event => event.status === 'confirmed').length >= 66)
	assert.ok(snapshot.events.filter(event => event.kind === 'banked').length >= 6)
	assert.ok(snapshot.events.every(event => event.evidence && event.sourceUrl.startsWith('https://x.com/')))
})

test('merging is idempotent and retains old history and direct corrections', () => {
	const feed = normalizeProdclubFeed(payload)
	assert.equal(mergeRadarFeeds(feed, feed).events.length, 3)
	assert.equal(mergeRadarFeeds(feed, empty).events.length, 3)
	const direct = { ...feed.events[0], evidence: 'direct-post', summary: '直接检测的修正' }
	assert.equal(mergeRadarFeeds({ ...feed, events: [direct] }, feed).events.find(event => event.id === direct.id).summary, direct.summary)
	const forecast = feed.events.find(event => event.status === 'scheduled')
	const confirmed = { ...forecast, status: 'confirmed', evidence: 'direct-post', eventAt: '2026-10-25T07:00:00Z' }
	const merged = mergeRadarFeeds(feed, { ...feed, events: [confirmed] })
	assert.equal(merged.events.length, 3)
	assert.equal(merged.events.find(event => event.id === confirmed.id).status, 'confirmed')
})

test('calendar separates reset/card counts on the same Beijing date', () => {
	const feed = normalizeProdclubFeed(payload)
	const calendar = buildCalendar(feed.events, '2026-10-25')
	const day = calendar.flat().find(day => day.key === '2026-10-24')
	assert.equal(day.resets, 1)
	assert.equal(day.banked, 1)
	assert.equal(day.count, 2)
	assert.equal(calendar.flat().find(day => day.key === '2026-10-25').count, 0)
	assert.equal(calendar.flat().length, 182)
	assert.equal(dayKey('2026-10-24T18:00:00Z'), '2026-10-25')
})

test('withdrawn upstream forecasts disappear instead of remaining as false countdowns', () => {
	const previous = normalizeProdclubFeed(payload)
	const current = normalizeProdclubFeed({ data: { ...payload.data, scheduled_reset: null } })
	const merged = mergeRadarFeeds(previous, current)
	assert.equal(merged.events.length, 2)
	assert.equal(getRadarStats(merged.events, now).upcoming, undefined)
})

test('forecast expiry never confirms a reset; cancellation never counts', () => {
	const feed = normalizeProdclubFeed(payload)
	const stats = getRadarStats(feed.events, now + 4 * 3600000)
	assert.equal(stats.upcoming, undefined)
	assert.equal(stats.awaitingConfirmation, true)
	assert.equal(stats.confirmed.length, 2)
	assert.equal(
		getRadarStats(
			feed.events.map(event => ({ ...event, status: 'cancelled' })),
			now
		).confirmed.length,
		0
	)
})

test('account configuration cannot inject extra X search operators', () => {
	assert.deepEqual(getAccounts('@ThSottiaux,thsottiaux'), ['thsottiaux'])
	assert.throws(() => getAccounts('someone OR from:attacker'))
})

test('X pagination, thread context, chronological order and cursor', async () => {
	const urls = []
	const result = await fetchAccountPosts({
		token: 'test',
		accounts: ['thsottiaux'],
		state: { query: 'thsottiaux', sinceId: '10' },
		fetcher: async url => {
			urls.push(new URL(url))
			const second = new URL(url).searchParams.has('next_token')
			return Response.json({
				data: [{ id: second ? '11' : '12', created_at: post.created_at, author_id: 'author', text: 'Done', referenced_tweets: [{ id: 'parent' }] }],
				includes: {
					users: [{ id: 'author', username: 'thsottiaux' }],
					tweets: [{ id: 'parent', text: 'Codex reset announcement', created_at: post.created_at }]
				},
				meta: second ? { result_count: 1 } : { result_count: 1, next_token: 'next' }
			})
		}
	})
	assert.deepEqual(
		result.posts.map(post => post.id),
		['11', '12']
	)
	assert.equal(result.posts[0].context[0].text, 'Codex reset announcement')
	assert.equal(urls[0].searchParams.get('since_id'), '10')
	assert.equal(result.newestId, '12')
	await assert.rejects(() =>
		fetchAccountPosts({
			token: 'test',
			accounts: ['thsottiaux'],
			fetcher: async () => Response.json({ data: [{ id: '12', created_at: post.created_at, author_id: 'unknown', text: 'reset' }], meta: { result_count: 1 } })
		})
	)
})

test('analysis uses strict structured output, then validates actual evidence', async () => {
	let request
	const events = await classifyPost(post, empty, {
		key: 'test',
		model: 'configured-model',
		fetcher: async (url, options) => {
			assert.equal(url, 'https://api.openai.com/v1/responses')
			request = JSON.parse(options.body)
			return Response.json({ status: 'completed', output: [{ content: [{ type: 'output_text', text: JSON.stringify({ events: [classification] }) }] }] })
		}
	})
	assert.equal(request.text.format.strict, true)
	assert.equal(request.store, false)
	assert.equal(events[0].eventAt, post.created_at.replace('Z', '.000Z'))
	assert.equal(events[0].timeBasis, 'announcement')
	for (const override of [
		{ confidence: 0.5 },
		{ scope: 'personal' },
		{ evidence: 'not present in post' },
		{ relatedEventId: 'invented-id' },
		{ eventAt: 'tomorrow' },
		{ status: 'cancelled' }
	]) {
		assert.equal(validateClassifications({ events: [{ ...classification, ...override }] }, post, empty, now).length, 0)
	}
})

test('ambiguous future time stays unknown and a related confirmation updates one event', () => {
	const planned = validateClassifications({ events: [{ ...classification, status: 'scheduled', eventAt: null }] }, post, empty, now)[0]
	assert.equal(planned.eventAt, null)
	assert.equal(getRadarStats([planned], now).upcoming, undefined)
	const confirmed = validateClassifications({ events: [{ ...classification, relatedEventId: planned.id }] }, post, { ...empty, events: [planned] }, now)[0]
	assert.equal(confirmed.id, planned.id)
	assert.equal(mergeRadarFeeds({ ...empty, events: [planned] }, { ...empty, events: [confirmed] }).events.length, 1)
})

test('processed post IDs prevent repeated model billing', async () => {
	const result = await monitorX(
		empty,
		{ query: 'thsottiaux', sinceId: '1', processedIds: ['500'] },
		{
			token: 'test',
			key: 'test',
			model: 'test',
			accounts: 'thsottiaux',
			fetcher: async url => {
				assert.equal(new URL(url).hostname, 'api.x.com')
				return Response.json({
					data: [{ ...post, author_id: 'author' }],
					includes: { users: [{ id: 'author', username: 'thsottiaux' }] },
					meta: { result_count: 1 }
				})
			}
		}
	)
	assert.equal(result.detected, 0)
	assert.equal(result.state.sinceId, '500')
})
