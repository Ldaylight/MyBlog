import fs from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { parseRadarFeed } from '../src/app/radar/data.ts'
import { fetchProdclubFeed, mergeRadarFeeds } from '../src/lib/radar/upstream.mjs'
import { monitorX } from './radar/x-monitor.mjs'

const file = fileURLToPath(new URL('../public/radar/events.json', import.meta.url))
const stateFile = fileURLToPath(new URL('../data/radar-monitor-state.json', import.meta.url))
const mode = process.env.RADAR_SYNC_MODE || 'upstream'
if (!['upstream', 'x', 'hybrid'].includes(mode)) throw new Error('RADAR_SYNC_MODE must be upstream, x or hybrid')

const previous = parseRadarFeed(JSON.parse(await fs.readFile(file, 'utf8')))
let next = previous
if (mode !== 'x') {
	let upstream
	for (let attempt = 0; attempt < 3; attempt++) {
		try {
			upstream = await fetchProdclubFeed()
			break
		} catch (error) {
			if (attempt === 2 && mode === 'upstream') throw error
			if (attempt === 2) console.warn('Upstream sync unavailable; continuing with independent X detection and existing history.')
		}
	}
	if (upstream) next = parseRadarFeed(mergeRadarFeeds(previous, upstream))
}
let nextState
let previousState
if (mode !== 'upstream') {
	if (!process.env.X_BEARER_TOKEN || !process.env.OPENAI_API_KEY || !process.env.RADAR_OPENAI_MODEL)
		throw new Error('Independent X detection needs X_BEARER_TOKEN, OPENAI_API_KEY and RADAR_OPENAI_MODEL')
	try {
		previousState = JSON.parse(await fs.readFile(stateFile, 'utf8'))
	} catch (error) {
		if (error.code !== 'ENOENT') throw error
		previousState = {}
	}
	const result = await monitorX(next, previousState, {
		token: process.env.X_BEARER_TOKEN,
		key: process.env.OPENAI_API_KEY,
		model: process.env.RADAR_OPENAI_MODEL,
		accounts: process.env.RADAR_X_ACCOUNTS || 'thsottiaux'
	})
	next = parseRadarFeed(mergeRadarFeeds(next, result.feed))
	nextState = result.state
}

// Do not create a commit or a Vercel deployment just because the check time changed.
const changed = JSON.stringify(previous.events) !== JSON.stringify(next.events)
if (changed) {
	const temporary = `${file}.tmp`
	await fs.writeFile(temporary, `${JSON.stringify(next, null, '\t')}\n`)
	await fs.rename(temporary, file)
}
if (nextState && JSON.stringify(previousState) !== JSON.stringify(nextState)) {
	await fs.mkdir(fileURLToPath(new URL('../data/', import.meta.url)), { recursive: true })
	await fs.writeFile(stateFile, `${JSON.stringify(nextState, null, '\t')}\n`)
}
console.log(`Radar sync (${mode}): ${next.events.length} records; ${changed ? 'history updated' : 'no history changes'}.`)
