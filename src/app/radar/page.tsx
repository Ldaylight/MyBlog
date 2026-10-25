import type { Metadata } from 'next'
import initialFeed from '@/../public/radar/events.json'
import { parseRadarFeed } from './data'
import RadarDashboard from './radar-dashboard'

export const metadata: Metadata = {
	title: 'Codex 重置雷达',
	description: '整理 Codex 额度重置与重置卡公告，查看重置倒计时、历史记录和日历。'
}

export default function RadarPage() {
	return <RadarDashboard initialFeed={parseRadarFeed(initialFeed)} />
}
