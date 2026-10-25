'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import useSWR from 'swr'
import { ArrowDown, ArrowUpRight, CalendarDays, Check, Clock3, Gift, History, Info, Radio, Radar, RefreshCw, Search, Signal, Sparkles } from 'lucide-react'
import { buildCalendar, dayKey, getRadarStats, parseRadarFeed, type RadarFeed, type ResetKind } from './data'
import styles from './radar.module.css'

const KIND_LABELS = { reset: '额度重置', banked: '重置卡' } as const
const dateFormat = new Intl.DateTimeFormat('zh-CN', {
	timeZone: 'Asia/Shanghai',
	year: 'numeric',
	month: 'long',
	day: 'numeric',
	hour: '2-digit',
	minute: '2-digit',
	hour12: false
})
const fullDateFormat = new Intl.DateTimeFormat('zh-CN', {
	timeZone: 'Asia/Shanghai',
	year: 'numeric',
	month: '2-digit',
	day: '2-digit',
	hour: '2-digit',
	minute: '2-digit',
	hour12: false
})

async function fetchFeed(url: string) {
	const response = await fetch(url, { cache: 'no-store' })
	if (!response.ok) throw new Error('暂时无法同步消息，请稍后重试。')
	return parseRadarFeed(await response.json())
}

function RadarVisual() {
	return (
		<div className={styles.radarVisual} aria-hidden='true'>
			<div className={styles.radarOrbit}>
				<div className={styles.radarSweep} />
				<span className={styles.radarAxisX} />
				<span className={styles.radarAxisY} />
				<span className={styles.radarRing} />
				<span className={styles.radarRing} />
				<span className={styles.radarRing} />
				<span className={styles.radarPoint} />
				<span className={styles.radarPoint} />
				<div className={styles.radarCore}>
					<Radar size={38} strokeWidth={1.25} />
				</div>
			</div>
			<div className={styles.radarCaption}>
				<span /> CODEX SIGNAL
			</div>
		</div>
	)
}

export default function RadarDashboard({ initialFeed }: { initialFeed: RadarFeed }) {
	const [now, setNow] = useState<number | null>(null)
	const [filter, setFilter] = useState<'all' | ResetKind>('all')
	const [query, setQuery] = useState('')
	const [visibleCount, setVisibleCount] = useState(12)
	const calendarRef = useRef<HTMLDivElement>(null)
	const {
		data = initialFeed,
		error,
		isValidating,
		mutate
	} = useSWR('/api/radar', fetchFeed, {
		fallbackData: initialFeed,
		refreshInterval: 60000,
		revalidateOnFocus: false,
		errorRetryCount: 2
	})

	useEffect(() => {
		setNow(Date.now())
		const timer = window.setInterval(() => setNow(Date.now()), 1000)
		return () => window.clearInterval(timer)
	}, [])

	const stats = useMemo(() => getRadarStats(data.events, now ?? 0), [data.events, now])
	const today = now === null ? null : dayKey(now)
	const calendar = useMemo(() => (today ? buildCalendar(stats.confirmed, today) : []), [stats.confirmed, today])
	useEffect(() => {
		const element = calendarRef.current
		if (!today || !element) return
		const alignToToday = () => {
			element.scrollLeft = element.scrollWidth
		}
		alignToToday()
		const observer = new ResizeObserver(alignToToday)
		observer.observe(element)
		return () => observer.disconnect()
	}, [today])
	const visibleEvents = useMemo(() => {
		const text = query.trim().toLocaleLowerCase()
		return data.events
			.filter(
				event =>
					(filter === 'all' || event.kind === filter) && (!text || `${event.title} ${event.summary} ${event.sourceName}`.toLocaleLowerCase().includes(text))
			)
			.sort((a, b) => Date.parse(b.announcedAt) - Date.parse(a.announcedAt))
	}, [data.events, filter, query])
	const remainingSeconds = stats.upcoming && now !== null ? Math.max(0, Math.ceil((Date.parse(stats.upcoming.eventAt!) - now) / 1000)) : null
	const elapsedSeconds = stats.lastReset && now !== null ? Math.max(0, Math.floor((now - Date.parse(stats.lastReset.eventAt!)) / 1000)) : null
	const countdown =
		remainingSeconds === null
			? elapsedSeconds === null
				? ['--', '--', '--']
				: [Math.floor(elapsedSeconds / 86400), Math.floor((elapsedSeconds % 86400) / 3600), Math.floor((elapsedSeconds % 3600) / 60)].map(value =>
						String(value).padStart(2, '0')
					)
			: [Math.floor(remainingSeconds / 3600), Math.floor((remainingSeconds % 3600) / 60), remainingSeconds % 60].map(value => String(value).padStart(2, '0'))
	const countdownUnits = remainingSeconds === null && elapsedSeconds !== null ? ['天', '小时', '分钟'] : ['小时', '分钟', '秒']
	const statusLabel = stats.upcoming ? '已有重置预告' : stats.awaitingConfirmation ? '等待生效确认' : '等待重置公告'
	useEffect(() => {
		setVisibleCount(12)
	}, [filter, query])

	function refresh() {
		// SWR exposes fetch errors while keeping the last successful records visible.
		void mutate().catch(() => undefined)
	}

	return (
		<div className={styles.page}>
			<div className={styles.topbar}>
				<div className={styles.eyebrow}>
					<Radio size={15} /> CODEX / RESET RADAR
				</div>
				<a href='https://chatgpt.com/codex/settings/usage' target='_blank' rel='noopener noreferrer' className={styles.usageLink}>
					查看我的额度 <ArrowUpRight size={15} />
				</a>
			</div>

			<section className={styles.hero} aria-labelledby='radar-title'>
				<div className={styles.heroContent}>
					<div className={styles.badge}>
						<span /> 关注每一次额度重置
					</div>
					<h1 id='radar-title'>
						Codex <span>重置雷达</span>
						<Sparkles className={styles.titleSpark} size={25} strokeWidth={1.5} />
					</h1>
					<p className={styles.intro}>
						把重置消息，放在一起。
						<br />
						额度重置、重置卡与历史公告，一眼掌握。
					</p>
					<div className={styles.countdownLabel}>
						<Clock3 size={15} /> {stats.upcoming ? '距离公告中的预计重置时间' : stats.lastReset ? '距离上次额度重置已过去' : statusLabel}
					</div>
					<div
						className={styles.countdown}
						aria-label={
							remainingSeconds === null
								? elapsedSeconds === null
									? '暂无可用的重置记录'
									: `距上次重置 ${countdown[0]} 天 ${countdown[1]} 小时 ${countdown[2]} 分钟`
								: `距预计重置还有 ${countdown[0]} 小时 ${countdown[1]} 分钟 ${countdown[2]} 秒`
						}>
						{countdown.map((value, index) => (
							<div className={styles.countdownGroup} key={index}>
								<strong>{value}</strong>
								<span>{countdownUnits[index]}</span>
							</div>
						))}
					</div>
					<div className={styles.announcement}>
						<div className={styles.announcementIcon}>
							<Signal size={20} />
						</div>
						<div>
							<strong>{stats.upcoming?.title ?? (stats.lastReset ? '上次重置记录' : statusLabel)}</strong>
							<p>
								{stats.upcoming
									? `${fullDateFormat.format(new Date(stats.upcoming.eventAt!))}（北京时间）`
									: stats.awaitingConfirmation
										? '预计时间已到，确认后再计入历史记录。'
										: stats.lastReset
											? `${fullDateFormat.format(new Date(stats.lastReset.eventAt!))}（北京时间）`
											: '收录已核实的预告后，这里会显示倒计时。'}
							</p>
						</div>
						{stats.upcoming && (
							<a href={stats.upcoming.sourceUrl} target='_blank' rel='noopener noreferrer' aria-label='查看重置预告原文'>
								<ArrowUpRight size={20} />
							</a>
						)}
					</div>
				</div>
				<RadarVisual />
				<div className={styles.heroFootnote}>
					<Info size={13} /> 独立整理，非官方公告 · 时间统一为 UTC+8
				</div>
			</section>

			<section className={styles.stats} aria-label='重置统计'>
				<div className={styles.statCard}>
					<div className={styles.statLabel}>
						<Clock3 size={16} /> 距上次额度重置
					</div>
					<div className={styles.statValue}>
						{stats.daysSinceReset === null ? '—' : stats.daysSinceReset.toFixed(1)}
						<span>天</span>
					</div>
					<div className={styles.statTrack}>
						<span
							style={{
								width:
									stats.daysSinceReset === null || stats.averageDays === null || stats.averageDays <= 0
										? '0%'
										: `${Math.min(100, (stats.daysSinceReset / stats.averageDays) * 100)}%`
							}}
						/>
					</div>
					<p>{stats.averageDays === null ? '收录足够的记录后显示平均间隔' : `历史平均间隔 ${stats.averageDays.toFixed(1)} 天，仅供参考`}</p>
				</div>
				<div className={styles.statCard}>
					<div className={styles.statLabel}>
						<History size={16} /> 累计收录
					</div>
					<div className={styles.statValue}>
						{stats.confirmed.length}
						<span>条历史记录</span>
					</div>
					<div className={styles.statBreakdown}>
						<span>
							<i />
							额度重置 {stats.resetCount}
						</span>
						<span>
							<i />
							重置卡 {stats.bankedCount}
						</span>
					</div>
					<p>按来源收录的历史事件统计</p>
				</div>
				<div className={styles.statCard}>
					<div className={styles.statLabel}>
						<CalendarDays size={16} /> 近 30 天
					</div>
					<div className={styles.statValue}>
						{stats.last30Days}
						<span>条历史记录</span>
					</div>
					<div className={styles.statNote}>
						<Gift size={16} />
						<span>不定期发生，留意官方消息</span>
					</div>
					<p>仅统计本站收录的记录</p>
				</div>
			</section>

			<section className={styles.panel} aria-labelledby='calendar-title'>
				<div className={styles.sectionHeading}>
					<div>
						<h2 id='calendar-title'>
							<CalendarDays size={19} /> 重置日历
						</h2>
						<p>过去半年，留住每一次重置的足迹</p>
					</div>
					<span className={styles.period}>最近 26 周</span>
				</div>
				<div className={styles.calendarScroll} ref={calendarRef} tabIndex={0} role='region' aria-label='重置日历，可横向滚动'>
					<div className={styles.calendar}>
						<div className={styles.weekdays}>
							<span>一</span>
							<span>三</span>
							<span>五</span>
							<span>日</span>
						</div>
						<div className={styles.calendarWeeks}>
							{calendar.map((week, index) => {
								const month = Number(week[0].key.slice(5, 7))
								const showMonth = index === 0 || week[0].key.slice(0, 7) !== calendar[index - 1][0].key.slice(0, 7)
								return (
									<div className={styles.calendarWeek} key={week[0].key}>
										<span className={styles.month}>{showMonth ? `${month}月` : ''}</span>
										{week.map(day => (
											<span
												key={day.key}
												className={`${styles.calendarDay} ${day.future ? styles.futureDay : ''} ${day.resets && day.banked ? styles.mixedDay : day.banked ? styles.bankedDay : day.resets ? styles.activeDay : ''}`}
												title={day.future ? `${day.key}：尚未到来` : `${day.key}：额度重置 ${day.resets} 次，重置卡 ${day.banked} 次`}
												aria-label={`${day.key}，${day.future ? '尚未到来' : `额度重置 ${day.resets} 次，重置卡 ${day.banked} 次`}`}
											/>
										))}
									</div>
								)
							})}
							{calendar.length === 0 && <div className={styles.calendarPlaceholder}>正在生成日历…</div>}
						</div>
					</div>
				</div>
				<div className={styles.calendarFooter}>
					<span>{stats.confirmed.length ? '鼠标悬停在日期上查看记录数量' : '尚未收录已确认事件，空白日期不代表当日没有重置'}</span>
					<div>
						<i />
						未收录
						<i className={styles.legendActive} />
						额度重置
						<i className={styles.legendBanked} />
						重置卡
					</div>
				</div>
			</section>

			<section className={styles.panel} aria-labelledby='history-title'>
				<div className={styles.sectionHeading}>
					<div>
						<h2 id='history-title'>
							<History size={19} /> 消息记录<span className={styles.countBadge}>{data.events.length}</span>
						</h2>
						<p>预告与已确认记录分开展示，每条消息可追溯来源</p>
					</div>
					<button type='button' className={styles.refreshButton} onClick={refresh} disabled={isValidating}>
						<RefreshCw size={15} className={isValidating ? styles.spinning : ''} />
						{isValidating ? '同步中' : '刷新记录'}
					</button>
				</div>
				{(error || data.sync?.stale) && (
					<div className={styles.error} role='alert'>
						自动同步暂时不可用，当前展示已有历史记录。请稍后点击刷新重试。
					</div>
				)}
				<div className={styles.filters}>
					<div className={styles.tabs} role='group' aria-label='消息类型'>
						{(['all', 'reset', 'banked'] as const).map(type => (
							<button
								type='button'
								key={type}
								aria-pressed={filter === type}
								className={filter === type ? styles.selectedTab : ''}
								onClick={() => setFilter(type)}>
								{type === 'all' ? '全部消息' : KIND_LABELS[type]}
							</button>
						))}
					</div>
					<label className={styles.search}>
						<Search size={16} />
						<input type='search' value={query} onChange={event => setQuery(event.target.value)} placeholder='搜索公告或来源' aria-label='搜索消息记录' />
					</label>
				</div>
				{visibleEvents.length ? (
					<ol className={styles.eventList}>
						{visibleEvents.slice(0, visibleCount).map(event => (
							<li key={event.id} className={styles.event}>
								<div className={`${styles.eventIcon} ${event.kind === 'banked' ? styles.giftIcon : ''}`}>
									{event.kind === 'banked' ? <Gift size={20} /> : <RefreshCw size={20} />}
								</div>
								<div className={styles.eventBody}>
									<div className={styles.eventMeta}>
										<span className={styles.eventTag}>{KIND_LABELS[event.kind]}</span>
										<span className={event.status === 'confirmed' ? styles.confirmed : styles.scheduled}>
											{event.status === 'confirmed' ? <Check size={12} /> : <Clock3 size={12} />}
											{event.status === 'cancelled'
												? '已取消'
												: event.status === 'confirmed'
													? event.evidence === 'source-observed'
														? '来源观察'
														: '已收录'
													: '预告 · 待确认'}
										</span>
										<time dateTime={event.announcedAt}>{dateFormat.format(new Date(event.announcedAt))} 发布</time>
									</div>
									<h3>{event.title}</h3>
									<p>{event.summary}</p>
									<a href={event.sourceUrl} target='_blank' rel='noopener noreferrer'>
										{event.sourceName}
										<ArrowUpRight size={13} />
									</a>
								</div>
							</li>
						))}
					</ol>
				) : (
					<div className={styles.emptyState}>
						<div>
							<Radio size={26} strokeWidth={1.4} />
						</div>
						<h3>{data.events.length ? '没有匹配的消息' : '静候下一条重置消息'}</h3>
						<p>{data.events.length ? '试试其他关键词，或切换消息类型。' : '本站尚未收录重置公告。有经过核实的消息后，会在这里留下记录。'}</p>
						{data.events.length ? (
							<button
								type='button'
								onClick={() => {
									setFilter('all')
									setQuery('')
								}}>
								清除筛选 <ArrowDown size={13} />
							</button>
						) : (
							<a href='https://developers.openai.com/codex/changelog' target='_blank' rel='noopener noreferrer'>
								浏览官方更新 <ArrowUpRight size={14} />
							</a>
						)}
					</div>
				)}
				{visibleEvents.length > visibleCount && (
					<button type='button' className={styles.loadMore} onClick={() => setVisibleCount(count => count + 12)}>
						显示更多记录（剩余 {visibleEvents.length - visibleCount} 条）
						<ArrowDown size={14} />
					</button>
				)}
				<div className={styles.feedFooter}>
					<span>
						<span className={styles.syncDot} />
						自动同步 · 每分钟刷新
					</span>
					<span>
						{data.sync
							? `${data.sync.mode === 'snapshot' ? '历史快照' : data.sync.mode === 'x' ? 'X 推文检测' : '参考站同步'} · ${fullDateFormat.format(new Date(data.sync.checkedAt))}`
							: data.updatedAt
								? `记录更新于 ${fullDateFormat.format(new Date(data.updatedAt))}`
								: '等待首次记录更新'}
					</span>
				</div>
			</section>

			<footer className={styles.footer}>
				<span>少一点等待，多一点创造。</span>
				<a href='https://developers.openai.com/codex/pricing' target='_blank' rel='noopener noreferrer'>
					了解 Codex 额度规则 <ArrowUpRight size={13} />
				</a>
				<p>
					历史及实时数据经{' '}
					<a href='https://prodclub.xyz/radar' target='_blank' rel='noopener noreferrer'>
						ProdClub
					</a>{' '}
					整理并保留原帖链接；来源观察记录不代表本站独立确认。公告时间不等于账户实际生效时间。
				</p>
			</footer>
		</div>
	)
}
