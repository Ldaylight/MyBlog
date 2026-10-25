'use client'

import Card from '@/components/card'
import Image from 'next/image'
import Link, { useLinkStatus } from 'next/link'
import { useEffect, useMemo, useState } from 'react'
import { useCenterStore } from '@/hooks/use-center'
import { CARD_SPACING } from '@/consts'
import ScrollOutlineSVG from '@/svgs/scroll-outline.svg'
import ScrollFilledSVG from '@/svgs/scroll-filled.svg'
import ProjectsFilledSVG from '@/svgs/projects-filled.svg'
import ProjectsOutlineSVG from '@/svgs/projects-outline.svg'
import AboutFilledSVG from '@/svgs/about-filled.svg'
import AboutOutlineSVG from '@/svgs/about-outline.svg'
import ShareFilledSVG from '@/svgs/share-filled.svg'
import ShareOutlineSVG from '@/svgs/share-outline.svg'
import WebsiteFilledSVG from '@/svgs/website-filled.svg'
import WebsiteOutlineSVG from '@/svgs/website-outline.svg'
import { usePathname } from 'next/navigation'
import clsx from 'clsx'
import { cn } from '@/lib/utils'
import { useSize } from '@/hooks/use-size'
import { useConfigStore } from '@/app/(home)/stores/config-store'
import { HomeDraggableLayer } from '@/app/(home)/home-draggable-layer'
import { LoaderCircle, Radar } from 'lucide-react'

const list = [
	{
		icon: ScrollOutlineSVG,
		iconActive: ScrollFilledSVG,
		label: '近期文章',
		href: '/blog'
	},
	{
		icon: ProjectsOutlineSVG,
		iconActive: ProjectsFilledSVG,
		label: '我的项目',
		href: '/projects'
	},
	{
		icon: AboutOutlineSVG,
		iconActive: AboutFilledSVG,
		label: '关于网站',
		href: '/about'
	},
	{
		icon: ShareOutlineSVG,
		iconActive: ShareFilledSVG,
		label: '推荐分享',
		href: '/share'
	},
	{
		icon: WebsiteOutlineSVG,
		iconActive: WebsiteFilledSVG,
		label: '优秀博客',
		href: '/bloggers'
	},
	{
		icon: Radar,
		iconActive: Radar,
		label: 'Codex 雷达',
		href: '/radar'
	}
]

const extraSize = 8

function NavigationPending() {
	const { pending } = useLinkStatus()
	if (!pending) return null
	return (
		<span role='status' data-nav-pending className='text-brand pointer-events-none absolute -right-1 -bottom-1 rounded-full bg-white p-0.5'>
			<LoaderCircle size={12} className='animate-spin' aria-hidden='true' />
			<span className='sr-only'>正在切换页面</span>
		</span>
	)
}

export default function NavCard() {
	const pathname = usePathname()
	const center = useCenterStore()
	const [show, setShow] = useState(false)
	const { maxSM } = useSize()
	const [hoveredIndex, setHoveredIndex] = useState<number>()
	const { siteContent, cardStyles } = useConfigStore()
	const styles = cardStyles.navCard
	const hiCardStyles = cardStyles.hiCard

	const activeIndex = useMemo(() => {
		const index = list.findIndex(item => pathname === item.href)
		return index >= 0 ? index : undefined
	}, [pathname])
	const highlightedIndex = hoveredIndex ?? activeIndex ?? 0

	useEffect(() => {
		setShow(true)
	}, [])

	let form = useMemo(() => {
		if (pathname == '/') return 'full'
		else if (pathname == '/write') return 'mini'
		else return 'icons'
	}, [pathname])
	if (maxSM) form = 'icons'

	const itemHeight = form === 'full' ? 52 : 28
	const compactWidth = maxSM ? Math.min(376, Math.max(280, center.width - 24)) : 376
	const iconGap = (compactWidth - 24 - 40 - list.length * 28) / list.length

	let position = useMemo(() => {
		if (form === 'full') {
			const x = styles.offsetX !== null ? center.x + styles.offsetX : center.x - hiCardStyles.width / 2 - styles.width - CARD_SPACING
			const y = styles.offsetY !== null ? center.y + styles.offsetY : center.y + hiCardStyles.height / 2 - styles.height
			return { x, y }
		}

		return {
			x: 24,
			y: 16
		}
	}, [form, center, styles, hiCardStyles])

	const size = useMemo(() => {
		if (form === 'mini') return { width: 64, height: 64 }
		else if (form === 'icons') return { width: compactWidth, height: 64 }
		else return { width: styles.width, height: styles.height }
	}, [form, styles, compactWidth])

	if (maxSM) position = { x: center.x - size.width / 2, y: 16 }

	if (show)
		return (
			<HomeDraggableLayer cardKey='navCard' x={position.x} y={position.y} width={styles.width} height={styles.height}>
				<Card
					interactive={false}
					animateEntrance={false}
					order={styles.order}
					width={size.width}
					height={size.height}
					x={position.x}
					y={position.y}
					className={clsx('z-30', form != 'full' && 'overflow-hidden', form === 'mini' && 'p-3', form === 'icons' && 'flex items-center p-3')}>
					{form === 'full' && siteContent.enableChristmas && (
						<>
							<img
								src='/images/christmas/snow-4.webp'
								alt='Christmas decoration'
								className='pointer-events-none absolute'
								style={{ width: 160, left: -18, top: -20, opacity: 0.9 }}
							/>
						</>
					)}

					<Link
						className='relative flex shrink-0 items-center gap-3 [&_img]:pointer-events-none [&_span]:pointer-events-none'
						href='/'
						aria-label='返回首页'
						style={form === 'icons' ? { marginRight: iconGap } : undefined}>
						<Image src='/images/avatar.png' alt='avatar' width={40} height={40} style={{ boxShadow: ' 0 12px 20px -5px #E2D9CE' }} className='rounded-full' />
						{form === 'full' && <span className='font-averia mt-1 text-2xl leading-none font-medium'>{siteContent.meta.title}</span>}
						{form === 'full' && <span className='text-brand mt-2 text-xs font-medium'>(开发中)</span>}
						<NavigationPending />
					</Link>

					{(form === 'full' || form === 'icons') && (
						<>
							{form !== 'icons' && <div className='text-secondary mt-6 text-sm uppercase'>General</div>}

							<div
								onPointerLeave={() => setHoveredIndex(undefined)}
								onBlur={event => {
									if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHoveredIndex(undefined)
								}}
								className={cn('relative mt-2 space-y-2', form === 'icons' && 'mt-0 flex items-center space-y-0')}
								style={form === 'icons' ? { gap: iconGap } : undefined}>
								<div
									className='pointer-events-none absolute max-w-[230px] rounded-full border transition-transform duration-150 ease-out motion-reduce:transition-none'
									style={{
										left: form === 'icons' ? -extraSize : 0,
										top: form === 'icons' ? -extraSize : 0,
										width: form === 'icons' ? itemHeight + extraSize * 2 : '100%',
										height: form === 'icons' ? itemHeight + extraSize * 2 : itemHeight,
										transform:
											form === 'icons' ? `translateX(${highlightedIndex * (itemHeight + iconGap)}px)` : `translateY(${highlightedIndex * (itemHeight + 8)}px)`,
										backgroundImage: 'linear-gradient(to right bottom, var(--color-border) 60%, var(--color-card) 100%)'
									}}
								/>

								{list.map((item, index) => (
									<Link
										key={item.href}
										href={item.href}
										aria-label={item.label}
										title={item.label}
										aria-current={pathname === item.href ? 'page' : undefined}
										onFocus={() => setHoveredIndex(index)}
										className={cn(
											'text-secondary text-md relative z-10 flex items-center gap-3 rounded-full px-5 py-3 [&_svg]:pointer-events-none',
											form === 'icons' && 'p-0'
										)}
										onMouseEnter={() => setHoveredIndex(index)}>
										<div className='pointer-events-none flex h-7 w-7 items-center justify-center' aria-hidden='true'>
											<item.iconActive
												className={cn('text-brand absolute h-7 w-7 transition-opacity duration-100', highlightedIndex === index ? 'opacity-100' : 'opacity-0')}
											/>
											<item.icon className={cn('absolute h-7 w-7 transition-opacity duration-100', highlightedIndex === index ? 'opacity-0' : 'opacity-100')} />
										</div>
										{form !== 'icons' && (
											<span className={clsx('pointer-events-none', index === highlightedIndex && 'text-primary font-medium')}>{item.label}</span>
										)}
										<NavigationPending />
									</Link>
								))}
							</div>
						</>
					)}
				</Card>
			</HomeDraggableLayer>
		)
}
