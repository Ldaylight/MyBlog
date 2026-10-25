'use client'

import { ANIMATION_DELAY } from '@/consts'
import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { useEffect, useState } from 'react'
import { useSize } from '@/hooks/use-size'

interface Props {
	className?: string
	order: number
	width: number
	height?: number
	x: number
	y: number
	children: React.ReactNode
	interactive?: boolean
	animateEntrance?: boolean
}

export default function Card({ children, order, width, height, x, y, className, interactive = true, animateEntrance = true }: Props) {
	const { maxSM, init } = useSize()
	let [show, setShow] = useState(!animateEntrance)
	if (maxSM && init) order = 0

	useEffect(() => {
		if (show) return
		if (x === 0 && y === 0) return
		const timer = setTimeout(
			() => {
				setShow(true)
			},
			order * ANIMATION_DELAY * 1000
		)
		return () => clearTimeout(timer)
	}, [x, y, show, order])

	if (show)
		return (
			<motion.div
				className={cn('card squircle', className)}
				initial={animateEntrance ? { opacity: 0, scale: 0.6, left: x, top: y, width, height } : false}
				animate={{ opacity: 1, scale: 1, left: x, top: y, width, height }}
				whileHover={interactive ? { scale: 1.05 } : undefined}
				whileTap={interactive ? { scale: 0.95 } : undefined}>
				{children}
			</motion.div>
		)

	return null
}
