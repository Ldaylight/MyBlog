export default function Loading() {
	return (
		<div role='status' aria-live='polite' className='mx-auto max-w-4xl px-6 pt-32 pb-12'>
			<div className='mb-6 h-7 w-40 rounded-xl bg-white/70 motion-safe:animate-pulse' aria-hidden='true' />
			<div className='space-y-4 rounded-3xl border bg-white/50 p-6' aria-hidden='true'>
				<div className='h-4 w-3/4 rounded bg-white/80 motion-safe:animate-pulse' />
				<div className='h-4 w-full rounded bg-white/80 motion-safe:animate-pulse' />
				<div className='h-4 w-2/3 rounded bg-white/80 motion-safe:animate-pulse' />
			</div>
			<p className='text-secondary mt-4 text-sm'>正在打开页面…</p>
		</div>
	)
}
