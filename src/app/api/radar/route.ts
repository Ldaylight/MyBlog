import { getRadarFeed } from '@/lib/radar/server'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

export async function GET() {
	return Response.json(await getRadarFeed(), { headers: { 'Cache-Control': 'no-store' } })
}
