export async function signAppJwt(appId: string, privateKeyPem: string): Promise<string> {
	const module = await import('jsrsasign')
	const { KJUR, KEYUTIL } = module.default ?? module
	const now = Math.floor(Date.now() / 1000)
	const header = { alg: 'RS256', typ: 'JWT' }
	const payload = { iat: now - 60, exp: now + 8 * 60, iss: appId }
	const key = KEYUTIL.getKey(privateKeyPem) as unknown as string
	return KJUR.jws.JWS.sign('RS256', JSON.stringify(header), JSON.stringify(payload), key)
}
