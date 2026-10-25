import assert from 'node:assert/strict'
import { generateKeyPairSync, verify } from 'node:crypto'
import test from 'node:test'
import { renderMarkdown } from '../src/lib/markdown-renderer.ts'
import { signAppJwt } from '../src/lib/github-jwt.ts'

test('plain Markdown keeps headings, links, task lists and a stable cached result', async () => {
	const markdown = '# 普通页面\n\n[链接](https://example.com)\n\n- [x] 已完成\n- **强调**'
	const result = await renderMarkdown(markdown)
	assert.match(result.html, /id="普通页面"/)
	assert.match(result.html, /href="https:\/\/example.com"/)
	assert.match(result.html, /checked disabled/)
	assert.match(result.html, /<strong>强调<\/strong>/)
	assert.deepEqual(result.toc, [{ id: '普通页面', text: '普通页面', level: 1 }])
	assert.strictEqual(await renderMarkdown(markdown), result)
})

test('inline and block math render on the first pass', async () => {
	const result = await renderMarkdown('# 公式\n\n行内 $x^2$\n\n$$\n\\frac{a}{b}\n$$\n')
	assert.match(result.html, /class="katex"/)
	assert.match(result.html, /katex-display/)
	assert.equal(result.toc.length, 1)
})

test('fenced code is highlighted and code headings stay out of the TOC', async () => {
	const result = await renderMarkdown('# 实际标题\n\n```python\n# not a heading\nprint("hello")\n```')
	assert.match(result.html, /data-code=/)
	assert.match(result.html, /shiki/)
	assert.equal(result.toc.length, 1)
	assert.equal(result.toc[0].text, '实际标题')
})

test('nested code and unknown languages retain literal code content', async () => {
	const result = await renderMarkdown('> ```python\n> print(1)\n> ```\n\n```unknown-language-for-test\n<script>alert(1)</script>\n```')
	assert.match(result.html, /shiki/)
	assert.match(result.html, /&lt;script&gt;/)
	assert.doesNotMatch(result.html, /<script>alert/)
})

test('concurrent documents keep their parser and code maps isolated', async () => {
	const [plain, code, math] = await Promise.all([
		renderMarkdown('# 并发纯文本\n\n内容'),
		renderMarkdown('# 并发代码\n\n```js\nconst unique = 42\n```'),
		renderMarkdown('# 并发公式\n\n$a+b$')
	])
	assert.equal(plain.toc[0].text, '并发纯文本')
	assert.doesNotMatch(plain.html, /SHIKI_CODE|unique|katex/)
	assert.match(code.html, /unique/)
	assert.match(math.html, /katex/)
})

test('aborted documents stop rendering and do not poison later renders', async () => {
	const controller = new AbortController()
	controller.abort()
	await assert.rejects(renderMarkdown('already cancelled', controller.signal), { name: 'AbortError' })
	const longDocument = Array.from({ length: 20 }, (_, i) => `\n\n\`\`\`js\nconst value${i} = ${i}\n\`\`\``).join('')
	const active = new AbortController()
	const pending = renderMarkdown(longDocument, active.signal)
	setTimeout(() => active.abort(), 0)
	await assert.rejects(pending, { name: 'AbortError' })
	assert.match((await renderMarkdown(longDocument)).html, /value19/)
})

test('lazy GitHub signing produces a valid RS256 JWT with the existing claims', async () => {
	const { privateKey, publicKey } = generateKeyPairSync('rsa', { modulusLength: 2048 })
	const pem = privateKey.export({ type: 'pkcs8', format: 'pem' })
	const token = await signAppJwt('test-app-id', pem)
	const [header, payload, signature] = token.split('.')
	assert.deepEqual(JSON.parse(Buffer.from(header, 'base64url').toString()), { alg: 'RS256', typ: 'JWT' })
	const claims = JSON.parse(Buffer.from(payload, 'base64url').toString())
	assert.equal(claims.iss, 'test-app-id')
	assert.equal(claims.exp - claims.iat, 540)
	assert.equal(verify('RSA-SHA256', Buffer.from(`${header}.${payload}`), publicKey, Buffer.from(signature, 'base64url')), true)
})
