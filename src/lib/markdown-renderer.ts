import { Marked, Renderer } from 'marked'
import type { Tokens } from 'marked'

export type TocItem = { id: string; text: string; level: number }

export interface MarkdownRenderResult {
	html: string
	toc: TocItem[]
}

export function slugify(text: string): string {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9\u4e00-\u9fa5\s-]/g, '')
		.trim()
		.replace(/\s+/g, '-')
}

let shikiPromise: Promise<typeof import('shiki') | null> | null = null
let katexPromise: Promise<typeof import('katex') | null> | null = null
const renderCache = new Map<string, MarkdownRenderResult>()

function loadShiki() {
	return (shikiPromise ??= import('shiki').catch(error => {
		shikiPromise = null
		console.warn('Failed to load shiki module:', error)
		return null
	}))
}

function loadKatex() {
	return (katexPromise ??= import('katex')
		.then(mod => (mod.default ?? mod) as typeof import('katex'))
		.catch(error => {
			katexPromise = null
			console.warn('Failed to load katex module:', error)
			return null
		}))
}

async function renderDocument(markdown: string, signal?: AbortSignal) {
	signal?.throwIfAborted()
	// Each document owns its parser; route changes must not accumulate global extensions.
	const parser = new Marked()
	let katex: typeof import('katex') | null = null
	const codeBlockMap = new Map<string, { html: string; original: string }>()

	// Render HTML with heading ids
	const renderer = new Renderer()

	renderer.heading = (token: Tokens.Heading) => {
		const id = slugify(token.text || '')
		return `<h${token.depth} id="${id}">${token.text}</h${token.depth}>`
	}

	renderer.code = (token: Tokens.Code) => {
		// Check if this code block was pre-processed
		const codeData = codeBlockMap.get(token.text)
		if (codeData) {
			// Add data-code attribute with original code for copy functionality
			// Escape HTML entities for attribute value
			const escapedCode = codeData.original.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
			if (codeData.html) {
				// Shiki highlighted code
				return `<pre data-code="${escapedCode}">${codeData.html}</pre>`
			}
			// Fallback for failed highlighting
			const escapedText = codeData.original.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
			return `<pre data-code="${escapedCode}"><code>${escapedText}</code></pre>`
		}
		// Fallback to default (inline code, not code block)
		return `<code>${token.text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code>`
	}

	renderer.listitem = (token: Tokens.ListItem) => {
		// Render inline markdown inside list items (e.g. links, emphasis)
		let inner = token.text
		let tokens = token.tokens

		if (token.task) tokens = tokens.slice(1)
		inner = parser.parser(tokens) as string

		if (token.task) {
			const checkbox = token.checked ? '<input type="checkbox" checked disabled />' : '<input type="checkbox" disabled />'
			return `<li class="task-list-item">${checkbox} ${inner}</li>\n`
		}

		return `<li>${inner}</li>\n`
	}

	const renderMath = (content: string, displayMode: boolean) => {
		if (!katex) {
			// Keep original delimiters if katex is not available
			return displayMode ? `$$${content}$$` : `$${content}$`
		}

		try {
			return katex.renderToString(content, {
				displayMode,
				throwOnError: false,
				output: 'html',
				strict: 'ignore'
			})
		} catch {
			return displayMode ? `$$${content}$$` : `$${content}$`
		}
	}

	// Register extensions BEFORE lexing so math gets tokenized on cold refresh.
	parser.use({
		renderer,
		extensions: [
			// Block math: $$ ... $$
			{
				name: 'mathBlock',
				level: 'block',
				start(src: string) {
					return src.indexOf('$$')
				},
				tokenizer(src: string) {
					const match = src.match(/^\$\$([\s\S]+?)\$\$(?:\n+|$)/)
					if (!match) return
					return {
						type: 'mathBlock',
						raw: match[0],
						text: match[1].trim()
					} as any
				},
				renderer(token: any) {
					return `${renderMath(token.text || '', true)}\n`
				}
			},
			// Inline math: $ ... $
			{
				name: 'mathInline',
				level: 'inline',
				start(src: string) {
					const idx = src.indexOf('$')
					return idx === -1 ? undefined : idx
				},
				tokenizer(src: string) {
					// Avoid $$ (block) and escaped dollars
					if (src.startsWith('$$')) return
					if (src.startsWith('\\$')) return

					const match = src.match(/^\$([^\n$]+?)\$/)
					if (!match) return

					const inner = match[1]
					// Heuristic: require some non-space content
					if (!inner || !inner.trim()) return

					return {
						type: 'mathInline',
						raw: match[0],
						text: inner.trim()
					} as any
				},
				renderer(token: any) {
					return renderMath(token.text || '', false)
				}
			}
		]
	})

	// Pre-process with marked lexer first (after extensions are registered)
	const tokens = parser.lexer(markdown)
	const codeTokens: Tokens.Code[] = []
	let hasMath = false
	parser.walkTokens(tokens, token => {
		if (token.type === 'code') codeTokens.push(token as Tokens.Code)
		if (token.type === 'mathInline' || token.type === 'mathBlock') hasMath = true
	})
	const [shiki, mathRenderer] = await Promise.all([codeTokens.length ? loadShiki() : null, hasMath ? loadKatex() : null])
	katex = mathRenderer
	signal?.throwIfAborted()

	// Extract TOC from parsed tokens (this correctly skips code blocks)
	const toc: TocItem[] = []
	function extractHeadings(tokenList: typeof tokens) {
		for (const token of tokenList) {
			if (token.type === 'heading' && token.depth <= 3) {
				// Use the parsed text (markdown syntax like links/code already stripped)
				const text = token.text
				const id = slugify(text)
				toc.push({ id, text, level: token.depth })
			}
			// Recursively check nested tokens (e.g., in blockquotes, lists)
			if ('tokens' in token && token.tokens) {
				extractHeadings(token.tokens as typeof tokens)
			}
		}
	}
	extractHeadings(tokens)

	// Pre-process code blocks with Shiki
	for (const [index, token] of codeTokens.entries()) {
		if (index > 0 && index % 4 === 0) await new Promise<void>(resolve => setTimeout(resolve, 0))
		signal?.throwIfAborted()
		if (token.type === 'code') {
			const codeToken = token as Tokens.Code
			const originalCode = codeToken.text
			const key = `__SHIKI_CODE_${codeBlockMap.size}__`

			if (shiki) {
				try {
					const html = await shiki.codeToHtml(originalCode, {
						lang: codeToken.lang || 'text',
						theme: 'one-light'
					})
					codeBlockMap.set(key, { html, original: originalCode })
					codeToken.text = key
				} catch {
					// Keep original if highlighting fails
					codeBlockMap.set(key, { html: '', original: originalCode })
					codeToken.text = key
				}
			} else {
				// Fallback when shiki is not available
				codeBlockMap.set(key, { html: '', original: originalCode })
				codeToken.text = key
			}
		}
	}
	signal?.throwIfAborted()
	const html = (parser.parser(tokens) as string) || ''

	return { html, toc, cacheable: (!codeTokens.length || Boolean(shiki)) && (!hasMath || Boolean(katex)) }
}

export async function renderMarkdown(markdown: string, signal?: AbortSignal): Promise<MarkdownRenderResult> {
	signal?.throwIfAborted()
	const cached = renderCache.get(markdown)
	if (cached) return cached
	const { html, toc, cacheable } = await renderDocument(markdown, signal)
	const result = { html, toc }
	if (cacheable && markdown.length <= 100000 && html.length <= 500000) {
		if (renderCache.size >= 8) renderCache.delete(renderCache.keys().next().value!)
		renderCache.set(markdown, result)
	}
	return result
}
