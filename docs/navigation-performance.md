# 导航卡顿定位与优化

## 已复现的根因

图标导航在 `onMouseEnter` / `onFocus` 时，把普通 SVG 换成另一个高亮 SVG 组件。按下与松开之间，原来的事件目标可能已被移除。浏览器收到 `pointerdown` 和 `pointerup`，却没有产生 `click`，因此首次点击不触发 Next.js 路由。这不是等待完成后一定会跳转的状态。

生产构建中的独立对照测试如下（每组连续切换三个不同页面）：

| 测试条件                | 首次点击未跳转 |
| ----------------------- | -------------- |
| 原版                    | 3 / 3          |
| 仅取消整栏缩放          | 3 / 3          |
| 仅禁止 SVG 接收指针事件 | 0 / 3          |
| 两项同时处理            | 0 / 3          |

修复后由稳定的 Link 元素接收事件，普通/高亮图标同时保留并通过透明度切换；装饰节点不接收指针事件。导航卡片不再整体悬停放大、按压缩小或延时入场；高亮条使用 CSS transform，移除了布局测量和 1.5 秒延时复位。

## 资源与渲染架构调整

- GitHub 的 RSA 签名库只在实际签名时导入。发布调用方等待异步签名结果；阅读页面无需先加载签名库。
- 首页设置弹窗、项目/分享/博客的新增弹窗以及保存服务按操作加载，阅读路径不再绑定全部编辑依赖。
- Markdown 先创建独立 Marked 实例、识别 token，再按需加载 Shiki 和 KaTeX。纯文字页面不加载代码高亮或公式库，列表中的代码块也能正常处理。
- 同时渲染的文档共用加载 Promise，各自持有 parser 和代码映射；不再每次渲染累积全局插件。小文档保留最多 8 条渲染缓存；长文档分批代码高亮并允许离开页面时取消后续工作。
- 音频设置 `preload=none`。原版在打开雷达等页面时也会请求背景音频；文件大小约 3.56 MB，实际预取字节由浏览器决定。现在用户点击播放才开始请求。
- 保留 Next.js Link 的预取和原生打开新标签页行为，增加 `useLinkStatus` 反馈和路由 `loading.tsx`，慢网时能够立即看到正在切换的状态。

实现依据：[Next.js 导航说明](https://nextjs.org/docs/app/getting-started/linking-and-navigating)、[按需加载说明](https://nextjs.org/docs/app/guides/lazy-loading)、[useLinkStatus](https://nextjs.org/docs/app/api-reference/functions/use-link-status)。

## 受控生产构建测试

使用本机 `next start`，Chromium，网络延迟 100 ms、下载 1 MiB/s、CPU 4 倍限速。桌面 1440×1000，手机模拟 390×844、DPR 2。两组均阻止外部字体及统计请求，其他测试条件一致。JavaScript 体积为 Resource Timing 的解码后大小，不等于压缩后的实际流量。

| 指标                         | 优化前   | 优化后                 |
| ---------------------------- | -------- | ---------------------- |
| 首页 JavaScript              | 1832 KiB | 1122 KiB（约减少 39%） |
| 首次关于页额外 JavaScript    | 482 KiB  | 0 KiB                  |
| 首次关于页内容显示，桌面     | 1280 ms  | 627 ms                 |
| 首次关于页内容显示，手机模拟 | 1222 ms  | 636 ms                 |

内容显示测量包含 100 ms 的稳定等待。以上是受控单次样本，不是线上 INP/P95，也不用于保证校园网中的网络延迟。热路由仍包含页面渲染、图片和动画成本，测试样本约 0.2–0.5 秒。

## 回归验证

- 桌面、390 px 和 320 px 手机模拟共 36 次点击/真实触摸事件，全部一次切换，无浏览器运行错误。
- 慢速路由请求期间能显示加载反馈。
- 设置弹窗、项目新增弹窗可在首次使用时加载；音频在播放之前无请求。
- 使用临时生成的测试密钥验证浏览器中的 RS256 签名；所有 GitHub 请求均模拟拦截，没有对外执行写入。
- 7 项 Markdown/签名测试和原有 11 项雷达测试通过，完整 TypeScript 检查和生产构建通过。
- 文章目录及站点内容配置保留原样。

本地逻辑测试：

```powershell
node --experimental-strip-types scripts/rendering.test.mjs
node --experimental-strip-types scripts/radar.test.mjs
pnpm exec tsc --noEmit
pnpm build
```
