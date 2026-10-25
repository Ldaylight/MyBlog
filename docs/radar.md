# Codex 重置雷达

本地路由 `/radar`；博客导航入口为「Codex 雷达」。

## 当前可直接使用的自动同步

服务端 `/api/radar` 读取参考站公开接口 `https://prodclub.xyz/api/v1/radar/codex`，缓存 60 秒，浏览器每分钟刷新。无需登录参考站、X 或配置模型密钥。历史来自该站，保留原推文链接和来源标记。首次导入为 66 条历史（60 条重置、6 条重置卡），另有 1 条预告。

这条路径是**自动同步参考站已解析的公告**，不声称本站独立读取和判断了 X 推文。第三方接口可能变更；失败时返回历史快照并显示提示。热力图空白日期仅代表未收录。

- 有明确的未来预告：使用接口 `scheduled_for` 的绝对时间，换算为北京时间（`Asia/Shanghai`）并倒计时，不再次猜测原帖中的 PST/PT。
- 没有未来预告：显示距最近一次额度重置的天、小时、分钟，以及上次记录的北京时间。
- 到预告时间不代表已生效，不会自动把预告计为重置。
- 日历：额度重置绿色、重置卡黄色；同一天两种事件使用双色。
- `observed` 记录标为「来源观察」，不冒充官方已确认。历史未提供精确生效时间时，采用来源记录/公告时间并说明时间口径。

## 无人访问时的后台同步

`.github/workflows/radar-sync.yml` 每小时第 7、17、27、37、47、57 分钟运行，也支持 Actions 手动运行。脚本读取公开接口、合并新增历史并去重，**仅记录变化时提交 `public/radar/events.json`**。工作流推送到默认分支并启用 Actions 后生效。

GitHub 定时任务可能延迟，不保证精确十分钟执行。仓库需要允许 Actions 的 `contents: write`，分支保护也可能限制机器人直接推送。检测状态使用 Actions cache，避免将每次检查时间提交到仓库。Vercel 保持现有 Git 集成；数据变化的提交可以触发部署。浏览器访问期间的数据更新不依赖部署，仍走服务端自动同步。

没有添加 Vercel 分钟级 Cron：Hobby 计划只支持每日运行一次。参见 [Vercel Cron 限制](https://vercel.com/docs/cron-jobs/usage-and-pricing)、[GitHub schedule 说明](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)。

本地同步命令（Node.js 22.15+）：

```sh
pnpm radar:sync
```

## 可选：独立监测 X 推文并理解文本

代码在 `scripts/radar/x-monitor.mjs`，由同一个后台脚本调用。默认只启用不需要密钥的参考站同步。独立检测需要在 GitHub 仓库 Settings → Secrets and variables → Actions 设置以下项目，不要把密钥写进代码、公开 JSON 或 `NEXT_PUBLIC_` 环境变量：

| 类型     | 名称                 | 用途                                                               |
| -------- | -------------------- | ------------------------------------------------------------------ |
| Variable | `RADAR_SYNC_MODE`    | `upstream`：参考站；`x`：独立 X 检测；`hybrid`：两者合并           |
| Variable | `RADAR_X_ACCOUNTS`   | 默认 `thsottiaux`（历史数据实际来源）；逗号分隔，最多 8 个可信账号 |
| Variable | `RADAR_OPENAI_MODEL` | 填入你的 API 项目可用且支持 Responses 结构化输出的模型 ID          |
| Secret   | `X_BEARER_TOKEN`     | X API 读取推文的凭证                                               |
| Secret   | `OPENAI_API_KEY`     | 用于文本理解，不读取个人 Codex 额度                                |

模板见 `docs/radar.env.example`。本地执行时需把变量加入当前 shell 环境；独立 Node 脚本不会自动读取 `.env.local`。

流程：

1. X Recent Search 读取指定账号的原帖与回复（排除转推），首次覆盖最近 7 天；后续使用 `since_id`，包含分页，避免漏掉两次检查间的新帖。认证失败、分页超限、作者不匹配时停止，不推进游标。
2. 用原帖、引用/回复上下文、近期已有事件调用 OpenAI Responses API，严格结构化提取重置/重置卡、预告/发生/取消、影响范围、明确时间、证据和置信度。
3. 普通用量窗口、个人账号重置、问题、否定表达、玩笑、低置信度和无可靠时间的预测不生成倒计时；没有时间的明确预告可收录但不猜时间。
4. 仅接受广泛用户事件、置信度至少 0.9、证据确实存在于原帖、可信账号和有效时间。简短的“Done”需要上下文能明确关联原预告，否则弃权。
5. 明确的发生公告可确认；预告时间到达不会自动确认。取消只能关联已有预告。确认后按来源 URL/事件 ID 去重，更新同一事件而不重复计数。

模型解析并非百分之百准确；保留原帖链接用于复核。当前没有配置这些密钥，**独立 X + 模型路径只做了模拟接口和逻辑测试，未声称已经实时运行**。X API 与模型可能产生费用，查看你账户的服务方案。当前实时网页继续读取参考站接口，`x`/`hybrid` 检测产生的记录通过 Git 数据部署加入，并优先保留直接检测的结果。

官方接口说明：[X Recent Search](https://docs.x.com/x-api/posts/search-recent-posts)、[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)。

## 数据与维护

`public/radar/events.json` 是可部署的历史快照，字段与校验在 `src/app/radar/data.ts`。`updatedAt` 表示最新公告时间，`sync.checkedAt` 是来源检查时间，两者不同。抓取失败不会把检查时间伪装成最新成功时间。

`status` 为 `scheduled` / `confirmed` / `cancelled`；`kind` 为 `reset` / `banked`。所有日期必须是带时区的 ISO 时间。`evidence` 区分 `source-record`、`source-observed`、`direct-post`；来源观察记录的事实准确性继承上游，不代表本站独立核实。

执行逻辑测试：

```sh
node --experimental-strip-types scripts/radar.test.mjs
```
