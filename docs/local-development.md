# 本地开发与线上内容同步

项目位置：`E:\MyBlog\2025-blog-public`。Git 远端为 `https://github.com/Ldaylight/MyBlog.git`，日常开发使用 `main`。

## 开始开发

在项目目录中执行：

```powershell
git status
git pull --ff-only origin main
pnpm install --frozen-lockfile
pnpm dev
```

打开 `http://localhost:2025`，雷达页为 `http://localhost:2025/radar`。依赖已安装且锁文件没有变化时，可以省略安装步骤。同步 `.env.local` 后需重启开发服务，公开环境变量在构建时生效。

如果 `git status` 显示未提交修改，先提交完成的工作，或妥善保存改动后再同步。不要使用强制推送或强制重置来处理网页端新增的文章。

## 为什么网页文章会比本地新

博客里的写作、删除、设置保存操作，通过 GitHub App 把更改提交到仓库，Vercel 再重新部署。本地目录不会随之自动更新，所以每次本地开发前都应先拉取远端。GitHub 上的文章和站点配置是同步基础。

## 环境配置

`.env.local` 只保留在本机，不提交。当前项目对应：

```dotenv
NEXT_PUBLIC_GITHUB_OWNER=Ldaylight
NEXT_PUBLIC_GITHUB_REPO=MyBlog
NEXT_PUBLIC_GITHUB_BRANCH=main
```

GitHub App ID 使用你已有的值；其他已有环境配置继续保留。Vercel 的环境变量在其项目设置中管理，本地文件不会自动覆盖它们。

## 验证与发布

```powershell
pnpm exec tsc --noEmit
node --experimental-strip-types scripts/radar.test.mjs
pnpm build
git diff
git add <本次需要提交的文件>
git commit -m "描述这次修改"
git fetch origin
git rebase origin/main
git push origin main
```

若其他人或网页端在开发期间更新了仓库，先合并远端的新提交，再推送。推送后可在 Vercel 查看部署状态；GitHub 推送成功本身不代表部署已经完成。

雷达的自动同步及可选 X 推文检测配置见 [radar.md](radar.md)。
