# 维护者手册

> 读者：本仓库的维护者/贡献者。安装使用者请看根目录 README。
> 原则：**main 同时是 `github:` 安装源与目录站文档源，不要直接推 main；npm 版本不可变、可精确回滚，是正式发布门禁。**

## 一次性设置

### 1. GitHub 分支保护

仓库 Settings → Branches → Add branch protection rule：

- Branch name pattern：`main`
- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging → 选 `ci`
- ✅ Require linear history（可选，保持历史干净）

### 2. CI 发布令牌

`.github/workflows/release.yml` 在 `v*` tag 上自动测试并 `npm publish`，需要一个 npm 令牌：

1. npmjs.com → Access Tokens → Generate New Token → **Granular Access Token**
2. 权限选本包 **Read and write**；Token 类型必须选 **Automation**（可绕过 2FA，官方为 CI 设计）
3. GitHub 仓库 Settings → Secrets and variables → Actions → New repository secret
   - Name：`NPM_TOKEN`
   - Value：`npm_xxxxxxxx...`

### 3. 目录站收录

仓库 About → Topics 添加 `dsh-plugin`（目录站周期同步自动收录）；也可主动到
<https://dshmarketplace.dev/submit> 提交仓库 URL（人工审核）。

## 日常发布流程

```sh
# 1. 功能分支开发（不要直接在 main 提交）
git checkout -b feat/xxx
git push -u origin feat/xxx          # 开 PR，ci 检查绿了再合并

# 2. 合并到 main 后发布
git checkout main && git pull
npm version patch                    # bump 版本 + 自动打 vX.Y.Z tag + 提交
git push origin main --follow-tags   # 推送后 CI 自动跑测试并发布 npm

# 重大变更用 minor（npm version minor），不兼容用 major
```

## 发布回滚

```sh
# npm：版本不可变，出问题直接回滚 tag 指向旧版本
npm dist-tag add dsh-agentic-router@1.2.0 latest

# 源码：直接 revert 再走一遍发布流程
git revert <坏提交> && npm version patch && git push origin main --follow-tags
```

## 版本号约定（SemVer）

- **patch**：bug 修复、文档、测试（用户无感知）
- **minor**：新功能（如新增工具、新奖励信号）
- **major**：破坏性变更（工具改名、数据格式不兼容）
