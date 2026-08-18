# dsh-agentic-router

> 🌐 语言切换 / Language: **[简体中文](./README.md)** · **[English](./README.en.md)**

DeepSeek Harness（DSH）插件：**学习型智能路由（agentic router）+ 数据飞轮**。

监听每次模型请求，按任务类型与复杂度把请求路由到最合适的模型档位；四个专家并行推荐、EXP3 元选择器决定听谁的；每回合按**质量代理信号**（工具失败、模型重试、延迟）计算奖励并回传，路由策略随使用越学越准。全部决策与奖励落盘，可审计、可重放。

## 架构

```
用户输入 → 分类(规则,<1ms) → 四专家推荐 → EXP3 元选择 → agent/request 切换模型
                                        ↑                        │
                 UCB/LinUCB/k-NN/EXP3 更新 ← 质量代理奖励 ← 回合收尾(工具失败/重试/延迟)
```

| 专家 | 算法 | 状态 |
|---|---|---|
| rule | 确定性阈值表（复杂度 ≤0.3→fast，≥0.7→strong，其余 mid） | 冷启动先验，常驻兜底 |
| UCB | 簇×档多臂老虎机，未探索臂上界∞、平局轮转 | 零样本即活跃 |
| LinUCB | 特征线性打分 + 探索项，梯度更新 | 影子，50 样本毕业 |
| k-NN | 经验池最近邻（k=5）聚合 | 影子，30 经验毕业 |
| EXP3 | 元选择器，重要性加权更新 | 常驻，冷启动偏 rule（权重 2.5） |

设计细节与路线图见 [docs/DESIGN.md](./docs/DESIGN.md)。本实现源自真实运行时上的动态原型验证（8 个实测缺陷已修复并固化为回归测试）。

## 安装

```sh
dsh plugin --profile web add dsh-agentic-router
```

> `--profile` 必填。安装后重启会话（或 profile），工具 schema 才会进入 prompt 组装。默认 **shadow 模式**（只记录与学习，不切换模型），确认策略表现后再切 active。

## 工具

| 工具 | 作用 |
|---|---|
| `agentic_router_stats` | 查看决策记录、四专家推荐、EXP3 权重、飞轮状态 |
| `agentic_router_set_mode` | 切换 shadow（默认）/ active（真切换）/ off（旁路） |
| `agentic_router_force` | 强制指定模型 id（active 生效），clear=true 清除 |
| `agentic_router_reset` | 清空飞轮学习状态，保留模式 |
| `agentic_router_set_prices` | 配置/查询某 provider 某模型（或档位）的价格 |

## 数据飞轮（落盘）

目录：`${DSH_HOME:-~/.dsh}/storages/dsh-agentic-router/`（`AGENTIC_ROUTER_HOME` 可覆盖）

- `decisions.jsonl` — 每条决策：任务类型/复杂度/四专家推荐/元层选择/实际路由/是否切换
- `rewards.jsonl` — 每条奖励：明细（工具失败/重试/延迟）+ 特征向量，**重启后重放恢复学习状态**；人类反馈结算（`feedback-settle` 行）在重放时精确修正
- 人类反馈：回合结束 45s 后查询 `messageFeedback`（Web UI 的 👍/👎），点踩 −0.5、点赞 +0.1，修正已应用的奖励（UCB 均值/EXP3 权重精确回退）
- `policy.json` — 模式与强制模型（原子写）

奖励公式：干净收尾 1 分起，工具失败 −0.2/次（封顶 0.6）、模型重试 −0.2/次（封顶 0.4）、
**成本** min(0.5, 20×回合成本元)、延迟 >30s −0.1 / >90s −0.3；出错回合记 0。

**成本信号（token 计费）**：透传 `llm/stream`，累加每个模型调用的 `usage` chunk
（输入/输出/缓存读/推理 token）；无 usage 的 provider 按文本长度估算（标 `estimated`）。

**价格表三级解析**（元/百万 token）：`provider→model` 精确价 > `provider→档位` 价 >
`*` 通用档位 > 保守兜底。**未知 provider 绝不套用 DeepSeek 价格**，用通用档位并标记
`pricingSource`（审计字段）。默认含 DeepSeek 官方口径，接入其他模型用
`agentic_router_set_prices` 工具配价（持久化到 policy.json），或安装时传 `config.prices`：

```js
{ 'deepseek-official': { 'deepseek-v4-pro': { in: 1, out: 12 } },
  openai: { 'gpt-x-pro': { in: 15, out: 60 } },
  '*': { fast: { in: 1, out: 4 }, mid: { in: 2, out: 8 }, strong: { in: 3, out: 16 } } }
```

## 分类器（规则先验，<1ms）

6 类任务（code/write/summarize/research/agentic/qa）+ 复杂度 0~1：
类型先验分（code/agentic 0.3，research 0.2，write 0.15…）+ 长度 + 代码块 + 复杂词加分 − 简单词减分（"简单/复杂"同时出现不惩罚）。

## 已知边界

- 模型档位靠 id 关键词识别（flash→fast、pro→strong）；可用模型多于一档时路由才真正切换
- 反馈结算窗口 45s（可配置 `settleDelayMs`），窗口内无反馈按代理奖励结算；重启会丢失未结算的 pending 条目；晚到的反馈可能归属到其后的回合
- 反馈依赖部署的 Web UI 实际产生数据（本插件只读取，不写入）
- 毕业策略（LinUCB/k-NN）需要数十个回合样本才会进元层池

## 开发

```sh
npm test        # node --test，零依赖
```

回归用例覆盖实测缺陷：qa 祈使句、mid 档回退、UCB 冷启动轮转、agent 键隔离、奖励 JSON 合法性、飞轮重放恢复等。

## 发布

```sh
npm publish                                     # 需 npm 账号
git push origin main                            # GitHub 仓库公开后
# 目录站：给仓库加 dsh-plugin topic 自动收录，或到 dshmarketplace.dev/submit 提交
```

## License

MIT
