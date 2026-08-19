# dsh-agentic-router

> 🌐 语言切换 / Language: **[简体中文](./README.md)** · **[English](./README.en.md)**

DeepSeek Harness（DSH）插件：**学习型智能路由（agentic router）+ 数据飞轮**。

每回合开始时（`agent/inbox/claimed`）按任务类型与复杂度把请求路由到最合适的模型档位；
四个专家并行推荐、EXP3 元选择器决定听谁的；每回合按**质量代理信号**
（工具失败、模型重试、成本、延迟）计算奖励并回传，路由策略随使用越学越准。
全部决策与奖励落盘，可审计、可重放。

**v1.4.0 起切换真实生效**：v1.3.0 及之前通过 `agent/request` 返回替换后的配置切模型，
但该返回值会被 DSH 内置的模型选择监听器强制改回，实际不生效；v1.4.0 改为在回合开始
向会话日志追加 `request/header` 事件（reason: `router`），走 DSH 官方模型选择优先级链
（本进程选择 → 会话日志最新 header → 默认模型）真实切换。

**v1.5.0 步骤级路由**：在回合级基线之上，`agent/pre-step` 按**上一步的工具调用**做
步骤级调整——重型工具（写代码/执行/子 agent 等，见 `HEAVY_TOOLS`）→ 下一步升级 strong
（pro）；轻量/纯文本 → 回到回合基线档。实现"简单步骤 flash、复杂步骤 pro"的近似形态
（步骤切换对下一步生效，一阶延迟）。步骤级决策落盘 `steps.jsonl`，`agentic_router_stats`
可见 `stepSwitches` 与 `stepLog`。

## 架构

```
用户输入 → 分类(规则,<1ms) → 四专家推荐 → EXP3 元选择 → request/header 追加(真实切换)
                                        ↑                        │
                 UCB/LinUCB/k-NN/EXP3 更新 ← 质量代理奖励 ← 回合收尾(工具失败/重试/成本/延迟)
```

| 专家 | 算法 | 状态 |
|---|---|---|
| rule | 确定性阈值表（复杂度 ≤0.3→fast，≥0.7→strong，其余 mid） | 冷启动先验，常驻兜底 |
| UCB | 簇×档多臂老虎机，未探索臂上界∞、平局轮转 | 零样本即活跃 |
| LinUCB | 特征线性打分 + 探索项，梯度更新 | 影子，50 样本毕业 |
| k-NN | 经验池最近邻（k=5）聚合 | 影子，30 经验毕业 |
| EXP3 | 元选择器，重要性加权更新 | 常驻，冷启动偏 rule（权重 2.5） |

设计细节与路线图见 [docs/DESIGN.md](./docs/DESIGN.md)。

## 安装

```sh
dsh plugin --profile web add dsh-agentic-router
```

> `--profile` 必填。安装后重启 Web，工具 schema 才会进入 prompt 组装。

### 切换到真实切换（active）

默认是 **shadow 模式**（只记录与学习，不切换模型）。切到真实切换，最省事的方式：重启后在任意 DSH 会话里对 agent 说：

> 把 agentic router 切成 active 模式

agent 会调用 `agentic_router_set_mode`（mode=active），并把模式持久化到
`~/.dsh/storages/dsh-agentic-router/policy.json`——**重启后依然 active**，无需每次设置。

也可以不走会话、首次启动即 active：在 profile 的 `cordis.patch.yml` 里给
`agentic-router` 条目加配置：

```yaml
- id: agentic-router
  config:
    mode: active
```

（两种方式选其一；一旦用工具存过模式，`policy.json` 的取值优先于 `config.mode`。）

## 切换机制（v1.4.0）

DSH 的 agent 每步请求最终由模型选择逻辑决定，读取优先级为：

```
本进程选择 → 会话日志最新 request/header → 默认模型
```

- 直接改 `agent/request` waterfall 的返回值：会被 `dsh-agent` 自带监听器强制改回「选定模型」——无效（v1.3.0 及之前的缺陷）
- 改 `llm/stream`：agent-loop 请求到达时已深度冻结，无法改写

v1.4.0 的正确做法：在 `agent/inbox/claimed`（回合开始、prompt 组装之前）向会话日志追加
`request/header` 事件，下一轮组装自然读到新模型。`agent/request` 只做只读观测
（记录每步真实出网模型）；插件中途热加载时由 `llm/stream` 在循环自身 header 落盘后补一次应用。
切到 fast/mid 档会去掉 `reasoningEffort`，让目标模型走自身默认 effort。

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

**开发、测试与发布流程（维护者）见 [docs/MAINTAINING.md](./docs/MAINTAINING.md)。**

## License

MIT
