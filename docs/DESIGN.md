# DSH Agentic Router 插件设计方案

> 目标：在 DeepSeek Harness 里做一个**学习型智能路由**插件——监听用户输入，按任务类型与复杂度把请求路由到最合适的模型 / 子 agent / 工具组合；每次路由的**决策与结果全部落盘**，形成数据飞轮，让路由判断随使用越来越准、越来越快。

> ⚠️ **v1.4.0 机制修正**：下文 1.1 的 `agent/request` 返回替换配置方案已废弃——实测该返回值会被 `dsh-agent` 内置的模型选择监听器强制改回「选定模型」，不生效。v1.4.0 改为在 `agent/inbox/claimed` 向会话日志追加 `request/header`（reason: `router`），走官方模型选择优先级链真实切换；`agent/request` 降级为只读观测，`llm/stream` 提供中途热加载的兜底应用。实现见 `dsh/index.js` 顶部注释与 README「切换机制」一节。

## 0. 一句话架构

```
用户输入 → 特征提取(快) → 路由器决策 → request/header 追加(真实切换) / 派发子 agent
                ↑                                    ↓
        策略(规则表+学习权重) ← 评分与在线学习 ← 结果回传(质量/成本/延迟落盘)
```

三个关键设计原则（对齐 DSH 架构文档）：

1. **只走官方扩展点**：路由动作通过 `agent/request` / `agent/pre-step` waterfall 实现；决策记录通过 SessionEventMap 扩展进入会话日志（"model-visible means logged"）。
2. **决策可解释、可撤销**：每次路由都写一条带理由的日志事件；提供 `router:force` 人工覆盖命令与开关。
3. **先影子后接管**：上线先跑 shadow mode（只记录不干预），数据够之后再接管。

## 1. 挂载点（引用自官方文档的确切签名）

### 1.1 模型级路由：`agent/request`（waterfall）

```ts
'agent/request'(this: Scoped<Agent>, payload: {
  agent: Agent; turn: number; step: number; signal: AbortSignal
}, next: () => Promise<LlmCallConfig>): Promise<LlmCallConfig>
```

> 官方语义：`await next()` 得到"机器本来会用的冻结调用配置"（首请求为 agent 选项，之后为已记录 header），**返回一个替代配置即可切换模型**；此 waterfall 不能改消息，模型可见内容必须走已记录通道。

这是**模型路由的官方入口**：路由器在这里拿到默认 provider/model，然后返回替换后的 `LlmCallConfig`（换 provider、换 model、换 reasoning 档位、改 maxTokens）。

### 1.2 输入监听与消息改写：`agent/pre-step`（waterfall）

```ts
'agent/pre-step'(this: Scoped<Agent>, payload: {
  agent: Agent; messages: UserMessage[]; turn: number; step: number; signal: AbortSignal
}, next: () => Promise<PreStepDecision>): Promise<PreStepDecision>
```

> 官方语义：拒绝一个提议 step 或**替换进入它的消息**；`next()` 保留当前消息。它是 request 派生前唯一的串行监听链。

在这里：观察/缓存本轮输入特征、把"路由上下文"注入消息（可选）、极端情况下直接拒掉不适合本 agent 的请求。

### 1.3 补充事件

| 事件 | 用途 |
|---|---|
| `agent/inbox/claimed`（emit） | 感知新输入被认领，触发特征预计算 |
| `agent/status` / `step/end` | 拿耗时、token 用量等结果信号 |
| `subagent/start` / `subagent/end` | 子 agent 级路由的观测点 |
| `turn/end`（durable） | 每个 turn 结束时做一次**结果评分与落盘** |

## 2. 路由决策链路（每次 turn）

```text
turn/start
  │ 1. 特征提取（<5ms，纯启发式）:
  │    - 任务类型: 代码/写作/问答/检索/agentic(工具编排)/翻译/数学…
  │    - 复杂度: 输入长度、是否含代码块、是否要求多步推理、历史轮次、
  │      关键词(debug/优化/重构/分析/总结…) → 0~1 分数
  │    - 上下文: agent preset、cwd、挂载工具集、会话历史长度
  │    - 约束: 延迟预算、成本预算、可用模型清单(来自部署配置或
  │      dsh-model-deploy 的容量数据)
  ▼
  │ 2. 路由决策（路由表 + 学习权重）
  │    策略候选:
  │      A 规则表: 任务类型×复杂度 → 候选路由集合(带成本/延迟先验)
  │      B 经验检索(k-NN): 当前任务 embedding → 历史相似任务中
  │        结果最好的路由(EvoRoute 思路)
  │      C 置信度低时: 小模型/LLM-judge 裁决一次(带缓存)
  │    输出: { target: model|agent|tools, reason, confidence }
  ▼
  │ 3. 执行路由
  │    - 模型级: agent/request 返回替换的 LlmCallConfig
  │    - 子agent级: 经 subagent provider 派发到对应 preset 的 agent
  │    - 工具级: 按任务类型增删本 turn 的工具 schema(agent-scoped)
  ▼
  │ 4. 记录(durable)
  │    routing/decision 事件: { taskType, complexity, features, chosen,
  │        alternatives, reason, confidence, mode: shadow|active }
  ▼
turn/end
  │ 5. 结果评分与回传(数据飞轮)
  │    routing/outcome 事件: { decisionId, success, quality, latencyMs,
  │        costTokens, retryCount, userEdited }
  └──→ 写入本地 flywheel 存储 → 触发在线学习
```

## 3. 分类器设计（复杂度 + 任务类型）

**第 1 层（默认，零延迟）**：确定性启发式 + 关键词/模式表。覆盖 90% 场景，<5ms。

- 任务类型：代码（文件/报错/函数/重构）、创作、问答、总结、翻译、agentic（"帮我做/查一下/跑一下"）、数据分析
- 复杂度信号：输入 token 数、代码块占比、工具调用历史、任务动词强度、是否需要多文件、会话历史长度
- 输出 0~1 复杂度分 + 类型标签（多标签）

**第 2 层（低置信度时，可关闭）**：小模型分类器（如 Flash 级）一次 prompt 分类，带结果缓存与超时（>300ms 放弃，回退第 1 层）。

**第 3 层（学习型）**：基于飞轮数据定期训练/更新——简单做法是"embedding + k-NN 经验检索"（在线可做），进阶做法是离线训练一个小型路由打分器（周级更新）。

## 4. 路由目标三层模型

| 层级 | 动作 | 挂载点 | 适用 |
|---|---|---|---|
| L1 模型路由 | 替换 provider/model/reasoning | `agent/request` | 同能力不同成本/质量档位（Flash vs Pro） |
| L2 子 agent 路由 | 派发给不同 preset 的 agent | subagent provider + `agent/pre-step` | 研究型 vs 编码型 vs 检索型任务 |
| L3 工具/能力路由 | 增删本 turn 工具 schema | tools 注册（agent-scoped） | 按任务裁剪工具面，降 prompt 成本、提速 |

**默认路由表（冷启动先验，随飞轮学习调整）**：

| 任务类型 | 复杂度低 | 复杂度高 |
|---|---|---|
| 闲聊/简单问答 | Flash 模型 | Pro 模型 |
| 创作/翻译 | Flash | Pro |
| 代码/调试 | Flash + 代码工具 | Pro + 代码 agent 子 agent |
| agentic 编排 | 本 agent 全工具 | Pro + workflow |
| 研究/检索 | 检索子 agent | 研究子 agent |

每个候选带 `成本(¥/1M tokens)、延迟先验(ms)、能力标签`，由配置声明，飞轮更新其**结果质量后验**。

## 5. 数据飞轮（核心闭环）

### 5.1 记录什么（durable，全部追加）

```jsonc
// routing/decision（turn 开始时）
{ "taskType": ["code"], "complexity": 0.72, "features": {...},
  "chosen": { "kind": "model", "provider": "…", "model": "…" },
  "alternatives": [...], "reason": "rule#code-high + exp-kNN@0.81",
  "confidence": 0.81, "mode": "shadow" }

// routing/outcome（turn 结束时）
{ "decisionId": "...", "success": true, "quality": 0.85,
  "latencyMs": 8420, "costTokens": { "in": 2400, "out": 810 },
  "retryCount": 0, "userEdited": false }
```

### 5.2 评分信号（怎么知道"路由好不好"）

- **成功**：turn 正常结束、无 tool error、无重试
- **质量代理**：用户是否编辑/重发消息、是否追加追问（正向）、是否中途打断（负向）
- **效率**：端到端延迟、token 成本
- 综合分：`score = α·quality + β·(1 - latency/预算) + γ·(1 - cost/预算)`，权重可配

### 5.3 学习机制（从易到难，方案支持三级）

1. **在线 Bandit**（上线即用）：每个"任务簇 × 候选路由"维护 UCB/ε-greedy 的收益估计；探索率随样本数衰减。简单、可解释、即时见效。
2. **经验检索 k-NN**（EvoRoute 思路）：把 (任务特征, 路由, 结果) 存为向量库，新任务检索最相似的历史经验，选历史收益最高的路由。冷启动与 1 并用。
3. **离线重训**（周级，可选）：用累积数据训练小型打分器/重排序器，替换第 2 层分类器。

### 5.4 飞轮存储

- 放在 DSH 数据目录下的插件专属存储（`~/.dsh/storages/dsh-agentic-router/`：`decisions.jsonl` 追加日志 + `policy.json` 可写回的策略权重 + 可选向量索引文件）
- 全部为本地明文 JSON/JSONL——用户可导出、可审计、可迁移

### 5.5 影子模式（安全启动的关键）

`mode: shadow`：决策照算、事件照记，但 `agent/request` 里**返回 `await next()` 原样配置**。跑 N 天/请求数后对比"shadow 路由 vs 实际默认路由"的评分分布，达标后一键切 `active`。

## 6. 存储与事件扩展

- 新增 durable 事件：`routing/decision`、`routing/outcome`（扩展 SessionEventMap，走会话日志，UI 可从日志渲染路由卡片）
- 模型可见的注入：`agent.inject()` 注入一行"本次已由 router 路由至 X（理由：…）"，模型可知情
- 策略文件读写经插件自己的存储工具（幂等、原子写、带版本号，支持回滚）

## 7. 工具与 UI

**模型工具**（注册到 `ctx.tools`）：
- `router_explain(turn?)` — 解释最近一次/指定 turn 的路由决策与依据
- `router_stats` — 各任务簇 × 路由的收益统计（飞轮数据摘要）
- `router_set_mode(mode, scope)` — shadow/active/off 切换

**人工命令**（`ctx.commands`）：
- `/router:force <模型|agent>` — 本次会话强制路由
- `/router:report` — 导出飞轮报告

**Client 侧（可选二期）**：路由卡片（复用 durable 事件渲染）+ 策略配置面板。

## 8. 包结构（对齐 dsh-model-deploy 的打包格式）

```
dsh-agentic-router/
├── package.json            # dsh.bundle.patch → cordis.patch.yml
├── cordis.patch.yml        # insert row: dsh-agentic-router
├── dsh/
│   ├── index.js            # apply(): 注册事件监听 + 工具 + 命令
│   ├── classify.js         # 第1/2层分类器（纯函数）
│   ├── policy.js           # 路由表 + bandit/kNN 策略 + 影子模式
│   ├── store.js            # 飞轮存储（jsonl 追加 + 原子写策略文件）
│   └── events.js           # routing/decision、routing/outcome 定义
├── test/                   # node --test：分类器/策略/存储/端到端 mock
├── README.md / README.en.md
└── LICENSE
```

`inject: ['tools', 'agents', 'commands']`；事件监听全部经 `ctx.on('agent/request', ...)` 注册（随插件卸载自动回收）。

## 9. 分阶段路线图

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 影子模式 | 特征+分类+决策+落盘，不改请求 | 日志出现 decision/outcome 事件；分类延迟 <5ms |
| P1 规则接管 | agent/request 真正切换模型；force 命令 | 简单问答走 Flash、复杂代码走 Pro，可解释 |
| P2 数据飞轮 | outcome 评分 + 在线 bandit + 统计 | 两周内成本/延迟下降且质量无回退（影子期数据可回放验证） |
| P3 子 agent 路由 | subagent 派发 + 工具面裁剪 | 研究型任务自动进检索子 agent |
| P4 学习升级 | k-NN 经验检索 + 策略热更新 UI | 新任务类型零配置自动学习出合理路由 |

## 10. 风险与护栏

- **误路由成本上限**：单会话/turn 成本预算，超限自动降级回默认模型
- **分类器自身延迟**：>5ms 直接走默认；LLM 分类器带 300ms 超时
- **可回退**：`/router:force default` 或全局 off；策略文件保留版本，支持一键回滚
- **隐私**：飞轮数据全部本地落盘，默认不出机
- **合规**：影子模式与 active 模式都写日志，任何路由可事后审计

## 11. 与 dsh-model-deploy（选型分析器）的联动

选型分析器提供**容量/成本先验**：部署里有哪些模型与硬件、各自 TTFT/吞吐/成本——这些直接作为路由表的候选集与成本/延迟先验；反过来，路由飞轮积累的**真实负载下的质量/延迟数据**可以回灌给选型分析器校准其估算系数。两者合起来就是"**规划（选型）→ 执行（路由）→ 反馈（飞轮）**"的完整闭环。

## 12. 验证方式

- 单元测试：分类器、策略、存储（node --test，零依赖）
- 离线回放：拿历史 session 日志重放，对比 shadow 路由与实际路由的评分分布
- 影子期 A/B：shadow 评分 vs 默认路由评分，达标才切 active
- 集成冒烟：mock ctx 注入 agent/request payload，验证 LlmCallConfig 替换与 next() 透传

---

## 13. v1.5.0 步骤级路由设计（已实现）

### 13.1 动机

回合级路由（v1.4.x）每回合决策一次、整回合同一模型。用户期望"单任务内简单步骤用 flash、复杂步骤用 pro"。步骤级路由在回合级基线之上做**按步骤升降档**。

### 13.2 时序依据（为什么对下一步生效）

loop 每步顺序：`inbox.claim()`（发 `agent/inbox/claimed`）→ `systemPrompt.assemble()`（快照模型）→ `agent/pre-step` waterfall → `buildRequest`。因此：

- **inbox 决策**在组装之前 → 回合第一步即生效（零延迟，v1.4.x 主路径）
- **pre-step 决策**在组装之后 → 追加 header 对**下一步**生效（一阶延迟，设计如此）

### 13.3 信号：上一步的工具调用

`session.deriveMessages()` 取最后一条 assistant 消息的 `tool-call` 块（只读叶子字段）：

| 上一步工具 | 下一步档位 |
|---|---|
| 重型：bash/pwsh/write/edit/batch_edit/run_code/workflow/subagent/subagent_fork/terminal/pty/mcp | **strong（pro）** |
| 轻量：read/glob/grep 等 | 回合基线档 |
| 无工具（纯文本，通常回合收尾） | 保持现状 |

### 13.4 决策与落盘

- 步骤档位 → `resolveTier` → 目标模型 ≠ 当前模型 → 追加 `request/header`（reason: `router-step`）
- 每步一条 `steps.jsonl`：`{turn, step, lastTools, stepTier, target, current, applied}`
- stats 暴露 `stepSwitches` 与 `stepLog`（最近 20 条）

### 13.5 边界与后续

- 奖励飞轮仍按回合结算（回合基线的策略选择），步骤级是确定性规则；未来可把步骤级升降档纳入学习（如按工具调用频率学阈值）
- 重型集合 `HEAVY_TOOLS` 是静态清单，可配置化
- 降级方向：重型步骤后的轻量步骤会回到基线（简单任务场景：写代码阶段 pro、前后问答 flash）

### 13.6 v1.6.0：跨 provider 解析、首回合延迟与 force 钉死

- `resolveTier` 解析顺序：同 provider 同档 → 其它 provider 同档（`crossProvider`，默认开）→ mid 回退（同 provider strong/fast → 跨 provider strong/fast）；`appendRouterHeader` 同步写 `config.provider` 与 `config.model`
- 首回合（`currentProvider` 为空）在 inbox 只挂起，等首个 `agent/request` 拿到真实 provider 后解析并落盘，再于 `llm/stream` 延迟应用
- `agentic_router_force` 期间跳过步骤级升降档（force 钉死模型）；force 支持 `model` 或 `provider/model`
