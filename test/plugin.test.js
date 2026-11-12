// 插件端到端测试（v1.4.0 机制）：inbox 决策 + request/header 追加真实切换；
// agent/request 只读透传；llm/stream 提供 request 路径的兜底应用。
import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { apply, name, inject } from '../dsh/index.js'
import { Store } from '../dsh/store.js'

// 会话 mock：requestHeader 反映最近追加的 header（模拟真实 fold 行为）；
// deriveMessages 返回可注入的最后一条 assistant 消息（供步骤级路由 v1.5.0 读取工具调用）
function mockSession(id, startModel = 'deepseek-v4-pro', opts = {}) {
  let model = startModel
  let provider = opts.provider || 'deepseek-official'
  let hasHeader = opts.noHeader ? false : true
  const events = []
  let lastAssistant = null
  return {
    id,
    events,
    append(type, data) {
      events.push({ type, data })
      if (type === 'request/header' && data && data.header && data.header.config) {
        if (typeof data.header.config.model === 'string') model = data.header.config.model
        if (typeof data.header.config.provider === 'string') provider = data.header.config.provider
        hasHeader = true
      }
    },
    requestHeader() {
      if (!hasHeader) return null
      return { config: { provider, model }, adapterDefaults: {} }
    },
    setLastAssistant(blocks) { lastAssistant = { role: 'assistant', content: blocks } },
    deriveMessages() {
      const base = [{ role: 'user', content: [{ type: 'text', text: 'u' }] }]
      return lastAssistant ? [...base, lastAssistant] : base
    },
  }
}

function mockCtx(opts = {}) {
  const listeners = {}
  const registered = []
  const providers = opts.providers || [{ id: 'deepseek-official', models: ['deepseek-flash', 'deepseek-v4-pro'] }]
  return {
    listeners,
    registered,
    get(k) {
      if (k === 'tools') {
        return { register: (def) => { registered.push(def); return () => {} } }
      }
      if (k === 'llm') {
        return {
          async listProviders() { return providers },
          async listModels(pid) {
            if (opts.slowModels) return new Promise(() => {}) // 永不 resolve：模拟档位表加载未完成
            const p = providers.find((x) => (x.id || x.name || x.provider) === pid)
            return (p ? p.models : []).map((id) => ({ id }))
          },
        }
      }
      if (k === 'messageFeedback') return opts.mf
      if (k === 'timer') return opts.timer
      if (k === 'sessions') return opts.sessions
      if (k === 'agents') return opts.agents
      return undefined
    },
    on(ev, fn) { listeners[ev] = fn; return () => {} },
  }
}

function payload(sessionId, turn, session) {
  return { agent: { sessionId, session: session || mockSession(sessionId) }, turn, step: 1, signal: {} }
}

const CODE_TASK = '重构下面这段 Python 代码，使其支持多线程并发执行并安全处理异常：\n\n```python\nimport time\ndef fetch(url):\n    time.sleep(1)\n```'
const QA_TASK = '请只用一句话回答：地球的天然卫星叫什么？'

test('插件声明', () => {
  assert.equal(name, 'agentic-router')
  assert.ok(inject.includes('tools'))
})

test('v1.4.0 机制：inbox 决策并追加 request/header 真实切换（shadow 不动、active 切换、request 只读透传）', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-'))
  const ctx = mockCtx()
  apply(ctx, { home })
  const inbox = ctx.listeners['agent/inbox/claimed']
  const req = ctx.listeners['agent/request']
  const stop = ctx.listeners['agent/turn-stopping']
  assert.ok(inbox && req && stop, '事件监听应全部挂载')
  assert.equal(ctx.registered.length, 5, '应注册 5 个工具')

  const session = mockSession('s1')
  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })
  const headers = () => session.events.filter((e) => e.type === 'request/header')

  // shadow 模式（默认）：复杂代码任务 → 记录决策，但不追加 header
  await inbox(P(1, CODE_TASK))
  assert.equal(headers().length, 0, 'shadow 模式不得切换')

  // 切 active
  const setMode = ctx.registered.find((t) => t.name === 'agentic_router_set_mode')
  await setMode.execute({ mode: 'active' }, {})

  // 简单问答 → 追加 flash header
  await inbox(P(2, QA_TASK))
  const appends = headers()
  assert.equal(appends.length, 1, 'active 模式应追加一次 request/header')
  assert.equal(appends[0].data.header.config.model, 'deepseek-flash', '简单任务应切 flash')
  assert.equal(appends[0].data.reason, 'router')

  // agent/request 必须只读透传（不再替换配置）
  const cfg = await req(payload('s1', 2, session), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  assert.equal(cfg.model, 'deepseek-v4-pro', 'agent/request 必须透传原配置')

  // 收尾 → 奖励落盘且元数据完整
  await stop(payload('s1', 2, session))
  const store = new Store(home)
  const rewards = store.loadJsonl('rewards.jsonl')
  assert.equal(rewards.length, 1)
  assert.ok(rewards[0].reward > 0)
  assert.equal(rewards[0].turn, 's1/2')
  assert.ok(Array.isArray(rewards[0].f) && rewards[0].f.length === 9, '奖励记录需含特征向量（供 replay）')

  const decisions = store.loadJsonl('decisions.jsonl')
  assert.equal(decisions.length, 2)
  assert.equal(decisions[1].targetModel, 'deepseek-flash')
  assert.equal(decisions[1].applied, true, 'header 追加成功应标记 applied')
  assert.equal(decisions[1].swapped, true, '真实切换应标记 swapped')

  const stats = await ctx.registered.find((t) => t.name === 'agentic_router_stats').execute({}, {})
  assert.equal(stats.totalDecisions, 2)
  assert.equal(stats.mode, 'active')
  assert.equal(stats.headerAppends, 1, 'stats 应统计追加次数')
  assert.equal(stats.lastActual.model, 'deepseek-v4-pro')
})

test('冷启动：档位表异步加载未完成时，inbox 仍用兜底档位立即切换（v1.4.1 修复）', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-coldstart-'))
  const ctx = mockCtx({ slowModels: true }) // listModels 永不 resolve，模拟冷启动
  apply(ctx, { home, mode: 'active' })
  const inbox = ctx.listeners['agent/inbox/claimed']
  assert.ok(inbox)

  const session = mockSession('s1') // 当前 pro
  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })

  // inbox 是同步的：调用后立即断言，不等待任何异步档位表加载
  inbox(P(1, QA_TASK))
  const appends = session.events.filter((e) => e.type === 'request/header')
  assert.equal(appends.length, 1, '冷启动下也应立即追加 header')
  assert.equal(appends[0].data.header.config.model, 'deepseek-flash', '兜底档位表应把简单问答切到 flash')
  assert.equal(appends[0].data.reason, 'router')
})

test('v1.4.2：UCB 不探索无候选的 mid 档——简单任务决策里 recs.ucb 不得为 mid', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-notier-'))
  const ctx = mockCtx() // 档位表只有 flash/pro（fast/strong），mid 无候选
  apply(ctx, { home, mode: 'active' })
  const inbox = ctx.listeners['agent/inbox/claimed']
  const session = mockSession('s1')
  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })

  // 连发 6 个简单问答，覆盖 UCB 冷启动轮转
  for (let i = 1; i <= 6; i++) inbox(P(i, QA_TASK + '（第' + i + '问）'))

  const store = new Store(home)
  const decisions = store.loadJsonl('decisions.jsonl')
  assert.equal(decisions.length, 6)
  for (const d of decisions) {
    assert.ok(!String(d.recs.ucb).startsWith('mid'), 'UCB 不得推荐 mid，实际 ' + d.recs.ucb)
    assert.ok(!String(d.recs.lin).startsWith('mid'), 'LinUCB 不得推荐 mid，实际 ' + d.recs.lin)
  }
})

test('v1.5.0 步骤级路由：重型工具调用 → 下一步升级 pro；轻量工具 → 回到回合基线 flash', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-step-'))
  const session = mockSession('s1', 'deepseek-flash') // 当前 flash
  const sessions = new Map([['s1', session]])
  const ctx = mockCtx({ sessions })
  apply(ctx, { home, mode: 'active' })
  const inbox = ctx.listeners['agent/inbox/claimed']
  const preStep = ctx.listeners['agent/pre-step']
  const stream = ctx.listeners['llm/stream']
  assert.ok(inbox && preStep && stream)

  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })
  const PRE = (t, s) => ({ agent: { sessionId: 's1', session }, turn: t, step: s, signal: {}, messages: [] })
  const headers = () => session.events.filter((e) => e.type === 'request/header')
  const drain = async (sid) => {
    const w = stream({ sessionId: sid, provider: 'deepseek-official' }, async function* () { yield { type: 'finish', reason: 'stop' } })
    for await (const c of w) { /* drain */ }
  }

  // 回合基线：简单问答 → flash（已是 flash，不追加）
  inbox(P(1, QA_TASK))
  assert.equal(headers().length, 0, '简单回合基线 flash，无需追加')

  // 上一步是纯文本（回合收尾）→ 步骤级不动
  session.setLastAssistant([{ type: 'text', text: 'ok' }])
  preStep(PRE(1, 2), async () => ({}))
  assert.equal(headers().length, 0, '纯文本步骤保持现状')

  // 上一步发起重型工具调用（write）→ 挂起升级 pro（pre-step 不直接追加，避免被 buildRequest 覆盖）
  session.setLastAssistant([{ type: 'tool-call', id: 'c1', name: 'write', arguments: '{}' }])
  preStep(PRE(1, 3), async () => ({}))
  assert.equal(headers().length, 0, 'pre-step 只挂起，不直接追加（防竞态）')
  // 本步流结束（loop header 已落盘）→ 延迟追加 pro
  await drain('s1')
  const h1 = headers()
  assert.equal(h1.length, 1, 'llm/stream 收尾应追加 header')
  assert.equal(h1[0].data.header.config.model, 'deepseek-v4-pro', '应升级到 pro')
  assert.equal(h1[0].data.reason, 'router-step', '步骤级切换 reason 应为 router-step')

  // 上一步转轻量工具（read）→ 挂起降回基线 flash
  session.setLastAssistant([{ type: 'tool-call', id: 'c2', name: 'read', arguments: '{}' }])
  preStep(PRE(1, 4), async () => ({}))
  await drain('s1')
  const h2 = headers()
  assert.equal(h2.length, 2, '轻量工具后应追加 header 降回 flash')
  assert.equal(h2[1].data.header.config.model, 'deepseek-flash', '应降回 flash')

  // stats 可见步骤级记录
  const stats = await ctx.registered.find((t) => t.name === 'agentic_router_stats').execute({}, {})
  assert.equal(stats.stepSwitches, 2, '应记录 2 次步骤级切换')
  assert.equal(stats.stepLog.length, 2)
})

test('兜底路径：request 决策 + llm/stream 延迟应用（中途热加载场景）', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-fallback-'))
  const session = mockSession('s1')
  const sessions = new Map([['s1', session]])
  const ctx = mockCtx({ sessions })
  apply(ctx, { home, mode: 'active' })
  const pre = ctx.listeners['agent/pre-step']
  const req = ctx.listeners['agent/request']
  const stream = ctx.listeners['llm/stream']
  assert.ok(pre && req && stream)

  await pre({ ...payload('s1', 1, session), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  const cfg = await req(payload('s1', 1, session), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  assert.equal(cfg.model, 'deepseek-v4-pro', 'request 路径也透传')

  // 流式调用结束 → 补应用切换
  const wrapped = stream({ sessionId: 's1', provider: 'deepseek-official' }, async function* () {
    yield { type: 'finish', reason: 'stop' }
  })
  for await (const c of wrapped) { /* drain */ }

  const appends = session.events.filter((e) => e.type === 'request/header')
  assert.equal(appends.length, 1, 'llm/stream 后应补应用一次 header')
  assert.equal(appends[0].data.header.config.model, 'deepseek-flash')
})

test('飞轮重放：新实例从 rewards.jsonl 恢复学习状态', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-replay-'))
  const ctx1 = mockCtx()
  apply(ctx1, { home })
  const stop = ctx1.listeners['agent/turn-stopping']
  const req = ctx1.listeners['agent/request']
  const pre = ctx1.listeners['agent/pre-step']
  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  await stop(payload('s1', 1))

  // 第二个实例（模拟重启）：应重放出 1 条奖励
  const ctx2 = mockCtx()
  apply(ctx2, { home })
  const stats = await ctx2.registered.find((t) => t.name === 'agentic_router_stats').execute({}, {})
  assert.equal(stats.flywheel.ucb.total, 1, '重启后 UCB 应恢复 1 次拉动')
  assert.equal(stats.flywheel.knnPoolSize, 1)
})

test('工具失败代理：tools/result 错误 → 奖励降分', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-toolerr-'))
  const ctx = mockCtx()
  apply(ctx, { home })
  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  const toolResult = ctx.listeners['tools/result']
  assert.ok(toolResult)

  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  toolResult({ agent: { sessionId: 's1' }, name: 'read' }, { status: 'error', error: 'not found' })
  await stop(payload('s1', 1))

  const store = new Store(home)
  const rewards = store.loadJsonl('rewards.jsonl')
  assert.equal(rewards.length, 1)
  assert.equal(rewards[0].breakdown.toolErrors, 1)
  assert.equal(rewards[0].reward, 0.8)
})

test('reset 工具清空飞轮', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-reset-'))
  const ctx = mockCtx()
  apply(ctx, { home })
  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  await stop(payload('s1', 1))
  await ctx.registered.find((t) => t.name === 'agentic_router_reset').execute({}, {})
  const stats = await ctx.registered.find((t) => t.name === 'agentic_router_stats').execute({}, {})
  assert.equal(stats.flywheel.ucb.total, 0)
  assert.equal(stats.totalDecisions, 0)
})

test('messageFeedback 结算：回合结束 45s 后按人类点踩修正奖励', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-mf-'))
  const negativeItem = { messageId: 'm1', rating: 'negative', createdAt: Date.now() + 60000 }
  const mf = { async list() { return { ok: true, value: { items: [negativeItem] } } } }
  let sweepFn = null
  const timer = { interval(fn) { sweepFn = fn; return () => {} } }
  const ctx = mockCtx({ mf, timer })
  apply(ctx, { home, settleDelayMs: 0 })

  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  await stop(payload('s1', 1))

  // 结算前：奖励 1，UCB mean 1
  const store = new Store(home)
  assert.equal(store.loadJsonl('rewards.jsonl')[0].reward, 1)

  // 触发结算轮询（settleDelayMs=0 立即到期）
  assert.ok(sweepFn, '应注册结算轮询')
  await sweepFn()

  const rewards = store.loadJsonl('rewards.jsonl')
  const settle = rewards.find((r) => r.type === 'feedback-settle')
  assert.ok(settle, '应产生 feedback-settle 记录')
  assert.equal(settle.negative, 1)
  assert.equal(settle.applied, 1)
  assert.equal(settle.newReward, 0.5)

  const stats = await ctx.registered.find((t) => t.name === 'agentic_router_stats').execute({}, {})
  assert.equal(stats.flywheel.ucb.arms['qa:fast'].mean, 0.5, '飞轮均值应被修正')
  assert.equal(stats.flywheel.pendingFeedback, 0, '结算后 pending 清空')
})

test('messageFeedback 缺失时优雅降级（不报错、不结算）', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-mfnone-'))
  const ctx = mockCtx({})
  apply(ctx, { home, settleDelayMs: 0 })
  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  await stop(payload('s1', 1))
  const store = new Store(home)
  assert.ok(store.loadJsonl('rewards.jsonl').every((r) => r.type !== 'feedback-settle'))
})

test('成本信号：llm/stream 透传并累加 usage，奖励按成本扣分', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-cost-'))
  const ctx = mockCtx({})
  apply(ctx, { home })
  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  const stream = ctx.listeners['llm/stream']
  assert.ok(stream, '应挂载 llm/stream 监听')

  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))

  // 模拟一次流式调用：文本 + usage chunk（strong 档 100k 输入 100k 输出）
  const chunks = []
  const wrapped = stream({ sessionId: 's1', provider: 'deepseek-official' }, async function* () {
    yield { type: 'text-delta', text: '月球' }
    yield { type: 'usage', usage: { inputTokens: 100000, outputTokens: 100000 } }
  })
  for await (const c of wrapped) chunks.push(c)
  assert.equal(chunks.length, 2, '所有 chunk 必须透传')
  assert.equal(chunks[0].text, '月球')

  await stop(payload('s1', 1))
  const store = new Store(home)
  const rewards = store.loadJsonl('rewards.jsonl')
  // strong 档成本 = (100000×1 + 100000×12)/1e6 = 1.3 元 → 惩罚封顶 0.5
  assert.equal(rewards[0].breakdown.costYuan, 1.3)
  assert.equal(rewards[0].breakdown.costPenalty, 0.5)
  assert.equal(rewards[0].reward, 0.5)
})

test('成本信号：无 usage 的 provider 走估算兜底（estimated 标记）', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-costest-'))
  const ctx = mockCtx({})
  apply(ctx, { home })
  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  const stream = ctx.listeners['llm/stream']

  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  const wrapped = stream({ sessionId: 's1', provider: 'deepseek-official' }, async function* () {
    yield { type: 'text-delta', text: 'H2O' }
    yield { type: 'finish', reason: 'stop' }
  })
  for await (const c of wrapped) { /* 透传 */ }
  await stop(payload('s1', 1))
  const rewards = new Store(home).loadJsonl('rewards.jsonl')
  assert.ok(rewards[0].breakdown.costYuan > 0, '估算成本应 > 0')
  assert.ok(rewards[0].breakdown.costYuan < 0.01, '小回合成本应很小')
})

test('set_prices：接入新 provider 后配置价格，成本按新价计算', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-price-'))
  const ctx = mockCtx({})
  apply(ctx, { home })
  const setPrices = ctx.registered.find((t) => t.name === 'agentic_router_set_prices')
  const stats = ctx.registered.find((t) => t.name === 'agentic_router_stats')

  // 默认无 openai 精确价 → 查询显示 prices 表
  const view = await setPrices.execute({ provider: 'openai' }, {})
  assert.ok(view.prices['*'], '默认含通用档位兜底')

  // 配置 openai/gpt-x-pro 价格
  const r = await setPrices.execute({ provider: 'openai', model: 'gpt-x-pro', priceIn: 15, priceOut: 60 }, {})
  assert.equal(r.updated, 'gpt-x-pro')
  const after = await stats.execute({}, {})
  assert.equal(after.prices.openai['gpt-x-pro'].out, 60, '价格表热更新')

  // 新 provider 决策 + 流式 usage → 成本按 15/60 计
  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  const stream = ctx.listeners['llm/stream']
  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: CODE_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'openai', model: 'gpt-x-pro' }))
  const wrapped = stream({ sessionId: 's1', provider: 'openai' }, async function* () {
    yield { type: 'usage', usage: { inputTokens: 1000, outputTokens: 1000 } }
  })
  for await (const c of wrapped) { /* 透传 */ }
  await stop(payload('s1', 1))
  const rewards = new Store(home).loadJsonl('rewards.jsonl')
  // (1000×15 + 1000×60)/1e6 = 0.075 元；来源 provider-model
  assert.ok(Math.abs(rewards[0].breakdown.costYuan - 0.075) < 1e-9)
  assert.equal(rewards[0].breakdown.pricingSource, 'provider-model')
})

test('未知 provider 不套用 DeepSeek 价格：用通用档位并标记来源', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-unknown-'))
  const ctx = mockCtx({})
  apply(ctx, { home })
  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  const stream = ctx.listeners['llm/stream']
  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  await req(payload('s1', 1), async () => ({ provider: 'mystery-vendor', model: 'mystery-fast' }))
  const wrapped = stream({ sessionId: 's1', provider: 'mystery-vendor' }, async function* () {
    yield { type: 'usage', usage: { inputTokens: 1000, outputTokens: 1000 } }
  })
  for await (const c of wrapped) { /* 透传 */ }
  await stop(payload('s1', 1))
  const rewards = new Store(home).loadJsonl('rewards.jsonl')
  // '*' fast = (1000×1 + 1000×4)/1e6 = 0.005
  assert.ok(Math.abs(rewards[0].breakdown.costYuan - 0.005) < 1e-9)
  assert.equal(rewards[0].breakdown.pricingSource, 'generic-tier')
})

test('v1.6.0 force：钉死模型，步骤级路由不再覆盖', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-force-'))
  const session = mockSession('s1', 'deepseek-flash')
  const sessions = new Map([['s1', session]])
  const agents = { list() { return [{ session }] } }
  const ctx = mockCtx({ sessions, agents })
  apply(ctx, { home, mode: 'active' })
  const inbox = ctx.listeners['agent/inbox/claimed']
  const preStep = ctx.listeners['agent/pre-step']
  const stream = ctx.listeners['llm/stream']
  const forceTool = ctx.registered.find((t) => t.name === 'agentic_router_force')

  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })
  const PRE = (t, s) => ({ agent: { sessionId: 's1', session }, turn: t, step: s, signal: {}, messages: [] })
  const drain = async (sid) => { const w = stream({ sessionId: sid, provider: 'deepseek-official' }, async function* () { yield { type: 'finish', reason: 'stop' } }); for await (const c of w) {} }
  const headers = () => session.events.filter((e) => e.type === 'request/header')

  inbox(P(1, QA_TASK)) // 基线 flash，已是 flash → 不追加
  assert.equal(headers().length, 0)

  await forceTool.execute({ model: 'deepseek-v4-pro' }, {}) // force 立即追加 header
  assert.equal(headers().length, 1)
  assert.equal(headers()[0].data.header.config.model, 'deepseek-v4-pro')

  session.setLastAssistant([{ type: 'tool-call', id: 'c1', name: 'bash', arguments: '{}' }])
  preStep(PRE(1, 2), async () => ({})) // 重型工具，本应升级；force 下跳过
  await drain('s1')
  assert.equal(headers().length, 1, 'force 下步骤级路由不得覆盖')
})

test('v1.6.0 首回合延迟决策：inbox 无 header → 首个 request 拿到 provider 后解析并应用', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-firstturn-'))
  const session = mockSession('s1', 'deepseek-v4-pro', { noHeader: true })
  const sessions = new Map([['s1', session]])
  const ctx = mockCtx({ sessions })
  apply(ctx, { home, mode: 'active' })
  const inbox = ctx.listeners['agent/inbox/claimed']
  const req = ctx.listeners['agent/request']
  const stream = ctx.listeners['llm/stream']
  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })

  inbox(P(1, QA_TASK))
  assert.equal(session.events.filter((e) => e.type === 'request/header').length, 0)
  assert.equal(new Store(home).loadJsonl('decisions.jsonl').length, 0, '首回合无 provider 前不落盘')

  const cfg = await req(payload('s1', 1, session), async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' }))
  assert.equal(cfg.model, 'deepseek-v4-pro', 'request 透传')
  const decisions = new Store(home).loadJsonl('decisions.jsonl')
  assert.equal(decisions.length, 1, 'request 后应落盘一条决策')
  assert.equal(decisions[0].targetModel, 'deepseek-flash')
  assert.equal(decisions[0].resolvedTier, 'fast')

  const wrapped = stream({ sessionId: 's1', provider: 'deepseek-official' }, async function* () { yield { type: 'finish', reason: 'stop' } })
  for await (const c of wrapped) {}
  const appends = session.events.filter((e) => e.type === 'request/header')
  assert.equal(appends.length, 1)
  assert.equal(appends[0].data.header.config.model, 'deepseek-flash')
})

test('v1.6.0 跨 provider：当前 provider 无目标档候选时切到其它 provider', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-cross-'))
  const session = mockSession('s1', 'alpha-pro', { provider: 'alpha' })
  const sessions = new Map([['s1', session]])
  const providers = [
    { id: 'alpha', models: ['alpha-pro'] },
    { id: 'beta', models: ['beta-flash'] },
  ]
  const ctx = mockCtx({ sessions, providers })
  apply(ctx, { home, mode: 'active' })
  await new Promise((r) => setTimeout(r, 0)) // 等权威档位表加载
  const inbox = ctx.listeners['agent/inbox/claimed']
  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })
  inbox(P(1, QA_TASK))
  const appends = session.events.filter((e) => e.type === 'request/header')
  assert.equal(appends.length, 1)
  assert.equal(appends[0].data.header.config.model, 'beta-flash')
  assert.equal(appends[0].data.header.config.provider, 'beta', '应切换 provider')
})

test('v1.6.0 crossProvider=false：无同 provider 候选时不跨、不切换', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-nocross-'))
  const session = mockSession('s1', 'alpha-pro', { provider: 'alpha' })
  const sessions = new Map([['s1', session]])
  const providers = [
    { id: 'alpha', models: ['alpha-pro'] },
    { id: 'beta', models: ['beta-flash'] },
  ]
  const ctx = mockCtx({ sessions, providers })
  apply(ctx, { home, mode: 'active', crossProvider: false })
  await new Promise((r) => setTimeout(r, 0))
  const inbox = ctx.listeners['agent/inbox/claimed']
  const P = (t, text) => ({ agent: { sessionId: 's1', session }, turn: t, step: 1, signal: {}, message: { role: 'user', content: [{ type: 'text', text }] } })
  inbox(P(1, QA_TASK))
  assert.equal(session.events.filter((e) => e.type === 'request/header').length, 0, 'crossProvider=false 不得跨 provider')
})

test('v1.6.0 价格合并：内置 deepseek-account 价对旧 policy 生效', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-pricemerge-'))
  new Store(home).savePolicy({ mode: 'active', forced: null, prices: { 'deepseek-official': { 'deepseek-v4-flash': { in: 0.2, out: 2 } }, '*': { fast: { in: 1, out: 4 }, mid: { in: 2, out: 8 }, strong: { in: 3, out: 16 } } } })
  const ctx = mockCtx({})
  apply(ctx, { home })
  const stats = await ctx.registered.find((t) => t.name === 'agentic_router_stats').execute({}, {})
  assert.ok(stats.prices['deepseek-account'], '应合并出 deepseek-account 内置价')
  assert.equal(stats.prices['deepseek-account']['deepseek-flash'].in, 0.2)
  assert.equal(stats.prices['deepseek-official']['deepseek-v4-flash'].in, 0.2, '旧 policy 覆盖保留')
})
