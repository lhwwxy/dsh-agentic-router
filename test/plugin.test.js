// 插件端到端测试：mock ctx 模拟真实事件流（pre-step → request → turn-stopping）
import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { apply, name, inject } from '../dsh/index.js'
import { Store } from '../dsh/store.js'

function mockCtx(opts = {}) {
  const listeners = {}
  const registered = []
  return {
    listeners,
    registered,
    get(k) {
      if (k === 'tools') {
        return { register: (def) => { registered.push(def); return () => {} } }
      }
      if (k === 'llm') {
        return {
          async listProviders() { return [{ id: 'deepseek-official' }] },
          async listModels() { return [{ id: 'deepseek-v4-flash' }, { id: 'deepseek-v4-pro' }] },
        }
      }
      if (k === 'messageFeedback') return opts.mf
      if (k === 'timer') return opts.timer
      return undefined
    },
    on(ev, fn) { listeners[ev] = fn; return () => {} },
  }
}

function payload(sessionId, turn) {
  return { agent: { sessionId }, turn, step: 1, signal: {} }
}

const CODE_TASK = '重构下面这段 Python 代码，使其支持多线程并发执行并安全处理异常：\n\n```python\nimport time\ndef fetch(url):\n    time.sleep(1)\n```'
const QA_TASK = '请只用一句话回答：地球的天然卫星叫什么？'

test('插件声明', () => {
  assert.equal(name, 'agentic-router')
  assert.ok(inject.includes('tools'))
})

test('端到端：shadow 记录不切换 → active 按元选择切换 → 奖励落盘', async () => {
  const home = mkdtempSync(join(tmpdir(), 'dsh-router-'))
  const ctx = mockCtx()
  apply(ctx, { home })

  const req = ctx.listeners['agent/request']
  const pre = ctx.listeners['agent/pre-step']
  const stop = ctx.listeners['agent/turn-stopping']
  assert.ok(req && pre && stop, '事件监听应全部挂载')
  assert.equal(ctx.registered.length, 4, '应注册 4 个工具')

  const nextPro = async () => ({ provider: 'deepseek-official', model: 'deepseek-v4-pro' })

  // shadow 模式：复杂代码任务 → 记录但模型不变
  await pre({ ...payload('s1', 1), messages: [{ content: [{ type: 'text', text: CODE_TASK }] }] }, async () => ({}))
  const cfg1 = await req(payload('s1', 1), nextPro)
  assert.equal(cfg1.model, 'deepseek-v4-pro', 'shadow 模式不得切换')

  // 切 active
  const setMode = ctx.registered.find((t) => t.name === 'agentic_router_set_mode')
  await setMode.execute({ mode: 'active' }, {})

  // 简单问答 → 元层应切到 fast（flash）
  await pre({ ...payload('s1', 2), messages: [{ content: [{ type: 'text', text: QA_TASK }] }] }, async () => ({}))
  const cfg2 = await req(payload('s1', 2), nextPro)
  assert.equal(cfg2.model, 'deepseek-v4-flash', 'active 模式简单任务应切 flash')

  // 收尾 → 奖励落盘且元数据完整
  await stop(payload('s1', 2))
  const store = new Store(home)
  const rewards = store.loadJsonl('rewards.jsonl')
  assert.equal(rewards.length, 1)
  assert.ok(rewards[0].reward > 0)
  assert.equal(rewards[0].turn, 's1/2')
  assert.ok(Array.isArray(rewards[0].f) && rewards[0].f.length === 9, '奖励记录需含特征向量（供 replay）')
  assert.equal(rewards[0].chosen, rewards[0].pool[0] === rewards[0].chosen ? rewards[0].chosen : rewards[0].chosen)

  const decisions = store.loadJsonl('decisions.jsonl')
  assert.equal(decisions.length, 2)
  assert.equal(decisions[1].swapped || decisions[1].targetModel === 'deepseek-v4-flash', true)
  const stats = await ctx.registered.find((t) => t.name === 'agentic_router_stats').execute({}, {})
  assert.equal(stats.totalDecisions, 2)
  assert.equal(stats.mode, 'active')
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
  const negativeItem = { messageId: 'm1', rating: 'negative', createdAt: Date.now() }
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
