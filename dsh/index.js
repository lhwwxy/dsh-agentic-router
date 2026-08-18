// dsh-agentic-router —— DeepSeek Harness 插件（Host 侧）
//
// 学习型智能路由：监听 agent/pre-step 与 agent/request，按任务类型与复杂度在
// 模型档位（fast/mid/strong）之间路由；四个专家（规则表 / UCB / LinUCB / k-NN）
// 并行推荐，EXP3 元选择器决定听谁的；回合结束时按质量代理信号
// （工具失败 + 模型重试 + 延迟）计算奖励并回传飞轮。
//
// 架构依据 docs/architecture.md：
//   - "Add a model-facing capability"    → ctx.tools 注册 4 个管理工具
//   - "Intercept a request"              → agent/request waterfall 替换 LlmCallConfig
//   - 所有监听经 ctx.on 注册，随插件卸载自动回收
// 飞轮数据落盘 ${DSH_HOME:-~/.dsh}/storages/dsh-agentic-router/，重启后重放恢复。

import { classify, featuresOf } from './classify.js'
import {
  tierOf, tierTarget, resolveTier,
  recommendRule, recommendUcb, recommendLin, recommendKnn,
  rewardOf, applyReward, correctReward,
  DEFAULT_PRICES, turnCostYuan,
} from './strategies.js'
import { metaPick, DEFAULT_WEIGHTS } from './meta.js'
import { Store, defaultHome } from './store.js'

export const name = 'agentic-router'
export const inject = ['tools']

const LIN_GRADUATION = 50   // LinUCB 毕业样本线
const KNN_GRADUATION = 30   // k-NN 毕业经验线
const SETTLE_DELAY_MS = 45000 // 反馈结算窗口：回合结束 45s 后查询 messageFeedback
const SWEEP_MS = 15000        // 结算轮询周期

function freshState() {
  return {
    mode: 'shadow',
    seq: 0,
    decisions: [],
    forced: null,
    pending: new Map(),
    modelTiers: null,
    turnDecision: {},
    openTurns: new Map(),
    pendingFeedback: [],
    recByTurn: new Map(),
    ucb: { total: 0, arms: {} },
    lin: {
      total: 0,
      tierN: { fast: 0, mid: 0, strong: 0 },
      tierW: { fast: new Array(9).fill(0), mid: new Array(9).fill(0), strong: new Array(9).fill(0) },
    },
    knn: { pool: [] },
    exp3: { total: 0, weights: { ...DEFAULT_WEIGHTS } },
  }
}

function now() {
  try { return Date.now() } catch { return null }
}

function agentKey(payload) {
  try {
    const a = payload && payload.agent
    if (a) {
      if (typeof a.sessionId === 'string') return a.sessionId
      if (typeof a.id === 'string') return a.id
      const s = a.session
      if (s && typeof s.id === 'string') return s.id
    }
  } catch {}
  return 'agent-' + (payload && payload.turn)
}

function fullKey(payload) { return agentKey(payload) + '/' + payload.turn }

function textOfMessages(messages) {
  const parts = []
  if (!Array.isArray(messages)) return ''
  for (const m of messages) {
    if (!m || typeof m !== 'object') continue
    const c = m.content
    if (typeof c === 'string') parts.push(c)
    else if (Array.isArray(c)) {
      for (const b of c) {
        if (b && typeof b === 'object' && b.type === 'text' && typeof b.text === 'string') parts.push(b.text)
      }
    }
  }
  return parts.join('\n').trim()
}

function isErrorResult(r) {
  if (r == null || typeof r !== 'object') return false
  if (r.ok === false || r.success === false) return true
  if (r.status === 'error' || r.status === 'failed' || r.status === 'ERROR') return true
  if (typeof r.error === 'object' && r.error !== null) return true
  if (typeof r.error === 'string' && r.error !== '') return true
  return false
}

function execAgentKey(exec) {
  try {
    if (!exec) return null
    const a = exec.agent
    if (a) {
      if (typeof a.sessionId === 'string') return a.sessionId
      if (typeof a.id === 'string') return a.id
    }
    if (typeof exec.sessionId === 'string') return exec.sessionId
    if (typeof exec.agentId === 'string') return exec.agentId
  } catch {}
  return null
}

async function refreshModelTiers(llm) {
  if (!llm || typeof llm.listProviders !== 'function') return null
  const providers = await llm.listProviders()
  const out = { providers: [], tiers: { fast: [], mid: [], strong: [] } }
  for (const p of providers) {
    const pid = p && (p.id || p.name || p.provider)
    if (!pid) continue
    let models = []
    try { models = await llm.listModels(pid) } catch { models = [] }
    const names = (models || []).map((m) => (m && (m.id || m.model || m.name)) || '').filter(Boolean)
    out.providers.push({ provider: pid, models: names })
    for (const n of names) out.tiers[tierOf(n)].push(pid + '/' + n)
  }
  return out
}

export function apply(ctx, config = {}) {
  const tools = ctx.get('tools')
  const llm = ctx.get('llm')
  const mf = ctx.get('messageFeedback')
  const timer = ctx.get('timer')
  const store = new Store(config.home || defaultHome())
  const settleDelayMs = typeof config.settleDelayMs === 'number' ? config.settleDelayMs : SETTLE_DELAY_MS
  const prices = config.prices || DEFAULT_PRICES
  const costWeight = typeof config.costWeight === 'number' ? config.costWeight : 20
  const costCap = typeof config.costCap === 'number' ? config.costCap : 0.5
  const policy = store.loadPolicy()

  const state = freshState()
  state.mode = policy.mode || config.mode || 'shadow'
  if (typeof policy.forced === 'string') state.forced = policy.forced

  // 飞轮恢复：重放 rewards.jsonl 重建 UCB/LinUCB/k-NN/EXP3；
  // feedback-settle 行在基础奖励重放后按 turn 精确修正
  const rewards = store.loadJsonl('rewards.jsonl')
  const byTurn = new Map()
  for (const rec of rewards) {
    if (rec.type === 'feedback-settle') continue
    if (typeof rec.reward === 'number' && Array.isArray(rec.f)) {
      applyReward(state, rec)
      byTurn.set(rec.turn, rec)
    }
  }
  for (const rec of rewards) {
    if (rec.type !== 'feedback-settle') continue
    const baseRec = byTurn.get(rec.turn)
    if (baseRec) correctReward(state, baseRec, rec.newReward)
  }
  state.recByTurn = byTurn
  state.decisions = store.loadJsonl('decisions.jsonl').slice(-60)
  state.seq = state.decisions.length ? state.decisions[state.decisions.length - 1].seq : 0

  const log = (msg) => console.log('[agentic-router]', msg)
  if (rewards.length) log('replayed ' + rewards.length + ' rewards; ucb.total=' + state.ucb.total)

  let mfWarned = false
  // 反馈结算：对已过结算窗口的回合查询 messageFeedback，按人类 👍/👎 修正奖励
  async function settleFeedback() {
    if (!mf || typeof mf.list !== 'function') {
      if (!mfWarned) { log('messageFeedback service absent — 反馈奖励停用'); mfWarned = true }
      return
    }
    const tNow = now()
    if (tNow == null) return
    const remaining = []
    for (const p of state.pendingFeedback) {
      if (tNow - p.t0 < settleDelayMs) { remaining.push(p); continue }
      try {
        const res = await mf.list({ sessionId: p.sessionId })
        let neg = 0
        let pos = 0
        if (res && res.ok === true && res.value && Array.isArray(res.value.items)) {
          for (const it of res.value.items) {
            if (typeof it.createdAt !== 'number' || it.createdAt < p.t0) continue
            if (it.rating === 'negative') neg += 1
            else if (it.rating === 'positive') pos += 1
          }
        }
        const newReward = Math.max(0, Math.min(1, Math.round((p.appliedReward - 0.5 * neg + 0.1 * pos) * 100) / 100))
        if (newReward !== p.appliedReward) {
          const baseRec = state.recByTurn.get(p.key)
          if (baseRec) correctReward(state, baseRec, newReward)
          store.appendJsonl('rewards.jsonl', {
            type: 'feedback-settle', turn: p.key, sessionId: p.sessionId,
            negative: neg, positive: pos, applied: p.appliedReward, newReward,
          })
          log('feedback settle ' + p.key + ': ' + p.appliedReward + ' -> ' + newReward + ' (neg=' + neg + ' pos=' + pos + ')')
        }
      } catch (e) {
        log('feedback settle failed: ' + String(e))
      }
    }
    state.pendingFeedback = remaining
  }

  if (timer && typeof timer.interval === 'function') {
    timer.interval(() => { settleFeedback().catch((e) => log('settle sweep failed: ' + String(e))) }, SWEEP_MS)
  }

  // ---- 事件挂载 ----
  ctx.on('agent/pre-step', (payload, next) => {
    try {
      const text = textOfMessages(payload && payload.messages)
      if (text) state.pending.set(agentKey(payload) + ':' + payload.turn, text)
    } catch (e) {
      log('pre-step capture failed: ' + String(e))
    }
    return next()
  })

  ctx.on('agent/request', async (payload, next) => {
    const cfg = await next()
    if (!cfg || typeof cfg !== 'object') return cfg
    try {
      if (!timer || typeof timer.interval !== 'function') settleFeedback().catch(() => {})
      const key = fullKey(payload)
      if (state.turnDecision[key]) return cfg
      const pkey = agentKey(payload) + ':' + payload.turn
      const text = state.pending.get(pkey) || ''
      state.pending.delete(pkey)
      const cls = classify(text)
      const f = featuresOf(cls, text)
      const current = typeof cfg.model === 'string' ? cfg.model : '(unknown)'
      const provider = typeof cfg.provider === 'string' ? cfg.provider : ''
      const recs = {
        rule: recommendRule(cls),
        ucb: recommendUcb(cls, state.ucb),
        lin: recommendLin(cls, f, state.lin),
        knn: recommendKnn(cls, f, state.knn),
      }
      if (state.forced) recs.rule = { tier: 'forced', reason: 'forced', conf: 1 }
      const exp3 = {
        ...state.exp3,
        linGraduated: state.lin.total >= LIN_GRADUATION,
        knnGraduated: state.knn.pool.length >= KNN_GRADUATION,
      }
      const meta = metaPick(recs, exp3)
      const chosenRec = meta.byName[meta.chosen] || recs.rule
      const resolved = state.forced
        ? { target: state.forced, used: 'forced', note: 'forced' }
        : resolveTier(chosenRec.tier, provider, state.modelTiers)
      const target = resolved.target
      state.seq += 1
      const record = {
        seq: state.seq,
        turn: key,
        mode: state.mode,
        taskType: cls.type,
        complexity: cls.complexity,
        inputPreview: text.slice(0, 100),
        currentModel: current,
        recs: {
          rule: recs.rule.tier + '(' + recs.rule.conf + ')',
          ucb: recs.ucb.tier + '(' + recs.ucb.conf + ')',
          lin: recs.lin.tier + '(' + recs.lin.conf + ')',
          knn: recs.knn.tier + '(' + recs.knn.conf + ')',
        },
        metaChosen: meta.chosen,
        metaProbs: meta.ps,
        metaPool: meta.pool,
        resolvedTier: resolved.used + (resolved.note ? ' [' + resolved.note + ']' : ''),
        targetModel: state.mode === 'active' ? target : null,
        swapped: false,
      }
      state.turnDecision[key] = {
        clsType: cls.type, f, recs, meta, chosen: meta.chosen,
        finalized: false, t0: now(), toolErrors: 0, retries: 0,
        currentModel: current, costWeight, costCap,
        estInputTokens: Math.ceil(text.length / 1.5),
      }
      state.openTurns.set(agentKey(payload), key)
      store.appendJsonl('decisions.jsonl', record)
      if (state.mode === 'active' && target && target !== current) {
        record.swapped = true
        state.decisions.push(record)
        if (state.decisions.length > 200) state.decisions.shift()
        log('SWAP -> ' + target + ' | ' + meta.chosen + ' | ' + chosenRec.reason + ' | ' + resolved.note)
        return { ...cfg, model: target }
      }
      state.decisions.push(record)
      if (state.decisions.length > 200) state.decisions.shift()
      log('decision #' + state.seq + ' type=' + cls.type + ' c=' + cls.complexity +
        ' meta=' + meta.chosen + ' tier=' + resolved.used + ' recs=' + JSON.stringify(record.recs))
    } catch (e) {
      log('decide failed: ' + String(e))
    }
    return cfg
  })

  ctx.on('agent/request-error', (payload, next) => {
    try {
      const td = state.turnDecision[fullKey(payload)]
      if (td && !td.finalized) {
        td.retries = (td.retries || 0) + 1
        log('retry on ' + fullKey(payload) + ' count=' + td.retries)
      }
    } catch (e) {
      log('request-error hook failed: ' + String(e))
    }
    return next()
  })

  // 成本信号：透传每次流式模型调用，累加 usage chunk 到当前 open turn；
  // 无 usage 的 provider 用文本长度估算（estimated=true）
  ctx.on('llm/stream', function (options, next) {
    return (async function* () {
      let usage = null
      let outTextLen = 0
      try {
        for await (const chunk of next()) {
          if (chunk && chunk.type === 'usage' && chunk.usage) usage = chunk.usage
          if (chunk && chunk.type === 'text-delta' && typeof chunk.text === 'string') outTextLen += chunk.text.length
          yield chunk
        }
      } finally {
        try {
          const sid = options && (options.sessionId || (options.session && options.session.id))
          const key = sid ? state.openTurns.get(sid) : null
          const td = key ? state.turnDecision[key] : null
          if (td && !td.finalized) {
            const u = td.usage || { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, reasoningTokens: 0, estimated: false }
            if (usage && typeof usage.inputTokens === 'number') {
              u.inputTokens += usage.inputTokens
              u.outputTokens += usage.outputTokens || 0
              u.cacheReadTokens += usage.cacheReadTokens || 0
              u.reasoningTokens += usage.reasoningTokens || 0
              u.estimated = false
            } else {
              u.inputTokens += td.estInputTokens || 0
              u.outputTokens += Math.round(outTextLen / 2)
              u.estimated = true
            }
            td.usage = u
            const tier = tierOf(typeof td.currentModel === 'string' ? td.currentModel : '')
            td.costYuan = turnCostYuan(u, tier, prices)
            log('usage on ' + key + ': in=' + u.inputTokens + ' out=' + u.outputTokens + (u.estimated ? ' (估)' : '') + ' cost=¥' + td.costYuan.toFixed(4))
          }
        } catch (e) {
          log('usage hook failed: ' + String(e))
        }
      }
    })()
  })

  ctx.on('tools/result', (exec, result) => {
    try {
      if (!isErrorResult(result)) return
      const ak = execAgentKey(exec)
      if (!ak) return
      const key = state.openTurns.get(ak)
      if (!key) return
      const td = state.turnDecision[key]
      if (td && !td.finalized) {
        td.toolErrors = (td.toolErrors || 0) + 1
        log('tool error on ' + key + ' count=' + td.toolErrors)
      }
    } catch (e) {
      log('tools/result hook failed: ' + String(e))
    }
  })

  const finalize = (payload, tag) => {
    try {
      const key = fullKey(payload)
      const td = state.turnDecision[key]
      if (!td || td.finalized) return
      td.finalized = true
      const t1 = now()
      const dtMs = t1 != null && td.t0 != null ? t1 - td.t0 : null
      const r = rewardOf(td, tag, dtMs)
      const rec = {
        turn: key,
        reward: r.reward,
        tag,
        breakdown: r.breakdown,
        chosen: td.chosen,
        pool: td.meta ? td.meta.pool : [],
        pChosen: td.meta ? td.meta.ps[td.chosen] || 0 : 0,
        clsType: td.clsType,
        ucbTier: td.recs.ucb.tier,
        linTier: td.recs.lin.tier,
        f: td.f,
      }
      applyReward(state, rec)
      store.appendJsonl('rewards.jsonl', rec)
      state.recByTurn.set(key, rec)
      if (state.recByTurn.size > 500) {
        const first = state.recByTurn.keys().next().value
        if (first !== undefined) state.recByTurn.delete(first)
      }
      if (mf && typeof mf.list === 'function' && td.t0 != null) {
        state.pendingFeedback.push({ key, sessionId: agentKey(payload), t0: td.t0, appliedReward: rec.reward })
        if (state.pendingFeedback.length > 200) state.pendingFeedback.shift()
      }
      const ak = agentKey(payload)
      if (state.openTurns.get(ak) === key) state.openTurns.delete(ak)
      log('reward ' + rec.reward + ' ' + tag + ' on ' + key + ' breakdown=' + JSON.stringify(r.breakdown))
    } catch (e) {
      log('finalize failed: ' + String(e))
    }
  }

  ctx.on('agent/turn-stopping', (payload) => finalize(payload, 'clean-close'))
  ctx.on('agent/error', (payload) => finalize(payload, 'error'))

  refreshModelTiers(llm).then((t) => {
    state.modelTiers = t
    log('model tiers: ' + JSON.stringify(t))
  })

  // ---- 工具 ----
  const OUT = (label) => ({
    schema: { type: 'object', additionalProperties: true },
    render: (_args, value) => [{ type: 'text', text: label + '\n' + JSON.stringify(value, null, 2) }],
  })
  const register = (def) => {
    try {
      tools.register(def)
      log('tool registered: ' + def.name)
    } catch (e) {
      log('tool register FAILED: ' + def.name + ' -> ' + String(e))
    }
  }

  register({
    name: 'agentic_router_stats',
    description: '查看 agentic router（UCB/LinUCB/k-NN 三策略 + EXP3 元选择器 + 质量代理奖励）的决策记录、策略权重与飞轮学习状态。',
    parameters: { type: 'object', properties: {} },
    output: OUT('agentic router 状态'),
    isConcurrencySafe: () => true,
    async execute() {
      return {
        mode: state.mode,
        forcedModel: state.forced,
        totalDecisions: state.seq,
        modelTiers: state.modelTiers,
        flywheel: {
          ucb: state.ucb,
          lin: { total: state.lin.total, tierN: state.lin.tierN, tierW: state.lin.tierW },
          knnPoolSize: state.knn.pool.length,
          exp3: state.exp3,
          pendingFeedback: state.pendingFeedback.length,
          recentRewards: store.loadJsonl('rewards.jsonl').slice(-20),
        },
        recent: state.decisions.slice(-20),
      }
    },
  })
  register({
    name: 'agentic_router_set_mode',
    description: '切换 agentic router 模式：shadow（决策与学习照常，但不切换模型）、active（按元选择器结果真正切换）、off（旁路）。',
    parameters: {
      type: 'object',
      properties: { mode: { type: 'string', enum: ['shadow', 'active', 'off'] } },
      required: ['mode'],
    },
    output: OUT('agentic router 模式'),
    isConcurrencySafe: () => true,
    async execute(args) {
      state.mode = args.mode
      store.savePolicy({ mode: state.mode, forced: state.forced })
      return { mode: state.mode }
    },
  })
  register({
    name: 'agentic_router_force',
    description: '强制后续请求使用指定模型 id（仅 active 模式生效）；clear=true 清除强制，恢复规则路由。',
    parameters: {
      type: 'object',
      properties: {
        model: { type: 'string', description: '目标模型 id（见 agentic_router_stats 的 modelTiers）' },
        clear: { type: 'boolean', description: '清除强制' },
      },
    },
    output: OUT('agentic router 强制路由'),
    isConcurrencySafe: () => true,
    async execute(args) {
      if (args && args.clear) {
        state.forced = null
        store.savePolicy({ mode: state.mode, forced: null })
        return { forced: null }
      }
      if (args && typeof args.model === 'string') {
        state.forced = args.model
        store.savePolicy({ mode: state.mode, forced: args.model })
        return { forced: args.model }
      }
      return { forced: state.forced, error: '需要 model 或 clear=true' }
    },
  })
  register({
    name: 'agentic_router_reset',
    description: '清空飞轮学习状态（决策与奖励历史），保留当前模式；用于重新开始学习或清理测试数据。',
    parameters: { type: 'object', properties: {} },
    output: OUT('agentic router 重置'),
    isConcurrencySafe: () => true,
    async execute() {
      store.resetFlywheel()
      state.ucb = { total: 0, arms: {} }
      state.lin = {
        total: 0,
        tierN: { fast: 0, mid: 0, strong: 0 },
        tierW: { fast: new Array(9).fill(0), mid: new Array(9).fill(0), strong: new Array(9).fill(0) },
      }
      state.knn = { pool: [] }
      state.exp3 = { total: 0, weights: { ...DEFAULT_WEIGHTS } }
      state.seq = 0
      state.decisions = []
      return { reset: true, home: store.home }
    },
  })
}
