// dsh-agentic-router —— DeepSeek Harness 插件（Host 侧）
//
// 学习型智能路由（v1.4.0 修复版）：按任务类型与复杂度在模型档位（fast/mid/strong）
// 之间路由；四个专家（规则表 / UCB / LinUCB / k-NN）并行推荐，EXP3 元选择器决定
// 听谁的；回合结束时按质量代理信号（工具失败 + 模型重试 + 成本 + 延迟）计算奖励
// 并回传飞轮；人类 👍/👎 反馈在结算窗口后修正奖励。
//
// ── 切换机制（v1.4.0 修复）────────────────────────────────────────────
// v1.3.0 及之前通过 `agent/request` waterfall 返回替换后的 LlmCallConfig 来切模型，
// 但该返回值会被 dsh-agent 内置的模型选择监听器强制改回「选定模型」，实际不生效。
// 正确做法是走 dsh 官方的模型选择优先级链：
//
//     本进程选择 → 会话日志最新 request/header → 默认模型
//
// 在 `agent/inbox/claimed`（回合开始、prompt 组装之前）向会话日志追加一条
// `request/header` 事件（reason: 'router'），下一轮组装就会读到新模型，真正生效。
// `agent/request` 只做只读观测（记录每步实际出网模型）；插件中途热加载时，
// `llm/stream` 在循环自身 header 落盘后补一次应用（deferred apply）。
//
// 其余架构依据 docs/architecture.md；所有监听经 ctx.on 注册，随插件卸载自动回收。
// 飞轮数据落盘 ${DSH_HOME:-~/.dsh}/storages/dsh-agentic-router/，重启后重放恢复。

import { classify, featuresOf } from './classify.js'
import {
  tierOf, tierTarget, resolveTier,
  recommendRule, recommendUcb, recommendLin, recommendKnn,
  rewardOf, applyReward, correctReward,
  DEFAULT_PRICES, turnCostYuan, resolvePrice,
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
    headerAppends: 0,
    lastActual: null,
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

function extractText(message) {
  if (message == null) return ''
  if (typeof message.text === 'string') return message.text
  const blocks = message.content
  if (Array.isArray(blocks)) {
    let out = ''
    for (const b of blocks) {
      if (b && typeof b === 'object' && b.type === 'text' && typeof b.text === 'string') out += b.text + '\n'
    }
    return out
  }
  return ''
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

// 会话句柄：agent 的 session（inbox / request 路径）
function sessionOf(payload) {
  try {
    const a = payload && payload.agent
    const s = a && a.session
    if (s && typeof s.append === 'function' && typeof s.requestHeader === 'function' && typeof s.id === 'string') return s
  } catch {}
  return null
}

function headerConfig(session) {
  try {
    const h = session.requestHeader()
    if (h && h.config && typeof h.config === 'object') return h.config
  } catch {}
  return null
}

function currentModel(session) {
  const c = headerConfig(session)
  return c && typeof c.model === 'string' ? c.model : null
}

function currentProvider(session) {
  const c = headerConfig(session)
  return c && typeof c.provider === 'string' ? c.provider : ''
}

// 核心切换：向会话日志追加 request/header（reason: 'router'），
// 下一轮 prompt 组装即读到新模型。fast/mid 档去掉 reasoningEffort，
// 让目标模型走自身默认 effort（避免 flash 不支持 max 而整步失败）。
function appendRouterHeader(session, targetModel) {
  try {
    const last = session.requestHeader()
    const header = {}
    if (last && typeof last === 'object') {
      for (const k of Object.keys(last)) header[k] = last[k]
    }
    const prevConfig = header.config && typeof header.config === 'object' ? header.config : {}
    const config = {}
    for (const k of Object.keys(prevConfig)) config[k] = prevConfig[k]
    config.model = targetModel
    if (tierOf(targetModel) !== 'strong') delete config.reasoningEffort
    header.config = config
    session.append('request/header', { header, reason: 'router' })
    return true
  } catch (e) {
    return false
  }
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
  const policy = store.loadPolicy()
  let prices = (policy.prices && Object.keys(policy.prices).length) ? policy.prices : (config.prices || DEFAULT_PRICES)
  const costWeight = typeof config.costWeight === 'number' ? config.costWeight : 20
  const costCap = typeof config.costCap === 'number' ? config.costCap : 0.5

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

  // ---- 决策流水线（inbox 与 request 兜底共用） ----
  // 返回 { key, td, record, target }；已决策过本回合则返回 null。
  // 档位表异步加载；决策前等待就绪（tiers-not-ready 会拿不到 target）。
  async function decideAndRecord(payload, text, current, provider, source) {
    if (!state.modelTiers) {
      try { await tiersReady } catch (e) { /* keep null */ }
    }
    const key = fullKey(payload)
    if (state.turnDecision[key]) return null
    const cls = classify(text)
    const f = featuresOf(cls, text)
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
    const td = {
      clsType: cls.type, f, recs, meta, chosen: meta.chosen,
      finalized: false, t0: now(), toolErrors: 0, retries: 0,
      currentModel: current, provider, costWeight, costCap,
      estInputTokens: Math.ceil(text.length / 1.5),
      target, applied: false, needsApply: false, actualModel: current,
    }
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
      applied: false,
      appliedAt: source,
      swapped: false,
    }
    td.rec = record
    state.turnDecision[key] = td
    state.openTurns.set(agentKey(payload), key)
    return { key, td, record, target }
  }

  // 决策记录落盘（在 applyDecision 之后调用，保证 applied/swapped 已是最终值）
  function commitRecord(record) {
    store.appendJsonl('decisions.jsonl', record)
    state.decisions.push(record)
    if (state.decisions.length > 200) state.decisions.shift()
  }

  // 应用切换：active 模式且目标与当前不同 → 追加 request/header。
  // 返回是否真实追加成功。
  function applyDecision(session, key, td, record, target) {
    if (state.mode !== 'active' || !target) return false
    const base = td.actualModel || td.currentModel
    if (base && target === base) return false
    const ok = session ? appendRouterHeader(session, target) : false
    td.applied = ok
    if (record) { record.applied = ok; record.swapped = ok }
    if (ok) {
      state.headerAppends++
      log('SWAP -> ' + target + ' | ' + td.chosen + ' | ' + (td.recs && td.recs[td.chosen] ? td.recs[td.chosen].reason : ''))
    }
    return ok
  }

  // ---- 事件挂载 ----
  // 回合开始（prompt 组装之前）：捕获用户消息并决策，active 模式立即应用切换。
  // 这是主路径——零延迟，回合第一步就按目标模型执行。
  ctx.on('agent/inbox/claimed', async (payload) => {
    try {
      const session = sessionOf(payload)
      const text = extractText(payload && payload.message)
      if (!text) return
      const current = session ? currentModel(session) : null
      const provider = session ? currentProvider(session) : ''
      const r = await decideAndRecord(payload, text, current, provider, 'inbox')
      if (!r) return
      applyDecision(session, r.key, r.td, r.record, r.target)
      commitRecord(r.record)
      log('decision #' + state.seq + ' type=' + r.td.clsType + ' c=' + r.td.f[1] +
        ' meta=' + r.td.chosen + ' tier=' + r.record.resolvedTier +
        ' applied=' + r.td.applied + ' target=' + (r.target || '-'))
    } catch (e) {
      log('inbox decide failed: ' + String(e))
    }
  })

  ctx.on('agent/pre-step', (payload, next) => {
    try {
      const text = textOfMessages(payload && payload.messages)
      if (text) state.pending.set(agentKey(payload) + ':' + payload.turn, text)
    } catch (e) {
      log('pre-step capture failed: ' + String(e))
    }
    return next()
  })

  // 只读观测 + 兜底决策。观测：记录每步真实出网模型（供 stats 验证切换生效）。
  // 兜底：inbox 未触发（插件中途热加载）时，在首个请求用 pending 文本决策，
  // 但不再替换配置——改由 llm/stream 在循环 header 落盘后补应用。
  ctx.on('agent/request', async (payload, next) => {
    const cfg = await next()
    if (!cfg || typeof cfg !== 'object') return cfg
    try {
      if (!timer || typeof timer.interval !== 'function') settleFeedback().catch(() => {})
      const key = fullKey(payload)
      const model = typeof cfg.model === 'string' ? cfg.model : null
      const td = state.turnDecision[key]
      if (td) {
        if (model) {
          td.actualModel = model
          state.lastActual = { model, turn: payload.turn, step: payload.step, key }
        }
        return cfg
      }
      const pkey = agentKey(payload) + ':' + payload.turn
      const text = state.pending.get(pkey) || ''
      state.pending.delete(pkey)
      const provider = typeof cfg.provider === 'string' ? cfg.provider : ''
      const r = await decideAndRecord(payload, text, model, provider, 'request')
      if (r) {
        r.td.needsApply = state.mode === 'active' && r.target && r.target !== r.td.currentModel
        commitRecord(r.record)
        if (r.td.needsApply) log('decision #' + state.seq + ' -> apply deferred to llm/stream')
        else log('decision #' + state.seq + ' type=' + r.td.clsType + ' meta=' + r.td.chosen + ' tier=' + r.record.resolvedTier)
      }
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
  // 无 usage 的 provider 用文本长度估算（estimated=true）。
  // 同时：request 路径的兜底决策在这里补应用切换（循环自身 header 已落盘，无竞态）。
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
            // 兜底应用：request 路径决策未应用时，在 header 落盘后补切换
            if (td.needsApply && !td.applied) {
              let session = null
              try {
                const sessions = ctx.get('sessions')
                if (sessions && typeof sessions.get === 'function') session = sessions.get(sid) || null
              } catch (e) { session = null }
              if (session) applyDecision(session, key, td, td.rec, td.target)
            }
            const u = td.usage || { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, reasoningTokens: 0, estimated: false }
            const model = (options && typeof options.model === 'string' && options.model) || td.currentModel || ''
            const tier = tierOf(model)
            const provider = (options && typeof options.provider === 'string' && options.provider) || td.provider || ''
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
            const rp = resolvePrice(provider, model, tier, prices)
            td.costYuan = turnCostYuan(u, provider, model, tier, prices)
            td.pricingSource = rp.source
            log('usage on ' + key + ': in=' + u.inputTokens + ' out=' + u.outputTokens + (u.estimated ? ' (估)' : '') + ' cost=¥' + td.costYuan.toFixed(4) + ' [' + rp.source + ']')
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

  // 档位表异步加载；decideAndRecord 在决策前 await 此 promise
  const tiersReady = refreshModelTiers(llm)
  tiersReady.then((t) => {
    state.modelTiers = t
    log('model tiers: ' + JSON.stringify(t))
  }).catch((e) => log('model tiers refresh failed: ' + String(e)))

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
    description: '查看 agentic router（UCB/LinUCB/k-NN 三策略 + EXP3 元选择器 + 质量代理奖励）的决策记录、策略权重、真实切换与飞轮学习状态。',
    parameters: { type: 'object', properties: {} },
    output: OUT('agentic router 状态'),
    isConcurrencySafe: () => true,
    async execute() {
      return {
        mode: state.mode,
        forcedModel: state.forced,
        totalDecisions: state.seq,
        modelTiers: state.modelTiers,
        prices,
        mechanism: 'request/header append (agent/inbox/claimed)',
        lastActual: state.lastActual,
        headerAppends: state.headerAppends,
        flywheel: {
          ucb: state.ucb,
          lin: { total: state.lin.total, tierN: state.lin.tierN, tierW: state.lin.tierW },
          knnPoolSize: state.knn.pool.length,
          exp3: state.exp3,
          pendingFeedback: state.pendingFeedback.length,
          prices,
          recentRewards: store.loadJsonl('rewards.jsonl').slice(-20),
        },
        recent: state.decisions.slice(-20),
        note: '切换真实性验证：headerAppends 为本进程追加的 request/header 次数；lastActual 为最近一步真实出网模型；decisions[].applied=true 且 swapped=true 即真实切换成功',
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
      store.savePolicy({ mode: state.mode, forced: state.forced, prices })
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
        store.savePolicy({ mode: state.mode, forced: null, prices })
        return { forced: null }
      }
      if (args && typeof args.model === 'string') {
        state.forced = args.model
        store.savePolicy({ mode: state.mode, forced: args.model, prices })
        // 立即对存活会话应用（下一请求生效）
        if (state.mode === 'active') {
          try {
            const agents = ctx.get('agents')
            if (agents && typeof agents.list === 'function') {
              for (const agent of agents.list()) {
                try {
                  const session = sessionOf({ agent })
                  if (session && currentModel(session) !== args.model) {
                    const ok = appendRouterHeader(session, args.model)
                    if (ok) state.headerAppends++
                  }
                } catch (e) { /* per-agent best effort */ }
              }
            }
          } catch (e) { /* best effort */ }
        }
        return { forced: args.model }
      }
      return { forced: state.forced, error: '需要 model 或 clear=true' }
    },
  })
  register({
    name: 'agentic_router_set_prices',
    description: '配置/查询某 provider 某模型（或档位）的价格（元/百万 tokens，{in, out}）。不带 priceIn/priceOut 时返回当前价格表。接入新模型后用它补价格，未知价格按保守值估算并标记 unknown。',
    parameters: {
      type: 'object',
      properties: {
        provider: { type: 'string', description: 'provider id，如 openai、anthropic、deepseek-official' },
        model: { type: 'string', description: '可选：精确模型 id；省略则设置该 provider 的档位价' },
        tier: { type: 'string', enum: ['fast', 'mid', 'strong'], description: '档位（未给 model 时必填）' },
        priceIn: { type: 'number', description: '输入价（元/百万 tokens），负数视为查询模式' },
        priceOut: { type: 'number', description: '输出价（元/百万 tokens），负数视为查询模式' },
      },
      required: ['provider'],
    },
    output: OUT('agentic router 价格表'),
    isConcurrencySafe: () => true,
    async execute(args) {
      if (typeof args.priceIn !== 'number' || typeof args.priceOut !== 'number' || args.priceIn < 0 || args.priceOut < 0) {
        return { prices, note: '查看模式：传 priceIn/priceOut 与 model 或 tier 进行设置' }
      }
      const byProvider = prices[args.provider] || (prices[args.provider] = {})
      if (args.model) byProvider[args.model] = { in: args.priceIn, out: args.priceOut }
      else if (args.tier) byProvider[args.tier] = { in: args.priceIn, out: args.priceOut }
      else return { error: '需要 model 或 tier', prices }
      store.savePolicy({ mode: state.mode, forced: state.forced, prices })
      return { prices, updated: args.model || (args.provider + '/' + args.tier) }
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
      state.headerAppends = 0
      state.lastActual = null
      return { reset: true, home: store.home }
    },
  })
}
