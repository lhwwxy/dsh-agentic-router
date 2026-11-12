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
  tierOf, tierTarget, tierHasAny, resolveTier,
  recommendRule, recommendUcb, recommendLin, recommendKnn,
  rewardOf, applyReward, correctReward,
  DEFAULT_PRICES, turnCostYuan, resolvePrice, mergePrices,
} from './strategies.js'
import { metaPick, DEFAULT_WEIGHTS } from './meta.js'
import { Store, defaultHome } from './store.js'

export const name = 'agentic-router'
export const inject = ['tools']

const LIN_GRADUATION = 50   // LinUCB 毕业样本线
const KNN_GRADUATION = 30   // k-NN 毕业经验线
const SETTLE_DELAY_MS = 45000 // 反馈结算窗口：回合结束 45s 后查询 messageFeedback
const SWEEP_MS = 15000        // 结算轮询周期
const TIERS_TTL_MS = 600000   // 模型目录 TTL：默认 10 分钟自动重新探测（provider/模型增删自愈）

// 空档位表：插件不在代码里写死任何模型清单。启动时优先读本地缓存（models.json，
// 上一次运行时探测的结果），无缓存则为空表——此时照常记录决策但不切换，
// 等 llm 探测（启动/手动/TTL/请求出错）完成后立即恢复决策能力。
function emptyTiers() {
  return { providers: [], tiers: { fast: [], mid: [], strong: [] } }
}

// 当前 provider 有候选模型的档位列表（UCB/LinUCB/k-NN 只在这些档位间探索）。
// provider 缺失或档位表不可用时回退到全部档位（保持专家函数原有行为）。
function availableTiersFor(provider, modelTiers, crossProvider) {
  const all = ['fast', 'mid', 'strong']
  if (!modelTiers) return all
  if (provider && !crossProvider) {
    const has = all.filter((t) => tierTarget(t, provider, modelTiers))
    return has.length ? has : all
  }
  // 跨 provider（或无 provider）：只看档位是否有候选（跨所有 provider），避免探索"幽灵档"
  const has = all.filter((t) => tierHasAny(t, modelTiers))
  return has.length ? has : all
}

function freshState() {
  return {
    mode: 'shadow',
    seq: 0,
    decisions: [],
    forced: null,
    crossProvider: true,
    pending: new Map(),
    modelTiers: emptyTiers(),
    catalogTs: null,
    catalogSource: 'empty',
    probing: false,
    lastProbeAt: 0,
    turnDecision: {},
    openTurns: new Map(),
    pendingFeedback: [],
    recByTurn: new Map(),
    headerAppends: 0,
    lastActual: null,
    stepLog: [],
    counters: { stepSwitches: 0 },
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

// 核心切换：向会话日志追加 request/header（reason: 'router' / 'router-step'），
// 下一轮 prompt 组装即读到新模型。fast/mid 档去掉 reasoningEffort，
// 让目标模型走自身默认 effort（避免 flash 不支持 max 而整步失败）。
function appendRouterHeader(session, targetModel, reason = 'router', targetProvider = null) {
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
    if (targetProvider && config.provider !== targetProvider) config.provider = targetProvider
    if (tierOf(targetModel) !== 'strong') delete config.reasoningEffort
    header.config = config
    session.append('request/header', { header, reason })
    return true
  } catch (e) {
    return false
  }
}

// 解析强制模型：支持 'model' 或 'provider/model'（跨厂商强制时需指定 provider）
function parseForced(forced) {
  if (typeof forced !== 'string' || !forced) return { provider: null, model: forced || null }
  const i = forced.indexOf('/')
  if (i > 0) return { provider: forced.slice(0, i), model: forced.slice(i + 1) }
  return { provider: null, model: forced }
}

// 步骤级路由（v1.5.0）：视为"重型工作"的工具调用——上一步出现它们时，
// 下一步升级到 strong（pro）处理其结果/继续执行；轻量工具保持回合基线。
const HEAVY_TOOLS = new Set([
  'bash', 'pwsh', 'write', 'edit', 'batch_edit', 'run_code', 'workflow',
  'subagent', 'subagent_fork', 'terminal', 'pty', 'mcp',
])

// 读取会话里最后一条 assistant 消息的工具调用名（只取叶子字段，不碰活数据）。
function lastToolCalls(session) {
  try {
    const msgs = typeof session.deriveMessages === 'function' ? session.deriveMessages() : null
    if (!Array.isArray(msgs)) return []
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i]
      if (!m || typeof m !== 'object' || m.role !== 'assistant') continue
      const blocks = m.content
      if (!Array.isArray(blocks)) return []
      const out = []
      for (const b of blocks) {
        if (b && typeof b === 'object' && b.type === 'tool-call' && typeof b.name === 'string') out.push(b.name)
      }
      return out
    }
  } catch (e) { /* read-only observation */ }
  return []
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
  const log = (msg) => console.log('[agentic-router]', msg)
  const tools = ctx.get('tools')
  const llm = ctx.get('llm')
  const mf = ctx.get('messageFeedback')
  const timer = ctx.get('timer')
  const store = new Store(config.home || defaultHome())
  const settleDelayMs = typeof config.settleDelayMs === 'number' ? config.settleDelayMs : SETTLE_DELAY_MS
  const policy = store.loadPolicy()
  const overlay = (policy.prices && Object.keys(policy.prices).length) ? policy.prices : (config.prices || {})
  let prices = mergePrices(DEFAULT_PRICES, overlay)
  const costWeight = typeof config.costWeight === 'number' ? config.costWeight : 20
  const costCap = typeof config.costCap === 'number' ? config.costCap : 0.5
  const crossProvider = config.crossProvider !== false

  const state = freshState()
  state.mode = policy.mode || config.mode || 'shadow'
  state.crossProvider = crossProvider
  if (typeof policy.forced === 'string') state.forced = policy.forced

  // 模型目录：不写死。先读上次探测的本地缓存（models.json），没有则为空表。
  // 缓存里的 provider 可能已被删除——可用性以本次 llm 探测为准，所以启动时无条件重探。
  const cached = store.loadModels()
  if (cached) {
    state.modelTiers = { providers: cached.providers, tiers: cached.tiers }
    state.catalogTs = typeof cached.ts === 'number' ? cached.ts : null
    state.catalogSource = 'cache'
    log('model catalog from cache: ' + cached.providers.length + ' providers (ts=' + cached.ts + ')')
  }
  const tiersTtlMs = typeof config.tiersTtlMs === 'number' ? config.tiersTtlMs : TIERS_TTL_MS

  // 探测：并发保护 + TTL + 失败保留旧缓存。成功后覆盖 state 并落盘 models.json。
  async function probeCatalog(reason) {
    if (state.probing) return null
    state.probing = true
    state.lastProbeAt = Date.now() // 无论成败都算"尝试过"，TTL 内不再重试，避免失败风暴
    try {
      const t = await refreshModelTiers(llm)
      if (t && t.providers.length) {
        state.modelTiers = t
        state.catalogTs = state.lastProbeAt
        state.catalogSource = 'live'
        try { store.saveModels({ ts: state.catalogTs, providers: t.providers, tiers: t.tiers }) } catch {}
        log('model catalog probed (' + reason + '): ' + t.providers.length + ' providers, tiers fast=' + t.tiers.fast.length + ' mid=' + t.tiers.mid.length + ' strong=' + t.tiers.strong.length)
        return t
      }
      log('model catalog probe empty (' + reason + '), keep cached')
      return null
    } catch (e) {
      log('model catalog probe failed (' + reason + '): ' + String(e))
      return null
    } finally {
      state.probing = false
    }
  }
  // TTL 惰性刷新：回合内每步 request 路径都会检查，过期则后台重探（不阻塞决策）。
  function maybeRefreshCatalog() {
    if (state.probing) return
    const stale = !state.catalogTs || (Date.now() - (state.lastProbeAt || 0) > tiersTtlMs)
    if (stale) probeCatalog('ttl').catch(() => {})
  }

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
  // 必须同步执行：模型目录来自本地缓存/运行时探测（见 freshState 与 probeCatalog），
  // 目录未就绪时照常记录决策但不切换，不 await、不延期——否则追加 header 会落到 prompt 组装之后。
  function decideAndRecord(payload, text, current, provider, source) {
    const key = fullKey(payload)
    if (state.turnDecision[key]) return null
    const cls = classify(text)
    const f = featuresOf(cls, text)
    // 只探索当前 provider 有候选模型的档位（跳过无候选的"幽灵档"，
    // 如 mid 无模型时探索 mid 只会白费并回退到最贵的 strong）
    const available = availableTiersFor(provider, state.modelTiers, state.crossProvider)
    const recs = {
      rule: recommendRule(cls),
      ucb: recommendUcb(cls, state.ucb, available),
      lin: recommendLin(cls, f, state.lin, available),
      knn: recommendKnn(cls, f, state.knn, available),
    }
    if (state.forced) recs.rule = { tier: 'forced', reason: 'forced', conf: 1 }
    const exp3 = {
      ...state.exp3,
      linGraduated: state.lin.total >= LIN_GRADUATION,
      knnGraduated: state.knn.pool.length >= KNN_GRADUATION,
    }
    const meta = metaPick(recs, exp3)
    const chosenRec = meta.byName[meta.chosen] || recs.rule
    let resolved = state.forced
      ? (() => { const pf = parseForced(state.forced); return { target: pf.model, provider: pf.provider, used: 'forced', note: 'forced' } })()
      : resolveTier(chosenRec.tier, provider, state.modelTiers, { crossProvider: state.crossProvider })
    // 首回合/无 provider：active 模式下等首个 request 拿到真实 provider 后再解析、应用与落盘
    const defer = state.mode === 'active' && !provider && !state.forced
    if (defer) resolved = { target: null, provider: null, used: 'pending', note: 'provider-unknown' }
    const target = resolved.target
    const targetProvider = resolved.provider || null
    state.seq += 1
    const td = {
      clsType: cls.type, f, recs, meta, chosen: meta.chosen,
      finalized: false, t0: now(), toolErrors: 0, retries: 0,
      currentModel: current, provider, costWeight, costCap,
      estInputTokens: Math.ceil(text.length / 1.5),
      target, targetProvider, applied: false, needsApply: false, actualModel: current,
      awaitProvider: defer, recordCommitted: false,
      // 回合级基线（供步骤级路由 v1.5.0 参考：无重型工具时保持该档）
      turnTier: target ? tierOf(target) : null, turnTarget: target,
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
      targetProvider: state.mode === 'active' ? targetProvider : null,
      deferred: defer,
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
  function commitRecord(record, td) {
    store.appendJsonl('decisions.jsonl', record)
    state.decisions.push(record)
    if (state.decisions.length > 200) state.decisions.shift()
    if (td) td.recordCommitted = true
  }

  // 应用切换：active 模式且目标与当前不同 → 追加 request/header。
  // 返回是否真实追加成功。
  function applyDecision(session, key, td, record, target) {
    if (state.mode !== 'active' || !target) return false
    const base = td.actualModel || td.currentModel
    if (base && target === base) return false
    const ok = session ? appendRouterHeader(session, target, 'router', td.targetProvider || null) : false
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
  ctx.on('agent/inbox/claimed', (payload) => {
    try {
      const session = sessionOf(payload)
      const text = extractText(payload && payload.message)
      if (!text) return
      const current = session ? currentModel(session) : null
      const provider = session ? currentProvider(session) : ''
      const r = decideAndRecord(payload, text, current, provider, 'inbox')
      if (!r) return
      applyDecision(session, r.key, r.td, r.record, r.target)
      if (!r.td.awaitProvider) commitRecord(r.record, r.td)
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
    // ---- 步骤级路由（v1.5.0）----
    // pre-step 在组装之后触发 → 此处追加 header 对下一步生效（一阶延迟，设计如此）。
    // 信号：上一步 assistant 消息的工具调用。重型工具（写代码/执行/子agent）→ 下一步
    // 升级 strong（pro）；轻量/无工具 → 回到回合基线档。
    try {
      const session = sessionOf(payload)
      if (!session) return next()
      const key = fullKey(payload)
      const td = state.turnDecision[key]
      if (!td || td.finalized) return next()
      if (state.forced) return next() // 强制模型时跳过步骤级升降档（force 钉死模型）
      const tools = lastToolCalls(session)
      if (!tools.length) return next() // 上一步是纯文本（通常回合收尾），保持现状
      const heavy = tools.some((n) => HEAVY_TOOLS.has(n))
      const stepTier = heavy ? 'strong' : (td.turnTier || null)
      if (!stepTier) return next()
      const provider = td.provider || currentProvider(session) || ''
      const resolved = resolveTier(stepTier, provider, state.modelTiers, { crossProvider: state.crossProvider })
      const target = resolved.target
      const targetProvider = resolved.provider || null
      const cur = currentModel(session)
      const changed = target && target !== cur
      // 注意：pre-step 在组装之后、buildRequest（loop 追加自己的 header）之前触发。
      // 这里直接追加会被本步 buildRequest 的 change(旧模型) 覆盖（与 v1.4.0 冷启动同款竞态）。
      // 因此只决策并挂起，真正追加推迟到 llm/stream 收尾（此时 loop 的 header 已落盘）。
      if (changed) { td.stepTarget = target; td.stepTargetProvider = targetProvider }
      const stepRec = { turn: key, step: payload.step, lastTools: tools.slice(0, 5), stepTier, target, provider: targetProvider, current: cur, applied: false, pending: changed, ts: Date.now() }
      if (changed) td.stepRec = stepRec
      state.stepLog.push(stepRec)
      if (state.stepLog.length > 30) state.stepLog.shift()
      try { store.appendJsonl('steps.jsonl', stepRec) } catch (e) { /* 落盘尽力而为 */ }
    } catch (e) {
      log('step routing failed: ' + String(e))
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
      maybeRefreshCatalog() // TTL 惰性刷新模型目录（不阻塞决策）
      const key = fullKey(payload)
      const model = typeof cfg.model === 'string' ? cfg.model : null
      const provider = typeof cfg.provider === 'string' ? cfg.provider : ''
      const td = state.turnDecision[key]
      if (td) {
        if (model) {
          td.actualModel = model
          state.lastActual = { model, turn: payload.turn, step: payload.step, key }
        }
        if (provider) td.provider = provider
        if (td.awaitProvider && !td.finalized) {
          const chosenRec = (td.recs && td.recs[td.chosen]) || td.recs.rule
          let resolved = state.forced
            ? (() => { const pf = parseForced(state.forced); return { target: pf.model, provider: pf.provider, used: 'forced', note: 'forced' } })()
            : resolveTier(chosenRec.tier, provider, state.modelTiers, { crossProvider: state.crossProvider })
          td.target = resolved.target
          td.targetProvider = resolved.provider || null
          td.turnTier = td.target ? tierOf(td.target) : null
          td.turnTarget = td.target
          td.awaitProvider = false
          if (td.rec) {
            td.rec.resolvedTier = resolved.used + (resolved.note ? ' [' + resolved.note + ']' : '')
            td.rec.targetModel = state.mode === 'active' ? td.target : null
            td.rec.targetProvider = state.mode === 'active' ? td.targetProvider : null
            td.rec.currentModel = model
          }
          if (!td.recordCommitted) commitRecord(td.rec, td)
          td.needsApply = state.mode === 'active' && td.target && td.target !== model
          log('decision #' + state.seq + ' resolved@request tier=' + td.rec.resolvedTier + ' target=' + (td.target || '-') + ' needsApply=' + td.needsApply)
        }
        return cfg
      }
      const pkey = agentKey(payload) + ':' + payload.turn
      const text = state.pending.get(pkey) || ''
      state.pending.delete(pkey)
      const r = decideAndRecord(payload, text, model, provider, 'request')
      if (r) {
        r.td.needsApply = state.mode === 'active' && r.target && r.target !== r.td.currentModel
        if (!r.td.awaitProvider) commitRecord(r.record, r.td)
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
      // 请求失败（可能是模型/provider 已下线）→ 立即触发重探模型目录，及时剔除失效项
      probeCatalog('request-error').catch(() => {})
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
            // 延迟应用（此时循环自身 header 已落盘，追加无竞态）：
            // 1) 步骤级路由挂起的升降档（router-step）
            // 2) request 路径兜底决策（needsApply）
            let session = null
            try {
              const sessions = ctx.get('sessions')
              if (sessions && typeof sessions.get === 'function') session = sessions.get(sid) || null
            } catch (e) { session = null }
            if (session) {
              if (td.stepTarget && !state.forced && (td.stepTarget !== currentModel(session) || (td.stepTargetProvider && td.stepTargetProvider !== currentProvider(session)))) {
                const ok = appendRouterHeader(session, td.stepTarget, 'router-step', td.stepTargetProvider || null)
                if (ok) {
                  state.counters.stepSwitches++
                  if (td.stepRec) { td.stepRec.applied = true }
                  log('STEP-SWAP -> ' + td.stepTarget + (td.stepTargetProvider ? ' (' + td.stepTargetProvider + ')' : '') + (td.stepRec ? ' [tools: ' + td.stepRec.lastTools.join(',') + ']' : ''))
                }
              }
              td.stepTarget = null
              td.stepTargetProvider = null
              if (td.needsApply && !td.applied) applyDecision(session, key, td, td.rec, td.target)
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
      if (!td.recordCommitted && td.rec) commitRecord(td.rec, td)
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

  // 启动即探测（无条件，确保删除的 provider 不会残留在缓存里）；
  // 有缓存时缓存已用于启动决策，探测完成后覆盖；无缓存时首个 request 也会触发探测。
  probeCatalog('startup').catch((e) => log('startup probe failed: ' + String(e)))

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
        crossProvider: state.crossProvider,
        totalDecisions: state.seq,
        modelTiers: state.modelTiers,
        modelCatalog: {
          source: state.catalogSource, // live（刚探测）/ cache（本地 models.json）/ empty（无目录，等探测）
          ts: state.catalogTs,
          lastProbeAt: state.lastProbeAt || null,
          probing: state.probing,
          ttlMs: tiersTtlMs,
          cacheFile: 'models.json',
        },
        prices,
        mechanism: 'request/header append (inbox turn-level + pre-step step-level v1.5.0; model catalog probed at runtime + cached locally)',
        lastActual: state.lastActual,
        headerAppends: state.headerAppends,
        stepSwitches: state.counters.stepSwitches,
        stepLog: state.stepLog.slice(-20),
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
        note: '切换真实性验证：headerAppends 为本进程追加的 request/header 次数；stepSwitches 为步骤级升级/降级次数（stepLog 有明细）；lastActual 为最近一步真实出网模型',
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
    description: '强制后续请求使用指定模型 id（仅 active 模式生效，支持 model 或 provider/model）；clear=true 清除强制，恢复规则路由。',
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
          const pf = parseForced(args.model)
          try {
            const agents = ctx.get('agents')
            if (agents && typeof agents.list === 'function') {
              for (const agent of agents.list()) {
                try {
                  const session = sessionOf({ agent })
                  if (session && currentModel(session) !== pf.model) {
                    const ok = appendRouterHeader(session, pf.model, 'router', pf.provider)
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
    name: 'agentic_router_refresh_models',
    description: '重新探测当前 DSH 的 provider/模型目录，更新路由档位并落盘到本地 models.json（不写死模型清单；provider/模型增删后用它立即刷新，默认也会按 TTL 自动刷新）。',
    parameters: { type: 'object', properties: {} },
    output: OUT('agentic router 模型目录刷新'),
    isConcurrencySafe: () => true,
    async execute() {
      const before = state.modelTiers
      const t = await probeCatalog('manual')
      const mt = t || state.modelTiers
      return {
        ok: !!t,
        changed: JSON.stringify(before) !== JSON.stringify(state.modelTiers),
        modelCatalog: {
          source: state.catalogSource,
          ts: state.catalogTs,
          probing: state.probing,
        },
        modelTiers: mt,
        note: '已按当前 DSH 实际 provider/模型重新探测并写入 models.json；被删除的 provider 不会保留',
      }
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
      store.clearModels()
      state.modelTiers = emptyTiers()
      state.catalogTs = null
      state.catalogSource = 'empty'
      probeCatalog('reset').catch(() => {}) // 重置后立即重新探测模型目录
      return { reset: true, home: store.home }
    },
  })
}
