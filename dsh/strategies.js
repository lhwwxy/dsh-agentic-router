// 四专家策略 + 奖励计算 + 飞轮更新（纯函数，state 从外部传入）
// 算法选择（见 docs/DESIGN.md）：
//   UCB      —— 簇×档臂的多臂老虎机（冷启动上界∞，平局轮转探索）
//   LinUCB   —— 特征线性打分 + 探索项（毕业线 50 样本）
//   k-NN     —— 经验池最近邻聚合（毕业线 30 经验）
//   rule     —— 确定性阈值表（冷启动先验与兜底）

export const FAST_RE = /flash|lite|mini|small|fast|air|nano|tiny|light/
export const STRONG_RE = /pro|max|ultra|plus|large|premium|reason|thinking/

export function tierOf(model) {
  if (!model) return 'unknown'
  if (FAST_RE.test(model)) return 'fast'
  if (STRONG_RE.test(model)) return 'strong'
  return 'mid'
}

export function tierTarget(tier, provider, modelTiers) {
  const list = modelTiers && modelTiers.tiers && modelTiers.tiers[tier] ? modelTiers.tiers[tier] : []
  for (const s of list) if (s.indexOf(provider + '/') === 0) return s.slice(provider.length + 1)
  return null
}

// 档位 → 模型 id 的解析；mid 档无候选时向上取 strong（保质量），再取 fast
export function resolveTier(tier, provider, modelTiers) {
  if (!modelTiers) return { target: null, used: tier, note: 'tiers-not-ready' }
  const target = tierTarget(tier, provider, modelTiers)
  if (target) return { target, used: tier, note: '' }
  if (tier === 'mid') {
    const strong = tierTarget('strong', provider, modelTiers)
    if (strong) return { target: strong, used: 'strong', note: 'mid->strong fallback' }
    const fast = tierTarget('fast', provider, modelTiers)
    if (fast) return { target: fast, used: 'fast', note: 'mid->fast fallback' }
  }
  return { target: null, used: tier, note: 'no-candidate' }
}

export function recommendRule(cls) {
  const want = cls.complexity >= 0.7 ? 'strong' : (cls.complexity <= 0.3 ? 'fast' : 'mid')
  return { tier: want, reason: 'rule threshold complexity=' + cls.complexity, conf: 0.5 }
}

// UCB：mean + c·√(lnN/n)；未探索臂上界∞，多个∞平局时按总拉动数轮转（避免永远选 fast）。
// availableTiers 由调用方传入（当前 provider 有候选模型的档位）——不探索无候选的"幽灵档"
// （如 mid 无模型时，探索 mid 只会白费并回退到最贵的 strong）。
export function recommendUcb(cls, ucbState, availableTiers = ['fast', 'mid', 'strong']) {
  const arms = ucbState.arms
  const scores = []
  for (const tier of availableTiers) {
    const arm = arms[cls.type + ':' + tier] || { n: 0, mean: 0 }
    const ucb = arm.n > 0 ? arm.mean + 1.2 * Math.sqrt(Math.log(ucbState.total + 1) / arm.n) : Infinity
    scores.push({ tier, ucb, arm })
  }
  let best = scores[0]
  for (const s of scores) if (s.ucb > best.ucb) best = s
  const infs = scores.filter((s) => s.ucb === Infinity)
  if (infs.length > 1) best = infs[ucbState.total % infs.length]
  return {
    tier: best.tier,
    reason: 'ucb ' + cls.type + ' n=' + best.arm.n + ' mean=' + (Math.round(best.arm.mean * 100) / 100) +
      ' bound=' + (isFinite(best.ucb) ? Math.round(best.ucb * 100) / 100 : 'inf'),
    conf: isFinite(best.ucb) ? 0.6 : 0.2,
  }
}

// LinUCB（近似实现）：每档线性权重 w·f + 探索项；梯度更新，探索项随档样本数衰减。
// availableTiers 同上：只评估有候选模型的档位。
export function recommendLin(cls, f, linState, availableTiers = ['fast', 'mid', 'strong']) {
  const tn = linState.tierN
  let best = null
  for (const tier of availableTiers) {
    const w = linState.tierW[tier] || []
    let dot = 0
    for (let i = 0; i < f.length; i++) dot += (w[i] || 0) * f[i]
    const bonus = tn[tier] > 0 ? 0.8 * Math.sqrt(Math.log(linState.total + 1) / tn[tier]) : 0.3
    const score = dot + bonus
    if (!best || score > best.score) best = { tier, score, dot }
  }
  return {
    tier: best.tier,
    reason: 'linucb score=' + Math.round(best.score * 100) / 100 + ' (dot=' + Math.round(best.dot * 100) / 100 + ')',
    conf: linState.total >= 50 ? 0.7 : 0.1,
  }
}

// k-NN：欧氏距离取 k=5 邻居，按邻居平均奖励选档。
// availableTiers 同上：只从有候选模型的档位里选最佳。
export function recommendKnn(cls, f, knnState, availableTiers = ['fast', 'mid', 'strong']) {
  const pool = knnState.pool
  if (pool.length < 5) return { tier: null, reason: 'knn cold (' + pool.length + '/5 seed)', conf: 0 }
  const scored = []
  for (const e of pool) {
    let d = 0
    for (let i = 0; i < f.length; i++) d += (e.f[i] - f[i]) * (e.f[i] - f[i])
    scored.push({ d, e })
  }
  scored.sort((a, b) => a.d - b.d)
  const k = Math.min(5, scored.length)
  const agg = { fast: { n: 0, s: 0 }, mid: { n: 0, s: 0 }, strong: { n: 0, s: 0 } }
  for (let i = 0; i < k; i++) {
    const t = scored[i].e.tier
    if (agg[t]) { agg[t].n += 1; agg[t].s += scored[i].e.r }
  }
  let best = null
  for (const tier of availableTiers) {
    const a = agg[tier]
    if (!best || (a.n > 0 && a.s / a.n > best.score)) best = { tier, score: a.n > 0 ? a.s / a.n : -1 }
  }
  return { tier: best.tier, reason: 'knn k=' + k + ' neighbors score=' + Math.round(best.score * 100) / 100, conf: Math.min(0.8, pool.length / 100) }
}

// ---- 成本信号（token 计费） ----
// 价格表（元/百万 tokens）三级解析：
//   1) prices[provider][model]          精确模型价（最高优先）
//   2) prices[provider][tier]           该 provider 的档位价
//   3) prices['*'][tier]                通用档位兜底（未知 provider 不套用 DeepSeek 价）
//   4) CONSERVATIVE                     保守值 + 标记 unknown，提醒用户补价格
export const DEFAULT_PRICES = {
  'deepseek-official': {
    'deepseek-v4-flash': { in: 0.2, out: 2 },
    'deepseek-v4-pro': { in: 1, out: 12 },
  },
  '*': {
    fast: { in: 1, out: 4 },
    mid: { in: 2, out: 8 },
    strong: { in: 3, out: 16 },
  },
}

const CONSERVATIVE = { in: 3, out: 16 }

// 解析价格并返回来源（透明标注：价格到底从哪来的）
export function resolvePrice(provider, model, tier, prices) {
  const table = prices || DEFAULT_PRICES
  const byProvider = table[provider]
  if (byProvider && byProvider[model]) return { price: byProvider[model], source: 'provider-model' }
  if (byProvider && byProvider[tier]) return { price: byProvider[tier], source: 'provider-tier' }
  const generic = table['*']
  if (generic && generic[tier]) return { price: generic[tier], source: 'generic-tier' }
  return { price: CONSERVATIVE, source: 'unknown-conservative' }
}

// 单回合成本（元）：缓存读计入输入，推理 token 计入输出
export function turnCostYuan(usage, provider, model, tier, prices) {
  if (!usage) return 0
  const { price } = resolvePrice(provider, model, tier, prices)
  const input = (usage.inputTokens || 0) + (usage.cacheReadTokens || 0)
  const output = (usage.outputTokens || 0) + (usage.reasoningTokens || 0)
  return (price.in * input + price.out * output) / 1e6
}

// 质量代理奖励：干净收尾 1 分起，扣工具失败(0.2/次，封顶 0.6)、模型重试(0.2/次，封顶 0.4)、
// 延迟(>90s −0.3, >30s −0.1)、成本(min(cap, weight × 元)，默认 weight=20 即每 0.01 元 −0.2)
export function rewardOf(td, tag, dtMs) {
  if (tag !== 'clean-close') {
    return { reward: 0, breakdown: { base: 0, toolErrors: td.toolErrors || 0, retries: td.retries || 0, dtMs, costYuan: td.costYuan || 0, costPenalty: 0, pricingSource: td.pricingSource || 'n/a' } }
  }
  let reward = 1
  const toolPenalty = Math.min(0.6, 0.2 * (td.toolErrors || 0))
  const retryPenalty = Math.min(0.4, 0.2 * (td.retries || 0))
  const costCap = typeof td.costCap === 'number' ? td.costCap : 0.5
  const costWeight = typeof td.costWeight === 'number' ? td.costWeight : 20
  const costYuan = typeof td.costYuan === 'number' ? td.costYuan : 0
  const costPenalty = Math.min(costCap, costWeight * costYuan)
  reward -= toolPenalty
  reward -= retryPenalty
  reward -= costPenalty
  if (dtMs != null) {
    if (dtMs > 90000) reward -= 0.3
    else if (dtMs > 30000) reward -= 0.1
  }
  return {
    reward: Math.max(0, Math.round(reward * 100) / 100),
    breakdown: { base: 1, toolErrors: td.toolErrors || 0, toolPenalty, retries: td.retries || 0, retryPenalty, costYuan, costPenalty, dtMs, pricingSource: td.pricingSource || 'n/a' },
  }
}

// 把一条奖励记录应用到飞轮（UCB/LinUCB/k-NN/EXP3），replay 与实时更新共用
export function applyReward(state, rec) {
  // 记录本次线性更新的学习率，供反馈结算修正精确回退
  rec.linLr = 1 / (1 + (state.lin.tierN[rec.linTier] || 0))
  const armKey = rec.clsType + ':' + rec.ucbTier
  const arm = state.ucb.arms[armKey] || (state.ucb.arms[armKey] = { n: 0, mean: 0 })
  arm.n += 1
  arm.mean = arm.mean + (rec.reward - arm.mean) / arm.n
  state.ucb.total += 1

  const lw = state.lin.tierW[rec.linTier] || []
  const lr = 1 / (1 + (state.lin.tierN[rec.linTier] || 0))
  let dot = 0
  for (let i = 0; i < rec.f.length; i++) dot += (lw[i] || 0) * rec.f[i]
  for (let i = 0; i < rec.f.length; i++) lw[i] = (lw[i] || 0) + lr * (rec.reward - dot) * rec.f[i]
  state.lin.tierW[rec.linTier] = lw
  state.lin.tierN[rec.linTier] += 1
  state.lin.total += 1

  state.knn.pool.push({ f: rec.f, tier: rec.ucbTier, r: rec.reward })
  if (state.knn.pool.length > 400) state.knn.pool.shift()

  if (rec.pChosen > 0) {
    state.exp3.weights[rec.chosen] = (state.exp3.weights[rec.chosen] || 0) *
      Math.exp(0.3 * (rec.reward / rec.pChosen) / rec.pool.length)
  }
  state.exp3.total += 1
}

// 反馈结算修正：把已应用奖励改为 newReward（人类在反馈窗口内点了 👍/👎 之后）
// UCB 均值修正（n 不变）；LinUCB 用原始学习率做精确梯度回退；k-NN 追加修正经验；
// EXP3 按乘性更新做精确逆修正。
export function correctReward(state, rec, newReward) {
  const delta = newReward - rec.reward
  if (!delta) return { delta: 0 }

  const arm = state.ucb.arms[rec.clsType + ':' + rec.ucbTier]
  if (arm && arm.n > 0) arm.mean = arm.mean + delta / arm.n

  const lw = state.lin.tierW[rec.linTier]
  if (lw && Array.isArray(rec.f)) {
    const lr = typeof rec.linLr === 'number' ? rec.linLr : 1 / (1 + (state.lin.tierN[rec.linTier] || 0))
    for (let i = 0; i < rec.f.length; i++) lw[i] = (lw[i] || 0) + lr * delta * rec.f[i]
  }

  if (state.knn && Array.isArray(rec.f)) state.knn.pool.push({ f: rec.f, tier: rec.ucbTier, r: newReward })

  if (rec.pChosen > 0 && rec.pool && rec.pool.length > 0) {
    state.exp3.weights[rec.chosen] = (state.exp3.weights[rec.chosen] || 0) *
      Math.exp(0.3 * delta / (rec.pChosen * rec.pool.length))
  }

  rec.reward = newReward
  return { delta }
}
