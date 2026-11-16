// EXP3 元选择器：把 {UCB, LinUCB, k-NN, rule} 当臂，按权重软最大采样，
// 被选策略按重要性加权更新（reward / p_chosen）。

const DEFAULT_WEIGHTS = { ucb: 1, linucb: 1, knn: 1, rule: 2.5 }

// 混合探索率 γ：采样概率 = (1-γ)·权重占比 + γ·均匀。
// 没有它，权重被反馈修正持续压到趋 0 的专家将永远不会再被选中——
// 无法在非平稳环境（任务分布漂移）中复苏；γ 给每个入池专家保底概率。
const GAMMA = 0.05

export function metaPick(recs, exp3) {
  const weights = exp3.weights
  const pool = ['ucb']
  const byName = { ucb: recs.ucb, rule: recs.rule }
  if (exp3.linGraduated && recs.lin && recs.lin.tier) {
    pool.push('linucb')
    byName.linucb = recs.lin
  }
  if (exp3.knnGraduated && recs.knn && recs.knn.tier) {
    pool.push('knn')
    byName.knn = recs.knn
  }
  pool.push('rule')

  let total = 0
  for (const n of pool) total += weights[n] || 0
  const gamma = typeof exp3.gamma === 'number' ? exp3.gamma : GAMMA
  const ps = {}
  if (total > 0) {
    for (const n of pool) ps[n] = (1 - gamma) * (weights[n] || 0) / total + gamma / pool.length
  } else {
    // 权重全 0（极端退化）时退化为均匀采样
    for (const n of pool) ps[n] = 1 / pool.length
  }
  let r = Math.random()
  let chosen = 'rule'
  for (const n of pool) {
    if (r < ps[n]) { chosen = n; break }
    r -= ps[n]
  }
  return { chosen, ps, pool, byName }
}

export { DEFAULT_WEIGHTS }
