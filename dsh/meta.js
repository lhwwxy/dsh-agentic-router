// EXP3 元选择器：把 {UCB, LinUCB, k-NN, rule} 当臂，按权重软最大采样，
// 被选策略按重要性加权更新（reward / p_chosen）。

const DEFAULT_WEIGHTS = { ucb: 1, linucb: 1, knn: 1, rule: 2.5 }

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
  const ps = {}
  for (const n of pool) ps[n] = (weights[n] || 0) / total
  let r = Math.random()
  let chosen = 'rule'
  for (const n of pool) {
    if (r < ps[n]) { chosen = n; break }
    r -= ps[n]
  }
  return { chosen, ps, pool, byName }
}

export { DEFAULT_WEIGHTS }
