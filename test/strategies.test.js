// 策略与飞轮测试：UCB 冷启动轮转 / mid 回退 / 奖励公式 / applyReward 更新
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  tierOf, tierTarget, resolveTier,
  recommendRule, recommendUcb, recommendLin, recommendKnn,
  rewardOf, applyReward, correctReward,
} from '../dsh/strategies.js'
import { metaPick, DEFAULT_WEIGHTS } from '../dsh/meta.js'

const MODEL_TIERS = {
  providers: [{ provider: 'deepseek-official', models: ['deepseek-v4-flash', 'deepseek-v4-pro'] }],
  tiers: { fast: ['deepseek-official/deepseek-v4-flash'], mid: [], strong: ['deepseek-official/deepseek-v4-pro'] },
}

test('tierOf 按 id 关键词分档', () => {
  assert.equal(tierOf('deepseek-v4-flash'), 'fast')
  assert.equal(tierOf('deepseek-v4-pro'), 'strong')
  assert.equal(tierOf('some-model-x'), 'mid')
})

test('UCB 冷启动：三个未探索臂∞平局时轮转，不会永远选 fast（实测修复）', () => {
  const st = { total: 0, arms: {} }
  const t0 = recommendUcb({ type: 'qa' }, st).tier
  st.total = 1
  const t1 = recommendUcb({ type: 'qa' }, st).tier
  st.total = 2
  const t2 = recommendUcb({ type: 'qa' }, st).tier
  assert.deepEqual([t0, t1, t2].sort(), ['fast', 'mid', 'strong'])
})

test('UCB 利用：三臂都有样本时高均值臂胜出（未探索臂上界∞是合法探索）', () => {
  const st = {
    total: 30,
    arms: {
      'qa:fast': { n: 10, mean: 0.2 },
      'qa:mid': { n: 10, mean: 0.1 },
      'qa:strong': { n: 10, mean: 1 },
    },
  }
  assert.equal(recommendUcb({ type: 'qa' }, st).tier, 'strong')
  // 反之：strong 未探索时，UCB 应合法地选择探索它
  const st2 = {
    total: 20,
    arms: { 'qa:fast': { n: 10, mean: 0.9 }, 'qa:mid': { n: 10, mean: 0.8 } },
  }
  assert.equal(recommendUcb({ type: 'qa' }, st2).tier, 'strong')
})

test('resolveTier：mid 档无候选时向上取 strong（实测修复）', () => {
  assert.deepEqual(resolveTier('mid', 'deepseek-official', MODEL_TIERS),
    { target: 'deepseek-v4-pro', used: 'strong', note: 'mid->strong fallback' })
  assert.equal(resolveTier('fast', 'deepseek-official', MODEL_TIERS).target, 'deepseek-v4-flash')
  assert.equal(resolveTier('strong', 'deepseek-official', MODEL_TIERS).target, 'deepseek-v4-pro')
})

test('规则表阈值：低→fast，中→mid，高→strong', () => {
  assert.equal(recommendRule({ complexity: 0.1 }).tier, 'fast')
  assert.equal(recommendRule({ complexity: 0.5 }).tier, 'mid')
  assert.equal(recommendRule({ complexity: 0.8 }).tier, 'strong')
})

test('质量代理奖励：工具失败 −0.2/次（封顶 0.6），重试 −0.2/次（封顶 0.4），延迟分档', () => {
  assert.equal(rewardOf({ toolErrors: 1, retries: 0 }, 'clean-close', 1000).reward, 0.8)
  assert.equal(rewardOf({ toolErrors: 5, retries: 0 }, 'clean-close', 1000).reward, 0.4)
  assert.equal(rewardOf({ toolErrors: 0, retries: 4 }, 'clean-close', 1000).reward, 0.6)
  assert.equal(rewardOf({ toolErrors: 0, retries: 0 }, 'clean-close', 40000).reward, 0.9)
  assert.equal(rewardOf({ toolErrors: 0, retries: 0 }, 'error', 1000).reward, 0)
})

test('applyReward：更新 UCB/LinUCB/k-NN/EXP3 四个学习器', () => {
  const state = {
    ucb: { total: 0, arms: {} },
    lin: { total: 0, tierN: { fast: 0, mid: 0, strong: 0 }, tierW: { fast: new Array(9).fill(0), mid: new Array(9).fill(0), strong: new Array(9).fill(0) } },
    knn: { pool: [] },
    exp3: { total: 0, weights: { ...DEFAULT_WEIGHTS } },
  }
  const rec = {
    reward: 0.8, clsType: 'qa', ucbTier: 'fast', linTier: 'fast',
    f: [1, 0.1, 0.01, 0, 0, 0, 0, 0, 1],
    chosen: 'rule', pChosen: 0.7, pool: ['ucb', 'rule'],
  }
  applyReward(state, rec)
  assert.equal(state.ucb.arms['qa:fast'].n, 1)
  assert.equal(state.ucb.arms['qa:fast'].mean, 0.8)
  assert.equal(state.lin.tierN.fast, 1)
  assert.equal(state.knn.pool.length, 1)
  assert.ok(state.exp3.weights.rule > DEFAULT_WEIGHTS.rule, '被选策略权重应上升')
})

test('metaPick：冷启动只有 ucb+rule 入池，rule 初始权重 2.5 占优', () => {
  const recs = {
    rule: { tier: 'fast', conf: 0.5 },
    ucb: { tier: 'fast', conf: 0.2 },
    lin: { tier: 'fast', conf: 0.1 },
    knn: { tier: null, conf: 0 },
  }
  const exp3 = { weights: { ...DEFAULT_WEIGHTS }, linGraduated: false, knnGraduated: false }
  const m = metaPick(recs, exp3)
  assert.deepEqual(m.pool.sort(), ['rule', 'ucb'])
  assert.ok(Math.abs(m.ps.rule - 2.5 / 3.5) < 1e-9)
  assert.ok(m.byName[m.chosen])
})

test('k-NN 冷启动：经验不足时给 null 档', () => {
  const r = recommendKnn({ type: 'qa' }, [1, 0.1, 0.01, 0, 0, 0, 0, 0, 1], { pool: [] })
  assert.equal(r.tier, null)
  assert.equal(r.conf, 0)
})

test('k-NN 推荐：邻居经验按均值选档', () => {
  const pool = []
  const f = [1, 0.1, 0.01, 0, 0, 0, 0, 0, 1]
  for (let i = 0; i < 10; i++) {
    pool.push({ f: [1, 0.11, 0.01, 0, 0, 0, 0, 0, 1], tier: 'strong', r: 1 })
    pool.push({ f: [1, 0.12, 0.01, 0, 0, 0, 0, 0, 1], tier: 'fast', r: 0.3 })
  }
  const r = recommendKnn({ type: 'qa' }, f, { pool })
  assert.equal(r.tier, 'strong')
  assert.ok(r.conf > 0)
})

test('correctReward：反馈修正精确回退 UCB 均值与 EXP3 权重', () => {
  const state = {
    ucb: { total: 1, arms: {} },
    lin: { total: 1, tierN: { fast: 1, mid: 0, strong: 0 }, tierW: { fast: new Array(9).fill(0), mid: new Array(9).fill(0), strong: new Array(9).fill(0) } },
    knn: { pool: [] },
    exp3: { total: 1, weights: { ...DEFAULT_WEIGHTS } },
  }
  const rec = {
    reward: 1, clsType: 'qa', ucbTier: 'fast', linTier: 'fast',
    f: [1, 0.1, 0.01, 0, 0, 0, 0, 0, 1],
    chosen: 'rule', pChosen: 0.7, pool: ['ucb', 'rule'],
  }
  applyReward(state, rec)
  assert.equal(state.ucb.arms['qa:fast'].mean, 1)
  const wBefore = state.exp3.weights.rule

  correctReward(state, rec, 0.5)
  assert.equal(state.ucb.arms['qa:fast'].n, 1, 'n 不变')
  assert.equal(state.ucb.arms['qa:fast'].mean, 0.5, '均值修正为 0.5')
  assert.ok(state.exp3.weights.rule < wBefore, 'EXP3 权重应乘性回退')
  assert.ok(Math.abs(state.exp3.weights.rule - DEFAULT_WEIGHTS.rule * Math.exp(0.3 * (0.5 / 0.7) / 2)) < 1e-9, 'EXP3 回退为精确逆运算')
  assert.equal(rec.reward, 0.5, '记录同步更新')
})

test('correctReward：无变化时不修正', () => {
  const state = {
    ucb: { total: 1, arms: {} },
    lin: { total: 1, tierN: { fast: 1, mid: 0, strong: 0 }, tierW: { fast: new Array(9).fill(0), mid: new Array(9).fill(0), strong: new Array(9).fill(0) } },
    knn: { pool: [] },
    exp3: { total: 1, weights: { ...DEFAULT_WEIGHTS } },
  }
  const rec = { reward: 0.8, clsType: 'qa', ucbTier: 'fast', linTier: 'fast', f: [1, 0, 0, 0, 0, 0, 0, 0, 1], chosen: 'rule', pChosen: 0.7, pool: ['ucb', 'rule'] }
  applyReward(state, rec)
  const snapshot = JSON.stringify(state.ucb)
  correctReward(state, rec, 0.8)
  assert.equal(JSON.stringify(state.ucb), snapshot)
})
