// 分类器回归测试：全部用例来自动态原型实测（含踩过的坑）
import test from 'node:test'
import assert from 'node:assert/strict'
import { classify, featuresOf } from '../dsh/classify.js'

test('qa 祈使句："请只用一句话回答…" → qa（修复前误判 agentic）', () => {
  const r = classify('请只用一句话回答：地球的天然卫星叫什么？（不要任何解释，只回答名称）')
  assert.equal(r.type, 'qa')
  assert.ok(r.complexity < 0.1, '极简问答复杂度应接近 0，实际 ' + r.complexity)
})

test('复杂代码（含代码块+重构）→ code 且复杂度 ≥ 0.7', () => {
  const r = classify('重构下面这段 Python 代码，使其支持多线程并发执行并安全处理异常：\n\n```python\nimport time\ndef fetch(url):\n    time.sleep(1)\n```')
  assert.equal(r.type, 'code')
  assert.ok(r.complexity >= 0.7, '复杂度应过 strong 阈值，实际 ' + r.complexity)
})

test('中等 agentic 设计任务 → agentic 且 0.3 < c < 0.7', () => {
  const r = classify('帮我设计一个博客系统的 MySQL 数据库 schema：需要用户、文章、评论、标签四个实体及它们的关联表。要求：给出完整建表 SQL（含主外键与索引），并解释其中 2 个关键索引的设计理由。')
  assert.equal(r.type, 'agentic')
  assert.ok(r.complexity > 0.3 && r.complexity < 0.7, '实际 ' + r.complexity)
})

test('中等写作任务 → write 且有类型先验分（修复前 c≈0.01）', () => {
  const r = classify('写一篇约 150 字的短文，向完全不懂技术的读者解释什么是 API，要求通俗易懂，并使用一个生活化的比喻。')
  assert.equal(r.type, 'write')
  assert.ok(r.complexity >= 0.1 && r.complexity <= 0.3, '实际 ' + r.complexity)
})

test('简单总结 → summarize 低复杂度', () => {
  const r = classify('用不超过两句话总结下面这段文字：机器学习模型在训练数据上表现好但在新数据上表现差，被称为过拟合。')
  assert.equal(r.type, 'summarize')
  assert.ok(r.complexity <= 0.3)
})

test('"从简单到复杂"同时出现时不触发简单惩罚（实测踩坑）', () => {
  const r = classify('你自己设计几轮问题，覆盖从简单到复杂的问题，跑一遍，然后看下决策情况是否符合逻辑，是否正常。')
  assert.ok(r.complexity > 0.2, '复杂关键词在场时简单词不应减分，实际 ' + r.complexity)
})

test('问候语 → other，复杂度 0', () => {
  const r = classify('你好')
  assert.equal(r.type, 'other')
  assert.equal(r.complexity, 0)
})

test('featuresOf 返回 9 维向量', () => {
  const cls = classify('帮我写一个 Python 脚本处理数据')
  const f = featuresOf(cls, '帮我写一个 Python 脚本处理数据')
  assert.equal(f.length, 9)
  assert.equal(f[0], 1)
  assert.ok(f[1] >= 0 && f[1] <= 1)
})
