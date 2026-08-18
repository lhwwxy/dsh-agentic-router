// 分类器：任务类型 + 复杂度（确定性规则，<1ms）
// 所有规则都来自动态原型实测，回归用例见 test/classify.test.js

const QA_IMPERATIVE = /(请|麻烦|帮我).{0,6}(回答|告诉|说)/
const CODE_RE = /def |function |class |import |require\(|```|\.py|\.ts|\.js|\.go|\.rs\b|bug|error|exception|traceback|报错|代码|函数|接口|脚本/
const WRITE_RE = /(写|创作|起草|润色|翻译|translate|write|blog|article|文案|邮件|报告)/
const SUMMARIZE_RE = /(总结|概括|summar|摘要|要点)/
const RESEARCH_RE = /(查|搜索|检索|research|调研|对比|最新|资料|文档)/
const AGENTIC_RE = /(帮我|请|做|搭建|构建|部署|实现|写一个|生成|build|setup|deploy|implement|create)/
const QA_RE = /[?？]|什么|为什么|怎么|如何|how|what|why|能否|是否/
const COMPLEX_RE = /(重构|优化|设计|分析|架构|debug|调试|排查|review|方案|复杂|多步|逐步|implement|design|architecture)/
const SIMPLE_RE = /(一句|简单|快速|quickly|short|hello|你好|谢谢)/

// 任务类型基础复杂度先验（实测调参）：代码与 agentic 任务天然更重
const PRIOR = { code: 0.3, agentic: 0.3, research: 0.2, write: 0.15, summarize: 0.05, qa: 0, other: 0 }

export function classify(text) {
  const t = text || ''
  const lower = t.toLowerCase()
  let type = 'other'
  if (QA_IMPERATIVE.test(lower)) type = 'qa'
  else if (CODE_RE.test(lower)) type = 'code'
  else if (WRITE_RE.test(lower)) type = 'write'
  else if (SUMMARIZE_RE.test(lower)) type = 'summarize'
  else if (RESEARCH_RE.test(lower)) type = 'research'
  else if (AGENTIC_RE.test(lower)) type = 'agentic'
  else if (QA_RE.test(lower)) type = 'qa'

  const complexKw = COMPLEX_RE.test(lower)
  const simpleKw = SIMPLE_RE.test(lower)
  let score = PRIOR[type] || 0
  score += Math.min(t.length / 2000, 1) * 0.2
  score += (t.match(/```/g) || []).length > 0 ? 0.25 : 0
  if (complexKw) score += 0.25
  // 同时出现"简单"与"复杂"关键词时不做简单惩罚（实测踩过的坑）
  if (simpleKw && !complexKw) score -= 0.25
  score = Math.max(0, Math.min(1, score))
  return { type, complexity: Math.round(score * 100) / 100 }
}

const TYPES = ['code', 'write', 'summarize', 'research', 'agentic', 'qa']

// LinUCB / k-NN 共享的特征向量：[1, 复杂度, 长度归一, 类型 one-hot ×6]
export function featuresOf(cls, text) {
  const t = text || ''
  const f = [1, cls.complexity, Math.min(t.length, 4000) / 4000]
  for (const ty of TYPES) f.push(cls.type === ty ? 1 : 0)
  return f
}
