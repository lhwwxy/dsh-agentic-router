// 飞轮持久化：决策/奖励追加 JSONL，策略（模式/强制模型）原子写 policy.json
// 默认目录：${DSH_HOME:-~/.dsh}/storages/dsh-agentic-router（可被 config.home 或
// AGENTIC_ROUTER_HOME 环境变量覆盖）。全部本地明文，可导出可审计。

import { mkdirSync, readFileSync, writeFileSync, renameSync, existsSync, appendFileSync, readdirSync, statSync, openSync, readSync, closeSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { homedir } from 'node:os'

export function defaultHome() {
  if (process.env.AGENTIC_ROUTER_HOME) return process.env.AGENTIC_ROUTER_HOME
  const base = process.env.DSH_HOME || join(homedir(), '.dsh')
  return join(base, 'storages', 'dsh-agentic-router')
}

export class Store {
  constructor(home) {
    this.home = home || defaultHome()
    mkdirSync(this.home, { recursive: true })
  }

  policyPath() { return join(this.home, 'policy.json') }
  jsonlPath(name) { return join(this.home, name) }

  loadPolicy() {
    try {
      if (!existsSync(this.policyPath())) return {}
      return JSON.parse(readFileSync(this.policyPath(), 'utf8'))
    } catch {
      return {}
    }
  }

  savePolicy(policy) {
    const p = this.policyPath()
    mkdirSync(dirname(p), { recursive: true })
    const tmp = p + '.tmp'
    writeFileSync(tmp, JSON.stringify(policy, null, 2))
    renameSync(tmp, p)
  }

  appendJsonl(name, obj) {
    appendFileSync(this.jsonlPath(name), JSON.stringify(obj) + '\n')
  }

  loadJsonl(name) {
    try {
      if (!existsSync(this.jsonlPath(name))) return []
      const out = []
      for (const line of readFileSync(this.jsonlPath(name), 'utf8').split('\n')) {
        if (!line.trim()) continue
        // 单行损坏只跳过该行，不丢整个历史（此前一行坏 = 全部丢，重放恢复会清零）
        try { out.push(JSON.parse(line)) } catch { /* skip corrupt line */ }
      }
      return out
    } catch {
      return []
    }
  }

  // 只读文件尾部（最多 64KB）解析最后 n 条：stats 等高频调用避免整文件解析
  loadJsonlTail(name, n = 20) {
    let fh = null
    try {
      const p = this.jsonlPath(name)
      if (!existsSync(p)) return []
      const size = statSync(p).size
      if (!size) return []
      const start = Math.max(0, size - 65536)
      const len = size - start
      const buf = Buffer.alloc(len)
      fh = openSync(p, 'r')
      readSync(fh, buf, 0, len, start)
      let text = buf.toString('utf8')
      if (start > 0) text = text.slice(text.indexOf('\n') + 1) // 丢弃可能被截断的首行
      const out = []
      for (const line of text.split('\n')) {
        if (!line.trim()) continue
        try { out.push(JSON.parse(line)) } catch { /* 截断行忽略 */ }
      }
      return out.slice(-n)
    } catch {
      return []
    } finally {
      if (fh != null) { try { closeSync(fh) } catch { /* best effort */ } }
    }
  }

  // 压实：只保留最后 keep 行（启动时调用；控制追加式日志的磁盘与重放开销）
  compactJsonl(name, keep) {
    try {
      const p = this.jsonlPath(name)
      if (!keep || !existsSync(p)) return
      const lines = readFileSync(p, 'utf8').split('\n').filter((l) => l.trim())
      if (lines.length <= keep) return
      const tmp = p + '.tmp'
      writeFileSync(tmp, lines.slice(-keep).join('\n') + '\n')
      renameSync(tmp, p)
    } catch { /* 尽力而为 */ }
  }

  // 清空飞轮（重置学习状态），保留 policy
  resetFlywheel() {
    for (const name of ['decisions.jsonl', 'rewards.jsonl']) {
      const p = this.jsonlPath(name)
      if (existsSync(p)) writeFileSync(p, '')
    }
  }

  // ---- 模型目录（运行时探测结果的本地缓存）----
  // 插件不在代码里写死任何模型清单：每次探测 llm.listProviders()/listModels() 后
  // 落到 models.json，重启后先读本地缓存，再按 TTL / 手动刷新重新探测。
  modelsPath() { return join(this.home, 'models.json') }

  loadModels() {
    try {
      if (!existsSync(this.modelsPath())) return null
      const c = JSON.parse(readFileSync(this.modelsPath(), 'utf8'))
      if (!c || !Array.isArray(c.providers) || !c.tiers) return null
      return c
    } catch {
      return null
    }
  }

  saveModels(catalog) {
    const p = this.modelsPath()
    mkdirSync(dirname(p), { recursive: true })
    const tmp = p + '.tmp'
    writeFileSync(tmp, JSON.stringify(catalog, null, 2))
    renameSync(tmp, p)
  }

  clearModels() {
    const p = this.modelsPath()
    if (existsSync(p)) writeFileSync(p, '')
  }

  listFiles() {
    try { return readdirSync(this.home) } catch { return [] }
  }
}
