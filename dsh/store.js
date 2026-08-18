// 飞轮持久化：决策/奖励追加 JSONL，策略（模式/强制模型）原子写 policy.json
// 默认目录：${DSH_HOME:-~/.dsh}/storages/dsh-agentic-router（可被 config.home 或
// AGENTIC_ROUTER_HOME 环境变量覆盖）。全部本地明文，可导出可审计。

import { mkdirSync, readFileSync, writeFileSync, renameSync, existsSync, appendFileSync, readdirSync } from 'node:fs'
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
      return readFileSync(this.jsonlPath(name), 'utf8')
        .split('\n')
        .filter((l) => l.trim())
        .map((l) => JSON.parse(l))
    } catch {
      return []
    }
  }

  // 清空飞轮（重置学习状态），保留 policy
  resetFlywheel() {
    for (const name of ['decisions.jsonl', 'rewards.jsonl']) {
      const p = this.jsonlPath(name)
      if (existsSync(p)) writeFileSync(p, '')
    }
  }

  listFiles() {
    try { return readdirSync(this.home) } catch { return [] }
  }
}
