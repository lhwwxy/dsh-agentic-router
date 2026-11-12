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
