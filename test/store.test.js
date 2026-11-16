// 存储层测试：损坏行容错 / 尾部读取 / JSONL 压实
import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, appendFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Store } from '../dsh/store.js'

function tmpHome() { return mkdtempSync(join(tmpdir(), 'dsh-router-store-')) }

test('loadJsonl：单行损坏只跳过该行，不丢整个历史（重放恢复不受影响）', () => {
  const s = new Store(tmpHome())
  s.appendJsonl('x.jsonl', { a: 1 })
  appendFileSync(s.jsonlPath('x.jsonl'), '{corrupt\n')
  s.appendJsonl('x.jsonl', { a: 2 })
  assert.deepEqual(s.loadJsonl('x.jsonl').map((r) => r.a), [1, 2])
})

test('loadJsonlTail：只解析尾部，等价于全量加载的最后 n 条', () => {
  const s = new Store(tmpHome())
  for (let i = 0; i < 500; i++) s.appendJsonl('big.jsonl', { i, pad: 'x'.repeat(64) })
  const full = s.loadJsonl('big.jsonl')
  assert.deepEqual(s.loadJsonlTail('big.jsonl', 20), full.slice(-20))
  assert.deepEqual(s.loadJsonlTail('big.jsonl', 3), full.slice(-3))
  assert.equal(s.loadJsonlTail('big.jsonl', 20).length, 20)
  // 空文件 / 不存在文件
  assert.deepEqual(s.loadJsonlTail('none.jsonl', 5), [])
})

test('compactJsonl：超过 keep 行时只保留最后 keep 行；未超限不动', () => {
  const s = new Store(tmpHome())
  for (let i = 0; i < 30; i++) s.appendJsonl('y.jsonl', { i })
  s.compactJsonl('y.jsonl', 10)
  const rows = s.loadJsonl('y.jsonl')
  assert.equal(rows.length, 10)
  assert.equal(rows[0].i, 20)
  assert.equal(rows[rows.length - 1].i, 29)
  s.compactJsonl('y.jsonl', 100)
  assert.equal(s.loadJsonl('y.jsonl').length, 10, '未超限时不应改动')
})
