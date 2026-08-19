# dsh-agentic-router

> 🌐 Language / 语言切换: **[English](./README.en.md)** · **[简体中文](./README.md)**

DeepSeek Harness (DSH) plugin: a **learning agentic router with a data flywheel**.

At the start of every turn (`agent/inbox/claimed`) it routes the request to the most appropriate model tier by task type and complexity, asks four experts in parallel and lets an EXP3 meta-selector decide who to trust. Every turn is scored with **quality-proxy signals** (tool failures, model retries, cost, latency) and the reward feeds back into the strategies — the router gets better as it runs. Every decision and reward is persisted locally: auditable and replayable.

**Real switching since v1.4.0**: v1.3.0 and earlier swapped models by returning a replaced config from the `agent/request` waterfall, but DSH's built-in model-selection listener overrides that value, so the swap never took effect. v1.4.0 instead appends a `request/header` event (reason: `router`) to the session log at turn start, going through DSH's official model-selection precedence chain (in-process selection → latest logged session `request/header` → default model).

## Architecture

```
user input → classify (rules, <1ms) → 4 expert recommendations → EXP3 meta-pick → request/header append (real switch)
                                            ↑                              │
              UCB/LinUCB/k-NN/EXP3 updates ← quality-proxy reward ← turn end (tool errors/retries/cost/latency)
```

| Expert | Algorithm | Status |
|---|---|---|
| rule | deterministic thresholds (complexity ≤0.3→fast, ≥0.7→strong, else mid) | cold-start prior, always in pool |
| UCB | cluster×tier multi-armed bandit; unexplored arms get ∞, ties rotate | active from day zero |
| LinUCB | linear feature scoring + exploration bonus, gradient updates | shadow; graduates at 50 samples |
| k-NN | k=5 nearest-neighbor aggregation over the experience pool | shadow; graduates at 30 experiences |
| EXP3 | meta-selector, importance-weighted updates | always on; cold-start biased to rule (weight 2.5) |

Design rationale and roadmap: [docs/DESIGN.md](./docs/DESIGN.md). The implementation was validated against a live runtime prototype first; eight defects found there are fixed and pinned as regression tests.

## Install

```sh
dsh plugin --profile web add dsh-agentic-router
```

> `--profile` is mandatory. Restart the session (or profile) after installing so the tool schemas join prompt assembly. Default mode is **shadow** (records and learns, never swaps); switch to active once the policy behaves.

## Tools

| Tool | Purpose |
|---|---|
| `agentic_router_stats` | decision log, four-expert recommendations, EXP3 weights, flywheel state |
| `agentic_router_set_mode` | shadow (default) / active (real swaps) / off (bypass) |
| `agentic_router_force` | force a model id (active mode only); clear=true to release |
| `agentic_router_reset` | wipe flywheel learning state, keep mode |
| `agentic_router_set_prices` | set/query per-provider per-model (or per-tier) prices |

## Data flywheel (persistence)

Directory: `${DSH_HOME:-~/.dsh}/storages/dsh-agentic-router/` (override with `AGENTIC_ROUTER_HOME`)

- `decisions.jsonl` — one line per decision: task type/complexity, all four recommendations, meta pick, actual route, swapped?
- `rewards.jsonl` — one line per reward with breakdown (tool errors/retries/latency) + feature vector; **replayed on startup to restore learning state**; human-feedback settlements (`feedback-settle` lines) correct the replayed rewards exactly
- Human feedback: 45s after a turn closes, the plugin polls `messageFeedback` (the Web UI 👍/👎); a thumbs-down costs −0.5, a thumbs-up adds +0.1, applied as an exact correction to the already-applied reward
- `policy.json` — mode and forced model (atomic write)

Reward: clean close starts at 1; tool failures −0.2 each (cap 0.6), model retries −0.2 each (cap 0.4),
**cost** min(0.5, 20 × turn cost in CNY), latency >30s −0.1 / >90s −0.3; errored turns score 0.

**Cost signal (token billing)**: the plugin wraps `llm/stream` and accumulates each model
call's `usage` chunk (input/output/cache-read/reasoning tokens); providers without usage
fall back to length-based estimates (flagged `estimated`).

**Three-level price resolution** (CNY per million tokens): `provider→model` exact price >
`provider→tier` price > `*` generic tier > conservative fallback. **Unknown providers
never inherit DeepSeek pricing** — they use the generic tier and carry a `pricingSource`
audit field. DeepSeek's official pricing ships as default; add other models at runtime via
the `agentic_router_set_prices` tool (persisted to policy.json) or at install via
`config.prices`:

```js
{ 'deepseek-official': { 'deepseek-v4-pro': { in: 1, out: 12 } },
  openai: { 'gpt-x-pro': { in: 15, out: 60 } },
  '*': { fast: { in: 1, out: 4 }, mid: { in: 2, out: 8 }, strong: { in: 3, out: 16 } } }
```

## Known limits

- Model tiers are recognized by id keywords (flash→fast, pro→strong); real swaps happen only when more than one tier exists
- Feedback settles after a 45s window (configurable via `settleDelayMs`); unsettled pending entries are lost on restart; late feedback may be attributed to a later turn
- Feedback only works where the deployment's Web UI actually writes it (this plugin only reads)
- LinUCB / k-NN need dozens of turns before they graduate into the meta pool

**Development, testing and release flow (maintainers): see [docs/MAINTAINING.md](./docs/MAINTAINING.md).**

## License

MIT
