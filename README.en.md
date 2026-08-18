# dsh-agentic-router

> 🌐 Language / 语言切换: **[English](./README.en.md)** · **[简体中文](./README.md)**

DeepSeek Harness (DSH) plugin: a **learning agentic router with a data flywheel**.

It watches every model request, routes it to the most appropriate model tier by task type and complexity, asks four experts in parallel and lets an EXP3 meta-selector decide who to trust. Every turn is scored with **quality-proxy signals** (tool failures, model retries, latency) and the reward feeds back into the strategies — the router gets better as it runs. Every decision and reward is persisted locally: auditable and replayable.

## Architecture

```
user input → classify (rules, <1ms) → 4 expert recommendations → EXP3 meta-pick → agent/request swap
                                           ↑                              │
              UCB/LinUCB/k-NN/EXP3 updates ← quality-proxy reward ← turn end (tool errors/retries/latency)
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
fall back to length-based estimates (flagged `estimated`). Default price table follows
DeepSeek's official pricing (CNY per million tokens), overridable:

```js
// config.prices / config.costWeight / config.costCap
{ fast: { in: 0.2, out: 2 }, mid: { in: 0.5, out: 6 }, strong: { in: 1, out: 12 } }
```

## Known limits

- Model tiers are recognized by id keywords (flash→fast, pro→strong); real swaps happen only when more than one tier exists
- Feedback settles after a 45s window (configurable via `settleDelayMs`); unsettled pending entries are lost on restart; late feedback may be attributed to a later turn
- Feedback only works where the deployment's Web UI actually writes it (this plugin only reads)
- LinUCB / k-NN need dozens of turns before they graduate into the meta pool

## Development

```sh
npm test        # node --test, zero dependencies
```

## Publish

```sh
npm publish                                     # requires npm account
git push origin main                            # after making the GitHub repo public
# directory listing: add the dsh-plugin topic, or submit at dshmarketplace.dev/submit
```

## License

MIT
