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
- `rewards.jsonl` — one line per reward with breakdown (tool errors/retries/latency) + feature vector; **replayed on startup to restore learning state**
- `policy.json` — mode and forced model (atomic write)

Reward: clean close starts at 1; tool failures −0.2 each (cap 0.6), model retries −0.2 each (cap 0.4), latency >30s −0.1 / >90s −0.3; errored turns score 0.

## Known limits

- Model tiers are recognized by id keywords (flash→fast, pro→strong); real swaps happen only when more than one tier exists
- Rewards are proxies: a mediocre-but-successful answer still scores full marks, so wire up UI feedback (messageFeedback) for production use
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
