# AI Trading Bot (paper money)

A self-learning crypto trading system that runs 24/7 for free and trades **fake money** on **real prices**.

**Dashboard (Polish):** https://ericsonart.github.io/ai-trading-bot/

## How it works

| Part | What it does | Where it runs |
|---|---|---|
| **Fast models** (`bot/model.py`) | Gradient-boosted trees trained on 3 years of 15-minute candles for 12 coins, one model per trading style. Each predicts whether a trade ends in profit **after fees**. Retrained every 12 h, tested walk-forward on ~1.5 years of unseen data. | GitHub Actions, every 15 min |
| **Dream map** (`bot/evolution.py`) | After each retrain the bot replays the recorded history under hundreds of strategy recipes (style, confidence threshold, target/stop/holding time, position size, entry rules) without trading them. It mutates and combines the best, and records every recipe in a map. A new recipe goes live only if it also wins on the newest 20% of history, which the search never saw. | cloud |
| **Agent league** (`bot/engine.py`) | The champion and the best challengers each trade their own paper account live at the same time. Live losers can't become champion; a challenger that clearly beats the champion live takes over. | cloud |
| **Learning from mistakes + internet** (`bot/rules.py`, `bot/research.py`) | Losing signals are mined for common patterns; once a day the brain reads arXiv trading research, crypto news, Babypips lessons and the Fear & Greed index. Both become rule ideas that enter the dream map (never applied untested). | cloud + PC |
| **Paper broker** (`bot/broker.py`) | 0.1% fee + 0.05% slippage per side, stop-loss, take-profit, time limit. Training labels, dreams and the broker share one exit function (tested). | cloud |
| **Accounts** | *Main* trades only while the champion has a proven edge. *Practice* trades every champion signal. Losing coins are benched for a day; a -5% day pauses entries. | cloud |
| **Brain** (`bot/brain.py`, `bot/llm.py`) | An LLM reviews results and sets bounded risk directives + lessons (in Polish). Local `mimo2.6:9b` when the PC is on; free `qwen3:4b` in the cloud otherwise; swappable (below). | PC hourly / cloud every 4 h |
| **CLM second opinion** (`bot/clm.py`) | Your local Contrastive Language Model judges each coin; its accuracy is tracked (60-day test: no skill yet, so it has no power). | PC, every 4 h |

## Honest status

Short-term crypto prediction is hard. A 180-day replay (monthly retrain + evolution, no peeking at the future) ended
around break-even while BTC rose ~27%. The main account therefore waits until the champion proves an edge on unseen data.
Don't put real money on a strategy that hasn't proven itself on paper first.

## Change the fake money

GitHub → Actions → **change-fake-money** → *Run workflow* → type an amount. Both accounts restart with that much;
old results are archived in `data/archive/`. (Locally: `python -m bot reset --cash 5000`.)

## Use a stronger (frontier) model as the brain

- **Cloud:** in the repo settings add a variable `BRAIN_PROVIDER` = `anthropic` (or `openai`), a variable `BRAIN_MODEL`
  (e.g. `claude-sonnet-5`), and the secret `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`; add `OPENAI_BASE_URL` for
  OpenRouter/Groq/etc.).
- **PC:** create `.runner/.env` with the same lines, e.g. `BRAIN_PROVIDER=anthropic`, `BRAIN_MODEL=claude-sonnet-5`,
  `ANTHROPIC_API_KEY=...`. The file is never committed.

## Real money (later, maybe)

Paper profits can't be withdrawn. The dashboard's **real-money readiness** checklist turns green only after 90+ days,
100+ trades, profit after fees, beating BTC buy-and-hold and a drawdown under 20%. Order execution is isolated in
`bot/broker.py`, so a live exchange adapter (trade-only API key, no withdrawals, small amount, kill switch) can be added
then, and tested with tiny sums first. It is deliberately not wired up yet.

## Commands

```
python -m bot run          # one trading cycle
python -m bot brain        # LLM review
python -m bot backtest     # replay the last 180 days honestly (with evolution)
python -m bot reset --cash 5000
python -m pytest -q        # tests
```

PC side (Task Scheduler, starts hidden at every login): `powershell -ExecutionPolicy Bypass -File local\install.ps1`
(remove with `local\uninstall.ps1`). Log: `.runner\.cache\local.log`.
