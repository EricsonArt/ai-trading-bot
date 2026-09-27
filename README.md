# AI Trading Bot (paper money)

A self-learning crypto trading system that runs 24/7 for free and trades **fake money** on **real prices**.

**Dashboard:** https://ericsonart.github.io/ai-trading-bot/

## How it works

| Part | What it does | Where it runs |
|---|---|---|
| **Fast model** (`bot/model.py`) | Gradient-boosted trees trained on 3 years of 15-minute candles (BTC, ETH, SOL, XRP, DOGE, ADA). Predicts whether a trade will end in profit **after fees**. Retrains every 12 h, tested walk-forward on ~1.5 years of unseen data, and only counts as having an *edge* if its picks beat blindly buying every time. | GitHub Actions, every 15 min |
| **Paper broker** (`bot/broker.py`) | $1,000 per account, 0.1% fee + 0.05% slippage per side, stop-loss, take-profit, time limit. Training labels replay these exact rules (tested). | cloud |
| **Two accounts** (`bot/engine.py`) | *Main* trades only while the edge is proven. *Practice* trades every signal so the model's real skill is measured live. Coins that keep losing get smaller positions or are benched for a day; a -5% day pauses new entries. | cloud |
| **Brain** (`bot/brain.py`) | An LLM reads a report (both accounts, trades, model quality, market, its own past lessons and how its last advice worked out) and sets bounded risk directives + writes lessons to memory. Local `mimo2.6:9b` via Ollama when the PC is on; free `qwen3:4b` on the cloud runner otherwise. | PC hourly / cloud every 4 h |
| **Learning loop** (`bot/rules.py`, `bot/research.py`) | After each retrain the bot mines its losing out-of-sample signals for common patterns; once a day the brain reads arXiv trading research, crypto news, Babypips lessons and the Fear & Greed index. Both produce candidate filter rules. A rule is adopted only if it raises profit per trade on unseen data (whole period *and* the most recent third), is re-tested after every retrain, and retired when it stops helping. | cloud judges; research on PC daily (cloud fallback) |
| **CLM second opinion** (`bot/clm.py`) | Your local Contrastive Language Model judges each coin from a plain-language market description. Every call is checked 2 days later, so its accuracy is known before anyone trusts it. | PC, every 4 h |
| **Dashboard** (`dashboard/`) | Static page on GitHub Pages; loads `data/dashboard.json` from this repo every minute. | GitHub Pages |

State lives in `data/` (committed by the bot); price history and the trained model live in the Actions cache (`.cache/`).

## Honest status

Short-term crypto prediction is hard: after fees, the model has **not** shown a reliable edge on unseen data yet
(replay of the practice account over the last 180 days: -13% while BTC rose +26%). That is exactly why the
main account waits in cash. Don't put real money on a strategy that hasn't proven itself on paper first.

## Commands

```
python -m bot run          # one trading cycle
python -m bot brain        # LLM review (needs Ollama)
python -m bot backtest     # replay the last 180 days honestly
python -m pytest -q        # tests
```

PC side (Task Scheduler, starts hidden at every login): `powershell -ExecutionPolicy Bypass -File local\install.ps1`
(remove with `local\uninstall.ps1`). Log: `.runner\.cache\local.log`.
