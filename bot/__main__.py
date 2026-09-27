"""Command line: python -m bot <command>

  run                      one trading cycle (the cloud runs this every 15 minutes)
  brain [--model M] [--source local|cloud] [--skip-if-fresh HOURS]
  backtest [--days N]      replay the last N days as the live bot would have traded
  report                   rebuild data/dashboard.json
  local                    the PC loop started at Windows login (brain + CLM + sync)
"""

import argparse

from . import config


def main():
    ap = argparse.ArgumentParser(prog="bot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    b = sub.add_parser("brain")
    b.add_argument("--model", default=config.LOCAL_BRAIN_MODEL)
    b.add_argument("--source", default="local")
    b.add_argument("--skip-if-fresh", type=float, default=0)
    bt = sub.add_parser("backtest")
    bt.add_argument("--days", type=int, default=180)
    sub.add_parser("report")
    sub.add_parser("local")
    args = ap.parse_args()

    if args.cmd == "run":
        from . import engine
        engine.run()
    elif args.cmd == "brain":
        from . import brain, report
        if args.skip_if_fresh and brain.fresh(args.skip_if_fresh):
            print(f"[brain] ran within the last {args.skip_if_fresh}h; skipping")
            return
        brain.run(args.model, args.source)
        report.write()
    elif args.cmd == "backtest":
        from . import backtest, report
        backtest.run(args.days)
        report.write()
    elif args.cmd == "report":
        from . import report
        report.write()
    elif args.cmd == "local":
        from . import local
        local.main()


if __name__ == "__main__":
    main()
