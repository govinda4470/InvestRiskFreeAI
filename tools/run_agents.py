#!/usr/bin/env python3
"""Run one or more paper auto-trade cycles for all registered users.

Examples:
    python tools/run_agents.py                 # one offline/demo cycle
    python tools/run_agents.py --live          # one delayed-yfinance cycle
    python tools/run_agents.py --loop --interval 300 --live

Use the loop only on an always-on machine with process supervision. Streamlit
Community Cloud does not guarantee long-running background processes.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investriskfree.auth import AuthStore  # noqa: E402
from investriskfree.autotrade import AutoTradeAgent  # noqa: E402
from investriskfree.paper import PaperBroker  # noqa: E402


def run_all(live_data: bool = False) -> tuple[int, int]:
    users = AuthStore().active_users()
    armed = 0
    failures = 0
    for user in users:
        broker = PaperBroker(user_id=user["id"])
        agent = AutoTradeAgent(broker)
        config = agent.config()
        if not config["enabled"]:
            continue
        armed += 1
        try:
            kronos_gate = None
            if config.get("require_kronos"):
                from investriskfree.data.loader import load_daily
                from investriskfree.kronos_forecast import KronosForecastService

                service = KronosForecastService()

                def kronos_gate(signal):
                    history = load_daily(
                        signal["symbol"], source="yfinance" if live_data else None
                    )
                    return service.long_signal_gate(signal, history)

            result = agent.run_once(live_data=live_data, kronos_gate=kronos_gate)
            print(
                f"[{user['username']}] {result.get('message')} "
                f"(run {result.get('run_id', '?')[:8]})",
                flush=True,
            )
            failures += int(not result.get("ok"))
        except Exception as exc:
            failures += 1
            print(f"[{user['username']}] ERROR: {exc}", file=sys.stderr, flush=True)
    print(f"Processed {armed} armed user(s); failures={failures}", flush=True)
    return armed, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="require current yfinance candles/quotes")
    parser.add_argument("--loop", action="store_true", help="keep running until terminated")
    parser.add_argument("--interval", type=int, default=300, help="seconds between cycles (minimum 60)")
    args = parser.parse_args()
    interval = max(60, int(args.interval))
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        _, failures = run_all(live_data=args.live)
        if not args.loop or stopping:
            return 1 if failures else 0
        deadline = time.monotonic() + interval
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1, deadline - time.monotonic()))
        if stopping:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
