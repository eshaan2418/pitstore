"""pitstore CLI.

    pitstore demo     # build features three ways on Online Retail II and run the leakage check
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def cmd_demo(_: argparse.Namespace) -> int:
    import pandas as pd

    from .store import EventTable, Feature, check_point_in_time

    inv = pd.read_parquet("data/invoices.parquet")
    buys = inv[~inv.is_return]
    t0 = time.perf_counter()
    tb = EventTable(buys.customer.to_numpy(), buys.t.to_numpy(),
                    {"revenue": buys.revenue.to_numpy()})
    print(f"event table: {len(tb):,} purchases, built in {1e3 * (time.perf_counter() - t0):.0f} ms")
    day = 86_400
    feats = [Feature("revenue_90d", "sum", "revenue", window=90 * day, delay=day),
             Feature("n_purchases_365d", "count", window=365 * day, delay=day)]
    rng = np.random.default_rng(0)
    ent = rng.choice(buys.customer.unique(), 2000)
    t = np.full(2000, int(pd.Timestamp("2011-06-01").timestamp()))
    end = int(inv.t.max()) + day
    pipelines = {
        "point-in-time (as of query time)": lambda tb_, f, e, q: tb_.historical_features(f, e, q),
        "snapshot join (as of dataset end)": lambda tb_, f, e, q: tb_.historical_features(
            f, e, np.full_like(q, end)),
        "off-by-one-window (+30 days)": lambda tb_, f, e, q: tb_.historical_features(
            f, e, q + 30 * day),
    }
    for name, fn in pipelines.items():
        bad = check_point_in_time(tb, feats, ent, t, n_checks=100, compute=fn)
        verdict = "PASS" if not bad else f"FAIL ({len(bad)}/100 queries saw future data)"
        print(f"  {name:38s} {verdict}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pitstore")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
