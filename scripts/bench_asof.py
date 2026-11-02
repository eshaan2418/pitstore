"""As-of window aggregation throughput: pitstore vs pandas vs a Python loop (synthetic data).

The pandas baseline implements the same semantics with the standard idiom: per-entity cumulative
sums on the event table, then two ``merge_asof`` lookups (at t − delay and t − delay − window,
strictly before) and a difference. Results are checked for equality before timing.
Writes eval/results/bench_asof.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pitstore.store import EventTable, Feature

F = Feature("sum_w", "sum", "x", window=3_600, delay=60)


def make(n_events: int, n_queries: int, n_ent: int, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    e = rng.integers(0, n_ent, n_events)
    t = rng.integers(0, 30 * 86_400, n_events)
    x = rng.normal(size=n_events)
    qe = rng.integers(0, n_ent, n_queries)
    qt = rng.integers(0, 30 * 86_400, n_queries)
    return e, t, x, qe, qt


def pandas_asof(e, t, x, qe, qt) -> np.ndarray:  # type: ignore[no-untyped-def]
    ev = pd.DataFrame({"e": e, "t": t, "x": x}).sort_values(["t", "e"], kind="stable")
    ev["cs"] = ev.groupby("e")["x"].cumsum()
    ev = ev.sort_values("t", kind="stable")
    q = pd.DataFrame({"e": qe, "qt": qt, "i": np.arange(len(qe))})
    out = {}
    for name, shift in (("hi", F.delay), ("lo", F.delay + (F.window or 0))):
        qq = q.assign(key=q.qt - shift).sort_values("key", kind="stable")
        m = pd.merge_asof(qq, ev[["t", "e", "cs"]], left_on="key", right_on="t", by="e",
                          allow_exact_matches=False, direction="backward")
        out[name] = m.sort_values("i")["cs"].fillna(0.0).to_numpy()
    return out["hi"] - out["lo"]


def loop(e, t, x, qe, qt) -> np.ndarray:  # type: ignore[no-untyped-def]
    by: dict[int, list[tuple[int, float]]] = {}
    for ei, ti, xi in zip(e.tolist(), t.tolist(), x.tolist(), strict=True):
        by.setdefault(ei, []).append((ti, xi))
    res = []
    for ei, ti in zip(qe.tolist(), qt.tolist(), strict=True):
        hi, lo = ti - F.delay, ti - F.delay - (F.window or 0)
        res.append(sum(v for tt, v in by.get(ei, []) if lo <= tt < hi))
    return np.array(res)


def timeit(fn, reps: int = 3) -> float:  # type: ignore[no-untyped-def]
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rows = []
    for n_ev, n_q in ((10_000, 10_000), (100_000, 100_000), (1_000_000, 1_000_000),
                      (10_000_000, 1_000_000)):
        e, t, x, qe, qt = make(n_ev, n_q, max(100, n_ev // 100))
        t0 = time.perf_counter()
        tb = EventTable(e, t, {"x": x})
        build = time.perf_counter() - t0
        ours = tb.compute(F, qe, qt)
        ref = pandas_asof(e, t, x, qe, qt)
        np.testing.assert_allclose(ours, ref, atol=1e-6)
        row = {"events": n_ev, "queries": n_q, "pitstore_build_s": build,
               "pitstore_query_s": timeit(lambda tb=tb, qe=qe, qt=qt: tb.compute(F, qe, qt)),
               "pandas_s": timeit(lambda e=e, t=t, x=x, qe=qe, qt=qt: pandas_asof(e, t, x, qe, qt),
                                  reps=1)}
        if n_ev <= 10_000:
            lr = loop(e, t, x, qe, qt)
            np.testing.assert_allclose(ours, lr, atol=1e-6)
            row["python_loop_s"] = timeit(lambda e=e, t=t, x=x, qe=qe, qt=qt: loop(e, t, x, qe, qt),
                                          reps=1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/bench_asof.json").write_text(json.dumps({"rows": rows, "feature": {
        "agg": F.agg, "window": F.window, "delay": F.delay}}, indent=1))


if __name__ == "__main__":
    main()
