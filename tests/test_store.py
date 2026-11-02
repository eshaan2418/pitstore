import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pitstore.store import EventTable, Feature, OnlineStore, check_point_in_time

FEATS = [
    Feature("n_all", "count"),
    Feature("n_10", "count", window=10),
    Feature("sum_10_d2", "sum", "x", window=10, delay=2),
    Feature("mean_all", "mean", "x"),
    Feature("last_x", "last", "x", delay=1),
    Feature("since", "time_since_last", window=50),
]


def brute(entity, time, x, f, qe, qt):
    out = []
    for e, t in zip(qe, qt, strict=True):
        hi = t - f.delay
        lo = -np.inf if f.window is None else hi - f.window
        m = (entity == e) & (time >= lo) & (time < hi)
        xs, ts = x[m], time[m]
        order = np.argsort(ts, kind="stable")
        xs, ts = xs[order], ts[order]
        if f.agg == "count":
            out.append(float(m.sum()))
        elif f.agg == "sum":
            out.append(float(xs.sum()))
        elif f.agg == "mean":
            out.append(float(xs.mean()) if m.any() else np.nan)
        elif f.agg == "last":
            out.append(float(xs[-1]) if m.any() else np.nan)
        else:
            out.append(float(t - ts[-1]) if m.any() else np.nan)
    return np.array(out)


events = st.integers(1, 60).flatmap(lambda n: st.tuples(
    st.lists(st.integers(0, 4), min_size=n, max_size=n),
    st.lists(st.integers(0, 100), min_size=n, max_size=n),
    st.lists(st.floats(-10, 10, allow_nan=False), min_size=n, max_size=n)))


@settings(max_examples=150, deadline=None)
@given(events, st.lists(st.tuples(st.integers(0, 6), st.integers(-5, 130)), min_size=1,
                        max_size=20))
def test_offline_matches_bruteforce(ev, queries):
    e, t, x = (np.array(v) for v in ev)
    tb = EventTable(e, t, {"x": x})
    qe = np.array([q[0] for q in queries])
    qt = np.array([q[1] for q in queries])
    got = tb.historical_features(FEATS, qe, qt)
    for f in FEATS:
        np.testing.assert_allclose(got[f.name], brute(e, t, x, f, qe, qt), rtol=1e-9,
                                   atol=1e-9, err_msg=f.name)


@settings(max_examples=60, deadline=None)
@given(events)
def test_online_matches_offline(ev):
    e, t, x = (np.array(v) for v in ev)
    tb = EventTable(e, t, {"x": x})
    online = OnlineStore(FEATS)
    order = np.lexsort((np.arange(len(t)), t))
    for i in order:
        # serve every entity right before ingesting this event
        for ent in range(5):
            on = online.get(ent, int(t[i]))
            off = tb.historical_features(FEATS, np.array([ent]), np.array([t[i]]))
            for f in FEATS:
                a, b = on[f.name], off[f.name][0]
                assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, f.name
        online.ingest(int(e[i]), int(t[i]), {"x": float(x[i])})


def test_leakage_check_passes_correct_and_catches_leaky():
    rng = np.random.default_rng(0)
    n = 2000
    e, t, x = rng.integers(0, 50, n), rng.integers(0, 10_000, n), rng.normal(size=n)
    tb = EventTable(e, t, {"x": x})
    qe, qt = rng.integers(0, 50, 300), rng.integers(1000, 10_000, 300)
    assert check_point_in_time(tb, FEATS, qe, qt, n_checks=100) == []

    def leaky(table, feats, ent, tt):  # off-by-delay: includes events up to t + 30
        return table.historical_features(
            [Feature("leak", "sum", "x", window=100, delay=-30)], ent, tt)

    bad = check_point_in_time(tb, [Feature("leak", "sum", "x", window=100)], qe, qt,
                              n_checks=100, compute=leaky)
    # exactly the sampled queries whose entity has an event in the leaked span [t, t + 30)
    picks = np.random.default_rng(0).choice(len(qe), 100, replace=False)
    expected = {int(i) for i in picks
                if ((tb.entity == qe[i]) & (tb.time >= qt[i]) & (tb.time < qt[i] + 30)).any()}
    assert set(bad) == expected and len(expected) > 0


def test_unknown_entity_and_empty_windows():
    tb = EventTable(np.array([1, 1]), np.array([5, 7]), {"x": np.array([1.0, 2.0])})
    out = tb.historical_features(FEATS, np.array([99, 1, 1]), np.array([10, 5, 6]))
    assert out["n_all"].tolist() == [0.0, 0.0, 1.0]
    assert np.isnan(out["mean_all"][0]) and np.isnan(out["since"][1])
