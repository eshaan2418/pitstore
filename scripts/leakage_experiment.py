"""How much does feature leakage inflate offline metrics? (Online Retail II, repurchase task)

Task: on the first day of each month r (2010-03 ... 2011-11), for every customer with at least
one purchase before r, predict whether they buy again within [r, r + 30 days).

Feature pipelines (same feature definitions, different as-of timestamps):
* correct      features as of r (1-day ingestion delay) -- what a production system can serve;
* snapshot     features as of the end of the dataset, joined to every row ("latest feature
               table" join): leaks everything after r;
* window_leak  features as of r + 30 days (off-by-one-window timestamp bug): leaks the label
               window itself.

Split by time: train r <= 2011-04, validation 2011-05..06 (early stopping), test 2011-07..11.
For each pipeline we report (a) the test AUC an analyst would see offline (model trained and
evaluated on that pipeline's features), and (b) the *deployed* AUC: the same model evaluated on
the correct features that production actually serves. 3 LightGBM seeds; 95% CIs from a
customer-clustered bootstrap over test rows. The future-perturbation check is run on each
pipeline. Writes eval/results/leakage.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pitstore.store import EventTable, Feature, check_point_in_time

DAY = 86_400
DELAY = DAY
FEATURES = [
    Feature("n_purchases_30d", "count", window=30 * DAY, delay=DELAY),
    Feature("n_purchases_90d", "count", window=90 * DAY, delay=DELAY),
    Feature("n_purchases_365d", "count", window=365 * DAY, delay=DELAY),
    Feature("revenue_30d", "sum", "revenue", window=30 * DAY, delay=DELAY),
    Feature("revenue_90d", "sum", "revenue", window=90 * DAY, delay=DELAY),
    Feature("revenue_365d", "sum", "revenue", window=365 * DAY, delay=DELAY),
    Feature("basket_mean_365d", "mean", "revenue", window=365 * DAY, delay=DELAY),
    Feature("lines_90d", "sum", "lines", window=90 * DAY, delay=DELAY),
    Feature("days_since_last", "time_since_last", delay=DELAY),
]
RETURN_FEATURES = [Feature("n_returns_365d", "count", window=365 * DAY, delay=DELAY)]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=4,
              verbose=-1)


def build(inv: pd.DataFrame) -> tuple[EventTable, EventTable, pd.DataFrame]:
    buys = inv[~inv.is_return]
    rets = inv[inv.is_return]
    tb = EventTable(buys.customer.to_numpy(), buys.t.to_numpy(),
                    {"revenue": buys.revenue.to_numpy(), "lines": buys.lines.to_numpy()})
    tr = EventTable(rets.customer.to_numpy(), rets.t.to_numpy(), {})
    refs = pd.date_range("2010-03-01", "2011-11-01", freq="MS")
    rows = []
    first = buys.groupby("customer").t.min()
    for r in refs:
        rt = int((r - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
        cust = first.index[first.to_numpy() < rt].to_numpy()
        rows.append(pd.DataFrame({"customer": cust, "ref": r, "t": rt}))
    lab = pd.concat(rows, ignore_index=True)
    # label: any purchase in [r, r + 30d): count over a 30-day window ending at r + 30d
    y = tb.compute(Feature("y", "count", window=30 * DAY), lab.customer.to_numpy(),
                   lab.t.to_numpy() + 30 * DAY)
    lab["y"] = (y > 0).astype(int)
    return tb, tr, lab


def features_at(tb: EventTable, tr: EventTable, ent: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    f = tb.historical_features(FEATURES, ent, t)
    f.update(tr.historical_features(RETURN_FEATURES, ent, t))
    return pd.DataFrame(f)


def cluster_bootstrap_auc(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n: int = 1000,
                          seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    inv = pd.read_parquet("data/invoices.parquet")
    tb, tr, lab = build(inv)
    ent, t = lab.customer.to_numpy(), lab.t.to_numpy()
    t_end = int(inv.t.max()) + DAY
    pipelines = {
        "correct": t,
        "snapshot": np.full_like(t, t_end),
        "window_leak": t + 30 * DAY,
    }
    X = {k: features_at(tb, tr, ent, tt) for k, tt in pipelines.items()}
    split = np.where(lab.ref <= "2011-04-01", "train",
                     np.where(lab.ref <= "2011-06-01", "val", "test"))
    out: dict = {"n_rows": len(lab), "positive_rate": float(lab.y.mean()),
                 "rows_per_split": {s: int((split == s).sum()) for s in ("train", "val", "test")},
                 "features": [f.name for f in FEATURES + RETURN_FEATURES], "pipelines": {},
                 "git_commit": commit}
    y = lab.y.to_numpy()
    te = split == "test"
    for name, tt in pipelines.items():
        # leakage check: compute features for the pipeline's timestamp, judged against r
        offset = tt - t

        def compute(table, feats, e, q, offset=offset, t=t):  # type: ignore[no-untyped-def]
            # map each checked query back to its pipeline timestamp
            idx = [int(np.nonzero((ent == ei) & (t == qi))[0][0]) for ei, qi in zip(e, q,
                                                                                    strict=True)]
            return table.historical_features(feats, e, q + offset[idx])

        viol = check_point_in_time(tb, FEATURES, ent, t, n_checks=200, compute=compute)
        rec: dict = {"leak_check_violations": len(viol), "leak_check_n": 200, "seeds": []}
        for seed in (0, 1, 2):
            Xtr, Xva = X[name][split == "train"], X[name][split == "val"]
            m = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(Xtr, y[split == "train"]),
                          2000, valid_sets=[lgb.Dataset(Xva, y[split == "val"])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            p_off = m.predict(X[name][te], num_iteration=m.best_iteration)
            p_dep = m.predict(X["correct"][te], num_iteration=m.best_iteration)
            g = ent[te]
            rec["seeds"].append({
                "seed": seed, "best_iteration": int(m.best_iteration),
                "offline_test_auc": float(roc_auc_score(y[te], p_off)),
                "offline_ci": cluster_bootstrap_auc(y[te], p_off, g),
                "deployed_test_auc": float(roc_auc_score(y[te], p_dep)),
                "deployed_ci": cluster_bootstrap_auc(y[te], p_dep, g),
                "val_auc": float(roc_auc_score(y[split == "val"], m.predict(
                    Xva, num_iteration=m.best_iteration))),
                "importance": dict(zip(X[name].columns, m.feature_importance("gain").tolist(),
                                       strict=True)),
            })
            print(name, seed, {k: v for k, v in rec["seeds"][-1].items()
                               if k.endswith("auc")}, flush=True)
        out["pipelines"][name] = rec
        print(name, "leak-check violations:", len(viol), flush=True)
    # random-split offline estimate for the snapshot pipeline (a common evaluation mistake)
    rng = np.random.default_rng(0)
    mask = rng.random(len(lab)) < 0.8
    aucs = []
    for seed in (0, 1, 2):
        m = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X["snapshot"][mask], y[mask]), 300)
        aucs.append(float(roc_auc_score(y[~mask], m.predict(X["snapshot"][~mask]))))
    out["snapshot_random_split_auc"] = aucs
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/leakage.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
