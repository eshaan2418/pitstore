# Deep dive: questions and answers

### 1. What exactly is "point-in-time correctness"?
Every feature value in a training row must equal what the serving system could have computed at that row's prediction time. Concretely, it may use only events with timestamp `< t − delay`, where `delay` is the ingestion lag. pitstore encodes this as half-open windows `[t − delay − window, t − delay)`. Both stores implement it, and both are tested against a brute-force reference.

### 2. Why did the leaky models do *worse* in deployment than the correct one, not just equally?
They learned relationships that only hold with future information. The snapshot model puts 64% of its split gain on 365-day purchase counts. Computed at the dataset end, those counts include the purchases the label is about. At serving time the same feature cannot contain the future, so the model's learned thresholds are miscalibrated. A correct model learns how much the *past* predicts the future and uses it properly: 0.797 deployed against 0.746 and 0.758.

### 3. Why is the off-by-one-window leak so much larger offline (0.994)?
Computed at `r + 30 days`, `days_since_last ≤ 30` almost exactly when the customer bought inside the label window, so the feature is nearly the label. That bug pattern (using the label window's end as the feature timestamp) is easy to make with SQL-style joins on "period end" dates.

### 4. How does the leakage check work, and what can't it catch?
For a sample of queries, it takes that entity's events at or after the cutoff and recomputes the features under two counterfactual futures: values and times randomised, and those events deleted. If any feature changes, the pipeline used future data. It is *sound*: a correct pipeline is independent of the future, so it never fails. It is not *complete*: a leaky feature that happens to produce the same value under both perturbations goes unnoticed. The test suite checks the leaky pipeline is flagged on exactly the queries whose entity has events in the leaked span, no more and no fewer.

### 5. Why did the first version of the check miss half the snapshot leaks?
It only moved future events to random times just after the cutoff. A snapshot feature computed over the 90 days before the dataset end never sees events moved to a few days after a mid-2011 cutoff. Its value changed only if the customer happened to have events in that final 90-day window, which many did not. Adding deletion as a second counterfactual raised the flagged count from 49 to 60 of 100, which equals the number of sampled customers who have any post-cutoff events (the maximum possible).

### 6. How do the as-of joins avoid a per-row loop?
Events are sorted by `entity_index << 40 | time_offset`. A window for `(entity, t)` maps to the key interval `[entity << 40 | (t − delay − window − t0), entity << 40 | (t − delay − t0))`, and its bounds are two vectorised `searchsorted` calls over all queries. Because the entity sits in the high bits, the interval can never cross into another entity's events. Counts are `hi − lo`, sums come from prefix sums, and "last" reads index `hi − 1`.

### 7. Where was build time going, and what fixed it?
`np.lexsort((time, entity))` took 18.7 s for 10M events, nearly all of the original 21 s build. Sorting the single composite key with a stable argsort takes 5.5 s and produces the same order, with input order kept for ties, which the `last` aggregate depends on. The build fell to 11.8 s, so build plus query beats pandas `merge_asof` at 10M events (12.1 s vs 23.1 s).

### 8. How is training/serving consistency verified?
A Hypothesis test generates random event streams with collisions, ties and out-of-window events, replays them in time order into the online store, and before each ingestion compares every entity's online features with the offline as-of features at that moment. Any off-by-one in window bounds or delay handling fails this test immediately.

### 9. Why include an ingestion delay at all?
Real pipelines see events late: logs are batched and stream processors lag. A model trained with features that assume zero delay is evaluated with fresher information than production will have. The delay makes that assumption explicit and testable. Here it is 1 day. Delay is part of the semantics in both stores and in the leakage check's cutoff.

### 10. Why the customer-clustered bootstrap?
Each customer appears in up to 5 test months, and those rows are correlated. Resampling rows independently would understate uncertainty. Resampling customers gives honest intervals: about ±0.008 AUC on 26,540 test rows.

### 11. What would a production version need?
Persistence (parquet partitions per entity range), incremental materialisation with watermarks for late events, incremental online aggregates with expiry instead of recompute-on-read, more aggregations (distinct counts via HyperLogLog, percentiles via t-digest), and the leakage check wired into CI for every feature pipeline change.

### 12. What would you do with 100x compute?
Repeat the leakage measurement across many public event datasets and leak types (target encoding computed on all data, normalisation statistics computed including test, duplicate rows across time) to estimate how large the offline/deployed gap typically is, and how often a sampled check like this one would have caught it.
