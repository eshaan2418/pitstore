# Model card

pitstore is a feature-computation library and ships no model. The leakage experiment trains
LightGBM classifiers at run time (`make eval`), only to measure how leakage changes evaluation.

- **Intended use:** computing training features that match what an online system can serve,
  and testing pipelines for future-data leakage before trusting offline metrics.
- **Not intended for:** deciding anything about real customers. The repurchase model is a
  measurement device, and its features describe purchase behaviour only.
- **Limitations:** supported aggregations are count, sum, mean, last value, and time since
  last event. Distinct counts and percentiles are not supported. The leakage check is sampled
  and sound but not complete (see README).
