# Design

## Semantics
For a query `(entity, t)`, a feature aggregates that entity's events with timestamp `e`
satisfying `t − delay − window ≤ e < t − delay`. The half-open upper bound means an event at
exactly the visibility cutoff is not yet visible. `delay` models ingestion lag, and `window=None`
means all history. Both stores implement exactly this, and both are tested against a
brute-force reference.

## Offline store
Events are sorted once by an int64 composite key `entity_index << 40 | (time − t0)`. For a batch
of queries, each window's bounds are two `searchsorted` calls on that key (entity-prefixed, so
a window can never spill into a neighbouring entity's events). Sums and means come from prefix
sums, and "last" and "time since last" read the element just before the upper bound. No
per-row Python runs, and a query batch costs O(M log N).

The first version sorted with `np.lexsort((time, entity))`, which took 18.7 s for 10M events.
A single stable argsort of the composite key takes 5.5 s and gives the same order, with ties
kept in input order (which `last` relies on).

## Online store
Per-entity deques hold events within the longest `window + delay` (or everything when a
feature is unbounded). Reads recompute aggregates with the same bounds. A property test
replays random event streams and asserts that before each event, every entity's online
features equal the offline as-of values at that time. That is the training/serving
consistency guarantee.

## Leakage check
`check_point_in_time` samples queries. For each, it takes the entity's events at or after
the cutoff and recomputes under two counterfactual futures: values and times randomised, and
those events deleted. A pipeline whose output changes depended on the future. It accepts any
callable pipeline, so it can test feature code that does not use pitstore at all. It is
sound (a correct pipeline never fails) but not complete. A leak that happens to leave the
value unchanged under both perturbations is missed. The first version only moved future
events, and it missed snapshot-join leaks whose windows ended far after the moved events.
