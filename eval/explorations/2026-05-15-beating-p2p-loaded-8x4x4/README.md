# Beating the P2P kernel on loaded 8×4×4

## Problem

The reference Pallas point-to-point AllToAll kernel measures **134541 gbps** on TPU
v5e with `slice=(8,4,4)` under the routing table at
`fixtures/routing_table_8x4x4_twist.json` ("loaded" routing — externally produced,
likely escape-VC + OCS aware). Our `orbit_greedy_full[lpt_tail_asc]` schedule with
sim-makespan **85** measures **132758 gbps** — 1.3% slower than P2P.

By linear scaling (throughput ∝ 1/makespan in steady state), we need
**sim-makespan ≤ 83** to definitively beat P2P, and ideally lower since the
simulator omits per-step barrier latency, HBM contention, and VC arbitration.

Physical-edge LB on this routing is **75** (max edge load over the AllToAll
workload). Whether LB is achievable on this non-translation-equivariant routing
is open — literal ILP with CBC is intractable at N=128 (1.37M binary variables).

## Goal

Find a schedule with sim-makespan ≤ 83 (and ideally ≤ 80) on the loaded routing
through algorithmic search. Each probe is documented in `RESULTS.md` with its
makespan, violation count, and runtime.

## Probes (cheapest first)

1. [01_ordering_sweep.py](01_ordering_sweep.py) — Deterministic orderings on existing greedies.
2. [02_random_orbit_shuffle.py](02_random_orbit_shuffle.py) — Random orbit orderings on `orbit_greedy_full`.
3. [03_cpsat_probe.py](03_cpsat_probe.py) — Google OR-Tools CP-SAT at decreasing `t_upper`.
4. [04_lp_rounding_probe.py](04_lp_rounding_probe.py) — LP relaxation + randomized rounding.
5. [05_local_search_probe.py](05_local_search_probe.py) — Local-search repair on best-found.

## Conclusions

(Filled at the end of the exploration — see [RESULTS.md](RESULTS.md) for the rolling table.)
