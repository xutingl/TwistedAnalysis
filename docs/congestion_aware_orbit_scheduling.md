# Congestion-Aware Orbit Schedules for All-to-All on Twisted Torus

## 1. Summary

All-to-all is the bandwidth-critical collective behind MoE dispatch/combine and
similar shuffle patterns. On a **symmetric torus**, the standard rotation order
(in round `r`, every device sends to the peer at offset `r`) is already
near-optimal: by symmetry every offset spreads its paths uniformly over the
links, so there is nothing for a smarter schedule to recover. On a **twisted
torus** this breaks — twist wraparounds make per-offset path load highly
uneven, and the rotation order, which is oblivious to paths, can stack several
heavy offsets onto the same links in adjacent rounds.

**Congestion-Aware Orbit Scheduling** closes this gap. It keeps the rotation
order's best property — each round is a *translation orbit*, a permutation in
which every device sends once and receives once — and adds the property the
rotation lacks: orbits are ordered (OrbitFull) or packed into small groups
(OrbitPack) so that no window of concurrently executing rounds concentrates
whole-path link load. The algorithm is **hardware-agnostic**; its only input is
the routing table.

Schedules are generated offline in seconds and compiled into a Pallas
all-to-all kernel, where the schedule takes effect as the kernel's
destination-table ordering. In ICISim, OrbitFull improves throughput by
**+6.5%** over the HLO-order baseline (13259 vs 12450 GB/s at DMA size 32);
OrbitPack performs on par with OrbitFull and reduces peak port queue depth by
**10.8%** on average (81.47 vs 91.35) and **3.9%** at the worst port (293 vs
305) at DMA size 2048. In the hardware experiment, the generated OrbitFull
kernel outperforms the Megablox P2P kernel by **+2.3%** (54400.18 vs 53155.67
gbps at array size 131072).

> **Terminology.** The "HLO order" baseline (ICISim) and the "Megablox P2P"
> kernel (hardware) both execute the same rotation destination order; the rest
> of this report refers to both as **the rotation order**.

## 2. Background

### 2.1 Topology: where the headroom is

On a symmetric torus with minimal routing, uniform all-to-all is perfectly
balanced by construction: every offset's paths tile the links uniformly, every
round of the rotation order loads every link equally, and the schedule meets
the bandwidth lower bound. Reordering rounds cannot help; there is **no
scheduling headroom on the symmetric torus**.

A twisted torus (shape family `{S, 2S}` — e.g. 4×4×8, where a wrap in the
twisted dimension also shifts the other coordinates) trades that symmetry for
a larger bisection at equal radix. The cost is path asymmetry: offsets whose
paths cross twist wraps load some links far more heavily than others, so the
per-round whole-path link load varies widely across the `N−1` offsets. The
rotation order fires offsets in index order, and index-adjacent offsets are
often *physically* similar — their heavy links coincide — so consecutive
rounds compound congestion on the same links. Measured on the loaded 4×4×8
routing, the rotation order's worst-case link load over a window of 6–24
consecutive rounds is roughly **2×** that of a congestion-aware order, decaying
to parity only when the window spans all `N−1` rounds (at which point every
schedule moves the same flows over the same routes). That variance is exactly
the headroom this scheduler harvests.

### 2.2 Routing

The routing table is the algorithm's only external constraint. It fixes, for
every `(src, dst)` pair, the physical path — and therefore fixes (a) the
bandwidth lower bound `LB` (max link load, e.g. 86 / 74 / 75 on the 4×4×8 cell
for dimension-order, LP-load-balanced, and the production routing
respectively), and (b) the *symmetry* the scheduler may rely on.

The relevant symmetry is **translation equivariance**:
`path(σ·u, σ·v) = σ·path(u, v)` for every translation `σ`. Dimension-order
routing satisfies it by construction; LP-based load-balanced routing and
production routings (which use twist-wrap edge classes and escape virtual
channels) do not. When equivariance fails, "orbit-class" capacity accounting —
charging one canonical path per orbit — overstates the feasible set, and a
schedule that verifies under it can still oversubscribe physical links.

Both schedulers below therefore do **full physical-edge accounting**: every
flow is charged along its actual path from the table. This makes the algorithm
valid for *any* routing table; the routing only changes the achievable quality
(its `LB` and its load variance across orbits), not the algorithm's
correctness.

### 2.3 Congestion spreading

The kernel issues each device's DMAs in destination-table order, and the
hardware admits only a bounded number of outstanding transfers. Since all
devices walk their tables in near-lockstep, the set of flows in flight at any
instant is approximately the union of a small **window of consecutive
destination-table columns**. Wherever many in-flight paths cross the same
port, its queue deepens and transfers serialize — port queue depth is the
observable, wall-clock the consequence.

The scheduler's lever is the column order. The objective is to **spread**
heavy-path orbits so that the maximum whole-path link load over *any* window
of `w` consecutive columns stays low (metric: `max_window_edge_load(w)`;
per-routing lower bound `LB(w) = ⌈w · LB / (N−1)⌉`). The effective window
width is set by DMA size and hardware queue depth, which is why the advantage
over the rotation order varies with DMA size (§5).

## 3. Orbit Scheduling

### 3.1 Orbits

The translation group of the torus acts on flows; the orbit of flow `(u, v)`
is the set of all `N` flows with the same source→destination offset. For
all-to-all this partitions the `N(N−1)` flows into `N−1` orbits (127 on the
128-device 4×4×8 cell), and each orbit is a **permutation**: every device
sends exactly once and receives exactly once. Scheduling in whole orbits
therefore guarantees, by construction, zero incast and identical per-device
work in every round — properties the rotation order also has, and which
device-jagged makespan-optimized schedules give up (to their measurable
detriment). Each orbit additionally carries a fixed whole-path load profile
over the links, computed directly from the routing table; on a twisted torus
these profiles vary widely across orbits, and that variance is what the
following two schedulers manage.

### 3.2 OrbitFull (`orbit_greedy_full`)

OrbitFull sequences orbits greedily under full physical-edge accounting:
orbits are processed heaviest-first (longest total path length, with a
tail-ascending tiebreak) and each is placed at the earliest round at which
every link on every one of its paths still has capacity. Heavy,
twist-crossing orbits are thereby forced apart, and each round's residual
capacity is filled with light orbits — the schedule interleaves heavy and
light instead of letting index order cluster the heavy ones. On the loaded
4×4×8 routing it schedules the 127 orbits with makespan 85 against the
routing's lower bound of 75.

### 3.3 OrbitPack(K, C) (`orbit_pack`)

OrbitPack makes the congestion bound explicit. It first-fit-decreasing packs
whole orbits into ordered *steps*, admitting an orbit into a step only if
(a) the step holds fewer than `K` orbits and (b) the union of whole-path link
loads of the step's orbits stays ≤ `C` on every link. Every step is thus a
union of at most `K` permutations — per-device sends = receives ≤ `K` — with a
certified congestion cap. The instantiation used throughout this report is
**OrbitPack(6, 3)**: `C = 3` equals the rotation order's *own* worst per-round
whole-path load, and `K = 6` packs the 127 orbits of the loaded 4×4×8 routing
into 27 steps (K = 2 → 64 steps, K = 3 → 43).

### 3.4 Why it helps

The design space has two axes, and the baselines each get one right. The
rotation order is orbit-atomic but path-oblivious: its congestion over small
windows runs ~2× the achievable minimum on twisted routings (§2.1).
Solver-based makespan-optimal schedules are path-aware but sacrifice orbit
atomicity: they finish in fewer nominal rounds but are device-jagged (unequal
per-device work per round, incast), and in practice fail to beat the rotation
order. Congestion-Aware Orbit Scheduling occupies the remaining corner —
**keep orbit atomicity, add path awareness** — and both variants realize it:
OrbitFull spreads heavy orbits so no window is dominated by them; OrbitPack
certifies a per-step cap `C`, so any execution window spanning `m` steps
carries at most `~mC` load on any link. Because all schedules converge at the
full-window limit, the advantage is concentrated where the in-flight window is
a small fraction of the schedule — equivalently, it is largest in the
DMA-size regimes where congestion, not aggregate bandwidth, binds.

## 4. Implementation

### 4.1 Schedule generation

Schedule generation is cheap, offline, and reproducible:

```
python scripts/generate_schedule.py \
    --routing-table fixtures/routing/routing_table_8x4x4_twist.json \
    --scheduler orbit_pack --k 6 --c 3
```

emits a flat JSON schedule (`{round, src, dst, path}` entries;
`fixtures/nonragged/schedule_<slice>_<router>_<scheduler>.json`). OrbitFull
runs the same way with `--scheduler orbit_greedy_full`. Both complete in
seconds on the 128-device cell. Verification is part of the pipeline:
`verify_capacity_step` checks OrbitPack's per-step cap `C`,
`max_window_edge_load` scores any schedule under the windowed-congestion
metric of §2.3, and `scripts/schedule_stats.py` reports step count, achieved
edge cap, and per-device DMA depth for any schedule JSON.

### 4.2 Pallas kernel generation

`pallas_kernel/gen_orbit_greedy_kernel.py` consumes a routing table plus a
schedule JSON and emits a self-contained Pallas all-to-all kernel. Key design
points:

- **The schedule reaches hardware as a destination ordering.** The generator
  builds a per-source destination table (column `k` = the `k`-th destination
  of each source, sorted by `(round, dst)`), and the kernel walks its column
  sequence issuing one remote DMA per column. All scheduling intent —
  spreading, packing, caps — is carried by this ordering; the kernel body is
  identical across schedules, so measured differences are attributable to the
  order alone.
- **Uniformity by construction.** Because every step is a union of
  permutations, all devices issue the same number of DMAs per step and the
  kernel is a single symmetric SPMD program — no per-device specialization,
  no idle devices, no incast hot spots.
- **Packet chunking.** Each per-destination payload is split into fixed-size
  packets; packet size is a tunable with a hardware-dependent sweet spot
  (32 KB on the platforms measured to date; both smaller and larger lose
  throughput).
- **Two issue modes.** The default all-up-front mode issues every DMA
  `.start()` back-to-back and waits once on the receive semaphore at the end —
  appropriate for hardware with deep DMA queues. For shallow-queue hardware, a
  per-step throttled mode (`--per-step-barrier`) issues one step's orbits,
  drains the *send* semaphore, then proceeds. Per-step drains must be
  SEND-only: the receive semaphore is drained exactly once at the end against
  the true total received bytes, because under non-uniform workloads a device
  receives a different byte count than it sends per step, and a per-step
  receive wait keyed on sent bytes deadlocks.
- **Destination-table placement.** The table ships either as an SMEM input or
  baked into the program (`--inline-destinations`, per-step destinations as
  `jax.lax.switch` branches) — a code-size vs. preamble-cost trade-off.

## 5. Results

### 5.1 ICISim throughput

*(to be filled)*

### 5.2 ICISim queue-depth analysis

*(to be filled)*

### 5.3 Hardware

*(to be filled)*
