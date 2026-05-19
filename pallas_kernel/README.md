# `pallas_kernel/` — orbit-greedy P2P AllToAll for twisted-torus topologies

This directory holds the Pallas TPU kernels used for actual TPU deployment of
the OrbitGreedy schedule. The kernels here are *generated* code: the
generator consumes a topology + router and emits a self-contained Python file
that replaces the default `_ragged_a2a_kernel_point_to_point` inside
`ragged_all_to_all`.

## Files

| File | Purpose |
|---|---|
| [reference_kernel.py](reference_kernel.py) | Reference `ragged_all_to_all` and `_ragged_a2a_kernel_point_to_point` extracted from `google3/learning/brain/research/megablox/collectives/ragged_all_to_all.py`. The orbit-greedy kernel is a drop-in for the P2P branch only. |
| [gen_orbit_greedy_kernel.py](gen_orbit_greedy_kernel.py) | Pipeline orchestrator. Either generates a routing table (via `--router`) or loads one (via `--routing-table`); generates a schedule from it; emits a kernel `.py` file. Persists the routing table and schedule as inspectable intermediates. |
| `outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py` | Generator output. One file per (topology, router, scheduler, order) combination. Current outputs: `orbit_greedy_8_4_4.py`, `orbit_greedy_full_8_4_4.py`, `literal_greedy_8_4_4.py`, `cpsat_literal_warm_8_4_4.py` (makespan-78 production recommendation for the loaded 8×4×4 routing; SMEM `dest_table_ref` input), and `cpsat_literal_warm_inline_8_4_4.py` (same schedule with destinations baked as compile-time `jax.lax.switch` branches via `--inline-destinations`; no SMEM input), `spread_greedy_k2_8_4_4.py`, and `spread_greedy_k2_inline_8_4_4.py` (per-device DMA-cap K=2 schedule; SMEM and inline variants). |

## What problem this kernel solves

The reference `_ragged_a2a_kernel_point_to_point` iterates destinations in
**rotation order** — equivalent to Latin-square round-robin:

```python
expert_offset = (my_id + 1) * groups_per_shard
group_idx = (i + expert_offset) % num_groups   # device d hits (d+1, d+2, …)
```

On a `{S, 2S}^n` twisted torus this is 4–9× sub-LB (per
[`../docs/orbit_greedy_optimality.md`](../docs/orbit_greedy_optimality.md)).
The generated kernel replaces this with the OrbitGreedy hop-0 firing order,
which achieves makespan = LB in the canonical paper model.

The kernel is *otherwise identical* to the reference: same DMA primitive
(`pltpu.make_async_remote_copy`), same packet chunking, same
`total_send_amount` / `total_recv_amount` drain pattern, same barrier setup.
Only the iteration order changes.

## Scheduler choice

The generator supports several scheduling algorithms via `--scheduler` (or
`--schedule-in <path>` to load a precomputed schedule from disk). Scheduler
performance **depends on the routing** — picking a scheduler in isolation is
not enough; the routing × scheduler pair determines the realized makespan.

### Algorithms

| Scheduler | Approach | Always physically feasible? |
|---|---|---|
| `orbit_greedy` (default) | Orbit greedy with full-physical-edge accounting (alias for `orbit_greedy_full` since 2026-05-15; the original `(dim, dir)`-keyed implementation was unsound on non-equivariant routings and was replaced) | Yes |
| `orbit_greedy_full` | Same algorithm; explicit name kept for clarity | Yes |
| `literal_greedy` | LMR-style per-flow earliest-feasible greedy on the literal `N(N-1)` flow set | Yes |
| `spread_greedy(k)` | `literal_greedy` plus a per-device cap of K outgoing AND K incoming DMAs per round. K=1 is P2P-style (each device sends/receives at most 1 DMA per round); K=∞ is `literal_greedy` | Yes |
| `cpsat_literal` | CP-SAT (OR-Tools) on the literal flow set; supports `--schedule-in` warm-starting | Yes (when CP-SAT returns FEASIBLE/OPTIMAL); may TIMEOUT |
| `ilp_literal` | Exact ILP on the literal flow set (PuLP/CBC) | Yes (when CBC returns); intractable at N=128 |

The post-schedule capacity verifier refuses to emit a kernel whose schedule
has any physical-edge collisions, so any wrong combination fails fast at
generation time.

### Routing × scheduler performance matrix (physical-edge model)

Numbers are `makespan` (lower is better; LB = max physical-edge load).

| Routing | N | LB | `orbit_greedy_full` | `literal_greedy` | `ilp_literal` | `cpsat_literal` (warm) | `spread_greedy(k=2)` |
|---|---:|---:|---:|---:|---|---:|---:|
| (2,4) ILP | 8 | 3 | **3** | 3 | 3 | 3 | — |
| (2,2,4) ILP | 16 | 5 | **5** | 6 | 5 (~1 s) | 5 | — |
| (2,4,4) ILP | 32 | 11 | 12 (+1) | 14 | **11** (~3 min) | 11 | — |
| (4,8) ILP | 32 | 21 | 22 (+1) | 25 | **21** (~85 min) | 21 | — |
| (8,4,4) loaded | 128 | 75 | 85 (+10) | 87 | intractable | **78** (+3, warm-started CP-SAT @4 h) | 92 (+17) |

**What this matrix shows:**

- On (2,4) and (2,2,4) ILP, `orbit_greedy_full` is LB-optimal.
- On (2,4,4) and (4,8) ILP, the literal ILP **proves** LB is achievable in
  the physical-edge model — so `orbit_greedy_full`'s LB+1 is a heuristic-
  level sub-optimality, not a fundamental bound. The gap is 1 step in both
  cases.
- On the loaded 8×4×4 TPU routing, the literal ILP is intractable (~1.37 M
  binary vars; CBC's LP relaxation alone runs indefinitely), but **CP-SAT
  scales**: cold CP-SAT @30 min/probe reaches makespan 80, and warm-starting
  from that incumbent at `t_upper=79` (4 h budget) reaches **makespan 78**.
  Whether LB=75 is achievable remains open: cold CP-SAT at `t_upper ∈ {77,
  76}` and LNS at 5–30% destroy both fail to escape makespan 78 (the
  makespan-78 schedule appears *structurally tight* — see
  [2026-05-16 exploration](../eval/explorations/2026-05-16-closing-gap-to-lb-75/)).
- `literal_greedy` consistently trails `orbit_greedy_full` by 5–14%. It's a
  fallback for routings where orbit symmetry breaks entirely; on routings
  with orbit structure it loses to the orbit-based variants.

### When to use which

| Routing class | Recommended scheduler | Why |
|---|---|---|
| DOR (any cell) | `orbit_greedy` | Translation-equivariant; orbit greedy is LB-tight or LB+1 |
| ILPRouter (small cells N ≤ 16) | `orbit_greedy` or `ilp_literal` | Both reach LB; ILP is the exact ground truth |
| ILPRouter (N=32) | `ilp_literal` if you have minutes; `orbit_greedy` otherwise | ILP closes the +1 gap that orbit greedy has on (2,4,4)/(4,8) |
| ILPRouter (N=128, i.e. 4×4×8) | `orbit_greedy` | ILP intractable; orbit greedy is the best practical choice |
| **Loaded TPU routing** | **`spread_greedy(k=2)` (current testbed candidate, fixtures shipped — makespan 92) — or `cpsat_literal` warm-started (`cpsatliteralwarm`, makespan 78, projected +7.5% but measured ~0 % on TPU)** — fall back to `orbit_greedy` (makespan 85) for the no-CP-SAT baseline | CP-SAT @4 h warm-started from the makespan-80 fixture finds makespan 78 at `t_upper=79`. The precomputed schedule is shipped; use it via `--schedule-in` rather than re-running the 4 h solve. The 2026-05-17 hypothesis: TPU wall-clock is dominated by per-device DMA-engine concurrency and ICI bandwidth, not by round count. spread_greedy(k=2) caps simultaneous DMAs per device per round to test this. |

### Example invocations

```bash
# spread_greedy(k=2) — per-device-DMA-capped headline; ship both regular
# and inline-destinations kernels:
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_spread_greedy_k2.json \
    --function-name _ragged_a2a_kernel_spread_greedy_k2_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py

# Other K values (1, 3, 4) ship as fixtures only; regenerate the kernel
# with the same command pattern, substituting the K in the schedule path
# and output filename.

# Loaded TPU routing on 8x4x4 — production recommendation:
# load the precomputed makespan-78 schedule and emit the kernel directly.
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json

# Same schedule, but with destinations inlined as jax.lax.switch branches
# (no SMEM dest_table_ref input). Use this if profiling on TPU shows that
# the per-step DEST_TABLE lookup is on the critical path:
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json \
    --inline-destinations \
    --function-name _ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4.py

# To regenerate the makespan-78 schedule from scratch (e.g., on a different
# routing table), run the warm-started CP-SAT probe from the 2026-05-16
# exploration. The kernel-generator CLI does NOT expose `--scheduler
# cpsat_literal` — schedules from CP-SAT are produced by direct Python
# scripts and consumed via `--schedule-in`. The probe takes ~4 h per
# t_upper at 8 workers:
#   python eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py

# Baseline: orbit_greedy (makespan 85, no CP-SAT compute needed):
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json

# Same routing, sanity-check with literal_greedy:
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --scheduler literal_greedy

# Small-cell ground-truth oracle (proves LB-achievability):
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 2,4,4 --router ilp \
    --scheduler ilp_literal --ilp-time-limit-s 600
```

## The twist (why per-source destinations)

On a `{S, 2S}^n` twisted torus, the group operation is **not** elementwise
modular addition. The wraparound in dim `k` adds `slice[k]` to *every* coord
then mods each by its own size (see
[`../twisted_analysis/topology/lattice.py`](../twisted_analysis/topology/lattice.py#L33-L43)).
For slice = (4, 4, 8):

- **x-wrap** (shift = 4): x mods to 0; y unchanged (`(y+4) mod 4 = y`);
  **z shifts by 4** (`(z+4) mod 8 ≠ z`). ← twist
- **y-wrap**: same — z shifts by 4.
- **z-wrap** (shift = 8): no effect (8 ≡ 0 mod 4, 8 ≡ 0 mod 8).

### Concrete mismatch with naïve `(src + δ) mod size`

Canonical path `[+x, +x, +x]` from origin → δ_canonical = `(3, 0, 0)`. No
wraparound during this walk; δ is just the displacement.

Walk the same path from source `(1, 0, 0)`:

```
(1,0,0) → (2,0,0) → (3,0,0) → wraparound on +x → ((4+4)%4, (0+4)%4, (0+4)%8) = (0, 0, 4)
```

The orbit member's destination from source `(1, 0, 0)` is **`(0, 0, 4)`**.

Naïve elementwise mod gives `((1+3)%4, (0+0)%4, (0+0)%8) = (0, 0, 0)`. **Off by
(0, 0, 4)**.

### Consequence

The kernel cannot compute destinations on-the-fly as `(my_coord + δ_O) mod size`.
Instead, the generator builds a `DEST_TABLE[N, K]` table by walking *each
source* through *each orbit's canonical path* using `Topology.neighbor()`,
which encodes the twist correctly. The table is baked into the generated
file as a numpy literal (~65 KB for 4×4×8). At runtime the kernel does
`dst_flat = DEST_TABLE[my_flat, k]` and decodes per-axis coords with simple
divmod — no twist logic in the kernel itself.

### Why not absorb the twist into the JAX `Mesh`?

The 4×4×8 twisted-torus group has the relation `a^4 = c^4` where `a` is the
+x generator and `c` is the +z generator (because 4 +x steps from origin
land at `(0, 0, 4)`, the same place as 4 +z steps). The group is therefore
not Cartesian — no `Z_a × Z_b × Z_c` relabeling of devices makes
elementwise modular arithmetic correct. JAX's standard `Mesh` only supports
Cartesian device assignments, so this option is closed.

## Usage

The kernel generator runs a 3-stage pipeline. Each stage's artifact is persisted under the project's standard directories so it can be inspected, reused, or regenerated independently.

```
[--router ilp|dor]                  [Stage 1]    fixtures/routing_table_<slice>_<router>.json
                  -- OR --
[--routing-table FILE]              (use existing)
                                          ↓
[--scheduler orbit_greedy --order …]  [Stage 2]    fixtures/schedule_<slice>_<...>_<order>.json
                                          ↓
                                    [Stage 3]    pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py
```

### Generate a kernel by also generating the routing table

Default: slice = 8,4,4, ILP routing, `lpt_tail_asc` order, no per-step barriers.

```bash
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 8,4,4 --router ilp
# [1/3] wrote routing table fixtures/routing_table_8x4x4_ilp.json
# [2/3] wrote schedule     fixtures/schedule_8x4x4_ilp_lpt_tail_asc.json
# [3/3] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_8_4_4.py
```

### Generate a kernel from an existing routing table

```bash
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json
# [2/3] wrote schedule     fixtures/schedule_8x4x4_loaded_lpt_tail_asc.json
# [3/3] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_8_4_4.py
```

The pre-shipped `fixtures/routing_table_8x4x4_twist.json` is a 4×4×8 TPU v5e twisted torus; it is stored with the largest dim first (slice `(8,4,4)`) to match the `{S, 2S}^n` flatten convention used elsewhere in the project.

### Run a single stage

```bash
# Stage 1 only — emit a routing table:
python scripts/generate_routing_table.py --slice 8,4,4 --router ilp

# Stage 2 only — emit a schedule from a routing table:
python scripts/generate_schedule.py \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --slice 8,4,4 \
    --scheduler orbit_greedy --order lpt_tail_asc
```

Common variants:

```bash
# DOR routing instead of ILP:
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 8,4,4 --router dor

# Different topology in the {S, 2S}^n family:
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 2,4,4 --router ilp
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,8 --router ilp

# Per-step barriers (forces stricter ordering, less pipelining):
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 8,4,4 --router ilp --per-step-barrier
```

### Wire into `ragged_all_to_all`

The orbit-greedy kernel has the same signature as
`_ragged_a2a_kernel_point_to_point` **plus one extra positional `Ref` input
`dest_table_ref`** (slot 6, between `num_packets_per_group_ref` and `x_ref`).
This is required because Pallas refuses to capture large constants as
closures — the destination table must be a `pallas_call` input.

**Integration steps:**

1. Copy the generated file into your project next to `reference_kernel.py`.

2. In `ragged_all_to_all` (or wherever you set `kernel = ...`), switch the
   kernel name:

   ```python
   from ._ragged_a2a_kernel_orbit_greedy_8_4_4 import (
       _ragged_a2a_kernel_orbit_greedy_8_4_4,
       build_pallas_call_kwargs,
   )
   kernel = _ragged_a2a_kernel_orbit_greedy_8_4_4
   ```

3. **Inject the destination table as an extra `pallas_call` input.** The
   helper `build_pallas_call_kwargs()` produces the JAX array (lazily, so no
   JAX backend init at import) and the SMEM `BlockSpec`:

   ```python
   kw = build_pallas_call_kwargs()
   dest_table = kw["dest_table"]                    # jnp.ndarray, int32[N, K]
   extra_in_spec = kw["extra_in_spec"]              # pl.BlockSpec(SMEM)
   alias_shift = kw["input_output_aliases_shift"]   # 1
   ```

4. **Modify `in_specs` to insert `extra_in_spec` at slot 6** (between the
   six scalar SMEM specs and the two ANY-memory specs for `x` and
   `existing_out`):

   ```python
   in_specs = [
       pl.BlockSpec(memory_space=pltpu.SMEM),  # input_offsets    (slot 0)
       pl.BlockSpec(memory_space=pltpu.SMEM),  # output_offsets   (slot 1)
       pl.BlockSpec(memory_space=pltpu.SMEM),  # send_sizes       (slot 2)
       pl.BlockSpec(memory_space=pltpu.SMEM),  # total_send       (slot 3)
       pl.BlockSpec(memory_space=pltpu.SMEM),  # total_recv       (slot 4)
       pl.BlockSpec(memory_space=pltpu.SMEM),  # num_packets      (slot 5)
       extra_in_spec,                          # dest_table       (slot 6)  ← NEW
       pl.BlockSpec(memory_space=pl.ANY),      # x                (slot 7)
       pl.BlockSpec(memory_space=pl.ANY)       # existing_out     (slot 8)
       if existing_out is not None else None,
   ]
   ```

5. **Pass `dest_table` as an extra positional input** to the
   `pallas_call(...)` call, between `num_packets_per_group` and `x`:

   ```python
   pallas_call(...)(
       input_offsets,
       output_offsets,
       send_sizes,
       total_send_amount,
       total_recv_amount,
       num_packets_per_group,
       dest_table,                # ← NEW
       x,
       existing_out,
   )
   ```

6. **Shift `input_output_aliases` keys by `alias_shift` (= 1).** The
   reference uses `{7: 0}` (existing_out aliases output); with this kernel
   it becomes `{8: 0}`.

7. **Call `ragged_all_to_all` with `axis_name` as a single flat string**
   spanning all `N = prod(slice)` devices:

   ```python
   out = ragged_all_to_all(
       x, routing_info,
       mesh=mesh,
       axis_name="x",          # flat single string (NOT a tuple)
       collective_id=k,
       ...
   )
   ```

   The orbit-greedy kernel does *not* decode per-axis coordinates — it
   calls `jax.lax.axis_index(axis_name)` once to get the flat device id and
   looks up destinations in `dest_table_ref`. The twist is baked into the
   table, not handled at runtime.

## Generator options reference

| Flag | Default | Meaning |
|---|---|---|
| `--slice` | required | Comma-separated topology shape, e.g. `8,4,4`. Must be in `{S, 2S}^n`. |
| `--router` | (when `--routing-table` is absent) `ilp` | `ilp` (load-balanced minimal) or `dor` (dimension-order). Mutually exclusive with `--routing-table`. |
| `--routing-table` | none | Path to an existing routing-table JSON. Skips stage 1; loads paths verbatim. |
| `--order` | `lpt_tail_asc` | OrbitGreedy ordering. `lpt_tail_asc` achieves makespan = LB on every doc cell. |
| `--per-step-barrier` | off | Insert dummy-DMA barriers between OrbitGreedy steps. |
| `--function-name` | `_ragged_a2a_kernel_orbit_greedy_<slice>` | Override the generated function name. |
| `--routing-table-out` | `./fixtures/routing_table_<slice>_<router>.json` | Where to save a generated routing table. |
| `--schedule-out` | `./fixtures/schedule_<slice>_<router_or_loaded>_<order>.json` | Where to save the schedule. |
| `--out` | `./pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py` | Output kernel path. |
| `--inline-destinations` | off | Bake per-step destinations into the kernel as compile-time `jax.lax.switch(my_flat, _DEST_BRANCHES_k)` branches instead of an SMEM `dest_table_ref` input. Drops the extra pallas_call input. Larger generated file but eliminates the per-step SMEM load from the inner critical path. Used to test whether SMEM DEST_TABLE lookup is a real wall-clock bottleneck on TPU. |

## Caveats / TODO

- **No TPU silicon yet.** The kernel cannot be tested locally. The generator
  output parses as Python (verified via `ast.parse`), but Pallas-specific
  semantics (semaphore behaviour, `device_id` resolution, SMEM bandwidth
  for the `dest_table_ref` lookup) need TPU validation.
- **`transpose=True` is not supported.** The orbit-to-destination map would
  need regeneration with reversed orbit direction. The kernel asserts.
- **Assumes 1 group per device** (uniform AllToAll). For ragged use cases,
  keep the reference kernel.
- **`ragged_collectives_utils` import path.** The generated file imports
  `from megablox.collectives import ragged_collectives_utils`. Adjust the
  import to match your project layout.
- **Compile time.** With `per_step_barrier=True` on 4×4×8 the generated file
  is ~100 KB / 740 lines (68 unrolled step blocks). JAX trace + Pallas
  lower-to-Mosaic time may be measurable on first call; subsequent calls hit
  the JIT cache.

## Validation plan (once TPU access lands)

1. **Correctness smoke test.** Generate a kernel for the smallest cell
   (`--slice 2,4`, 8 devices), feed a uniform AllToAll with distinct
   per-source payloads, verify `out[i] == in_on_device_i`.
2. **Performance baseline.** Run reference rotation P2P kernel on a
   `{S, 2S}` sub-mesh (e.g., 4×8) with bandwidth-bound payload. Record
   wall-clock.
3. **OrbitGreedy comparison.** Same sub-mesh, same payload, swap to the
   generated kernel. Compute the ratio.
4. **Decision rule (from earlier analysis):** if observed speedup ≥ 1.5×,
   the ordering wins are surviving the ICI router; proceed to 4×4×8. If
   < 1.2×, the ICI fabric is largely neutralizing the ordering, and the
   next investment should be in a multihop kernel (see
   [`../twisted_analysis/schedules/orbit_greedy.py`](../twisted_analysis/schedules/orbit_greedy.py)
   for the LB-relevant hop schedule).

## Dispatch-path tuning options (2026-05-18)

Two orthogonal CLI flags target the kernel's scalar dispatch path. They emit different kernel variants from the same schedule; choose between them at generation time. Both flags only apply to the default codepath (NOT `--per-step-barrier`, NOT `--inline-destinations`).

### `--packed-state` (Option A)

Insert a one-time preamble fori_loop that builds a per-source packed-state array `_my_state[K, 4]` carrying `(dst, sizes_ref[dst], input_offsets_ref[dst], output_offsets_ref[dst])` for each orbit `k`. The main hot loop then reads from `_my_state` instead of issuing 3 dependent SMEM reads per iteration.

- **Preamble cost**: 1 fori_loop of K iterations, run once per kernel call (K ≈ N-1 = 127).
- **Hot-loop savings**: 3 SMEM reads × `(N-1) × num_packets` iterations per call eliminated (replaced by 4 reads from `_my_state` — but those are from a single small array with high locality, so the compiler likely fuses them into 1 wide read).
- **Caller integration**: identical to the default kernel — no extra inputs or scratch needed.

### `--wait-batch-size N` (Option B)

Group the inner loop's `(N-1) × num_packets` DMA issues into batches of `N` and insert a `make_async_copy(...).wait()` drain after each batch. The default kernel issues all DMAs and drains only at the very end; this flag inserts intermediate drains to test whether the DMA-engine in-flight queue is the bottleneck.

- `N = 0` (default): no intermediate drains, identical to the legacy kernel.
- `N = 127`: one drain per packet_idx (recommended starting probe).
- `N = 64` or smaller: more frequent drains, less in-flight concurrency. Worth probing if `N=127` shows improvement.

The kernel maintains a running `cum_bytes` counter and drains to it via the standard `make_async_copy(o_ref.at[pl.ds(0, cum_bytes)], ..., send_sem).wait()` pattern.

### Generated variants

| Variant | Source kernel | Flag combination |
|---|---|---|
| `_ragged_a2a_kernel_cpsat_literal_warm_packed_8_4_4.py` | cpsat_literal_warm (makespan 78) | `--packed-state` |
| `_ragged_a2a_kernel_cpsat_literal_warm_pipelined_8_4_4.py` | cpsat_literal_warm (makespan 78) | `--wait-batch-size 127` |

Both are produced from the same schedule fixture (`fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json`); structural correctness is verified by `tests/test_gen_kernel_options.py`. TPU wall-clock correctness must be verified out of band by the operator.

## Related

- [docs/orbit_greedy_optimality.md](../docs/orbit_greedy_optimality.md) — the
  optimality theory (König + Smith).
- [twisted_analysis/schedules/orbit_greedy.py](../twisted_analysis/schedules/orbit_greedy.py)
  — the reference Python implementation of the OrbitGreedy schedule.
- [twisted_analysis/topology/lattice.py](../twisted_analysis/topology/lattice.py)
  — the twisted-torus topology + the `neighbor()` function the generator
  uses to bake the destination table.
