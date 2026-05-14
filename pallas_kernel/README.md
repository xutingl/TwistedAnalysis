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
| [gen_orbit_greedy_kernel.py](gen_orbit_greedy_kernel.py) | Generator. Takes a topology and router, emits a topology-specific kernel `.py` file. |
| `_ragged_a2a_kernel_orbit_greedy_<slice>.py` | Generator output. One file per (topology, router, order) combination. Pre-generated default: `_ragged_a2a_kernel_orbit_greedy_4_4_8.py` (slice=(4,4,8), ILP router). |

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

### Generate a kernel

Default: slice = 4,4,8, ILP routing, `lpt_tail_asc` order, no per-step barriers.

```bash
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8
# → writes pallas_kernel/_ragged_a2a_kernel_orbit_greedy_4_4_8.py
```

Common variants:

```bash
# DOR routing instead of ILP:
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --router dor

# Different topology in the {S, 2S}^n family:
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 2,4,4
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,8

# Custom axis names matching your JAX mesh:
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --axis-names mesh_x,mesh_y,mesh_z

# Per-step barriers (forces stricter ordering, less pipelining):
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --per-step-barrier
```

### Wire into `ragged_all_to_all`

1. Copy the generated file into your project next to `reference_kernel.py`.
2. In `ragged_all_to_all` (or wherever you set `kernel = ...`), change:

   ```python
   kernel = _ragged_a2a_kernel_point_to_point
   ```

   to:

   ```python
   kernel = _ragged_a2a_kernel_orbit_greedy_4_4_8
   ```

   The signature is identical. If you want both kernels co-resident, route
   on a new `KernelImpl.ORBIT_GREEDY` enum value.

3. **Call `ragged_all_to_all` with a tuple `axis_name`** matching the topology
   dimension order:

   ```python
   out = ragged_all_to_all(
       x, routing_info,
       mesh=mesh,
       axis_name=("x", "y", "z"),     # tuple of 3 mesh axis names
       collective_id=k,
       ...
   )
   ```

   The reference kernel accepts `axis_name: str | tuple[str, ...]`; the
   orbit-greedy kernel *requires* a tuple of length matching the topology
   `ndim`. Assert at line 1 of the kernel catches misuse.

## Generator options reference

| Flag | Default | Meaning |
|---|---|---|
| `--slice` | required | Comma-separated topology shape, e.g. `4,4,8`. Must be in `{S, 2S}^n`. |
| `--router` | `ilp` | `ilp` (load-balanced minimal — gives lower LB) or `dor` (dimension-order). |
| `--order` | `lpt_tail_asc` | OrbitGreedy ordering. `lpt_tail_asc` achieves makespan = LB on every doc cell. |
| `--axis-names` | `x,y,z,...` | Mesh axis names baked into doc comments. Runtime is flexible (uses `axis_name[i]`). |
| `--per-step-barrier` | off | Insert dummy-DMA barriers between OrbitGreedy steps. Stricter ordering, less pipelining. |
| `--function-name` | `_ragged_a2a_kernel_orbit_greedy_<slice>` | Override the generated function name. |
| `--out` | `./pallas_kernel/_ragged_a2a_kernel_orbit_greedy_<slice>.py` | Output path. |

## Caveats / TODO

- **No TPU silicon yet.** The kernel cannot be tested locally. The generator
  output parses as Python (verified via `ast.parse`), but Pallas-specific
  semantics (semaphore behaviour, `device_id` resolution under multi-axis mesh,
  closure-capture placement of `DEST_TABLE`) need TPU validation.
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

## Related

- [docs/orbit_greedy_optimality.md](../docs/orbit_greedy_optimality.md) — the
  optimality theory (König + Smith).
- [twisted_analysis/schedules/orbit_greedy.py](../twisted_analysis/schedules/orbit_greedy.py)
  — the reference Python implementation of the OrbitGreedy schedule.
- [twisted_analysis/topology/lattice.py](../twisted_analysis/topology/lattice.py)
  — the twisted-torus topology + the `neighbor()` function the generator
  uses to bake the destination table.
