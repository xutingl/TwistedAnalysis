# Dispatch-Path Improvements (Options A and B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `pallas_kernel/gen_orbit_greedy_kernel.py` with two new orthogonal code-generation options that target the kernel's scalar dispatch path: **Option A** (packed per-row state precompute, replaces N×3 dependent SMEM reads with one combined precompute pass + N×1 indexed reads in the hot loop) and **Option B** (pipelined semaphore waits, drains the DMA semaphore after every batch of `B` issued DMAs instead of only at the end). Then generate two new `cpsat_literal_warm` variants for TPU measurement.

**Architecture:** Two independent CLI flags on the existing kernel generator (`--packed-state` for Option A, `--wait-batch-size N` for Option B). Each flag changes the emitted kernel source in a localized way: Option A inserts a preamble fori_loop that builds a per-source packed state array `_my_state[K, 4]` carrying `(dst, total_size, input_offset, output_offset)` for each orbit `k`, then rewrites the hot loop to read from it; Option B wraps the existing flat hot loop in an outer batched-fori_loop that issues `B` DMAs then drains via `make_async_copy(...).wait()` on the running cumulative byte counter. Flags are independent so future plans can combine them.

**Tech Stack:** Python 3.11, JAX Pallas TPU kernels, the existing `_ragged_a2a_kernel_orbit_greedy_*` codegen pattern. Tests use structural assertions on the generated `.py` source (regex / substring matching + AST parse) because we cannot run TPU kernels in this environment.

---

## Background

**The current per-step scalar work in the `cpsat_literal_warm` kernel** (see [`pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py:319-356`](pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py#L319-L356)):

```python
def _body(i, _state):
    packet_idx = lax.div(i, _NUM_ORBITS)         # 1 scalar div
    k = lax.rem(i, _NUM_ORBITS)                  # 1 scalar rem
    dst_flat = dest_table_ref[my_flat, k]         # SMEM read #1
    _issue_packet(packet_idx, dst_flat, ...)
    return _state

def _issue_packet(packet_idx, group_idx, dst_device_id):
    size = lax.min(packet_size,
                   lax.max(sizes_ref[group_idx] - packet_idx * packet_size, 0))  # SMEM read #2
    input_offset = input_offsets_ref[group_idx] + packet_idx * packet_size       # SMEM read #3
    output_offset = output_offsets_ref[group_idx] + packet_idx * packet_size      # SMEM read #4
    ...
    pltpu.make_async_remote_copy(...).start()
```

Per iteration: 4 dependent SMEM reads, of which 3 are keyed on `dst_flat` (= the destination retrieved from the first read). The total inner-loop count is `(N-1) × num_packets`; for N=128 with 4 packets per flow that's 508 iterations per device.

**The wait pattern**: all DMAs share `send_sem` and `recv_sem`; the only `.wait()` is at the very end, on the cumulative send/recv totals. So the kernel is already maximally pipelined at the issue point. Option B asks the inverse question: is the pipeline *too* deep (in-flight queue overflow / back-pressure)?

## Empirical signals motivating the work

- Inline kernel = **−50%** measured throughput → the scalar path costs about half the wall-clock somewhere; the per-step SMEM reads + arithmetic are the largest candidate.
- 64 KB packets = **−7.1%** measured throughput → VMEM/pipeline-depth is constrained on the large-payload side; Option B's intermediate drains might let the pipeline reset cleanly between batches.
- All scheduled variants (cpsat_warm makespan 78, K=2 makespan 92, K=1 makespan 145) measure within 0.1% of each other → schedule structure is not the bottleneck; per-DMA setup is.

## File structure

| File | Responsibility |
|---|---|
| `pallas_kernel/gen_orbit_greedy_kernel.py` (modified) | Add `--packed-state` and `--wait-batch-size` CLI flags; thread them through `generate_kernel_source()`; emit the new code blocks when enabled |
| `tests/test_gen_kernel_options.py` (new) | Structural assertions on the generated source (e.g., presence of `_my_state` preamble, presence of per-batch `.wait()` calls) — does NOT run kernels on TPU |
| `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_packed_8_4_4.py` (new, auto-generated) | Kernel variant with Option A only |
| `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_pipelined_8_4_4.py` (new, auto-generated) | Kernel variant with Option B only |
| `pallas_kernel/README.md` (modified) | Document the two new flags and what they do |
| `README.md` (root, modified) | Add the two new kernels to the TPU-measurement table as "to be measured" entries |

The two CLI flags are independent (orthogonal). The plan generates one kernel with A only and one with B only, but they can be combined in a future probe by passing both flags.

---

### Task 1: Branch setup + add CLI flag stubs (no-ops)

**Files:**
- Modify: `pallas_kernel/gen_orbit_greedy_kernel.py:555-563` (CLI section near `--inline-destinations`)

- [ ] **Step 1: Create feature branch**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git checkout -b feature/dispatch-path-improvements
```

- [ ] **Step 2: Add `--packed-state` and `--wait-batch-size` flags as no-op stubs**

Edit `pallas_kernel/gen_orbit_greedy_kernel.py`. Find the `--inline-destinations` argparse block and add the two new flags directly after it (around line 563):

```python
    p.add_argument(
        "--packed-state",
        action="store_true",
        help="Option A: precompute a per-source packed state array "
             "_my_state[K, 4] holding (dst, sizes_ref[dst], "
             "input_offsets_ref[dst], output_offsets_ref[dst]) for each orbit k, "
             "then rewrite the hot loop to read from it. Trades a one-time "
             "K-iteration preamble for an N(N-1)*num_packets-iteration savings "
             "of 3 dependent SMEM reads per inner-loop step.",
    )
    p.add_argument(
        "--wait-batch-size",
        type=int,
        default=0,
        help="Option B: drain send/recv semaphores after every B issued DMAs "
             "using make_async_copy(cumulative_bytes).wait(). 0 (default) means "
             "no intermediate drain (current behavior, all DMAs issued before "
             "single final wait). Typical experiment values: 127 (= N-1, one "
             "drain per packet_idx) or 64 (half-N).",
    )
```

Also add these to the function signature of `generate_kernel_source(...)` near line 135 with defaults that preserve current behavior:

```python
def generate_kernel_source(
    *,
    slice_: tuple[int, ...],
    router_name_for_doc: str,
    scheduler_name: str,
    order: str,
    per_step_barrier: bool,
    function_name: str | None,
    dest_table: np.ndarray,
    orbit_steps: list[list[int]],
    inline_destinations: bool = False,
    packed_state: bool = False,
    wait_batch_size: int = 0,
) -> str:
```

And pass them through in `main()` near line 699 where `generate_kernel_source(...)` is called:

```python
    src = generate_kernel_source(
        slice_=slice_,
        router_name_for_doc=router_doc,
        scheduler_name=scheduler_for_doc,
        order=args.order,
        per_step_barrier=args.per_step_barrier,
        function_name=args.function_name,
        dest_table=dest_table,
        orbit_steps=orbit_steps,
        inline_destinations=args.inline_destinations,
        packed_state=args.packed_state,
        wait_batch_size=args.wait_batch_size,
    )
```

- [ ] **Step 3: Verify the flags parse**

```bash
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py --help 2>&1 | grep -E "packed-state|wait-batch-size"
```

Expected output: two lines mentioning `--packed-state` and `--wait-batch-size N`.

- [ ] **Step 4: Verify generator still works with NO new flags (preserves current behavior)**

Re-run an existing kernel generation and check the output is identical:

```bash
mkdir -p /tmp/gen_check
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json \
    --out /tmp/gen_check/kernel_baseline.py
diff /tmp/gen_check/kernel_baseline.py pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py | head -5
```

Expected: no differences (the file shipped on main was generated this way).

- [ ] **Step 5: Commit**

```bash
git add pallas_kernel/gen_orbit_greedy_kernel.py
git commit -m "feat(gen): add --packed-state and --wait-batch-size CLI stubs (no-op)"
```

---

### Task 2: Test for Option A structural changes

**Files:**
- Create: `tests/test_gen_kernel_options.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gen_kernel_options.py`:

```python
"""Structural tests for the new --packed-state and --wait-batch-size kernel-gen options.

These tests do NOT run kernels on TPU; they verify the GENERATED SOURCE has the
expected structural markers (preamble loops, batched waits). TPU correctness must
be verified out-of-band by the operator.
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pallas_kernel.gen_orbit_greedy_kernel import generate_kernel_source


def _tiny_dest_table(n: int = 4):
    """4-device round-robin dest table, shape (4, 3)."""
    K = n - 1
    table = np.zeros((n, K), dtype=np.int32)
    for s in range(n):
        col = 0
        for d in range(n):
            if d == s:
                continue
            table[s, col] = d
            col += 1
    return table, [[0], [1], [2]]  # trivial orbit_steps with one orbit per step


def _gen(**overrides):
    table, steps = _tiny_dest_table()
    defaults = dict(
        slice_=(4,),
        router_name_for_doc="test",
        scheduler_name="test",
        order="lpt",
        per_step_barrier=False,
        function_name="_test_kernel",
        dest_table=table,
        orbit_steps=steps,
        inline_destinations=False,
        packed_state=False,
        wait_batch_size=0,
    )
    defaults.update(overrides)
    return generate_kernel_source(**defaults)


def test_packed_state_emits_preamble_table():
    """--packed-state must emit a _my_state preamble fori_loop that fills the
    per-source packed state from dest_table_ref + sizes/input/output refs."""
    src = _gen(packed_state=True)
    assert "_my_state" in src, "packed_state must declare a _my_state variable"
    assert "input_offsets_ref" in src
    assert "sizes_ref" in src
    assert "output_offsets_ref" in src
    # The preamble must run BEFORE the main hot loop. We assert that the
    # _my_state build appears in the source earlier than the main fori_loop's
    # `_body` definition.
    state_pos = src.find("_my_state")
    body_pos = src.find("def _body(")
    assert 0 <= state_pos < body_pos, (
        "preamble _my_state initialization must precede the main loop"
    )


def test_packed_state_hot_loop_reads_packed_array():
    """In the main hot loop, the per-step SMEM reads must be replaced with
    reads from _my_state."""
    src = _gen(packed_state=True)
    # Find the main `_body` definition block
    body_start = src.find("def _body(")
    assert body_start != -1
    # Find the next top-level function or end of _body block (use heuristic:
    # _body is followed by `jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, ...)`)
    body_end = src.find("jax.lax.fori_loop(0, _NUM_ORBITS * num_packets", body_start)
    assert body_end != -1, "could not locate main fori_loop call after _body"
    body_block = src[body_start:body_end]
    # The body must reference _my_state to pick up at least one of the packed fields
    assert "_my_state" in body_block, (
        "main loop body must read from _my_state when packed_state=True"
    )


def test_packed_state_output_parses_as_python():
    """Generated source must be syntactically valid Python."""
    src = _gen(packed_state=True)
    ast.parse(src)


def test_packed_state_off_emits_old_pattern():
    """Without --packed-state, the hot loop must still use the old direct
    SMEM-lookup pattern."""
    src = _gen(packed_state=False)
    assert "dest_table_ref[my_flat, k]" in src, (
        "default mode must still read dst directly from dest_table_ref"
    )
    assert "_my_state" not in src
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_gen_kernel_options.py -v
```

Expected: 4 tests, with `test_packed_state_emits_preamble_table`, `test_packed_state_hot_loop_reads_packed_array`, and `test_packed_state_output_parses_as_python` all FAILING (since packed_state currently does nothing) and `test_packed_state_off_emits_old_pattern` PASSING (current behavior is the "off" case).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gen_kernel_options.py
git commit -m "test(gen): structural tests for --packed-state option (failing)"
```

---

### Task 3: Implement Option A code generation

**Files:**
- Modify: `pallas_kernel/gen_orbit_greedy_kernel.py:289-374` (kernel body emission section)

- [ ] **Step 1: Add the preamble emission in the default (non-`per_step_barrier`, non-`inline_destinations`) code path**

Edit `pallas_kernel/gen_orbit_greedy_kernel.py` and locate the section that emits the main loop, around line 356-374. The current code emits:

```python
        L.append(f'    _NUM_ORBITS = {K}')
        L.append('    def _body(i, _state):')
        L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
        L.append('        k = lax.rem(i, _NUM_ORBITS)')
        L.append('        dst_flat = dest_table_ref[my_flat, k]')
        L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
        L.append('        return _state')
        L.append('')
        L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
```

Replace this block with a conditional: if `packed_state` is False emit the existing block; if True emit the new block. The full replacement:

```python
        L.append(f'    _NUM_ORBITS = {K}')
        if packed_state:
            # --- Option A: build per-source packed state, then read it in hot loop ---
            L.append('')
            L.append('    # ---- Option A: per-source packed state preamble ----')
            L.append('    # Build _my_state[k] = (dst, sizes_ref[dst], input_offsets_ref[dst],')
            L.append('    #                       output_offsets_ref[dst]) for each orbit k.')
            L.append('    # Trades a K-iteration one-time fori_loop for the elimination of')
            L.append('    # 3 dependent SMEM reads per main-loop iteration.')
            L.append('    import jax.numpy as jnp')
            L.append('    _initial_state = jnp.zeros((_NUM_ORBITS, 4), dtype=jnp.int32)')
            L.append('    def _build_state(k, state):')
            L.append('        dst = dest_table_ref[my_flat, k]')
            L.append('        state = state.at[k, 0].set(dst)')
            L.append('        state = state.at[k, 1].set(sizes_ref[dst])')
            L.append('        state = state.at[k, 2].set(input_offsets_ref[dst])')
            L.append('        state = state.at[k, 3].set(output_offsets_ref[dst])')
            L.append('        return state')
            L.append('    _my_state = jax.lax.fori_loop(0, _NUM_ORBITS, _build_state, _initial_state)')
            L.append('')
            L.append('    # ---- main loop reads packed state ----')
            L.append('    def _body(i, _state):')
            L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
            L.append('        k = lax.rem(i, _NUM_ORBITS)')
            L.append('        dst_flat = _my_state[k, 0]')
            L.append('        size_total = _my_state[k, 1]')
            L.append('        in_off_base = _my_state[k, 2]')
            L.append('        out_off_base = _my_state[k, 3]')
            L.append('        size = lax.min(packet_size, lax.max(size_total - packet_idx * packet_size, 0))')
            L.append('        input_offset = in_off_base + packet_idx * packet_size')
            L.append('        output_offset = out_off_base + packet_idx * packet_size')
            L.append('        @pl.when(size > 0)')
            L.append('        def _():')
            L.append('            if axis_size_local > 1:')
            L.append('                pltpu.make_async_remote_copy(')
            L.append('                    x_ref.at[pl.ds(input_offset, size)],')
            L.append('                    o_ref.at[pl.ds(output_offset, size)],')
            L.append('                    device_id={axis_name: dst_flat},')
            L.append('                    send_sem=send_sem,')
            L.append('                    recv_sem=recv_sem,')
            L.append('                ).start()')
            L.append('            else:')
            L.append('                pltpu.make_async_copy(')
            L.append('                    x_ref.at[pl.ds(input_offset, size)],')
            L.append('                    o_ref.at[pl.ds(output_offset, size)],')
            L.append('                    sem=send_sem,')
            L.append('                ).start()')
            L.append('        return _state')
            L.append('')
            L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
        else:
            L.append('    def _body(i, _state):')
            L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
            L.append('        k = lax.rem(i, _NUM_ORBITS)')
            L.append('        dst_flat = dest_table_ref[my_flat, k]')
            L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
            L.append('        return _state')
            L.append('')
            L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
```

**Note**: `--packed-state` only works in the non-`per_step_barrier` and non-`inline_destinations` code path (i.e., the default `cpsat_literal_warm` pattern). Add a compatibility check near the top of `generate_kernel_source` (right after the assertions):

```python
    if packed_state and per_step_barrier:
        raise ValueError("--packed-state is incompatible with --per-step-barrier")
    if packed_state and inline_destinations:
        raise ValueError("--packed-state is incompatible with --inline-destinations")
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_gen_kernel_options.py::test_packed_state_emits_preamble_table tests/test_gen_kernel_options.py::test_packed_state_hot_loop_reads_packed_array tests/test_gen_kernel_options.py::test_packed_state_output_parses_as_python tests/test_gen_kernel_options.py::test_packed_state_off_emits_old_pattern -v
```

Expected: 4 passed.

- [ ] **Step 3: Verify baseline kernel regeneration is still identical (no regression for default flags)**

```bash
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json \
    --out /tmp/gen_check/kernel_baseline_after_A.py
diff /tmp/gen_check/kernel_baseline_after_A.py pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py
```

Expected: no differences. (The default codepath must be byte-identical.)

- [ ] **Step 4: Commit**

```bash
git add pallas_kernel/gen_orbit_greedy_kernel.py
git commit -m "feat(gen): implement Option A --packed-state preamble + hot-loop rewrite"
```

---

### Task 4: Generate the cpsat_literal_warm packed-state kernel

**Files:**
- Create: `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_packed_8_4_4.py`

- [ ] **Step 1: Generate the kernel**

```bash
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json \
    --packed-state \
    --function-name _ragged_a2a_kernel_cpsat_literal_warm_packed_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_packed_8_4_4.py
```

Expected: 4-stage output ending with `[4/4] wrote kernel ...`.

- [ ] **Step 2: Verify the file parses and contains the expected structural markers**

```bash
.venv/bin/python -c "
import ast
src = open('pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_packed_8_4_4.py').read()
ast.parse(src)
print('parses OK')
assert '_my_state' in src, 'missing _my_state preamble'
assert '_build_state' in src, 'missing _build_state inner fn'
assert 'dest_table_ref[my_flat, k]' not in src.split('def _body')[1].split('jax.lax.fori_loop(0, _NUM_ORBITS * num_packets')[0], \
    'main loop should NOT contain direct dest_table_ref read'
print('structural OK')
"
```

Expected: prints "parses OK" then "structural OK".

- [ ] **Step 3: Commit**

```bash
git add pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_packed_8_4_4.py
git commit -m "feat(kernel): generate cpsat_literal_warm packed variant (Option A)"
```

---

### Task 5: Test for Option B structural changes

**Files:**
- Modify: `tests/test_gen_kernel_options.py` (append)

- [ ] **Step 1: Add failing tests for `--wait-batch-size`**

Append to `tests/test_gen_kernel_options.py`:

```python
def test_wait_batch_size_zero_is_unchanged():
    """wait_batch_size=0 must produce identical output to no flag."""
    src_default = _gen(wait_batch_size=0)
    src_off = _gen()
    assert src_default == src_off


def test_wait_batch_size_127_emits_outer_packet_loop_with_intermediate_drain():
    """wait_batch_size=127 wraps the main loop in an outer per-packet fori_loop
    that drains after every 127 issued DMAs."""
    src = _gen(wait_batch_size=127)
    # Must NOT use the old flat fori_loop over _NUM_ORBITS * num_packets:
    assert "jax.lax.fori_loop(0, _NUM_ORBITS * num_packets" not in src, (
        "wait_batch_size>0 must replace the flat outer loop with a batched outer loop"
    )
    # Must emit an intermediate drain wait inside the outer loop:
    assert ".wait()" in src
    # Must keep a running cumulative byte counter:
    assert "cum_bytes" in src, "expected a running cum_bytes counter for partial drain"


def test_wait_batch_size_output_parses_as_python():
    src = _gen(wait_batch_size=127)
    ast.parse(src)
```

- [ ] **Step 2: Run to verify failures**

```bash
.venv/bin/python -m pytest tests/test_gen_kernel_options.py -v -k wait_batch_size
```

Expected: `test_wait_batch_size_zero_is_unchanged` PASSES; the other two FAIL because nothing has been implemented yet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gen_kernel_options.py
git commit -m "test(gen): structural tests for --wait-batch-size option (failing)"
```

---

### Task 6: Implement Option B code generation

**Files:**
- Modify: `pallas_kernel/gen_orbit_greedy_kernel.py:355-391` (default-path code-emission section, AFTER Task 3 changes)

- [ ] **Step 1: Add compatibility check + emit batched outer loop when `wait_batch_size > 0`**

Near the top of `generate_kernel_source`, after the existing `packed_state` compatibility checks added in Task 3, add:

```python
    if wait_batch_size and per_step_barrier:
        raise ValueError("--wait-batch-size is incompatible with --per-step-barrier")
    if wait_batch_size and inline_destinations:
        raise ValueError("--wait-batch-size is incompatible with --inline-destinations")
    if wait_batch_size and packed_state:
        raise ValueError(
            "--wait-batch-size combined with --packed-state not supported in this pass; "
            "run them as separate experiments."
        )
    if wait_batch_size < 0:
        raise ValueError(f"--wait-batch-size must be >= 0, got {wait_batch_size}")
```

In the existing default-path block (the `else` branch from Task 3, where `packed_state=False`), wrap the main loop emission in another conditional on `wait_batch_size`:

Replace the existing:
```python
        else:
            L.append('    def _body(i, _state):')
            L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
            L.append('        k = lax.rem(i, _NUM_ORBITS)')
            L.append('        dst_flat = dest_table_ref[my_flat, k]')
            L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
            L.append('        return _state')
            L.append('')
            L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
```

with:

```python
        elif wait_batch_size > 0:
            # --- Option B: batched outer loop with intermediate drain ---
            L.append(f'    _WAIT_BATCH_SIZE = {wait_batch_size}')
            L.append('    # ---- Option B: batched issue with intermediate semaphore drain ----')
            L.append('    # Total inner iterations = _NUM_ORBITS * num_packets. We group them')
            L.append('    # into batches of _WAIT_BATCH_SIZE issued DMAs, draining after each.')
            L.append('    def _orbit_body(j, state):')
            L.append('        i, cum_bytes = state')
            L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
            L.append('        k = lax.rem(i, _NUM_ORBITS)')
            L.append('        dst_flat = dest_table_ref[my_flat, k]')
            L.append('        size = lax.min(')
            L.append('            packet_size,')
            L.append('            lax.max(sizes_ref[dst_flat] - packet_idx * packet_size, 0),')
            L.append('        )')
            L.append('        input_offset = input_offsets_ref[dst_flat] + packet_idx * packet_size')
            L.append('        output_offset = output_offsets_ref[dst_flat] + packet_idx * packet_size')
            L.append('        @pl.when(size > 0)')
            L.append('        def _():')
            L.append('            if axis_size_local > 1:')
            L.append('                pltpu.make_async_remote_copy(')
            L.append('                    x_ref.at[pl.ds(input_offset, size)],')
            L.append('                    o_ref.at[pl.ds(output_offset, size)],')
            L.append('                    device_id={axis_name: dst_flat},')
            L.append('                    send_sem=send_sem,')
            L.append('                    recv_sem=recv_sem,')
            L.append('                ).start()')
            L.append('            else:')
            L.append('                pltpu.make_async_copy(')
            L.append('                    x_ref.at[pl.ds(input_offset, size)],')
            L.append('                    o_ref.at[pl.ds(output_offset, size)],')
            L.append('                    sem=send_sem,')
            L.append('                ).start()')
            L.append('        return (i + 1, cum_bytes + size)')
            L.append('')
            L.append('    def _batch_body(batch_idx, cum_bytes):')
            L.append('        i_start = batch_idx * _WAIT_BATCH_SIZE')
            L.append('        # Use a fori_loop of length _WAIT_BATCH_SIZE to issue the batch')
            L.append('        # while accumulating cum_bytes:')
            L.append('        (_, cum_bytes) = jax.lax.fori_loop(')
            L.append('            0, _WAIT_BATCH_SIZE, _orbit_body, (i_start, cum_bytes)')
            L.append('        )')
            L.append('        # Drain after this batch:')
            L.append('        pltpu.make_async_copy(')
            L.append('            o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('            o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('            send_sem,')
            L.append('        ).wait()')
            L.append('        if axis_size_local > 1:')
            L.append('            pltpu.make_async_copy(')
            L.append('                o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('                o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('                recv_sem,')
            L.append('            ).wait()')
            L.append('        return cum_bytes')
            L.append('')
            L.append('    _total_iters = _NUM_ORBITS * num_packets')
            L.append('    _num_batches = _total_iters // _WAIT_BATCH_SIZE')
            L.append('    cum_bytes = jax.lax.fori_loop(0, _num_batches, _batch_body, 0)')
            L.append('    # Handle any remainder (when _total_iters is not a multiple of _WAIT_BATCH_SIZE):')
            L.append('    _remainder = _total_iters - _num_batches * _WAIT_BATCH_SIZE')
            L.append('    def _remainder_body(j, state):')
            L.append('        i, cum_bytes = state')
            L.append('        return _orbit_body(j, state)')
            L.append('    (_, cum_bytes) = jax.lax.fori_loop(')
            L.append('        0, _remainder, _remainder_body,')
            L.append('        (_num_batches * _WAIT_BATCH_SIZE, cum_bytes),')
            L.append('    )')
        else:
            L.append('    def _body(i, _state):')
            L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
            L.append('        k = lax.rem(i, _NUM_ORBITS)')
            L.append('        dst_flat = dest_table_ref[my_flat, k]')
            L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
            L.append('        return _state')
            L.append('')
            L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
```

Also, since the batched-wait version drains DMAs as it goes, the final outer drain at the end of the kernel becomes redundant for the batched path. Inspect lines ~376-391 (the final `total_send_amount_ref / total_recv_amount_ref` drain). Wrap THAT block in a `if not wait_batch_size:` guard:

```python
        if wait_batch_size:
            L.append('    # final drain not needed: all batches drained inside the loop')
            L.append('    # but still wait for any tail bytes in the remainder:')
            L.append('    pltpu.make_async_copy(')
            L.append('        o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('        o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('        send_sem,')
            L.append('    ).wait()')
            L.append('    if axis_size_local > 1:')
            L.append('        pltpu.make_async_copy(')
            L.append('            o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('            o_ref.at[pl.ds(0, cum_bytes)],')
            L.append('            recv_sem,')
            L.append('        ).wait()')
        else:
            L.append('    send_amount = total_send_amount_ref[0]')
            L.append('    recv_amount = total_recv_amount_ref[0]')
            L.append('    if enable_checks:')
            L.append('        pl.debug_check(send_amount >= 0, "send_amount<0")')
            L.append('        pl.debug_check(recv_amount >= 0, "recv_amount<0")')
            L.append('    pltpu.make_async_copy(')
            L.append('        o_ref.at[pl.ds(0, send_amount)],')
            L.append('        o_ref.at[pl.ds(0, send_amount)],')
            L.append('        send_sem,')
            L.append('    ).wait()')
            L.append('    if axis_size_local > 1:')
            L.append('        pltpu.make_async_copy(')
            L.append('            o_ref.at[pl.ds(0, recv_amount)],')
            L.append('            o_ref.at[pl.ds(0, recv_amount)],')
            L.append('            recv_sem,')
            L.append('        ).wait()')
```

(Replace the existing unconditional version of that final-drain block accordingly.)

- [ ] **Step 2: Run all tests**

```bash
.venv/bin/python -m pytest tests/test_gen_kernel_options.py -v
```

Expected: 7 passed (4 from Task 2 + 3 from Task 5).

- [ ] **Step 3: Verify baseline kernel regeneration is still identical**

```bash
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json \
    --out /tmp/gen_check/kernel_baseline_after_B.py
diff /tmp/gen_check/kernel_baseline_after_B.py pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py
```

Expected: no differences.

- [ ] **Step 4: Commit**

```bash
git add pallas_kernel/gen_orbit_greedy_kernel.py
git commit -m "feat(gen): implement Option B --wait-batch-size batched drain"
```

---

### Task 7: Generate the cpsat_literal_warm pipelined kernel

**Files:**
- Create: `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_pipelined_8_4_4.py`

- [ ] **Step 1: Generate with `--wait-batch-size 127`**

We choose batch size 127 (= N-1, one drain per packet_idx pass) as the initial experimental value. This is a single drain after each "full sweep across destinations" — semantically equivalent to per-packet-iteration semaphore reset.

```bash
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json \
    --wait-batch-size 127 \
    --function-name _ragged_a2a_kernel_cpsat_literal_warm_pipelined_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_pipelined_8_4_4.py
```

Expected: 4-stage output ending with `[4/4] wrote kernel ...`.

- [ ] **Step 2: Verify structural markers**

```bash
.venv/bin/python -c "
import ast
src = open('pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_pipelined_8_4_4.py').read()
ast.parse(src)
print('parses OK')
assert '_WAIT_BATCH_SIZE = 127' in src
assert '_batch_body' in src
assert '_orbit_body' in src
assert 'cum_bytes' in src
print('structural OK')
"
```

Expected: prints "parses OK" then "structural OK".

- [ ] **Step 3: Commit**

```bash
git add pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_pipelined_8_4_4.py
git commit -m "feat(kernel): generate cpsat_literal_warm pipelined variant (Option B, batch=127)"
```

---

### Task 8: Documentation

**Files:**
- Modify: `pallas_kernel/README.md`
- Modify: `README.md` (root)

- [ ] **Step 1: Add a section to `pallas_kernel/README.md` documenting the new flags**

Append the following section to `pallas_kernel/README.md` (read the current file first to pick a coherent insertion point — likely near the end, before any "examples" or after the existing "When to use which kernel" matrix):

```markdown
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
```

- [ ] **Step 2: Add the new kernels to the root `README.md` measurement table**

Open `README.md` and find the TPU v5e measurement table (around line 68-74). Append two new rows AFTER the existing `spread_greedy(k=2)` line:

Replace this:
```markdown
| `spread_greedy(k=2)` (2-way pipelining) | 92 | 91 | 132764 gbps | —1.3 % |
```

With:
```markdown
| `spread_greedy(k=2)` (2-way pipelining) | 92 | 91 | 132764 gbps | —1.3 % |
| `cpsat_literal_warm_packed` (Option A: packed SMEM preamble) | 78 | ≤ 78 | TBD (to measure on TPU) | TBD |
| `cpsat_literal_warm_pipelined` (Option B: batch-127 wait drain) | 78 | ≤ 78 | TBD (to measure on TPU) | TBD |
```

- [ ] **Step 3: Commit**

```bash
git add pallas_kernel/README.md README.md
git commit -m "docs: dispatch-path option A/B variants + integration notes"
```

---

## Self-review notes

**Spec coverage:**
- "Implement option A" → Tasks 2–4 (test, implement, generate variant).
- "Implement option B" → Tasks 5–7 (test, implement, generate variant).
- "Different options that the kernel generator will support" → Task 1 (CLI flag stubs); Tasks 3, 6 (implementations are independent; orthogonality enforced by the cross-incompatibility check that disallows combining A+B in this pass, marking it explicitly as future work).
- "Two new variants of the cpsat_warm schedule's kernel" → Tasks 4, 7 (generate `_packed_` and `_pipelined_` kernel files).

**Risks the implementer should flag (not pre-emptively work around):**
- The Option A code emits `jnp.zeros((_NUM_ORBITS, 4), dtype=jnp.int32)` and uses `state.at[k, X].set(...)` inside a fori_loop. Pallas may or may not lower this to in-place SMEM writes; if the compiler produces extra copies, the precompute could be more expensive than expected. Worth measuring on TPU.
- The Option B remainder loop (when `_total_iters` is not a multiple of `_WAIT_BATCH_SIZE`) shares the same `_orbit_body` to keep DRY, but uses a different state structure (`(i_start, cum_bytes)`). If the implementer encounters fori_loop state-shape complaints, restructure to a single carried tuple.
- Both options assume `axis_size_local > 1` is the runtime case. Single-device fallback uses `pltpu.make_async_copy` instead of `make_async_remote_copy`; both code paths need the same change.
- Neither flag is compatible with `--per-step-barrier` or `--inline-destinations`; the generator raises `ValueError` if combined. This is intentional — combining is out of scope for this plan.

**TPU runtime testing is out of scope.** This plan produces structurally validated kernel source files. The operator must (separately) compile and measure them on TPU v5e and append results to README.md's measurement table.
