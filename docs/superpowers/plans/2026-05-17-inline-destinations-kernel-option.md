# `--inline-destinations` Kernel Generator Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--inline-destinations` flag to `pallas_kernel/gen_orbit_greedy_kernel.py` that emits a kernel variant whose per-step destinations are baked as compile-time constants in `jax.lax.switch(my_flat, _DEST_BRANCHES_k)` calls, rather than read at runtime from an SMEM `dest_table_ref` input. Then generate the cpsat_literal_warm makespan-78 kernel with the new flag and store it in `pallas_kernel/outputs/`.

**Architecture:** The flag changes three things in the generated kernel: (a) drops the `dest_table_ref` parameter from the kernel signature (no extra SMEM input needed); (b) emits, at module level, K tuples `_DEST_BRANCHES_0, ..., _DEST_BRANCHES_{K-1}`, each a length-N tuple of constant-returning lambdas built via a small `_branches(consts)` helper; (c) replaces the `dest_table_ref[my_flat, k]` lookups with `jax.lax.switch(my_flat, _DEST_BRANCHES_k)`. For the `per_step_barrier=False` path this also requires unrolling the K loop into explicit per-k statements inside `_body` (since `switch` cannot take a runtime k). For the `per_step_barrier=True` path the K loop is already Python-unrolled by `for k in step`, so we only swap the destination-lookup mechanism. The `_DEST_TABLE_NP` numpy literal is still emitted (for inspection / debug / a `build_pallas_call_kwargs()` parity) but is not used by the kernel itself.

**Tech Stack:** Python 3 source-string generation (the current generator pattern: `L: list[str] = []; L.append(...)`), pytest, no new third-party deps. The motivating hypothesis is option 5 from the post-mortem on the makespan-78 kernel's hardware measurement (`Why = 132764 gbps ≈ orbit_greedy 132758 gbps`): SMEM `DEST_TABLE` lookup may be a non-trivial fraction of the per-step critical path; inlining destinations as compile-time constants per device gives Mosaic the chance to fold them into immediate operands, eliminating that load.

---

## File Structure

**Modify:**
- `pallas_kernel/gen_orbit_greedy_kernel.py` — Add `inline_destinations: bool = False` to `generate_kernel_source`; thread through CLI as `--inline-destinations`; emit different module-level constants and kernel body when True.
- `tests/test_gen_orbit_greedy_kernel_pipeline.py` — Add 3 new tests for the inline variant.
- `pallas_kernel/README.md` — Document the new flag, add the new kernel file to the outputs list.
- `README.md` (repo root) — Add the new kernel file to the layout description.

**Create (generator output):**
- `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4.py` — Generated kernel with destinations inlined.

The plan does NOT touch:
- `twisted_analysis/` (no scheduler-side changes)
- `fixtures/` (no new schedule, reusing the existing makespan-78 fixture)
- Any other kernel outputs (the existing 4 kernels are unchanged)

---

### Task 1: Add `inline_destinations` parameter to `generate_kernel_source` and emit the inline variant

**Files:**
- Modify: `pallas_kernel/gen_orbit_greedy_kernel.py:112-389` (the `generate_kernel_source` function)
- Test: `tests/test_gen_orbit_greedy_kernel_pipeline.py`

The current `generate_kernel_source` always emits a kernel whose `_body` reads `dest_table_ref[my_flat, k]` (line 313 for `per_step_barrier=False`, lines 350+360 for `per_step_barrier=True`). The new parameter switches to per-step `jax.lax.switch` calls.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gen_orbit_greedy_kernel_pipeline.py`:

```python
def test_inline_destinations_emits_branches_and_drops_dest_table_ref():
    """With inline_destinations=True the generated source:
      - drops the `dest_table_ref` parameter from the kernel signature,
      - contains `_DEST_BRANCHES_0` ... `_DEST_BRANCHES_{K-1}` module constants,
      - contains `jax.lax.switch(my_flat, _DEST_BRANCHES_` lookups,
      - does NOT contain `dest_table_ref[my_flat, k]` lookups,
      - still parses as Python (ast.parse).
    """
    import ast
    from pallas_kernel.gen_orbit_greedy_kernel import (
        generate_kernel_source, _dest_table_and_orbit_steps_from_schedule,
    )
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy_full
    from twisted_analysis.topology import Topology, ILPRouter
    from twisted_analysis.io.routing_table import (
        save_routing_table, load_routing_table,
    )
    import tempfile, os
    t = Topology(slice=(2, 4))
    r = ILPRouter(topology=t)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        save_routing_table(t, r, tmp)
        table = load_routing_table(tmp)
    finally:
        os.unlink(tmp)
    sch = schedule_from_orbit_greedy_full(t, table)
    dt, steps = _dest_table_and_orbit_steps_from_schedule(sch, t.n_nodes)

    src = generate_kernel_source(
        slice_=(2, 4), router_name_for_doc="ILP",
        scheduler_name="orbit_greedy_full", order="lpt_tail_asc",
        per_step_barrier=False, function_name=None,
        dest_table=dt, orbit_steps=steps,
        inline_destinations=True,
    )
    ast.parse(src)  # must parse cleanly
    K = dt.shape[1]
    for k in range(K):
        assert f"_DEST_BRANCHES_{k} = _branches(" in src, \
            f"missing _DEST_BRANCHES_{k} tuple"
    assert "jax.lax.switch(my_flat, _DEST_BRANCHES_" in src, \
        "no inline switch lookup emitted"
    assert "dest_table_ref[my_flat, k]" not in src, \
        "old lookup pattern still present"
    # Signature must not include dest_table_ref:
    sig_line = next(line for line in src.splitlines()
                    if "def _ragged_a2a_kernel_" in line)
    sig_start = src.index(sig_line)
    sig_end = src.index("):", sig_start)
    sig = src[sig_start:sig_end]
    assert "dest_table_ref" not in sig, \
        "dest_table_ref still in kernel signature"


def test_inline_destinations_per_step_barrier_true_also_works():
    """The inline variant must also work in per_step_barrier=True mode."""
    import ast
    from pallas_kernel.gen_orbit_greedy_kernel import (
        generate_kernel_source, _dest_table_and_orbit_steps_from_schedule,
    )
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy_full
    from twisted_analysis.topology import Topology, ILPRouter
    from twisted_analysis.io.routing_table import (
        save_routing_table, load_routing_table,
    )
    import tempfile, os
    t = Topology(slice=(2, 4))
    r = ILPRouter(topology=t)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        save_routing_table(t, r, tmp)
        table = load_routing_table(tmp)
    finally:
        os.unlink(tmp)
    sch = schedule_from_orbit_greedy_full(t, table)
    dt, steps = _dest_table_and_orbit_steps_from_schedule(sch, t.n_nodes)

    src = generate_kernel_source(
        slice_=(2, 4), router_name_for_doc="ILP",
        scheduler_name="orbit_greedy_full", order="lpt_tail_asc",
        per_step_barrier=True, function_name=None,
        dest_table=dt, orbit_steps=steps,
        inline_destinations=True,
    )
    ast.parse(src)
    assert "dest_table_ref[my_flat, k]" not in src
    assert "jax.lax.switch(my_flat, _DEST_BRANCHES_" in src


def test_inline_destinations_default_false_preserves_old_behavior():
    """Without the flag, the source must be byte-identical to the current
    output (regression guard — the refactor must not change the default
    code path)."""
    from pallas_kernel.gen_orbit_greedy_kernel import (
        generate_kernel_source, _dest_table_and_orbit_steps_from_schedule,
    )
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy_full
    from twisted_analysis.topology import Topology, ILPRouter
    from twisted_analysis.io.routing_table import (
        save_routing_table, load_routing_table,
    )
    import tempfile, os
    t = Topology(slice=(2, 4))
    r = ILPRouter(topology=t)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        save_routing_table(t, r, tmp)
        table = load_routing_table(tmp)
    finally:
        os.unlink(tmp)
    sch = schedule_from_orbit_greedy_full(t, table)
    dt, steps = _dest_table_and_orbit_steps_from_schedule(sch, t.n_nodes)

    src_default = generate_kernel_source(
        slice_=(2, 4), router_name_for_doc="ILP",
        scheduler_name="orbit_greedy_full", order="lpt_tail_asc",
        per_step_barrier=False, function_name=None,
        dest_table=dt, orbit_steps=steps,
    )
    src_explicit = generate_kernel_source(
        slice_=(2, 4), router_name_for_doc="ILP",
        scheduler_name="orbit_greedy_full", order="lpt_tail_asc",
        per_step_barrier=False, function_name=None,
        dest_table=dt, orbit_steps=steps,
        inline_destinations=False,
    )
    assert src_default == src_explicit, "default arg must not change output"
    assert "dest_table_ref[my_flat, k]" in src_default, \
        "default code path lost the SMEM lookup"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_gen_orbit_greedy_kernel_pipeline.py -v -x -k inline_destinations`

Expected: 3 tests FAIL with `TypeError: generate_kernel_source() got an unexpected keyword argument 'inline_destinations'`.

- [ ] **Step 3: Add the parameter to `generate_kernel_source` signature**

In `pallas_kernel/gen_orbit_greedy_kernel.py`, change the function signature (around line 112) from:

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
) -> str:
```

to:

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
) -> str:
```

- [ ] **Step 4: Add helper to emit per-step branch tuples**

Right above `generate_kernel_source` (i.e., between `_dest_table_literal` and `generate_kernel_source`, around line 107), add:

```python
def _dest_branches_literal(table: np.ndarray) -> str:
    """Emit per-step branch-tuple constants: _DEST_BRANCHES_<k> = _branches((...)).

    `table` is the [N, K] destination table; for each step k we emit a length-N
    tuple of integer destinations indexed by device id (my_flat). The kernel
    body calls `_branches(...)` to wrap them in constant-returning lambdas
    suitable for `jax.lax.switch`.
    """
    N, K = table.shape
    lines: list[str] = []
    lines.append("# ---- per-step destination branches (inline_destinations=True) ----")
    lines.append("# _branches(consts) -> tuple of N constant-returning lambdas, suitable")
    lines.append("# as the second argument to jax.lax.switch. The `c=c` default-arg")
    lines.append("# binding is required to capture per-lambda; without it every lambda")
    lines.append("# would close over the same loop variable and return the last value.")
    lines.append("def _branches(consts):")
    lines.append("    import jax.numpy as jnp")
    lines.append("    return tuple(lambda c=c: jnp.int32(c) for c in consts)")
    lines.append("")
    for k in range(K):
        col_ints = ", ".join(f"{int(v):3d}" for v in table[:, k])
        lines.append(f"_DEST_BRANCHES_{k} = _branches(({col_ints},))")
    return "\n".join(lines)
```

- [ ] **Step 5: Emit the branch constants when `inline_destinations=True`**

In `generate_kernel_source`, locate the block that emits `_DEST_TABLE_NP` and `_ORBIT_STEPS` (around lines 186-197). RIGHT AFTER that block (after the line emitting the closing `]` of `_ORBIT_STEPS` and the blank line), insert:

```python
    if inline_destinations:
        L.append(_dest_branches_literal(dest_table))
        L.append('')
```

This keeps `_DEST_TABLE_NP` in the file (for inspection/comparison) and adds the branch tuples below it.

- [ ] **Step 6: Adjust the kernel signature when `inline_destinations=True`**

In `generate_kernel_source`, locate the kernel-signature block (around lines 201-221). Currently it emits an unconditional `'    dest_table_ref,  # int32[N, K] ...'` line at the slot-6 position. Wrap that line in a conditional:

Find:

```python
    L.append('    dest_table_ref,  # int32[N, K] in SMEM — pass as extra pallas_call input')
```

Replace with:

```python
    if not inline_destinations:
        L.append('    dest_table_ref,  # int32[N, K] in SMEM — pass as extra pallas_call input')
```

- [ ] **Step 7: Update the integration docstring (top of file) when `inline_destinations=True`**

Locate the integration-instructions block (around lines 157-173). The instructions tell the user to insert an extra SMEM in_spec and pass `_DEST_TABLE_NP`. In inline mode this is wrong. Replace the docstring block with a conditional:

Find:

```python
    L.append('Integration. The kernel has the same signature as')
    L.append('`_ragged_a2a_kernel_point_to_point` PLUS one extra positional Ref')
    L.append('input `dest_table_ref` (slot 6, between `num_packets_per_group_ref`')
    L.append('and `x_ref`). To use:')
    L.append(f'  1. Copy this file next to reference_kernel.py.')
    L.append(f'  2. Set `kernel = {function_name}` in ragged_all_to_all.')
    L.append('  3. Insert an extra SMEM in_spec at slot 6 (before x):')
    L.append('         pl.BlockSpec(memory_space=pltpu.SMEM)')
    L.append('     and pass `jnp.asarray(_DEST_TABLE_NP)` as the corresponding')
    L.append('     extra positional input to the pallas_call.')
    L.append('  4. Shift `input_output_aliases` keys by +1: e.g. {7: 0} → {8: 0}.')
    L.append('  5. Call ragged_all_to_all(...) with a single flat axis name, e.g.')
    L.append('         axis_name="x"')
    L.append('     The orbit-greedy kernel does NOT decode per-axis coords; it')
    L.append('     reads `my_flat = jax.lax.axis_index(axis_name)` once.')
```

Replace with:

```python
    if inline_destinations:
        L.append('Integration (inline-destinations variant). The kernel has the SAME')
        L.append('signature as `_ragged_a2a_kernel_point_to_point` — no extra inputs.')
        L.append('Per-step destinations are baked as compile-time constants via')
        L.append('`jax.lax.switch(my_flat, _DEST_BRANCHES_k)`. To use:')
        L.append(f'  1. Copy this file next to reference_kernel.py.')
        L.append(f'  2. Set `kernel = {function_name}` in ragged_all_to_all.')
        L.append('  3. Do NOT insert an extra SMEM in_spec (this kernel has no')
        L.append('     `dest_table_ref` input).')
        L.append('  4. Keep `input_output_aliases` keys at their reference positions.')
        L.append('  5. Call ragged_all_to_all(...) with a single flat axis name, e.g.')
        L.append('         axis_name="x"')
        L.append('     The kernel reads `my_flat = jax.lax.axis_index(axis_name)` once.')
    else:
        L.append('Integration. The kernel has the same signature as')
        L.append('`_ragged_a2a_kernel_point_to_point` PLUS one extra positional Ref')
        L.append('input `dest_table_ref` (slot 6, between `num_packets_per_group_ref`')
        L.append('and `x_ref`). To use:')
        L.append(f'  1. Copy this file next to reference_kernel.py.')
        L.append(f'  2. Set `kernel = {function_name}` in ragged_all_to_all.')
        L.append('  3. Insert an extra SMEM in_spec at slot 6 (before x):')
        L.append('         pl.BlockSpec(memory_space=pltpu.SMEM)')
        L.append('     and pass `jnp.asarray(_DEST_TABLE_NP)` as the corresponding')
        L.append('     extra positional input to the pallas_call.')
        L.append('  4. Shift `input_output_aliases` keys by +1: e.g. {7: 0} → {8: 0}.')
        L.append('  5. Call ragged_all_to_all(...) with a single flat axis name, e.g.')
        L.append('         axis_name="x"')
        L.append('     The orbit-greedy kernel does NOT decode per-axis coords; it')
        L.append('     reads `my_flat = jax.lax.axis_index(axis_name)` once.')
```

- [ ] **Step 8: Replace the body's `dest_table_ref[my_flat, k]` lookups with `jax.lax.switch` when inline**

For the `per_step_barrier=False` path (around lines 307-318), the body currently is:

```python
    if not per_step_barrier:
        L.append('    # ---- main orbit loop: packet outer, OrbitGreedy order inner ----')
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

Replace the block with a conditional. Find that block and substitute:

```python
    if not per_step_barrier:
        L.append('    # ---- main orbit loop: packet outer, OrbitGreedy order inner ----')
        L.append(f'    _NUM_ORBITS = {K}')
        if inline_destinations:
            L.append('    def _body(packet_idx, _state):')
            for k in range(K):
                L.append(f'        dst_flat = jax.lax.switch(my_flat, _DEST_BRANCHES_{k})')
                L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
            L.append('        return _state')
            L.append('')
            L.append('    jax.lax.fori_loop(0, num_packets, _body, None)')
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

(The rest of the `if not per_step_barrier:` block — the `send_amount/recv_amount` drain section, lines 319-334 — is unchanged.)

- [ ] **Step 9: Replace the `per_step_barrier=True` body's lookups with `jax.lax.switch` when inline**

For the `per_step_barrier=True` path (around lines 335-378), the body has two lookups to change:

1. `_issue_orbit(k)` definition (line 349-355) uses `dest_table_ref[my_flat, k]`.
2. `_drain_step(step_indices)` definition (lines 357-371) uses `dest_table_ref[my_flat, k]` inside its loop.

Find the `else:` branch (after the `if not per_step_barrier:` block ends) and the lines that define `_issue_orbit` and `_drain_step`:

```python
    else:
        L.append('    _self_bytes = sizes_ref[my_flat]')
        L.append('    pltpu.make_async_copy(')
        L.append('        o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('        o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('        send_sem,')
        L.append('    ).wait()')
        L.append('    if axis_size_local > 1:')
        L.append('        pltpu.make_async_copy(')
        L.append('            o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('            o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('            recv_sem,')
        L.append('        ).wait()')
        L.append('')
        L.append('    def _issue_orbit(k):')
        L.append('        dst_flat = dest_table_ref[my_flat, k]')
        L.append('        dst_dev = {axis_name: dst_flat}')
        L.append('        def _pb(packet_idx, _state):')
        L.append('            _issue_packet(packet_idx, dst_flat, dst_dev)')
        L.append('            return _state')
        L.append('        jax.lax.fori_loop(0, num_packets, _pb, None)')
        L.append('')
        L.append('    def _drain_step(step_indices):')
        L.append('        cum = 0')
        L.append('        for k in step_indices:')
        L.append('            cum = cum + sizes_ref[dest_table_ref[my_flat, k]]')
        L.append('        pltpu.make_async_copy(')
        L.append('            o_ref.at[pl.ds(0, cum)],')
        L.append('            o_ref.at[pl.ds(0, cum)],')
        L.append('            send_sem,')
        L.append('        ).wait()')
        L.append('        if axis_size_local > 1:')
        L.append('            pltpu.make_async_copy(')
        L.append('                o_ref.at[pl.ds(0, cum)],')
        L.append('                o_ref.at[pl.ds(0, cum)],')
        L.append('                recv_sem,')
        L.append('            ).wait()')
        L.append('')
        for t, step in enumerate(orbit_steps):
            L.append(f'    # ---- OrbitGreedy step {t} ({len(step)} orbit(s)) ----')
            for k in step:
                L.append(f'    _issue_orbit({k})')
            L.append(f'    _drain_step({step!r})')
            L.append('')
```

Replace the `_issue_orbit` definition, the `_drain_step` definition, and the per-step emission loop with the inline-aware versions. The cleanest change: when `inline_destinations`, redefine `_issue_orbit` to take a branches tuple, redefine `_drain_step` to take a list of branches tuples, and emit per-step calls with branch-tuple identifiers. Substitute the block from `L.append('    def _issue_orbit(k):')` through the end of the per-step `for t, step` loop with:

```python
        if inline_destinations:
            L.append('    def _issue_orbit_inlined(branches):')
            L.append('        dst_flat = jax.lax.switch(my_flat, branches)')
            L.append('        dst_dev = {axis_name: dst_flat}')
            L.append('        def _pb(packet_idx, _state):')
            L.append('            _issue_packet(packet_idx, dst_flat, dst_dev)')
            L.append('            return _state')
            L.append('        jax.lax.fori_loop(0, num_packets, _pb, None)')
            L.append('')
            L.append('    def _drain_step_inlined(branches_list):')
            L.append('        cum = 0')
            L.append('        for branches in branches_list:')
            L.append('            dst_idx = jax.lax.switch(my_flat, branches)')
            L.append('            cum = cum + sizes_ref[dst_idx]')
            L.append('        pltpu.make_async_copy(')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            send_sem,')
            L.append('        ).wait()')
            L.append('        if axis_size_local > 1:')
            L.append('            pltpu.make_async_copy(')
            L.append('                o_ref.at[pl.ds(0, cum)],')
            L.append('                o_ref.at[pl.ds(0, cum)],')
            L.append('                recv_sem,')
            L.append('            ).wait()')
            L.append('')
            for t, step in enumerate(orbit_steps):
                L.append(f'    # ---- OrbitGreedy step {t} ({len(step)} orbit(s)) ----')
                for k in step:
                    L.append(f'    _issue_orbit_inlined(_DEST_BRANCHES_{k})')
                branches_args = ", ".join(f"_DEST_BRANCHES_{k}" for k in step)
                L.append(f'    _drain_step_inlined(({branches_args},))')
                L.append('')
        else:
            L.append('    def _issue_orbit(k):')
            L.append('        dst_flat = dest_table_ref[my_flat, k]')
            L.append('        dst_dev = {axis_name: dst_flat}')
            L.append('        def _pb(packet_idx, _state):')
            L.append('            _issue_packet(packet_idx, dst_flat, dst_dev)')
            L.append('            return _state')
            L.append('        jax.lax.fori_loop(0, num_packets, _pb, None)')
            L.append('')
            L.append('    def _drain_step(step_indices):')
            L.append('        cum = 0')
            L.append('        for k in step_indices:')
            L.append('            cum = cum + sizes_ref[dest_table_ref[my_flat, k]]')
            L.append('        pltpu.make_async_copy(')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            send_sem,')
            L.append('        ).wait()')
            L.append('        if axis_size_local > 1:')
            L.append('            pltpu.make_async_copy(')
            L.append('                o_ref.at[pl.ds(0, cum)],')
            L.append('                o_ref.at[pl.ds(0, cum)],')
            L.append('                recv_sem,')
            L.append('            ).wait()')
            L.append('')
            for t, step in enumerate(orbit_steps):
                L.append(f'    # ---- OrbitGreedy step {t} ({len(step)} orbit(s)) ----')
                for k in step:
                    L.append(f'    _issue_orbit({k})')
                L.append(f'    _drain_step({step!r})')
                L.append('')
```

The `(_DEST_BRANCHES_5,)` (trailing comma) syntax matters when `step` has length 1 — without the trailing comma `(_DEST_BRANCHES_5)` would be a plain identifier, not a tuple.

- [ ] **Step 10: Update `build_pallas_call_kwargs()` for inline mode**

Locate the helper at the bottom of the function (around lines 381-388). Replace:

```python
    L.append('')
    L.append('def build_pallas_call_kwargs():')
    L.append('    """Helper for inserting _DEST_TABLE_NP as an extra pallas_call input."""')
    L.append('    import jax.numpy as jnp')
    L.append('    return {')
    L.append('        "dest_table": jnp.asarray(_DEST_TABLE_NP),')
    L.append('        "extra_in_spec": pl.BlockSpec(memory_space=pltpu.SMEM),')
    L.append('        "input_output_aliases_shift": 1,')
    L.append('    }')
```

with:

```python
    L.append('')
    if inline_destinations:
        L.append('def build_pallas_call_kwargs():')
        L.append('    """Inline-destinations variant: no extra pallas_call input needed."""')
        L.append('    return {')
        L.append('        "dest_table": None,')
        L.append('        "extra_in_spec": None,')
        L.append('        "input_output_aliases_shift": 0,')
        L.append('    }')
    else:
        L.append('def build_pallas_call_kwargs():')
        L.append('    """Helper for inserting _DEST_TABLE_NP as an extra pallas_call input."""')
        L.append('    import jax.numpy as jnp')
        L.append('    return {')
        L.append('        "dest_table": jnp.asarray(_DEST_TABLE_NP),')
        L.append('        "extra_in_spec": pl.BlockSpec(memory_space=pltpu.SMEM),')
        L.append('        "input_output_aliases_shift": 1,')
        L.append('    }')
```

- [ ] **Step 11: Run all tests**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_gen_orbit_greedy_kernel_pipeline.py -v`

Expected: all pre-existing tests still PASS (Step 11 verifies the regression-guard test in particular: `test_inline_destinations_default_false_preserves_old_behavior`). The 3 new tests PASS as well.

- [ ] **Step 12: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add pallas_kernel/gen_orbit_greedy_kernel.py tests/test_gen_orbit_greedy_kernel_pipeline.py
git commit -m "gen_orbit_greedy_kernel: add inline_destinations option (bake per-step dests as compile-time switch branches)"
```

---

### Task 2: Add `--inline-destinations` CLI flag

**Files:**
- Modify: `pallas_kernel/gen_orbit_greedy_kernel.py:409+` (the `main()` argument parser)

- [ ] **Step 1: Add the CLI flag to the argparse setup**

In `main()`, after the existing kwarg-style flags (e.g., near `--per-step-barrier`), add:

```python
    p.add_argument(
        "--inline-destinations",
        action="store_true",
        help="Bake per-step destinations into the kernel as compile-time "
             "switch branches (jax.lax.switch(my_flat, _DEST_BRANCHES_k)) "
             "instead of SMEM lookup. Drops the dest_table_ref input. "
             "Larger generated file but eliminates the per-step SMEM load "
             "from the inner critical path.",
    )
```

To find the right spot: locate `p.add_argument("--per-step-barrier"` and add the new flag just below it.

- [ ] **Step 2: Thread the flag through to `generate_kernel_source`**

In `main()`, find the call site of `generate_kernel_source(...)` (search for the function-call), and add `inline_destinations=args.inline_destinations` to the call's kwargs.

- [ ] **Step 3: Verify the CLI accepts the flag**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py --help 2>&1 | grep inline-destinations
```

Expected: a line `--inline-destinations` appears in the help output.

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add pallas_kernel/gen_orbit_greedy_kernel.py
git commit -m "gen_orbit_greedy_kernel CLI: add --inline-destinations flag"
```

---

### Task 3: Generate the cpsat_literal_warm inline kernel

**Files (generator output):**
- Create: `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4.py`

- [ ] **Step 1: Run the generator with `--inline-destinations` and `--schedule-in` pointing at the makespan-78 fixture**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
.venv/bin/python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json \
    --inline-destinations \
    --function-name _ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4.py
```

Expected output (last few lines):
```
[2/4] loaded schedule    fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json (16256 entries)
[3/4] verified schedule  (16256 flows, 0 violations)
[4/4] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4.py (<size> bytes)
```

If the generator script does not accept `--function-name` exactly as written (some scripts use `--function_name` or only set it automatically), inspect the help with `--help` and use the actual flag name. If a function-name override is unavailable, accept the default name `_ragged_a2a_kernel_orbit_greedy_8_4_4` — the file path matters more than the inner function name, but renaming via Edit after generation is acceptable too.

- [ ] **Step 2: Verify the generated kernel parses, doesn't reference `dest_table_ref`, contains `_DEST_BRANCHES_`**

```bash
.venv/bin/python -c "
import ast
src = open('pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4.py').read()
ast.parse(src)
assert 'dest_table_ref[my_flat, k]' not in src, 'old lookup pattern still present'
assert 'jax.lax.switch(my_flat, _DEST_BRANCHES_' in src, 'no inline switch'
assert '_DEST_BRANCHES_0 = _branches(' in src, 'no branch tuples'
print('OK: inline kernel verified')
print(f'  size: {len(src):,} bytes')
print(f'  lines: {src.count(chr(10)):,}')
"
```

Expected: `OK: inline kernel verified` plus the file's size (likely 200–800 KB) and line count.

- [ ] **Step 3: Commit the new kernel file**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_inline_8_4_4.py
git commit -m "kernel: generate cpsat_literal_warm makespan-78 with inline destinations"
```

---

### Task 4: Update READMEs to document the new flag and kernel file

**Files:**
- Modify: `pallas_kernel/README.md`
- Modify: `README.md` (repo root)

- [ ] **Step 1: Add the new kernel to `pallas_kernel/README.md` Files table**

Locate the row that currently reads:

```markdown
| `outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py` | Generator output. One file per (topology, router, scheduler, order) combination. Current outputs: `orbit_greedy_8_4_4.py`, `orbit_greedy_full_8_4_4.py`, `literal_greedy_8_4_4.py`, and `cpsat_literal_warm_8_4_4.py` (makespan-78, current production recommendation for the loaded 8×4×4 routing). |
```

Replace with:

```markdown
| `outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py` | Generator output. One file per (topology, router, scheduler, order) combination. Current outputs: `orbit_greedy_8_4_4.py`, `orbit_greedy_full_8_4_4.py`, `literal_greedy_8_4_4.py`, `cpsat_literal_warm_8_4_4.py` (makespan-78 production recommendation for the loaded 8×4×4 routing; SMEM `dest_table_ref` input), and `cpsat_literal_warm_inline_8_4_4.py` (same schedule with destinations baked as compile-time `jax.lax.switch` branches via `--inline-destinations`; no SMEM input). |
```

- [ ] **Step 2: Add `--inline-destinations` to the `pallas_kernel/README.md` Generator options table**

Locate the "Generator options reference" table (around the bottom of the file, the section starting `| Flag | Default | Meaning |`). Add a new row at the end of that table:

```markdown
| `--inline-destinations` | off | Bake per-step destinations into the kernel as compile-time `jax.lax.switch(my_flat, _DEST_BRANCHES_k)` branches instead of an SMEM `dest_table_ref` input. Drops the extra pallas_call input. Larger generated file but eliminates the per-step SMEM load from the inner critical path. Used to test whether SMEM DEST_TABLE lookup is a real wall-clock bottleneck on TPU. |
```

- [ ] **Step 3: Add an example invocation to `pallas_kernel/README.md`**

In the "Example invocations" code block (the section that shows the four current `python pallas_kernel/gen_orbit_greedy_kernel.py ...` commands), append after the first (precomputed makespan-78) example:

```bash
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
```

- [ ] **Step 4: Add the new kernel file to root `README.md` layout description**

Locate the line in the root `README.md` Layout section that currently reads:

```markdown
- `pallas_kernel/` — Pallas TPU kernel generator (consumes a routing table + schedule, emits `outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py`). Current outputs include `orbit_greedy_8_4_4`, `orbit_greedy_full_8_4_4`, `literal_greedy_8_4_4`, and `cpsat_literal_warm_8_4_4` (the makespan-78 production recommendation for the loaded 8×4×4 routing).
```

Replace with:

```markdown
- `pallas_kernel/` — Pallas TPU kernel generator (consumes a routing table + schedule, emits `outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py`). Current outputs include `orbit_greedy_8_4_4`, `orbit_greedy_full_8_4_4`, `literal_greedy_8_4_4`, `cpsat_literal_warm_8_4_4` (the makespan-78 production recommendation for the loaded 8×4×4 routing), and `cpsat_literal_warm_inline_8_4_4` (same schedule with `--inline-destinations`: per-step destinations baked as `jax.lax.switch` branches instead of an SMEM `dest_table_ref` input).
```

- [ ] **Step 5: Commit the README updates**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add pallas_kernel/README.md README.md
git commit -m "docs: document --inline-destinations flag and the cpsat_literal_warm_inline kernel"
```

---

## Post-implementation hand-off

After Task 4, the work is complete. Tests pass; the new kernel file is committed under `pallas_kernel/outputs/`; both READMEs document the flag and the file. The implementer-or-controller may then invoke `superpowers:finishing-a-development-branch` to merge / PR / hand off.

The inline kernel's wall-clock performance vs. the SMEM-table variant cannot be tested without TPU silicon — that comparison is the empirical question the flag exists to answer, and is out of scope for this plan.
