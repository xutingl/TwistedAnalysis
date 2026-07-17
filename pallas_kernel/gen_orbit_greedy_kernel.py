"""Generator for orbit-greedy P2P AllToAll Pallas TPU kernels.

Pipeline:
    [router OR --routing-table FILE] -> routing-table JSON
                  -> schedule_from_orbit_greedy -> schedule JSON
                  -> _DEST_TABLE_NP, _ORBIT_STEPS literals
                  -> kernel .py source

All three artifacts are written to disk before the kernel is emitted, so the
intermediate routing table and schedule can be inspected independently.

Usage (CLI):

    # Generate routing-table from a router:
    python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --router ilp

    # Reuse an existing routing-table:
    python pallas_kernel/gen_orbit_greedy_kernel.py \\
        --slice 8,4,4 \\
        --routing-table fixtures/routing/routing_table_8x4x4_twist.json

Default outputs:
    routing table: ./fixtures/routing/routing_table_<slice>_<router>.json
    schedule:      ./fixtures/nonragged/schedule_<slice>_<router>_<order>.json
    kernel:        ./pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make `python pallas_kernel/gen_orbit_greedy_kernel.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import (
    save_routing_table, load_routing_table,
)
from twisted_analysis.io.schedule import (
    save_schedule,
    schedule_from_algorithm,
)
from twisted_analysis.schedules.verify import (
    schedule_step_count,
    verify_capacity,
    verify_capacity_step,
)
from twisted_analysis.topology import Topology, DORRouter, ILPRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dest_table_and_orbit_steps_from_schedule(
    schedule: list[dict], n: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Build _DEST_TABLE_NP[n, K] and _ORBIT_STEPS from schedule entries.

    Strategy:
      * Sort schedule by (round, dst). For each src, the per-round destination
        sequence becomes the columns of _DEST_TABLE_NP[src].
      * `_ORBIT_STEPS[t]` = list of column indices `k` whose hop-0 round equals
        the t-th distinct round value. Derived from src=0's row.

    Invariants the kernel actually depends on:
      1. Rounds-by-column agree across sources: for every column k, the round
         value at row src=0 equals the round at row src=s for all s. (This
         holds because all sources in the same orbit fire hop-0 at the same
         OrbitGreedy step.)
      2. Each row of `_DEST_TABLE_NP` is a permutation of `{0..n-1} \\ {src}`.

    What is *not* guaranteed (and the kernel does not need): column k may
    correspond to different orbits across sources whenever a round contains
    multiple orbits and per-source `dst` values break ties differently.
    """
    by_src: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        by_src[e["src"]].append((e["round"], e["dst"]))
    if any(len(by_src[s]) != n - 1 for s in range(n)):
        raise RuntimeError(
            f"schedule does not cover full AllToAll: "
            f"each source needs {n - 1} entries"
        )
    for s in range(n):
        by_src[s].sort()

    K = n - 1
    table = np.zeros((n, K), dtype=np.int32)
    for src in range(n):
        for k, (_round, dst) in enumerate(by_src[src]):
            table[src, k] = dst

    rounds_src0 = [r for (r, _d) in by_src[0]]
    by_step: dict[int, list[int]] = defaultdict(list)
    for k, r in enumerate(rounds_src0):
        by_step[r].append(k)
    orbit_steps = [by_step[r] for r in sorted(by_step.keys())]
    return table, orbit_steps


def _dest_table_literal(table: np.ndarray) -> str:
    """Compact module-level literal for an int32 [N, K] table."""
    rows = ["    [" + ", ".join(f"{v:3d}" for v in row) + "]"
            for row in table]
    return "np.array([\n" + ",\n".join(rows) + ",\n], dtype=np.int32)"


# ---------------------------------------------------------------------------
# Source builder
# ---------------------------------------------------------------------------

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
    lines.append("# The `c=c` default-arg binding captures per-lambda; without it every")
    lines.append("# lambda closes over the same loop variable and returns the last value.")
    lines.append("def _branches(consts):")
    lines.append("    import jax.numpy as jnp")
    lines.append("    return tuple(lambda c=c: jnp.int32(c) for c in consts)")
    lines.append("")
    for k in range(K):
        col_ints = ", ".join(f"{int(v):3d}" for v in table[:, k])
        lines.append(f"_DEST_BRANCHES_{k} = _branches(({col_ints},))")
    return "\n".join(lines)


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
    """Emit kernel source from already-computed dest_table + orbit_steps.

    Pure function: takes the schedule-derived arrays as inputs and emits the
    .py source string. Does NOT touch disk and does NOT call routers.
    """
    n_dim = len(slice_)
    N = 1
    for s in slice_:
        N *= s

    if function_name is None:
        slice_str = "_".join(str(s) for s in slice_)
        function_name = f"_ragged_a2a_kernel_orbit_greedy_{slice_str}"

    K = dest_table.shape[1]
    if K != N - 1:
        raise RuntimeError(f"Expected dest_table cols = {N - 1}; got {K}")
    if packed_state and per_step_barrier:
        raise ValueError("--packed-state is incompatible with --per-step-barrier")
    if packed_state and inline_destinations:
        raise ValueError("--packed-state is incompatible with --inline-destinations")
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
    makespan_hop0 = len(orbit_steps)

    L: list[str] = []

    L.append('"""Orbit-greedy P2P AllToAll Pallas TPU kernel.')
    L.append('')
    L.append('AUTO-GENERATED — DO NOT EDIT BY HAND.')
    L.append('')
    L.append(f'Topology:        slice={slice_}  (N={N} devices, ndim={n_dim})')
    L.append(f'Router:          {router_name_for_doc}')
    L.append(f'Scheduler:       {scheduler_name}')
    L.append(f'OrbitGreedy:     order={order!r}')
    L.append(f'Per-step barrier: {per_step_barrier}')
    L.append(f'Hop-0 steps:     {makespan_hop0}')
    L.append('')
    L.append('Generated from:  routing-table JSON + schedule JSON')
    L.append('')
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
    L.append('')
    L.append('Use `build_pallas_call_kwargs()` below for a copy-pasteable example.')
    L.append('"""')
    L.append('from __future__ import annotations')
    L.append('')
    L.append('import jax')
    L.append('import numpy as np')
    L.append('from jax import lax')
    L.append('from jax.experimental import pallas as pl')
    L.append('from jax.experimental.pallas import tpu as pltpu')
    L.append('')
    L.append('from megablox.collectives import ragged_collectives_utils  # type: ignore')
    L.append('')
    L.append('')
    L.append('# ----------------------------- baked schedule -------------------------------')
    L.append(f'_DEST_TABLE_NP = {_dest_table_literal(dest_table)}')
    L.append(f'assert _DEST_TABLE_NP.shape == ({N}, {K}), (')
    L.append('    f"_DEST_TABLE_NP shape mismatch: {_DEST_TABLE_NP.shape}"')
    L.append(')')
    L.append('')
    L.append(f'# _ORBIT_STEPS[t] = orbit indices firing at OrbitGreedy step t. '
             f'{makespan_hop0} steps total.')
    L.append('_ORBIT_STEPS = [')
    for t, step in enumerate(orbit_steps):
        L.append(f'    {step!r},  # step {t} ({len(step)} concurrent orbit(s))')
    L.append(']')
    L.append('')
    if inline_destinations:
        L.append(_dest_branches_literal(dest_table))
        L.append('')
    L.append('')

    L.append(f'def {function_name}(')
    L.append('    input_offsets_ref,')
    L.append('    output_offsets_ref,')
    L.append('    sizes_ref,')
    L.append('    total_send_amount_ref,')
    L.append('    total_recv_amount_ref,')
    L.append('    num_packets_per_group_ref,')
    if not inline_destinations:
        L.append('    dest_table_ref,  # int32[N, K] in SMEM — pass as extra pallas_call input')
    L.append('    x_ref,')
    L.append('    _,')
    L.append('    o_ref,')
    L.append('    scratch_ref,')
    L.append('    send_sem,')
    L.append('    recv_sem,')
    L.append('    scratch_sems,')
    L.append('    *,')
    L.append('    axis_name,')
    L.append('    transpose,')
    L.append('    packet_size,')
    L.append('    enable_checks: bool = False,')
    L.append('):')
    L.append(f'    """Orbit-greedy P2P AllToAll kernel for slice={slice_}.')
    L.append('')
    if inline_destinations:
        L.append('    Signature: same as `_ragged_a2a_kernel_point_to_point` — no extra inputs.')
        L.append('    Destinations are baked as compile-time constants via `jax.lax.switch`.')
        L.append('    Other differences vs reference:')
        L.append('      * Iteration order = OrbitGreedy firing order (vs rotation).')
    else:
        L.append('    Signature: same as `_ragged_a2a_kernel_point_to_point` PLUS one extra')
        L.append('    Ref input `dest_table_ref` (int32[N, K] in SMEM, slot 6). Other')
        L.append('    differences vs reference:')
        L.append('      * Iteration order = OrbitGreedy firing order (vs rotation).')
        L.append('      * Destinations are looked up in `dest_table_ref` (twist-aware).')
    L.append('      * `transpose=True` is NOT supported (would need regen).')
    L.append('      * Assumes 1 group per device (uniform AllToAll).')
    L.append('      * `axis_name` is a flat string (e.g. "x"), as in the reference.')
    L.append('    """')
    if packed_state:
        L.append('    # Option A: scratch_ref is the per-source packed state buffer')
        L.append('    # (shape (K, 4) int32 SMEM, allocated by the caller via scratch_shapes).')
    else:
        L.append('    assert scratch_ref is None')
        L.append('    del scratch_ref')
    L.append('    assert scratch_sems is None')
    L.append('    del scratch_sems')
    L.append('    assert not transpose, (')
    L.append('        "transpose=True not supported by orbit-greedy kernel; use reference."')
    L.append('    )')
    L.append('')
    L.append('    my_flat = jax.lax.axis_index(axis_name)')
    L.append('    axis_size_local = jax.lax.axis_size(axis_name)')
    L.append('')
    L.append('    num_groups = sizes_ref.shape[0]')
    L.append(f'    assert num_groups == {N}, (')
    L.append(f'        f"Expected num_groups={N} (uniform AllToAll on {slice_}); got {{num_groups}}"')
    L.append('    )')
    L.append('    groups_per_shard, r = divmod(num_groups, axis_size_local)')
    L.append('    assert r == 0 and groups_per_shard == 1, (')
    L.append('        "orbit-greedy kernel assumes 1 group per device"')
    L.append('    )')
    L.append('')
    L.append('    if axis_size_local > 1:')
    L.append('        ragged_collectives_utils.main_barrier(')
    L.append('            axis_name,')
    L.append('            barrier_type=ragged_collectives_utils.BarrierType.ALL_TO_ALL,')
    L.append('        )')
    L.append('')
    L.append('    num_packets = num_packets_per_group_ref[0]')
    L.append('')
    L.append('    def _issue_packet(packet_idx, group_idx, dst_device_id):')
    L.append('        size = lax.min(')
    L.append('            packet_size,')
    L.append('            lax.max(sizes_ref[group_idx] - packet_idx * packet_size, 0),')
    L.append('        )')
    L.append('        input_offset = input_offsets_ref[group_idx] + packet_idx * packet_size')
    L.append('        output_offset = output_offsets_ref[group_idx] + packet_idx * packet_size')
    L.append('')
    L.append('        if enable_checks:')
    L.append('            pl.debug_check(sizes_ref[group_idx] >= 0, "size<0")')
    L.append('            pl.debug_check(size >= 0, "transfer size<0")')
    L.append('            pl.debug_check(input_offset >= 0, "input_offset<0")')
    L.append('            pl.debug_check(output_offset >= 0, "output_offset<0")')
    L.append('            pl.debug_check(')
    L.append('                input_offset + size <= x_ref.shape[0],')
    L.append('                "input_offset+size > x_ref.shape[0]",')
    L.append('            )')
    L.append('            pl.debug_check(')
    L.append('                output_offset + size <= o_ref.shape[0],')
    L.append('                "output_offset+size > o_ref.shape[0]",')
    L.append('            )')
    L.append('')
    L.append('        @pl.when(size > 0)')
    L.append('        def _():')
    L.append('            if axis_size_local > 1:')
    L.append('                pltpu.make_async_remote_copy(')
    L.append('                    x_ref.at[pl.ds(input_offset, size)],')
    L.append('                    o_ref.at[pl.ds(output_offset, size)],')
    L.append('                    device_id=dst_device_id,')
    L.append('                    send_sem=send_sem,')
    L.append('                    recv_sem=recv_sem,')
    L.append('                ).start()')
    L.append('            else:')
    L.append('                pltpu.make_async_copy(')
    L.append('                    x_ref.at[pl.ds(input_offset, size)],')
    L.append('                    o_ref.at[pl.ds(output_offset, size)],')
    L.append('                    sem=send_sem,')
    L.append('                ).start()')
    L.append('')
    L.append('    def _self_body(packet_idx, _state):')
    L.append('        _issue_packet(packet_idx, my_flat, {axis_name: my_flat})')
    L.append('        return _state')
    L.append('')
    L.append('    jax.lax.fori_loop(0, num_packets, _self_body, None)')
    L.append('')

    if not per_step_barrier:
        L.append('    # ---- main orbit loop: packet outer, OrbitGreedy order inner ----')
        if inline_destinations:
            L.append('    def _body(packet_idx, _state):')
            for k in range(K):
                L.append(f'        dst_flat = jax.lax.switch(my_flat, _DEST_BRANCHES_{k})')
                L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
            L.append('        return _state')
            L.append('')
            L.append('    jax.lax.fori_loop(0, num_packets, _body, None)')
        else:
            L.append(f'    _NUM_ORBITS = {K}')
            if packed_state:
                # --- Option A: build per-source packed state in SMEM scratch ref ---
                L.append('')
                L.append('    # ---- Option A: per-source packed state preamble ----')
                L.append('    # Build scratch_ref[k] = (dst, sizes_ref[dst], input_offsets_ref[dst],')
                L.append('    #                         output_offsets_ref[dst]) for each orbit k.')
                L.append('    # scratch_ref is an SMEM scratch buffer of shape (K, 4) int32, allocated')
                L.append('    # by the caller via `scratch_shapes=[pltpu.SMEM((K, 4), jnp.int32)]` on')
                L.append('    # the pallas_call. Subscript writes into Pallas SMEM refs do NOT lower')
                L.append('    # to scatter (unlike `jnp.zeros(...).at[k].set(...)`), which is required')
                L.append('    # for TPU lowering compatibility.')
                L.append('    def _build_state(k, _):')
                L.append('        dst = dest_table_ref[my_flat, k]')
                L.append('        scratch_ref[k, 0] = dst')
                L.append('        scratch_ref[k, 1] = sizes_ref[dst]')
                L.append('        scratch_ref[k, 2] = input_offsets_ref[dst]')
                L.append('        scratch_ref[k, 3] = output_offsets_ref[dst]')
                L.append('        return _')
                L.append('    jax.lax.fori_loop(0, _NUM_ORBITS, _build_state, None)')
                L.append('')
                L.append('    # ---- main loop reads packed state from scratch_ref ----')
                L.append('    def _body(i, _state):')
                L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
                L.append('        k = lax.rem(i, _NUM_ORBITS)')
                L.append('        dst_flat = scratch_ref[k, 0]')
                L.append('        size_total = scratch_ref[k, 1]')
                L.append('        in_off_base = scratch_ref[k, 2]')
                L.append('        out_off_base = scratch_ref[k, 3]')
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
                L.append('    (_, cum_bytes) = jax.lax.fori_loop(')
                L.append('        0, _remainder, _orbit_body,')
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
        L.append('')
        if wait_batch_size:
            L.append('    # final drain: wait for any tail bytes from the remainder loop')
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
    else:
        # Per-step SEND drain ONLY. Issue one OrbitGreedy step's DMAs, then wait
        # for THIS device's own sends to complete before issuing the next step.
        # This bounds outstanding DMAs to one step's width (the TPU v4 / "pfc"
        # fix; v4 has a smaller DMA descriptor queue than v5).
        #
        # The RECV side is drained ONCE at the very end for the true total
        # `total_recv_amount_ref[0]`, exactly like the reference / non-barrier
        # path. A per-step recv drain is INCORRECT: it would have to wait on
        # recv_sem for THIS device's *send* byte-count (`cum`), but a device
        # receives a different number of bytes than it sends -- per step AND in
        # total -- under a ragged / non-uniform all-to-all. A send-keyed recv
        # wait therefore blocks on bytes that never arrive => deadlock on EVERY
        # TPU generation (this was the bug in the original --per-step-barrier).
        L.append('    # ---- per-step SEND drain; single RECV drain at the end ----')
        L.append('    _self_bytes = sizes_ref[my_flat]')
        L.append('    pltpu.make_async_copy(')
        L.append('        o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('        o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('        send_sem,')
        L.append('    ).wait()')
        L.append('')
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
            L.append('        # SEND drain only: wait for this step\'s own sends.')
            L.append('        cum = 0')
            L.append('        for branches in branches_list:')
            L.append('            dst_idx = jax.lax.switch(my_flat, branches)')
            L.append('            cum = cum + sizes_ref[dst_idx]')
            L.append('        pltpu.make_async_copy(')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            send_sem,')
            L.append('        ).wait()')
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
            L.append('        # SEND drain only: wait for this step\'s own sends.')
            L.append('        cum = 0')
            L.append('        for k in step_indices:')
            L.append('            cum = cum + sizes_ref[dest_table_ref[my_flat, k]]')
            L.append('        pltpu.make_async_copy(')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            o_ref.at[pl.ds(0, cum)],')
            L.append('            send_sem,')
            L.append('        ).wait()')
            L.append('')
            for t, step in enumerate(orbit_steps):
                L.append(f'    # ---- OrbitGreedy step {t} ({len(step)} orbit(s)) ----')
                for k in step:
                    L.append(f'    _issue_orbit({k})')
                L.append(f'    _drain_step({step!r})')
                L.append('')
        # Single final RECV drain for the TRUE total (ragged-safe), exactly like
        # the reference kernel. All sends were already drained per step above.
        L.append('    # ---- final RECV drain: wait for the true total received ----')
        L.append('    recv_amount = total_recv_amount_ref[0]')
        L.append('    if enable_checks:')
        L.append('        pl.debug_check(recv_amount >= 0, "recv_amount<0")')
        L.append('    if axis_size_local > 1:')
        L.append('        pltpu.make_async_copy(')
        L.append('            o_ref.at[pl.ds(0, recv_amount)],')
        L.append('            o_ref.at[pl.ds(0, recv_amount)],')
        L.append('            recv_sem,')
        L.append('        ).wait()')

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
        if packed_state:
            L.append('    """Packed-state variant: dest_table + a scratch SMEM buffer for _my_state.')
            L.append('')
            L.append('    The caller must thread `scratch_shapes` into the pallas_call so')
            L.append('    `scratch_ref` resolves to an SMEM Ref of shape (K, 4) int32. Without')
            L.append('    it the kernel will receive scratch_ref=None and fail.')
            L.append('    """')
            L.append('    import jax.numpy as jnp')
            L.append(f'    _K = {K}')
            L.append('    return {')
            L.append('        "dest_table": jnp.asarray(_DEST_TABLE_NP),')
            L.append('        "extra_in_spec": pl.BlockSpec(memory_space=pltpu.SMEM),')
            L.append('        "input_output_aliases_shift": 1,')
            L.append('        "scratch_shapes": [pltpu.SMEM((_K, 4), jnp.int32)],')
            L.append('    }')
        else:
            L.append('    """Helper for inserting _DEST_TABLE_NP as an extra pallas_call input."""')
            L.append('    import jax.numpy as jnp')
            L.append('    return {')
            L.append('        "dest_table": jnp.asarray(_DEST_TABLE_NP),')
            L.append('        "extra_in_spec": pl.BlockSpec(memory_space=pltpu.SMEM),')
            L.append('        "input_output_aliases_shift": 1,')
            L.append('    }')
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------

def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _build_router(name: str, topology: Topology):
    name = name.lower()
    if name == "ilp":
        return ILPRouter(topology=topology), "ILP"
    if name == "dor":
        return DORRouter(topology=topology), "DOR"
    raise ValueError(f"unknown router: {name!r} (choose ilp|dor)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate orbit-greedy P2P Pallas kernel source via the "
                    "router -> scheduler -> kernel pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8")
    p.add_argument("--routing-table", default=None, type=Path,
                   help="Load this routing-table JSON instead of generating one.")
    p.add_argument("--router", default=None, choices=["ilp", "dor"],
                   help="Router used to generate the routing table "
                        "(ignored if --routing-table is given). Default: ilp.")
    p.add_argument("--order", default="lpt_tail_asc",
                   choices=["lpt_tail_asc", "lpt", "spt", "tail_asc"])
    p.add_argument(
        "--scheduler",
        default="orbit_greedy",
        choices=["orbit_greedy", "orbit_greedy_full", "literal_greedy", "ilp_literal"],
        help="Which scheduling algorithm to run on the routing table. "
             "orbit_greedy: original (dim, dir)-keyed greedy — only correct on "
             "translation-equivariant routings (DOR, ILP). "
             "orbit_greedy_full: same greedy but keyed on full physical-edge sets — "
             "correct on any translation-symmetric workload. "
             "literal_greedy: LMR-style per-flow earliest-feasible greedy. "
             "ilp_literal: exact ILP on the literal N*(N-1) flow set (small cells only). "
             "Ignored when --schedule-in is given.",
    )
    p.add_argument(
        "--schedule-in",
        default=None,
        type=Path,
        help="Load an existing schedule JSON instead of running --scheduler. "
             "Useful when the schedule was produced by an expensive offline "
             "search (e.g. CP-SAT, LP-rounding, manual edit) and should not "
             "be regenerated. The routing table is still re-read to verify the "
             "schedule's paths match its declared routing.",
    )
    p.add_argument(
        "--ilp-time-limit-s",
        type=int,
        default=600,
        help="Time limit (s) for the ilp_literal solver. Ignored otherwise.",
    )
    p.add_argument("--per-step-barrier", action="store_true",
                   help="Bound outstanding DMAs for small-DMA-queue targets "
                        "(e.g. TPU v4 / 'pfc'): issue one OrbitGreedy step's "
                        "DMAs, drain SEND sem, then next step; RECV sem is "
                        "drained once at the end for the true total "
                        "(ragged-safe). Without it all DMAs are issued up front "
                        "(fine on TPU v5, hangs on v4).")
    p.add_argument(
        "--inline-destinations",
        action="store_true",
        help="Bake per-step destinations into the kernel as compile-time "
             "switch branches (jax.lax.switch(my_flat, _DEST_BRANCHES_k)) "
             "instead of SMEM lookup. Drops the dest_table_ref input. "
             "Larger generated file but eliminates the per-step SMEM load "
             "from the inner critical path.",
    )
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
    p.add_argument(
        "--capacity-model",
        choices=["staggered", "step"],
        default="staggered",
        help="Which capacity model the stage-3 verifier enforces. "
             "'staggered' (default): hop i of a flow fires at round + i, one "
             "flow per directed edge per time — the orbit_greedy/literal "
             "schedulers' model. 'step': each round is one barrier-delimited "
             "step; all DMAs of a step occupy their WHOLE path together and "
             "cross-step interactions are ignored (the --per-step-barrier "
             "execution model; required for orbit_pack schedules, which are "
             "staggered-infeasible by design).",
    )
    p.add_argument(
        "--step-edge-cap",
        type=int,
        default=None,
        help="Max whole-path edge load per step for --capacity-model step "
             "(e.g. the orbit_pack 'c' the schedule was built with). "
             "Required when --capacity-model step.",
    )
    p.add_argument("--function-name", default=None)
    p.add_argument("--routing-table-out", default=None, type=Path,
                   help="Where to save the generated routing table "
                        "(default: ./fixtures/routing/routing_table_<slice>_<router>.json). "
                        "Ignored if --routing-table is given.")
    p.add_argument("--schedule-out", default=None, type=Path,
                   help="Where to save the schedule "
                        "(default: ./fixtures/nonragged/schedule_<slice>_<router_or_loaded>_<order>.json)")
    p.add_argument("--out", default=None, type=Path,
                   help="Output kernel path "
                        "(default: ./pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py)")
    args = p.parse_args(argv)

    if args.capacity_model == "step" and args.step_edge_cap is None:
        p.error("--step-edge-cap is required when --capacity-model step")
    if args.capacity_model != "step" and args.step_edge_cap is not None:
        p.error("--step-edge-cap only applies to --capacity-model step")

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    slice_slug = "x".join(str(s) for s in slice_)        # e.g. "4x4x8" — used in fixtures filenames
    slice_kern = "_".join(str(s) for s in slice_)        # e.g. "4_4_8" — used in kernel filename + function name
    fixtures = _HERE.parent / "fixtures"

    # Stage 1: routing table.
    if args.routing_table is not None:
        if args.router is not None:
            raise SystemExit(
                "Conflict: pass --routing-table OR --router, not both."
            )
        rt_path = args.routing_table
        router_slug = "loaded"
        router_doc = "loaded"
    else:
        router_slug = args.router or "ilp"
        router, router_disp = _build_router(router_slug, topology)
        rt_path = args.routing_table_out or (
            fixtures / "routing" / f"routing_table_{slice_slug}_{router_slug}.json"
        )
        save_routing_table(topology, router, rt_path)
        print(f"[1/4] wrote routing table {rt_path}", file=sys.stderr)
        router_doc = f"{router_disp}Router"

    table = load_routing_table(rt_path)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table {rt_path} has {len(table)} sources; "
            f"slice {slice_} expects {topology.n_nodes}"
        )

    # Stage 2: schedule (load from disk or run scheduler).
    if args.schedule_in is not None:
        from twisted_analysis.io.schedule import load_schedule
        schedule = load_schedule(args.schedule_in)
        # Verify the schedule's paths agree with the routing table.
        mismatched = 0
        for entry in schedule:
            s, d = entry["src"], entry["dst"]
            expected = list(table[s][d])
            if list(entry["path"]) != expected:
                mismatched += 1
        if mismatched > 0:
            raise SystemExit(
                f"--schedule-in {args.schedule_in}: {mismatched} entries have "
                f"paths that disagree with --routing-table {rt_path}. "
                f"The schedule and routing must match."
            )
        sched_path = args.schedule_in
        print(f"[2/4] loaded schedule    {sched_path} ({len(schedule)} entries)", file=sys.stderr)
    else:
        sched_kwargs = {}
        if args.scheduler in ("orbit_greedy", "orbit_greedy_full"):
            sched_kwargs["order"] = args.order
        elif args.scheduler == "literal_greedy":
            # literal_greedy has its own valid orders; map lpt_tail_asc/lpt -> lpt.
            sched_kwargs["order"] = "lpt" if args.order == "lpt_tail_asc" else args.order
        elif args.scheduler == "ilp_literal":
            sched_kwargs["time_limit_s"] = args.ilp_time_limit_s

        schedule = schedule_from_algorithm(
            args.scheduler, topology, table, **sched_kwargs,
        )
        sched_path = args.schedule_out or (
            fixtures / "nonragged"
            / f"schedule_{slice_slug}_{router_slug}_{args.scheduler}_{args.order}.json"
        )
        save_schedule(schedule, sched_path)
        print(f"[2/4] wrote schedule     {sched_path}", file=sys.stderr)

    # Stage 3 (new): verify capacity under the selected model.
    if args.capacity_model == "step":
        step_violations = verify_capacity_step(
            schedule, max_edge_load=args.step_edge_cap,
        )
        if step_violations:
            print(
                f"\nERROR: schedule has {len(step_violations)} step-model "
                f"capacity violation(s) at --step-edge-cap {args.step_edge_cap}. "
                f"First 3: {step_violations[:3]}",
                file=sys.stderr,
            )
            raise SystemExit(
                f"refusing to emit kernel: step-model capacity violation "
                f"count = {len(step_violations)} at edge cap "
                f"{args.step_edge_cap} (routing={rt_path})"
            )
        print(
            f"[3/4] verified schedule  ({len(schedule)} flows, 0 step-model "
            f"violations at edge cap {args.step_edge_cap}, "
            f"{schedule_step_count(schedule)} barrier steps)",
            file=sys.stderr,
        )
    else:
        violations = verify_capacity(schedule)
        if violations:
            if args.schedule_in is None:
                # Remove the now-misleading schedule file so it doesn't
                # linger as if it were valid. Never delete a user-provided
                # --schedule-in file — it may be an expensive offline result
                # whose intended model is 'step' (e.g. orbit_pack).
                sched_path.unlink(missing_ok=True)
            print(
                f"\nERROR: schedule has {len(violations)} physical-edge capacity violation(s). "
                f"First 3: {violations[:3]}",
                file=sys.stderr,
            )
            raise SystemExit(
                f"refusing to emit kernel for violating schedule "
                f"(scheduler={args.scheduler}, routing={rt_path}); "
                f"capacity violation count = {len(violations)}. "
                f"If this schedule targets barrier-step execution "
                f"(orbit_pack), pass --capacity-model step --step-edge-cap C."
            )
        print(f"[3/4] verified schedule  ({len(schedule)} flows, 0 violations)", file=sys.stderr)

    # Stage 4: kernel.
    dest_table, orbit_steps = _dest_table_and_orbit_steps_from_schedule(
        schedule, topology.n_nodes,
    )
    if args.per_step_barrier:
        # --per-step-barrier bakes per-step DMA-drain calls into the unrolled
        # kernel using src=0's column-to-round map. For per-flow schedulers
        # (literal_greedy, ilp_literal), each source has an independent round
        # mapping, so the unrolled kernel would be wrong on every source != 0.
        # Refuse to emit such a kernel.
        from collections import defaultdict
        _by_src: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for _e in schedule:
            _by_src[_e["src"]].append((_e["round"], _e["dst"]))
        for _s in _by_src:
            _by_src[_s].sort()
        _rounds_src0 = [r for (r, _d) in _by_src[0]]
        for _s in _by_src:
            if _s == 0:
                continue
            _rounds_s = [r for (r, _d) in _by_src[_s]]
            if _rounds_s != _rounds_src0:
                raise SystemExit(
                    f"--per-step-barrier requires a translation-symmetric "
                    f"schedule (orbit_greedy / orbit_greedy_full), but "
                    f"scheduler={args.scheduler!r} produced an asymmetric "
                    f"schedule (sources 0 and {_s} differ at column 0 onward). "
                    f"Drop --per-step-barrier or use an orbit-based scheduler."
                )
    scheduler_for_doc = (
        f"loaded-from {args.schedule_in.name}" if args.schedule_in is not None
        else args.scheduler
    )
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
    out_path = args.out or (
        _HERE / "outputs" / f"_ragged_a2a_kernel_orbit_greedy_{slice_kern}.py"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(src)
    print(f"[4/4] wrote kernel       {out_path} ({len(src):,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
