"""Orbit-greedy P2P AllToAll Pallas TPU kernel.

AUTO-GENERATED — DO NOT EDIT BY HAND.

Topology:        slice=(2, 2, 4)  (N=16 devices, ndim=3)
Router:          loaded
Scheduler:       loaded-from schedule_torus_2x2x4_cpsat_literal_warm.json
OrbitGreedy:     order='lpt_tail_asc'
Per-step barrier: False
Hop-0 steps:     8

Generated from:  routing-table JSON + schedule JSON

Integration. The kernel has the same signature as
`_ragged_a2a_kernel_point_to_point` PLUS one extra positional Ref
input `dest_table_ref` (slot 6, between `num_packets_per_group_ref`
and `x_ref`). To use:
  1. Copy this file next to reference_kernel.py.
  2. Set `kernel = _ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4` in ragged_all_to_all.
  3. Insert an extra SMEM in_spec at slot 6 (before x):
         pl.BlockSpec(memory_space=pltpu.SMEM)
     and pass `jnp.asarray(_DEST_TABLE_NP)` as the corresponding
     extra positional input to the pallas_call.
  4. Shift `input_output_aliases` keys by +1: e.g. {7: 0} → {8: 0}.
  5. Call ragged_all_to_all(...) with a single flat axis name, e.g.
         axis_name="x"
     The orbit-greedy kernel does NOT decode per-axis coords; it
     reads `my_flat = jax.lax.axis_index(axis_name)` once.

Use `build_pallas_call_kwargs()` below for a copy-pasteable example.
"""
from __future__ import annotations

import jax
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from megablox.collectives import ragged_collectives_utils  # type: ignore


# ----------------------------- baked schedule -------------------------------
_DEST_TABLE_NP = np.array([
    [  8,  11,  12,  14,   5,   6,   3,  15,   9,   7,  10,   4,  13,   1,   2],
    [  5,   8,  11,  13,  10,   7,  14,   2,  12,   6,  15,   4,   9,   0,   3],
    [  6,  12,  13,  14,   8,   9,   4,  15,  11,   7,  10,   0,   5,   1,   3],
    [  5,  10,  11,  15,  14,   4,   8,   9,   1,   6,  12,  13,   0,   7,   2],
    [  0,   1,   2,   8,  13,   9,  12,   3,  14,  15,  10,  11,   7,   5,   6],
    [  9,  13,  14,  15,   3,  10,   6,   7,   8,  11,  12,   2,   0,   1,   4],
    [  2,   9,  10,  12,   4,  13,  15,   0,   3,   1,   8,  11,   5,  14,   7],
    [  9,  10,  11,  15,  12,   2,  13,  14,   0,   1,   8,   4,   5,   3,   6],
    [  0,   3,   4,   6,  13,  14,  11,   1,  12,   7,   2,  15,   5,  10,   9],
    [  4,   5,   7,  13,   1,   2,   3,  14,   6,  12,  15,   0,  10,   8,  11],
    [  2,   6,   9,  12,   0,  13,   5,   3,   1,   4,  15,   7,   8,  11,  14],
    [  2,   5,   7,  15,   3,   8,   0,   4,   1,  14,  12,   6,  13,   9,  10],
    [  0,   5,   8,  14,   4,   7,   1,   2,  11,   3,   6,   9,  10,  15,  13],
    [  1,   3,   6,   9,   0,   7,  10,   8,  15,   2,   5,   4,  11,  14,  12],
    [  2,   5,  10,  12,   4,   9,   1,   6,   7,  13,   0,  11,   3,   8,  15],
    [  1,   3,   6,  11,   4,   7,   0,   8,   2,   5,  10,   9,  12,  13,  14],
], dtype=np.int32)
assert _DEST_TABLE_NP.shape == (16, 15), (
    f"_DEST_TABLE_NP shape mismatch: {_DEST_TABLE_NP.shape}"
)

# _ORBIT_STEPS[t] = orbit indices firing at OrbitGreedy step t. 8 steps total.
_ORBIT_STEPS = [
    [0, 1, 2, 3],  # step 0 (4 concurrent orbit(s))
    [4, 5],  # step 1 (2 concurrent orbit(s))
    [6],  # step 2 (1 concurrent orbit(s))
    [7],  # step 3 (1 concurrent orbit(s))
    [8],  # step 4 (1 concurrent orbit(s))
    [9, 10],  # step 5 (2 concurrent orbit(s))
    [11, 12],  # step 6 (2 concurrent orbit(s))
    [13, 14],  # step 7 (2 concurrent orbit(s))
]


def _ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4(
    input_offsets_ref,
    output_offsets_ref,
    sizes_ref,
    total_send_amount_ref,
    total_recv_amount_ref,
    num_packets_per_group_ref,
    dest_table_ref,  # int32[N, K] in SMEM — pass as extra pallas_call input
    x_ref,
    _,
    o_ref,
    scratch_ref,
    send_sem,
    recv_sem,
    scratch_sems,
    *,
    axis_name,
    transpose,
    packet_size,
    enable_checks: bool = False,
):
    """Orbit-greedy P2P AllToAll kernel for slice=(2, 2, 4).

    Signature: same as `_ragged_a2a_kernel_point_to_point` PLUS one extra
    Ref input `dest_table_ref` (int32[N, K] in SMEM, slot 6). Other
    differences vs reference:
      * Iteration order = OrbitGreedy firing order (vs rotation).
      * Destinations are looked up in `dest_table_ref` (twist-aware).
      * `transpose=True` is NOT supported (would need regen).
      * Assumes 1 group per device (uniform AllToAll).
      * `axis_name` is a flat string (e.g. "x"), as in the reference.
    """
    assert scratch_ref is None
    del scratch_ref
    assert scratch_sems is None
    del scratch_sems
    assert not transpose, (
        "transpose=True not supported by orbit-greedy kernel; use reference."
    )

    my_flat = jax.lax.axis_index(axis_name)
    axis_size_local = jax.lax.axis_size(axis_name)

    num_groups = sizes_ref.shape[0]
    assert num_groups == 16, (
        f"Expected num_groups=16 (uniform AllToAll on (2, 2, 4)); got {num_groups}"
    )
    groups_per_shard, r = divmod(num_groups, axis_size_local)
    assert r == 0 and groups_per_shard == 1, (
        "orbit-greedy kernel assumes 1 group per device"
    )

    if axis_size_local > 1:
        ragged_collectives_utils.main_barrier(
            axis_name,
            barrier_type=ragged_collectives_utils.BarrierType.ALL_TO_ALL,
        )

    num_packets = num_packets_per_group_ref[0]

    def _issue_packet(packet_idx, group_idx, dst_device_id):
        size = lax.min(
            packet_size,
            lax.max(sizes_ref[group_idx] - packet_idx * packet_size, 0),
        )
        input_offset = input_offsets_ref[group_idx] + packet_idx * packet_size
        output_offset = output_offsets_ref[group_idx] + packet_idx * packet_size

        if enable_checks:
            pl.debug_check(sizes_ref[group_idx] >= 0, "size<0")
            pl.debug_check(size >= 0, "transfer size<0")
            pl.debug_check(input_offset >= 0, "input_offset<0")
            pl.debug_check(output_offset >= 0, "output_offset<0")
            pl.debug_check(
                input_offset + size <= x_ref.shape[0],
                "input_offset+size > x_ref.shape[0]",
            )
            pl.debug_check(
                output_offset + size <= o_ref.shape[0],
                "output_offset+size > o_ref.shape[0]",
            )

        @pl.when(size > 0)
        def _():
            if axis_size_local > 1:
                pltpu.make_async_remote_copy(
                    x_ref.at[pl.ds(input_offset, size)],
                    o_ref.at[pl.ds(output_offset, size)],
                    device_id=dst_device_id,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                ).start()
            else:
                pltpu.make_async_copy(
                    x_ref.at[pl.ds(input_offset, size)],
                    o_ref.at[pl.ds(output_offset, size)],
                    sem=send_sem,
                ).start()

    def _self_body(packet_idx, _state):
        _issue_packet(packet_idx, my_flat, {axis_name: my_flat})
        return _state

    jax.lax.fori_loop(0, num_packets, _self_body, None)

    # ---- main orbit loop: packet outer, OrbitGreedy order inner ----
    _NUM_ORBITS = 15
    def _body(i, _state):
        packet_idx = lax.div(i, _NUM_ORBITS)
        k = lax.rem(i, _NUM_ORBITS)
        dst_flat = dest_table_ref[my_flat, k]
        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})
        return _state

    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)

    send_amount = total_send_amount_ref[0]
    recv_amount = total_recv_amount_ref[0]
    if enable_checks:
        pl.debug_check(send_amount >= 0, "send_amount<0")
        pl.debug_check(recv_amount >= 0, "recv_amount<0")
    pltpu.make_async_copy(
        o_ref.at[pl.ds(0, send_amount)],
        o_ref.at[pl.ds(0, send_amount)],
        send_sem,
    ).wait()
    if axis_size_local > 1:
        pltpu.make_async_copy(
            o_ref.at[pl.ds(0, recv_amount)],
            o_ref.at[pl.ds(0, recv_amount)],
            recv_sem,
        ).wait()

def build_pallas_call_kwargs():
    """Helper for inserting _DEST_TABLE_NP as an extra pallas_call input."""
    import jax.numpy as jnp
    return {
        "dest_table": jnp.asarray(_DEST_TABLE_NP),
        "extra_in_spec": pl.BlockSpec(memory_space=pltpu.SMEM),
        "input_output_aliases_shift": 1,
    }