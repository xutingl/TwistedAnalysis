"""Orbit-greedy P2P AllToAll Pallas TPU kernel.

AUTO-GENERATED — DO NOT EDIT BY HAND.

Topology:        slice=(2, 4, 4)  (N=32 devices, ndim=3)
Router:          loaded
Scheduler:       loaded-from schedule_torus_2x4x4_cpsat_literal_warm.json
OrbitGreedy:     order='lpt_tail_asc'
Per-step barrier: False
Hop-0 steps:     16

Generated from:  routing-table JSON + schedule JSON

Integration. The kernel has the same signature as
`_ragged_a2a_kernel_point_to_point` PLUS one extra positional Ref
input `dest_table_ref` (slot 6, between `num_packets_per_group_ref`
and `x_ref`). To use:
  1. Copy this file next to reference_kernel.py.
  2. Set `kernel = _ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4` in ragged_all_to_all.
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
    [ 16,  19,  20,  24,  30,  23,   7,   8,   3,   5,  10,  21,  26,  17,  22,   4,  25,  11,  12,  29,   2,  27,  31,  13,  18,  15,  28,   9,  14,   1,   6],
    [ 10,  13,  17,  23,  25,  20,  18,  29,  11,  16,   6,   7,  12,  14,  31,  15,  19,  30,  21,  28,  22,   4,  24,   8,  27,   9,  26,   2,   5,   0,   3],
    [ 10,  11,  22,  26,  28,  25,  20,  30,  31,  17,  24,  29,   1,   8,  12,  23,   5,  18,  19,  15,   9,   6,  21,  13,  14,   7,  16,  27,   0,   3,   4],
    [  5,   6,  19,  23,  27,   9,  26,  29,  30,   4,  13,  20,   8,  17,  24,  14,  16,  31,  15,  22,  21,  28,  11,  18,   7,  12,   0,  25,  10,   1,   2],
    [  0,   9,  12,  18,  28,  16,  17,  14,  21,  23,  19,  26,  15,  20,  13,  29,   3,  10,  27,   1,  30,  11,  24,  22,  25,   8,  31,   2,   7,   5,   6],
    [ 12,  21,  25,  27,  29,   3,  28,  14,  19,   0,  15,  13,  22,   1,   6,   8,  11,  17,  24,   9,  16,   7,  10,  20,  31,  26,  18,  23,  30,   2,   4],
    [  2,   5,   8,  22,  30,  10,  27,  19,   0,  29,  16,  25,  14,  15,   9,  26,  13,  18,  31,  23,  28,  21,  11,  12,  17,  20,   3,  24,   1,   4,   7],
    [  9,  14,  15,  21,  31,  26,  28,  10,  18,  25,  13,  24,   8,  27,   3,  20,  22,   2,  29,  16,  17,  19,   1,  30,   4,  11,  12,   0,  23,   5,   6],
    [  2,  16,  19,  24,  30,   1,   6,  17,  10,  27,  11,  28,  21,  26,   5,  20,  29,  13,  22,   0,  31,  14,  23,  18,  25,   3,   4,   7,  12,  15,   9],
    [ 12,  17,  23,  25,  27,   2,  31,   5,  22,  21,  30,   1,  24,  28,   0,   4,  20,   7,  18,  14,  29,   3,  26,   6,  13,  15,  16,  10,  19,   8,  11],
    [  4,  18,  22,  26,  31,  20,  21,   3,   0,   7,   2,  12,  23,   5,  14,  15,  28,  24,  27,   9,  16,  25,  19,  29,  30,   8,  17,   1,   6,  13,  11],
    [  1,   3,  13,  19,  24,   5,  16,   7,  28,  22,  31,  30,  18,  29,   2,  15,   6,  25,  26,   0,  12,  20,  23,   4,  17,  14,  27,   8,  21,   9,  10],
    [  4,  16,  18,  20,  27,  15,  26,   1,  28,   6,  25,   5,   3,  30,  23,  29,  10,  17,   8,  21,   7,  22,   0,  31,  19,  24,   2,   9,  11,  13,  14],
    [ 11,  17,  21,  29,  30,  14,  25,  18,  22,  27,  10,   1,  16,   6,  31,  19,  24,   4,  23,  26,   2,   9,   5,  28,   0,   7,   8,   3,  20,  12,  15],
    [  6,  20,  22,  24,  31,   5,   4,  17,   2,  27,   8,  23,  18,  21,  25,  26,   7,   3,  16,  19,  30,  11,   0,   1,  10,  28,  29,   9,  13,  12,  15],
    [  9,  18,  23,  29,  31,  11,  24,   8,  19,  28,   0,  16,  21,  20,  26,  27,   3,  12,  17,   4,   7,  25,   6,  30,   2,  13,   1,  10,   5,  22,  14],
    [  0,   1,   4,   8,  14,   2,  23,   5,  12,   3,  10,   7,  28,  11,  22,  24,  25,  26,  15,   9,  20,  31,  13,  21,  30,  29,   6,  27,  19,  17,  18],
    [  1,   9,  29,  30,  31,  13,  14,   6,  11,   0,  15,   7,  26,   8,  27,  18,  23,   2,  19,  21,  24,   3,  28,   5,  20,   4,  12,  10,  22,  16,  25],
    [  4,   8,   9,  10,  26,  27,   3,  16,   1,   0,   5,  21,  13,  28,   6,  29,  12,  25,  14,  15,  23,  24,   7,  22,   2,  31,  11,  30,  17,  19,  20],
    [  8,  11,  13,  15,  27,  20,  22,  25,   1,  28,  29,  30,   9,  14,   4,  31,   6,  17,   7,  12,  10,   2,   5,   0,  26,  24,   3,  16,  23,  18,  21],
    [  2,   4,   5,  12,  30,   0,  17,   3,  13,  26,   6,   9,   7,  18,  29,  31,  10,  15,  22,  14,  27,   1,  24,  16,  25,  19,   8,  11,  23,  21,  28],
    [  3,   5,  10,  13,  15,   8,  31,  16,  19,   7,  28,  23,  24,   0,  11,  30,   1,  12,   2,  29,  18,   4,  25,   6,  26,  27,   9,  14,  17,  22,  20],
    [ 12,  14,  24,  25,  30,   0,   9,   7,   6,  17,   1,   3,  16,   2,  15,  11,   4,  29,  27,  28,   8,  13,  10,  31,   5,  19,  26,  18,  21,  20,  23],
    [  9,  11,  15,  26,  31,   4,  13,   2,  29,  12,  19,  18,  21,   3,   6,  20,   8,  25,  27,   0,  10,  17,   1,  30,  28,   5,  24,   7,  14,  16,  22],
    [  0,  14,  16,  18,  21,   5,  26,   1,  20,  10,  15,  23,   3,  13,  22,   6,  17,  11,  28,  27,   7,  12,   8,  19,   4,   9,   2,  29,  31,  25,  30],
    [  1,   6,   7,  17,  21,   8,  14,   4,  11,  13,  20,  10,  29,   2,  22,   5,  23,  28,  15,  16,  12,   0,   3,   9,  30,  18,  19,  31,  26,  24,  27],
    [  2,   7,  10,  12,  16,   6,  19,   3,  20,  17,  15,  30,   8,  31,   5,   0,   4,  11,  22,  23,   9,  14,  29,   1,  18,  13,  21,  25,  28,  24,  27],
    [  3,  11,  12,  21,  31,  19,  24,  14,  15,   1,   4,  10,  29,   8,   2,  20,  25,   9,  18,   6,   5,  22,   0,   7,  13,  30,  16,  23,  17,  28,  26],
    [  0,   2,   4,   7,  20,   9,  27,  17,  13,   1,  10,  16,  23,  18,  19,   5,  26,  11,   8,  21,   3,  12,  14,   6,  15,  22,  25,  24,  31,  29,  30],
    [  3,   5,  21,  22,  23,  14,  11,  16,  25,  30,   0,  13,   9,  12,  19,   8,   6,  27,   7,  24,   1,  26,   4,  31,  10,  15,  17,  18,   2,  20,  28],
    [  6,  10,  17,  22,  24,  15,  18,  16,  23,   2,   5,  11,  21,  26,   4,  13,   1,   9,  14,   7,  12,  25,   3,   8,  19,  20,  27,   0,  29,  28,  31],
    [  6,   7,  17,  23,  27,   2,  29,   5,   8,   9,  12,  19,  24,  14,  20,  10,   0,   3,  13,  18,  11,  28,   4,  21,  16,  25,  15,  26,   1,  22,  30],
], dtype=np.int32)
assert _DEST_TABLE_NP.shape == (32, 31), (
    f"_DEST_TABLE_NP shape mismatch: {_DEST_TABLE_NP.shape}"
)

# _ORBIT_STEPS[t] = orbit indices firing at OrbitGreedy step t. 16 steps total.
_ORBIT_STEPS = [
    [0, 1, 2, 3, 4],  # step 0 (5 concurrent orbit(s))
    [5],  # step 1 (1 concurrent orbit(s))
    [6, 7],  # step 2 (2 concurrent orbit(s))
    [8],  # step 3 (1 concurrent orbit(s))
    [9, 10],  # step 4 (2 concurrent orbit(s))
    [11, 12],  # step 5 (2 concurrent orbit(s))
    [13, 14],  # step 6 (2 concurrent orbit(s))
    [15, 16],  # step 7 (2 concurrent orbit(s))
    [17, 18],  # step 8 (2 concurrent orbit(s))
    [19],  # step 9 (1 concurrent orbit(s))
    [20, 21],  # step 10 (2 concurrent orbit(s))
    [22],  # step 11 (1 concurrent orbit(s))
    [23, 24],  # step 12 (2 concurrent orbit(s))
    [25, 26],  # step 13 (2 concurrent orbit(s))
    [27, 28],  # step 14 (2 concurrent orbit(s))
    [29, 30],  # step 15 (2 concurrent orbit(s))
]


def _ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4(
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
    """Orbit-greedy P2P AllToAll kernel for slice=(2, 4, 4).

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
    assert num_groups == 32, (
        f"Expected num_groups=32 (uniform AllToAll on (2, 4, 4)); got {num_groups}"
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
    _NUM_ORBITS = 31
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