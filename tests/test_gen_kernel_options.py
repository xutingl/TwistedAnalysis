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


def test_packed_state_emits_preamble_scratch_writes():
    """--packed-state must emit a preamble fori_loop that writes the per-source
    packed state into scratch_ref (SMEM ref, allocated by the caller).

    We use scratch_ref (not jnp.zeros + state.at[].set()) because the latter
    lowers to lax.scatter, which Pallas TPU does not support.
    """
    src = _gen(packed_state=True)
    assert "scratch_ref[k, 0]" in src, (
        "packed_state must write dst into scratch_ref[k, 0]"
    )
    assert "scratch_ref[k, 1]" in src
    assert "scratch_ref[k, 2]" in src
    assert "scratch_ref[k, 3]" in src
    assert "input_offsets_ref" in src
    assert "sizes_ref" in src
    assert "output_offsets_ref" in src
    # Must NOT use the scatter-emitting pattern:
    assert ".at[k, 0].set" not in src, (
        "must not use state.at[k, X].set() — that emits lax.scatter which "
        "Pallas TPU lowering does not support"
    )
    # The preamble must run BEFORE the main hot loop.
    preamble_pos = src.find("scratch_ref[k, 0] =")
    body_pos = src.find("def _body(")
    assert 0 <= preamble_pos < body_pos, (
        "preamble scratch_ref writes must precede the main loop"
    )


def test_packed_state_hot_loop_reads_from_scratch_ref():
    """In the main hot loop, the per-step SMEM reads must be replaced with
    reads from scratch_ref."""
    src = _gen(packed_state=True)
    body_start = src.find("def _body(")
    assert body_start != -1
    body_end = src.find("jax.lax.fori_loop(0, _NUM_ORBITS * num_packets", body_start)
    assert body_end != -1, "could not locate main fori_loop call after _body"
    body_block = src[body_start:body_end]
    assert "scratch_ref[k, 0]" in body_block, (
        "main loop body must read dst from scratch_ref[k, 0] when packed_state=True"
    )
    # The body must NOT make the old direct lookups:
    assert "dest_table_ref[my_flat, k]" not in body_block
    assert "sizes_ref[dst" not in body_block


def test_packed_state_build_pallas_call_kwargs_emits_scratch_shapes():
    """build_pallas_call_kwargs() must publish scratch_shapes so the caller
    allocates the SMEM scratch buffer that scratch_ref binds to."""
    src = _gen(packed_state=True)
    assert "scratch_shapes" in src, (
        "packed_state variant must expose scratch_shapes via build_pallas_call_kwargs"
    )
    assert "pltpu.SMEM" in src


def test_packed_state_does_not_assert_scratch_ref_is_none():
    """The kernel body must not assert scratch_ref is None when packed_state=True
    (it's expected to be the SMEM scratch buffer)."""
    src = _gen(packed_state=True)
    assert "assert scratch_ref is None" not in src


def test_packed_state_output_parses_as_python():
    """Generated source must be syntactically valid Python."""
    src = _gen(packed_state=True)
    ast.parse(src)


def test_packed_state_off_emits_old_pattern():
    """Without --packed-state, the hot loop must still use the old direct
    SMEM-lookup pattern and assert scratch_ref is None."""
    src = _gen(packed_state=False)
    assert "dest_table_ref[my_flat, k]" in src, (
        "default mode must still read dst directly from dest_table_ref"
    )
    assert "scratch_ref[k," not in src
    assert "assert scratch_ref is None" in src


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
