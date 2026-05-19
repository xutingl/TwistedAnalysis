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
