"""Generator for orbit-greedy P2P AllToAll Pallas TPU kernels.

Produces a self-contained Python file containing a kernel function with the
**same signature as `_ragged_a2a_kernel_point_to_point` plus one extra Ref
input `dest_table_ref`**:

  reference: `(i + (my_id+1) * groups_per_shard) % num_groups`  (rotation)
  generated: `dest_table_ref[my_flat, i]`                       (OrbitGreedy)

`_DEST_TABLE_NP` is an `[N, N-1]` int32 NumPy constant at module level
(no JAX dependency at import). It encodes, for every source device flat-id
and every orbit (in OrbitGreedy step order), the destination device's
flat-id under the *group composition* of the twisted torus. Per-source
baking is required because on a {S, 2S}^n twisted torus the group operation
is NOT elementwise modular addition — see README.md.

The generated kernel takes the table as a kernel *input* (via Pallas
`in_specs`), not a closure constant: Pallas raises "captures constants
[i32[N,K]]" if the table is closed over instead of passed in.

`axis_name` is a **single flat string** (e.g. `"x"`) — the kernel calls
`jax.lax.axis_index(axis_name)` once to obtain the flat device id. The
twist-aware destination lookup then makes per-axis decode unnecessary.

Usage (CLI):

    python pallas_kernel/gen_orbit_greedy_kernel.py \\
        --slice 4,4,8 \\
        [--router ilp|dor] \\
        [--order lpt_tail_asc|lpt|spt|tail_asc] \\
        [--per-step-barrier] \\
        [--out FILE]

Default output: ./pallas_kernel/_ragged_a2a_kernel_orbit_greedy_<slice>.py

The generated kernel cannot be tested without TPU silicon. See README.md
§Validation for the smoke-test plan.
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

from twisted_analysis.topology import Topology, DORRouter, ILPRouter
from twisted_analysis.schedules.orbit_greedy import (
    _canonical_paths, _edge_orbit_load, _ordered_orbits, _emit_orbit_greedy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walk_path(topology: Topology, source, path) -> tuple:
    """Walk `path` (sequence of DirectedLink) from `source`, return endpoint."""
    node = source
    for _u, _v, dim, dir in path:
        node = topology.neighbor(node, dim, dir)
    return node


def _build_dest_table(
    topology: Topology,
    ordered_orbits: list,
    canonical_paths: dict,
) -> np.ndarray:
    """For every (source_flat, orbit_idx) -> destination_flat.

    Returns: int32 [N, K] where N = num devices, K = num orbits.
    Flat-id convention: `flat = dim0 * prod(slice[1:]) + dim1 * prod(slice[2:]) + ...`
    """
    nodes = list(topology.nodes())
    flat_of = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    K = len(ordered_orbits)
    table = np.zeros((N, K), dtype=np.int32)
    for src in nodes:
        for k, orbit_id in enumerate(ordered_orbits):
            dst = _walk_path(topology, src, canonical_paths[orbit_id])
            table[flat_of[src], k] = flat_of[dst]
    return table


def _hop0_steps(assignment: dict, ordered_orbits: list) -> list[list[int]]:
    """Group orbit indices by their hop-0 firing time `t_0^O`."""
    t0: dict = {}
    for (orbit_id, hop_i, t), _v in assignment.items():
        if hop_i == 0:
            t0[orbit_id] = t
    by_step: dict = defaultdict(list)
    for k, orbit_id in enumerate(ordered_orbits):
        by_step[t0[orbit_id]].append(k)
    return [by_step[t] for t in sorted(by_step.keys())]


def _dest_table_literal(table: np.ndarray) -> str:
    """Compact module-level literal for an int32 [N, K] table."""
    rows = ["    [" + ", ".join(f"{v:3d}" for v in row) + "]"
            for row in table]
    return "np.array([\n" + ",\n".join(rows) + ",\n], dtype=np.int32)"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_kernel_source(
    topology: Topology,
    router=None,
    *,
    order: str = "lpt_tail_asc",
    function_name: str | None = None,
    per_step_barrier: bool = False,
    router_name_for_doc: str | None = None,
) -> str:
    """Generate the orbit-greedy P2P kernel as Python source.

    Args:
        topology: TwistedAnalysis Topology. Slice in the {S, 2S}^n family.
        router: Router instance. None -> ILPRouter (recommended).
        order: OrbitGreedy ordering. Default "lpt_tail_asc" (achieves
            makespan = LB on every doc cell).
        function_name: Override generated function name.
        per_step_barrier: Emit per-OrbitGreedy-step barriers (best-effort,
            via dummy-DMA wait). Default False (rely on source-FIFO).
        router_name_for_doc: Display name for the router (default: class name).

    Returns:
        Generated Python source as a string.
    """
    if router is None:
        router = ILPRouter(topology=topology)

    n_dim = topology.ndim
    slice_ = tuple(topology.slice)
    N = topology.n_nodes

    if function_name is None:
        slice_str = "_".join(str(s) for s in slice_)
        function_name = f"_ragged_a2a_kernel_orbit_greedy_{slice_str}"

    if router_name_for_doc is None:
        router_name_for_doc = type(router).__name__

    canon = _canonical_paths(topology, router)
    edge_load = _edge_orbit_load(canon)
    ordered_orbits = _ordered_orbits(canon, edge_load, order)
    assignment = _emit_orbit_greedy(topology, router, order)

    K = len(ordered_orbits)
    if K != N - 1:
        raise RuntimeError(
            f"Expected {N - 1} non-trivial orbits, got {K}"
        )

    dest_table = _build_dest_table(topology, ordered_orbits, canon)
    orbit_steps = _hop0_steps(assignment, ordered_orbits)
    makespan_hop0 = len(orbit_steps)
    LB = max(edge_load.values())
    observed_makespan = max(t for (_, _, t) in assignment.keys()) + 1

    L = []

    # ---- header ------------------------------------------------------------
    L.append('"""Orbit-greedy P2P AllToAll Pallas TPU kernel.')
    L.append('')
    L.append('AUTO-GENERATED — DO NOT EDIT BY HAND.')
    L.append('')
    L.append(f'Topology:        slice={slice_}  (N={N} devices, ndim={n_dim})')
    L.append(f'Router:          {router_name_for_doc}')
    L.append(f'OrbitGreedy:     order={order!r}')
    L.append(f'Per-step barrier: {per_step_barrier}')
    L.append(f'Bandwidth LB:    {LB}')
    L.append(f'OrbitGreedy makespan (model): {observed_makespan} '
             f'(LB-ratio = {observed_makespan / LB:.3f})')
    L.append(f'Hop-0 steps:     {makespan_hop0}')
    L.append('')
    L.append('Regenerate via:')
    barrier_flag = " --per-step-barrier" if per_step_barrier else ""
    L.append(f'    python pallas_kernel/gen_orbit_greedy_kernel.py \\')
    L.append(f'        --slice {",".join(str(s) for s in slice_)} \\')
    L.append(f'        --router {router_name_for_doc.replace("Router", "").lower()} \\')
    L.append(f'        --order {order}{barrier_flag}')
    L.append('')
    L.append('Integration. The kernel has the same signature as')
    L.append('`_ragged_a2a_kernel_point_to_point` PLUS one extra positional Ref')
    L.append('input `dest_table_ref` (slot 6, between `num_packets_per_group_ref`')
    L.append('and `x_ref`). To use:')
    L.append('  1. Copy this file next to reference_kernel.py.')
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
    L.append('')
    L.append('See README.md for the twist explanation and integration steps.')
    L.append('"""')
    L.append('from __future__ import annotations')
    L.append('')
    L.append('import jax')
    L.append('import numpy as np')
    L.append('from jax import lax')
    L.append('from jax.experimental import pallas as pl')
    L.append('from jax.experimental.pallas import tpu as pltpu')
    L.append('')
    L.append('# Adjust this import to match where ragged_collectives_utils lives in')
    L.append('# your project. The reference P2P kernel uses the same barrier.')
    L.append('from megablox.collectives import ragged_collectives_utils  # type: ignore')
    L.append('')
    L.append('')
    L.append('# ----------------------------- baked schedule -------------------------------')
    L.append('# _DEST_TABLE_NP[src_flat, k] = flat destination device id for orbit k')
    L.append('# (in OrbitGreedy firing order) from source `src_flat`.')
    L.append('#')
    L.append('# Plain numpy at module level (NO jax.numpy here — importing this file')
    L.append('# must NOT trigger JAX backend initialization, or imports of sibling')
    L.append('# kernel files in the same module can fail).')
    L.append('#')
    L.append(f'# Per-source baking required because the {slice_} twisted-torus group')
    L.append('# composition is NOT elementwise modular. See README.md §Twist.')
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
    L.append('')

    # ---- function header --------------------------------------------------
    L.append(f'def {function_name}(')
    L.append('    input_offsets_ref,')
    L.append('    output_offsets_ref,')
    L.append('    sizes_ref,')
    L.append('    total_send_amount_ref,')
    L.append('    total_recv_amount_ref,')
    L.append('    num_packets_per_group_ref,')
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
    L.append('    Signature: same as `_ragged_a2a_kernel_point_to_point` PLUS one extra')
    L.append('    Ref input `dest_table_ref` (int32[N, K] in SMEM, slot 6). Other')
    L.append('    differences vs reference:')
    L.append('      * Iteration order = OrbitGreedy firing order (vs rotation).')
    L.append('      * Destinations are looked up in `dest_table_ref` (twist-aware).')
    L.append('      * `transpose=True` is NOT supported (would need regen).')
    L.append('      * Assumes 1 group per device (uniform AllToAll).')
    L.append('      * `axis_name` is a flat string (e.g. "x"), as in the reference.')
    L.append('    """')
    L.append('    assert scratch_ref is None')
    L.append('    del scratch_ref')
    L.append('    assert scratch_sems is None')
    L.append('    del scratch_sems')
    L.append('    assert not transpose, (')
    L.append('        "transpose=True not supported by orbit-greedy kernel; use reference."')
    L.append('    )')
    L.append('')
    L.append('    # Flat device id under `axis_name` (a single mesh axis spanning all')
    L.append(f'    # {N} devices). The twist is baked into dest_table_ref, so we do not')
    L.append('    # decode per-axis coordinates here.')
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

    # ---- common inner helpers (defined inside the kernel) ----------------
    L.append('    num_packets = num_packets_per_group_ref[0]')
    L.append('')
    L.append('    def _issue_packet(packet_idx, group_idx, dst_device_id):')
    L.append('        size = lax.min(')
    L.append('            packet_size,')
    L.append('            lax.max(sizes_ref[group_idx] - packet_idx * packet_size, 0),')
    L.append('        )')
    L.append('        input_offset = input_offsets_ref[group_idx] + packet_idx * packet_size')
    L.append('        output_offset = output_offsets_ref[group_idx] + packet_idx * packet_size')
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
    L.append('    # ---- self copy (group_idx = my_flat) ----')
    L.append('    def _self_body(packet_idx, _state):')
    L.append('        _issue_packet(packet_idx, my_flat, {axis_name: my_flat})')
    L.append('        return _state')
    L.append('')
    L.append('    jax.lax.fori_loop(0, num_packets, _self_body, None)')
    L.append('')

    if not per_step_barrier:
        # ---- single flat fori_loop, final drain ----
        # packet_idx is OUTER, orbit-index is INNER — matches the reference's
        # round-robin pattern (each consecutive iteration targets a different
        # physical outgoing link, so per-link DMAs pipeline), and preserves
        # the orbit-greedy column-permutation invariant per packet round.
        L.append('    # ---- main orbit loop: packet outer, OrbitGreedy order inner ----')
        L.append(f'    _NUM_ORBITS = {K}  # = axis_size - 1')
        L.append('    def _body(i, _state):')
        L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
        L.append('        k = lax.rem(i, _NUM_ORBITS)')
        L.append('        dst_flat = dest_table_ref[my_flat, k]')
        L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
        L.append('        return _state')
        L.append('')
        L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
        L.append('')
        L.append('    # ---- final drain (same dummy-DMA pattern as reference) ----')
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
        # ---- per-step barrier variant ----
        # Drain the self copy first (we can't bundle it with any orbit step).
        L.append('    # Drain the self copy before issuing OrbitGreedy step 0:')
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

    L.append('')

    # ---- integration helper -----------------------------------------------
    L.append('')
    L.append(f'def build_pallas_call_kwargs():')
    L.append('    """Return a dict of pallas_call kwargs/inputs for this kernel.')
    L.append('')
    L.append('    The returned dict has 3 keys:')
    L.append('      * "dest_table": jnp.ndarray to pass as the extra positional input')
    L.append('         (slot 6, before x).')
    L.append('      * "extra_in_spec": pl.BlockSpec for the dest_table input (SMEM).')
    L.append('      * "input_output_aliases_shift": int 1 — add to every key in the')
    L.append('         original `input_output_aliases` dict to account for the new')
    L.append('         slot. The reference uses `{7: 0}`; with this kernel use `{8: 0}`.')
    L.append('')
    L.append('    Call this inside `ragged_all_to_all` (NOT at module load) so the')
    L.append('    `jnp.asarray` of the table happens only when JAX is initialized.')
    L.append('    """')
    L.append('    import jax.numpy as jnp')
    L.append('    return {')
    L.append('        "dest_table": jnp.asarray(_DEST_TABLE_NP),')
    L.append('        "extra_in_spec": pl.BlockSpec(memory_space=pltpu.SMEM),')
    L.append('        "input_output_aliases_shift": 1,')
    L.append('    }')
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
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
        description="Generate orbit-greedy P2P Pallas kernel source.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8")
    p.add_argument("--router", default="ilp", choices=["ilp", "dor"],
                   help="Routing table (default: ilp)")
    p.add_argument("--order", default="lpt_tail_asc",
                   choices=["lpt_tail_asc", "lpt", "spt", "tail_asc"])
    p.add_argument("--per-step-barrier", action="store_true",
                   help="Emit per-OrbitGreedy-step barriers (best-effort).")
    p.add_argument("--function-name", default=None)
    p.add_argument("--out", default=None,
                   help="Output path (default: ./pallas_kernel/_ragged_a2a_kernel_orbit_greedy_<slice>.py)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    router, router_disp = _build_router(args.router, topology)

    src = generate_kernel_source(
        topology,
        router=router,
        order=args.order,
        function_name=args.function_name,
        per_step_barrier=args.per_step_barrier,
        router_name_for_doc=f"{router_disp}Router",
    )

    if args.out is None:
        slice_str = "_".join(str(s) for s in slice_)
        out_path = _HERE / f"_ragged_a2a_kernel_orbit_greedy_{slice_str}.py"
    else:
        out_path = Path(args.out)
    out_path.write_text(src)
    print(f"wrote {out_path} ({len(src):,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
