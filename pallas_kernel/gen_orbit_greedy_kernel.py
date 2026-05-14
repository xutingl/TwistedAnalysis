"""Generator for orbit-greedy P2P AllToAll Pallas TPU kernels.

Produces a self-contained Python file containing a kernel function with the
same signature as `_ragged_a2a_kernel_point_to_point` (see reference_kernel.py
in this directory), differing only in the *order* in which each device
issues its remote DMA sends:

  reference: `(i + (my_id+1) * groups_per_shard) % num_groups`  (rotation)
  generated: `DEST_TABLE[my_flat, i]`                            (OrbitGreedy)

`DEST_TABLE` is an `[N, N-1]` int32 constant baked into the generated file. It
encodes, for every source device coord and every orbit (in OrbitGreedy step
order), the destination device's flat id under the *group composition* of the
twisted torus. Per-source baking is required because on a {S, 2S}^n twisted
torus the group operation is NOT elementwise modular addition — see README.md.

Usage (CLI):

    python pallas_kernel/gen_orbit_greedy_kernel.py \\
        --slice 4,4,8 \\
        [--router ilp|dor] \\
        [--order lpt_tail_asc|lpt|spt|tail_asc] \\
        [--axis-names x,y,z] \\
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


def _coord_decode_exprs(
    flat_var: str, slice_: tuple[int, ...]
) -> tuple[str, ...]:
    """Per-axis decode expressions for `flat_var` given the slice layout."""
    exprs = []
    suffix_prod = 1
    for s in reversed(slice_):
        if suffix_prod == 1:
            exprs.append(f"{flat_var} % {s}")
        else:
            exprs.append(f"({flat_var} // {suffix_prod}) % {s}")
        suffix_prod *= s
    return tuple(reversed(exprs))


def _flat_encode_expr(
    coord_vars: tuple[str, ...], slice_: tuple[int, ...]
) -> str:
    """Flat-id expression from per-axis variables and slice."""
    terms = []
    suffix_prod = 1
    for var, s in zip(reversed(coord_vars), reversed(slice_)):
        if suffix_prod == 1:
            terms.append(var)
        else:
            terms.append(f"{var} * {suffix_prod}")
        suffix_prod *= s
    return " + ".join(reversed(terms))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_kernel_source(
    topology: Topology,
    router=None,
    *,
    order: str = "lpt_tail_asc",
    axis_names: tuple[str, ...] | None = None,
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
        axis_names: Mesh axis names, one per topology dim. Defaults to
            ("x", "y", "z", ...).
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

    if axis_names is None:
        default_axes = ("x", "y", "z", "w", "v", "u")
        if n_dim > len(default_axes):
            raise ValueError(f"No default axis names for ndim={n_dim}")
        axis_names = default_axes[:n_dim]
    if len(axis_names) != n_dim:
        raise ValueError(
            f"axis_names length {len(axis_names)} != ndim {n_dim}"
        )

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

    my_flat_encode = _flat_encode_expr(
        tuple(f"my_{a}" for a in axis_names), slice_
    )
    dst_decode = _coord_decode_exprs("dst_flat", slice_)
    # device_id dicts use *runtime* axis names from `axis_name[i]` so the user
    # can pass any mesh-axis name tuple matching topology dim order.
    device_id_dst = "{" + ", ".join(
        f"axis_name[{i}]: dst_{a}" for i, a in enumerate(axis_names)
    ) + "}"
    device_id_self = "{" + ", ".join(
        f"axis_name[{i}]: my_{a}" for i, a in enumerate(axis_names)
    ) + "}"
    axis_index_block = "\n".join(
        f"    my_{a} = jax.lax.axis_index(axis_name[{i}])"
        for i, a in enumerate(axis_names)
    )
    axis_size_expr = " * ".join(
        f"jax.lax.axis_size(axis_name[{i}])" for i in range(len(axis_names))
    )

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
    L.append(f'Mesh axes:       {axis_names}')
    L.append('')
    L.append('Regenerate via:')
    barrier_flag = " --per-step-barrier" if per_step_barrier else ""
    L.append(f'    python pallas_kernel/gen_orbit_greedy_kernel.py \\')
    L.append(f'        --slice {",".join(str(s) for s in slice_)} \\')
    L.append(f'        --router {router_name_for_doc.replace("Router", "").lower()} \\')
    L.append(f'        --order {order}{barrier_flag}')
    L.append('')
    L.append('Drop-in replacement for `_ragged_a2a_kernel_point_to_point`. To use:')
    L.append('  1. Copy this file next to reference_kernel.py.')
    L.append('  2. In ragged_all_to_all() change the line')
    L.append('         kernel = _ragged_a2a_kernel_point_to_point')
    L.append('     to:')
    L.append(f'         kernel = {function_name}')
    L.append('  3. Call ragged_all_to_all(...) with')
    L.append(f'         axis_name={axis_names}')
    L.append('     (a tuple of mesh axes corresponding to topology dims).')
    L.append('')
    L.append('See README.md for the twist explanation and integration steps.')
    L.append('"""')
    L.append('from __future__ import annotations')
    L.append('')
    L.append('import jax')
    L.append('import jax.numpy as jnp')
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
    L.append('# DEST_TABLE[src_flat, k] = flat destination device id for orbit k')
    L.append('# (in OrbitGreedy firing order) from source `src_flat`.')
    L.append('#')
    L.append(f'# Flat layout: my_flat = {my_flat_encode}')
    L.append(f'#   over mesh axes {axis_names} with sizes {slice_}.')
    L.append('#')
    L.append(f'# Per-source baking required because the {slice_} twisted-torus group')
    L.append('# composition is NOT elementwise modular. See README.md §Twist.')
    L.append(f'_DEST_TABLE_NP = {_dest_table_literal(dest_table)}')
    L.append(f'assert _DEST_TABLE_NP.shape == ({N}, {K}), (')
    L.append('    f"DEST_TABLE shape mismatch: {_DEST_TABLE_NP.shape}"')
    L.append(')')
    L.append('# JAX captures this as a closure constant when the kernel is traced.')
    L.append('DEST_TABLE = jnp.asarray(_DEST_TABLE_NP)')
    L.append('')
    L.append(f'# ORBIT_STEPS[t] = orbit indices firing at OrbitGreedy step t. {makespan_hop0} steps total.')
    L.append('ORBIT_STEPS = [')
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
    L.append('    Signature matches `_ragged_a2a_kernel_point_to_point`. Differences:')
    L.append(f'      * `axis_name` MUST be a {n_dim}-tuple matching slice={slice_}.')
    L.append('      * Iteration order = OrbitGreedy firing order (vs rotation).')
    L.append('      * Destinations are looked up in DEST_TABLE (twist-aware).')
    L.append('      * `transpose=True` is NOT supported (would need regen).')
    L.append('      * Assumes 1 group per device (uniform AllToAll).')
    L.append('    """')
    L.append('    assert scratch_ref is None')
    L.append('    del scratch_ref')
    L.append('    assert scratch_sems is None')
    L.append('    del scratch_sems')
    L.append('    assert not transpose, (')
    L.append('        "transpose=True not supported by orbit-greedy kernel; use reference."')
    L.append('    )')
    L.append(f'    assert isinstance(axis_name, tuple) and len(axis_name) == {n_dim}, (')
    L.append(f'        f"axis_name must be a {n_dim}-tuple matching slice={slice_}; got {{axis_name=}}"')
    L.append('    )')
    L.append('')
    L.append('    # Per-axis indices on this device:')
    L.append(axis_index_block)
    L.append(f'    my_flat = {my_flat_encode}')
    L.append(f'    axis_size_local = {axis_size_expr}')
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
    L.append(f'        _issue_packet(packet_idx, my_flat, {device_id_self})')
    L.append('        return _state')
    L.append('')
    L.append('    jax.lax.fori_loop(0, num_packets, _self_body, None)')
    L.append('')

    if not per_step_barrier:
        # ---- single fori_loop over orbits, final drain ----
        L.append('    # ---- main orbit loop: OrbitGreedy firing order ----')
        L.append('    def _orbit_body(k, _state):')
        L.append('        dst_flat = DEST_TABLE[my_flat, k]')
        for a, expr in zip(axis_names, dst_decode):
            L.append(f'        dst_{a} = {expr}')
        L.append(f'        dst_dev = {device_id_dst}')
        L.append('')
        L.append('        def _packet_body(packet_idx, _state2):')
        L.append('            _issue_packet(packet_idx, dst_flat, dst_dev)')
        L.append('            return _state2')
        L.append('')
        L.append('        jax.lax.fori_loop(0, num_packets, _packet_body, None)')
        L.append('        return _state')
        L.append('')
        L.append(f'    jax.lax.fori_loop(0, {K}, _orbit_body, None)')
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
        L.append('        dst_flat = DEST_TABLE[my_flat, k]')
        for a, expr in zip(axis_names, dst_decode):
            L.append(f'        dst_{a} = {expr}')
        L.append(f'        dst_dev = {device_id_dst}')
        L.append('        def _pb(packet_idx, _state):')
        L.append('            _issue_packet(packet_idx, dst_flat, dst_dev)')
        L.append('            return _state')
        L.append('        jax.lax.fori_loop(0, num_packets, _pb, None)')
        L.append('')
        L.append('    def _drain_step(step_indices):')
        L.append('        cum = 0')
        L.append('        for k in step_indices:')
        L.append('            cum = cum + sizes_ref[DEST_TABLE[my_flat, k]]')
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
    p.add_argument("--axis-names", default=None,
                   help="Comma-separated mesh axis names, e.g. x,y,z")
    p.add_argument("--per-step-barrier", action="store_true",
                   help="Emit per-OrbitGreedy-step barriers (best-effort).")
    p.add_argument("--function-name", default=None)
    p.add_argument("--out", default=None,
                   help="Output path (default: ./pallas_kernel/_ragged_a2a_kernel_orbit_greedy_<slice>.py)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    router, router_disp = _build_router(args.router, topology)
    axis_names = (
        tuple(args.axis_names.split(",")) if args.axis_names else None
    )

    src = generate_kernel_source(
        topology,
        router=router,
        order=args.order,
        axis_names=axis_names,
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
