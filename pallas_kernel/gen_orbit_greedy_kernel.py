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
        --slice 4,4,8 \\
        --routing-table fixtures/routing_table_4x4x8_twist.json

Default outputs:
    routing table: ./fixtures/routing_table_<slice>_<router>.json
    schedule:      ./fixtures/schedule_<slice>_<router>_<order>.json
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
from twisted_analysis.io.schedule import save_schedule, schedule_from_orbit_greedy
from twisted_analysis.topology import Topology, DORRouter, ILPRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dest_table_and_orbit_steps_from_schedule(
    schedule: list[dict], n: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Build _DEST_TABLE_NP[n, K] and _ORBIT_STEPS from schedule entries.

    Strategy:
      * Sort schedule by (round, src). For each src, the per-round destination
        sequence becomes the columns of _DEST_TABLE_NP[src].
      * The k-th column corresponds to one orbit; all sources agree on the
        round-of-column-k, so derive _ORBIT_STEPS once from src=0.
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

def generate_kernel_source(
    *,
    slice_: tuple[int, ...],
    router_name_for_doc: str,
    order: str,
    per_step_barrier: bool,
    function_name: str | None,
    dest_table: np.ndarray,
    orbit_steps: list[list[int]],
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
    makespan_hop0 = len(orbit_steps)

    L: list[str] = []

    L.append('"""Orbit-greedy P2P AllToAll Pallas TPU kernel.')
    L.append('')
    L.append('AUTO-GENERATED — DO NOT EDIT BY HAND.')
    L.append('')
    L.append(f'Topology:        slice={slice_}  (N={N} devices, ndim={n_dim})')
    L.append(f'Router:          {router_name_for_doc}')
    L.append(f'OrbitGreedy:     order={order!r}')
    L.append(f'Per-step barrier: {per_step_barrier}')
    L.append(f'Hop-0 steps:     {makespan_hop0}')
    L.append('')
    L.append('Generated from:  routing-table JSON + schedule JSON')
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
    L.append('')

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
    L.append(f'    """Orbit-greedy P2P AllToAll kernel for slice={slice_}."""')
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
        L.append(f'    _NUM_ORBITS = {K}')
        L.append('    def _body(i, _state):')
        L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
        L.append('        k = lax.rem(i, _NUM_ORBITS)')
        L.append('        dst_flat = dest_table_ref[my_flat, k]')
        L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
        L.append('        return _state')
        L.append('')
        L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
        L.append('')
        L.append('    send_amount = total_send_amount_ref[0]')
        L.append('    recv_amount = total_recv_amount_ref[0]')
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
    L.append('def build_pallas_call_kwargs():')
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
    p.add_argument("--per-step-barrier", action="store_true",
                   help="Emit per-OrbitGreedy-step barriers (best-effort).")
    p.add_argument("--function-name", default=None)
    p.add_argument("--routing-table-out", default=None, type=Path,
                   help="Where to save the generated routing table "
                        "(default: ./fixtures/routing_table_<slice>_<router>.json). "
                        "Ignored if --routing-table is given.")
    p.add_argument("--schedule-out", default=None, type=Path,
                   help="Where to save the schedule "
                        "(default: ./fixtures/schedule_<slice>_<router_or_loaded>_<order>.json)")
    p.add_argument("--out", default=None, type=Path,
                   help="Output kernel path "
                        "(default: ./pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py)")
    args = p.parse_args(argv)

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
            fixtures / f"routing_table_{slice_slug}_{router_slug}.json"
        )
        save_routing_table(topology, router, rt_path)
        print(f"[1/3] wrote routing table {rt_path}", file=sys.stderr)
        router_doc = f"{router_disp}Router"

    table = load_routing_table(rt_path)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table {rt_path} has {len(table)} sources; "
            f"slice {slice_} expects {topology.n_nodes}"
        )

    # Stage 2: schedule.
    schedule = schedule_from_orbit_greedy(topology, table, order=args.order)
    sched_path = args.schedule_out or (
        fixtures / f"schedule_{slice_slug}_{router_slug}_{args.order}.json"
    )
    save_schedule(schedule, sched_path)
    print(f"[2/3] wrote schedule     {sched_path}", file=sys.stderr)

    # Stage 3: kernel.
    dest_table, orbit_steps = _dest_table_and_orbit_steps_from_schedule(
        schedule, topology.n_nodes,
    )
    src = generate_kernel_source(
        slice_=slice_,
        router_name_for_doc=router_doc,
        order=args.order,
        per_step_barrier=args.per_step_barrier,
        function_name=args.function_name,
        dest_table=dest_table,
        orbit_steps=orbit_steps,
    )
    out_path = args.out or (
        _HERE / "outputs" / f"_ragged_a2a_kernel_orbit_greedy_{slice_kern}.py"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(src)
    print(f"[3/3] wrote kernel       {out_path} ({len(src):,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
