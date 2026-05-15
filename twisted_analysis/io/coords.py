"""Flatten / unflatten device coordinates.

Convention (matches pallas_kernel/gen_orbit_greedy_kernel.py:78):
    flat = c0 * prod(slice[1:]) + c1 * prod(slice[2:]) + ... + c_{n-1} * 1
i.e. dim 0 is most significant. Node (i, j, k) on slice=(4,4,8) maps to
flat = i*32 + j*8 + k.
"""
from __future__ import annotations
from typing import Sequence


def _strides(slice_: Sequence[int]) -> tuple[int, ...]:
    n = len(slice_)
    out = [1] * n
    for d in range(n - 2, -1, -1):
        out[d] = out[d + 1] * slice_[d + 1]
    return tuple(out)


def flatten(node: Sequence[int], slice_: Sequence[int]) -> int:
    if len(node) != len(slice_):
        raise ValueError(
            f"node has {len(node)} dims; slice has {len(slice_)}"
        )
    for d, (c, s) in enumerate(zip(node, slice_)):
        if not (0 <= c < s):
            raise ValueError(f"coord {c} out of range [0, {s}) at dim {d}")
    strides = _strides(slice_)
    return sum(c * st for c, st in zip(node, strides))


def unflatten(flat: int, slice_: Sequence[int]) -> tuple[int, ...]:
    n = 1
    for s in slice_:
        n *= s
    if not (0 <= flat < n):
        raise ValueError(f"flat={flat} out of range [0, {n})")
    strides = _strides(slice_)
    out = []
    rem = flat
    for st in strides:
        out.append(rem // st)
        rem = rem % st
    return tuple(out)
