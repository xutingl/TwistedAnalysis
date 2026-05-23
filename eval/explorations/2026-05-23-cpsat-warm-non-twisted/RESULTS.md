# Results — CP-SAT-Warm Non-Twisted Torus (2×2×4, 2×4×4)

| Cell | N | LB | orbit_greedy_full | CP-SAT makespan | Gap to LB | Runtime | Kernel |
|---|---:|---:|---:|---:|---:|---:|---|
| 2×2×4 | 16 | 8 | 13 | 8 | 0 | 0.8s | `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py` |
| 2×4×4 | 32 | 16 | 33 | 16 | 0 | 321.0s | `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py` |

Both cells reach `Gap to LB = 0`, i.e. both schedules are LB-tight — provably optimal under the physical-edge capacity model. `Gap to LB` is `CP-SAT makespan − LB`; when nonzero the `t_uppers_tried` field in `results.json` shows which step succeeded.
