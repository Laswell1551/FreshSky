# FreshSky reproducibility package

This package accompanies **FreshSky: Energy-Aware UAV Uplink Scheduling over Correlated Air–Ground Blockage**, an ICC 2027 submission draft by Jiaqi Lin, Shenhu Zhang, and Shi Yan. Corresponding author: Shi Yan (yanshi01@bupt.edu.cn). It contains code, cached map data, individual run records, and numerical certificates for the current manuscript. The experiment code is MIT licensed; map databases retain ODbL 1.0.

## Requirements and commands

Python 3.12 was used. Install the versions in requirements.txt. NumPy, pandas, SciPy, matplotlib, requests, and PyMuPDF are sufficient. No GPU, remote account, network download, or training service is required for reproduction after dependencies are installed.

From the package root:

    python scripts/reproduce.py --deep-geometry

This validates recorded headline numbers, independent CMDP certificates, seed counts, physical route samples in the height-grid model, all primary LoS arrays, and finite radio balances.

    python scripts/reproduce.py --full --deep-geometry

This regenerates the numerical experiments and six-panel current collage. Existing raw geometry caches are used offline. To force regeneration of primary patrol paths, remove only data/osm_patrol_*_geo*.npz; city_multitrajectory.py will recreate them. To regenerate the rolling shift curves, remove results/shift_power_traces.npz; plot_grid.py checks regenerated per-seed metrics against the recorded CSV before caching them.

Individual experiments:

    python scripts/run_icc.py
    python scripts/battery_check.py
    python scripts/small_cmdp_check.py
    python scripts/mismatch_check.py
    python scripts/mismatch_check.py --unguarded
    python scripts/city_multitrajectory.py
    python scripts/check_strengthened_results.py --deep-geometry
    python scripts/plot_grid.py

run_icc.py also reproduces the initial one-trajectory wrapped map stress test; those city outputs are historical and are superseded by city_multitrajectory.py for the current manuscript's Table III / Fig. 2(f). Always run the latter stage before plotting the final collage. Runtime timings can be regenerated with scripts/runtime_check.py.

## Protocol

- Main synthetic comparisons: 12 sources, three bands, 3000 slots, first 300 discarded, seeds 0–4; main target 3 mW per source. Figures and tables report sample SD.
- Static-price selection: prices 0.02–3.2, separate seeds 50 and 51, selected price 0.1. No evaluation seed participates in tuning.
- Partial observation: 4000 slots, scoring starts at 300, qL=0.90/qN=0.02, weights 1/5, cap 120, V=6, per-UAV allowance 0.4 mW. One potential outcome per UAV and slot is pre-generated and shared across policies.
- Finite reserve: the separate shifted full-information tests use 900 and 720 mJ per source. New mismatch checks use 148 mJ per source over their 3700-slot scored mission. Existing calibrated results are unchanged by this nonbinding guard. The guarded Memoryless equivalence was checked on all 15 seed/dwell combinations.
- Small exact reference: N=2, M=1, known current Markov channel states, fixed attempt powers 1/1.2 mW, individual allowances 0.3/0.36 mW, qL/qN as above, each channel transition 0.125, event occupancy 0.08 and off probability 0.04, weights 1/5, caps 8/12. Each policy uses 60000 slots and discards 10000; seeds 0–4. The joint LP enforces the per-slot action limit directly, not just an average fleet constraint. Both policies see the same state sequence and potential deliveries. V is swept over 2, 6, 20, 50, 200; the paper reports the predeclared V=6 case.
- Model mismatch: true channel/outcome processes stay fixed. Model A uses 0.5× transitions and estimated likelihoods 0.80/0.10; B uses 2× transitions (clipped at one) and 0.95/0.005. Both keep individual queues and the same hard reserve. A separate unguarded run records the finite-window overage. The same-history true-model belief provides the one-slot sensitivity reference; it never sees the hidden state.
- Cities: frozen OSM maps for Manhattan/London, 6-m grids, five geometry seeds 1–5. Each fleet follows obstacle-avoiding free-cell paths with at least 5-m height-grid clearance, altitude 80–120 m, speed 6–14 m/s, and no boundary wrapping. Each 2000-slot path calibrates its own frozen model on the first 300 slots. Five paired delivery/event seeds 0–4 score the remaining 1700 slots. Both policies have 272 mJ per source and a 1.6-mW allowance. Statistics first average the five stochastic repetitions for each path, then take the mean and sample SD over the five independent path means. Plotted dots are those path means.

## Evidence mapping

| Paper item | Source |
|---|---|
| Table II; Fig. 2(a) curves | stationary_* and V_sweep_* CSVs |
| Fig. 2(a) small-reference annotation | small_cmdp_* CSV / JSON / NPZ; independent Bellman dual check |
| Fig. 2(b),(c) | shift_* CSVs; shift_power_traces.npz |
| Fig. 2(d) | belief_* and mismatch_* CSVs |
| Fig. 2(e) | belief_* component comparisons (unguarded) |
| Table III; Fig. 2(f) | city_multitrajectory_* CSVs; osm_patrol_* caches |
| Mission reserve test | battery_reserve_* CSVs |
| Correctness checks | manifest.json, strengthened_checks.json, memoryless_guard_equivalence.json |

The small LP certificates and empirical cost gaps do not establish a global AoI competitive ratio or large-system optimality. The queue/window proof allows finite-window excess; the physical radio reserve guard enforces a different hard constraint. Map data and ray marching are simulated geometry, not flight channel measurements. The Manhattan result is not a universal improvement claim.

## Maps and provenance

Building data © OpenStreetMap contributors, ODbL: https://www.openstreetmap.org/copyright . Raw cached data, converted heights, and derived traces retain this attribution. Explicit height is preferred; building:levels × 3 m is next; otherwise 10 m is used. The actual paper describes unequal height coverage between cities. The archive's ARTIFACT_MANIFEST.json lists relative file paths and SHA-256 hashes. 
