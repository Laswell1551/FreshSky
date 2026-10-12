# FreshSky reproducibility protocol — v1.1.0

Public code: https://github.com/Laswell1551/FreshSky . Authors: Jiaqi Lin, Shenhu Zhang, Shi Yan; corresponding author Shi Yan (yanshi01@bupt.edu.cn). Code is MIT licensed. OpenStreetMap data and derived map traces retain ODbL 1.0 attribution. This software release does not assert conference acceptance.

## Commands

Python 3.12 with the pinned requirements.txt is sufficient. After dependency installation, all checks and experiments run offline without a GPU or external account.

    python scripts/reproduce.py --deep-geometry

This validates seed records and their summaries, actual invariant priors, physical speed units, matched dual parameters, independent LP occupation/Bellman certificates, common radio balances, and all ten primary urban LoS arrays.

    python scripts/reproduce.py --full --deep-geometry

This regenerates the experiments and the existing six-panel collage. The finite-state weighted CMDP LPs take several minutes in total on the tested CPU. Their saved policy caches include a protocol key and independently checkable occupation measures. A full run always refreshes the shift trace cache. plot_grid.py verifies its scored mean power against each recorded seed.

Individual stages: run_icc.py, battery_check.py, cmdp_sensitivity.py, small_cmdp_check.py, mismatch_check.py (also --unguarded), city_multitrajectory.py, check_results.py, protocol_checks.py, check_strengthened_results.py (--deep-geometry), plot_grid.py, all under scripts/. Runtime can be remeasured with runtime_check.py.

## Matched protocol

- Full information: 12 sources, three 20-MHz bands, 100-ms slots, 3000 slots, first 300 discarded, seeds 0–4, individual target 3 mW, max power 100 mW, weights 1/10, cap 500, default V=200. Radial sinusoidal patrols range from 50 to 400 m at height 100 m, with per-source peak speed 6–14 m/s. Phase uses elapsed seconds t*tau.
- Fixed price: eight candidates 0.02–3.2; disjoint calibration seeds 50/51 with 1500 slots. Select the lowest-age candidate satisfying every calibration UAV's scored budget. Selected price: 0.1.
- Matched dual: score w*d-lambda*c, step 1/V, zero initialization. FreshSky's normalized queue uses deduct-then-spend; projected dual uses max(lambda+step*(p-budget),0). These agree away from the floor and share the positive-score expenditure envelope. Both policies are swept over the same five V values.
- Tuned dual: zero initialization, candidates 0.003/0.01/0.03/0.1/0.3, the same disjoint calibration protocol. Select minimum age among budget-feasible candidates; if none is feasible, record that fact and select minimum excess. Selected step: 0.003; calibration met the tested budget.
- Weighted CMDP: Tang-style finite-state occupation relaxation adapted to current event weight times expected next age. Two event states, age cap 80, eight deliverable power bins plus a blocked bin, 6000 independent calibration slots per mission geometry. Bin upper powers make the LP power accounting conservative. One coupled LP enforces individual nominal power and average total activation <=3. Random requests are truncated uniformly to the instantaneous limit; zero-mass high-age states inherit service. Thus its large-fleet empirical cost is not an exact full-environment optimum. Nominal budget scale 0.5/0.75/1.0 is calibrated on seeds 50/51; selected scale: 1.0. Separate sensitivity uses (cap,bins)=(40,4),(80,4),(80,8) with this frozen scale. Saved occupation measures support independent flow, power, activation and objective checks. Legacy Tang-CMDP uses plain AoI and average-bin powers and remains a secondary artifact.
- Partial observation: 4000 slots, scoring at slot 300, qL/qN=0.90/0.02, weights 1/5, cap 120, V=6, target 0.4 mW. Blocked mean dwell is 2/8/20 slots. After clipping the transition probabilities, both the latent chain initialization and Memoryless prior use its actual invariant distribution. Potential deliveries and events are generated per source/slot and shared by all policies. Belief dual matches FreshSky-B's information, zero initialization and step 1/6.
- Finite missions: every compared full-information method uses the same attempt filter and 900/720 mJ per source over 300 s. Feedback controls use 148/118.4 mJ over the 3700-slot scored mission. Record remaining balance, whole/scored power, age, and last-30-second age/delivery rate/source-service fraction. A hard reserve is distinct from an average-power target.
- Small exact reference: N=2, M=1, observed current Markov channel, powers 1/1.2 mW, targets 0.3/0.36 mW, weights 1/5, qL/qN=0.90/0.02, caps 8/12. Joint occupancy LP enforces the per-slot action limit. Five paired 60000-slot runs discard 10000. The predeclared V=6 empirical mean gaps are 2.75%/3.03%; independent Bellman inequalities and primal/dual agreement verify each reference.
- Mismatch: same true channel/outcomes. Model A halves transitions and uses likelihoods 0.80/0.10; B doubles transitions, clipped at one, and uses 0.95/0.005. A 148-mJ hard reserve is shared; unguarded runs expose finite-window excess. One-slot error uses a true-model belief with the same observed history, never hidden states.
- Cities: Manhattan/London OSM 6-m grids, five feasible geometry seeds 1–5 each, five paired event/delivery seeds 0–4 each. Routes remain within bounds with at least 5-m height-grid clearance, 80–120-m altitude, speeds 6–14 m/s and no boundary wrapping. Each 2000-slot path freezes its model from the first 300 slots; score the remaining 1700 with 272 mJ/source and 1.6 mW target. Average stochastic seeds within each path before taking mean/SD over the five path means. Map replay is simulated geometry, not flight measurements. The older single wrapped-patrol city_* files are secondary, not Table III.

## Evidence mapping

| Paper item | Recorded sources |
|---|---|
| Table II | stationary_*; fixed_price_tuning.csv, dual_tuning.csv, weighted_cmdp_tuning.csv |
| Fig. 2(a) | V_sweep_*, matched_pareto_*, separate small_cmdp_* |
| Fig. 2(b),(c) | shift_* and shift_power_traces.npz |
| Fig. 2(d),(e) | belief_* and mismatch_* |
| Table III; Fig. 2(f) | city_multitrajectory_* and osm_patrol_* caches |
| Finite reserve and terminal service | battery_reserve_* and partial_battery_* |
| Weighted adapter discretization | cmdp_sensitivity_* and weighted_cmdp_policies/*.npz |
| Independent checks | measured_claims.json, protocol_checks.json, strengthened_checks.json |

Means and error bars use sample SD. Measured finite-window power and small-model cost gaps are empirical. The sample-path theorem gives bounded individual debt and an explicit additive expenditure envelope; the reserve filter enforces allocated finite energy. Neither establishes a global AoI competitive ratio.

## Map provenance

© OpenStreetMap contributors, https://www.openstreetmap.org/copyright, ODbL 1.0. Height priority: explicit height, levels*3 m, then 10 m default. Bounds: Manhattan (40.7485,-73.9911,40.7593,-73.9769), London (51.5100,-0.0900,51.5200,-0.0740). SHA-256 and byte sizes are listed in ARTIFACT_MANIFEST.json. Internal reviews and manuscript-change notes are excluded from the public repository.
