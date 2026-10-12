# FreshSky

Experimental code for energy-aware uplink scheduling of multiple UAV sources over correlated air–ground blockage.

FreshSky combines the expected decrease in importance-weighted age with a separate virtual energy queue for each UAV. FreshSky-B adds a Bayesian LoS belief updated from ACK/NACK. A remaining-radio-energy filter enforces a finite mission allocation.

Authors: **Jiaqi Lin, Shenhu Zhang, and Shi Yan**, Beijing University of Posts and Telecommunications. Corresponding author: Shi Yan, yanshi01@bupt.edu.cn.

## Quick start

Python 3.12 is recommended. The recorded runs used Python 3.12.7.

~~~bash
git clone https://github.com/Laswell1551/FreshSky.git
cd FreshSky
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python scripts/reproduce.py --deep-geometry
~~~

The last command checks the supplied results, independently validates the small CMDP optimality certificates, checks finite radio balances and patrol feasibility, and re-ray-marches all ten primary city traces. After installing dependencies, numerical reproduction works offline without a GPU or an external account.

To regenerate all experiments and the existing six-panel figure:

~~~bash
python scripts/reproduce.py --full --deep-geometry
~~~

Individual commands and all seeds, parameters, scoring windows, calibration rules, and uncertainty conventions are in [PROTOCOL.md](PROTOCOL.md).

## Experiments

| Experiment | Entry point | Recorded evidence |
|---|---|---|
| Full-information scheduling, value ablation, control sweep and mission shift | scripts/run_icc.py | results/stationary_*, V_sweep_*, shift_* |
| Feedback memory and individual/fleet budget ablations | scripts/run_icc.py | results/belief_* |
| Finite radio reserve | scripts/battery_check.py | results/battery_guard_*, battery_reserve_* |
| Exact joint two-source CMDP reference | scripts/small_cmdp_check.py | results/small_cmdp_* |
| Transition/likelihood mismatch, with and without a hard reserve | scripts/mismatch_check.py; add --unguarded for the latter | results/mismatch_* |
| Five independent feasible patrol sets per city, five paired stochastic repetitions per set | scripts/city_multitrajectory.py | results/city_multitrajectory_* |
| Independent numerical and geometric checks | scripts/check_strengthened_results.py --deep-geometry | results/strengthened_checks.json |
| Rebuild the 2 × 3 experimental figure | scripts/plot_grid.py | figures/grid_results.pdf / .svg / .png |

For the current urban evaluation, use **city_multitrajectory_\*** and **osm_patrol_\*_geo\*.npz**. The older single wrapped-trace results city_* and osm_traces_* remain solely to reproduce the original synthetic/map stress-test entry point. They are not the current multi-patrol comparison.

## Main recorded observations

The value-aware rule reduces weighted age by 9.3% relative to the value-agnostic queue rule in the main five-seed simulation. Under partial observation, the eight-slot blocked-dwell case gives a 13.9% reduction relative to the memoryless rule.

The matched two-source CMDP benchmark gives mean empirical gaps of 2.75% and 3.03% at age caps 8 and 12 for V=6. These are small-model measurements, not a general competitive-ratio guarantee.

Across the five independent patrol sets per city, London improves by 5.6% in capped age and 15.2% in uncapped age. Manhattan's mean increases by 1.1% and 4.3%, with substantial path variation. Error bars use the SD of five path means, each averaging five paired event/delivery seeds. Maps and ray marching provide simulated channels rather than measured flight links.

The deterministic queue/window guarantee allows an explicit additive finite-window excess. The radio reserve filter supplies the separate hard finite-allocation constraint.

## Layout

- scripts/: experiment drivers, schedulers, environments, independent checks, and plotting.
- data/: cached OSM maps and reproducible geometry traces.
- results/: per-run CSVs, summaries, configuration manifests, CMDP certificates, and shift traces.
- figures/: the system diagram and current experimental collage.
- ARTIFACT_MANIFEST.json: SHA-256 hashes of the release files.
- CITATION.cff: software citation metadata.

## License and data attribution

The original experiment and plotting code is licensed under [MIT](LICENSE). OpenStreetMap databases and their derived geometric traces retain the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/), as described in [DATA_LICENSE.md](DATA_LICENSE.md).

Building data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). Keep the attribution and data license when redistributing map caches or derived traces.

Please use the repository's **Cite this repository** information when referencing this software. This repository does not assert conference acceptance or a DOI.
