# FreshSky experiment releases

## v1.1.0 — 2026-10-12

This release corrects the experimental protocol used in v1.0.0. Its rerun CSVs and figures supersede the earlier synthetic comparisons.

- Synthetic sinusoidal patrol phase uses elapsed seconds, with peak radial speeds of 6–14 m/s. The old phase treated slot indices as seconds and produced unrealistic speeds.
- Partial-observation priors use the invariant distribution of the actual clipped transition matrix. The old dwell-two prior disagreed with that matrix.
- A projected-dual control matches step 1/V and zero initialization; a separate dual control is calibrated on disjoint seeds 50/51. The old comparison used a different step and initial price.
- An event-weighted, conservative-power finite-state CMDP adapter replaces the plain-AoI adapter in the main comparison. LP occupation measures are saved and independently verified. Age/power-bin sensitivity is recorded. The legacy plain-AoI run remains a secondary artifact.
- Shared hard radio reserves compare FreshSky and the controls, including last-30-second service metrics.
- Checks recompute summaries and comparisons from run records rather than requiring predetermined performance improvements.

The six-panel layout and original system diagram are preserved. The separate two-source exact CMDP benchmark and cached feasible urban paths are unchanged; the latter are replayed and their geometry is independently rechecked. v1.0.0 remains in Git history for traceability.

## v1.0.0 — 2026-10-12

Initial public experiment code, map caches, and recorded evidence.
