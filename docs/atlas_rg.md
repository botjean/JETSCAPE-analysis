# Opt-In Groomed-Radius Analysis

`jetscape_analysis.analysis.atlas_rg` implements an optional absolute groomed
splitting-angle analysis. `simulation_covariance` accumulates event-level
moments for correlated statistical propagation. Existing analysis defaults
are unchanged.

The supplied example card uses full anti-kt R=0.4 jets, ungroomed
pT>158 GeV, |y|<2.1, and Soft Drop beta=0, zcut=0.2 with a strict passing
condition. The observable is the absolute splitting angle, not angle/R.
Failed grooming is not counted as a positive-rg measurement, and unit-area
normalization is not substituted for absolute yields.

The signed-prong implementation is a model-side background convention, not
an experimental prescription. The example card is diagnostic: production
export requires independent bin-boundary evidence and particle-definition
provenance through the module's explicit validation gates.

The optional native extension is built by `scripts/build_atlas_rg.py`.
Synthetic behavior and input-validation tests are in `tests/test_atlas_rg.py`.
This package contains no simulation bank, fitted parameters, posterior results,
or project-specific execution/reproduction snapshots.
