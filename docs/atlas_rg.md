# Opt-In Groomed-Radius Analysis

`jetscape_analysis.analysis.atlas_rg` implements an optional absolute groomed
splitting-angle analysis. `simulation_covariance` supplies reusable count-moment
and ratio-covariance math APIs. Existing analysis defaults are unchanged.

The supplied example card uses full anti-kt R=0.4 jets, ungroomed
pT>158 GeV, |y|<2.1, and Soft Drop beta=0, zcut=0.2 with a strict passing
condition. The observable is the absolute splitting angle, not angle/R.
Failed grooming is not counted as a positive-rg measurement, and unit-area
normalization is not substituted for absolute yields.

The signed-prong implementation is a model-side background convention, not
an experimental prescription. The example card is diagnostic: production
export requires independent bin-boundary evidence and particle-definition
provenance through the module's explicit validation gates.

Supported configuration groomers are `signed_prongs` and diagnostic
`lund_reference`, both implemented by the supplied native extension. The
unsupported `upstream_signed_prongs` mode is rejected; no unpublished backend or
runtime build manifest is required.

## Covariance APIs

`simulation_covariance` provides only `EventMoments`, `histogram_counts`,
`ratio_map`, and `ratio_bank`. There is no saved-STAT/dR12 extraction adapter,
file-joining utility, or covariance CLI. Ordinary STAT output is not a supported
input: its writer drops empty events and does not preserve the all-event weights
and immutable event identities needed for a reliable join.

Callers must supply already selected, caller-aligned per-event counts, include
empty events, and establish immutable event identity and source binding before
combining observables. Equal weights/pThat or row counts do not establish a
join. No named-column lookup or missing-column-to-zero inference is performed.
An explicitly supplied empty value list (or `None`) to `histogram_counts` means
an empty observable in that known event, not an absent producer column.
`EventMoments.update` appends the all-event weight denominator; `ratio_map`
returns values and a Jacobian for caller-defined linear numerator/denominator
maps. `ratio_bank` assumes independent AA designs and one independent shared pp
reference; it does not infer pairing or cross-design AA covariance.

The optional native extension is built by `scripts/build_atlas_rg.py`.
Synthetic behavior and input-validation tests are in `tests/test_atlas_rg.py`.
This package contains no simulation bank, fitted parameters, posterior results,
or project-specific execution/reproduction snapshots.
