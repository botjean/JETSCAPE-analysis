# Opt-In ATLAS Groomed Radius

`jetscape_analysis.analysis.atlas_rg` and `simulation_covariance` are copied
byte-for-byte from the paper analysis sources. Existing dR12 sources, its card,
and all STAT production defaults are unchanged.

Full anti-kt R=0.4 jets use ungroomed pT>158 GeV and |y|<2.1. Soft Drop has
beta=0, zcut=0.2 and a strict passing condition. The observable is the absolute
splitting angle, not angle/R. Failed grooming is not a positive-rg measurement.
The explicit signed-prong convention is a Hybrid background prescription, not
an ATLAS instruction. No unit-area normalization substitutes for absolute yields.

Both supplied rg cards are diagnostic. The exact first positive integration
boundary and the historical particle decay/ancestry convention remain qualified.
The module's production-export gates are deliberately preserved.

The old `signed_prongs` native module is retained for its existing API and
regressions. The nominal paper bank used `upstream_signed_prongs`, with the
strict-first precision wrapper in the separate study snapshot. Building only
`scripts/build_atlas_rg.py` does not reproduce that bank.

See [the study README](../studies/hybrid_lres/README.md) for source pins, optional
builds, tests, and the distinction between scrutiny and exact execution replay.
