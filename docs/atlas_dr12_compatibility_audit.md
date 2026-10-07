# ATLAS dr12: differences from the previous Hybrid analysis

This is an optional new-observable runner, not a drop-in reproduction of the
cluster's existing analysis. Comparing against dev-hybrid-model 057ea21b and
the inspected cluster dev-stat-observables dff91df. The cluster source/config
was read without modifying it. Existing observable code, tables and Bayesian
inference settings have not been changed by this work.

## Defaults that should have matched and now do

| Item | Old pipeline | Original draft | Current local correction |
| --- | --- | --- | --- |
| Undecayed pi0 | Retained in full jets | Aborted on lifetime <=30 ps | Retain by default; named optional decay_gg sensitivity test |
| Charged PID selection | Fixed STAT charged list | Any known charged, sufficiently long-lived particle | Same fixed STAT list as old code |
| Input four-vectors | Used as supplied | Used as supplied after lifetime guard | Unchanged in keep mode; explicit mass restoration only for decayed pi0, with energy-shift QA |
| MC errors | Per-jet ROOT Sumw2 | Event-level outer products | Per-jet default restored; event-level option explicit |
| Cross-section pooling | Arithmetic mean per histogram input file, pooled counts/weights | Required identical sigma | Legacy input-file mean restored with explicit original-file identity |
| Grooming selection | First passing Lund split in GroomerShop | Custom walk rejecting negative prongs | Upstream Lund default; custom walk explicit alternative |

The original veto was an input guard motivated by the experimental definition,
not a removal of pion energy. Nevertheless, making it mandatory was an
unapproved incompatibility with actual Hybrid production. Retaining pi0 is the
legacy model approximation, not exact reproduction of ATLAS particle-level
decays. The optional gamma-gamma test does not implement all branching channels.

## Other differences, not changed silently in this correction

| Item | Difference and disposition |
| --- | --- |
| Calorimetric particle selection | Old generic full-jet inputs include muons/neutrinos; new observable excludes them, as ATLAS specifies. Retained and explicitly documented. |
| Jet definition | Selected anti-kt R=0.2 jets reclustered into R=1, charged tracks ghost-associated and reclustered with kt R=2.5, not the usual C/A Soft Drop tree. These are measurement-specific requirements, not changes to the existing observables. |
| Negative composite sign | New native recombiner preserves a negative sign via marker -1 instead of heppy's default marker 0. Documented in the original draft; effect on existing production not established. Not a claim of bitwise legacy reproduction. |
| Negative branches and charged fraction | Signed charged scalar-pT sum and rejection of net-negative small jets retained. Default Lund grooming does not add the custom negative-prong veto. These remain disclosed model-side conventions. |
| Grooming implementation | Upstream LundGenerator, selecting the first passing split on the ATLAS kt tree. Compared against the custom walk and native SoftDrop on real events; not a reproduction of the entire old container. |
| MC statistical errors | Default per_jet reproduces ROOT Sumw2; event uses event-level products and can change errors for multiple jets. Both raw accumulators retained. |
| Independent-input normalization | v4 accepts different sigma estimates and preserves the old arithmetic mean per original histogram input file. Explicit normalization identity makes computational splitting and hierarchical merging invariant. Different original parquet parts are still distinct entries, even from one generator job. |
| Centrality combination | v4 implements the user-approved 50:50 mixture of separately normalized 0--5 and 5--10 spectra, with independent component errors and common pp propagated once. This testing assumption is explicit, not claimed identical to raw-count pooling or final geometry weighting. |
| Empty bins | Export refuses zero/nonpositive AA or pp bins rather than silently emitting zero-error predictions. Intentional validation guard; does not justify removing those data bins. |
| Export/provenance | Explicit design IDs, exact bin matching, duplicate-source checks, input hashes, no-overwrite outputs, and complete supplied-design coverage. Validation restrictions, not physics selections. |
| Software path | New isolated Python 3.12/native FastJet runner; old cluster workflow uses ROOT/heppy/container and a different repository branch. Separate staging is required. |

## Confirmed unchanged choices

- Hybrid status mapping: shower 0, positive wake 1, negative wake 2; outgoing
  partons -2 excluded; pp selects status 0.
- No pre-cancellation of negative wake before signed clustering in this mode.
- Event weights retained; normalization uses all events, not only selected jets.
- No unit-area normalization of the measured spectra.
- No event-outlier rejection. The actual old STAT_5020.yaml also has
  `do_event_outlier_rejection: false`.
- No detector smearing/efficiency correction added to truth predictions.
- No changes to design coordinates, emulators, priors or experimental covariance.
- Only shared existing Python edit in the PR remains the bounded backward-seek
  reader fix. The legacy analysis driver itself was not edited.

## Current tests and remaining gate

The pi0 paired pilot uses the same input events and weights in both modes and
compares cut flow and spectra. Its difference errors use per-event paired
differences, not independent errors assigned to two copies of the same sample.
Zero differences in a small sample do not establish a zero production effect.
No full-statistics RAA or inference conclusion follows from this check.

The gamma-gamma mode preserves signed three-momentum; reconstructing the PDG
mass gives a logged rounding-scale energy change for float32 inputs. Exact
daughter masslessness, four-vector conservation on the restored mass shell,
wake signs and deterministic seeds are regression-tested.

Run the comparison with the isolated test environment and optional plotting
dependency `matplotlib==3.10.7`:

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_atlas_dr12_pi0.py \
  --pp /path/to/pp.parquet --aa05 /path/to/aa05.parquet \
  --aa510 /path/to/aa510.parquet --max-events 5000 \
  --output /new/local/comparison-directory
```

Before cluster production: confirm generator conventions and review the
remaining disclosed model choices. Local normalization, mixture and exported
error tests now pass. Do not label this entire runner "unchanged legacy analysis".

## v4 compatibility checks, 2026-10-07

All 74 tests pass after the final native rebuild. Analytic tests cover the
legacy cross-section mean, computational split/hierarchical merge invariance,
per-jet versus event-level errors, centrality mixing, common-pp propagation and
exported bin values/errors. Old-schema output is rejected rather than mixed.

Five algorithm paths were compared on the same 14,370 downloaded events:
sign-preserving/custom walk, independently implemented heppy marker -1/custom
walk, sign-preserving/upstream Lund, sign-preserving/upstream SoftDrop, and old
heppy marker 0/upstream Lund. All selected jets and both spectra agree in this
pilot (156 pp, 97 central PbPb and 91 peripheral PbPb selected jets).
Marker zero does change the soft negative-jet counts. Synthetic counterexamples
also show that marker zero can change selected jets and that the custom
negative-prong veto can differ from Lund selection. Thus neither equivalence is
claimed universally. Production retains the sign-preserving recombiner and
restores upstream Lund selection; marker zero is QA-only.

The reference formulas were checked against the cluster's heppy fjtools.cxx,
GroomerShop.cc and LundGenerator.cc using read-only access. This is algorithmic
comparison on the new measurement's kt tree, not execution of the old container.
Full report: `validation/atlas_dr12/legacy_compatibility_20261007/comparison.json`.
No cluster jobs or files were changed for these checks.

## Real-event paired pi0 pilot, 2026-10-07 (before v4 default restoration)

User-approved download of the first parquet part from pp Run0001 and PbPb
Run0100/Run0500. All computation below ran locally, not on the cluster. Inputs
and full-file hashes are recorded in the result summary. Only design 0 and one
decay seed (0) were tested; this is not a full-statistics campaign.

| Sample | Events | Selected jets, keep -> decay | Integrated dr12 spectrum change |
| --- | ---: | ---: | ---: |
| pp | 5000 | 156 -> 155 | -0.917% |
| PbPb 0--5% | 5000 | 97 -> 94 | -3.464% |
| PbPb 5--10% | 4370 (whole part) | 91 -> 90 | -0.336% |

With the explicitly approved 50:50 mixture of separately normalized PbPb
spectra and the common pp spectrum, the integrated dr12 ratio changes from
0.43008 to 0.42614 (-0.916%). This is a noisy pilot ratio, not a production RAA
prediction or precision estimate of the decay systematic. Most bins do not
change, but sparse individual bins can move by tens of percent (about -46% in
one 0--5% dr12 bin), with comparably large paired statistical uncertainty.
The sqrt(d12) comparison shows the same sparse-bin caution.

No negative Soft-Drop splits were encountered in this pilot. That is not a
validation of the custom negative-prong prescription in all events. The largest
logged mass-shell energy restoration was 0.000121 GeV per decayed pi0.

Outputs: `validation/atlas_dr12/pi0_real_event_sensitivity_20261007/`, including
`summary.json`, and PDF/PNG comparisons for both observables. Both PNGs were
visually inspected. These outputs are local ignored validation artifacts.
