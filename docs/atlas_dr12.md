# ATLAS charged kt-splitting predictions

## Scope and provenance

Base: `JETSCAPE/JETSCAPE-analysis`, branch `dev-hybrid-model`, commit
`057ea21b8da6a0784e9cd557d967edceaec363aa`. Local feature branch:
`codex/atlas-dr12-hybrid`. This is the upstream branch implementing Hybrid ASCII,
statuses and signed wake analysis. The precise commit that produced the existing
collaborator tables is **not recorded in those tables and is not established**.

This optional runner uses that branch's reader and normalization convention.
It does not change existing observable definitions, tables, Bayesian code,
emulators, priors, or fits. It is separate from the legacy ROOT/STAT driver:
that driver's Hybrid centrality fallback is hard-coded and its 2-D histogram
path fails for a single pT interval. Neither path is used here.

Files added: native reconstruction `analysis/atlas_dr12.cc`, Python event/
histogram/table driver `analysis/atlas_dr12.py`, config, build script,
tests and synthetic smoke test. The **only existing Python change** is bounding
the ASCII reader's backward seek at byte zero; files shorter than its read
buffer previously raised `ValueError`. Existing parser regressions are included.

## Definition

Implemented from [arXiv:2504.04805v2, sections 4-5](https://arxiv.org/html/2504.04805v2):

- Particle-level anti-kt R=0.2, pT>35 GeV, |eta|<3; exclude muons/neutrinos
  from calorimetric inputs. Stable particles must have lifetime >30 ps.
- Recluster selected small jets with anti-kt R=1.0; |eta|<1.3 and pT>200 GeV.
- Ghost-associate charged tracks, pT>4 GeV and |eta|<2.5, using 1 eV ghosts
  in that same R=1.0 clustering. Restore track four-vectors afterward.
- Require charged scalar-sum-pT / large-jet pT >0.2 before grooming.
- Cluster associated tracks with kt R=2.5, then decluster that tree with
  Soft Drop zcut=0.15, beta=0. **No C/A reclustering.** Grooming failures do not
  enter the spectra. Multiple kt jets raise an error rather than dropping tracks.
- Record Delta R12 using rapidity/azimuth and sqrt(d12)=min(pT1,pT2)*Delta R12.
  Config selects 200 <= pT <251 GeV (the reconstruction additionally requires
  pT>200), 0-10%, and the audited Figure 2 bins. Overflow/underflow are counted.

Muon exclusion is for the calorimetric inputs; charged stable muons can enter
the track list. No detector efficiency or detector-quality cuts are applied to
truth particles. The input format has no ancestry: correct decay/primary-particle
production must be confirmed, not reconstructed from a PID list. Short-lived
final-state particles cause an error: dropping an undecayed pi0 would lose its
energy, not implement the measurement. Unknown PDG/status values also fail.

### Hybrid wake prescription

Status -2 partons are excluded. pp uses status 0 only. PbPb uses 0/1 positively
and 2 negatively, as the Hybrid branch specifies. Small-jet and charged kt
clustering use signed four-vector recombination. Net-negative small jets cannot
seed physical large jets. The charged scalar sum is signed; negative branches
cannot serve as physical Soft-Drop prongs and are skipped when declustering.
Ghosts carry association labels, not wake signs; original signed tracks are
restored before the charged clustering. There is no double subtraction.

This is an explicit **model-side wake prescription**, not something specified
by the experimental particle-level definition. Compare it with the collaborator's
production prescription on real events before accepting physics predictions.
In particular, heppy's `NegativeEnergyRecombiner()` defaults its negative
composite marker to zero. The implementation here uses -1 (equivalent to explicitly
constructing its recombiner with a negative marker), preserving the sign through
subsequent merges. The existing STAT code is not altered. This finding alone
does not establish that the previous table production was affected.

FastJet clustering is native C++, bound through pybind11. This avoids the
FastJet wheel's Python-recombiner callback, which crashed in local tests.
It does not replace clustering with a Python approximation.

## Build and test (WSL/Linux)

From the repository root, in an isolated environment:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-atlas-dr12.txt
.venv/bin/python scripts/build_atlas_dr12.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_atlas_dr12.py
```

The pinned environment requires Python 3.12 and a Linux platform supported by
the wheels (the FastJet wheel requires glibc >=2.27), plus a C++17 compiler.
Use cluster modules or a compatible container if the login node is older; do
not replace dependencies silently in the inference environment. The build uses headers/library from the pinned
FastJet wheel and embeds that environment's library path; rebuild after moving
the environment. No ROOT/heppy installation is required for this isolated driver.

Tests exercise analytic two-prong kinematics, kt versus C/A, signed composites,
ghost association/uniqueness and momentum invariance, track and jet thresholds,
rapidity rather than eta, grooming failures, particle selections, explicit
centrality, event weights, multi-jet statistical errors, ASCII/parquet input,
chunk pooling, absolute RAA and Bayesian-format output. The persisted smoke
uses **synthetic**, not Hybrid-production, events, with known RAA=0.5 in all
15 published Delta R12 bins. Its outputs are under
`validation/atlas_dr12/synthetic_smoke`. The smoke refuses to overwrite a run.
Use `--output /new/path` to repeat it. Optional-feature tests skip if the native
module is not built; run the build command above before validating this feature.
Set `ATLAS_DR12_DATA_DIR` to the independently extracted experimental Data
directory to also check the exact bin edges against the archive; no private/local
data files are required for the portable tests.

## Generate one sample

Use the actual **event-level** Hybrid files, not the existing binned prediction
tables: the latter cannot reconstruct this new observable. ASCII and the branch's
standard event parquet format are accepted. A sample must belong to one generator
pThat stratum and, for PbPb, the declared centrality interval.

```bash
.venv/bin/python -m jetscape_analysis.analysis.atlas_dr12 analyze \
  --config config/atlas_dr12_5020.yaml --input /path/to/hybrid_pp.out \
  --system pp --stratum pthat200_400 --output results/dr12/pp_chunk0.npz \
  --confirm-particle-level-input --per-binary-hard-process

.venv/bin/python -m jetscape_analysis.analysis.atlas_dr12 analyze \
  --config config/atlas_dr12_5020.yaml --input /path/to/hybrid_PbPb_design0.out \
  --system PbPb --centrality 0 10 --design-id 0 --stratum pthat200_400 \
  --output results/dr12/design0_chunk0.npz \
  --confirm-particle-level-input --per-binary-hard-process
```

The confirmation flags are substantive: verify generator decays, that the input
is not detector-level or minimum-bias/UE-embedded, and that the header cross
sections/event weights have the STAT hard-process normalization. These cannot
be inferred from the file name. Centrality here declares the sample interval;
it does not turn a 0-5% sample into a 0-10% sample. Geometry weighting across
centrality sub-bins must be established separately. No outlier rejection is
silently imposed. `--cross-section` overrides the header only when explicitly set.

Every sample writes raw weighted bin sums, event-level sum-of-outer-products,
all-event weight sums, input hash, config and raw/weighted cut flow (NPZ + JSON).
PbPb design IDs are mandatory and checked when merging and exporting; the
common pp reference has no design ID. Confirm the declared ID against the
actual generator parameters: the event format does not encode them. The stratum
label must describe identical generator phase space across independent chunks.
Normalize by `sigma * sum(w*n_bin) / sum(w_all_events)`, including events with
zero selected jets. Do not normalize each substructure spectrum to unit area.
For per-binary hard-process Hybrid samples, RAA is the AA/pp cross-section ratio;
do **not** apply an additional experimental TAA factor. Cross-section units
must match. These conventions must be checked against the generator production.

Pool independent chunks of the same pThat stratum before normalization:

```bash
.venv/bin/python -m jetscape_analysis.analysis.atlas_dr12 merge \
  --inputs results/dr12/pp_chunk0.npz results/dr12/pp_chunk1.npz \
  --output results/dr12/pp_pthat200_400.npz
```

Merge requires identical sample definitions and cross sections. If independent
jobs estimate different cross sections, choose the production-level common
cross section explicitly rather than silently averaging. Different, nonoverlapping
pThat strata are listed separately in the export manifest; their normalized
cross sections are added. User-supplied stratum labels cannot prove phase-space
coverage or nonoverlap: check generator cuts. Repeated stratum/source hashes fail.

## Export Bayesian tables

Example manifest, with paths relative to the manifest (list **every** existing
design ID, not just the abbreviated example below):

```yaml
observable: dr12_atlas
pp: [pp_pthat200_400.npz, pp_pthat400_800.npz]
design_points:
  0: [design0_pthat200_400.npz, design0_pthat400_800.npz]
  # Add all remaining design IDs; preserve the existing unit-square design file.
data_table: Data__5020__PbPb__inclusive_jet__dr12_atlas__R1.0_pt200-251_trk4_zcut0.15_beta0__0-10.dat
design_table: Design__Lres-E-loss.dat
```

```bash
.venv/bin/python -m jetscape_analysis.analysis.atlas_dr12 export \
  --manifest /path/to/manifest.yaml --output results/dr12/tables
```

The output uses the existing `Data/`, `Design/`, `Prediction/` layout with
`design_pointN` columns and values/errors files. Data bins and design IDs are
validated; original Data and Design files are copied unchanged. The error table
contains propagated **MC statistics**, not experimental errors. A covariance
sidecar retains same-event bin covariance; the current Bayesian error-table
reader does not automatically use this sidecar. A shared pp reference also
correlates predictions across design points. Cross-section estimation uncertainty
is not added; per-event MC errors treat sigma/sumw as fixed, like the existing
histogram normalization. Zero numerator/denominator bins refuse export because
their naive Gaussian MC error would be misleading.

sqrt(d12) is recorded simultaneously and can be exported with a separate
manifest. **Do not fit it jointly with Delta R12 as independent data**: both use
the same jets. Experimental covariance/source information is still missing from
the supplied archive. Converted total errors support disclosed correlation
hypotheses, not a recovered ATLAS covariance.

## Remaining physics validation

Local verification on 2026-10-07: the complete repository pytest suite and the
end-to-end synthetic smoke test were run. Report:
`validation/atlas_dr12/pr-review-tests.xml`. The command-line smoke
reproduced RAA=0.5 in all 15 Delta R12 bins with zero numerical residual; report:
`validation/atlas_dr12/synthetic_smoke/validation.json`. These are software checks,
not a closure claim on actual Hybrid/ATLAS spectra.

No real Hybrid events were available for this implementation test. Before
inference: confirm the generation/weight/centrality/decay conventions with the
collaborator; reproduce their pp spectrum and at least one published/existing
Hybrid benchmark; inspect cut flow, signed-branch rate and statistical coverage;
then generate the full design bank and validate a separate Delta R12 emulator.
The code is tested; agreement of real physics predictions is not yet established.

## Cluster handoff

Start with pp and one PbPb design point, not the full bank. Return the NPZ/JSON
sample outputs first so we can inspect normalization, cut flow and per-bin MC
coverage before spending production time. Once those checks pass, run the same
configuration for every design point and export the tables. Return the entire
export directory, including `QA/`, together with the sample JSON files and job
logs. Keep the large event files on the cluster. The PR can be tested directly
on its feature branch; it need not be merged before this pilot validation.
