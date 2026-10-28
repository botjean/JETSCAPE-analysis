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

- Anti-kt R=0.2, pT>35 GeV, |eta|<3; exclude muons/neutrinos from
  calorimetric inputs. ATLAS defines stable particles with lifetime >30 ps;
  the default Hybrid input convention below is an explicit approximation to
  that definition, not a claim that generator decays have been matched.
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
production must be confirmed, not reconstructed from a PID list. By default,
`pi0_treatment: keep` retains the supplied Hybrid particles, including undecayed
pi0, without a lifetime veto. Charged PID selection matches the old STAT list.
Unknown PDG/status values still fail. This replaces the draft v2 lifetime
rejection, which was incompatible with the production files.

`pi0_treatment: decay_gg` is an explicit sensitivity test: force each retained
pi0 into two isotropic photons, with both daughters inheriting the wake sign.
It is not a full branching-ratio decay model (in particular, no Dalitz channel)
or a correction for every other generator decay convention. The random seed is
configured by `pi0_decay_seed`; event-content hashing makes file chunking/order
irrelevant. Float32 input can erase the small pi0 invariant mass, so the decay
restores the PDG mass at fixed three-momentum. Energy changes are recorded in
`particle_qa`; changes exceeding 1e-5 times max(1 GeV, E) fail. No other species
are decayed. The default keep mode changes no particle four-vectors.

Version v4 records treatment, grooming, error and normalization conventions and
refuses to combine different definitions or older output. See
[the behavioral audit](atlas_dr12_compatibility_audit.md).

### Hybrid wake prescription

Status -2 partons are excluded. pp uses status 0 only. PbPb uses 0/1 positively
and 2 negatively, as the Hybrid branch specifies. Small-jet and charged kt
clustering use signed four-vector recombination. Net-negative small jets cannot
seed physical large jets. The charged scalar sum is signed. The default
`grooming: lund` selects the first passing split from upstream LundGenerator,
matching the old GroomerShop selection on the specified kt tree. The previous
custom walk that skips negative prongs is available only through explicit
`grooming: signed_prongs`; these prescriptions are not universally equivalent.
Ghosts carry association labels, not wake signs; original signed tracks are
restored before the charged clustering. There is no double subtraction.

This is an explicit **model-side wake prescription**, not something specified
by the experimental particle-level definition. Reference-comparison tools do
not establish production-wide equivalence of the available prescriptions.
In particular, heppy's `NegativeEnergyRecombiner()` defaults its negative
composite marker to zero. The implementation here uses -1 (equivalent to explicitly
constructing its recombiner with a negative marker), preserving the sign through
subsequent merges. Config `negative_recombiner: sign_preserving` is the supported
production mode; the marker-zero implementation is a QA reference only.
The existing STAT code is not altered. This finding alone
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
Set `ATLAS_DR12_DATA_DIR` to the independently extracted table root containing
`Data/` to also check the exact bin edges against the archive; no private/local
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
it does not turn a 0-5% sample into a 0-10% sample. With a 0-10% target config,
analyze its inputs using `--centrality 0 5` and `--centrality 5 10`, respectively,
then use the explicit mixture below. The example testing assumption is equal
weights of separately normalized spectra, not pooled raw counts. Production
geometry weights must be established separately. No outlier rejection is
silently imposed. `--cross-section` overrides the header only when explicitly set.

Every sample writes raw weighted bin sums, per-jet squared weights, event-level
sum-of-outer-products, all-event weight sums, cross-section entries keyed by
original input identity, input hash, config and raw/weighted cut flow (NPZ + JSON).
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

Merge requires identical sample definitions, but permits different cross-section
estimates from independent original inputs. `legacy_input_file_mean` pools
the raw counts and all-event weights, then uses the arithmetic mean of sigma
over original histogram input files, matching the inspected ROOT path. This is
per original input file, not once per generator job or once per event. It is a
compatibility convention, not a new optimal cross-section estimator.
The default normalization identity is the input SHA256. When computationally
subdividing one original file, supply the same `--normalization-id` for those
pieces so sigma is counted once; do not give different original parquet files
one shared job identity. Same-identity sigma disagreements fail. Different, nonoverlapping
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

For separately generated subcentralities, replace a design point's list with:

```yaml
design_points:
  0:
    centrality_components:
      - centrality: [0, 5]
        weight: 0.5
        samples: [design0_0-5_pooled.npz]
      - centrality: [5, 10]
        weight: 0.5
        samples: [design0_5-10_pooled.npz]
```

Components must cover the target without gaps/overlaps, have disjoint sources,
and weights summing to one. Each component is normalized first. Independent
component covariances receive squared weights; the common pp denominator is
propagated once after mixing. Independence of the generated subcentrality
samples is an explicit assumption.

```bash
.venv/bin/python -m jetscape_analysis.analysis.atlas_dr12 export \
  --manifest /path/to/manifest.yaml --output results/dr12/tables
```

The output uses the existing `Data/`, `Design/`, `Prediction/` layout with
`design_pointN` columns and values/errors files. Data bins and design IDs are
validated; original Data and Design files are copied unchanged. The error table
contains propagated **MC statistics**, not experimental errors. Default
`mc_error: per_jet` uses sum(w^2 * number_of_jets_in_bin), matching ROOT Sumw2.
Optional `mc_error: event` uses event-level outer products, including same-event
bin covariance. Both raw accumulators are saved; the exported covariance sidecar
uses the selected convention (diagonal by default). The Bayesian error-table
reader does not automatically use this sidecar. A shared pp reference also
correlates predictions across design points; that cross-design covariance is
not stored in the per-design sidecar. Cross-section estimation and denominator
weight-sum uncertainty are not added: sigma/sumw is treated as fixed, like the
existing histogram normalization. Zero numerator/denominator bins refuse export because
their naive Gaussian MC error would be misleading.

sqrt(d12) is recorded simultaneously and can be exported with a separate
manifest. **Do not fit it jointly with Delta R12 as independent data**: both use
the same jets. Experimental covariance/source information is still missing from
the supplied archive. Converted total errors support disclosed correlation
hypotheses, not a recovered ATLAS covariance.

## Remaining physics validation

Synthetic tests cover varying-sigma pooling, computational split/merge
invariance, explicit centrality mixing, common-pp error propagation and exported
values/errors. The command-line smoke checks an analytic RAA=0.5 example.
These are software checks, not closure on actual Hybrid/ATLAS spectra.

The optional reference-comparison script accepts caller-provided input files
and compares the available recombination/grooming paths on the same ATLAS kt
tree. It does not reproduce an entire external analysis container. Synthetic
counterexamples demonstrate why the alternatives must not be assumed equivalent.

Before inference: confirm the generation/weight/centrality/decay conventions with the
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
