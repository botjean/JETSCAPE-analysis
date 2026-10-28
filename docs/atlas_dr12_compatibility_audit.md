# ATLAS charged-kt analysis conventions

This optional observable runner does not replace existing STAT analyses.
Its measurement definition is documented in `atlas_dr12.md`; the distinctions
below matter when comparing implementations or pooling simulation outputs.

## Reconstruction

- Full-jet input retains undecayed pi0 by default, following the existing Hybrid
  input convention. The optional `decay_gg` mode is a sensitivity check, not a
  complete decay model or an experimental correction.
- Charged tracks use the existing fixed STAT charged-PID list. Calorimetric
  muons and neutrinos are excluded for this measurement.
- Selected anti-kt R=0.2 jets are reclustered with anti-kt R=1. Charged tracks
  are ghost-associated, restored to their original momenta, and reclustered
  with kt R=2.5. The Soft Drop selection is performed on this kt tree, not C/A.
- Default grooming selects the first passing upstream Lund split. The explicit
  `signed_prongs` alternative applies an additional negative-prong veto.
- Signed recombination preserves negative composite signs with marker -1.
  The marker-zero heppy reference is diagnostic only. Neither recombiner nor
  grooming alternatives are assumed equivalent for all possible events.
- Wake signs and grooming are model-side conventions, not ATLAS prescriptions.
  No detector correction, event-outlier cut or modification of existing
  observable reconstruction is introduced.

## Normalization and statistics

- All-event weight sums include events with no selected jet. Absolute yields
  are not replaced by unit-area distributions.
- Cross-section estimates are averaged per original histogram input file,
  not per event or computational subdivision. Original-input identities are
  retained so splitting a file cannot change its normalization weight.
- Default `per_jet` statistical errors follow ROOT Sumw2. The opt-in `event`
  mode uses event-level products and retains same-event bin covariance.
- Separately normalized centrality components may be mixed with explicit
  weights. The example equal-weight mixture is an assumption, not a supplied
  geometry determination; component errors use squared weights and the shared
  pp reference is propagated once.
- The exported per-design sidecar does not contain shared-pp cross-design
  covariance. Downstream inference must not invent independence from this
  omission. Generator cross sections and weight-sum normalization are fixed.

## Reviewable checks

Synthetic tests cover signs, ghost association, first-passing grooming,
cross-section pooling, computational split/merge invariance, per-jet versus
event errors, centrality mixing and table export. The command-line smoke uses
synthetic inputs. `scripts/validate_atlas_dr12_compatibility.py` accepts explicit
caller-provided files to compare the available reference algorithms; it does
not require or distribute a particular production sample.

Software parity is not physics closure. Before production, establish the
particle/decay conventions, generator normalization, centrality weights and
agreement with an appropriate published or independently verified spectrum.
