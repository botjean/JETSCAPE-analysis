# Hybrid Lres: Paper Analysis Source Review

This is a review package, not a new analysis or a claim of portable event replay.
No inference, simulation, SSH, job scheduling, raw events, or fitted artifacts
are included. The unrelated upstream STAT selection patch is out of scope.

The review branch stays based on `29060da6a216ae1d68fb6d1b3bdcd889d6a2a6cf`.
The user-directed publication target is a STACKED fork PR in
`botjean/JETSCAPE-analysis`, base `codex/atlas-dr12-hybrid`, not `main`.
The existing upstream dR12 PR20 and old branch are not changed by this package.

## Three Layers

1. Canonical `jetscape_analysis/analysis/atlas_rg.{py,cc}` and
   `simulation_covariance.py`, their tests and diagnostic rg cards are exact
   imports. The committed dR12 baseline remains unchanged.
2. `executed/` preserves the relevant original relative directory layout and
   byte-identical sources. It contains bounded transport, seed and geometry
   solvers, strict-first/precision policies, hardest-root dR12, event-count
   collectors and the transitive import/hash dependencies needed for scrutiny.
   Old-looking calibration/ratio helpers here are used dependencies, not a
   request to revive their old workflows.
3. `verify_sources.py`, `run_checks.py`, `build_native.py`, and the review tests
   are NEW tooling. They are not the executed cluster drivers. `provenance/`
   contains unedited execution receipts; historical absolute paths in those
   receipts and snapshots are evidence, never defaults of the new tooling.

`SOURCE_MANIFEST.json` maps every imported file to its workspace-relative origin,
or `archive!member`, SHA256, size, role and transformation. All imports are
byte-identical. New helper files are intentionally outside that imported-source
manifest and are reviewed as normal diffs. `CHANGED_FILES.txt` lists this change.
The current manifest contains 82 packaged files. The four heppy files are NOT
redistributed: their paths, SHA256, sizes and commit are in `external_dependencies`.
The previous 86-file manifest and 348-pass native evidence are retained as
historical metadata only under `provenance/review_v1_86_files/`.

## Scientific Identities

Nominal: the rg `precision_continuation/timeout_completion` lineage, including
strict-first recovery of the exact constituent-retention error. Successful
strict calls are returned untouched. The precision backend is NOT a universal
relaxation. Native old-observable tables remain the nominal central predictions;
moment-derived means must not silently replace them.

Wake: bounded separate-sign transport, fixed a=0 and a=1, with 31 physical design
IDs (30 training, holdout 38). It uses the later FastJet hardest-root dR12 policy
and same-ray seed-coordinate repair. Transformed inputs do not receive the
original float32/six-digit precision recovery. Original checkpoints, signed-root
checkpoints and later executions have distinct identities; their compatibility
certificate does not make them bitwise replays of one binary.

Do not combine the nominal 39-training-design bank with this reduced endpoint
bank as if they were matched statistics. Exact rg first-bin boundaries and legacy
particle lifetime/decay ancestry remain conditional. This is an extreme
paper-inspired sensitivity, not a calibrated theoretical uncertainty.

## Verification Without Native Compilation

From the repository root, using an existing Python environment:

```sh
python studies/hybrid_lres/verify_sources.py
python studies/hybrid_lres/run_checks.py
```

These source/pure-Python checks do not need heppy. To additionally validate an
already available checkout without copying any of its sources:

```sh
python studies/hybrid_lres/verify_sources.py --heppy-root "$HEPPY_ROOT"
```

Pure tests need Python >=3.11, numpy, scipy, pytest and PyYAML. No dependency is
installed by these commands. Each historical test group runs in its own process
to isolate legacy bare-import aliases. Groups have a five-minute limit. The runner
disables pytest plugin autoload and bytecode writes; logs/receipts go only under
ignored `validation/`. Missing dependencies fail visibly, not as a passing run.

The runner executes the original synthetic bounded-transport, polygon,
certificate, thinning, symmetry, seed-scaling, collector-wiring and strict-first
adversarial tests. Tests requiring omitted raw events/captured seed files are
explicitly skipped by the NEW `executed/conftest.py`. The synthetic strict-prefix
tamper test runs; controller replay is excluded. No test imports an omitted
scheduling controller.
Default pytest discovery excludes the historical execution snapshot; use the
runner rather than recursively collecting `executed/`.

## Optional Local Native Builds

Use an explicit environment with a C++17 compiler, Python development headers,
the Linux `fastjet==3.5.2.0` wheel and `pybind11==3.1.0`. Existing
`requirements-atlas-dr12.txt` records the original analysis environment, not a
request to install a large stack. New builds are opt-in:

```sh
PAPER_NATIVE_BUILD=1 python studies/hybrid_lres/build_native.py --target canonical
PAPER_NATIVE_BUILD=1 python studies/hybrid_lres/build_native.py --target upstream --heppy-root "$HEPPY_ROOT"
PAPER_NATIVE_BUILD=1 python studies/hybrid_lres/build_native.py --target precision --heppy-root "$HEPPY_ROOT"
PAPER_NATIVE_BUILD=1 python studies/hybrid_lres/build_native.py --target hardest
PAPER_NATIVE_BUILD=1 python studies/hybrid_lres/build_native.py --target wake --heppy-root "$HEPPY_ROOT"
python studies/hybrid_lres/run_checks.py --native
```

`CXX` optionally selects the compiler. Canonical/upstream builds write ignored
extensions alongside the canonical analysis modules; other extensions and build
receipts go in ignored `build/hybrid_lres`. Existing binaries are never replaced.
The upstream manifest fingerprints the actual wheel libraries, retaining the
canonical runtime checks. New build receipts are NEW local provenance, never
substitutes for the executed binary hashes in the historical receipts. GPL and
heppy licensing qualifications are in `native/THIRD_PARTY.md`.
`HEPPY_ROOT` is user-supplied, with no default location. Only upstream, precision
and wake targets require it: the checkout must be at commit
`16fcefbdca7a106e6e74fd53e904e302335dcce3`, and all four external source hashes
must match exactly. The build reads those files in place; it never downloads or
copies them into this package. Canonical and hardest targets do not need heppy.

## Explicit Replay Boundary

The preserved collectors are NOT runnable event-replay entrypoints here. They
refer to missing historical data, sealed payloads, approval controls, runtime
aliases and controller lookups. Some aliases exist only inside old archives;
`nominal_payload/` retains the relevant reuse-proof files. The manifest documents
these identities without rewriting any validator or claiming its old paths are
portable. No raw input can be reconstructed from the included source receipts.

Event-level replay would require separately supplied count/input artifacts,
validated remapping of every historical dependency and new equivalence tests.
That work, new fits, and a production default change are outside this package.

See `TEST_RESULTS.md` for the actual bounded validation performed for this review.
