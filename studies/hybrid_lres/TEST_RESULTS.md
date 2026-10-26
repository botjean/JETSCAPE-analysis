# Release-Fix Validation: 2026-10-26

**82 packaged imports byte-identical; four heppy sources external-only.**
GroomerShop.cc/.hh and fjtools.cxx/.hh are absent from the source package. Their
paths, sizes, exact SHA256 and pinned commit remain dependency metadata. GPL
LundGenerator.cc/.hh and the supplied license text retain their original notices.
No license is assigned to the external heppy files.

Default source verification passed without a heppy checkout. Optional verification
against the existing local checkout at `16fcefbdca7a106e6e74fd53e904e302335dcce3`
also passed all four hashes. No sources were copied from that checkout.
Committed dR12 Python/C++ sources and card remain unchanged.

Current bounded run after the release fix: **238 passed, 5 explicitly skipped**.

| Group | Passed | Skipped |
| --- | ---: | ---: |
| Source/package/external dependency guards and covariance algebra | 24 | 0 |
| Original bounded transport, projection, certificate, thinning, symmetry, polygon tests | 150 | 1 |
| Original seed-scaling and collector-wiring tests | 52 | 3 |
| Original strict-first and synthetic prefix-tamper tests | 12 | 1 |

New release guards cover absent packaged heppy files, preserved GPL notices,
exact external hashes, rejection of wrong commits/modified sources/symlinks,
and refusal of all three heppy-dependent builds without `--heppy-root` before
native imports or compilation. The skips need omitted event-derived fixtures.

## Historical Native Evidence

The accepted **348-passed/15-skipped** native/local run is preserved unchanged in
`provenance/review_v1_86_files/TEST_EVIDENCE.json`, together with its original
manifest and report. It tested the pre-release-fix 86-file package using existing
local binaries. It is historical evidence, not a new native run on this revision.
No physics implementation source changed in this release fix.

Current `TEST_EVIDENCE.json` records the new source/pure-Python checks and links
the historical evidence by hash. Native compilation and event replay were NOT
rerun. The optional build helper now reads heppy only from an explicit supplied
checkout and requires both the pinned commit and exact source hashes.

## Environment And Scope

Used the existing Python 3.12 environment with numpy 2.5.3, scipy 1.17.1,
pytest 9.1.1 and PyYAML 6.0.3. SciPy was read from the existing local vendor
directory via explicit PYTHONPATH; it is not distributed. No dependency was
installed or modified. No originals, old branches or PR20 were altered.

No raw data, source downloads, network access, SSH, jobs, compilation, event
processing, fits, commits or pushes were performed. The stacked fork-PR base
remains `codex/atlas-dr12-hybrid` at `29060da6a216ae1d68fb6d1b3bdcd889d6a2a6cf`.
