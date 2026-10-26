# Local Review Validation: 2026-10-26

Source verification: **86 imported files byte-identical**, plus unchanged
committed dR12 Python/C++ sources and card. The new source-package tests also
bind the nominal rg adapter/native source and endpoint seed/hardest-root sources
to the final executed control receipts. These are source identities, not a new
event-replay certification.

Bounded test run: **348 passed, 15 explicitly skipped, no failures**.

| Group | Passed | Skipped |
| --- | ---: | ---: |
| Source/package guards and covariance algebra | 16 | 0 |
| Original bounded transport, projection, certificate, thinning, symmetry, polygon tests | 150 | 1 |
| Original seed-scaling and collector-wiring tests | 52 | 3 |
| Original strict-first and synthetic prefix-tamper tests | 12 | 1 |
| Canonical rg and upstream integration tests | 46 | 0 |
| Original precision-boundary adversarial tests | 68 | 10 |
| New local hardest-root native smoke/guard tests | 4 | 0 |

The skips require deliberately omitted raw event or captured event-derived
fixtures. They are not silently replaced by invented inputs. The complete old
hardest-root native suite is preserved for scrutiny but was not run: it requires
omitted event fixtures and builds additional reference extensions.

The complete imported-source mapping is `SOURCE_MANIFEST.json`; the machine
test receipt, versions, commands, output and reused local binary hashes are
`TEST_EVIDENCE.json`. Default discovery retains the existing pytest configuration
and excludes the execution snapshot. The runner explicitly selects finite test
groups in isolated Python processes.

## Environment And Scope

Used an existing Python 3.12 analysis environment with numpy 2.5.3, scipy 1.17.1,
pytest 9.1.1, PyYAML 6.0.3, FastJet 3.5.2.0 and pybind11 3.1.0. SciPy was read
from an existing local vendor directory, supplied explicitly through PYTHONPATH;
it is not copied into this package. No environment was installed or modified.

Optional native checks reused existing local compiled extensions, copied into
ignored paths in this clone. No compilation was performed. Their exact hashes
and source locations are evidence metadata, not default build paths. These
local binaries are not distributed and are not claimed bitwise equal to all
cluster binaries. The NEW opt-in build helper was tested for fail-closed refusal
without its environment flag, but its compilation path was not exercised.

No original source files were changed. No raw events, SSH, downloads, cluster
jobs, event processing, emulation or inference fits were performed. No commit or
push was made. The heppy license ambiguity remains a public-release blocker.
