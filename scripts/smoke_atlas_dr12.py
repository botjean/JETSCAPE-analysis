"""Persist an explicitly synthetic ASCII -> native reconstruction -> Bayesian table smoke test."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "validation/atlas_dr12/synthetic_smoke")
OUT = parser.parse_args().output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
config = yaml.safe_load((ROOT / "config/atlas_dr12_5020.yaml").read_text())
edges = config["observables"]["dr12_atlas"]["edges"]
centers = 0.5 * (np.array(edges[:-1]) + edges[1:])
for system in ("pp", "PbPb"):
    records = []
    for i, angle in enumerate(centers):
        records += [
            f"# event {i}\nweight 1 cross 10 X 0 Y 0 cross_err 0 pthat 250\n",
            "150 0 0 0.13957 211 0\n",
            f"{80 * math.cos(angle):.15g} {80 * math.sin(angle):.15g} 0 0.13957 211 0\nend\n",
        ]
    if system == "PbPb":
        for i in range(len(centers), 2 * len(centers)):
            records += [f"# event {i}\nweight 1 cross 10 X 0 Y 0 cross_err 0 pthat 250\n", "1 0 0 0.13957 211 0\nend\n"]
    input_path = OUT / f"SYNTHETIC_hybrid_{system}.out"
    input_path.write_text("".join(records))
    cmd = [
        sys.executable,
        "-m",
        "jetscape_analysis.analysis.atlas_dr12",
        "analyze",
        "--input",
        str(input_path),
        "--config",
        "config/atlas_dr12_5020.yaml",
        "--output",
        str(OUT / f"{system}.npz"),
        "--system",
        system,
        "--stratum",
        "toy",
        "--confirm-particle-level-input",
        "--per-binary-hard-process",
    ]
    if system == "PbPb":
        cmd += ["--centrality", "0", "10", "--design-id", "0"]
    subprocess.run(cmd, cwd=ROOT, check=True)
data_name = "Data__5020__PbPb__inclusive_jet__dr12_atlas__R1.0_pt200-251_trk4_zcut0.15_beta0__0-10.dat"
np.savetxt(
    OUT / data_name,
    np.column_stack([edges[:-1], edges[1:], np.ones(15), np.ones(15)]),
    header="SYNTHETIC test data, not ATLAS measurement\nLabel xmin xmax y stat",
)
design_name = "Design__SYNTHETIC.dat"
(OUT / design_name).write_text(
    "# Version 1.0\n# Parameter u_Lres u_kappa\n# Design point indices (row index): 0\n0.5 0.5\n"
)
manifest = dict(
    observable="dr12_atlas",
    pp=["pp.npz"],
    design_points={0: ["PbPb.npz"]},
    data_table=data_name,
    design_table=design_name,
)
(OUT / "manifest.yaml").write_text(yaml.safe_dump(manifest))
subprocess.run(
    [
        sys.executable,
        "-m",
        "jetscape_analysis.analysis.atlas_dr12",
        "export",
        "--manifest",
        str(OUT / "manifest.yaml"),
        "--output",
        str(OUT / "tables"),
    ],
    cwd=ROOT,
    check=True,
)
values = np.loadtxt(next((OUT / "tables/Prediction").glob("*values.dat")))
np.testing.assert_allclose(values, 0.5, atol=1e-12, rtol=0)
report = dict(
    status="PASS",
    synthetic=True,
    bins=int(values.size),
    expected_raa=0.5,
    max_absolute_error=float(np.max(np.abs(values - 0.5))),
    explanation="Identical accepted jets; twice as many generated AA events. No shape normalization.",
)
(OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
