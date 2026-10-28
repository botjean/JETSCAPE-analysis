"""Opt-in ATLAS full-jet RAA(rg) extraction; never changes existing STAT outputs.

Unfolded particle-level definition: arXiv:2211.11470, sections 4.1--4.4.
The staged experimental tables do NOT establish the actual histogram edges.
Production export requires independently verified edges and input provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from particle import Particle

from jetscape_analysis.analysis import _atlas_rg
from jetscape_analysis.analysis.atlas_dr12 import particle_rows, read_events

VERSION = "atlas-rg-v1"
PT_BINS = ((158, 200), (200, 315), (315, 501))


def reconstruction_hashes():
    from jetscape_analysis.analysis import atlas_dr12

    paths = [Path(__file__), Path(_atlas_rg.__file__), Path(atlas_dr12.__file__)]
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def validate_config(config, production=False):
    if config.get("sqrt_s") != 5020 or config.get("centrality") != [0, 10]:
        raise ValueError("This implementation requires 5020 GeV and target centrality [0,10]")
    if config.get("particle_level") not in ("atlas_30ps", "legacy_diagnostic"):
        raise ValueError("Declare particle_level atlas_30ps or legacy_diagnostic")
    if config.get("groomer") not in ("signed_prongs", "lund_reference"):
        raise ValueError("Declare signed_prongs or diagnostic lund_reference grooming")
    edges = np.asarray(config["rg_edges"], dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
        or edges[0] < 0
        or edges[-1] != 0.4
    ):
        raise ValueError("Require finite increasing rg edges within [0,.4]")
    if production and (
        config["particle_level"] != "atlas_30ps"
        or config["groomer"] != "signed_prongs"
        or config.get("binning_status") != "verified"
        or not config.get("binning_source")
    ):
        raise ValueError("Production requires verified binning evidence and the declared particle-level prescription")
    return edges


def rows_for_event(event, system, config):
    rows = particle_rows(event, system)
    if config["particle_level"] == "atlas_30ps":
        for pid, status in zip(event["particle_ID"], event["status"], strict=True):
            if status == -2 or (system == "pp" and status != 0):
                continue
            if abs(int(pid)) in (12, 13, 14, 16):
                continue
            lifetime = Particle.from_pdgid(int(pid)).lifetime  # particle uses ns
            if lifetime is None or np.isnan(lifetime):
                raise ValueError(f"Cannot verify the >30 ps particle definition for PDG {pid}")
            if lifetime <= 0.030:
                raise ValueError(
                    f"PDG {pid} violates the >30 ps particle definition; re-hadronize, do not drop its energy"
                )
    return rows


def count_event(event, config, system):
    """Unweighted vector to append to other observables BEFORE joint moments.

    Its three pT blocks are absolute tagged yields. Untagged, underflow and
    overflow are separate cutflow entries, never folded into the first bin.
    """
    validate_config(config)
    return count_rows(rows_for_event(event, system, config), config)


def count_rows(rows, config):
    """Count explicitly prepared rows, including paired input-treatment controls.

    This low-level hook does not certify particle-level provenance. Production
    callers must still use the input and export gates; count_event checks the
    declared particle prescription before reaching this identical counter.
    """
    edges = validate_config(config)
    result = _atlas_rg.reconstruct(rows, config["groomer"])
    counts = np.zeros((3, len(edges) - 1))
    qa = Counter(result["cutflow"])
    for i, (low, high) in enumerate(PT_BINS):
        selected = [j for j in result["jets"] if j[0] > 158 and low <= j[0] < high]
        tagged = np.array([j[1] for j in selected if j[3]], dtype=float)
        visible = tagged[(tagged > 0) & (tagged >= edges[0]) & (tagged < edges[-1])]
        counts[i] = np.histogram(visible, edges)[0]
        qa[f"pt{i}_untagged"] += sum(not j[3] for j in selected)
        qa[f"pt{i}_underflow"] += int(np.count_nonzero((tagged <= 0) | (tagged < edges[0])))
        qa[f"pt{i}_overflow"] += int(np.count_nonzero(tagged >= edges[-1]))
        qa[f"pt{i}_tagged"] += len(tagged)
    return counts.ravel(), qa


def analyze(event_streams, design_ids, config, system, centrality=None, pairing_id=None, input_provenance=None):
    """Joint moments across bins, pT slices AND paired AA designs.

    Each design has its own all-event denominator. A declared pairing cohort
    is mandatory for multiple AA designs; equal pThat alone is not evidence.
    Generator cross sections are retained as fixed recorded estimates.
    """
    edges = validate_config(config)
    if system not in ("pp", "PbPb"):
        raise ValueError("Expected pp or PbPb")
    n = len(event_streams)
    if n == 0 or len(design_ids) != n or len(set(design_ids)) != n:
        raise ValueError("Need one stream for each unique design ID")
    if system == "pp" and (design_ids != [None] or centrality is not None):
        raise ValueError("Common pp has no design ID or centrality")
    if system == "PbPb":
        if (
            centrality is None
            or len(centrality) != 2
            or not 0 <= centrality[0] < centrality[1] <= 10
            or any(type(i) is not int or i < 0 for i in design_ids)
        ):
            raise ValueError("Declare actual AA centrality and nonnegative design IDs")
        if n > 1 and not pairing_id:
            raise ValueError("Multiple designs require explicit common-event pairing provenance")
    if config["particle_level"] == "atlas_30ps" and not input_provenance:
        raise ValueError("Require provenance confirming >30 ps hadronization and no weak-decay feed-down")
    features = 3 * (len(edges) - 1)
    width = features + 1
    total, outer = np.zeros(n * width), np.zeros((n * width, n * width))
    jet_sumw2 = np.zeros((n, features))
    sigmas = [None] * n
    cutflow = [Counter() for _ in design_ids]
    events = 0
    seen_identities = set()
    for batch in itertools.zip_longest(*event_streams):
        if any(e is None for e in batch):
            raise ValueError("Paired streams have unequal event counts")
        if n > 1:
            identities = []
            for event in batch:
                candidates = [event[k] for k in ("event_ID", "event_number", "event_id") if k in event]
                if (
                    not candidates
                    or any(not np.isfinite(v) or int(v) != v or v < 0 for v in candidates)
                    or any(v != candidates[0] for v in candidates)
                ):
                    raise ValueError("Paired streams require unambiguous event identities")
                identities.append(int(candidates[0]))
            if len(set(identities)) != 1 or identities[0] in seen_identities:
                raise ValueError("Paired event identities differ or repeat within one cohort")
            seen_identities.add(identities[0])
            for key in ("event_weight", "pt_hat"):
                values = [e.get(key) for e in batch]
                if any(v is None or not np.isfinite(v) for v in values) or any(v != values[0] for v in values):
                    raise ValueError(f"Paired event alignment failed for {key}")
        vector = np.zeros(n * width)
        for i, event in enumerate(batch):
            weight, sigma = float(event["event_weight"]), float(event["cross_section"])
            if not np.isfinite(weight) or weight < 0 or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Invalid event weight or recorded cross section")
            if sigmas[i] is not None and not np.isclose(sigma, sigmas[i], rtol=1e-6, atol=0):
                raise ValueError("Cross section changes inside one input file")
            sigmas[i] = sigma
            actual = event.get("centrality", -1)
            if not np.isfinite(actual):
                raise ValueError("Nonfinite event centrality")
            if centrality is not None and actual >= 0 and not centrality[0] <= actual < centrality[1]:
                raise ValueError("Event centrality contradicts sample provenance")
            counts, qa = count_event(event, config, system)
            vector[i * width : (i + 1) * width] = weight * np.r_[counts, 1.0]
            jet_sumw2[i] += weight**2 * counts
            cutflow[i].update(qa)
        total += vector
        outer += np.outer(vector, vector)
        events += 1
    if events == 0 or any(total[(i + 1) * width - 1] <= 0 for i in range(n)):
        raise ValueError("Empty or zero-weight sample")
    return dict(total=total, outer=outer, jet_sumw2=jet_sumw2), dict(
        version=VERSION,
        config=deepcopy(config),
        system=system,
        centrality=centrality,
        design_ids=design_ids,
        pairing_id=pairing_id,
        input_provenance=input_provenance,
        events=events,
        cross_sections=sigmas,
        cutflow=[dict(c) for c in cutflow],
        normalization="per_binary_hard_process",
        covariance="joint_event_poisson_with_denominators",
        reconstruction_sha256=reconstruction_hashes(),
    )


def spectrum(arrays, info):
    """Propagate event-denominator fluctuations, not just per-jet Sumw2."""
    n = len(info["design_ids"])
    f = 3 * (len(validate_config(info["config"])) - 1)
    if (
        n == 0
        or len(set(info["design_ids"])) != n
        or len(info["cross_sections"]) != n
        or arrays["jet_sumw2"].shape != (n, f)
        or info.get("normalization") != "per_binary_hard_process"
        or info.get("covariance") != "joint_event_poisson_with_denominators"
    ):
        raise ValueError("Sample shape, normalization or covariance contract mismatch")
    width = f + 1
    total, raw = arrays["total"], arrays["outer"]
    if total.shape != (n * width,) or raw.shape != (n * width, n * width):
        raise ValueError("Inconsistent moment dimensions")
    if not np.isfinite(total).all() or not np.isfinite(raw).all():
        raise ValueError("Nonfinite moments")
    if not np.allclose(raw, raw.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Nonsymmetric raw moments")
    jac, means = np.zeros((n * f, n * width)), np.zeros((n, f))
    for i, sigma in enumerate(info["cross_sections"]):
        block = total[i * width : (i + 1) * width]
        den = block[-1]
        if den <= 0 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Invalid all-event normalization")
        means[i] = sigma * block[:-1] / den
        jac[i * f : (i + 1) * f, i * width : i * width + f] = np.eye(f) * sigma / den
        jac[i * f : (i + 1) * f, (i + 1) * width - 1] = -sigma * block[:-1] / den**2
    covariance = jac @ raw @ jac.T
    return means, covariance.reshape(n, f, n, f)


def ratio_bank(aa, caa, pp, cpp):
    """Retain measured AA cross-design blocks and the ONE shared pp reference."""
    aa, caa, pp, cpp = map(np.asarray, (aa, caa, pp, cpp))
    n, f = aa.shape
    if caa.shape != (n, f, n, f) or pp.shape != (f,) or cpp.shape != (f, f):
        raise ValueError("Mismatched full spectrum covariance")
    if not all(np.isfinite(x).all() for x in (aa, caa, pp, cpp)) or np.any(pp <= 0) or np.any(aa <= 0):
        raise ValueError("Empty/nonpositive or nonfinite bin: do not export a zero-error prediction")
    derivative = -aa / pp**2
    cov = caa / np.einsum("b,c->bc", pp, pp)[None, :, None, :]
    cov += np.einsum("ib,bc,jc->ibjc", derivative, cpp, derivative)
    return aa / pp, cov


def save(path, arrays, info):
    path = Path(path)
    if path.suffix != ".npz" or path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Use a new .npz output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def load(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    info = json.loads(path.with_suffix(".json").read_text())
    if info["version"] != VERSION:
        raise ValueError("Extraction version mismatch")
    return arrays, info


def export_tables(manifest_path, output):
    """Export only complete audited design banks; no placeholder predictions."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    manifest = yaml.safe_load(manifest_path.read_text())

    def resolve(path):
        return (manifest_path.parent / path).resolve()

    pp_arrays, pp_info = load(resolve(manifest["pp"]))
    config = pp_info["config"]
    edges = validate_config(config, production=True)
    if (
        pp_info["system"] != "pp"
        or pp_info["design_ids"] != [None]
        or pp_info["centrality"] is not None
        or not pp_info["input_provenance"]
        or not pp_info.get("source_sha256")
        or not pp_info.get("reconstruction_sha256")
        or not isinstance(pp_info.get("pairing_id"), str)
        or not pp_info["pairing_id"].strip()
    ):
        raise ValueError("Invalid common pp sample")
    pp_all, cpp_all = spectrum(pp_arrays, pp_info)
    pp, cpp = pp_all[0], cpp_all[0, :, 0, :]
    aa = caa = ids = None
    intervals, weights, used = [], [], set(pp_info.get("source_sha256", []))
    cohorts = {pp_info["pairing_id"]}
    for component in manifest["aa_components"]:
        arrays, info = load(resolve(component["sample"]))
        if (
            info["system"] != "PbPb"
            or info["config"] != config
            or not info["input_provenance"]
            or (ids is not None and ids != info["design_ids"])
            or info.get("reconstruction_sha256") != pp_info["reconstruction_sha256"]
            or not isinstance(info.get("pairing_id"), str)
            or not info["pairing_id"].strip()
        ):
            raise ValueError("AA and pp definitions or design ordering differ")
        sources = set(info.get("source_sha256", []))
        if not sources or sources.intersection(used):
            raise ValueError("Missing source provenance or reused pp/centrality inputs")
        if info["pairing_id"] in cohorts:
            raise ValueError("Shared pp/AA or cross-centrality cohort requires measured cross-component moments")
        cohorts.add(info["pairing_id"])
        used.update(sources)
        weight = float(component["weight"])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Invalid explicit per-binary centrality mixture weight")
        value, cov = spectrum(arrays, info)
        aa = weight * value if aa is None else aa + weight * value
        caa = weight**2 * cov if caa is None else caa + weight**2 * cov
        ids = info["design_ids"]
        intervals.append(info["centrality"])
        weights.append(weight)
    intervals.sort()
    if (
        not intervals
        or intervals[0][0] != 0
        or intervals[-1][1] != 10
        or any(a[1] != b[0] for a, b in zip(intervals[:-1], intervals[1:]))
        or not np.isclose(sum(weights), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Centrality components must cover [0,10] with explicit weights summing to one")
    means, covariance = ratio_bank(aa, caa, pp, cpp)
    design = resolve(manifest["design_table"])
    label = next(line for line in design.read_text().splitlines() if "Design point indices" in line)
    order = [int(x) for x in label.split(":", 1)[1].split()]
    design_values = np.loadtxt(design, ndmin=2)
    if (
        not design.name.startswith("Design__")
        or not np.isfinite(design_values).all()
        or len(set(order)) != len(order)
        or set(order) != set(ids)
        or len(design_values) != len(order)
    ):
        raise ValueError("Design IDs must cover the supplied unchanged design table exactly")
    permutation = [ids.index(i) for i in order]
    means = means[permutation]
    covariance = np.ascontiguousarray(covariance[permutation, :, :, :][:, :, permutation, :])
    flat = covariance.reshape(means.size, means.size)
    if not np.allclose(flat, flat.T, rtol=1e-10, atol=1e-14) or np.linalg.eigvalsh(flat).min() < -1e-10 * max(
        np.diag(flat).max(), 1e-30
    ):
        raise ValueError("Exported covariance is not symmetric positive semidefinite")
    # Only tolerated negative diagonal roundoff; never sqrt a negative variance.
    roundoff = np.flatnonzero(np.diag(flat) < 0)
    flat[roundoff, roundoff] = 0
    errors = np.sqrt(np.diag(flat)).reshape(means.shape)
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite propagated simulation errors")
    data_paths = [resolve(p) for p in manifest["data_tables"]]
    expected_edges = np.column_stack([edges[:-1], edges[1:]])
    if len(data_paths) != 3:
        raise ValueError("Require three exclusive pT tables, not an overlapping inclusive table")
    for i, path in enumerate(data_paths):
        expected = f"Data__5020__PbPb__inclusive_jet__rg_atlas__R0.4_pt{i}__0-10.dat"
        data = np.loadtxt(path, ndmin=2)
        if (
            path.name != expected
            or data.shape[0] != len(edges) - 1
            or not np.allclose(data[:, :2], expected_edges, rtol=0, atol=1e-12)
        ):
            raise ValueError("Experimental table identity/bin edges differ from verified extraction")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("Data", "Prediction", "Design", "QA"):
        (output / folder).mkdir()
    parameterization = design.stem.removeprefix("Design__")
    for i, data in enumerate(data_paths):
        stem = f"Prediction__{parameterization}__{data.stem.removeprefix('Data__')}"
        block = slice(i * (len(edges) - 1), (i + 1) * (len(edges) - 1))
        header = f"Version 2.0\nData {data.name}\nDesign {design.name}\n" + " ".join(f"design_point{x}" for x in order)
        for name, values in (
            ("values", means[:, block]),
            ("errors", errors[:, block]),
        ):
            np.savetxt(output / "Prediction" / f"{stem}__{name}.dat", values.T, header=header)
        shutil.copy2(data, output / "Data" / data.name)
    shutil.copy2(design, output / "Design" / design.name)
    names = [f"5020__PbPb__inclusive_jet__rg_atlas__R0.4_pt{i}__0-10" for i in range(3) for _ in edges[:-1]]
    np.savez_compressed(
        output / "QA/rg_simulation_covariance.npz",
        covariance=covariance,
        central_values=means,
        design_ids=order,
        observable_names=names,
        xmin=np.tile(edges[:-1], 3),
        xmax=np.tile(edges[1:], 3),
    )
    (output / "QA/manifest.json").write_text(
        json.dumps(
            dict(
                version=VERSION,
                manifest=manifest,
                config=config,
                roundoff_diagonal_clips=roundoff.tolist(),
                limitation="rg-only covariance is NOT independent of existing observables. Append count_event vectors to the common paired extractor before a joint inference; do not invent block-diagonal cross-observable noise.",
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--inputs", nargs="+", type=Path, required=True)
    analysis.add_argument("--design-ids", nargs="+", type=int)
    analysis.add_argument("--config", type=Path, required=True)
    analysis.add_argument("--system", choices=("pp", "PbPb"), required=True)
    analysis.add_argument("--centrality", nargs=2, type=int)
    analysis.add_argument("--pairing-id")
    analysis.add_argument("--input-provenance", required=True)
    analysis.add_argument("--per-binary-hard-process", action="store_true", required=True)
    analysis.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_tables(args.manifest, args.output)
        return
    config = yaml.safe_load(args.config.read_text())
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Refusing to overwrite an existing extraction")
    ids = args.design_ids if args.system == "PbPb" else [None]
    hashes = []
    for path in args.inputs:
        with path.open("rb") as stream:
            hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    if len(set(hashes)) != len(hashes):
        parser.error("Repeated particle file")
    arrays, info = analyze(
        [read_events(p) for p in args.inputs],
        ids or [],
        config,
        args.system,
        args.centrality,
        args.pairing_id,
        args.input_provenance,
    )
    info["source_sha256"] = hashes
    info["sources"] = [str(p.resolve()) for p in args.inputs]
    save(args.output, arrays, info)
    print(json.dumps(info["cutflow"], indent=2))


if __name__ == "__main__":
    main()
