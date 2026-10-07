"""ATLAS charged kt-splitting analysis; isolated from the existing STAT driver.

Run ``python -m jetscape_analysis.analysis.atlas_dr12 --help``.
Input normalization follows the STAT convention: sigma * sum(w*n) / sum(w).
PbPb input must represent hard processes per binary collision, NOT minimum bias.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from particle import Particle
import yaml

from jetscape_analysis.analysis import _atlas_dr12

VERSION = "atlas-dr12-v2"


@lru_cache(maxsize=None)
def particle_properties(pid):
    particle = Particle.from_pdgid(pid)
    lifetime = particle.lifetime  # particle uses ns: 30 ps = 0.03 ns.
    if lifetime is None or lifetime <= 0.03:
        raise ValueError(f"PDG {pid} is not verified stable (>30 ps). Fix generator decays, not particle cuts.")
    if particle.charge is None:
        raise ValueError(f"Unknown charge for PDG {pid}")
    return abs(pid) not in (12, 13, 14, 16), particle.charge != 0


def particle_rows(event, system):
    """Convert the branch's Hybrid status convention without losing wake signs."""
    keys = ("px", "py", "pz", "E", "particle_ID", "status")
    arrays = [event[key] for key in keys]
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("Inconsistent particle array lengths")
    rows = []
    for px, py, pz, energy, pid, status in zip(*arrays, strict=True):
        if not np.isfinite(pid) or int(pid) != pid:
            raise ValueError(f"Invalid PDG ID {pid}")
        if status == -2:  # outgoing hard partons, not final-state particles
            continue
        if status not in (0, 1, 2):
            raise ValueError(f"Unexpected Hybrid particle status {status}")
        if system == "pp" and status != 0:
            continue  # matches dev-hybrid-model: no wake in the pp reference
        calo, track = particle_properties(int(pid))
        rows.append([px, py, pz, energy, -1 if status == 2 else 1, int(calo), int(track)])
    return rows


def read_events(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        import awkward as ak
        from jetscape_analysis.analysis.reader import parse_ascii

        with path.open() as stream:
            if parse_ascii.determine_model_from_file(stream) != "hybrid":
                raise ValueError("This runner currently accepts Hybrid events only")
        for chunk in parse_ascii.read(path, events_per_chunk=1024):
            yield from ak.to_list(chunk)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    if config["sqrt_s"] != 5020:
        raise ValueError("This measurement is at 5020 GeV")
    low, high = config["jet_pt"]
    if not 200 <= low < high:
        raise ValueError("Require jet pT interval above 200 GeV")
    low, high = config["centrality"]
    if not 0 <= low < high <= 100:
        raise ValueError("Invalid centrality interval")
    if not config["observables"]:
        raise ValueError("No observables configured")
    columns = {"dr12_atlas": 1, "sqrt_d12_atlas": 2}
    for name, block in config["observables"].items():
        edges = np.asarray(block["edges"], float)
        if len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError(f"Invalid bin edges for {name}")
        if name not in columns or block["column"] != columns[name]:
            raise ValueError("Observable name/column mismatch: dr12_atlas=1, sqrt_d12_atlas=2")


def analyze_events(events, config, system, centrality, stratum, cross_section=None, design_id=None):
    validate_config(config)
    if system not in ("pp", "PbPb"):
        raise ValueError("Expected pp or PbPb")
    if system == "PbPb" and list(centrality or []) != config["centrality"]:
        raise ValueError("Declare the actual PbPb sample centrality, matching the config; no automatic fallback")
    if system == "PbPb" and (type(design_id) is not int or design_id < 0):
        raise ValueError("PbPb samples require a nonnegative integer design_id")
    if system == "pp" and (design_id is not None or centrality is not None):
        raise ValueError("The common pp reference must not have a design ID or centrality")
    if not isinstance(stratum, str) or not stratum.strip():
        raise ValueError("A nonempty generator stratum label is required")
    cuts, weighted_cuts = Counter(), Counter()
    output = {}
    for name, block in config["observables"].items():
        n = len(block["edges"]) - 1
        output[f"{name}_sumw"] = np.zeros(n)
        output[f"{name}_event_outer"] = np.zeros((n, n))
    weight_sum = weight_sum2 = 0.0
    sigma = cross_section
    low, high = config["jet_pt"]
    for event in events:
        weight = float(event["event_weight"])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Expected nonnegative finite Hybrid event weights")
        if cross_section is None:
            current = float(event["cross_section"])
            if sigma is None:
                sigma = current
            if not np.isclose(current, sigma, rtol=1e-6, atol=0):
                raise ValueError("Cross section changes inside one sample")
        if system == "PbPb" and event.get("centrality", -1) >= 0:
            if not centrality[0] <= event["centrality"] < centrality[1]:
                raise ValueError("Event centrality contradicts declared sample centrality")
        weight_sum += weight  # includes events with no selected jets
        weight_sum2 += weight**2
        cuts["events"] += 1
        weighted_cuts["events"] += weight
        result = _atlas_dr12.reconstruct(particle_rows(event, system))
        jets = [j for j in result["jets"] if low <= j[0] < high]
        result["cutflow"]["jets_in_pt_interval"] = len(jets)
        cuts.update(result["cutflow"])
        weighted_cuts.update({k: weight * v for k, v in result["cutflow"].items()})
        for name, block in config["observables"].items():
            values = np.asarray([j[block["column"]] for j in jets])
            edges = np.asarray(block["edges"])
            # Match ROOT's [low, high) convention, including the upper overflow.
            counts = np.histogram(values[(values >= edges[0]) & (values < edges[-1])], edges)[0]
            contribution = weight * counts
            output[f"{name}_sumw"] += contribution
            output[f"{name}_event_outer"] += np.outer(contribution, contribution)
            cuts[f"{name}_underflow"] += int(np.sum(values < edges[0]))
            cuts[f"{name}_overflow"] += int(np.sum(values >= edges[-1]))
    if not weight_sum > 0 or sigma is None or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Empty/zero-weight sample or invalid cross section")
    metadata = dict(
        version=VERSION,
        config=config,
        system=system,
        centrality=centrality,
        design_id=design_id,
        stratum=stratum,
        cross_section=float(sigma),
        weight_sum=weight_sum,
        weight_sum2=weight_sum2,
        cutflow=dict(cuts),
        weighted_cutflow=dict(weighted_cuts),
        normalization="per_binary_hard_process",
        mc_error="event-cluster Poisson, fixed sigma/sumw",
    )
    return output, metadata


def save_sample(path, arrays, metadata):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Output must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata, sort_keys=True))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_sample(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files if k != "metadata"}, json.loads(str(data["metadata"]))


def merge_samples(paths, output):
    """Merge independent chunks of one stratum before normalizing, never sum their cross sections."""
    if len(set(map(str, paths))) != len(paths):
        raise ValueError("Duplicate input sample")
    arrays, metadata = load_sample(paths[0])
    sources = set(metadata["sources"])
    for path in paths[1:]:
        other, info = load_sample(path)
        for key in (
            "version",
            "config",
            "system",
            "centrality",
            "design_id",
            "stratum",
            "cross_section",
            "normalization",
        ):
            if info[key] != metadata[key]:
                raise ValueError(f"Cannot merge mismatched {key}")
        if sources.intersection(info["sources"]):
            raise ValueError("Repeated input events detected by source hash")
        sources.update(info["sources"])
        for key in arrays:
            arrays[key] += other[key]
        for key in ("weight_sum", "weight_sum2"):
            metadata[key] += info[key]
        for key in ("cutflow", "weighted_cutflow"):
            metadata[key] = dict(Counter(metadata[key]) + Counter(info[key]))
    metadata["sources"] = sorted(sources)
    save_sample(output, arrays, metadata)


def spectrum(paths, observable, expected_system, design_id=None):
    value = covariance = config = None
    strata, sources = set(), set()
    for path in paths:
        arrays, info = load_sample(path)
        if info["version"] != VERSION or info["system"] != expected_system:
            raise ValueError("Sample version/system mismatch")
        if info["design_id"] != design_id:
            raise ValueError("Sample design_id differs from the requested design point")
        if info["normalization"] != "per_binary_hard_process":
            raise ValueError("Unsupported normalization")
        if info["stratum"] in strata or sources.intersection(info["sources"]):
            raise ValueError("Repeated stratum/source: merge independent chunks before constructing a spectrum")
        strata.add(info["stratum"])
        sources.update(info["sources"])
        if config is not None and config != info["config"]:
            raise ValueError("Mismatched reconstruction/binning configs")
        config = info["config"]
        factor = info["cross_section"] / info["weight_sum"]
        y = factor * arrays[f"{observable}_sumw"]
        cov = factor**2 * arrays[f"{observable}_event_outer"]
        value = y if value is None else value + y
        covariance = cov if covariance is None else covariance + cov
    if config is None:
        raise ValueError("Empty sample list")
    return value, covariance, config, sources


def ratio(aa, caa, pp, cpp):
    if not all(np.all(np.isfinite(x)) for x in (aa, caa, pp, cpp)):
        raise ValueError("Nonfinite MC spectrum/covariance")
    if np.any(pp <= 0) or np.any(aa <= 0):
        raise ValueError("Empty/nonpositive MC bin: generate more events; do not export zero-error predictions")
    raa = aa / pp
    # Independent pp/AA samples, including shared-event bin covariance inside each sample.
    cov = caa / np.outer(pp, pp) + cpp * np.outer(aa / pp**2, aa / pp**2)
    return raa, cov


def export_tables(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    base = manifest_path.parent
    resolve = lambda p: (base / p).resolve()
    observable = manifest["observable"]
    pp_paths = [resolve(p) for p in manifest["pp"]]
    pp, cpp, config, pp_sources = spectrum(pp_paths, observable, "pp")
    values, errors, covariances, ids = [], [], {}, []
    for design_id, paths in sorted(manifest["design_points"].items(), key=lambda kv: int(kv[0])):
        if not str(design_id).isdigit():
            raise ValueError("Design IDs must be nonnegative integers")
        aa, caa, aa_config, sources = spectrum([resolve(p) for p in paths], observable, "PbPb", int(design_id))
        if aa_config != config:
            raise ValueError("pp and PbPb analysis definitions differ")
        if sources.intersection(pp_sources):
            raise ValueError("pp and PbPb share events; the independent-sample error formula is not valid")
        y, cov = ratio(aa, caa, pp, cpp)
        values.append(y)
        errors.append(np.sqrt(np.diag(cov)))
        covariances[f"design_point{int(design_id)}"] = cov
        ids.append(int(design_id))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Need unique design point IDs")
    data_path = resolve(manifest["data_table"])
    data = np.loadtxt(data_path, ndmin=2)
    edges = np.asarray(config["observables"][observable]["edges"])
    if data.shape[0] != len(edges) - 1 or not np.allclose(
        data[:, :2], np.column_stack([edges[:-1], edges[1:]]), atol=1e-12, rtol=0
    ):
        raise ValueError("Data bin edges differ from generated prediction bins")
    lo, hi = config["centrality"]
    ptlo, pthi = config["jet_pt"]
    expected = f"Data__5020__PbPb__inclusive_jet__{observable}__R1.0_pt{ptlo}-{pthi}_trk4_zcut0.15_beta0__{lo}-{hi}.dat"
    if data_path.name != expected:
        raise ValueError(f"Data name does not match selection; expected {expected}")
    design_path = resolve(manifest["design_table"])
    if not design_path.name.startswith("Design__"):
        raise ValueError("Use the existing Design__PARAMETERIZATION.dat file unchanged")
    header = next(line for line in design_path.read_text().splitlines() if "Design point indices" in line)
    design_ids = [int(x) for x in header.split(":", 1)[1].split()]
    design = np.loadtxt(design_path, ndmin=2)
    if design.shape[0] != len(design_ids) or not np.all(np.isfinite(design)):
        raise ValueError("Design table rows do not match declared IDs or contain nonfinite values")
    if ids != sorted(design_ids):
        raise ValueError("Prediction IDs must cover the supplied design table exactly")
    parameterization = design_path.stem.removeprefix("Design__")
    stem = f"Prediction__{parameterization}__{data_path.stem.removeprefix('Data__')}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("Prediction", "Data", "Design", "QA"):
        (output / folder).mkdir()
    header = f"Version 2.0\nData {data_path.name}\nDesign {design_path.name}\n" + " ".join(
        f"design_point{i}" for i in ids
    )
    for kind, array in (("values", values), ("errors", errors)):
        np.savetxt(output / "Prediction" / f"{stem}__{kind}.dat", np.asarray(array).T, header=header)
    shutil.copy2(data_path, output / "Data" / data_path.name)
    shutil.copy2(design_path, output / "Design" / design_path.name)
    np.savez_compressed(output / "QA" / "prediction_mc_covariance.npz", **covariances)
    (output / "QA" / "manifest_resolved.json").write_text(
        json.dumps(
            dict(
                manifest=manifest,
                manifest_base=str(base),
                config=config,
                version=VERSION,
                note="MC covariance sidecar is not automatically consumed by Bayesian; pp sample shared across design columns.",
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Analyze one Hybrid ASCII/parquet sample")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--config", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--system", choices=("pp", "PbPb"), required=True)
    analyze.add_argument("--centrality", nargs=2, type=int)
    analyze.add_argument("--design-id", type=int, help="Required for PbPb; recorded and checked during merge/export")
    analyze.add_argument(
        "--stratum", required=True, help="Generator pThat stratum; same label for chunks of one stratum"
    )
    analyze.add_argument(
        "--cross-section", type=float, help="Explicit same-unit generator cross section, otherwise use header"
    )
    analyze.add_argument(
        "--confirm-particle-level-input",
        action="store_true",
        required=True,
        help="Confirm decays/stable-particle convention and no detector simulation or underlying-event embedding",
    )
    analyze.add_argument(
        "--per-binary-hard-process",
        action="store_true",
        required=True,
        help="Confirm sigma/sumw normalization; PbPb events are per binary hard process, not minimum bias",
    )
    merge = sub.add_parser("merge", help="Pool independent chunks of the SAME generation stratum")
    merge.add_argument("--inputs", nargs="+", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export", help="Write Bayesian prediction values/errors using a design manifest")
    export.add_argument("--manifest", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "analyze":
        config = load_config(args.config)
        arrays, metadata = analyze_events(
            read_events(args.input),
            config,
            args.system,
            args.centrality,
            args.stratum,
            args.cross_section,
            args.design_id,
        )
        with args.input.open("rb") as stream:
            metadata["sources"] = [hashlib.file_digest(stream, "sha256").hexdigest()]
        metadata["input"] = str(args.input.resolve())
        save_sample(args.output, arrays, metadata)
        print(json.dumps(metadata["cutflow"], indent=2))
    elif args.command == "merge":
        merge_samples(args.inputs, args.output)
    else:
        export_tables(args.manifest, args.output)


if __name__ == "__main__":
    main()
