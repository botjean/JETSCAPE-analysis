"""Resume committed moments with explicit strict-success compatibility proofs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_prefix(expected, original_backend, effective_backend, policy_sha256):
    """No relabelling: retain original execution identity and both old hashes."""
    directory = Path(expected["directory"])
    path = directory / "CHECKPOINT.json"
    if sha(path) != expected["metadata_sha256"]:
        raise ValueError("Committed metadata changed")
    meta = json.loads(path.read_text())
    if (meta["checkpoint_sha256"] != expected["checkpoint_sha256"]
        or sha(path.with_name("checkpoint.npz")) != expected["checkpoint_sha256"]
        or meta["reconstruction_sha256"] != original_backend
        or meta["completed_chunks"] != expected["chunks"]
        or meta["completed_chunks"] != meta["batch"]["chunks"][:len(expected["chunks"])]):
        raise ValueError("Committed prefix/backend changed")
    with np.load(path.with_name("checkpoint.npz"), allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    proof = dict(status="STRICT_SUCCESS_IDENTITY_PRESERVED", source_directory=str(directory),
        source_metadata_sha256=expected["metadata_sha256"],
        source_checkpoint_sha256=expected["checkpoint_sha256"],
        original_execution_backend=original_backend, effective_backend=effective_backend,
        strict_first_policy_sha256=policy_sha256, chunks=expected["chunks"],
        original_slurm_job_id=meta["slurm_job_id"],
        basis="Exact original native call and count_rows return on success; fallback is unreachable for these committed events")
    return arrays, deepcopy(meta["reports"]), proof


def publish(output, arrays, reports, batch, plan, payload, started, reuse, wrapper):
    from jetscape_analysis.analysis import atlas_rg as rg
    n = 1 if batch["tag"] == "pp" else 40
    shapes = dict(old_total=(n*136,), new_total=(n*40,), old_new=(n*136, n*40), new_new=(n*40, n*40))
    if n == 1:
        shapes["old_old"] = (136, 136)
    if set(arrays) != set(shapes) or any(v.shape != shapes[k] or not np.isfinite(v).all() for k, v in arrays.items()):
        raise ValueError("Invalid moment layout/values")
    used = sum(p.stat().st_size for p in Path(plan["recovery"]["legacy_root"]).rglob("*") if p.is_file())
    if used + sum(v.nbytes for v in arrays.values()) + 2**20 > plan["storage_cap_bytes"]:
        raise ValueError("Cumulative storage cap reached")
    chunks = [r["chunk_index"] for r in reports]
    if chunks != batch["chunks"][:len(chunks)]:
        raise ValueError("Not a unique committed prefix")
    pending = output / "checkpoint.pending.npz"
    np.savez_compressed(pending, **arrays)
    pending.replace(output / "checkpoint.npz")
    meta = dict(status="COMPLETE" if chunks == batch["chunks"] else "PARTIAL", batch=batch,
        completed_chunks=chunks, reports=reports, seconds=time.monotonic()-started,
        maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        collector_sha256=sha(__file__), manifest_sha256=sha(payload / "MANIFEST.json"),
        checkpoint_sha256=sha(output / "checkpoint.npz"), new_events_generated=False,
        rg_edges=plan["recovery"]["rg_config"]["rg_edges"], pi0_treatment="keep",
        reconstruction_sha256=rg.reconstruction_hashes(plan["recovery"]["rg_config"]),
        reuse_proof=reuse, finite_moment_shapes_verified=True,
        strict_first_audit=dict(wrapper.audit), fallback_event_audit=wrapper.events,
        slurm_job_id=os.environ["SLURM_JOB_ID"])
    pending = output / "CHECKPOINT.pending.json"
    pending.write_text(json.dumps(meta, indent=2) + "\n")
    pending.replace(output / "CHECKPOINT.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    payload = Path(__file__).resolve().parent
    root = payload.parent
    plan = json.loads((payload / "MANIFEST.json").read_text())
    extension = plan["precision_continuation"]
    batch = next(b for b in [extension["gate"], *plan["batches"]] if b["id"] == args.batch)
    output = root / "batches" / args.batch
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    from jetscape_analysis.analysis import atlas_dr12 as dr, atlas_rg as rg
    import _rg_precision_candidate as precision
    from strict_first import install
    cfg = plan["recovery"]["rg_config"]
    if rg.reconstruction_hashes(cfg) != extension["old_backend"]:
        raise ValueError("Strict executed backend differs from the pinned predecessor")
    build = json.loads((payload / "precision_build.json").read_text())
    if build["runtime_dependencies"] != rg._upstream_runtime_libraries():
        raise ValueError("Precision backend loaded different FastJet libraries")
    wrapper = install(rg, precision, payload / "strict_first.py")
    effective = rg.reconstruction_hashes(cfg)
    legacy = load("original_collector", payload / "legacy_collect_reduced.py")
    legacy.BASELINE_CONFIG = deepcopy(legacy.RG_CONFIG)
    legacy.RG_CONFIG = cfg
    original = load("original_rg", payload / "legacy_rg.py")
    previous = load("previous_collector", payload / "collect_reduced.py")

    def prefix(expected):
        if expected["stable"]:
            return stable_prefix(expected, extension["old_backend"], effective, sha(payload / "strict_first.py"))
        return previous.verify_prefix(Path(expected["directory"]), expected,
            original, rg, dr, legacy, cfg)

    arrays, reports, proof = None, [], None
    if args.batch == "pp_00":
        pp_arrays, pp_reports, pp_proof = prefix(extension["prefixes"]["pp_reused"])
        pp_output = root / "batches/pp_reused"
        pp_output.mkdir(exist_ok=False)
        publish(pp_output, pp_arrays, pp_reports, dict(id="pp_reused", tag="pp", chunks=[0], gate=True),
                plan, payload, started, pp_proof, wrapper)
    if args.batch in extension["prefixes"]:
        arrays, reports, proof = prefix(extension["prefixes"][args.batch])
        publish(output, arrays, reports, batch, plan, payload, started, proof, wrapper)
    if args.batch == "aa510_00":
        gate_dir = root / "batches/precision_gate"
        gate = json.loads((gate_dir / "CHECKPOINT.json").read_text())
        if (gate["status"] != "COMPLETE" or gate["completed_chunks"] != [35]
            or gate["reconstruction_sha256"] != effective
            or sha(gate_dir / "checkpoint.npz") != gate["checkpoint_sha256"]
            or [r["chunk_index"] for r in reports] != [12, 23]):
            raise ValueError("Precision gate cannot be pooled exactly once")
        with np.load(gate_dir / "checkpoint.npz", allow_pickle=False) as data:
            for key in arrays:
                arrays[key] += data[key]
        reports.extend(deepcopy(gate["reports"]))
        proof["precision_gate_metadata_sha256"] = sha(gate_dir / "CHECKPOINT.json")
        publish(output, arrays, reports, batch, plan, payload, started, proof, wrapper)
    producer = load("unchanged_producer", legacy.BASE / "simulation-covariance-20261007/simulation_covariance.py")
    for path, checksum in json.loads((payload / "external_pins.json").read_text()).items():
        if sha(path) != checksum:
            raise ValueError("External source/input changed: " + path)
    config = dr.load_config(legacy.BASE / "atlas-dr12-pilot-20261007/config/atlas_dr12_5020.yaml")
    layout = json.loads((payload / "layout_hpt10.json").read_text())
    for chunk in batch["chunks"][len(reports):]:
        before = dict(wrapper.audit)
        first_event = len(wrapper.events)
        stats, report = legacy.extract_chunk(producer, batch["tag"], chunk, layout, config, batch["gate"])
        report["strict_first_audit"] = {k: v-before.get(k, 0) for k, v in wrapper.audit.items()}
        report["fallback_events"] = deepcopy(wrapper.events[first_event:])
        if args.batch == "precision_gate" and wrapper.audit["recovered_events"] <= 0:
            raise ValueError("Known failing gate did not exercise the repair")
        if arrays is None:
            arrays = stats
        else:
            for key in arrays:
                arrays[key] += stats[key]
        reports.append(report)
        publish(output, arrays, reports, batch, plan, payload, started, proof, wrapper)
        print("PRECISION_CHUNK_COMPLETE", args.batch, chunk, flush=True)
    print("PRECISION_BATCH_COMPLETE", args.batch, flush=True)


if __name__ == "__main__":
    main()
