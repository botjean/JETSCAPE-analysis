"""Run bounded local review tests in isolated processes; never run collectors."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXECUTED = HERE / "executed"
WAKE = EXECUTED / "wake_bounded_transport_20261017"
NOMINAL = EXECUTED / "atlas_rg_inference_20261015/recovery_20261016"
SEED = WAKE / "endpoint_bank/physical_prior_restriction/seed_scaling_recovery"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="Also require the optional native extensions")
    args = parser.parse_args()
    groups = {
        "source_and_covariance": [HERE / "tests/test_source_package.py", ROOT / "tests/test_simulation_covariance.py"],
        "bounded": [WAKE / (name + ".py") for name in (
            "test_transport", "test_bounded_projection", "test_thinning", "test_certificate",
            "test_float32_certificate", "test_symmetries", "test_polygon_repro")],
        "seed": [SEED / "test_seed_scaling.py", SEED / "test_collector_patch.py"],
        "strict_first": [NOMINAL / "precision_continuation/test_strict_first.py"],
    }
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment["PYTHONPATH"] = os.pathsep.join([
        str(ROOT), str(ROOT / "build/hybrid_lres"), environment.get("PYTHONPATH", "")])
    if args.native:
        required = ["jetscape_analysis.analysis._atlas_dr12", "jetscape_analysis.analysis._atlas_rg",
                    "jetscape_analysis.analysis._rg_upstream", "_rg_precision_candidate", "_wake_dr12_diagnostic"]
        probe = "import importlib.util; " + "; ".join(
            f"assert importlib.util.find_spec({name!r}) is not None, 'Missing native module: {name}'"
            for name in required)
        subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=environment, check=True)
        groups["canonical_native"] = [ROOT / "tests/test_atlas_rg.py", ROOT / "tests/test_atlas_rg_upstream.py"]
        groups["precision_native"] = [NOMINAL / "precision_candidate_local/test_precision.py"]
        groups["hardest_native"] = [HERE / "tests/test_hardest_local.py"]
    output = HERE / "validation"
    output.mkdir(exist_ok=True)
    results = []
    for name, paths in groups.items():
        command = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                   *[str(p.relative_to(ROOT)) for p in paths]]
        started = time.monotonic()
        try:
            run = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            code, log = run.returncode, run.stdout
        except subprocess.TimeoutExpired as error:
            code, log = 124, f"Group exceeded the 300-second limit: {error}"
        (output / f"{name}.txt").write_text(log)
        results.append(dict(group=name, returncode=code, seconds=round(time.monotonic() - started, 3),
                            command=command, output=log))
        print(f"\n[{name}]\n{log}", flush=True)
    versions = {}
    for name in ("numpy", "scipy", "pytest", "PyYAML", "fastjet", "pybind11"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    receipt = dict(kind="NEW_LOCAL_REVIEW_TESTS_NOT_EVENT_REPLAY", python=sys.version,
                   dependencies=versions, native_requested=args.native, groups=results)
    (output / "checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return int(any(item["returncode"] != 0 for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
