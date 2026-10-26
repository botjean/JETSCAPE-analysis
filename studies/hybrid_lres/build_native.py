"""Explicit local source builds, not the historical cluster build procedure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sysconfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EX = HERE / "executed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("canonical", "upstream", "precision", "hardest", "wake"))
    parser.add_argument("--heppy-root", type=Path, help="User-supplied pinned checkout; required for upstream/precision/wake")
    args = parser.parse_args()
    if os.environ.get("PAPER_NATIVE_BUILD") != "1":
        parser.error("Explicit opt-in required: PAPER_NATIVE_BUILD=1")
    from verify_sources import verify, verify_heppy

    verify()
    needs_heppy = args.target in ("upstream", "precision", "wake")
    if needs_heppy and args.heppy_root is None:
        parser.error("--heppy-root is required for upstream, precision and wake targets")
    external = verify_heppy(args.heppy_root) if needs_heppy else {}
    vendor = HERE / "native/vendor"
    if needs_heppy:
        # Quoted includes search the external header's directory before -I paths.
        shadow = external["cpptools/src/fjcontrib/custom/GroomerShop.hh"].with_name("LundGenerator.hh")
        if shadow.exists() and sha(shadow) != sha(vendor / "LundGenerator.hh"):
            parser.error("External header would shadow the pinned GPL LundGenerator.hh")
    import fastjet
    import pybind11
    fj = Path(fastjet.__file__).resolve().parent
    lib = next((fj / p for p in ("lib64", "lib") if (fj / p / "libfastjet.so").is_file()), None)
    if lib is None:
        parser.error("Expected a Linux FastJet wheel with libfastjet.so")
    analysis = ROOT / "jetscape_analysis/analysis"
    build = ROOT / "build/hybrid_lres"
    nominal = EX / "atlas_rg_inference_20261015/recovery_20261016"
    specifications = {
        "canonical": [("_atlas_dr12", analysis / "atlas_dr12.cc", analysis, False),
                      ("_atlas_rg", analysis / "atlas_rg.cc", analysis, False)],
        "upstream": [("_rg_upstream", EX / "atlas_rg_implementation_20261015/upstream_native/rg_upstream.cc", analysis, True)],
        "precision": [("_rg_precision_candidate", nominal / "precision_candidate_local/generated_native.cc", build, True)],
        "hardest": [("_wake_dr12_diagnostic", EX / "wake_bounded_transport_20261017/endpoint_bank/upstream_hardest_recovery/dr12_policy.cc", build, False)],
        "wake": [("_wake_native", EX / "wake_sensitivity_20261017/native.cc", build, True)],
    }
    extension = sysconfig.get_config_var("EXT_SUFFIX")
    libraries = {str((lib / n).resolve()): sha(lib / n) for n in ("libfastjet.so", "libfastjettools.so")}
    specs = specifications[args.target]
    outputs = [directory / (name + extension) for name, _, directory, _ in specs]
    if any(p.exists() for p in outputs):
        parser.error("Refuse to replace an existing local extension")
    build.mkdir(parents=True, exist_ok=True)
    for (name, source, directory, isolated), target in zip(specs, outputs):
        sources = [source]
        command = shlex.split(os.environ.get("CXX", "c++")) + ["-O2", "-std=c++17", "-shared", "-fPIC"]
        if isolated:
            command += ["-fvisibility=hidden", "-Wl,-Bsymbolic"]
            sources += [external["cpptools/src/fjcontrib/custom/GroomerShop.cc"], vendor / "LundGenerator.cc"]
        external_includes = [external[path].parent for path in (
            "cpptools/src/fjcontrib/custom/GroomerShop.hh", "cpptools/src/fjext/fjtools.hh")] if isolated else []
        command += ["-I" + str(p) for p in (
            pybind11.get_include(), sysconfig.get_path("include"), fj / "include", vendor, *external_includes)]
        command += [str(p) for p in sources] + ["-L" + str(lib), "-Wl,-rpath," + str(lib)]
        if not isolated:
            command += ["-lfastjetcontribfragile"]
        command += ["-lfastjettools", "-lfastjet", "-o", str(target)]
        subprocess.run(command, check=True)
        pins = {str(p.relative_to(ROOT)): sha(p) for p in [source, *vendor.iterdir()] if p.is_file()}
        receipt = dict(kind="NEW_LOCAL_BUILD_NOT_EXECUTED_CLUSTER_BINARY", command=command,
                       source_sha256=pins, binary_sha256=sha(target), runtime_dependencies=libraries,
                       external_source_sha256={name: sha(path) for name, path in external.items()},
                       heppy_commit="16fcefbdca7a106e6e74fd53e904e302335dcce3" if needs_heppy else None)
        (build / (name + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
        if name == "_rg_upstream":
            (analysis / "_rg_upstream_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
