"""Build the reviewed additional backend only, inside the compute gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig

import fastjet
import pybind11


def main():
    root = Path(__file__).resolve().parent
    native = root / "native"
    target = root / ("_rg_precision_candidate" + sysconfig.get_config_var("EXT_SUFFIX"))
    if target.exists():
        raise ValueError("Refuse to replace an existing precision backend")
    fj = Path(fastjet.__file__).resolve().parent
    lib = next(p for p in (fj / "lib64", fj / "lib") if (p / "libfastjet.so").exists())
    sources = [native / n for n in ("generated_native.cc", "GroomerShop.cc", "LundGenerator.cc")]
    command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-fvisibility=hidden",
               "-Wl,-Bsymbolic", "-Wall", "-Wextra"]
    command += ["-I" + str(p) for p in (pybind11.get_include(), sysconfig.get_path("include"), fj / "include", native)]
    command += [str(p) for p in sources]
    command += ["-L" + str(lib), "-Wl,-rpath," + str(lib), "-lfastjettools", "-lfastjet", "-o", str(target)]
    subprocess.run(command, check=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(command=command, fastjet_version=fastjet.__version__,
        source_sha256={str(p): sha(p) for p in [Path(__file__), target, *sources]},
        runtime_dependencies={str((lib / n).resolve()): sha(lib / n) for n in ("libfastjet.so", "libfastjettools.so")},
        policy="unchanged strict result; precision boundary only after explicit retention failure")
    manifest = root / "precision_build.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    with (root / "NATIVE_SHA256SUMS").open("a") as stream:
        for p in (target, manifest):
            stream.write(sha(p) + "  " + str(p.relative_to(root)) + "\n")
    print("PRECISION_BACKEND_BUILT", sha(target), flush=True)


if __name__ == "__main__":
    main()
