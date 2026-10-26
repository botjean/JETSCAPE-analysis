"""Verify imported source bytes without importing any analysis or controller."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "studies/hybrid_lres/SOURCE_MANIFEST.json").read_text())
    seen = set()
    for record in manifest["files"]:
        name = record["path"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or name in seen:
            raise ValueError(f"Unsafe or duplicate source path: {name}")
        seen.add(name)
        target = root / name
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes review checkout: {name}")
        data = target.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Imported source differs: {name}")
        if record["transformation"] != "byte-identical":
            raise ValueError(f"Unreviewed transformation: {name}")
    for name, expected in manifest["preserved_base_files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Committed dR12 baseline changed: {name}")
    return len(seen)


def verify_heppy(heppy_root, root=ROOT):
    """Read a user-supplied checkout; never fetch, copy or install external code."""
    manifest = json.loads((Path(root) / "studies/hybrid_lres/SOURCE_MANIFEST.json").read_text())
    dependency, = manifest["external_dependencies"]
    checkout = Path(heppy_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE).strip()
        top = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("--heppy-root must be a local Git checkout root") from error
    if Path(top).resolve() != checkout or commit != dependency["commit"]:
        raise ValueError("External heppy checkout is not at the pinned root/commit")
    result = {}
    for record in dependency["files"]:
        path = PurePosixPath(record["path"])
        if path.is_absolute() or ".." in path.parts or record["path"] in result:
            raise ValueError("Unsafe or duplicate external heppy path")
        target = checkout / path
        if target.is_symlink() or not target.resolve().is_relative_to(checkout):
            raise ValueError("External heppy source escapes checkout")
        data = target.read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"External heppy source differs: {path}")
        result[record["path"]] = target
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heppy-root", type=Path, help="Optionally verify a supplied pinned heppy checkout")
    args = parser.parse_args()
    print(f"Verified {verify()} packaged sources and preserved dR12 baseline")
    if args.heppy_root is not None:
        print(f"Verified {len(verify_heppy(args.heppy_root))} external heppy sources (not packaged)")
