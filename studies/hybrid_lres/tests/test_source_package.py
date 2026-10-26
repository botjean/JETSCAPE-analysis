"""New source-package guards, not physics or event-replay certification."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("paper_source_verifier", HERE / "verify_sources.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_all_imports_and_committed_dr12_are_byte_exact():
    assert verifier.verify() == 82


def test_heppy_is_external_only_and_gpl_lund_notices_remain():
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    dependency, = manifest["external_dependencies"]
    assert dependency["commit"] == "16fcefbdca7a106e6e74fd53e904e302335dcce3"
    assert {Path(record["path"]).name for record in dependency["files"]} == {
        "GroomerShop.cc", "GroomerShop.hh", "fjtools.cxx", "fjtools.hh"}
    packaged = {record["path"] for record in manifest["files"]}
    for record in dependency["files"]:
        assert record["former_package_path"] not in packaged
        assert not (ROOT / record["former_package_path"]).exists()
    for name in ("LundGenerator.cc", "LundGenerator.hh"):
        assert "GNU General Public License" in (HERE / "native/vendor" / name).read_text()
    assert (HERE / "native/vendor/COPYING.LundPlane").is_file()


@pytest.fixture
def synthetic_external_checkout(tmp_path, monkeypatch):
    root = tmp_path / "package"
    checkout = tmp_path / "external"
    checkout.mkdir()
    records = []
    for name in ("GroomerShop.cc", "GroomerShop.hh", "fjtools.cxx", "fjtools.hh"):
        data = ("synthetic test data: " + name).encode()
        (checkout / name).write_bytes(data)
        records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = root / "studies/hybrid_lres/SOURCE_MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(external_dependencies=[dict(commit="pinned-test-commit", files=records)])))

    def git_read(command, **kwargs):
        return ("pinned-test-commit" if command[-1] == "HEAD" else str(checkout)) + "\n"

    monkeypatch.setattr(verifier.subprocess, "check_output", git_read)
    return root, checkout


def test_external_checkout_exact_hashes(synthetic_external_checkout):
    root, checkout = synthetic_external_checkout
    assert len(verifier.verify_heppy(checkout, root)) == 4


def test_external_checkout_rejects_changed_source(synthetic_external_checkout):
    root, checkout = synthetic_external_checkout
    (checkout / "GroomerShop.cc").write_bytes(b"different")
    with pytest.raises(ValueError, match="External heppy source differs"):
        verifier.verify_heppy(checkout, root)


def test_external_checkout_rejects_wrong_commit(synthetic_external_checkout, monkeypatch):
    root, checkout = synthetic_external_checkout
    monkeypatch.setattr(verifier.subprocess, "check_output", lambda command, **kwargs: "wrong-commit\n")
    with pytest.raises(ValueError, match="pinned root/commit"):
        verifier.verify_heppy(checkout, root)


def test_external_checkout_rejects_symlink(synthetic_external_checkout):
    root, checkout = synthetic_external_checkout
    file = checkout / "GroomerShop.cc"
    file.unlink()
    file.symlink_to(checkout / "GroomerShop.hh")
    with pytest.raises(ValueError, match="escapes checkout"):
        verifier.verify_heppy(checkout, root)


def test_no_raw_events_binaries_or_scheduling_controllers_imported():
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    for record in manifest["files"]:
        path = Path(record["path"])
        assert path.suffix not in {".npz", ".parquet", ".so", ".slurm"}
        assert path.name not in {"cambridge.sh", "launch.py", "recovery.py", "controller.py"}


def test_tampered_source_fails_closed(tmp_path):
    source = HERE / "SOURCE_MANIFEST.json"
    manifest = json.loads(source.read_text())
    record = manifest["files"][0]
    manifest["files"] = [record]
    manifest["preserved_base_files"] = {}
    target = tmp_path / "studies/hybrid_lres/SOURCE_MANIFEST.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(manifest))
    imported = tmp_path / record["path"]
    imported.parent.mkdir(parents=True, exist_ok=True)
    imported.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Imported source differs"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize("name", ["../outside", "/absolute/path"])
def test_manifest_cannot_escape_checkout(tmp_path, name):
    target = tmp_path / "studies/hybrid_lres/SOURCE_MANIFEST.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(dict(files=[dict(path=name)], preserved_base_files={})))
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        verifier.verify(tmp_path)


def test_build_requires_explicit_opt_in_before_loading_native_dependencies():
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PAPER_NATIVE_BUILD", None)
    result = subprocess.run([sys.executable, str(HERE / "build_native.py"), "--target", "upstream"],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "PAPER_NATIVE_BUILD=1" in result.stderr


@pytest.mark.parametrize("target", ["upstream", "precision", "wake"])
def test_heppy_build_requires_checkout_before_native_import_or_compilation(target):
    environment = dict(os.environ, PAPER_NATIVE_BUILD="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(HERE / "build_native.py"), "--target", target],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "--heppy-root is required" in result.stderr


def test_nominal_and_endpoint_pins_match_imported_source_identities():
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    imports = {record["source_workspace_relative"]: record["sha256"] for record in manifest["files"]}
    base = HERE / "provenance/atlas_rg_inference_20261015/recovery_20261016/precision_continuation"
    nominal = json.loads((base / "timeout_completion/CONTROL_MANIFEST.json").read_text())
    for filename in ("atlas_rg.py", "atlas_dr12.py"):
        path = "JETSCAPE-analysis-atlas-dr12/jetscape_analysis/analysis/" + filename
        actual = imports.get(path, manifest["preserved_base_files"].get(path.split("/", 1)[1]))
        assert actual == nominal["effective_backend"][filename]
    source = "atlas_rg_implementation_20261015/upstream_native/rg_upstream.cc"
    assert imports[source] == nominal["original_source_pins"]["native/rg_upstream.cc"]
    endpoint = HERE / "provenance/wake_bounded_transport_20261017/endpoint_bank/physical_prior_restriction/seed_scaling_recovery"
    control = json.loads((endpoint / "CONTROL.json").read_text())
    policy = "wake_bounded_transport_20261017/endpoint_bank/upstream_hardest_recovery/dr12_policy.cc"
    assert imports[policy] == control["policy_source_sha256"]
    for filename in ("seed_scaling.py", "collector_patch.py"):
        source = "wake_bounded_transport_20261017/endpoint_bank/physical_prior_restriction/seed_scaling_recovery/" + filename
        assert imports[source] == control["numerical_source_sha256"][filename]
