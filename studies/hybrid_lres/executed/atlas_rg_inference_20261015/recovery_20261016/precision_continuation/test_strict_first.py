from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strict_first import RETENTION_ERROR, StrictFirst

OPTIONS = dict(negative_marker=-1, ca_radius=1000, zero_energy_scale=.0001,
               grooming_policy="signed_prongs", require_all_constituents=True)


def fail(error):
    def call(*args, **kwargs):
        raise error
    return SimpleNamespace(reconstruct=call)


def test_success_returns_identical_object_and_never_calls_fallback():
    value = {"jets": [(200., .1, 0., True)]}
    wrapper = StrictFirst(SimpleNamespace(reconstruct=lambda *a, **k: value),
                          fail(AssertionError("Fallback must not be called")))
    assert wrapper.reconstruct([], **OPTIONS) is value
    assert wrapper.audit == {"unchanged_successes": 1}


@pytest.mark.parametrize("error", [RuntimeError("Other numerical failure"),
    ValueError(RETENTION_ERROR), RuntimeError(RETENTION_ERROR + " ")])
def test_unrelated_failure_is_not_retried(error):
    wrapper = StrictFirst(fail(error), fail(AssertionError("No broad catch")))
    with pytest.raises(type(error), match=str(error)):
        wrapper.reconstruct([], **OPTIONS)
    assert not wrapper.audit


def test_only_exact_failure_and_configuration_are_recovered():
    calls = []
    value = dict(precision_boundary_interventions=2,
                 reclustering=[dict(all_constituents_retained=True)])
    def precision(rows, **kwargs):
        calls.append((rows, kwargs))
        return value
    rows = [[1, 2, 3]]
    wrapper = StrictFirst(fail(RuntimeError(RETENTION_ERROR)), SimpleNamespace(reconstruct=precision))
    assert wrapper.reconstruct(rows, **OPTIONS) is value
    assert calls == [(rows, dict(**OPTIONS, source_decimal_digits=6, source_float32=True))]
    assert wrapper.audit["recovered_events"] == 1
    with pytest.raises(ValueError, match="exact frozen"):
        wrapper.reconstruct(rows, **{**OPTIONS, "zero_energy_scale": .001})


@pytest.mark.parametrize("value", [dict(), dict(precision_boundary_interventions=0),
    dict(precision_boundary_interventions=1, reclustering=[]),
    dict(precision_boundary_interventions=1, reclustering=[dict(all_constituents_retained=False)])])
def test_no_silent_acceptance_without_precision_and_retention(value):
    wrapper = StrictFirst(fail(RuntimeError(RETENTION_ERROR)),
                          SimpleNamespace(reconstruct=lambda *a, **k: value))
    with pytest.raises(RuntimeError):
        wrapper.reconstruct([], **OPTIONS)


def test_actual_known_failure_and_unchanged_old_successes():
    work = HERE.parents[2]
    sys.path.insert(0, str(work / "JETSCAPE-analysis-atlas-dr12"))
    sys.path.insert(0, str(HERE.parent / "precision_candidate_local"))
    sys.path.insert(0, str(work / "atlas_rg_implementation_20261015/upstream_native"))
    from jetscape_analysis.analysis import atlas_rg as rg
    import _rg_precision_candidate as precision
    strict = rg._upstream_backend()
    wrapper = StrictFirst(strict, precision)
    rows = np.load(HERE.parent / "diagnostic_aa510_chunk35/event_2025.npz")["rows"].tolist()
    with pytest.raises(RuntimeError, match=RETENTION_ERROR):
        strict.reconstruct(rows, **OPTIONS)
    result = wrapper.reconstruct(rows, **OPTIONS)
    assert result["precision_boundary_interventions"] == 2
    assert all(d["all_constituents_retained"] for d in result["reclustering"])
    simple = [[220., 0., 0., 220., 1., 1., 1.]]
    np.testing.assert_array_equal(wrapper.reconstruct(simple, **OPTIONS)["jets"],
                                  strict.reconstruct(simple, **OPTIONS)["jets"])


def test_candidate_exception_is_not_swallowed():
    wrapper = StrictFirst(fail(RuntimeError(RETENTION_ERROR)),
                          fail(ValueError("Incompatible source precision")))
    with pytest.raises(ValueError, match="Incompatible source"):
        wrapper.reconstruct([], **OPTIONS)
    assert wrapper.audit["retention_failures"] == 1
    assert wrapper.audit["recovered_events"] == 0


def test_install_scoped_counter_and_provenance(tmp_path, monkeypatch):
    from strict_first import install
    import hashlib
    policy = tmp_path / "strict_first.py"
    policy.write_text("test policy")
    binary = tmp_path / "_rg_precision_candidate.so"
    binary.write_bytes(b"test binary")
    (tmp_path / "precision_build.json").write_text("{}")
    result = dict(jets=[], cutflow={})
    backend = SimpleNamespace(reconstruct=lambda *a, **k: result)
    class RG:
        def __init__(self):
            self._upstream_backend = lambda: backend
            self.reconstruction_hashes = lambda config=None: {"strict.so": "unchanged"}
            self.count_rows = lambda rows, config: self._upstream_backend().reconstruct(rows, **OPTIONS)
            self.count_event = lambda event, config, system: self.count_rows(event["rows"], config)
    rg = RG()
    precision = SimpleNamespace(__file__=str(binary), reconstruct=fail(AssertionError("must not retry")).reconstruct)
    wrapper = install(rg, precision, policy)
    assert rg.count_event(dict(rows=[], event_ID=12), {}, "pp") is result
    assert rg._upstream_backend() is backend
    assert wrapper.context is None
    assert rg.reconstruction_hashes()["strict.so"] == "unchanged"
    assert rg.reconstruction_hashes()["strict_first.py"] == hashlib.sha256(policy.read_bytes()).hexdigest()


def test_stable_prefix_rejects_tampering(tmp_path):
    import json
    import hashlib
    from collect_continuation import stable_prefix
    np.savez_compressed(tmp_path / "checkpoint.npz", new_total=np.array([1.]))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    meta = dict(checkpoint_sha256=digest(tmp_path / "checkpoint.npz"),
        reconstruction_sha256={"strict": "old"}, completed_chunks=[12, 23],
        batch=dict(chunks=[12, 23, 35]), reports=[dict(chunk_index=12), dict(chunk_index=23)],
        slurm_job_id="123")
    path = tmp_path / "CHECKPOINT.json"
    path.write_text(json.dumps(meta))
    expected = dict(directory=str(tmp_path), metadata_sha256=digest(path),
        checkpoint_sha256=meta["checkpoint_sha256"], chunks=[12, 23])
    arrays, reports, proof = stable_prefix(expected, {"strict": "old"}, {"strict": "old", "policy": "new"}, "policyhash")
    assert proof["original_execution_backend"] == {"strict": "old"}
    assert proof["original_slurm_job_id"] == "123"
    with pytest.raises(ValueError, match="backend changed"):
        stable_prefix(expected, {"strict": "wrong"}, {}, "policyhash")
    with (tmp_path / "checkpoint.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="backend changed"):
        stable_prefix(expected, {"strict": "old"}, {}, "policyhash")
