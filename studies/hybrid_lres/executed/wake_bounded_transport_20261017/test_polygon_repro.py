"""Captured AA510 chunk35/design7/event1483/a0.25 polygon regression.

Arguments were captured at geometry.polygon_angles during the bounded deform.
Constructor/project tests are synthetic. One integration test reads only the
captured local event; no test performs native reconstruction.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("polygon_repro_test_subject", HERE / "bounded_projection.py")
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)


def captured_polygon():
    pt = np.array([
        1.3928208717318535,
        0.5533391631370856,
        1.4621894493086556,
        1.2634907586575121,
        1.349752779765797,
        1.2201284612306154,
        1.6992371006566405,
        0.15139042847762207,
        0.20953106805819965,
        0.19994440069773733,
        0.3205887412158673,
        0.780963713636926,
        0.4879203304442757,
        2.0693784313514514,
        1.4352354829293579,
        8.112090939696778e-18,
        1.0680811914139567,
        0.17467543043562686,
        0.23137199937258215,
        0.09807690709614174,
        0.4322251078511112,
        0.49969976913667524,
        0.1274103893328044,
        1.4790761493520423,
        0.18980269055101995,
        0.19974990855909705,
        0.18173918072105705,
        0.35542347692567355,
        8.112090939696778e-18,
        0.4447890706619013,
    ], dtype=np.float64)
    preferred = np.array([
        2.5995357248795266,
        2.2776259348093753,
        1.4525603894845651,
        -2.776713042529982,
        -0.9665567240471774,
        2.7909446598198038,
        -3.056353522435013,
        -0.40944325977661716,
        0.7809242116374329,
        -2.2497213458175787,
        1.0839839034750207,
        1.5804066375351493,
        -0.1085947832390372,
        -1.1105291829277801,
        -1.035711796451182,
        -3.141592653589793,
        1.1804240495782072,
        -0.13135848982781653,
        0.5276289292314005,
        0.8217862126650535,
        -2.903152710573406,
        -2.126324749422034,
        -1.7879201578913524,
        0.555743136569141,
        1.0285800053996144,
        0.3369562694203364,
        0.9811525413152012,
        -1.9918280529675358,
        -0.0,
        -2.5214531896594274,
    ], dtype=np.float64)
    target = np.array([-1.132674556006441, -1.4155157996751182], dtype=np.float64)
    return pt, preferred, target


def test_actual_event1483_polygon_constructor_preserves_transverse_target():
    pt, preferred, target = captured_polygon()
    before_pt, before_preferred, before_target = pt.copy(), preferred.copy(), target.copy()
    radius = np.hypot(*target)
    assert max(0., 2 * pt.max() - pt.sum()) <= radius <= pt.sum()
    angles = projection.construct_phi(pt, preferred, target)
    assert angles.shape == pt.shape
    assert np.isfinite(angles).all()
    actual = np.array([(pt * np.cos(angles)).sum(), (pt * np.sin(angles)).sum()])
    tolerance = 2e-12 * max(1., pt.sum(), np.hypot(*target))
    assert abs(actual - target).max() <= tolerance
    small = pt <= 64 * np.finfo(float).eps * pt.sum()
    assert small.sum() == 2
    np.testing.assert_array_equal(angles[small], preferred[small])
    assert (pt[small] > 0).all()
    np.testing.assert_allclose(np.hypot(pt * np.cos(angles), pt * np.sin(angles)), pt,
                               atol=0., rtol=1e-15)
    np.testing.assert_array_equal(pt, before_pt)
    np.testing.assert_array_equal(preferred, before_preferred)
    np.testing.assert_array_equal(target, before_target)


def assert_project_guards(result, pt, masses2, target):
    assert result.shape == (len(pt), 4)
    assert result.dtype == np.dtype(np.float64)
    assert np.isfinite(result).all()
    assert (result[:, 3] >= 0).all()
    np.testing.assert_allclose(result.sum(0), target, atol=2e-8, rtol=2e-12)
    np.testing.assert_allclose(result[:, 3] ** 2 - (result[:, :3] ** 2).sum(1),
                               masses2, atol=2e-7, rtol=1e-7)
    actual_pt = np.hypot(result[:, 0], result[:, 1])
    np.testing.assert_allclose(actual_pt, pt, atol=1e-12, rtol=1e-12)
    assert (actual_pt <= pt * (1 + 1e-12) + 1e-12).all()


def test_full_projection_with_captured_polygon_preserves_every_row_and_pt():
    pt, preferred, transverse = captured_polygon()
    masses2 = np.full(len(pt), .13957 ** 2)
    target = np.r_[transverse, 0., np.sqrt(masses2 + pt ** 2).sum()]
    result, qa = projection.project(target, pt, masses2, np.zeros(len(pt)), preferred)
    assert_project_guards(result, pt, masses2, target)
    assert qa["analytic_constructor_available"]
    assert qa["pt_damping"] == pytest.approx(1., abs=2e-12)
    tiny = pt < 1e-16
    assert (np.hypot(result[tiny, 0], result[tiny, 1]) > 0).all()


def fail_constructor(*args, **kwargs):
    raise RuntimeError("Synthetic optional analytic constructor failure")


def triangle_case():
    pt, masses2 = np.ones(3), np.full(3, .1)
    preferred = np.array([0., 2 * np.pi / 3, -2 * np.pi / 3])
    target = np.array([(pt * np.cos(preferred)).sum(), (pt * np.sin(preferred)).sum(),
                       0., np.sqrt(masses2 + pt ** 2).sum()])
    return pt, masses2, preferred, target


def test_optional_constructor_failure_keeps_valid_newton_solution(monkeypatch):
    pt, masses2, preferred, target = triangle_case()
    monkeypatch.setattr(projection, "construct_phi", fail_constructor)
    result, qa = projection.project(target, pt, masses2, np.zeros(3), preferred)
    assert not qa["analytic_constructor_available"]
    assert_project_guards(result, pt, masses2, target)


def test_failure_of_both_starts_remains_fatal(monkeypatch):
    pt, masses2, preferred, target = triangle_case()
    monkeypatch.setattr(projection, "match_phi", lambda *args: None)
    monkeypatch.setattr(projection, "construct_phi", fail_constructor)
    with pytest.raises(RuntimeError, match="Synthetic optional analytic constructor failure"):
        projection.project(target, pt, masses2, np.zeros(3), preferred)


def test_optional_constructor_failure_cannot_bypass_final_guards(monkeypatch):
    pt, masses2, _, target = triangle_case()
    monkeypatch.setattr(projection, "match_phi", lambda *args: np.zeros(3))
    monkeypatch.setattr(projection, "construct_phi", fail_constructor)
    with pytest.raises(AssertionError):
        projection.project(target, pt, masses2, np.zeros(3), np.zeros(3))


def test_actual_event1483_with_calibrated_transport():
    pq = pytest.importorskip('pyarrow.parquet', reason='Real-parquet integration uses native analysis environment')

    path = (HERE.parent / "atlas_rg_inference_20261015/recovery_20261016"
            / "diagnostic_aa510_chunk35/input.parquet")
    events = pq.read_table(path, filters=[("event_ID", "=", 1483)]).to_pylist()
    assert len(events) == 1
    spec = importlib.util.spec_from_file_location("polygon_repro_transport", HERE / "transport.py")
    transport = importlib.util.module_from_spec(spec)
    previous_path = sys.path.copy()
    sys.path.insert(0, str(HERE))
    try:
        spec.loader.exec_module(transport)
    finally:
        sys.path[:] = previous_path
    event = events[0]
    original = transport.REFERENCE.four_vectors(event)
    old_status = np.asarray(event["status"])
    signed_before = (original[np.isin(old_status, (1, 2))]
                     * np.where(old_status[np.isin(old_status, (1, 2))] == 1, 1, -1)[:, None]).sum(0)
    for a in (.25, .5, .75, 1.):
        observations = []
        result, qa = transport.deform(event, a, observer=observations.append)
        assert qa["whole_event_fallback"] == 0
        assert qa["pt_envelope_ratio"] <= 1 + 1e-12
        assert len(observations) == 1
        final = transport.REFERENCE.four_vectors(result)
        status = np.asarray(result["status"])
        wake = np.isin(status, (1, 2))
        signed_after = (final[wake] * np.where(status[wake] == 1, 1, -1)[:, None]).sum(0)
        np.testing.assert_allclose(signed_after, signed_before, atol=2e-7, rtol=2e-10)
        selected = observations[0]["selected_original_indices"]
        old_mass2 = transport.REFERENCE.mass2(original)
        old_pt = np.hypot(original[:, 0], original[:, 1])
        eligible = np.isin(old_status, (1, 2)) & (old_mass2 >= 0)
        eligible &= ~((old_pt == 0) & (old_mass2 == 0))
        keep = np.flatnonzero(~eligible | np.isin(np.arange(len(original)), selected))
        positions = np.searchsorted(keep, selected)
        np.testing.assert_array_equal(result["particle_ID"], np.asarray(event["particle_ID"])[keep])
        np.testing.assert_allclose(transport.REFERENCE.mass2(final), old_mass2[keep], atol=2e-7, rtol=1e-7)
        assert (np.hypot(final[positions, 0], final[positions, 1])
                <= observations[0]["pt_caps"] * (1 + 1e-12) + 1e-12).all()
        for sign, target in zip((1, 2), observations[0]["targets_lab"]):
            rows = old_status[selected] == sign
            np.testing.assert_allclose(final[positions][rows].sum(0), target, atol=2e-7, rtol=2e-10)
