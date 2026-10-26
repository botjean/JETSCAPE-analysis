"""Synthetic bounded-transport integration tests; no native reconstruction."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bounded_transport_test_subject", HERE / "transport.py")
transport = importlib.util.module_from_spec(SPEC)
previous_path = sys.path.copy()
sys.path.insert(0, str(HERE))
try:
    SPEC.loader.exec_module(transport)
finally:
    sys.path[:] = previous_path


def four(pt, masses2, phi, y):
    pt, masses2, phi, y = map(np.asarray, (pt, masses2, phi, y))
    mt = np.sqrt(pt ** 2 + masses2)
    return np.column_stack((pt * np.cos(phi), pt * np.sin(phi),
                            mt * np.sinh(y), mt * np.cosh(y)))


@pytest.fixture
def paper(monkeypatch):
    settings = dict(parameters=dict(positive_pt_scale=1.25, negative_pt_scale=1.1,
                                   positive_phi_scale=.85, positive_y_scale=.9),
                    amplitudes=[.7, .6])
    monkeypatch.setattr(transport, "description", lambda a: settings)
    return settings


def synthetic_event(collinear=False):
    count = 20
    status = np.r_[[-2, -2, 0, 0], np.ones(8, int), np.full(8, 2)]
    pid = np.array([21, 21, 211, -211] + [211, -211, 2212, -2212, 111, 111, 111, 111] * 2)
    pt = np.r_[[10., 9., 4., 3.], np.ones(8), np.full(8, .2)]
    phi = np.linspace(-2., 2., count)
    if collinear:
        phi[4:12] = 0.
        phi[12:] = np.pi
    y = np.linspace(-.4, .4, count)
    vectors = four(pt, np.full(count, .13957 ** 2), phi, y)
    event = dict(zip(("px", "py", "pz", "E"), vectors.T))
    event.update(particle_ID=pid, status=status, event_ID=731, event_weight=.75,
                 pt_hat=40., vertex_x=.1, vertex_y=-.2, centrality=2., cross_section=1.)
    return event


def assert_event_contract(original, modified, qa, observation, paper):
    selected = observation["selected_original_indices"]
    old_status, new_status = np.asarray(original["status"]), np.asarray(modified["status"])
    active = np.isin(old_status, [1, 2])
    frozen = np.setdiff1d(np.flatnonzero(active), selected)
    old_vectors = transport.REFERENCE.four_vectors(original)
    old_masses = transport.REFERENCE.mass2(old_vectors)
    excluded = (old_masses[frozen] < 0) | ((np.hypot(old_vectors[frozen, 0], old_vectors[frozen, 1]) == 0)
                                        & (old_masses[frozen] == 0))
    keep = np.sort(np.r_[np.flatnonzero(~active), selected, frozen[excluded]])
    np.testing.assert_array_equal(modified["status"], old_status[keep])
    np.testing.assert_array_equal(modified["particle_ID"], np.asarray(original["particle_ID"])[keep])
    vectors = transport.REFERENCE.four_vectors(modified)
    unchanged = ~np.isin(keep, selected)
    np.testing.assert_array_equal(vectors[unchanged], old_vectors[keep[unchanged]])
    positions = np.searchsorted(keep, selected)
    pt = np.hypot(vectors[positions, 0], vectors[positions, 1])
    scales = np.where(old_status[selected] == 1,
                      paper["parameters"]["positive_pt_scale"], paper["parameters"]["negative_pt_scale"])
    caps = np.hypot(old_vectors[selected, 0], old_vectors[selected, 1]) * scales
    np.testing.assert_allclose(observation["pt_caps"], caps, atol=1e-12, rtol=1e-12)
    assert (pt <= caps * (1 + 1e-12) + 1e-12).all()
    np.testing.assert_allclose(transport.REFERENCE.mass2(vectors[positions]), old_masses[selected], atol=2e-7, rtol=1e-7)
    assert qa["whole_event_fallback"] == 0
    assert qa["pt_envelope_ratio"] <= 1 + 1e-12
    for sign, target in zip((1, 2), observation["targets_lab"]):
        rows = old_status[selected] == sign
        np.testing.assert_allclose(vectors[positions][rows].sum(0), target, atol=2e-7, rtol=2e-10)
        assert vectors[new_status == sign, 3].sum() <= old_vectors[old_status == sign, 3].sum() + 2e-7
        for species in (211, 2212):
            before = ((old_status == sign) & (np.asarray(original["particle_ID"]) == species)).sum()
            before -= ((old_status == sign) & (np.asarray(original["particle_ID"]) == -species)).sum()
            after = ((new_status == sign) & (np.asarray(modified["particle_ID"]) == species)).sum()
            after -= ((new_status == sign) & (np.asarray(modified["particle_ID"]) == -species)).sum()
            assert before == after
    for key in ("event_ID", "event_weight", "pt_hat", "vertex_x", "vertex_y", "centrality", "cross_section"):
        assert modified[key] == original[key]


def test_a0_is_exact_identity_without_loading_parameters(monkeypatch):
    event = {"sentinel": object()}
    def forbidden(a):
        raise AssertionError("Baseline must not obtain deformation parameters")
    monkeypatch.setattr(transport, "description", forbidden)
    result, qa = transport.deform(event, 0.)
    assert result is event
    assert qa["baseline_identity"] == 1


@pytest.mark.parametrize("seed", [2, 19, 20261018])
def test_synthetic_transport_charge_determinism_caps_and_nonwake_identity(seed, paper):
    event = synthetic_event()
    observations = []
    result, qa = transport.deform(event, .5, seed=seed, observer=observations.append)
    again, again_qa = transport.deform(event, .5, seed=seed)
    for key in transport.PARTICLE_KEYS:
        np.testing.assert_array_equal(result[key], again[key])
    assert qa == again_qa
    assert len(observations) == 1
    assert_event_contract(event, result, qa, observations[0], paper)


def test_infeasible_thinned_population_restores_all_original_rows(monkeypatch, paper):
    event = synthetic_event(collinear=True)
    monkeypatch.setattr(transport, "conditioned_indices", lambda pid, amplitude, rng: np.array([0, 1]))
    observations = []
    result, qa = transport.deform(event, .5, observer=observations.append)
    assert qa["population_restored"] == 1
    assert qa["restored_rows"] == 12
    assert qa["deleted_positive"] == qa["deleted_negative"] == 0
    assert qa["selected_positive"] == qa["original_positive"] == 8
    assert qa["selected_negative"] == qa["original_negative"] == 8
    assert_event_contract(event, result, qa, observations[0], paper)


def test_zero_mt_and_spacelike_wake_rows_are_frozen_exactly(paper):
    event = synthetic_event()
    for key, values in zip(("px", "py", "pz", "E"), [[0., 1.], [0., 0.], [1., 0.], [1., .5]]):
        event[key] = np.r_[event[key], values]
    event["status"] = np.r_[event["status"], [1, 2]]
    event["particle_ID"] = np.r_[event["particle_ID"], [22, 111]]
    observations = []
    result, qa = transport.deform(event, .5, observer=observations.append)
    assert qa["frozen_zero_mt"] == 1
    assert qa["frozen_spacelike"] == 1
    assert_event_contract(event, result, qa, observations[0], paper)


@pytest.mark.parametrize("singleton_status", [1, 2])
def test_singleton_sign_is_projected_at_original_budget_not_silently_bypassed(singleton_status, paper):
    event = synthetic_event()
    statuses = np.asarray(event["status"])
    singleton_row = np.flatnonzero(statuses == singleton_status)[0]
    keep = (statuses != singleton_status) | (np.arange(len(statuses)) == singleton_row)
    for key in transport.PARTICLE_KEYS:
        event[key] = np.asarray(event[key])[keep]
    observations = []
    result, qa = transport.deform(event, .5, observer=observations.append)
    assert len(observations) == 1
    assert qa["actual_negative_fraction"] == 1.
    assert qa["original_positive" if singleton_status == 1 else "original_negative"] == 1
    assert qa["selected_positive" if singleton_status == 1 else "selected_negative"] == 1
    assert_event_contract(event, result, qa, observations[0], paper)


@pytest.mark.parametrize("absent_status", [1, 2])
def test_originally_absent_sign_stays_absent(absent_status, paper):
    event = synthetic_event()
    keep = np.asarray(event["status"]) != absent_status
    for key in transport.PARTICLE_KEYS:
        event[key] = np.asarray(event[key])[keep]
    observations = []
    result, qa = transport.deform(event, .5, observer=observations.append)
    assert not np.any(np.asarray(result["status"]) == absent_status)
    assert_event_contract(event, result, qa, observations[0], paper)


@pytest.mark.parametrize("case", ["no_axis", "degenerate"])
def test_declared_passthroughs_are_event_identity(case, paper):
    event = synthetic_event()
    if case == "no_axis":
        event["status"][event["status"] == -2] = 0
    else:
        for key in transport.PARTICLE_KEYS:
            event[key] = event[key][:8]
    result, qa = transport.deform(event, .5)
    assert result is event
    assert qa["no_axis_events" if case == "no_axis" else "degenerate_events"] == 1


@pytest.mark.parametrize("a", [-.1, 1.1, np.nan, True, [.5]])
def test_invalid_global_a_raises_value_error(a):
    with pytest.raises(ValueError):
        transport.deform(synthetic_event(), a)


def test_subunit_paper_hardening_is_rejected(paper):
    paper["parameters"]["positive_pt_scale"] = .9
    with pytest.raises(ValueError, match="certificate"):
        transport.deform(synthetic_event(), .5)
