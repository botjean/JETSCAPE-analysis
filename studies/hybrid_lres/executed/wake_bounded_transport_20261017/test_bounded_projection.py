"""Bounded synthetic projection checks; no event files or native backend."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bounded_projection_test_subject", HERE / "bounded_projection.py")
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)


def four(pt, masses2, phi, y):
    pt, masses2, phi, y = map(np.asarray, (pt, masses2, phi, y))
    mt = np.sqrt(pt ** 2 + masses2)
    return np.column_stack((pt * np.cos(phi), pt * np.sin(phi),
                            mt * np.sinh(y), mt * np.cosh(y)))


def assert_contract(result, caps, masses2, target):
    assert result.shape == (len(caps), 4)
    assert result.dtype == np.dtype(np.float64)
    assert np.isfinite(result).all()
    assert (result[:, 3] >= 0).all()
    pt = np.hypot(result[:, 0], result[:, 1])
    assert np.all(pt <= np.asarray(caps) * (1 + 1e-12) + 1e-12)
    np.testing.assert_allclose(result.sum(0), target, atol=2e-8, rtol=2e-12)
    shell = result[:, 3] ** 2 - (result[:, :3] ** 2).sum(1)
    np.testing.assert_allclose(shell, masses2, atol=2e-7, rtol=1e-7)
    assert result[:, 3].max() <= target[3] + 2e-8


@pytest.mark.parametrize("count", [2, 3, 8, 16])
@pytest.mark.parametrize("hardening", [1., 1.25, 2.])
def test_full_original_population_has_d_equals_inverse_h_certificate(count, hardening):
    rng = np.random.default_rng(6100 + count)
    for _ in range(6):
        old_pt, masses2 = rng.uniform(.1, 2., count), rng.uniform(.02, .7, count)
        phi, y = rng.uniform(-np.pi, np.pi, count), rng.uniform(-1.5, 1.5, count)
        target = four(old_pt, masses2, phi, y).astype(np.longdouble).sum(0)
        caps = hardening * old_pt
        interval = projection.bounds(target, caps, masses2)
        assert interval is not None
        low, high = interval
        assert low - 2e-12 <= 1 / hardening <= high + 2e-12
        result, qa = projection.project(target, caps, masses2, .8 * y, .8 * phi)
        assert 0 <= qa["pt_damping"] <= 1 + 1e-12
        assert_contract(result, caps, masses2, target)


@pytest.mark.parametrize("count", [2, 3, 8])
def test_original_shapes_are_preserved_when_h_is_one(count):
    rng = np.random.default_rng(6200 + count)
    pt, masses2 = rng.uniform(.2, 2., count), rng.uniform(.02, .5, count)
    phi, y = rng.uniform(-2., 2., count), rng.uniform(-.8, .8, count)
    original = four(pt, masses2, phi, y)
    result, qa = projection.project(original.astype(np.longdouble).sum(0), pt, masses2, y, phi)
    np.testing.assert_allclose(result, original, atol=2e-9, rtol=2e-11)
    assert qa["pt_damping"] == pytest.approx(1., abs=2e-12)


@pytest.mark.parametrize("rotation", [0., .7, -2.])
def test_collinear_polygon_boundary_is_feasible(rotation):
    pt, masses2 = np.array([1., 2., 3.]), np.full(3, .1)
    original = four(pt, masses2, np.full(3, rotation), np.array([-.4, .2, .6]))
    target = original.astype(np.longdouble).sum(0)
    result, _ = projection.project(target, pt, masses2, [-.2, .1, .3], [.4, -.3, 1.1])
    assert_contract(result, pt, masses2, target)


def test_singleton_requires_mass_equality_not_only_a_lower_bound():
    assert projection.bounds([0., 0., 0., 2.], [1.], [1.]) is None
    target = np.array([.6, .8, .7, np.sqrt(1. + 1. + .7 ** 2)])
    interval = projection.bounds(target, [2.], [1.])
    assert interval is not None
    np.testing.assert_allclose(interval, [.5, .5], atol=2e-12)
    result, _ = projection.project(target, [2.], [1.], [.1], [.2])
    assert_contract(result, [2.], [1.], target)


def test_zero_mt_padding_does_not_create_internal_rapidity_freedom():
    impossible = [0., 0., 0., 2.]
    assert projection.bounds(impossible, [1.], [1.]) is None
    assert projection.bounds(impossible, [1., 0., 0.], [1., 0., 0.]) is None
    target = np.array([.6, 0., .2, np.sqrt(1. + .6 ** 2 + .2 ** 2)])
    result, _ = projection.project(target, [1., 0., 0.], [1., 0., 0.],
                                    [.1, .1, .1], [.2, -.3, .5])
    assert_contract(result, [1., 0., 0.], [1., 0., 0.], target)
    np.testing.assert_array_equal(result[1:], np.zeros((2, 4)))


def test_zero_pt_massive_rows_have_a_valid_threshold_configuration():
    target = np.array([0., 0., 0., 3.])
    result, _ = projection.project(target, [0., 0.], [1., 4.], [0., 0.], [.2, -.8])
    assert_contract(result, [0., 0.], [1., 4.], target)


def test_failed_thinned_subset_has_a_feasible_full_population_certificate():
    positive = np.array([8., 0., 0., 9.])
    negative = np.array([-.8, 0., 0., 1.6])
    masses2 = np.full(12, .02)
    signs = np.r_[np.ones(8), -np.ones(4)]
    caps = np.r_[np.full(8, 1.25), np.full(4, .22)]
    subset = np.array([0, 1, 8, 9])
    assert projection.choose_budgets(positive, negative, caps[subset], masses2[subset], signs[subset], .6) is None
    match = projection.choose_budgets(positive, negative, caps, masses2, signs, .6)
    assert match is not None
    fraction, targets = match
    assert .6 <= fraction <= 1.
    np.testing.assert_allclose(targets[0] - targets[1], positive - negative, atol=1e-12)
    for sign, target in zip((1, -1), targets):
        assert projection.bounds(target, caps[signs == sign], masses2[signs == sign]) is not None


@pytest.mark.parametrize("target,caps,masses2", [
    ([0., 0., 0., np.nan], [1., 1.], [.1, .1]),
    ([0., 0., 0., np.inf], [1., 1.], [.1, .1]),
    ([0., 0., 0., 3.], [-1., 2.], [.1, .1]),
    ([0., 0., 0., 3.], [1., 1.], [-.1, .1]),
    ([0., 0., 0., 3.], [np.nan, 1.], [.1, .1]),
    ([0., 0., 0., 3.], [1., 1.], [.1]),
    ([0., 0., 0., 3.], [], []),
    ([0., 0., 0.], [1., 1.], [.1, .1]),
])
def test_bounds_invalid_inputs_raise_value_error(target, caps, masses2):
    with pytest.raises(ValueError):
        projection.bounds(target, caps, masses2)


@pytest.mark.parametrize("requested", [-.1, 0., 1.1, np.nan, np.inf, True, [.5]])
def test_budget_invalid_requested_fraction_raises_value_error(requested):
    with pytest.raises(ValueError):
        projection.choose_budgets([0., 0., 0., 5.], [0., 0., 0., 3.],
                                  np.ones(4), np.full(4, .1), [1, 1, -1, -1], requested)


def test_budget_invalid_signs_raise_value_error():
    with pytest.raises(ValueError):
        projection.choose_budgets([0., 0., 0., 5.], [0., 0., 0., 3.],
                                  np.ones(4), np.full(4, .1), [0, 1, -1, -1], .5)


def test_project_nonfinite_preferences_raise_value_error():
    with pytest.raises(ValueError):
        projection.project([0., 0., 0., 3.], [1., 1.], [.1, .1], [np.nan, 0.], [0., 0.])
