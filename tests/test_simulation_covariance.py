"""Analytic checks of caller-aligned event-count covariance utilities."""
import numpy as np
import pytest

from jetscape_analysis.analysis.simulation_covariance import (
    EventMoments, histogram_counts, ratio_map, ratio_bank,
)
from jetscape_analysis.analysis import simulation_covariance


def test_multiple_jets_same_event_and_cross_observable_covariance():
    moments = EventMoments(3)
    moments.update([2, 1, 0], 3)
    moments.update([0, 0, 1], 2)
    moments.update([0, 0, 0], 4)
    vectors = np.array([[6, 3, 0, 3], [0, 0, 2, 2], [0, 0, 0, 4]])
    np.testing.assert_array_equal(moments.total, vectors.sum(0))
    np.testing.assert_array_equal(moments.outer, vectors.T @ vectors)
    assert moments.outer[0, 0] == 36  # not 2 * 3**2, the per-jet answer
    assert moments.outer[0, 1] == 18
    assert moments.total[-1] == 9
    assert moments.outer[-1, -1] == 29


def test_root_edges_and_theta_untagged_normalization():
    counts, norm = histogram_counts([-1, 0, .1, .999, 1, 2], [0, .1, 1])
    np.testing.assert_array_equal(counts, [1, 2])
    assert norm == 4


def test_normalization_induces_negative_bin_correlations():
    # Multinomial fraction covariance emerges from independent Poisson counts.
    total = np.array([40., 60.])
    values, jac = ratio_map(total, np.eye(2), np.ones((2, 2)))
    cov = jac @ np.diag(total) @ jac.T
    np.testing.assert_allclose(values, [.4, .6])
    np.testing.assert_allclose(cov, [[.0024, -.0024], [-.0024, .0024]])
    np.testing.assert_allclose(cov.sum(1), 0, atol=1e-17)


def test_ratio_jacobian_finite_difference():
    total = np.array([4., 8., 15.])
    a = np.array([[1., 0, 0], [0, 1., 0]])
    b = np.array([[0., 0, 1], [1., 1, 0]])
    _, jac = ratio_map(total, a, b, factors=[2., 3.])
    numerical = np.empty_like(jac)
    for i in range(3):
        step = np.eye(3)[i] * 1e-5
        numerical[:, i] = (ratio_map(total+step, a, b, [2., 3.])[0]
                            - ratio_map(total-step, a, b, [2., 3.])[0]) / 2e-5
    np.testing.assert_allclose(jac, numerical, atol=1e-10)


def test_shared_pp_creates_cross_design_and_cross_observable_covariance():
    aa = np.array([[2., 4.], [3., 5.]])
    ca = np.array([[[.4, .1], [.1, .5]], [[.3, .05], [.05, .6]]])
    pp = np.array([5., 6.])
    cp = np.array([[.2, .08], [.08, .3]])
    values, cov = ratio_bank(aa, ca, pp, cp)
    np.testing.assert_allclose(values, aa/pp)
    np.testing.assert_allclose(cov[0, :, 1, :], np.outer(aa[0]/pp**2, aa[1]/pp**2)*cp)
    np.testing.assert_allclose(cov[0, :, 0, :], ca[0]/np.outer(pp, pp) + np.outer(aa[0]/pp**2, aa[0]/pp**2)*cp)
    assert np.linalg.eigvalsh(cov.reshape(4, 4)).min() > 0


@pytest.mark.parametrize('name', ['extract', 'main', 'validate_alignment', 'digest'])
def test_no_saved_stat_adapter_or_weight_only_alignment(name):
    assert not hasattr(simulation_covariance, name)


@pytest.mark.parametrize('values', [[np.nan], [[1., 2.]]])
def test_invalid_observables_fail(values):
    with pytest.raises(ValueError):
        histogram_counts(values, [0, 1])


def test_empty_pp_bins_fail_instead_of_zero_error():
    with pytest.raises(ValueError, match='empty pp'):
        ratio_bank(np.ones((2, 1)), np.ones((2, 1, 1)), np.zeros(1), np.eye(1))
