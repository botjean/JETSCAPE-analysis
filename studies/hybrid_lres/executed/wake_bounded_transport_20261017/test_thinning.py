"""Pure local tests of the conditional finite-population thinning law."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bounded_thinning_test_subject", HERE / "thinning.py")
thinning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(thinning)
conditioned_indices = thinning.conditioned_indices


@pytest.mark.parametrize("amplitude", [.01, .3, .8, 1.])
def test_species_charges_determinism_and_retained_row_contract(amplitude):
    pid = np.array([111] * 7 + [211] * 5 + [-211] * 2 + [2212] * 2 + [-2212] * 4
                   + [321, -321, 22, 221, 331])
    before = pid.copy()
    for seed in range(25):
        selected = conditioned_indices(pid, amplitude, np.random.default_rng(seed))
        repeated = conditioned_indices(pid, amplitude, np.random.default_rng(seed))
        np.testing.assert_array_equal(selected, repeated)
        assert len(selected) >= 2
        assert selected.dtype.kind == "i"
        assert np.all(np.diff(selected) > 0)
        assert ((selected >= 0) & (selected < len(pid))).all()
        for species in (211, 2212, 321):
            old = (pid == species).sum() - (pid == -species).sum()
            new = (pid[selected] == species).sum() - (pid[selected] == -species).sum()
            assert new == old
        if amplitude == 1.:
            np.testing.assert_array_equal(selected, np.arange(len(pid)))
    np.testing.assert_array_equal(pid, before)


@pytest.mark.parametrize("pid", [111, -111, 211, -211, 2212, -2212])
@pytest.mark.parametrize("amplitude", [.01, .5, 1.])
def test_singleton_is_identity_without_consuming_rng(pid, amplitude):
    rng, untouched = np.random.default_rng(401), np.random.default_rng(401)
    np.testing.assert_array_equal(conditioned_indices([pid], amplitude, rng), [0])
    assert rng.random() == untouched.random()


def test_empty_group_stays_empty():
    assert conditioned_indices([], .3, np.random.default_rng(402)).shape == (0,)


@pytest.mark.parametrize("pid", [[111, 111], [211, -211], [211, 211],
                                [211, -2212], [111, 211], [-2212, -2212]])
@pytest.mark.parametrize("amplitude", [.001, .4, 1.])
def test_two_rows_must_both_survive(pid, amplitude):
    for seed in range(5):
        np.testing.assert_array_equal(conditioned_indices(pid, amplitude, np.random.default_rng(seed)), [0, 1])


@pytest.mark.parametrize("pid", [[111, 111], [111, 111, 111], [211, -211], [111, 211]])
@pytest.mark.parametrize("amplitude", [1e-200, np.nextafter(0., 1.)])
def test_tiny_valid_amplitudes_do_not_underflow_the_conditioning(pid, amplitude):
    with np.errstate(all="raise"):
        selected = conditioned_indices(pid, amplitude, np.random.default_rng(403))
    assert len(selected) >= 2


class NeutralUniforms:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)

    def permutation(self, values):
        raise AssertionError("Neutral units must not be particle/antiparticle paired")


@pytest.mark.parametrize("amplitude", [.1, .25, .5, .9])
def test_three_neutral_analytic_p2_p3_and_conditional_branch_boundaries(amplitude):
    a = amplitude
    first = (2 - a) / (3 - 2 * a)
    second_after_keep = 1 / (2 - a)
    delta = 1e-10
    cases = [
        ([first - delta, second_after_keep - delta, a - delta], [0, 1, 2]),
        ([first - delta, second_after_keep - delta, a + delta], [0, 1]),
        ([first - delta, second_after_keep + delta, .5], [0, 2]),
        ([first + delta, .5, .5], [1, 2]),
    ]
    for uniforms, expected in cases:
        np.testing.assert_array_equal(conditioned_indices([111] * 3, a, NeutralUniforms(uniforms)), expected)
    normalization = 3 * a ** 2 * (1 - a) + a ** 3
    p3 = first * second_after_keep * a
    pairs = [first * second_after_keep * (1 - a),
             first * (1 - second_after_keep), 1 - first]
    assert p3 == pytest.approx(a ** 3 / normalization, abs=2e-15)
    np.testing.assert_allclose(pairs, np.full(3, a ** 2 * (1 - a) / normalization), atol=2e-15)
    assert sum(pairs) == pytest.approx(3 * a ** 2 * (1 - a) / normalization, abs=2e-15)
    assert p3 + sum(pairs) == pytest.approx(1., abs=2e-15)


def test_three_neutral_sample_frequencies_match_analytic_p2_p3():
    amplitude, count = .3, 4000
    rng = np.random.default_rng(404)
    sizes = np.array([len(conditioned_indices([111] * 3, amplitude, rng)) for _ in range(count)])
    normalization = 3 * amplitude ** 2 * (1 - amplitude) + amplitude ** 3
    expected_p3 = amplitude ** 3 / normalization
    expected_p2 = 3 * amplitude ** 2 * (1 - amplitude) / normalization
    tolerance = 6 * np.sqrt(expected_p3 * (1 - expected_p3) / count)
    assert np.mean(sizes == 3) == pytest.approx(expected_p3, abs=tolerance)
    assert np.mean(sizes == 2) == pytest.approx(expected_p2, abs=tolerance)


@pytest.mark.parametrize("pid", [[0], [211, 0], [111.], [211.5], [True],
                                [[211, -211]], 111, [np.nan], [np.inf], ["111"],
                                [np.iinfo(np.int64).min], np.array([2 ** 63], dtype=np.uint64)])
def test_invalid_particle_ids_raise(pid):
    with pytest.raises(ValueError):
        conditioned_indices(pid, .5, np.random.default_rng(405))


@pytest.mark.parametrize("amplitude", [0., -.1, 1.01, np.nan, np.inf, -np.inf,
                                      True, False, [.5], "0.5", .5 + 0j, None])
def test_invalid_amplitudes_raise_even_for_passthrough(amplitude):
    for pid in ([], [111], [111, 111]):
        with pytest.raises(ValueError):
            conditioned_indices(pid, amplitude, np.random.default_rng(406))
