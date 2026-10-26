"""Bounded local seed-coordinate regressions; no event, native or cluster runs."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
GEOMETRY = WORK/'wake_sign_separated_20261017/projection.py'
CAPTURE = HERE.parent/'failure_event4245/failed_seed.npz'
ERROR = 'Cannot bracket per-sign rapidity seed'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load('_seed_scaling_frozen_geometry_tests', GEOMETRY)
subject = load('_seed_scaling_recovery_tests', HERE/'seed_scaling.py')
original = geometry.rapidity_seed


@pytest.fixture(scope='module', autouse=True)
def original_source_is_untouched():
    checksum = hashlib.sha256(GEOMETRY.read_bytes()).hexdigest()
    yield
    assert hashlib.sha256(GEOMETRY.read_bytes()).hexdigest() == checksum
    assert geometry.rapidity_seed is original


@pytest.fixture
def captured():
    with np.load(CAPTURE, allow_pickle=False) as archive:
        assert set(archive.files) == {'mt', 'preferred', 'target'}
        return tuple(archive[key].copy() for key in ('mt', 'preferred', 'target'))


def target_for(mt, rapidity):
    mt, rapidity = np.asarray(mt, np.longdouble), np.asarray(rapidity, np.longdouble)
    return np.array([0., 0., np.sum(mt*np.sinh(rapidity)), np.sum(mt*np.cosh(rapidity))],
                    dtype=np.longdouble)


def assert_longitudinal_closure(mt, rapidity, target):
    assert np.isfinite(rapidity).all() and np.max(np.abs(rapidity)) <= 30
    mt, rapidity = np.asarray(mt, np.longdouble), np.asarray(rapidity, np.longdouble)
    np.testing.assert_allclose(np.sum(mt*np.sinh(rapidity)), target[2], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(np.sum(mt*np.cosh(rapidity)), target[3], atol=2e-12, rtol=2e-12)


def test_actual_captured_failure_recovers_same_two_particle_root(captured):
    mt, preferred, target = captured
    snapshots = [value.copy() for value in captured]
    with pytest.raises(RuntimeError, match='^Cannot bracket per-sign rapidity seed$'):
        original(mt, preferred, target)
    wrapper = subject.SeedRescaler(original)
    actual = wrapper(mt, preferred, target)
    np.testing.assert_allclose(actual, [3.457760563638426, -.7499315306245303], atol=2e-13, rtol=0)
    # Independent two-body identity: M_L^2=mT1^2+mT2^2+2*mT1*mT2*cosh(delta_y).
    invariant = (target[3]-target[2])*(target[3]+target[2])
    difference = np.arccosh((invariant-mt[0]**2-mt[1]**2)/(2*mt[0]*mt[1]))
    assert actual[0]-actual[1] == pytest.approx(float(difference), abs=2e-13)
    assert_longitudinal_closure(mt, actual, target)
    for value, snapshot in zip(captured, snapshots):
        np.testing.assert_array_equal(value, snapshot)
    assert len(wrapper.repairs) == 1 and wrapper.repairs[0]['policy'] == subject.POLICY
    assert wrapper.repairs[0]['event'] is None
    assert wrapper.repairs[0]['particles'] == 2
    assert wrapper.repairs[0]['coordinate_scale'] == pytest.approx(1.7164188801377378e-6)


def test_success_returns_exact_object_without_copy_normalization_or_context_read():
    mt, preferred, target = object(), object(), object()
    returned, calls = np.array([.2, -.1]), []
    def successful(*args):
        calls.append(args)
        return returned
    def forbidden_context():
        pytest.fail('Successful legacy calls must not inspect repair context')
    wrapper = subject.SeedRescaler(successful, context=forbidden_context)
    assert wrapper(mt, preferred, target) is returned
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0], (mt, preferred, target)))
    assert wrapper.original is successful and wrapper.repairs == []


@pytest.mark.parametrize('mt,preferred,rapidities', [
    ([1., 2.], [-.2, .3], [-1.4, 2.6]),
    ([.7, 1.1, 2.3], [-1., .1, .8], [-2., .2, 1.6]),
    ([1., 2.], [1., 1.], [0., 0.]),
    ([1., 2., 0.], [-.4, .5, 0.], [-1., 2., .2]),
])
def test_real_legacy_success_is_bitwise_unchanged(mt, preferred, rapidities):
    target = target_for(mt, rapidities)
    expected = original(mt, preferred, target)
    wrapper = subject.SeedRescaler(original)
    actual = wrapper(mt, preferred, target)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    assert wrapper.repairs == []


@pytest.mark.parametrize('spread,offset', [(1e-12, 0.), (1e-11, 0.), (1e-9, 7.), (1e-7, -7.)])
def test_adversarial_tiny_but_nondegenerate_spread_preserves_root(spread, offset):
    mt = np.array([1., 2.], dtype=np.longdouble)
    preferred = np.array([offset-spread/2, offset+spread/2], dtype=np.longdouble)
    target = target_for(mt, [-1.4, 2.6])
    with pytest.raises(RuntimeError, match='^Cannot bracket per-sign rapidity seed$'):
        original(mt, preferred, target)
    wrapper = subject.SeedRescaler(original)
    result = wrapper(mt, preferred, target)
    np.testing.assert_allclose(result, [-1.4, 2.6], atol=2e-12, rtol=0)
    assert_longitudinal_closure(mt, result, target)
    assert len(wrapper.repairs) == 1


def test_retry_uses_same_original_with_longdouble_centered_positive_normalization():
    mt = np.array([.7, 1.1, 2.3], np.longdouble)
    preferred = np.array([2e-7, -5e-7, 1e-7], np.longdouble)
    target, returned = np.ones(4), np.array([.1, -.2, .3])
    calls, error = [], RuntimeError(ERROR)
    def solver(*args):
        calls.append(args)
        if len(calls) == 1:
            raise error
        return returned
    wrapper = subject.SeedRescaler(solver)
    assert wrapper(mt, preferred, target) is returned
    assert len(calls) == 2
    assert calls[0][0] is mt and calls[0][1] is preferred and calls[0][2] is target
    assert calls[1][0] is mt and calls[1][2] is target
    direction = preferred-preferred.mean()
    norm = np.max(np.abs(direction))
    assert norm > 0 and calls[1][1].dtype == np.dtype(np.longdouble)
    np.testing.assert_array_equal(calls[1][1], direction/norm)
    assert wrapper.repairs[0]['coordinate_scale'] == float(norm)
    assert wrapper.repairs[0]['old_preferred_spread'] == float(np.ptp(direction))
    assert wrapper.repairs[0]['final_rapidity_min'] == returned.min()
    assert wrapper.repairs[0]['final_rapidity_max'] == returned.max()


@pytest.mark.parametrize('error', [
    RuntimeError('different numerical failure'), RuntimeError(ERROR+' '),
    RuntimeError('prefix '+ERROR), ValueError(ERROR), ArithmeticError(ERROR),
])
def test_other_errors_are_reraised_as_the_same_exception(error):
    calls = []
    def solver(*args):
        calls.append(args)
        raise error
    wrapper = subject.SeedRescaler(solver)
    with pytest.raises(type(error)) as caught:
        wrapper([1., 2.], [-1e-6, 1e-6], [0., 0., 0., 4.])
    assert caught.value is error and len(calls) == 1 and wrapper.repairs == []


@pytest.mark.parametrize('mt,preferred', [
    ([0., 0.], [-1e-7, 1e-7]),
    ([1., 0.], [-1e-7, 1e-7]),
    ([1., 1., 0.], [0., 0., 1e-5]),
    ([1., 2.], [0., 5e-13]),
    ([1., 2.], [2., 2.]),
    ([1.], [1.]),
    ([1., -1.], [-1e-6, 1e-6]),
    ([1., np.nan], [-1e-6, 1e-6]),
    ([1., 2.], [np.inf, 1.]),
    ([1., 2.], [[-1e-6, 1e-6]]),
])
def test_degenerate_or_invalid_retry_inputs_remain_fail_closed(mt, preferred):
    calls, error = [], RuntimeError(ERROR)
    def solver(*args):
        calls.append(args)
        raise error
    wrapper = subject.SeedRescaler(solver)
    with pytest.raises(RuntimeError) as caught:
        wrapper(mt, preferred, [0., 0., 0., 4.])
    assert caught.value is error and len(calls) == 1 and wrapper.repairs == []


@pytest.mark.parametrize('mt,preferred', [
    ([0., 0.], [-1e-9, 1e-9]),
    ([1., 0.], [-1e-9, 1e-9]),
    ([1., 1., 0.], [0., 0., 1e-5]),
])
def test_real_zero_mt_support_cannot_gain_artificial_mass_freedom(mt, preferred):
    errors = []
    def solver(*args):
        try:
            return original(*args)
        except RuntimeError as error:
            errors.append(error)
            raise
    wrapper = subject.SeedRescaler(solver)
    with pytest.raises(RuntimeError, match='^Cannot bracket per-sign rapidity seed$') as caught:
        wrapper(mt, preferred, [0., 0., 0., 3.])
    assert len(errors) == 1 and caught.value is errors[0] and wrapper.repairs == []


def test_zero_mt_row_with_valid_active_ray_still_uses_same_solver():
    mt = np.array([1., 2., 0.])
    preferred = np.array([-1e-9, 1e-9, 0.])
    target = target_for(mt, [-1.4, 2.6, .6])
    result = subject.SeedRescaler(original)(mt, preferred, target)
    np.testing.assert_allclose(result, [-1.4, 2.6, .6], atol=2e-12, rtol=0)
    assert_longitudinal_closure(mt, result, target)


@pytest.mark.parametrize('mt,target,message', [
    ([1., 2.], [0., 0., 0., -4.], 'Non-future'),
    ([1., 2.], [0., 0., 4., 4.], 'Non-future'),
    ([1., 2.], [0., 0., 0., 2.], 'longitudinal mass is infeasible'),
    ([1.], [0., 0., 0., 2.], 'Singleton cannot change'),
])
def test_original_physical_feasibility_guards_do_not_route_to_retry(mt, target, message):
    errors = []
    def solver(*args):
        try:
            return original(*args)
        except RuntimeError as error:
            errors.append(error)
            raise
    wrapper = subject.SeedRescaler(solver)
    with pytest.raises(RuntimeError, match=message) as caught:
        wrapper(mt, np.linspace(-1e-6, 1e-6, len(mt)), target)
    assert len(errors) == 1 and caught.value is errors[0] and wrapper.repairs == []


def test_retry_still_enforces_original_rapidity_guard_and_records_no_false_success():
    mt = np.array([1., 2.], np.longdouble)
    target = target_for(mt, [30.1, -3.])
    errors = []
    def solver(*args):
        try:
            return original(*args)
        except RuntimeError as error:
            errors.append(error)
            raise
    wrapper = subject.SeedRescaler(solver)
    with pytest.raises(RuntimeError, match='exceeds the original rapidity guard') as caught:
        wrapper(mt, [1e-6, -1e-6], target)
    assert len(errors) == 2 and str(errors[0]) == ERROR
    assert caught.value is errors[1] and wrapper.repairs == []


def test_a0_success_is_unmodified_and_a1_repair_context_is_snapshotted(captured):
    event = dict(a=0., event_ID=4245, tag='aa510', chunk=158, design=31, nested=['before'])
    wrapper = subject.SeedRescaler(original, context=lambda: event)
    target = target_for([1., 2.], [-1., 2.])
    expected = original([1., 2.], [-.1, .2], target)
    np.testing.assert_array_equal(wrapper([1., 2.], [-.1, .2], target), expected)
    assert wrapper.repairs == []
    event['a'] = 1.
    wrapper(*captured)
    event['a'], event['nested'][0] = 0., 'after'
    np.testing.assert_array_equal(wrapper([1., 2.], [-.1, .2], target), expected)
    assert len(wrapper.repairs) == 1
    assert wrapper.repairs[0]['event'] == dict(a=1., event_ID=4245, tag='aa510', chunk=158, design=31, nested=['before'])


def test_install_wraps_only_private_geometry_and_refuses_duplicate_install(captured):
    private = SimpleNamespace(rapidity_seed=original)
    wrapper = subject.install(private, context=lambda: dict(a=1.))
    assert private.rapidity_seed is wrapper and wrapper.original is original
    assert geometry.rapidity_seed is original
    private.rapidity_seed(*captured)
    with pytest.raises(ValueError, match='already installed'):
        subject.install(private)
    assert private.rapidity_seed is wrapper
    with pytest.raises(ValueError, match='already installed'):
        subject.SeedRescaler(wrapper)


@pytest.mark.parametrize('solver,context', [(None, None), (original, {}), (original, 1)])
def test_invalid_wrapper_constructor_contract_rejected(solver, context):
    with pytest.raises(TypeError):
        subject.SeedRescaler(solver, context=context)
