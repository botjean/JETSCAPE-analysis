"""Local production-shaped wiring to real bounded projection; no native/jobs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = load('_collector_patch_tests', HERE/'collector_patch.py')


def deform_template(event, a):
    if a == 0:
        return event
    return project(event['target'], event['caps'], event['masses2'], event['preferred'], event['phi'])


@pytest.fixture
def bank():
    projection = load('_collector_patch_private_bounded_geometry', WORK/'wake_bounded_transport_20261017/bounded_projection.py')
    mt, y, phi = np.array([1., 2.]), np.array([2.5, -1.2]), np.array([0., np.pi])
    target = np.array([(mt*np.cos(phi)).sum(), (mt*np.sin(phi)).sum(),
                       (mt*np.sinh(y)).sum(), (mt*np.cosh(y)).sum()])
    event = dict(event_ID=4245, caps=mt, masses2=np.zeros(2), target=target,
                 preferred=np.array([1e-6, -1e-6]), phi=phi)
    col = SimpleNamespace(deform=FunctionType(deform_template.__code__, {'project': projection.project}))
    fixture = SimpleNamespace(col=col, projection=projection, event=event, inject=lambda ctx: ctx,
        installed=[], calls=[], error=None, context=dict(tag='aa510', chunk=158, design=31),
        config=dict(a_nodes=[0., 1.], design_ids=[31, 32]))
    fixture.counts = np.ones((1, 2, 172), dtype=np.int16)
    fixture.fingerprints = np.array([[4245, 1., 2., .3, .4]])
    fixture.report = dict(events=1, hadron_baseline_exact=True, guard_token=object())

    def reconstruction(reference):
        result = (SimpleNamespace(context=None), [])
        fixture.installed.append((reference, result))
        return result

    def collect(raw, saved, config, system, centrality, *, context, wrapper, unresolved):
        if fixture.error is not None:
            raise fixture.error
        for a in config['a_nodes']:
            # This is the ordering in the unchanged production collect_one.
            wrapper.context = fixture.inject(dict(context, event_ID=fixture.event['event_ID'], a=a))
            result = col.deform(fixture.event, a)
            fixture.calls.append((deepcopy(wrapper.context), result))
        wrapper.context = None
        return fixture.counts, fixture.fingerprints, fixture.report

    col.install_reconstruction, col.collect_one = reconstruction, collect
    fixture.run = lambda wrapper: col.collect_one('raw', 'saved', fixture.config, 'PbPb', [5, 10],
        context=fixture.context, wrapper=wrapper, unresolved=[])
    return fixture


def test_real_project_uses_live_context_before_deform_and_adds_source_proof(bank):
    original_seed = bank.projection.geometry.rapidity_seed
    checksum = subject.sha(original_seed.__code__.co_filename)
    with pytest.raises(RuntimeError, match='Cannot bracket per-sign rapidity seed'):
        bank.col.deform(bank.event, 1.)
    repair = subject.install(bank.col)
    returned = bank.col.install_reconstruction('reference')
    assert returned is bank.installed[-1][1]
    wrapper, _ = returned
    counts, identity, report = bank.run(wrapper)
    assert counts is bank.counts and identity is bank.fingerprints
    assert wrapper.context is None and len(repair.repairs) == 1
    assert report['numerical_seed_repairs'][0]['event'] == dict(bank.context, event_ID=4245, a=1.)
    assert report['numerical_seed_policy'] == dict(policy=subject.seed_scaling.POLICY, source_sha256={
        'collector_patch.py': subject.sha(HERE/'collector_patch.py'),
        'seed_scaling.py': subject.sha(HERE/'seed_scaling.py'), 'original_geometry': checksum})
    assert set(bank.report) == {'events', 'hadron_baseline_exact', 'guard_token'}
    assert report['guard_token'] is bank.report['guard_token']
    assert bank.calls[0][1] is bank.event
    four, checks = bank.calls[1][1]
    np.testing.assert_allclose(four.sum(0), bank.event['target'], atol=2e-8, rtol=2e-12)
    np.testing.assert_allclose(np.hypot(four[:, 0], four[:, 1]), bank.event['caps'], atol=1e-12, rtol=1e-12)
    assert checks['pt_damping'] == 1.
    assert subject.sha(original_seed.__code__.co_filename) == checksum


def test_original_success_outputs_equal_and_report_objects_not_mutated(bank):
    bank.event['preferred'] = np.array([.2, -.3])
    wrapper, _ = bank.col.install_reconstruction('reference')
    expected = bank.run(wrapper)
    original_four = bank.calls[1][1][0].copy()
    repair = subject.install(bank.col)
    wrapper, _ = bank.col.install_reconstruction('reference')
    actual = bank.run(wrapper)
    assert actual[0] is expected[0] and actual[1] is expected[1]
    for key, value in expected[2].items():
        assert actual[2][key] is value
    np.testing.assert_array_equal(bank.calls[-1][1][0], original_four)
    assert repair.repairs == actual[2]['numerical_seed_repairs'] == []


def test_per_collection_reset_and_old_reports_are_independent_snapshots(bank):
    repair = subject.install(bank.col)
    wrapper, _ = bank.col.install_reconstruction('reference')
    first = bank.run(wrapper)[2]
    snapshot = deepcopy(first['numerical_seed_repairs'])
    bank.event['preferred'] = np.array([.2, -.3])
    bank.context['design'] = 32
    second = bank.run(wrapper)[2]
    assert repair.repairs == second['numerical_seed_repairs'] == []
    assert first['numerical_seed_repairs'] == snapshot
    bank.event['preferred'] = np.array([1e-6, -1e-6])
    third = bank.run(wrapper)[2]
    assert len(third['numerical_seed_repairs']) == 1
    assert third['numerical_seed_repairs'][0]['event']['design'] == 32
    assert first['numerical_seed_repairs'][0]['event']['design'] == 31


def test_duplicate_install_rejected_without_replacing_entrypoints(bank):
    repair = subject.install(bank.col)
    entrypoints = bank.col.install_reconstruction, bank.col.collect_one
    with pytest.raises(ValueError, match='already installed'):
        subject.install(bank.col)
    assert entrypoints == (bank.col.install_reconstruction, bank.col.collect_one)
    assert bank.projection.geometry.rapidity_seed is repair


def test_latest_actual_reconstruction_wrapper_is_required(bank):
    subject.install(bank.col)
    old, _ = bank.col.install_reconstruction('old')
    current, _ = bank.col.install_reconstruction('current')
    with pytest.raises(RuntimeError, match='canonical live wrapper'):
        bank.run(old)
    report = bank.run(current)[2]
    assert report['numerical_seed_repairs'][0]['event']['event_ID'] == 4245


@pytest.mark.parametrize('change', ['missing', 'tag', 'chunk', 'design', 'a', 'event_type', 'unknown_event'])
def test_missing_or_unknown_repair_event_identity_is_fatal(bank, change):
    subject.install(bank.col)
    wrapper, _ = bank.col.install_reconstruction('reference')
    def changed(context):
        if change == 'missing': return None
        key, value = dict(tag=('tag', 'aa05'), chunk=('chunk', 7), design=('design', 39),
            a=('a', .5), event_type=('event_ID', 4245.), unknown_event=('event_ID', 999))[change]
        return dict(context, **{key: value})
    bank.inject = changed
    with pytest.raises(RuntimeError, match='identity|unknown returned event'):
        bank.run(wrapper)


def test_original_failure_forwarded_unchanged_and_next_collection_is_clean(bank):
    repair = subject.install(bank.col)
    wrapper, _ = bank.col.install_reconstruction('reference')
    bank.run(wrapper)
    bank.error = RuntimeError('original conservation/native guard')
    with pytest.raises(RuntimeError) as caught:
        bank.run(wrapper)
    assert caught.value is bank.error and repair.repairs == []
    bank.error = None
    assert len(bank.run(wrapper)[2]['numerical_seed_repairs']) == 1


def test_repair_outside_collection_cannot_acquire_stale_event_context(bank):
    subject.install(bank.col)
    wrapper, _ = bank.col.install_reconstruction('reference')
    wrapper.context = dict(bank.context, event_ID=4245, a=1.)
    with pytest.raises(RuntimeError, match='active source-event identity'):
        bank.col.deform(bank.event, 1.)


def test_unknown_geometry_routing_rejected_without_collector_changes(bank):
    old = bank.col.install_reconstruction, bank.col.collect_one
    bank.col.deform = lambda *args: None
    with pytest.raises(ValueError, match='routing'):
        subject.install(bank.col)
    assert old == (bank.col.install_reconstruction, bank.col.collect_one)
