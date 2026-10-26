"""Preflight the packaged bounded method, optionally using recorded native cases."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
METHOD_POLICY = 'bounded-separate-sign-v1'
CORE_SHA256 = 'cfcc831231bfefbd12585b2dc2e5d93df6bcce7024a5b87228a482d0357eba8f'
NODES = (0., .25, .5, .75, 1.)
METHOD_PATHS = (
    'wake_bounded_transport_20261017/transport.py',
    'wake_bounded_transport_20261017/bounded_projection.py',
    'wake_bounded_transport_20261017/thinning.py',
    'wake_bounded_transport_20261017/bank/collect_bounded.py',
    'wake_bounded_transport_20261017/bank/verify_runtime.py',
    'wake_sign_separated_20261017/projection.py',
    'wake_nuisance_20261017/global_a.py',
    'wake_nuisance_20261017/collect_gate.py',
    'wake_nuisance_20261017/full_bank/collect_full.py',
    'wake_nuisance_20261017/full_bank/compact.py',
    'wake_nuisance_20261017/full_bank/storage.py',
    'wake_paper_morph_20261017/transport.py',
    'wake_paper_morph_20261017/CALIBRATION.json',
    'wake_paper_morph_20261017/calibrate.py',
    'wake_sensitivity_20261017/deform.py',
    'wake_sensitivity_20261017/run_study.py',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def method_sources():
    """Portable runtime-root-relative paths, never machine-specific paths."""
    result = {name: sha(WORK / name) for name in METHOD_PATHS}
    if result['wake_bounded_transport_20261017/bounded_projection.py'] != CORE_SHA256:
        raise RuntimeError('Bounded projection differs from the reviewed source pin')
    return result


def load_bounded():
    # The legacy code uses bare imports. Isolate its dependencies explicitly so
    # another collector's cached transport/projection cannot silently be reused.
    dependencies = (
        ('deform', 'wake_sensitivity_20261017/deform.py'),
        ('calibrate', 'wake_paper_morph_20261017/calibrate.py'),
        ('global_a', 'wake_nuisance_20261017/global_a.py'),
        ('bounded_projection', 'wake_bounded_transport_20261017/bounded_projection.py'),
        ('thinning', 'wake_bounded_transport_20261017/thinning.py'),
    )
    previous = {name: sys.modules.get(name) for name, _ in dependencies}
    old_path = sys.path[:]
    try:
        sys.path[:0] = [str(WORK / 'wake_paper_morph_20261017'),
                       str(WORK / 'wake_sensitivity_20261017')]
        for name, relative in dependencies:
            load(name, WORK / relative)
        bounded = load('_bounded_bank_transport', HERE.parent / 'transport.py')
        if Path(bounded.project.__code__.co_filename).resolve() != (HERE.parent / 'bounded_projection.py').resolve():
            raise RuntimeError('Bounded transport imported a different projection')
        if Path(bounded.project.__globals__['geometry'].__file__).resolve() != (
                WORK / 'wake_sign_separated_20261017/projection.py').resolve():
            raise RuntimeError('Bounded transport imported a different geometry')
        # Populate lazy canonical-node calibration caches while the portable
        # dependency bindings are active, not a previous task's bare imports.
        for a in NODES:
            bounded.description(a)
            model = bounded.description.__globals__['_transport'](a).Model
            if Path(model.amplitudes.__code__.co_filename).resolve() != (
                    WORK / 'wake_paper_morph_20261017/calibrate.py').resolve():
                raise RuntimeError('Bounded transport imported a different calibration')
        return bounded
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def synthetic_event(singleton=False):
    pt = np.linspace(.4, 1.8, 8)
    four = np.column_stack((pt, np.zeros((8, 2)), np.sqrt(pt**2 + .13957**2)))
    four = np.vstack(([10., 0., 0., 10.], four))
    pid, status = [21] + [211, -211]*4, [-2] + [1]*8
    if singleton:
        forward = np.array([.8, 0., 2*np.sinh(5.935), 2*np.cosh(5.935)], np.float32).astype(float)
        four = np.vstack((four, forward))
        pid.append(211)
        status.append(2)
    event = dict(event_ID=321 + int(singleton), event_weight=1., pt_hat=40.,
                 vertex_x=.1, vertex_y=.2, centrality=2., cross_section=1.,
                 particle_ID=np.array(pid), status=np.array(status))
    event.update({key: four[:, i] for i, key in enumerate(('px', 'py', 'pz', 'E'))})
    return event


def check_event(bounded, event, a):
    observed = []
    modified, checks = bounded.deform(event, a, observer=observed.append)
    if checks['whole_event_fallback']:
        raise AssertionError('Undeclared whole-event fallback')
    if a == 0:
        if modified is not event or checks['baseline_identity'] != 1:
            raise AssertionError('Original a=0 is not exact identity')
    before, after = map(bounded.REFERENCE.four_vectors, (event, modified))
    status, final_status = map(np.asarray, (event['status'], modified['status']))
    pid, final_pid = map(np.asarray, (event['particle_ID'], modified['particle_ID']))
    if not np.isfinite(after).all():
        raise AssertionError('Nonfinite final four-vector')
    wake, final_wake = np.isin(status, (1, 2)), np.isin(final_status, (1, 2))
    original_signed = (before[wake]*np.where(status[wake] == 1, 1, -1)[:, None]).sum(0)
    final_signed = (after[final_wake]*np.where(final_status[final_wake] == 1, 1, -1)[:, None]).sum(0)
    np.testing.assert_allclose(final_signed, original_signed, atol=2e-7, rtol=2e-10)
    for sign in (1, 2):
        old_energy = before[status == sign, 3].sum()
        energy = after[final_status == sign, 3].sum()
        if energy > old_energy + 2e-7 + 2e-10*abs(old_energy):
            raise AssertionError('Unsigned sign energy increased')
        for species in np.unique(abs(pid)):
            if species in (22, 111, 221, 331):
                continue
            old_charge = np.count_nonzero((status == sign) & (pid == species)) - np.count_nonzero((status == sign) & (pid == -species))
            charge = np.count_nonzero((final_status == sign) & (final_pid == species)) - np.count_nonzero((final_status == sign) & (final_pid == -species))
            if old_charge != charge:
                raise AssertionError('Species charge changed')
    if observed:
        info, = observed
        selected = info['selected_original_indices']
        eligible = wake & (bounded.REFERENCE.mass2(before) >= 0)
        eligible &= ~((np.hypot(before[:, 0], before[:, 1]) == 0) & (bounded.REFERENCE.mass2(before) == 0))
        alive = ~eligible
        alive[selected] = True
        indices = np.flatnonzero(alive)
        np.testing.assert_array_equal(final_pid, pid[indices])
        np.testing.assert_array_equal(final_status, status[indices])
        unchanged = ~np.isin(indices, selected)
        np.testing.assert_array_equal(after[unchanged], before[indices[unchanged]])
        np.testing.assert_allclose(bounded.REFERENCE.mass2(after), bounded.REFERENCE.mass2(before[indices]), atol=2e-7, rtol=1e-7)
        projected = after[np.searchsorted(indices, selected)]
        pt, caps = np.hypot(projected[:, 0], projected[:, 1]), info['pt_caps']
        if np.any(pt > caps*(1+1e-12)+1e-12):
            raise AssertionError('Particle exceeds the paper pT envelope')
        for sign, target in zip((1, 2), info['targets_lab']):
            np.testing.assert_allclose(projected[status[selected] == sign].sum(0), target, atol=2e-7, rtol=2e-10)
    else:
        np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(final_pid, pid)
        np.testing.assert_array_equal(final_status, status)
    return modified, checks


def read_fixtures(path):
    with np.load(path, allow_pickle=False) as data:
        required = ('events_json', 'labels', 'input_sha256', 'event_ids')
        if not all(name in data.files for name in required):
            raise ValueError('Fixtures require events_json, labels, input_sha256, event_ids')
        strings, labels, inputs, ids = [data[key] for key in required]
        if (any(x.ndim != 1 for x in (strings, labels, inputs, ids)) or not len(strings)
                or any(len(x) != len(strings) for x in (labels, inputs, ids))
                or any(x.dtype.kind != 'U' for x in (strings, labels, inputs))
                or ids.dtype != np.dtype('int64') or len(set(labels.tolist())) != len(labels)):
            raise ValueError('Malformed or duplicated runtime fixture identities')
        records = []
        for text, label, source, eid in zip(strings, labels, inputs, ids):
            if len(source) != 64 or any(c not in '0123456789abcdef' for c in source):
                raise ValueError('Invalid fixture input SHA256')
            event = json.loads(str(text))
            if event['event_ID'] != int(eid):
                raise ValueError('Fixture event_ID mismatch')
            for key in ('px', 'py', 'pz', 'E', 'particle_ID', 'status'):
                event[key] = np.asarray(event[key])
            records.append((str(label), str(source), event))
        return records


def verify(fixtures=None, reference=None):
    if (fixtures is None) != (reference is None):
        raise ValueError('--fixtures and --reference must be supplied together')
    before = method_sources()
    bounded = load_bounded()
    synthetic = []
    for singleton in (False, True):
        event = synthetic_event(singleton)
        checks = [check_event(bounded, event, a)[1] for a in NODES]
        synthetic.append(dict(case='float32_singleton' if singleton else 'collinear_certificate',
                              nodes=list(NODES), maxima={key: max(q[key] for q in checks)
                              for key in ('signed_p4_residual', 'mass2_residual', 'pt_envelope_ratio')}))
    report = dict(status='BOUNDED_RUNTIME_PASS', method_policy=METHOD_POLICY,
                  method_sources=before, synthetic_cases=synthetic, native_cases=[],
                  native_event_node_checks=0, strict_transformed_checks=0,
                  native_source_sha256={})
    if fixtures is not None:
        fixture_sha = sha(fixtures)
        cases = read_fixtures(fixtures)
        full = load('_bounded_runtime_collector', HERE / 'collect_bounded.py')
        config_path = HERE / 'config.json'
        if not config_path.exists():
            config_path = WORK / 'wake_nuisance_20261017/gate_config.json'
        config = json.loads(config_path.read_text())
        config_sha = sha(config_path)
        old_dr, old_backend = full.pilot.dr, full.pilot.rg._upstream_backend
        try:
            wrapper, unresolved = full.install_reconstruction(Path(reference).resolve())
            native_before = full.native_sources(wrapper, reference)
            for label, input_sha, event in cases:
                checks = []
                outside = []
                for a in NODES:
                    wrapper.context = dict(fixture=label, event_ID=int(event['event_ID']), a=a)
                    unresolved.clear()
                    modified, qa = check_event(bounded, event, a)
                    vector, _ = full.pilot.reconstruction(modified, 'PbPb', config['jet_blocks'])
                    if (not np.isfinite(vector).all() or np.any(vector != np.rint(vector)) or vector[-1] != 1):
                        raise AssertionError('Invalid native runtime count vector')
                    checks.append(qa)
                    if unresolved:
                        outside.append(dict(a=a, jets=list(unresolved)))
                    report['native_event_node_checks'] += 1
                report['native_cases'].append(dict(label=label, input_sha256=input_sha,
                    event_ID=int(event['event_ID']), nodes=list(NODES),
                    maxima={key: max(q[key] for q in checks) for key in
                            ('signed_p4_residual', 'mass2_residual', 'pt_envelope_ratio')},
                    unresolved_dr12_outside_fit=outside))
            expected = 4*len(cases)
            if wrapper.audit.get('strict_transformed_successes', 0) != expected:
                raise AssertionError('Incomplete strict transformed native coverage')
            if native_before != full.native_sources(wrapper, reference):
                raise RuntimeError('Native policy source changed during preflight')
            report.update(strict_transformed_checks=expected, native_audit=dict(wrapper.audit),
                          original_a0_precision_events=wrapper.events,
                          native_source_sha256=native_before, fixtures_sha256=fixture_sha,
                          reconstruction_sources=full.reconstruction_sources(wrapper, reference),
                          config_sha256=config_sha)
        finally:
            full.pilot.dr, full.pilot.rg._upstream_backend = old_dr, old_backend
        if fixture_sha != sha(fixtures) or config_sha != sha(config_path):
            raise RuntimeError('Runtime fixtures/config changed during preflight')
    if before != method_sources():
        raise RuntimeError('Method source changed during preflight')
    report['sources_unchanged'] = True
    report['limitation'] = 'Runtime regression only; not bank coverage or physical calibration.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify(args.fixtures, args.reference)
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        with args.output.open('x') as stream:
            stream.write(encoded)
    print(encoded, end='')
    print('BOUNDED_RUNTIME_PASS', flush=True)


if __name__ == '__main__':
    main()
