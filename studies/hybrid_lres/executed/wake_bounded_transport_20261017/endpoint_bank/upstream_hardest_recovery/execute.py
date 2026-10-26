"""Original checkpoint collector with explicit mixed-origin hardest-root reuse."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
POLICY = 'fastjet-keep-only-hardest-v1'
IDENTITY = 'hardest_root_execution'
ORIGINAL_FIELDS = (
    'plan_sha256', 'config_sha256', 'collector_sha256', 'method_policy',
    'method_sources', 'native_source_sha256', 'reconstruction_sources',
    'checkpoint_policy', 'checkpoint_execution_sha256')
OLD_EXTENSION = 'signed_root_execution'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def checkpoint_proofs(certificate):
    identities = certificate['identities']
    for digest, origin in identities.items():
        if (identity_digest(origin) != digest
                or set(origin) not in (set(ORIGINAL_FIELDS), set(ORIGINAL_FIELDS) | {OLD_EXTENSION})
                or (OLD_EXTENSION in origin and origin[OLD_EXTENSION].get('policy')
                    != 'unique-leading-positive-kt-root-v1')):
            raise ValueError('Invalid original checkpoint identity')
    allowed = {}
    for proof in certificate['checkpoints']:
        task, design = proof['task_id'], proof['design_id']
        key = (task, design)
        if (type(task) is not int or task not in range(40)
                or type(design) is not int or design not in range(40)
                or key in allowed or proof['source_identity_sha256'] not in identities):
            raise ValueError('Duplicate or invalid checkpoint certificate')
        allowed[key] = proof
    if certificate.get('checkpoint_count', len(allowed)) != len(allowed):
        raise ValueError('Incomplete checkpoint certificate')
    return allowed


def load_file(name, path):
    if name in sys.modules:
        raise RuntimeError('Refuse a cached module: '+name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != Path(path).resolve():
        raise RuntimeError('Unexpected module origin')
    return module


def load_runtime(root=HERE):
    control = json.loads((root/'CONTROL.json').read_text())
    runtime, reference = Path(control['prepared_runtime']), Path(control['reference'])
    receipt = json.loads((root/'NATIVE.json').read_text())
    source, native = root/'dr12_policy.cc', root/'native'/receipt['filename']
    if (native.parent.resolve() != (root/'native').resolve()
            or native.is_symlink() or sha(source) != control['policy_source_sha256']
            or receipt['source_sha256'] != sha(source) or sha(native) != receipt['sha256']):
        raise ValueError('Unpinned hardest-root source/native')
    diagnostic = load_file('_wake_dr12_diagnostic', native)
    helper = runtime/'wake_bounded_transport_20261017/bank/checkpoint_collect.py'
    if sha(helper) != control['checkpoint_execution_sha256']:
        raise ValueError('Changed original checkpoint collector')
    cp = load_file('_hardest_root_original_checkpoint', helper)
    col = cp.load_collector()
    certificate_path = root/'COMPATIBILITY.json'
    certificate = json.loads(certificate_path.read_text())
    if (sha(certificate_path) != control['compatibility_sha256']
            or certificate['policy'] != POLICY or len(checkpoint_proofs(certificate)) != 937
            or certificate['base_native_sources'] != control['base_native_sources']):
        raise ValueError('Changed mixed-origin compatibility certificate')
    extension = dict(policy=POLICY, bootstrap_sha256=sha(__file__),
        policy_source_sha256=sha(source), native_sha256=sha(native),
        compatibility_sha256=sha(certificate_path), control_sha256=sha(root/'CONTROL.json'))
    return control, reference, diagnostic, cp, col, certificate, extension


def install(cp, col, certificate, extension, base_native_sources):
    """Restore certified original records without relabelling either old origin."""
    original_identity, original_load = cp._identity, cp.load_design
    original_install, original_collect = col.install_reconstruction, col.collect_one
    allowed = checkpoint_proofs(certificate)
    origins = certificate['identities']
    state = dict(audit=[], reused={}, plan=None)

    def compatible(origin, current):
        if any(origin[key] != current[key] for key in ORIGINAL_FIELDS
               if key != 'native_source_sha256'):
            raise ValueError('Old checkpoint plan/config/method differs from current source')
        native = origin['native_source_sha256']
        if (set(native) != set(base_native_sources)
                or any(native[key] != value for key, value in base_native_sources.items()
                       if key != 'wake_dr12_diagnostic')):
            raise ValueError('Old checkpoint changed a non-dR12 native backend')

    def identity(*args, **kwargs):
        result = original_identity(*args, **kwargs)
        state['plan'] = args[1] if len(args) > 1 else kwargs.get('plan')
        expected_native = dict(base_native_sources, wake_dr12_diagnostic=extension['native_sha256'])
        if result['native_source_sha256'] != expected_native:
            raise RuntimeError('Unexpected changed native backend')
        for origin in origins.values():
            compatible(origin, result)
        return dict(result, **{IDENTITY: extension})

    def restore(path, expected, source, config):
        key = (expected['task']['id'], expected['design_id'])
        if key not in allowed:
            return original_load(path, expected, source, config)
        proof = allowed[key]
        if Path(path).is_symlink() or sha(path) != proof['sha256']:
            raise ValueError('Old immutable checkpoint differs from certificate')
        origin = origins[proof['source_identity_sha256']]
        compatible(origin, expected)
        old_expected = {key: expected[key] for key in ('status', 'task', 'chunk', 'design_id')}
        old_expected.update(origin)
        counts, fingerprints, record = original_load(path, old_expected, source, config)
        actual = {key: record[key] for key in ORIGINAL_FIELDS}
        if OLD_EXTENSION in record:
            actual[OLD_EXTENSION] = record[OLD_EXTENSION]
        if (IDENTITY in record or actual != origin or record['slurm_job_id'] != proof['slurm_job_id']):
            raise ValueError('Old execution identity differs from certificate')
        state['reused'][key] = dict(proof)
        return counts, fingerprints, record

    def reconstruction(reference):
        wrapper, unresolved = original_install(reference)
        original = col.pilot.dr._atlas_dr12.reconstruct

        def audited(*args, **kwargs):
            result = original(*args, **kwargs)
            for row in result['remnant_audit']:
                if wrapper.context is None:
                    raise RuntimeError('Root audit has no source-event identity')
                required = ('selected_root_pt', 'selected_root_sign', 'selected_root_p4',
                            'selected_root_m2', 'equal_pt_roots', 'remnants')
                if any(key not in row for key in required):
                    raise ValueError('Incomplete general hardest-root audit')
                state['audit'].append(dict(event=dict(wrapper.context), **row))
            return result

        col.pilot.dr._atlas_dr12.reconstruct = audited
        return wrapper, unresolved

    def collect(*args, **kwargs):
        context = kwargs.get('context', {})
        if allowed:
            if state['plan'] is None:
                raise RuntimeError('Missing canonical plan for forbidden-recollection guard')
            tasks = [task for task in state['plan']['tasks']
                     if task['tag'] == context.get('tag') and task['chunks'] == [context.get('chunk')]]
            if len(tasks) != 1:
                raise ValueError('Collection has no unique canonical task context')
            if (tasks[0]['id'], context.get('design')) in allowed:
                raise ValueError('Certified old checkpoint cannot be recomputed')
        state['audit'].clear()
        counts, fingerprints, report = original_collect(*args, **kwargs)
        valid_ids = set(fingerprints[:, 0])
        if any(row['event']['event_ID'] not in valid_ids for row in state['audit']):
            raise ValueError('Root audit references an unknown event')
        report['dr12_hardest_root_policy'] = POLICY
        report['dr12_hardest_root_audit'] = list(state['audit'])
        report['dr12_hardest_root_audit_scope'] = 'all reconstructed jets; join weights by event_ID'
        return counts, fingerprints, report

    cp._identity, cp.load_design = identity, restore
    col.install_reconstruction, col.collect_one = reconstruction, collect
    return state


def check_prefixes(root, certificate, plan, task=None):
    """A certified old result must never fall through to new-event collection."""
    for (current, design), proof in checkpoint_proofs(certificate).items():
        if task is not None and current != task:
            continue
        canonical = plan['tasks'][current]
        if canonical['id'] != current:
            raise ValueError('Noncanonical plan task ordering')
        chunk, = canonical['chunks']
        expected = f'task_{current}/results/checkpoints/chunk_{chunk:04d}/design_{design:02d}.npz'
        if proof['filename'] != expected:
            raise ValueError('Certified checkpoint path differs from canonical task')
        path = Path(root)/expected
        if path.is_symlink() or not path.is_file() or sha(path) != proof['sha256']:
            raise ValueError('Certified old checkpoint missing or changed; never recompute it')


def verify_fixtures(control, reference, root):
    import numpy as np
    import _wake_dr12_diagnostic as native
    keys = {'rows_a0p0', 'rows_a0p25', 'rows_a0p5', 'rows_a0p75', 'rows_a1p0'}
    for event in (3287, 4409):
        with np.load(root/f'diagnostic_rows_{event}.npz', allow_pickle=False) as archive:
            if set(archive.files) != keys:
                raise ValueError('Incomplete five-node diagnostic archive')
            for key in archive.files:
                result = native.reconstruct(archive[key])
                if any(200 <= row[0] < 251 for row in result['unresolved']):
                    raise ValueError(f'Diagnosed selected jet still unresolved: {event}/{key}')
                exercise = (event == 3287 and key in ('rows_a0p5', 'rows_a1p0')) or (
                    event == 4409 and key == 'rows_a1p0')
                if exercise and not any(200 <= row['jet_pt'] < 251 for row in result['remnant_audit']):
                    raise ValueError(f'Hardest-root policy was not exercised: {event}/{key}')
                if event == 4409 and key == 'rows_a1p0':
                    if (any(200 <= row[0] < 251 for row in result['jets'])
                            or result['cutflow']['grooming_failed'] < 1):
                        raise ValueError('Event 4409 must be an ordinary grooming failure')
    script = Path(control['prepared_runtime'])/'wake_bounded_transport_20261017/bank/verify_runtime.py'
    previous = sys.argv
    try:
        sys.argv = [str(script), '--fixtures', str(script.with_name('runtime_cases.npz')),
                    '--reference', str(reference)]
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous
    print('HARDEST_ROOT_FIXTURES_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'fixtures', 'collect'))
    parser.add_argument('--task', type=int)
    args = parser.parse_args()
    control, reference, diagnostic, cp, col, certificate, extension = load_runtime()
    plan = json.loads((cp.HERE/'PLAN.json').read_text())
    check_prefixes(control['original_remote_root'], certificate, plan, args.task)
    install(cp, col, certificate, extension, control['base_native_sources'])
    if args.action == 'check':
        wrapper, _ = col.install_reconstruction(reference)
        cp._identity(col, plan, cp.HERE/'PLAN.json', cp.HERE/'config.json', wrapper, reference)
        expected = dict(control['base_native_sources'], wake_dr12_diagnostic=extension['native_sha256'])
        if col.native_sources(wrapper, reference) != expected:
            raise ValueError('Loaded native origins differ from pinned runtime')
        print('HARDEST_ROOT_IMPORTS_PASS', json.dumps(extension, sort_keys=True), flush=True)
    elif args.action == 'fixtures':
        verify_fixtures(control, reference, HERE)
    else:
        if args.task not in range(2, 40):
            raise ValueError('Only the remaining canonical tasks may run')
        output = Path(control['original_remote_root'])/f'task_{args.task}'/'results'
        cp.run_task(args.task, reference, output, collector=col)


if __name__ == '__main__':
    main()
