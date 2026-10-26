"""Resume immutable restricted checkpoints with the declared numerical wrapper."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
POLICY = 'same-ray-rapidity-rescale-on-scale-limit-v1'
IDENTITY = 'numerical_seed_execution'
OLD = 'original-mixed-source-certificate'
PREVIOUS = 'restricted-hardest-root-success'
ORIGINAL_FIELDS = ('plan_sha256', 'config_sha256', 'collector_sha256', 'method_policy',
    'method_sources', 'native_source_sha256', 'reconstruction_sources',
    'checkpoint_policy', 'checkpoint_execution_sha256')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    if name in sys.modules:
        raise RuntimeError('Refuse a cached execution module: '+name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def numerical_identity(root, control):
    root = Path(root)
    sources = dict(control['numerical_source_sha256'])
    inventory = sha(root/'REUSE_INVENTORY.json')
    if (set(sources) != {'seed_scaling.py', 'collector_patch.py', 'original_geometry'}
            or any(sha(root/name) != sources[name] for name in ('seed_scaling.py', 'collector_patch.py'))
            or any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                   for v in sources.values())
            or inventory != control['reuse_inventory_sha256']):
        raise ValueError('Changed approved numerical sources or reuse inventory')
    return dict(policy=POLICY, execution_sha256=sha(root/'execute.py'),
                source_sha256=sources, reuse_inventory_sha256=inventory,
                previous_control_sha256=control['previous_subset_control_sha256'])


def reuse_proofs(inventory, certificate, old_control, previous_root, plan):
    selected = set(plan['original_physics_designs'])
    originals = {(p['task_id'], p['design_id']): p for p in certificate['checkpoints']
                 if p['design_id'] in selected}
    proofs, seen_originals = {}, set()
    for row in inventory['checkpoints']:
        task, design = row['task_id'], row['design_id']
        key = (task, design)
        if (type(task) is not int or task not in range(40) or type(design) is not int
                or design not in selected or key in proofs or plan['tasks'][task]['id'] != task
                or not isinstance(row['slurm_job_id'], str) or not row['slurm_job_id'].isdigit()
                or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256'])):
            raise ValueError('Invalid or duplicate certified checkpoint identity')
        if row['origin'] == OLD:
            expected = originals.get(key)
            if (expected is None or any(row.get(k) != expected.get(k) for k in (
                    'sha256', 'slurm_job_id', 'source_identity_sha256'))
                    or row['filename'] != old_control['original_remote_root']+'/'+expected['filename']):
                raise ValueError('Original checkpoint differs from the existing certificate')
            seen_originals.add(key)
        elif row['origin'] == PREVIOUS:
            chunk, = plan['tasks'][task]['chunks']
            expected = str(Path(previous_root)/f'task_{task}/results/checkpoints/'
                           f'chunk_{chunk:04d}/design_{design:02d}.npz')
            if task < 2 or key in originals or row['filename'] != expected:
                raise ValueError('Previous restricted checkpoint path/origin differs')
        else:
            raise ValueError('Unknown checkpoint execution origin')
        proofs[key] = row
    if seen_originals != set(originals) or inventory['checkpoint_count'] != len(proofs):
        raise ValueError('Incomplete immutable checkpoint inventory')
    return proofs


def check_prefixes(proofs, task=None):
    for (current, _), proof in proofs.items():
        if task is None or current == task:
            path = Path(proof['filename'])
            if path.is_symlink() or not path.is_file() or sha(path) != proof['sha256']:
                raise ValueError('Certified checkpoint missing or changed; never recompute it')


def install(cp, col, proofs, identity, previous_execution, unwrapped_load):
    """Layer after the old install; old records never acquire the new identity."""
    prior_identity, prior_restore, prior_collect = cp._identity, cp.load_design, col.collect_one
    state = dict(plan=None)

    def current_identity(*args, **kwargs):
        result = prior_identity(*args, **kwargs)
        state['plan'] = args[1] if len(args) > 1 else kwargs.get('plan')
        if result.get('hardest_root_execution') != previous_execution or IDENTITY in result:
            raise ValueError('Previous execution identity changed or numerical layer duplicated')
        return dict(result, **{IDENTITY: identity})

    def restore(path, expected, source, config):
        proof = proofs.get((expected['task']['id'], expected['design_id']))
        if proof is None:
            return prior_restore(path, expected, source, config)
        if (proof['origin'] not in (OLD, PREVIOUS) or str(path) != proof['filename']
                or Path(path).is_symlink() or sha(path) != proof['sha256']):
            raise ValueError('Certified checkpoint path/hash differs')
        old_expected = {key: value for key, value in expected.items() if key != IDENTITY}
        if old_expected.get('hardest_root_execution') != previous_execution:
            raise ValueError('Wrong previous execution during restore')
        loader = prior_restore if proof['origin'] == OLD else unwrapped_load
        counts, fingerprints, record = loader(path, old_expected, source, config)
        if IDENTITY in record or record['slurm_job_id'] != proof['slurm_job_id']:
            raise ValueError('Certified checkpoint numerical identity or original job differs')
        if proof['origin'] == PREVIOUS:
            keys = set(ORIGINAL_FIELDS)
            keys.add('hardest_root_execution')
            actual = {key: value for key, value in record.items() if key in keys or key.endswith('_execution')}
            if actual != {key: old_expected[key] for key in keys}:
                raise ValueError('Unknown or changed previous checkpoint execution identity')
        return counts, fingerprints, record

    def collect(*args, **kwargs):
        context, plan = kwargs.get('context', {}), state['plan']
        if plan is None:
            raise RuntimeError('Missing canonical plan for certified-recollection guard')
        tasks = [task for task in plan['tasks'] if task['tag'] == context.get('tag')
                 and task['chunks'] == [context.get('chunk')]]
        if len(tasks) != 1 or (tasks[0]['id'], context.get('design')) in proofs:
            raise ValueError('Unknown collection or certified checkpoint would be recomputed')
        return prior_collect(*args, **kwargs)

    cp._identity, cp.load_design, col.collect_one = current_identity, restore, collect


def load_runtime(root=HERE):
    root = Path(root)
    control = read(root/'CONTROL.json')
    previous = Path(control['previous_subset_remote_root'])
    if (sha(previous/'execute.py') != control['previous_subset_execute_sha256']
            or sha(previous/'CONTROL.json') != control['previous_subset_control_sha256']
            or any((root/name).read_bytes() != (previous/name).read_bytes() for name in ('PLAN.json', 'config.json'))):
        raise ValueError('Changed previous execution or copied plan/config')
    identity = numerical_identity(root, control)
    old = load('_seed_previous_subset_execution', previous/'execute.py')
    old_control, reference, _, cp, col, certificate, execution = old.load_runtime(root=previous)
    if execution != control['previous_hardest_root_execution']:
        raise ValueError('Actual previous hardest-root execution differs from approval')
    proofs = reuse_proofs(read(root/'REUSE_INVENTORY.json'), certificate, old_control, previous, read(root/'PLAN.json'))
    if len(proofs) != 870 or sum(row['origin'] == PREVIOUS for row in proofs.values()) != 157:
        raise ValueError('Expected exactly 713 old and 157 previous restricted checkpoints')
    unwrapped = cp.load_design
    old.install(cp, col, certificate, execution, old_control['base_native_sources'])
    install(cp, col, proofs, identity, execution, unwrapped)
    geometry = col.deform.__globals__['project'].__globals__['geometry']
    if sha(geometry.rapidity_seed.__code__.co_filename) != identity['source_sha256']['original_geometry']:
        raise ValueError('Actual loaded seed geometry differs from approval')
    patch = load('_seed_numerical_collector_patch', root/'collector_patch.py')
    patch.install(col)
    return control, old, old_control, reference, cp, col, proofs, identity


def verify_gate_fixture(root, control, col, wrapper, unresolved):
    import numpy as np
    import pyarrow.parquet as pq
    root = Path(root)
    paths = [Path(control['diagnostic_'+kind+'_path']) for kind in ('raw', 'saved')]
    pins = [control['diagnostic_'+kind+'_sha256'] for kind in ('raw', 'saved')]
    if any(not path.is_absolute() or sha(path) != digest for path, digest in zip(paths, pins)):
        raise ValueError('Changed original gate input')
    tables = [pq.read_table(path) for path in paths]
    positions, = np.nonzero(tables[0]['event_ID'].to_numpy() == 4245)
    if len(positions) != 1 or tables[0].num_rows != tables[1].num_rows:
        raise ValueError('Missing/duplicate real gate event or unmatched source rows')
    fixtures = root/'gate_fixture'
    fixtures.mkdir(exist_ok=True)
    fixture_paths = []
    for kind, table in zip(('raw', 'saved'), tables):
        path, row = fixtures/(kind+'.parquet'), table.slice(int(positions[0]), 1)
        if not path.exists():
            with path.open('xb') as stream:
                pq.write_table(row, stream)
        if not pq.read_table(path).equals(row):
            raise ValueError('One-row gate fixture changed original values')
        fixture_paths.append(path)
    before = col.method_sources()
    counts, fingerprints, report = col.collect_one(*fixture_paths,
        read(root/'config.json'), 'PbPb', [5, 10], context=dict(tag='aa510', chunk=158, design=31),
        wrapper=wrapper, unresolved=unresolved)
    repairs = report['numerical_seed_repairs']
    event = dict(tag='aa510', chunk=158, design=31, event_ID=4245, a=1.)
    if (counts.shape != (1, 2, 172) or fingerprints.shape != (1, 5) or fingerprints[0, 0] != 4245
            or not np.isfinite(counts).all() or np.any(counts[..., -1] != 1)
            or len(repairs) != 1 or repairs[0]['event'] != event or repairs[0]['policy'] != POLICY
            or report['numerical_seed_policy'] != dict(policy=POLICY, source_sha256=control['numerical_source_sha256'])
            or report['native_event_node_checks'] != 2 or report['strict_transformed_checks'] != 1
            or any(report.get(key) is not True for key in ('hadron_baseline_exact', 'baseline_a0_identity_exact', 'no_events_dropped'))
            or any(report['qa'][str(a)]['whole_event_fallback'] != 0 for a in (0., 1.))
            or wrapper.context is not None or before != col.method_sources()
            or any(sha(path) != digest for path, digest in zip(paths, pins))):
        raise ValueError('Real collector numerical/native gate failed')
    print('NUMERICAL_SEED_COLLECTOR_GATE_PASS', flush=True)


def complete_gate(root, proofs):
    root = Path(root)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or len(proofs) != 870:
        raise ValueError('Gate requires its actual raw allocation and all 870 checkpoints')
    record = dict(status='SEED_SCALING_GATE_PASS', slurm_job_id=job,
                  control_sha256=sha(root/'CONTROL.json'), checkpoint_count=len(proofs))
    with (root/'GATE_COMPLETE.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    print('SEED_SCALING_GATE_PASS', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'gate', 'collect'))
    parser.add_argument('--task', type=int)
    args = parser.parse_args()
    control, old, old_control, reference, cp, col, proofs, identity = load_runtime()
    check_prefixes(proofs, None if args.action == 'gate' else args.task)
    plan = read(HERE/'PLAN.json')
    if args.action in ('check', 'gate'):
        wrapper, unresolved = col.install_reconstruction(reference)
        cp._identity(col, plan, HERE/'PLAN.json', HERE/'config.json', wrapper, reference)
        print('NUMERICAL_SEED_IMPORTS_PASS', json.dumps(identity, sort_keys=True), flush=True)
        if args.action == 'gate':
            subprocess.run([sys.executable, str(Path(control['previous_subset_remote_root'])/'execute.py'),
                            'fixtures'], check=True)
            verify_gate_fixture(HERE, control, col, wrapper, unresolved)
            complete_gate(HERE, proofs)
    else:
        if args.task not in range(2, 40):
            raise ValueError('Only canonical remaining tasks 2 through 39 may collect')
        paths = {design: Path(row['filename']) for (task, design), row in proofs.items() if task == args.task}
        cp.run_task(args.task, reference, HERE/f'task_{args.task}/results', plan_path=HERE/'PLAN.json',
                    config_path=HERE/'config.json', collector=col, checkpoint_paths=paths)


if __name__ == '__main__':
    main()
