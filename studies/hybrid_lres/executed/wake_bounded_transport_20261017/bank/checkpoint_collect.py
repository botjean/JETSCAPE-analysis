"""Paired a=0/a=1 checkpoints around the unchanged bounded-bank collector."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
POLICY = 'per-design-atomic-v1'
COLLECTOR = 'wake_bounded_transport_20261017/bank/collect_bounded.py'
TRANSPORT = 'wake_bounded_transport_20261017/transport.py'
NODES = [0., 1.]
CHUNKS = [7, 12, 23, 35, 46, 54, 62, 74, 86, 90, 104, 117, 128, 130, 143, 158, 164, 177, 180, 198]
RETENTION_ERROR = 'Upstream C/A reclustering lost jet constituents'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_collector():
    """Use the packaged original by file, never a cached bare module name."""
    path = HERE/'collect_bounded.py'
    if not path.exists():
        path = HERE.parent/'bank/collect_bounded.py'
    spec = importlib.util.spec_from_file_location('_endpoint_original_bounded', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError('Wrong original collector import')
    module.runtime.NODES = tuple(NODES)
    return module


def _scope(plan, config, task_id):
    if (config.get('design_ids') != list(range(40))
            or any(type(i) is not int for i in config['design_ids'])
            or config.get('a_nodes') != NODES or config.get('max_events') is not None
            or plan.get('original_physics_designs') != list(range(40))
            or plan.get('a_nodes') != NODES or plan.get('aa_chunks') != CHUNKS):
        raise ValueError('Wrong canonical design/node/chunk scope')
    tasks = plan['tasks']
    if ([t['id'] for t in tasks] != list(range(40))
            or any(type(t['id']) is not int for t in tasks)
            or {(t['tag'], tuple(t['chunks'])) for t in tasks}
            != {(tag, (chunk,)) for tag in ('aa05', 'aa510') for chunk in CHUNKS}
            or type(task_id) is not int or not 0 <= task_id < 40):
        raise ValueError('Wrong canonical task coverage/identity')
    return tasks[task_id]


def _identity(col, plan, plan_path, config_path, wrapper, reference):
    execution = sha(__file__)
    if plan.get('checkpoint_execution_sha256') != execution:
        raise ValueError('Checkpoint execution differs from approved plan')
    methods = col.method_sources()
    if methods != col._source_at_import or methods.get(COLLECTOR) != sha(col.__file__):
        raise RuntimeError('Original collector/helper sources changed')
    for function, name in ((col.pack, 'compact.py'), (col.write_checked, 'storage.py')):
        if Path(function.__code__.co_filename).resolve() != (col.FULL/name).resolve():
            raise RuntimeError('Original storage helper import changed')
    native = col.native_sources(wrapper, reference)
    reconstruction = col.reconstruction_sources(wrapper, reference)
    if plan.get('reconstruction_sources') != reconstruction:
        raise RuntimeError('Loaded native/policy differs from approved plan')
    return dict(plan_sha256=sha(plan_path), config_sha256=sha(config_path),
        collector_sha256=methods[COLLECTOR], method_policy=col.METHOD_POLICY,
        method_sources=methods, native_source_sha256=native,
        reconstruction_sources=reconstruction, checkpoint_policy=POLICY,
        checkpoint_execution_sha256=execution)


def _array_sha(value):
    value = np.asarray(value)
    return hashlib.sha256(_json([value.dtype.str, list(value.shape)])
                          + value.tobytes(order='C')).hexdigest()


def _validate_counts(counts, fingerprints, config):
    n = len(counts)
    if (counts.shape != (n, len(NODES), 172) or counts.dtype != np.dtype('int16')
            or not 0 < n <= 50000 or fingerprints.shape != (n, 5)
            or fingerprints.dtype != np.dtype('float64') or not np.isfinite(fingerprints).all()
            or len(np.unique(fingerprints[:, 0])) != n or np.any(fingerprints[:, 1] < 0)
            or not fingerprints[:, 1].sum() > 0 or np.any(counts[..., -1] != 1)):
        raise ValueError('Invalid complete design counts/fingerprints')
    # Use the collector's hadron-then-native order, including the appended r_g.
    old = [(b['name'], lo, hi) for b in config['layout']
           for lo, hi in zip(b['edges'][:-1], b['edges'][1:])]
    blocks = [b for b in config['layout'] if '__hadron__' in b['name']] + config['jet_blocks']
    features = [(b['name'], lo, hi) for b in blocks
                for lo, hi in zip(b['edges'][:-1], b['edges'][1:])]
    if (len(old) != 134 or len(features) != 170 or features[:134] != old
            or any('__rg_atlas__' not in name for name, _, _ in features[134:])):
        raise ValueError('Invalid original 134 plus appended 36 r_g feature layout')
    unsigned = ['__hadron__' not in name for name, _, _ in features] + [True, True]
    if np.any(counts[..., unsigned] < 0):
        raise ValueError('Invalid feature layout or negative jet/normalization counts')


def _audit(audit, recoveries, task, chunk, design, fingerprints):
    n = len(fingerprints)
    if (not isinstance(audit, dict) or not isinstance(recoveries, list)
            or any(type(v) is not int or v < 0 for v in audit.values())):
        raise ValueError('Invalid strict audit integer deltas')
    recovered = audit.get('recovered_events', 0)
    if (audit.get('strict_transformed_successes', 0) != n*(len(NODES)-1)
            or audit.get('unchanged_successes', 0)+recovered != n*len(NODES)
            or audit.get('retention_failures', 0) != recovered or recovered != len(recoveries)
            or audit.get('precision_boundary_interventions', 0) < recovered
            or (not recovered and audit.get('precision_boundary_interventions', 0))):
        raise ValueError('Incomplete strict audit or transformed recovery')
    ids = set(fingerprints[:, 0])
    for row in recoveries:
        event = row.get('event', {})
        if (event.get('a') != 0 or event.get('tag') != task['tag']
                or event.get('chunk') != chunk or event.get('design') != design
                or type(event.get('event_ID')) is not int or event['event_ID'] not in ids
                or row.get('trigger') != RETENTION_ERROR or not row.get('decisions')):
            raise ValueError('Recovery is not bound to an original a=0 source event')


def _validate_sample(sample, source, n):
    if (any(sample.get(k) != v for k, v in source.items()) or sample.get('events') != n
            or sample.get('native_event_node_checks') != n*len(NODES)
            or sample.get('strict_transformed_checks') != n*(len(NODES)-1)
            or any(sample.get(k) is not True for k in
                   ('hadron_baseline_exact', 'baseline_a0_identity_exact', 'no_events_dropped'))
            or any(sample.get('qa', {}).get(str(a), {}).get('whole_event_fallback') != 0 for a in NODES)):
        raise ValueError('Design report/source proof changed or incomplete')


def _atomic_write(path, data, col, root, cap):
    """Uncommitted partials survive crashes; only the immutable link commits."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError('Refuse to overwrite a committed output')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.partial')
    col.write_checked(partial, data, root, cap)
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    # Keep the shared size scan from counting the transient hardlink twice.
    with (Path(root)/'STORAGE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.link(partial, path)
        partial.unlink()
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_design(path, counts, fingerprints, record, col, root, cap):
    arrays = col.pack(counts[:, None])
    arrays.update(fingerprints=fingerprints, design_ids=np.array([record['design_id']]),
                  a_nodes=np.array(NODES))
    record = dict(record, array_sha256={k: _array_sha(v) for k, v in arrays.items()})
    record['metadata_sha256'] = hashlib.sha256(_json(record)).hexdigest()
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays, metadata_json=np.array(_json(record).decode()))
        _atomic_write(path, buffer.getvalue(), col, root, cap)


def load_design(path, expected, source, config):
    """Reject an invalid committed checkpoint; never silently recompute it."""
    if Path(path).is_symlink():
        raise ValueError('Checkpoint must be an ordinary immutable file')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'baseline_counts', 'delta_counts', 'fingerprints',
                                  'design_ids', 'a_nodes', 'metadata_json'}:
            raise ValueError('Wrong design checkpoint members')
        arrays = {k: archive[k] for k in archive.files if k != 'metadata_json'}
        record = json.loads(str(archive['metadata_json'].item()))
    checksum = record.pop('metadata_sha256', None)
    if checksum != hashlib.sha256(_json(record)).hexdigest():
        raise ValueError('Checkpoint metadata checksum changed')
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint execution/source/task identity changed')
    if record.get('sample', {}).get('design_id') != expected['design_id']:
        raise ValueError('Checkpoint sample design identity changed')
    if record.get('array_sha256') != {k: _array_sha(v) for k, v in arrays.items()}:
        raise ValueError('Checkpoint count/fingerprint checksum changed')
    base, delta = arrays['baseline_counts'], arrays['delta_counts']
    n = len(base)
    if (base.shape != (n, 1, 172) or delta.shape != (n, 1, len(NODES)-1, 172)
            or base.dtype != np.dtype('int16') or delta.dtype != np.dtype('int16')
            or not np.array_equal(arrays['design_ids'], [expected['design_id']])
            or not np.array_equal(arrays['a_nodes'], NODES)):
        raise ValueError('Wrong compact design count layout/identity')
    full = np.concatenate((base[:, :, None].astype(np.int32),
                           base[:, :, None].astype(np.int32)+delta.astype(np.int32)), axis=2)[:, 0]
    if np.any(full < -32768) or np.any(full > 32767):
        raise ValueError('Restored design count range exceeded')
    counts, fingerprints = full.astype(np.int16), arrays['fingerprints']
    _validate_counts(counts, fingerprints, config)
    _validate_sample(record['sample'], source, n)
    _audit(record['rg_strict_first_audit'], record['rg_precision_events'], expected['task'],
           expected['chunk'], expected['design_id'], fingerprints)
    return counts, fingerprints, record


def verify_final_metadata(metadata, plan, *, execution_sha256=None):
    """Additional local-controller check; ordinary collector checks still apply."""
    expected = execution_sha256 or sha(__file__)
    rows = metadata.get('design_checkpoint_records', [])
    if (not _digest(expected) or plan.get('checkpoint_execution_sha256') != expected
            or metadata.get('checkpoint_execution_sha256') != expected
            or metadata.get('checkpoint_policy') != POLICY
            or [row.get('design_id') for row in rows] != list(range(40))
            or any(type(row.get('design_id')) is not int or not _digest(row.get('sha256')) for row in rows)
            or metadata.get('collector_sha256') != metadata.get('method_sources', {}).get(COLLECTOR)):
        raise ValueError('Invalid checkpoint execution certificate')
    return True


def _same_archive(path, arrays):
    if Path(path).is_symlink():
        raise ValueError('Final counts must not be a symlink')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays) or any(
                archive[k].dtype != v.dtype or not np.array_equal(archive[k], v)
                for k, v in arrays.items()):
            raise ValueError('Existing final count output differs from completed designs')


def run_task(task_id, reference, output, *, plan_path=None, config_path=None, collector=None):
    """Run/restart one approved AA chunk; no submission, retry, or physics logic."""
    col = collector if collector is not None else load_collector()
    col.runtime.NODES = tuple(NODES)
    plan_path, config_path = Path(plan_path or HERE/'PLAN.json'), Path(config_path or HERE/'config.json')
    plan, config = json.loads(plan_path.read_text()), json.loads(config_path.read_text())
    task = _scope(plan, config, task_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError('Output must not be a symlink')
    root, cap = output.parent.parent, plan['storage_bytes_cap']
    with (output/'.checkpoint_execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = output/'CHECKPOINT.json'
        if target.is_symlink():
            raise ValueError('Final metadata must not be a symlink')
        if target.exists() and json.loads(target.read_text()).get('slurm_job_id') != os.environ.get('SLURM_JOB_ID'):
            raise ValueError('Completed output belongs to a different job; approved reuse certificate required')
        wrapper, unresolved = col.install_reconstruction(reference)
        identity = _identity(col, plan, plan_path, config_path, wrapper, reference)

        def unchanged():
            if _identity(col, plan, plan_path, config_path, wrapper, reference) != identity:
                raise RuntimeError('Plan/config/method/native changed during checkpoint execution')

        chunk, = task['chunks']
        common, matrices, samples, certificates = None, None, [], []
        total_audit, recoveries = Counter(), []
        for column, design in enumerate(config['design_ids']):
            unchanged()
            paths, info, source = col.source_paths(Path(plan['source_bank']), task['tag'], design, chunk)
            if info['system'] != 'PbPb' or info['centrality'] != ([0, 5] if task['tag'] == 'aa05' else [5, 10]):
                raise ValueError('Wrong source system/centrality')
            expected = dict(status='DESIGN_COMPLETE', task=task, chunk=chunk,
                            design_id=design, **identity)
            path = output/'checkpoints'/f'chunk_{chunk:04d}'/f'design_{design:02d}.npz'
            if path.exists() or path.is_symlink():
                counts, fingerprints, record = load_design(path, expected, source, config)
            else:
                before, previous = dict(wrapper.audit), _json(wrapper.events)
                count_events = len(wrapper.events)
                counts, fingerprints, report = col.collect_one(paths['raw'], paths['saved'], config,
                    info['system'], info['centrality'], context=dict(tag=task['tag'], chunk=chunk, design=design),
                    wrapper=wrapper, unresolved=unresolved)
                if _json(wrapper.events[:count_events]) != previous:
                    raise RuntimeError('Original reconstruction recovery history changed')
                delta = {key: value-before.get(key, 0) for key, value in wrapper.audit.items()}
                if any(key not in wrapper.audit and value for key, value in before.items()):
                    raise RuntimeError('Original reconstruction audit history changed')
                record = dict(expected, sample=dict(design_id=design, **source, **report),
                    rg_strict_first_audit=delta, rg_precision_events=wrapper.events[count_events:],
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'))
                _validate_counts(counts, fingerprints, config)
                _validate_sample(record['sample'], source, len(counts))
                _audit(delta, record['rg_precision_events'], task, chunk, design, fingerprints)
                unchanged()
                if col.source_paths(Path(plan['source_bank']), task['tag'], design, chunk)[2] != source:
                    raise RuntimeError('Original input source changed during collection')
                if common is not None:
                    np.testing.assert_array_equal(common, fingerprints)
                save_design(path, counts, fingerprints, record, col, root, cap)
                # Exercise exactly the restore validator before accepting the commit.
                counts, fingerprints, record = load_design(path, expected, source, config)
            if col.source_paths(Path(plan['source_bank']), task['tag'], design, chunk)[2] != source:
                raise RuntimeError('Original input source changed during checkpoint validation')
            if common is None:
                common = fingerprints
                matrices = np.empty((len(counts), 40, len(NODES), 172), np.int16)
            else:
                np.testing.assert_array_equal(common, fingerprints)
            matrices[:, column] = counts
            samples.append(record['sample'])
            total_audit.update(record['rg_strict_first_audit'])
            recoveries.extend(record['rg_precision_events'])
            certificates.append(dict(design_id=design, sha256=sha(path), filename=str(path.relative_to(output))))
            print('WAKE_FULL_DESIGN_COMPLETE', task_id, chunk, design, flush=True)
        unchanged()
        arrays = col.pack(matrices)
        arrays.update(fingerprints=common, design_ids=np.array(config['design_ids']), a_nodes=np.array(NODES))
        path = output/f'chunk_{chunk:04d}.npz'
        if path.exists():
            _same_archive(path, arrays)
        else:
            with io.BytesIO() as buffer:
                np.savez_compressed(buffer, **arrays)
                _atomic_write(path, buffer.getvalue(), col, root, cap)
        record = dict(chunk=chunk, filename=path.name, sha256=sha(path), bytes=path.stat().st_size,
            events=len(common), samples=samples,
            paired_fingerprint_sha256=hashlib.sha256(common.astype('<f8').tobytes()).hexdigest())
        metadata = dict(status='COMPLETE', task=task, records=[record], global_a=True, no_new_events=True,
            **identity, transport_sha256=identity['method_sources'][TRANSPORT], sources_unchanged=True,
            rg_strict_first_audit=dict(total_audit), rg_precision_events=recoveries,
            design_checkpoint_records=certificates, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        verify_final_metadata(metadata, plan)
        unchanged()
        if target.exists():
            previous = json.loads(target.read_text())
            if previous != metadata:
                raise ValueError('Existing final metadata differs from completed designs')
            metadata = previous
        else:
            _atomic_write(target, (json.dumps(metadata, indent=2, allow_nan=False)+'\n').encode(), col, root, cap)
        print('WAKE_FULL_TASK_COMPLETE', task_id, flush=True)
        return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_task(args.task, args.reference, args.output)


if __name__ == '__main__':
    main()
