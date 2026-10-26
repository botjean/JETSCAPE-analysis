"""Source-pinned full-event wake bank. No new events or failed-task retries."""
from collections import Counter
import argparse
import hashlib
import itertools
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
WORK = STUDY.parent
sys.path[:0] = [str(STUDY), str(WORK/'wake_sensitivity_20261017'),
               str(WORK/'wake_paper_morph_20261017'),
               str(WORK/'JETSCAPE-analysis-atlas-dr12')]
from collect_gate import feature_counts, hadrons, legacy_counts, sha, FIELDS
from global_a import deform, _transport
from compact import pack
from conservation import RecoverProjection
from storage import write_checked
import run_study as pilot
from jetscape_analysis.analysis import atlas_dr12 as reader

CONSERVATION = {}


def install_conservation(config):
    for a in config['a_nodes']:
        if a == 0:
            continue
        module = _transport(a)
        if not isinstance(module.project_signed, RecoverProjection):
            module.project_signed = RecoverProjection(module.project_signed)
        CONSERVATION[a] = module.project_signed
    return CONSERVATION


def install_reconstruction(reference):
    """Reuse the already-reviewed precision routing and dR12 diagnostic copy."""
    sys.path.insert(0, str(reference))
    import _rg_precision_candidate as precision
    from strict_first import StrictFirst
    import _wake_dr12_diagnostic as diagnostic
    strict = pilot.rg._upstream_backend()
    class ScopedPrecision(StrictFirst):
        def reconstruct(self, rows, *args, **kwargs):
            # The old six-digit/float32 source declaration is valid only before
            # deformation. Do not pretend nonlinear float64 outputs are stored
            # original particles or invent a transported numerical error bound.
            if self.context is not None and self.context.get('a', 0) != 0:
                result = self.strict.reconstruct(rows, *args, **kwargs)
                self.audit['unchanged_successes'] += 1
                self.audit['strict_transformed_successes'] += 1
                return result
            return super().reconstruct(rows, *args, **kwargs)

    wrapper = ScopedPrecision(strict, precision)
    pilot.rg._upstream_backend = lambda: wrapper
    unresolved = []

    def reconstruct(rows, **kwargs):
        result = diagnostic.reconstruct(rows, **kwargs)
        for jet in result['unresolved']:
            if 200 <= jet[0] < 251:
                raise RuntimeError('Undefined selected dR12 splitting; no dropping or guessed bin')
            unresolved.append([float(x) for x in jet])
        return result

    pilot.dr = SimpleNamespace(particle_rows=reader.particle_rows, CHARGED_PIDS=reader.CHARGED_PIDS,
                              _atlas_dr12=SimpleNamespace(reconstruct=reconstruct))
    return wrapper, unresolved


def source_paths(source, tag, design, chunk):
    directory = source/('pp' if tag == 'pp' else f'point_{design:02d}_{tag}')
    meta_path = directory/'pooled.json'
    info = json.loads(meta_path.read_text())
    interval = None if tag == 'pp' else ([0, 5] if tag == 'aa05' else [5, 10])
    if info['design_id'] != design or info['centrality'] != interval or info['partial']:
        raise ValueError('Original source identity changed')
    if len(info['records']) != (161 if tag == 'pp' else 200):
        raise ValueError('Incomplete original chunk bank')
    record = info['records'][chunk]
    paths = {}
    for key, marker in (('raw', 'final_state_hadrons'), ('saved', '_observables_')):
        matching = [Path(p) for p in record['sources'] if marker in Path(p).name]
        if len(matching) != 1:
            raise ValueError('Ambiguous original raw/observable file')
        paths[key] = matching[0]
        if sha(paths[key]) != record['sources'][str(paths[key])]:
            raise ValueError('Original input checksum changed')
    sigma = float(info['cross_section'])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid fixed full-production generator cross section')
    return paths, info, dict(metadata_sha256=sha(meta_path),
        sources={str(p): sha(p) for p in paths.values()}, fixed_full_cross_section=sigma)


def collect_one(raw, saved, config, system, centrality, *, limit=None,
                context=None, wrapper=None, unresolved=None):
    import pyarrow.parquet as pq
    raw_file, obs_file = pq.ParquetFile(raw), pq.ParquetFile(saved)
    n = raw_file.metadata.num_rows if limit is None else limit
    if raw_file.metadata.num_rows != obs_file.metadata.num_rows or n <= 0:
        raise ValueError('Different/incomplete raw and saved event coverage')
    if n > raw_file.metadata.num_rows or n > 50000:
        raise ValueError('Unexpected source event count exceeds the memory contract')
    nodes = config['a_nodes'] if system == 'PbPb' else [0.]
    hadron_blocks = [b for b in config['layout'] if '__hadron__' in b['name']]
    nhadron = sum(len(b['edges'])-1 for b in hadron_blocks)
    columns = {'event_weight_all', 'event_weight', 'pt_hat'}
    columns.update(b.get('column_pp' if system == 'pp' else 'column_aa', '') for b in config['layout'])
    schema = set(obs_file.schema_arrow.names)
    if not {'event_weight_all', 'pt_hat'} <= schema:
        raise ValueError('Missing mandatory saved-event alignment fields')
    obs = (r for batch in obs_file.iter_batches(batch_size=256, columns=sorted(columns & schema))
           for r in batch.to_pylist())
    counts = np.empty((n, len(nodes), 172), np.int16)
    identity = np.empty((n, len(FIELDS)))
    baseline_diff = np.zeros(134)
    baseline_changed = np.zeros(134, np.int64)
    native_only = np.array([block.get('source') == 'dr12' for block in config['layout']
                            for _ in block['edges'][:-1]])
    qa = {str(a): Counter() for a in nodes}
    maxima = {str(a): Counter() for a in nodes}
    outside_selected_dr12 = []
    chunk_sigma = None
    started = time.monotonic()
    events = 0
    for event, row in itertools.islice(itertools.zip_longest(reader.read_events(raw), obs), n):
        if event is None or row is None:
            raise ValueError('Raw/saved streams ended at different events')
        weight = float(event['event_weight'])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Invalid event weight')
        if row['event_weight_all'] != weight or row['pt_hat'] != event['pt_hat']:
            raise ValueError('Raw/saved event alignment mismatch')
        if row.get('event_weight') is not None and row['event_weight'] != weight:
            raise ValueError('Wrong accepted-event weight')
        if system == 'PbPb' and not centrality[0] <= event['centrality'] < centrality[1]:
            raise ValueError('Wrong event centrality')
        sigma = float(event['cross_section'])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Invalid within-chunk generator cross section')
        if chunk_sigma is not None and not np.isclose(sigma, chunk_sigma, rtol=1e-6):
            raise ValueError('Cross section changes inside a source file')
        chunk_sigma = sigma
        identity[events] = [event[k] for k in FIELDS]
        if not np.isfinite(identity[events]).all():
            raise ValueError('Nonfinite event fingerprint')
        for node, a in enumerate(nodes):
            if wrapper is not None:
                wrapper.context = dict(context or {}, event_ID=event['event_ID'], a=a)
            if a in CONSERVATION:
                CONSERVATION[a].context = dict(context or {}, event_ID=event['event_ID'], a=a)
            if unresolved is not None:
                unresolved.clear()
            modified, checks = deform(event, a)
            other, _ = pilot.reconstruction(modified, system, config['jet_blocks'])
            vector = np.r_[hadrons(modified, hadron_blocks, config, centrality, system), other]
            if vector.shape != (172,) or not np.isfinite(vector).all() or np.any(vector != np.rint(vector)):
                raise ValueError('Noninteger or malformed native count vector')
            if np.max(np.abs(vector)) > np.iinfo(np.int16).max or vector[-1] != 1:
                raise ValueError('Count range/normalization contract failed')
            counts[events, node] = vector
            if unresolved:
                outside_selected_dr12.append(dict(context or {}, event_ID=event['event_ID'], a=a,
                                                   jets=list(unresolved)))
            qa[str(a)].update({key: checks[key] for key in ('deleted_positive', 'deleted_negative',
                'frozen_spacelike', 'no_axis_events', 'degenerate_events', 'conditional_resampling_draws')})
            for key in ('signed_p4_residual', 'mass2_residual', 'correction_rms_radians'):
                maxima[str(a)][key] = max(maxima[str(a)][key], checks[key])
        old = legacy_counts(row, config['layout'], centrality, system)
        np.testing.assert_array_equal(counts[events, 0, :nhadron], old[:nhadron])
        diff = counts[events, 0, :134]-old
        diff[native_only] = 0
        baseline_diff += weight*diff
        baseline_changed += diff != 0
        events += 1
    if events != n or len(np.unique(identity[:, 0])) != n:
        raise ValueError('Missing/duplicate source events')
    if wrapper is not None:
        wrapper.context = None
    return counts, identity, dict(events=events, seconds=time.monotonic()-started,
        input_chunk_cross_section=chunk_sigma, baseline_weighted_difference=baseline_diff.tolist(),
        baseline_changed_entries=baseline_changed.tolist(), hadron_baseline_exact=True,
        qa={k: dict(v) for k, v in qa.items()}, maxima={k: dict(v) for k, v in maxima.items()},
        unresolved_dr12_outside_fit=outside_selected_dr12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_path = HERE/'PLAN.json'
    plan = json.loads(plan_path.read_text())
    task = next(t for t in plan['tasks'] if t['id'] == args.task)
    config = json.loads((HERE/'config.json').read_text())
    if args.output.exists():
        raise ValueError('No task overwrite or automatic retry')
    args.output.mkdir(parents=True)
    wrapper, unresolved = install_reconstruction(args.reference)
    install_conservation(config)
    records = []
    for chunk in task['chunks']:
        common = None
        designs = [None] if task['tag'] == 'pp' else config['design_ids']
        matrices = None
        sample_records = []
        for column, design in enumerate(designs):
            paths, info, source = source_paths(Path(plan['source_bank']), task['tag'], design, chunk)
            counts, identity, report = collect_one(paths['raw'], paths['saved'], config,
                info['system'], info['centrality'], context=dict(tag=task['tag'], chunk=chunk, design=design),
                wrapper=wrapper, unresolved=unresolved)
            if common is None:
                common = identity
                matrices = np.empty((len(counts), len(designs), counts.shape[1], counts.shape[2]), np.int16)
            else:
                np.testing.assert_array_equal(common, identity)
            matrices[:, column] = counts
            sample_records.append(dict(design_id=design, **source, **report))
            print('WAKE_FULL_DESIGN_COMPLETE', task['id'], chunk, design, round(report['seconds'], 2), flush=True)
        path = args.output/f'chunk_{chunk:04d}.npz'
        arrays = pack(matrices)
        arrays.update(fingerprints=common, design_ids=np.array([-1] if task['tag'] == 'pp' else designs),
                      a_nodes=np.array([0.] if task['tag'] == 'pp' else config['a_nodes']))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        write_checked(path, buffer.getbuffer(), args.output.parent.parent, plan['storage_bytes_cap'])
        buffer.close()
        record = dict(chunk=chunk, filename=path.name, sha256=sha(path), bytes=path.stat().st_size,
            events=len(common), samples=sample_records,
            paired_fingerprint_sha256=hashlib.sha256(common.astype('<f8').tobytes()).hexdigest())
        records.append(record)
        metadata = dict(status='COMPLETE' if len(records) == len(task['chunks']) else 'PARTIAL',
            task=task, records=records, global_a=True, no_new_events=True,
            plan_sha256=sha(plan_path), config_sha256=sha(HERE/'config.json'),
            collector_sha256=sha(__file__), transport_sha256=sha(WORK/'wake_paper_morph_20261017/transport.py'),
            rg_strict_first_audit=dict(wrapper.audit), rg_precision_events=wrapper.events,
            conservation_sha256=sha(HERE/'conservation.py'),
            conservation_recovery={str(a): dict(audit=dict(solver.audit), events=solver.recoveries)
                                   for a, solver in CONSERVATION.items()},
            slurm_job_id=__import__('os').environ.get('SLURM_JOB_ID'))
        # AA tasks contain exactly one chunk, so immutable checkpoint writes do
        # not replace partial results or duplicate a completed source file.
        write_checked(args.output/'CHECKPOINT.json', (json.dumps(metadata, indent=2)+'\n').encode(),
                      args.output.parent.parent, plan['storage_bytes_cap'])
        print('WAKE_FULL_CHUNK_COMPLETE', task['id'], chunk, path.stat().st_size, flush=True)
        del arrays, matrices, counts
    print('WAKE_FULL_TASK_COMPLETE', task['id'], flush=True)


if __name__ == '__main__':
    main()
