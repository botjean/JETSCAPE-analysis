"""Measured rg/old-overlap moments on saved chunks, with old physics unchanged."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
from collect_low_pt_gate import extract

BASE = Path('/rds/project/rds-hCZCEbPdvZ8/jeandp')
ROOT = BASE / 'atlas-rg-reduced-20261015'
FIELDS = ['event_ID', 'event_weight', 'pt_hat', 'vertex_x', 'vertex_y']
EDGES = [0, .003, .0041776, .0063246, .0095748, .0144956, .0219452,
         .0332232, .0502972, .0761462, .115279, .174524, .264215, .4]
RG_CONFIG = dict(sqrt_s=5020, centrality=[0, 10], particle_level='legacy_diagnostic',
                 groomer='signed_prongs', rg_edges=EDGES,
                 binning_status='publication_reconstruction_unverified')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprints(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=FIELDS)
    result = np.column_stack([table[k].to_numpy() for k in FIELDS])
    ids = result[:, 0]
    if (not np.isfinite(result).all() or np.any(ids < 0) or
            np.any(ids != np.floor(ids)) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Invalid or nonunique event identities')
    return result


def capture(producer, paths, layout, info, config, previous, identity):
    from jetscape_analysis.analysis import atlas_dr12 as dr
    from jetscape_analysis.analysis import atlas_rg as rg
    reader = dr.read_events
    vectors, cutflow = [], Counter()
    rg_seconds = 0.

    def with_rg(path):
        nonlocal rg_seconds
        if Path(path).resolve() != paths[1].resolve():
            raise ValueError('Unexpected raw-event reader path')
        for i, event in enumerate(reader(path)):
            if i >= len(identity):
                raise ValueError('Extra raw event')
            np.testing.assert_array_equal([event[k] for k in FIELDS], identity[i])
            start = time.monotonic()
            counts, qa = rg.count_event(event, RG_CONFIG, info['system'])
            vectors.append(float(event['event_weight']) * np.r_[counts, 1.])
            cutflow.update(qa)
            rg_seconds += time.monotonic() - start
            yield event

    dr.read_events = with_rg
    try:
        arrays, report, old, projection = extract(producer, paths, layout, info, config, previous)
    finally:
        dr.read_events = reader
    new = np.asarray(vectors)
    if new.shape != (len(old), 40) or len(old) != len(identity) or not np.isfinite(new).all():
        raise ValueError('Old/new event vector layout/alignment changed')
    np.testing.assert_array_equal(new[:, -1], old[:, -1])
    return arrays, report, old, new, projection, dict(rg_seconds=rg_seconds, cutflow=dict(cutflow))


def delta_statistics(old, new, pp=False):
    if (old.ndim != 2 or new.ndim != 2 or len(old) != len(new) or
            not np.isfinite(old).all() or not np.isfinite(new).all()):
        raise ValueError('Invalid aligned event matrices')
    result = dict(old_total=old.sum(0), new_total=new.sum(0),
                  old_new=old.T @ new, new_new=new.T @ new)
    if pp:
        result['old_old'] = old.T @ old
    return result


def assert_old_reference(old, reference, design_ids):
    with np.load(reference, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['design_ids'], design_ids)
        if old.shape[1] != len(design_ids) * int(z['vector_width']) or len(old) != int(z['events']):
            raise ValueError('Old reference dimensions changed')
        np.testing.assert_allclose(old.sum(0), z['total'], rtol=1e-10, atol=1e-18)
        target = z['outer']
        # Row blocks avoid several full covariance-sized comparison temporaries.
        for first in range(0, old.shape[1], 256):
            last = min(first + 256, old.shape[1])
            np.testing.assert_allclose(old[:, first:last].T @ old,
                                       target[first:last], rtol=1e-10, atol=1e-18)


def extract_chunk(producer, tag, job, layout, config, gate):
    from jetscape_analysis.analysis import atlas_dr12 as dr
    pp = tag == 'pp'
    designs = [None] if pp else list(range(40))
    interval = None if pp else ([0, 5] if tag == 'aa05' else [5, 10])
    old_blocks, new_blocks, records = [], [], []
    identity = None
    features = norms = None
    for design in designs:
        directory = BASE / 'simulation-covariance-20261007/bank' / (
            'pp' if pp else f'point_{design:02d}_{tag}')
        metadata = directory / 'pooled.json'
        info = json.loads(metadata.read_text())
        if (info['partial'] or info['design_id'] != design or info['centrality'] != interval or
                info['extraction_sha256'] != digest(producer.__file__) or
                info['dr12_source_sha256'] != digest(dr.__file__) or
                len(info['records']) != (161 if pp else 200)):
            raise ValueError('Original sample identity/source contract changed')
        record = info['records'][job]
        paths = []
        for marker in ('_observables_', 'final_state_hadrons', '_cross_section_'):
            matching = [Path(p) for p in record['sources'] if marker in Path(p).name]
            if len(matching) != 1:
                raise ValueError('Ambiguous original input: ' + marker)
            paths.append(matching[0])
        current = fingerprints(paths[1])
        if identity is None:
            identity = current
        else:
            np.testing.assert_array_equal(current, identity)
        previous_path = directory / f'chunk_{job:04d}.npz'
        with np.load(previous_path, allow_pickle=False) as previous:
            arrays, report, old, new, projection, extra = capture(
                producer, paths, layout, info, config, previous, current)
        if len(report['features']) != 134 or len(info['features']) != 102 or report['sources'] != record['sources']:
            raise ValueError('Original input/feature contract changed')
        if features is None:
            features, norms = report['features'], report['normalization_features']
        if features != report['features'] or norms != report['normalization_features']:
            raise ValueError('Mixed old layouts')
        old_blocks.append(old)
        new_blocks.append(new)
        records.append(dict(design_id=design, chunk_index=job, events=report['events'],
                            sources=report['sources'], old_info_sha256=digest(metadata),
                            old_chunk_sha256=digest(previous_path), **extra))
        print('REDUCED_DESIGN_COMPLETE', tag, job, design, len(old), extra['rg_seconds'], flush=True)
    old = np.concatenate(old_blocks, axis=1)
    new = np.concatenate(new_blocks, axis=1)
    width = old_blocks[0].shape[1]
    del old_blocks, new_blocks, arrays, projection, current
    if gate or pp:
        task = job + (400 if pp else (0 if tag == 'aa05' else 200))
        low = BASE / 'hadron-low-pt-covariance-20261014'
        reference = (low / 'gate' / f'gate_{task:04d}.npz' if task in (0, 200) else
                     low / 'bank' / f'chunk_{task:04d}.npz')
        assert_old_reference(old, reference, [-1] if pp else range(40))
    stats = delta_statistics(old, new, pp)
    if not np.isfinite(stats['old_new']).all():
        raise ValueError('Nonfinite old/new moments')
    return stats, dict(tag=tag, chunk_index=job, events_per_design=len(identity),
                       design_ids=[-1] if pp else list(range(40)), old_vector_width=width,
                       new_vector_width=40, old_features=features, old_normalization_features=norms,
                       full_old_block_checked=bool(gate or pp), records=records,
                       event_fingerprint_sha256=hashlib.sha256(identity.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    spec = json.loads((run / 'MANIFEST.json').read_text())
    batch = next(b for b in spec['batches'] if b['id'] == args.batch)
    output = ROOT / 'batches' / args.batch
    if output.exists():
        raise ValueError('No automatic overwrite/retry of a batch')
    output.mkdir()
    producer_path = BASE / 'simulation-covariance-20261007/simulation_covariance.py'
    module = importlib.util.spec_from_file_location('unchanged_producer', producer_path)
    producer = importlib.util.module_from_spec(module)
    module.loader.exec_module(producer)
    from jetscape_analysis.analysis import atlas_dr12 as dr
    for name, expected in json.loads((run / 'external_pins.json').read_text()).items():
        if digest(name) != expected:
            raise ValueError('External input/source pin changed: ' + name)
    layout = json.loads((run / 'layout_hpt10.json').read_text())
    config = dr.load_config(BASE / 'atlas-dr12-pilot-20261007/config/atlas_dr12_5020.yaml')
    accumulator, reports = None, []
    started = time.monotonic()
    for job in batch['chunks']:
        stats, report = extract_chunk(producer, batch['tag'], job, layout, config, batch['gate'])
        if accumulator is None:
            accumulator = stats
        else:
            if set(accumulator) != set(stats):
                raise ValueError('Mixed sufficient statistics')
            for key in stats:
                accumulator[key] += stats[key]
        reports.append(report)
        payload_bytes = sum(x.nbytes for x in accumulator.values())
        used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        if used + payload_bytes + 2**20 > spec['storage_cap_bytes']:
            raise ValueError('New-storage cap would be exceeded')
        pending = output / 'checkpoint.pending.npz'
        np.savez_compressed(pending, **accumulator)
        pending.replace(output / 'checkpoint.npz')
        metadata = dict(status='COMPLETE' if len(reports) == len(batch['chunks']) else 'PARTIAL',
                        batch=batch, completed_chunks=[r['chunk_index'] for r in reports],
                        reports=reports, seconds=time.monotonic()-started,
                        maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        collector_sha256=digest(__file__), manifest_sha256=digest(run / 'MANIFEST.json'),
                        checkpoint_sha256=digest(output / 'checkpoint.npz'),
                        new_events_generated=False, rg_edges=EDGES, pi0_treatment='keep')
        temporary = output / 'CHECKPOINT.pending.json'
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(output / 'CHECKPOINT.json')
        print('REDUCED_CHUNK_CHECKPOINT', batch['id'], job, metadata['seconds'], flush=True)
        del stats, report
    print('REDUCED_BATCH_COMPLETE', batch['id'], time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
