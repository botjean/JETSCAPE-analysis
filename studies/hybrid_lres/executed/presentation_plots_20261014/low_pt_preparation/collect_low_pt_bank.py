"""Extend the successful low-pT gate to the remaining original AA/pp chunks."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from collect_low_pt_gate import extract


def task_identity(task):
    if not 0 <= task <= 560 or task in (0, 200):
        raise ValueError('Only the 559 approved new extraction tasks are allowed')
    if task >= 400:
        return 'pp', task-400, [None]
    return ('aa05' if task < 200 else 'aa510'), task % 200, list(range(40))


def source_paths(record):
    result = []
    for marker in ('_observables_', 'final_state_hadrons', '_cross_section_'):
        matches = [Path(path) for path in record['sources'] if marker in Path(path).name]
        if len(matches) != 1:
            raise ValueError('Ambiguous input: '+marker)
        result.append(matches[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('source', 'producer', 'layout', 'old-joint', 'dr12-config', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    tag, job, designs = task_identity(args.task)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite completed or partial outputs')
    spec = importlib.util.spec_from_file_location('unchanged_bank_producer', args.producer)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    from jetscape_analysis.analysis import atlas_dr12 as dr
    import pyarrow.parquet as pq
    config = dr.load_config(args.dr12_config)
    layout = json.loads(args.layout.read_text())
    vectors, records = [], []
    fingerprint = projection = None
    started = time.monotonic()
    for design in designs:
        directory = args.source/('pp' if design is None else f'point_{design:02d}_{tag}')
        info_path = directory/'pooled.json'
        info = json.loads(info_path.read_text())
        interval = None if tag == 'pp' else ([0, 5] if tag == 'aa05' else [5, 10])
        if info['partial'] or info['centrality'] != interval or info['design_id'] != design:
            raise ValueError('Original sample identity changed')
        if info['extraction_sha256'] != producer.digest(args.producer):
            raise ValueError('Original producer source changed')
        if len(info['records']) != (161 if tag == 'pp' else 200):
            raise ValueError('Original chunk inventory changed')
        record = info['records'][job]
        paths = source_paths(record)
        fields = ['event_ID', 'event_weight', 'pt_hat', 'vertex_x', 'vertex_y']
        table = pq.read_table(paths[1], columns=fields)
        current = np.column_stack([table[field].to_numpy() for field in fields])
        if not np.isfinite(current).all():
            raise ValueError('Nonfinite event fingerprint')
        if fingerprint is None:
            fingerprint = current
        else:
            np.testing.assert_array_equal(current, fingerprint)
        old_chunk = directory/f'chunk_{job:04d}.npz'
        with np.load(old_chunk, allow_pickle=False) as previous:
            arrays, report, captured, mapping = extract(
                producer, paths, layout, info, config, previous)
        if len(info['features']) != 102 or len(report['features']) != 134:
            raise ValueError('Unexpected feature layout')
        if report['sources'] != record['sources'] or len(captured) != len(fingerprint):
            raise ValueError('Original inputs or event alignment changed')
        if projection is None:
            projection = mapping
        else:
            np.testing.assert_array_equal(projection, mapping)
        vectors.append(captured)
        records.append(dict(design_id=design, source_hashes=report['sources'],
            old_metadata_sha256=producer.digest(info_path), old_chunk_sha256=producer.digest(old_chunk),
            events=report['events'], features=report['features'],
            normalization_features=report['normalization_features'],
            dr12_source_sha256=report['dr12_source_sha256']))
        print('LOW_PT_DESIGN_COMPLETE', tag, job, design, len(captured), flush=True)
    width = vectors[0].shape[1]
    matrix = np.concatenate(vectors, axis=1)
    del vectors, captured, table, current
    total, outer = matrix.sum(0), matrix.T @ matrix
    del matrix
    if tag != 'pp':
        old_indices = np.concatenate([projection+design*width for design in range(40)])
        with np.load(args.old_joint/f'chunk_{args.task:04d}.npz', allow_pickle=False) as old:
            np.testing.assert_allclose(total[old_indices], old['total'], rtol=1e-10, atol=1e-18)
            np.testing.assert_allclose(outer[np.ix_(old_indices, old_indices)],
                                       old['outer'], rtol=1e-10, atol=1e-18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.stem+'.pending.npz')
    np.savez_compressed(pending, total=total, outer=outer,
        design_ids=np.array([-1] if tag == 'pp' else range(40)),
        events=len(fingerprint), vector_width=width)
    pending.replace(args.output)
    metadata = dict(status='LOW_PT_BANK_CHUNK_COMPLETE', task=args.task, tag=tag,
        job_index=job, events=len(fingerprint), features=134, original_features=102,
        old_block_reproduced=True, paired_event_fingerprints_identical=True,
        records=records, seconds=time.monotonic()-started, new_event_generation=False,
        collector_sha256=producer.digest(Path(__file__)),
        gate_helper_sha256=producer.digest(Path(__file__).with_name('collect_low_pt_gate.py')),
        producer_sha256=producer.digest(args.producer), layout_sha256=producer.digest(args.layout),
        output_sha256=producer.digest(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('LOW_PT_BANK_CHUNK_COMPLETE', args.task, metadata['seconds'], flush=True)


if __name__ == '__main__':
    main()
