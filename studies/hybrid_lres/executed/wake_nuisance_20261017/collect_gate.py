"""Small paired raw-event gate; no new events, inference, or automatic retry."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path[:0] = [str(WORK/'wake_sensitivity_20261017'),
               str(WORK/'JETSCAPE-analysis-atlas-dr12')]
import run_study as pilot
from global_a import deform

FIELDS = ('event_ID', 'event_weight', 'pt_hat', 'vertex_x', 'vertex_y')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def feature_counts(values, edges):
    values = np.asarray([] if values is None else values, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Expected finite scalar selected observables')
    return np.histogram(values[(values >= edges[0]) & (values < edges[-1])], edges)[0]


def hadrons(event, blocks, config, centrality, system):
    pt = np.hypot(event['px'], event['py'])
    eta = np.arcsinh(np.divide(event['pz'], pt, out=np.zeros_like(pt), where=pt > 0))
    status, pid = np.asarray(event['status']), np.asarray(event['particle_ID'])
    valid = np.isin(status, (0, 1, 2)) if system == 'PbPb' else status == 0
    valid &= np.isin(np.abs(pid), list(pilot.dr.CHARGED_PIDS))
    signs = np.where(status == 2, -1., 1.)
    values = []
    for block in blocks:
        kind = block['name'].split('__')[3]
        settings = config['hadron'][kind]
        enabled = system == 'pp' or block['centrality'] == list(centrality)
        selected = valid & (np.abs(eta) < settings['eta_cut'])
        selected &= (pt > settings['pt'][0]) & (pt < settings['pt'][1])
        if not enabled:
            selected[:] = False
        edges = block['edges']
        selected &= (pt >= edges[0]) & (pt < edges[-1])
        values.extend(np.histogram(pt[selected], edges, weights=signs[selected])[0])
    return np.asarray(values)


def legacy_counts(row, layout, centrality, system):
    values = []
    for block in layout:
        n = len(block['edges'])-1
        enabled = system == 'pp' or (block['centrality'][0] <= centrality[0]
                                    and centrality[1] <= block['centrality'][1])
        if not enabled or block.get('source') == 'dr12':
            values.extend([0.]*n)
            continue
        column = block['column_pp' if system == 'pp' else 'column_aa']
        values.extend(feature_counts(row.get(column), block['edges']))
    return np.asarray(values)


def collect(raw, saved, centrality, system, config, limit):
    import pyarrow.parquet as pq
    nodes = config['a_nodes'] if system == 'PbPb' else [0.]
    hadron_blocks = [b for b in config['layout'] if '__hadron__' in b['name']]
    columns = {'event_weight_all', 'event_weight', 'pt_hat'}
    columns.update(b.get('column_pp' if system == 'pp' else 'column_aa', '')
                   for b in config['layout'])
    file = pq.ParquetFile(saved)
    required = {'event_weight_all', 'pt_hat'}
    if not required <= set(file.schema_arrow.names):
        raise ValueError('Saved observables lack alignment fields')
    saved_rows = (r for batch in file.iter_batches(batch_size=256,
                  columns=sorted(columns & set(file.schema_arrow.names))) for r in batch.to_pylist())
    vectors, legacy, fingerprints, qa = [], [], [], {str(a): Counter() for a in nodes}
    maxima = {str(a): Counter() for a in nodes}
    cross_section = None
    started = time.monotonic()
    for event, row in itertools.islice(itertools.zip_longest(pilot.dr.read_events(raw), saved_rows), limit):
        if event is None or row is None:
            raise ValueError('Different raw/saved event coverage')
        weight = float(event['event_weight'])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Invalid weight')
        if row['event_weight_all'] != weight or row['pt_hat'] != event['pt_hat']:
            raise ValueError('Raw/saved event alignment mismatch')
        if row.get('event_weight') is not None and row['event_weight'] != weight:
            raise ValueError('Accepted-event weight differs')
        if system == 'PbPb' and not centrality[0] <= event['centrality'] < centrality[1]:
            raise ValueError('Unexpected event centrality')
        sigma = float(event['cross_section'])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Invalid fixed generator cross section')
        if cross_section is not None and not np.isclose(cross_section, sigma, rtol=1e-6):
            raise ValueError('Cross section changes inside chunk')
        cross_section = sigma
        identities = np.asarray([event[k] for k in FIELDS], float)
        if not np.isfinite(identities).all():
            raise ValueError('Invalid event fingerprint')
        fingerprints.append(identities)
        current = []
        for a in nodes:
            modified, checks = deform(event, a)
            # Undefined reconstruction raises; no event or selected splitting is dropped.
            other, _ = pilot.reconstruction(modified, system, config['jet_blocks'])
            counts = np.r_[hadrons(modified, hadron_blocks, config, centrality, system), other]
            if not np.isfinite(counts).all():
                raise ValueError('Nonfinite counts')
            current.append(weight*counts)
            qa[str(a)].update({k: checks[k] for k in
                ('deleted_positive', 'deleted_negative', 'frozen_spacelike',
                 'no_axis_events', 'degenerate_events', 'conditional_resampling_draws')})
            for key in ('signed_p4_residual', 'mass2_residual', 'correction_rms_radians'):
                maxima[str(a)][key] = max(maxima[str(a)][key], checks[key])
        vectors.append(current)
        legacy.append(weight*legacy_counts(row, config['layout'], centrality, system))
    matrix, old, identity = np.asarray(vectors), np.asarray(legacy), np.asarray(fingerprints)
    if len(identity) != limit or len(np.unique(identity[:, 0])) != limit:
        raise ValueError('Incomplete gate or duplicate event IDs')
    # The first 76 features are hadrons and must reproduce saved production at a=0.
    nhadron = sum(len(b['edges'])-1 for b in hadron_blocks)
    np.testing.assert_array_equal(matrix[:, 0, :nhadron], old[:, :nhadron])
    return matrix, old, identity, dict(events=limit, seconds=time.monotonic()-started,
        cross_section=cross_section, qa={k: dict(v) for k, v in qa.items()},
        maxima={k: dict(v) for k, v in maxima.items()}, hadron_baseline_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', choices=('aa05', 'aa510'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE/'gate_config.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('No overwrite or automatic retry')
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    centrality = [0, 5] if args.tag == 'aa05' else [5, 10]
    matrices, old, records = [], [], []
    common = None
    for design in config['design_ids']:
        metadata = args.source/f'point_{design:02d}_{args.tag}/pooled.json'
        info = json.loads(metadata.read_text())
        if info['design_id'] != design or info['centrality'] != centrality or info['partial']:
            raise ValueError('Wrong source sample identity')
        record = info['records'][0]
        paths = {}
        for key, marker in (('raw', 'final_state_hadrons'), ('saved', '_observables_')):
            matches = [Path(p) for p in record['sources'] if marker in Path(p).name]
            if len(matches) != 1:
                raise ValueError('Ambiguous original path')
            paths[key] = matches[0]
            if sha(paths[key]) != record['sources'][str(paths[key])]:
                raise ValueError('Original input checksum changed')
        vectors, legacy, identity, report = collect(paths['raw'], paths['saved'],
            centrality, 'PbPb', config, config['max_events'])
        if common is None:
            common = identity
        else:
            np.testing.assert_array_equal(common, identity)
        matrices.append(vectors)
        old.append(legacy)
        records.append(dict(design_id=design, metadata_sha256=sha(metadata),
                            sources={str(p): sha(p) for p in paths.values()}, **report))
        print('WAKE_GATE_DESIGN_COMPLETE', args.tag, design, flush=True)
    # Gram factors, not an enormous dense outer product: no correlation is lost.
    events = np.stack(matrices, axis=1)
    legacy = np.stack(old, axis=1)
    np.savez_compressed(args.output/'events.npz', weighted_event_vectors=events,
        legacy_weighted_vectors=legacy, fingerprints=common,
        design_ids=np.array(config['design_ids']), a_nodes=np.array(config['a_nodes']))
    result = dict(status='GATE_COMPLETE_REQUIRES_INTERPOLATION_AND_BASELINE_REVIEW',
        tag=args.tag, records=records, vector_axes=['event', 'design', 'a', 'counts_and_normalizations'],
        covariance_factor='Uncentered weighted event-Poisson Gram factor; paired across all axes',
        global_a=True, config_sha256=sha(args.config), collector_sha256=sha(__file__),
        output_sha256=sha(args.output/'events.npz'), no_new_events=True)
    (args.output/'COMPLETE.json').write_text(json.dumps(result, indent=2)+'\n')
    print('WAKE_GATE_COMPLETE', args.tag, flush=True)


if __name__ == '__main__':
    main()
