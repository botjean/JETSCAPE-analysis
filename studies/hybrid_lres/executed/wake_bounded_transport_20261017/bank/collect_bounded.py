"""Source-pinned bounded full-event bank, preserving native/alignment contracts."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
STUDY = WORK / 'wake_nuisance_20261017'
FULL = STUDY / 'full_bank'
sys.path[:0] = [str(STUDY), str(FULL), str(WORK / 'wake_sensitivity_20261017'),
               str(WORK / 'wake_paper_morph_20261017'), str(WORK / 'JETSCAPE-analysis-atlas-dr12')]
spec = importlib.util.spec_from_file_location('_bounded_bank_runtime', HERE / 'verify_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
METHOD_POLICY = runtime.METHOD_POLICY
method_sources = runtime.method_sources
_source_at_import = method_sources()
bounded = runtime.load_bounded()
deform = bounded.deform
from collect_gate import hadrons, legacy_counts, sha, FIELDS
from compact import pack
from storage import write_checked
import run_study as pilot
from jetscape_analysis.analysis import atlas_dr12 as reader

for module, path in (
        (sys.modules[hadrons.__module__], STUDY/'collect_gate.py'),
        (sys.modules[pack.__module__], FULL/'compact.py'),
        (sys.modules[write_checked.__module__], FULL/'storage.py'),
        (pilot, WORK/'wake_sensitivity_20261017/run_study.py'),
        (reader, WORK/'JETSCAPE-analysis-atlas-dr12/jetscape_analysis/analysis/atlas_dr12.py')):
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError(f'Collector imported an unpackaged helper: {path.name}')

FLAGS = ('baseline_identity', 'no_axis_events', 'degenerate_events', 'population_restored',
         'budget_relaxed', 'whole_event_fallback')
TOTALS = ('frozen_spacelike', 'frozen_zero_mt', 'restored_rows', 'original_positive',
          'original_negative', 'selected_positive', 'selected_negative',
          'deleted_positive', 'deleted_negative')
METRICS = ('requested_negative_fraction', 'actual_negative_fraction',
           'pt_damping_positive', 'pt_damping_negative', 'pt_envelope_ratio',
           'unsigned_energy_ratio', 'signed_p4_residual', 'mass2_residual',
           'sign_budget_residual', 'correction_rms_radians', 'maximum_sign_correction',
           'maximum_charged_wake_pt', 'maximum_wake_pt')


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


def native_sources(wrapper, reference):
    import strict_first
    import _wake_dr12_diagnostic as diagnostic
    reference = Path(reference)
    precision, = reference.glob('_rg_precision_candidate*.so')
    if sha(strict_first.__file__) != sha(reference / 'strict_first.py') or sha(wrapper.precision.__file__) != sha(precision):
        raise RuntimeError('Loaded native precision/policy differs from --reference')
    return {key: sha(path) for key, path in {
        'strict_first.py': strict_first.__file__, 'rg_strict_backend': wrapper.strict.__file__,
        'rg_precision_candidate': wrapper.precision.__file__,
        'wake_dr12_diagnostic': diagnostic.__file__, 'wake_native': pilot.native.__file__,
    }.items()}


def reconstruction_sources(wrapper, reference):
    """Approved native-reference identities, independent of task temp paths."""
    native_sources(wrapper, reference)
    paths = {
        'native_reference/strict_first.py': Path(reference) / 'strict_first.py',
        'native_reference/' + Path(wrapper.strict.__file__).name: wrapper.strict.__file__,
        'native_reference/' + Path(wrapper.precision.__file__).name: wrapper.precision.__file__,
        'JETSCAPE-analysis-atlas-dr12/jetscape_analysis/analysis/atlas_dr12.py': reader.__file__,
        'JETSCAPE-analysis-atlas-dr12/jetscape_analysis/analysis/atlas_rg.py': pilot.rg.__file__,
    }
    return {name: sha(path) for name, path in paths.items()}


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
    if not np.array_equal(nodes, runtime.NODES if system == 'PbPb' else [0.]):
        raise ValueError('Wrong canonical a-node coverage')
    hadron_blocks = [b for b in config['layout'] if '__hadron__' in b['name']]
    nhadron = sum(len(b['edges'])-1 for b in hadron_blocks)
    columns = {'event_weight_all', 'event_weight', 'pt_hat'}
    columns.update(b.get('column_pp' if system == 'pp' else 'column_aa', '') for b in config['layout'])
    schema = set(obs_file.schema_arrow.names)
    if not {'event_weight_all', 'pt_hat'} <= schema:
        raise ValueError('Missing mandatory saved-event alignment fields')
    obs = (r for batch in obs_file.iter_batches(batch_size=256, columns=sorted(columns & schema))
           for r in batch.to_pylist())
    counts, identity = np.empty((n, len(nodes), 172), np.int16), np.empty((n, len(FIELDS)))
    baseline_diff, baseline_changed = np.zeros(134), np.zeros(134, np.int64)
    native_only = np.array([b.get('source') == 'dr12' for b in config['layout'] for _ in b['edges'][:-1]])
    qa, maxima, weighted = [{str(a): Counter() for a in nodes} for _ in range(3)]
    minima = {str(a): {key: float('inf') for key in METRICS} for a in nodes}
    outside_selected_dr12, chunk_sigma, total_weight = [], None, 0.
    started, events = time.monotonic(), 0
    audit_start = wrapper.audit.get('strict_transformed_successes', 0) if wrapper is not None else 0
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
        total_weight += weight
        for node, a in enumerate(nodes):
            event_context = dict(context or {}, event_ID=event['event_ID'], a=a)
            if wrapper is not None:
                wrapper.context = event_context
            if unresolved is not None:
                unresolved.clear()
            try:
                modified, checks = deform(event, a)
                if a == 0 and (modified is not event or checks['baseline_identity'] != 1):
                    raise AssertionError('Original a=0 is not exact identity')
                if checks['whole_event_fallback']:
                    raise AssertionError('Undeclared whole-event fallback')
                other, _ = pilot.reconstruction(modified, system, config['jet_blocks'])
            except Exception as error:
                raise RuntimeError(f'Bounded/native failure {event_context}: {error}') from error
            vector = np.r_[hadrons(modified, hadron_blocks, config, centrality, system), other]
            if vector.shape != (172,) or not np.isfinite(vector).all() or np.any(vector != np.rint(vector)):
                raise ValueError('Noninteger or malformed native count vector')
            if np.max(np.abs(vector)) > np.iinfo(np.int16).max or vector[-1] != 1:
                raise ValueError('Count range/normalization contract failed')
            counts[events, node] = vector
            if unresolved:
                outside_selected_dr12.append(dict(event_context, jets=list(unresolved)))
            # Include frozen/passthrough tracks, which are absent from core's
            # selected-particle maxima, so reported spectra cannot hide tails.
            checks = dict(checks)
            wake = np.isin(np.asarray(modified['status']), (1, 2))
            charged = np.isin(abs(np.asarray(modified['particle_ID'])), bounded.CHARGED_PIDS)
            pt = np.hypot(modified['px'], modified['py'])
            checks['maximum_wake_pt'] = float(pt[wake].max(initial=0))
            checks['maximum_charged_wake_pt'] = float(pt[wake & charged].max(initial=0))
            checks['correction_above_0.5'] = int(checks['correction_rms_radians'] > .5)
            checks['correction_above_1.0'] = int(checks['correction_rms_radians'] > 1.)
            flag_keys = FLAGS + ('correction_above_0.5', 'correction_above_1.0')
            for key in flag_keys + TOTALS + METRICS:
                value = float(checks[key])
                if not np.isfinite(value):
                    raise ValueError(f'Nonfinite bounded QA metric: {key}')
                if key in flag_keys and value not in (0., 1.):
                    raise ValueError(f'Nonboolean bounded QA flag: {key}')
                if key in flag_keys + TOTALS:
                    if value < 0 or value != int(value):
                        raise ValueError(f'Invalid bounded QA row count: {key}')
                    qa[str(a)][key] += int(value)
                weighted[str(a)][key] += weight*value
                maxima[str(a)][key] = max(maxima[str(a)][key], value)
                if key in METRICS:
                    minima[str(a)][key] = min(minima[str(a)][key], value)
        old = legacy_counts(row, config['layout'], centrality, system)
        np.testing.assert_array_equal(counts[events, 0, :nhadron], old[:nhadron])
        diff = counts[events, 0, :134]-old
        diff[native_only] = 0
        baseline_diff += weight*diff
        baseline_changed += diff != 0
        events += 1
    if events != n or len(np.unique(identity[:, 0])) != n:
        raise ValueError('Missing/duplicate source events')
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError('Nonpositive/nonfinite total event weight')
    expected_strict = n*(len(nodes)-1)
    if wrapper is not None:
        wrapper.context = None
        if wrapper.audit.get('strict_transformed_successes', 0)-audit_start != expected_strict:
            raise AssertionError('Incomplete strict transformed native coverage')
    flag_keys = FLAGS + ('correction_above_0.5', 'correction_above_1.0')
    return counts, identity, dict(events=events, seconds=time.monotonic()-started,
        input_chunk_cross_section=chunk_sigma, baseline_weighted_difference=baseline_diff.tolist(),
        baseline_changed_entries=baseline_changed.tolist(), hadron_baseline_exact=True,
        qa={k: dict(v) for k, v in qa.items()}, maxima={k: dict(v) for k, v in maxima.items()},
        minima=minima, total_event_weight=total_weight,
        event_weighted_sums={k: dict(v) for k, v in weighted.items()},
        event_weighted_means={k: {key: value/total_weight for key, value in v.items()} for k, v in weighted.items()},
        event_weighted_flag_fractions={k: {key: v[key]/total_weight for key in flag_keys} for k, v in weighted.items()},
        native_event_node_checks=n*len(nodes), strict_transformed_checks=expected_strict,
        baseline_a0_identity_exact=True, no_events_dropped=True,
        unresolved_dr12_outside_fit=outside_selected_dr12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_path, config_path = HERE/'PLAN.json', HERE/'config.json'
    plan, config = json.loads(plan_path.read_text()), json.loads(config_path.read_text())
    task = next(t for t in plan['tasks'] if t['id'] == args.task)
    pins, plan_sha, config_sha = method_sources(), sha(plan_path), sha(config_path)
    if pins != _source_at_import:
        raise RuntimeError('Method source changed after collector import')
    args.output.mkdir(parents=True, exist_ok=False)
    wrapper, unresolved = install_reconstruction(args.reference)
    native_pins = native_sources(wrapper, args.reference)
    reconstruction_pins = reconstruction_sources(wrapper, args.reference)
    if plan.get('reconstruction_sources') != reconstruction_pins:
        raise RuntimeError('Loaded reconstruction sources differ from the approved plan')
    records = []
    for chunk in task['chunks']:
        common = None
        designs = [None] if task['tag'] == 'pp' else config['design_ids']
        matrices, sample_records = None, []
        for column, design in enumerate(designs):
            paths, info, source = source_paths(Path(plan['source_bank']), task['tag'], design, chunk)
            counts, identity, report = collect_one(paths['raw'], paths['saved'], config,
                info['system'], info['centrality'], context=dict(tag=task['tag'], chunk=chunk, design=design),
                wrapper=wrapper, unresolved=unresolved)
            if source['sources'] != {str(p): sha(p) for p in paths.values()}:
                raise RuntimeError('Input source changed during collection')
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
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **arrays)
            write_checked(path, buffer.getbuffer(), args.output.parent.parent, plan['storage_bytes_cap'])
        records.append(dict(chunk=chunk, filename=path.name, sha256=sha(path), bytes=path.stat().st_size,
            events=len(common), samples=sample_records,
            paired_fingerprint_sha256=hashlib.sha256(common.astype('<f8').tobytes()).hexdigest()))
        if pins != method_sources() or native_pins != native_sources(wrapper, args.reference):
            raise RuntimeError('Method/native source changed during collection')
        if plan_sha != sha(plan_path) or config_sha != sha(config_path):
            raise RuntimeError('Plan/config source changed during collection')
        print('WAKE_FULL_CHUNK_COMPLETE', task['id'], chunk, path.stat().st_size, flush=True)
        del arrays, matrices, counts
    metadata = dict(status='COMPLETE', task=task, records=records, global_a=True, no_new_events=True,
        plan_sha256=plan_sha, config_sha256=config_sha, collector_sha256=sha(__file__),
        transport_sha256=pins['wake_bounded_transport_20261017/transport.py'],
        method_policy=METHOD_POLICY, method_sources=pins, native_source_sha256=native_pins,
        reconstruction_sources=reconstruction_pins,
        sources_unchanged=True, rg_strict_first_audit=dict(wrapper.audit), rg_precision_events=wrapper.events,
        slurm_job_id=__import__('os').environ.get('SLURM_JOB_ID'))
    write_checked(args.output/'CHECKPOINT.json', (json.dumps(metadata, indent=2, allow_nan=False)+'\n').encode(),
                  args.output.parent.parent, plan['storage_bytes_cap'])
    print('WAKE_FULL_TASK_COMPLETE', task['id'], flush=True)


if __name__ == '__main__':
    main()
