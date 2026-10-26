"""Paired endpoint validation and unchanged ratio/noise algebra with 80 samples."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('_endpoint_bounded_proofs', HERE.parent/'bank/bounded_data.py')
bounded = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bounded)
algebra = bounded.parent
sha, save_json, feature_keys = bounded.sha, bounded.save_json, bounded.feature_keys
PREFIX = bounded.PREFIX
EXECUTION_SOURCE = PREFIX+'checkpoint_collect.py'
NODES = [0., 1.]
PP_PROOF_SHA256, PP_SHA256 = bounded.PP_PROOF_SHA256, bounded.PP_SHA256


def _sample_proofs(record, events, nhadron):
    if [s['design_id'] for s in record['samples']] != list(range(40)):
        raise ValueError('Unexpected endpoint design coverage')
    for sample in record['samples']:
        sources = sample.get('sources', {})
        difference = np.asarray(sample.get('baseline_weighted_difference', []))
        changed = np.asarray(sample.get('baseline_changed_entries', []))
        sigma = sample.get('fixed_full_cross_section', np.nan)
        if (sample.get('events') != events or sample.get('hadron_baseline_exact') is not True
                or sample.get('baseline_a0_identity_exact') is not True or sample.get('no_events_dropped') is not True
                or sample.get('native_event_node_checks') != events*2 or sample.get('strict_transformed_checks') != events
                or not bounded._digest(sample.get('metadata_sha256')) or len(sources) != 2
                or not all(isinstance(k, str) and bounded._digest(v) for k, v in sources.items())
                or sum('final_state_hadrons' in Path(k).name for k in sources) != 1
                or sum('_observables_' in Path(k).name for k in sources) != 1
                or not np.isfinite(sigma) or sigma <= 0
                or difference.shape != (134,) or not np.isfinite(difference).all()
                or changed.shape != (134,) or changed.dtype.kind not in 'iu'
                or np.any(changed < 0) or np.any(changed > events)
                or np.any(difference[:nhadron] != 0) or np.any(changed[:nhadron] != 0)):
            raise ValueError('Missing endpoint source/baseline/native proof')
        qa = sample.get('qa', {})
        if set(qa) != {'0.0', '1.0'} or any(qa[str(a)].get('whole_event_fallback') != 0 for a in NODES):
            raise ValueError('Missing endpoint zero-fallback QA')
        for row in sample.get('unresolved_dr12_outside_fit', []):
            if any(200 <= jet[0] < 251 for jet in row['jets']):
                raise ValueError('Undefined selected dR12 was not fatal')


def _audit(info, task, events):
    audit, rows = info.get('rg_strict_first_audit'), info.get('rg_precision_events')
    if (not isinstance(audit, dict) or not isinstance(rows, list)
            or any(type(v) is not int or v < 0 for v in audit.values())):
        raise ValueError('Invalid endpoint strict audit')
    recovered = audit.get('recovered_events', 0)
    if (audit.get('strict_transformed_successes') != events*40
            or audit.get('unchanged_successes', 0)+recovered != events*40*2
            or audit.get('retention_failures', 0) != recovered or len(rows) != recovered
            or audit.get('precision_boundary_interventions', 0) < recovered
            or (not recovered and audit.get('precision_boundary_interventions', 0))):
        raise ValueError('Incomplete endpoint strict coverage or transformed recovery')
    for row in rows:
        event = row.get('event', {})
        if (event.get('a') != 0 or event.get('tag') != task['tag']
                or event.get('chunk') != task['chunks'][0]
                or type(event.get('design')) is not int or not 0 <= event['design'] < 40
                or type(event.get('event_ID')) is not int
                or row.get('trigger') != bounded._RETENTION_ERROR or not row.get('decisions')):
            raise ValueError('Recovery is not bound to an original endpoint event')


class NoiseProjection:
    """Same accumulator as the full bank; only flattened sample count differs."""

    def project_noise(self, retained, omitted, training, targets=(), permutation=None):
        training, targets = np.asarray(training), np.asarray(targets)
        samples, features = self.shape[0]*self.shape[1], self.shape[2]
        for rows in (training, targets):
            if (rows.ndim != 1 or (rows.size and not np.issubdtype(rows.dtype, np.integer))
                    or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= samples)):
                raise ValueError('Invalid flattened endpoint sample IDs')
        if not len(training) or np.intersect1d(training, targets).size:
            raise ValueError('Empty training set or overlapping noisy targets')
        if permutation is not None:
            permutation = np.asarray(permutation)
            if not np.array_equal(np.sort(permutation), np.arange(features)):
                raise ValueError('Feature mapping must be a complete permutation')
        if (retained.ndim != 2 or omitted.ndim != 2
                or retained.shape[1] != features or omitted.shape[1] != features):
            raise ValueError('Misaligned endpoint noise projection maps')
        if not hasattr(self, 'maps'):
            self.summarize()
        acc = algebra.NoiseAccumulator(retained, omitted, training, targets, features)
        for factors in self.factor_blocks(permutation):
            acc.add(factors)
        derivative = self.maps['pp_derivative'].reshape(samples, features)
        covariance = self.pp['covariance']
        if permutation is not None:
            derivative = derivative[:, permutation]
            covariance = covariance[np.ix_(permutation, permutation)]
        acc.add_pp(derivative, covariance)
        return acc.result()


class EndpointBank(NoiseProjection, bounded.BoundedBank):
    """Strict two-node source proof; retain paired cross-a noise for shift QA."""

    def __init__(self, root=HERE, pp_source=None):
        self.root = Path(root)
        self.config = json.loads((self.root/'config.json').read_text())
        self.plan = json.loads((self.root/'PLAN.json').read_text())
        pins = bounded._payload_pins(self.root, json.loads((self.root/'APPROVAL.json').read_text()))
        execution = self.plan.get('checkpoint_execution_sha256')
        if (not bounded._digest(execution) or pins.get(EXECUTION_SOURCE) != execution
                or sha(HERE/'checkpoint_collect.py') != execution):
            raise ValueError('Endpoint execution helper differs from approved payload/plan')
        self.features = feature_keys(self.config)
        self.ids, self.nodes = np.asarray(self.config['design_ids'], int), np.asarray(self.config['a_nodes'])
        self.shape = (len(self.ids), len(self.nodes), len(self.features))
        reference = json.loads((bounded.LEGACY_ROOT/'config.json').read_text())
        if (self.shape != (40, 2, 170) or len(set(self.features)) != 170
                or self.features != feature_keys(reference) or self.config['design_ids'] != list(range(40))
                or any(type(i) is not int for i in self.config['design_ids'])
                or self.config['a_nodes'] != NODES or self.config.get('max_events') is not None
                or self.plan.get('original_physics_designs') != list(range(40))
                or self.plan.get('a_nodes') != NODES or self.plan.get('aa_chunks') != list(bounded.CHUNKS)
                or self.plan.get('pp_reused_chunks') != list(range(161))
                or self.plan.get('method_policy') != bounded.METHOD_POLICY
                or self.plan.get('no_new_events') is not True):
            raise ValueError('Changed approved two-endpoint scope')
        reconstruction = bounded._source_map(self.plan.get('reconstruction_sources'))
        names = [PurePosixPath(k).name for k in reconstruction]
        if (names.count('strict_first.py') != 1 or names.count('atlas_dr12.py') != 1 or names.count('atlas_rg.py') != 1
                or sum(n.startswith('_rg_upstream') and n.endswith('.so') for n in names) != 1
                or sum(n.startswith('_rg_precision_candidate') and n.endswith('.so') for n in names) != 1
                or any(reconstruction[k] != pins[k] for k in reconstruction.keys() & pins.keys())):
            raise ValueError('Incomplete or inconsistent native/policy identity')
        coverage = [(t['tag'], tuple(t['chunks'])) for t in self.plan['tasks']]
        if (len(coverage) != 40 or set(coverage) != {(tag, (c,)) for tag in ('aa05', 'aa510') for c in bounded.CHUNKS}
                or [t['id'] for t in self.plan['tasks']] != list(range(40))
                or any(type(t['id']) is not int for t in self.plan['tasks'])):
            raise ValueError('Incomplete endpoint task coverage')
        nhadron = sum(len(b['edges'])-1 for b in self.config['layout'] if '__hadron__' in b['name'])
        self.inputs, methods = [], None
        for task in self.plan['tasks']:
            directory = self.root/'download'/f'task_{task["id"]}'
            info = json.loads((directory/'CHECKPOINT.json').read_text())
            rows = info.get('design_checkpoint_records', [])
            if (info.get('status') != 'COMPLETE' or info.get('task') != task
                    or info.get('plan_sha256') != sha(self.root/'PLAN.json')
                    or info.get('config_sha256') != sha(self.root/'config.json')
                    or len(info.get('records', [])) != 1 or info.get('method_policy') != bounded.METHOD_POLICY
                    or info.get('sources_unchanged') is not True or info.get('no_new_events') is not True
                    or info.get('checkpoint_policy') != 'per-design-atomic-v1'
                    or info.get('checkpoint_execution_sha256') != execution
                    or [r.get('design_id') for r in rows] != list(range(40))
                    or any(type(r.get('design_id')) is not int or not bounded._digest(r.get('sha256')) for r in rows)):
                raise ValueError('Changed/incomplete endpoint checkpoint proof')
            sources = bounded._source_map(info.get('method_sources'))
            if (not set(bounded.METHOD_SOURCES) <= sources.keys() or any(pins.get(k) != v for k, v in sources.items())
                    or (methods is not None and methods != sources)
                    or info.get('collector_sha256') != pins[bounded.COLLECTOR_SOURCE]
                    or info.get('transport_sha256') != pins[bounded.TRANSPORT_SOURCE]
                    or any(k in info for k in ('conservation_sha256', 'conservation_recovery',
                                               'projection_source_sha256', 'projection_adapter_sha256'))):
                raise ValueError('Endpoint method/collector sources differ from payload')
            methods = sources
            if info.get('reconstruction_sources') != reconstruction:
                raise ValueError('Endpoint reconstruction differs from plan')
            bounded._native_identity(info, reconstruction)
            record = info['records'][0]
            events, name = (5630 if task['tag'] == 'aa05' else 4370), f'chunk_{task["chunks"][0]:04d}.npz'
            path = directory/name
            if (record.get('filename') != name or record.get('chunk') != task['chunks'][0]
                    or record.get('events') != events or not bounded._digest(record.get('paired_fingerprint_sha256'))
                    or sha(path) != record.get('sha256') or record.get('bytes') != path.stat().st_size):
                raise ValueError('Changed endpoint count path/coverage/checksum')
            _sample_proofs(record, events, nhadron)
            _audit(info, task, events)
            self.inputs.append((task, path, info))
        pp_source = bounded.LEGACY_ROOT/'pp_reuse' if pp_source is None else Path(pp_source)
        proof_path, self.pp_path = pp_source/'PROOF.json', pp_source/'pp.npz'
        if (self.plan.get('pp_reuse_proof_sha256') != PP_PROOF_SHA256
                or sha(proof_path) != PP_PROOF_SHA256 or sha(self.pp_path) != PP_SHA256):
            raise ValueError('Changed fixed shared-pp reference')
        proof = json.loads(proof_path.read_text())
        if (proof.get('complete_pp_chunks') != 161 or proof.get('chunks') != list(range(161))
                or proof.get('raw_total_and_outer_reproduced') is not True or proof.get('output_sha256') != PP_SHA256):
            raise ValueError('Incomplete exact shared-pp reuse proof')
        with np.load(self.pp_path, allow_pickle=False) as archive:
            self.pp = {k: archive[k].copy() for k in archive.files}
        if (self.pp['mean'].shape != (170,) or self.pp['covariance'].shape != (170, 170)
                or not np.isfinite(self.pp['mean']).all() or np.any(self.pp['mean'] <= 0)
                or not np.isfinite(self.pp['covariance']).all()):
            raise ValueError('Invalid shared-pp moments')
        self._controls = {p: sha(p) for p in (self.root/'PLAN.json', self.root/'config.json',
            self.root/'APPROVAL.json', self.root/'payload.tar.gz', proof_path, self.pp_path,
            HERE/'checkpoint_collect.py', HERE/'endpoint_data.py', HERE.parent/'bank/bounded_data.py')}

    def raw_blocks(self, batch=256):
        unsigned = np.array(['__hadron__' not in name for name, _, _ in self.features]+[True, True])
        for task, path, info in self.inputs:
            with np.load(path, allow_pickle=False) as archive:
                base, delta, identity = (archive[k] for k in ('baseline_counts', 'delta_counts', 'fingerprints'))
                np.testing.assert_array_equal(archive['design_ids'], self.ids)
                np.testing.assert_array_equal(archive['a_nodes'], self.nodes)
            n = 5630 if task['tag'] == 'aa05' else 4370
            if (base.shape != (n, 40, 172) or delta.shape != (n, 40, 1, 172)
                    or base.dtype != np.int16 or delta.dtype != np.int16 or identity.shape != (n, 5)
                    or not np.isfinite(identity).all() or len(np.unique(identity[:, 0])) != n
                    or np.any(identity[:, 1] < 0)):
                raise ValueError('Malformed two-endpoint counts/fingerprints')
            if hashlib.sha256(identity.astype('<f8').tobytes()).hexdigest() != info['records'][0]['paired_fingerprint_sha256']:
                raise ValueError('Changed paired endpoint fingerprints')
            for start in range(0, n, batch):
                stop = min(start+batch, n)
                original = base[start:stop].astype(np.int32)
                values = np.concatenate((original[:, :, None], original[:, :, None]+delta[start:stop].astype(np.int32)), axis=2)
                if np.any(values[..., -1] != 1) or np.any(values[..., unsigned] < 0):
                    raise ValueError('Negative jet/normalization or invalid event count')
                yield int(task['tag'] == 'aa510'), values*identity[start:stop, 1, None, None, None]

    def summarize(self):
        out, identity = self.root/'export', self.input_identity()
        target, marker = out/'means.npz', out/'COMPLETE.json'
        if marker.exists():
            proof = json.loads(marker.read_text())
            if proof['inputs'] != identity or proof['output_sha256'] != sha(target):
                raise ValueError('Changed completed endpoint export')
            with np.load(target, allow_pickle=False) as archive:
                result = {k: archive[k].copy() for k in archive.files}
            self.maps = algebra.ratio_maps(result['raw_totals'], self.config, result['cross_sections'], self.pp['mean'])
            return result
        if target.exists():
            raise ValueError('Uncommitted endpoint export; do not overwrite')
        totals = np.zeros((2,)+self.shape[:2]+(172,))
        squares = np.zeros_like(totals)
        for i, values in self.raw_blocks():
            totals[i] += values.sum(0)
            squares[i] += (values*values).sum(0)
        sigma = np.full((2, len(self.ids), 1, 1), np.nan)
        for task, _, info in self.inputs:
            i = int(task['tag'] == 'aa510')
            values = np.asarray([s['fixed_full_cross_section'] for s in info['records'][0]['samples']])
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('Invalid full-production cross sections')
            if np.isfinite(sigma[i]).all():
                np.testing.assert_array_equal(sigma[i, :, 0, 0], values)
            sigma[i, :, 0, 0] = values
        self.maps = algebra.ratio_maps(totals, self.config, sigma, self.pp['mean'])
        if np.any(self.maps['mean'] <= 0):
            raise ValueError('Empty endpoint prediction bins')
        variance = self.maps['pp_derivative']**2*np.diag(self.pp['covariance'])
        for factors in self.factor_blocks():
            variance += (factors*factors).sum(0).reshape(self.shape)
        result = dict(raw_totals=totals, raw_sumw2=squares, cross_sections=sigma,
            central_values=self.maps['mean'], measured_variance=variance, pp_mean=self.pp['mean'],
            pp_covariance=self.pp['covariance'], design_ids=self.ids, a_nodes=self.nodes, features=np.asarray(self.features, str))
        out.mkdir(exist_ok=True)
        pending = target.with_name('means.pending.npz')
        with pending.open('xb') as stream:
            np.savez_compressed(stream, **result)
        target.hardlink_to(pending)
        pending.unlink()
        save_json(marker, dict(status='PAIRED_ENDPOINT_COUNTS_EXPORTED_NOT_EMULATOR_CERTIFIED',
            inputs=identity, output_sha256=sha(target), events_per_design=200000,
            old_reference_sampling_identical=False, note='Exact paired a0/a1 factors retained; no nuisance integration'))
        return result

    def factor_blocks(self, permutation=None, batch=128):
        if not hasattr(self, 'maps'):
            self.summarize()
        for i, values in self.raw_blocks(batch):
            factors = algebra.aa_factors(values, i, self.maps).reshape(len(values), self.shape[0]*self.shape[1], self.shape[2])
            yield factors if permutation is None else factors[..., permutation]


class FixedEndpointView(NoiseProjection):
    """Select a real endpoint, never interpolate or treat it as a nuisance axis."""

    def __init__(self, bank, a):
        if isinstance(a, bool) or a not in (0, 1) or not np.array_equal(bank.nodes, NODES):
            raise ValueError('Only fixed a=0 or a=1 is supported')
        self.bank, self.a, self.index = bank, int(a), int(a)
        self.root, self.features, self.ids, self.config, self.pp = bank.root, bank.features, bank.ids, bank.config, bank.pp
        self.nodes, self.shape = np.array([float(a)]), (40, 1, 170)

    def summarize(self):
        result = dict(self.bank.summarize())
        for key in ('raw_totals', 'raw_sumw2'):
            result[key] = result[key][:, :, self.index:self.index+1]
        for key in ('central_values', 'measured_variance'):
            result[key] = result[key][:, self.index:self.index+1]
        result['a_nodes'] = self.nodes
        self.maps = dict(self.bank.maps, pp_derivative=self.bank.maps['pp_derivative'][:, self.index:self.index+1])
        return result

    def factor_blocks(self, permutation=None, batch=128):
        for factors in self.bank.factor_blocks(permutation, batch):
            yield factors[:, self.index::2]

    def input_identity(self):
        return dict(self.bank.input_identity(), fixed_endpoint_a=self.a)


FullBank = EndpointBank
