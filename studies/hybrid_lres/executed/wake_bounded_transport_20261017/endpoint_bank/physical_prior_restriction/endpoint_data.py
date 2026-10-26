"""Restricted physical designs with unchanged endpoint ratio and noise algebra."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
WORK = ORIGINAL.parents[1]
DESIGN_IDS = [0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22,
              24, 25, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parent = load('_restricted_original_endpoint_data', ORIGINAL/'endpoint_data.py')
bounded, algebra = parent.bounded, parent.algebra
sha, save_json, feature_keys = parent.sha, parent.save_json, parent.feature_keys
NODES = parent.NODES
PP_PROOF_SHA256, PP_SHA256 = parent.PP_PROOF_SHA256, parent.PP_SHA256


def positions(ids, selected):
    """Map physical labels to array positions; never use labels as offsets."""
    ids, selected = np.asarray(ids), np.asarray(selected)
    if (ids.ndim != 1 or selected.ndim != 1 or ids.dtype.kind not in 'iu'
            or (selected.size and selected.dtype.kind not in 'iu')
            or len(set(ids.tolist())) != len(ids) or len(set(selected.tolist())) != len(selected)):
        raise ValueError('Invalid physical-design labels')
    lookup = {int(value): i for i, value in enumerate(ids)}
    if any(int(value) not in lookup for value in selected):
        raise ValueError('Physical design is outside the selected source bank')
    return np.array([lookup[int(value)] for value in selected], dtype=int)


def _events(task):
    return 5630 if task['tag'] == 'aa05' else 4370


def _validated_context(root):
    """Use only the approved controller's pure-local execution lookup."""
    approval = json.loads((root/'APPROVAL.json').read_text())
    prepared = json.loads((root/'PREPARED.json').read_text())
    pins = prepared.get('source_sha256', {})
    required = {str((root/name).relative_to(WORK)) for name in (
        'recovery.py', 'endpoint_data.py', 'endpoint_pipeline.py', 'finish.py',
        'checkpoint_collect.py', 'execute.py', 'PLAN.json', 'config.json', 'CONTROL.json')}
    if (approval.get('approved') is not True or not required <= pins.keys()
            or approval.get('prepared_sha256') != sha(root/'PREPARED.json')
            or approval.get('request_sha256') != sha(root/'RECOVERY_REQUEST.txt')
            or prepared.get('request_sha256') != approval['request_sha256']):
        raise ValueError('Restricted study requires explicit approved local controls')
    for name, digest in pins.items():
        if not bounded._portable(name) or not bounded._digest(digest) or sha(WORK/name) != digest:
            raise ValueError('Changed approved restricted local control: '+str(name))
    controller = load('_restricted_verified_local_controller', root/'recovery.py')
    context = controller.validated_local_context(root)
    jobs = context['actual_job_ids']
    if (set(jobs) != {str(i) for i in range(40)}
            or any(not isinstance(value, str) or not value.isdigit() for value in jobs.values())):
        raise ValueError('Incomplete actual allocation lookup')
    return context


def _sample_proofs(record, events, nhadron, ids):
    if [sample['design_id'] for sample in record['samples']] != list(ids):
        raise ValueError('Unexpected physical-design sample coverage')
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
                or not np.isfinite(sigma) or sigma <= 0 or difference.shape != (134,)
                or not np.isfinite(difference).all() or changed.shape != (134,)
                or changed.dtype.kind not in 'iu' or np.any(changed < 0) or np.any(changed > events)
                or np.any(difference[:nhadron] != 0) or np.any(changed[:nhadron] != 0)):
            raise ValueError('Missing restricted source/baseline/native proof')
        qa = sample.get('qa', {})
        if set(qa) != {'0.0', '1.0'} or any(qa[str(a)].get('whole_event_fallback') != 0 for a in NODES):
            raise ValueError('Missing endpoint zero-fallback QA')
        for row in sample.get('unresolved_dr12_outside_fit', []):
            if any(200 <= jet[0] < 251 for jet in row['jets']):
                raise ValueError('Undefined selected dR12 was not fatal')


def _audit(info, task, events, ids):
    audit, rows = info.get('rg_strict_first_audit'), info.get('rg_precision_events')
    if (not isinstance(audit, dict) or not isinstance(rows, list)
            or any(type(value) is not int or value < 0 for value in audit.values())):
        raise ValueError('Invalid restricted strict audit')
    recovered, n = audit.get('recovered_events', 0), len(ids)
    if (audit.get('strict_transformed_successes') != events*n
            or audit.get('unchanged_successes', 0)+recovered != events*n*2
            or audit.get('retention_failures', 0) != recovered or len(rows) != recovered
            or audit.get('precision_boundary_interventions', 0) < recovered
            or (not recovered and audit.get('precision_boundary_interventions', 0))):
        raise ValueError('Incomplete strict coverage or transformed recovery')
    for row in rows:
        event = row.get('event', {})
        if (event.get('a') != 0 or event.get('tag') != task['tag']
                or event.get('chunk') != task['chunks'][0] or type(event.get('design')) is not int
                or event['design'] not in ids or type(event.get('event_ID')) is not int
                or row.get('trigger') != bounded._RETENTION_ERROR or not row.get('decisions')):
            raise ValueError('Recovery is not an original selected source event')


class EndpointBank(parent.EndpointBank):
    """Full-40 old gates and new 31-design archives, never rewritten metadata."""

    def __init__(self, root=HERE, pp_source=None):
        self.root = Path(root).resolve()
        context = _validated_context(self.root)
        control = json.loads((self.root/'CONTROL.json').read_text())
        self.plan = json.loads((self.root/'PLAN.json').read_text())
        self.config = json.loads((self.root/'config.json').read_text())
        old_plan = json.loads((ORIGINAL/'PLAN.json').read_text())
        old_config = json.loads((ORIGINAL/'config.json').read_text())
        pins = bounded._payload_pins(ORIGINAL, json.loads((ORIGINAL/'APPROVAL.json').read_text()))
        paths = [self.root/name for name in ('PLAN.json', 'config.json', 'CONTROL.json', 'PREPARED.json',
                 'APPROVAL.json', 'SUBMITTED.json', 'DOWNLOADED.json', 'checkpoint_collect.py', 'execute.py', 'recovery.py')]
        paths += [HERE/'endpoint_data.py', HERE/'endpoint_pipeline.py', ORIGINAL/'endpoint_data.py',
                  ORIGINAL/'endpoint_pipeline.py', ORIGINAL/'PLAN.json', ORIGINAL/'config.json',
                  ORIGINAL/'checkpoint_collect.py', ORIGINAL/'payload.tar.gz', ORIGINAL/'APPROVAL.json',
                  ORIGINAL.parent/'bank/bounded_data.py']
        expected_pins = {
            'payload_sha256': ORIGINAL/'payload.tar.gz', 'original_plan_sha256': ORIGINAL/'PLAN.json',
            'original_config_sha256': ORIGINAL/'config.json',
            'original_checkpoint_execution_sha256': ORIGINAL/'checkpoint_collect.py',
            'plan_sha256': self.root/'PLAN.json', 'config_sha256': self.root/'config.json',
            'checkpoint_execution_sha256': self.root/'checkpoint_collect.py'}
        if any(control.get(key) != sha(path) for key, path in expected_pins.items()):
            raise ValueError('Changed original/restricted source or scope bridge')
        if (self.config != dict(old_config, design_ids=DESIGN_IDS)
                or self.plan != dict(old_plan, original_physics_designs=DESIGN_IDS,
                                    checkpoint_execution_sha256=control['checkpoint_execution_sha256'],
                                    physics_design_event_instances=len(DESIGN_IDS)*old_plan['AA_statistics_per_design'])):
            raise ValueError('Only the physical-design subset and helper identity may change')
        self.ids, self.nodes = np.array(DESIGN_IDS), np.array(NODES)
        self.features = feature_keys(self.config)
        self.shape = (len(self.ids), 2, len(self.features))
        if self.shape != (31, 2, 170) or self.features != feature_keys(old_config):
            raise ValueError('Changed restricted feature/node scope')
        receipt_path = self.root/'NATIVE.json'
        if not receipt_path.exists():
            receipt_path = ORIGINAL/'upstream_hardest_recovery/NATIVE.json'
        certificate_path = self.root/'COMPATIBILITY.json'
        if not certificate_path.exists():
            certificate_path = ORIGINAL/'upstream_hardest_recovery/COMPATIBILITY.json'
        receipt, certificate = [json.loads(path.read_text()) for path in (receipt_path, certificate_path)]
        if (sha(receipt_path) != control['native_receipt_sha256']
                or receipt['source_sha256'] != control['policy_source_sha256']
                or sha(certificate_path) != control['compatibility_sha256']
                or certificate['policy'] != 'fastjet-keep-only-hardest-v1'):
            raise ValueError('Changed upstream native or compatibility receipt')
        proofs = {(row['task_id'], row['design_id']): row for row in certificate['checkpoints']}
        if len(proofs) != 937 or len(certificate['checkpoints']) != 937:
            raise ValueError('Incomplete immutable mixed-origin certificate')
        bridge = {label: {key: control[prefix+key] for key in (
            'plan_sha256', 'config_sha256', 'checkpoint_execution_sha256')}
            for label, prefix in (('old', 'original_'), ('new', ''))}
        extension = dict(policy=certificate['policy'], bootstrap_sha256=sha(self.root/'execute.py'),
            policy_source_sha256=control['policy_source_sha256'], native_sha256=receipt['sha256'],
            compatibility_sha256=sha(certificate_path), control_sha256=sha(self.root/'CONTROL.json'),
            physical_prior_design_ids=DESIGN_IDS, scope_bridge=bridge)
        native = dict(control['base_native_sources'], wake_dr12_diagnostic=receipt['sha256'])
        nhadron = sum(len(block['edges'])-1 for block in self.config['layout'] if '__hadron__' in block['name'])
        self.inputs, self.source_ids = [], {}
        reconstruction = self.plan['reconstruction_sources']
        for task in self.plan['tasks']:
            tid = task['id']
            directory = self.root/'download'/f'task_{tid}'
            metadata_path = directory/'CHECKPOINT.json'
            info = json.loads(metadata_path.read_text())
            ids = list(range(40)) if tid < 2 else DESIGN_IDS
            expected = dict(status='COMPLETE', task=task, method_policy=bounded.METHOD_POLICY,
                checkpoint_policy='per-design-atomic-v1', sources_unchanged=True, no_new_events=True,
                global_a=True, reconstruction_sources=reconstruction,
                plan_sha256=control['original_plan_sha256'] if tid < 2 else control['plan_sha256'],
                config_sha256=control['original_config_sha256'] if tid < 2 else control['config_sha256'],
                checkpoint_execution_sha256=control['original_checkpoint_execution_sha256'] if tid < 2
                                            else control['checkpoint_execution_sha256'],
                native_source_sha256=control['base_native_sources'] if tid < 2 else native,
                slurm_job_id=context['actual_job_ids'][str(tid)])
            if any(info.get(key) != value for key, value in expected.items()) or len(info.get('records', [])) != 1:
                raise ValueError('Actual job/source/checkpoint identity differs from approved scope')
            if tid < 2:
                if sha(metadata_path) != control['gate_files'][f'task_{tid}/results/CHECKPOINT.json']:
                    raise ValueError('Changed immutable full-40 gate metadata')
            elif info.get('hardest_root_execution') != extension:
                raise ValueError('Changed restricted hardest-root execution identity')
            sources = bounded._source_map(info.get('method_sources'))
            if (not set(bounded.METHOD_SOURCES) <= sources.keys()
                    or any(pins.get(key) != value for key, value in sources.items())
                    or info.get('collector_sha256') != pins[bounded.COLLECTOR_SOURCE]
                    or info.get('transport_sha256') != pins[bounded.TRANSPORT_SOURCE]
                    or any(key in info for key in ('conservation_sha256', 'conservation_recovery',
                                                   'projection_source_sha256', 'projection_adapter_sha256'))):
                raise ValueError('Changed original physics/collector source proof')
            bounded._native_identity(info, reconstruction)
            rows = info.get('design_checkpoint_records', [])
            if ([row.get('design_id') for row in rows] != ids
                    or any(type(row['design_id']) is not int or not bounded._digest(row.get('sha256')) for row in rows)):
                raise ValueError('Changed checkpoint physical-design coverage')
            if tid >= 2:
                for row in rows:
                    proof = proofs.get((tid, row['design_id']))
                    if proof is not None:
                        path = control['original_remote_root']+'/'+proof['filename']
                        if (row['sha256'] != proof['sha256'] or row.get('filename') != path
                                or row.get('reused_external_checkpoint') is not True):
                            raise ValueError('Changed external immutable checkpoint origin')
                    elif (row.get('reused_external_checkpoint') is not False
                          or row.get('filename') != control['recovery_remote_root']+f'/task_{tid}/results/checkpoints/'
                              f'chunk_{task["chunks"][0]:04d}/design_{row["design_id"]:02d}.npz'):
                        raise ValueError('Changed new checkpoint location/origin')
            record, events = info['records'][0], _events(task)
            name = f'chunk_{task["chunks"][0]:04d}.npz'
            raw = directory/name
            if (record.get('filename') != name or record.get('chunk') != task['chunks'][0]
                    or record.get('events') != events or sha(raw) != record.get('sha256')
                    or record.get('bytes') != raw.stat().st_size
                    or not bounded._digest(record.get('paired_fingerprint_sha256'))):
                raise ValueError('Changed restricted count file/coverage')
            _sample_proofs(record, events, nhadron, ids)
            _audit(info, task, events, ids)
            self.source_ids[tid] = np.array(ids)
            self.inputs.append((task, raw, info))
        pp_source = bounded.LEGACY_ROOT/'pp_reuse' if pp_source is None else Path(pp_source)
        proof_path, self.pp_path = pp_source/'PROOF.json', pp_source/'pp.npz'
        proof = json.loads(proof_path.read_text())
        if (self.plan.get('pp_reuse_proof_sha256') != PP_PROOF_SHA256 or sha(proof_path) != PP_PROOF_SHA256
                or sha(self.pp_path) != PP_SHA256 or proof.get('output_sha256') != PP_SHA256
                or proof.get('complete_pp_chunks') != 161 or proof.get('chunks') != list(range(161))
                or proof.get('raw_total_and_outer_reproduced') is not True):
            raise ValueError('Changed fixed shared-pp proof/moments')
        with np.load(self.pp_path, allow_pickle=False) as archive:
            self.pp = {key: archive[key].copy() for key in archive.files}
        if (self.pp['mean'].shape != (170,) or self.pp['covariance'].shape != (170, 170)
                or not np.isfinite(self.pp['mean']).all() or np.any(self.pp['mean'] <= 0)
                or not np.isfinite(self.pp['covariance']).all()):
            raise ValueError('Invalid fixed shared-pp moments')
        paths += [receipt_path, certificate_path, proof_path, self.pp_path]
        self._controls = {path: sha(path) for path in paths}

    def raw_blocks(self, batch=256):
        unsigned = np.array(['__hadron__' not in name for name, _, _ in self.features]+[True, True])
        for task, path, info in self.inputs:
            source_ids = self.source_ids[task['id']]
            with np.load(path, allow_pickle=False) as archive:
                base, delta, identity = (archive[key] for key in ('baseline_counts', 'delta_counts', 'fingerprints'))
                np.testing.assert_array_equal(archive['design_ids'], source_ids)
                np.testing.assert_array_equal(archive['a_nodes'], self.nodes)
            n, count = _events(task), len(source_ids)
            if (base.shape != (n, count, 172) or delta.shape != (n, count, 1, 172)
                    or base.dtype != np.int16 or delta.dtype != np.int16 or identity.shape != (n, 5)
                    or not np.isfinite(identity).all() or len(np.unique(identity[:, 0])) != n
                    or np.any(identity[:, 1] < 0) or not identity[:, 1].sum() > 0
                    or hashlib.sha256(identity.astype('<f8').tobytes()).hexdigest()
                       != info['records'][0]['paired_fingerprint_sha256']):
                raise ValueError('Malformed physical-design counts/fingerprints')
            columns = positions(source_ids, self.ids)
            for start in range(0, n, batch):
                stop = min(start+batch, n)
                original = base[start:stop, columns].astype(np.int32)
                changed = delta[start:stop, columns].astype(np.int32)
                values = np.concatenate((original[:, :, None], original[:, :, None]+changed), axis=2)
                if np.any(values[..., -1] != 1) or np.any(values[..., unsigned] < 0):
                    raise ValueError('Negative jet/normalization or invalid event count')
                yield int(task['tag'] == 'aa510'), values*identity[start:stop, 1, None, None, None]

    def summarize(self):
        out, identity = self.root/'export', self.input_identity()
        target, marker = out/'means.npz', out/'COMPLETE.json'
        if marker.exists():
            proof = json.loads(marker.read_text())
            if proof['inputs'] != identity or proof['output_sha256'] != sha(target):
                raise ValueError('Changed completed restricted export')
            with np.load(target, allow_pickle=False) as archive:
                result = {key: archive[key].copy() for key in archive.files}
            self.maps = algebra.ratio_maps(result['raw_totals'], self.config, result['cross_sections'], self.pp['mean'])
            return result
        if target.exists():
            raise ValueError('Uncommitted restricted export; preserve it')
        totals, squares = np.zeros((2,)+self.shape[:2]+(172,)), np.zeros((2,)+self.shape[:2]+(172,))
        for i, values in self.raw_blocks():
            totals[i] += values.sum(0)
            squares[i] += (values*values).sum(0)
        sigma = np.full((2, len(self.ids), 1, 1), np.nan)
        for task, _, info in self.inputs:
            i = int(task['tag'] == 'aa510')
            samples = {sample['design_id']: sample for sample in info['records'][0]['samples']}
            values = np.array([samples[int(design)]['fixed_full_cross_section'] for design in self.ids])
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('Invalid selected production cross sections')
            if np.isfinite(sigma[i]).all():
                np.testing.assert_array_equal(sigma[i, :, 0, 0], values)
            sigma[i, :, 0, 0] = values
        self.maps = algebra.ratio_maps(totals, self.config, sigma, self.pp['mean'])
        if np.any(self.maps['mean'] <= 0):
            raise ValueError('Empty restricted prediction bins')
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
        save_json(marker, dict(status='RESTRICTED_PAIRED_ENDPOINT_COUNTS_EXPORTED_NOT_EMULATOR_CERTIFIED',
            inputs=identity, output_sha256=sha(target), physical_design_ids=self.ids.tolist(), events_per_design=200000,
            note='Original exact ratio/event-factor/shared-pp algebra; fixed endpoints only'))
        return result


class FixedEndpointView(parent.FixedEndpointView):
    def __init__(self, bank, a):
        super().__init__(bank, a)
        self.shape = (len(bank.ids), 1, len(bank.features))

    def project_noise(self, retained, omitted, training, targets=(), permutation=None):
        return parent.NoiseProjection.project_noise(self, retained, omitted,
            positions(self.ids, training), positions(self.ids, targets), permutation)


FullBank = EndpointBank
