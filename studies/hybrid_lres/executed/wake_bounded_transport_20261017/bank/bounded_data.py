"""Bounded-source validation around the unchanged full-bank export algebra.

Collector API: method_policy == METHOD_POLICY; method_sources maps portable
WORK-relative source paths to SHA256. PLAN and checkpoints carry the same
reconstruction_sources map (portable strict/policy paths, never task temp paths).
Existing task/record/source proofs and rg_strict_first_audit remain mandatory.
Source validation, signed-hadron validation, and input identity differ from the
original FullBank; count decoding and export/noise algebra remain unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
LEGACY_ROOT = WORK/'wake_nuisance_20261017/full_bank'
sys.path.insert(0, str(LEGACY_ROOT))
_spec = importlib.util.spec_from_file_location('_bounded_original_full_data', LEGACY_ROOT/'full_data.py')
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
if Path(parent.ratio_maps.__code__.co_filename).resolve() != LEGACY_ROOT/'ratio_factors.py':
    raise ImportError('Unexpected full-bank ratio algebra module')

sha, save_json, feature_keys = parent.sha, parent.save_json, parent.feature_keys
METHOD_POLICY = 'bounded-separate-sign-v1'
PREFIX = 'wake_bounded_transport_20261017/bank/'
COLLECTOR_SOURCE = PREFIX+'collect_bounded.py'
TRANSPORT_SOURCE = 'wake_bounded_transport_20261017/transport.py'
METHOD_SOURCES = (
    COLLECTOR_SOURCE, PREFIX+'verify_runtime.py', TRANSPORT_SOURCE,
    'wake_bounded_transport_20261017/thinning.py',
    'wake_bounded_transport_20261017/bounded_projection.py',
    'wake_sign_separated_20261017/projection.py',
    'wake_nuisance_20261017/global_a.py',
    'wake_paper_morph_20261017/transport.py',
    'wake_paper_morph_20261017/calibrate.py',
    'wake_paper_morph_20261017/CALIBRATION.json',
    'wake_sensitivity_20261017/deform.py',
    'wake_sensitivity_20261017/run_study.py',
    'wake_nuisance_20261017/collect_gate.py',
    'wake_nuisance_20261017/full_bank/collect_full.py',
    'wake_nuisance_20261017/full_bank/compact.py',
    'wake_nuisance_20261017/full_bank/storage.py',
)
CHUNKS = (7, 12, 23, 35, 46, 54, 62, 74, 86, 90, 104, 117, 128, 130, 143, 158, 164, 177, 180, 198)
PP_PROOF_SHA256 = 'eee4ccabba12fd44b2e4e1acef47dc92517cbf3b80f71a7ae37892e45f3748a1'
PP_SHA256 = '5d9fdf11693f292cf46b783eaa0d86edc4ad90dcea3b63504abed40e79bfc3f1'
_DIGEST = re.compile(r'[0-9a-f]{64}')
_RETENTION_ERROR = 'Upstream C/A reclustering lost jet constituents'


def _digest(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _portable(name):
    return (isinstance(name, str) and bool(name) and not name.startswith('/')
            and '\\' not in name and str(PurePosixPath(name)) == name
            and not any(part in ('.', '..') for part in name.split('/')))


def _source_map(value):
    if (not isinstance(value, dict) or not value
            or any(not _portable(k) or not _digest(v) for k, v in value.items())):
        raise ValueError('Invalid portable source identity map')
    return value


def _payload_pins(root, approved):
    payload = root/'payload.tar.gz'
    if approved.get('approved') is not True or sha(payload) != approved.get('approved_payload_sha256'):
        raise ValueError('Unapproved or changed source payload')
    with tarfile.open(payload) as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError('Duplicate payload member')
        manifest = archive.extractfile('runtime/SOURCE_SHA256SUMS')
        if manifest is None:
            raise ValueError('Missing payload source manifest')
        pins = {}
        for line in manifest.read().decode().splitlines():
            parts = line.split('  ', 1)
            if (len(parts) != 2 or not _digest(parts[0]) or not _portable(parts[1])
                    or parts[1] in pins):
                raise ValueError('Invalid or duplicate payload source pin')
            pins[parts[1]] = parts[0]
        required = set(METHOD_SOURCES) | {PREFIX+'PLAN.json', PREFIX+'config.json'}
        if not required <= pins.keys():
            raise ValueError('Payload is not the complete bounded-method payload')
        for name, expected in pins.items():
            member = archive.getmember('runtime/'+name)
            if not member.isfile():
                raise ValueError('Source pin is not an ordinary payload file')
            with archive.extractfile(member) as stream:
                if parent.hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Payload source bytes disagree with manifest: '+name)
    for name in ('PLAN.json', 'config.json'):
        if sha(root/name) != pins[PREFIX+name]:
            raise ValueError('Local inputs differ from approved payload: '+name)
    return pins


def _strict_audit(info, task, events):
    audit, recoveries = info.get('rg_strict_first_audit'), info.get('rg_precision_events')
    if (not isinstance(audit, dict) or not isinstance(recoveries, list)
            or any(type(v) is not int or v < 0 for v in audit.values())):
        raise ValueError('Missing/invalid strict reconstruction audit')
    recovered = audit.get('recovered_events', 0)
    if (audit.get('strict_transformed_successes') != events*40*4
            or audit.get('unchanged_successes', 0)+recovered != events*40*5
            or audit.get('retention_failures', 0) != recovered
            or recovered != len(recoveries)
            or audit.get('precision_boundary_interventions', 0) < recovered
            or (not recovered and audit.get('precision_boundary_interventions', 0))):
        raise ValueError('Incomplete strict reconstruction or transformed fallback')
    for row in recoveries:
        event = row.get('event', {})
        if (event.get('a') != 0 or event.get('tag') != task['tag']
                or event.get('chunk') != task['chunks'][0]
                or type(event.get('design')) is not int or not 0 <= event['design'] < 40
                or type(event.get('event_ID')) is not int
                or row.get('trigger') != _RETENTION_ERROR or not row.get('decisions')):
            raise ValueError('Precision recovery is not bound to an original source event')


def _native_identity(info, reconstruction):
    native = info.get('native_source_sha256', {})
    if not isinstance(native, dict) or any(not _digest(v) for v in native.values()):
        raise ValueError('Missing/invalid loaded native identity')
    for label, prefix in (('strict_first.py', 'strict_first.py'),
                          ('rg_strict_backend', '_rg_upstream'),
                          ('rg_precision_candidate', '_rg_precision_candidate')):
        expected = [value for path, value in reconstruction.items()
                    if PurePosixPath(path).name.startswith(prefix)]
        if len(expected) != 1 or native.get(label) != expected[0]:
            raise ValueError('Loaded strict native/policy differs from approved identity')


def _sample_proofs(record, events, nhadron):
    samples = record['samples']
    if [s['design_id'] for s in samples] != list(range(40)):
        raise ValueError('Unexpected count design coverage')
    for sample in samples:
        sources = sample.get('sources', {})
        difference = np.asarray(sample.get('baseline_weighted_difference', []))
        changed = np.asarray(sample.get('baseline_changed_entries', []))
        sigma = sample.get('fixed_full_cross_section', np.nan)
        if (sample.get('events') != events or sample.get('hadron_baseline_exact') is not True
                or sample.get('baseline_a0_identity_exact') is not True
                or sample.get('no_events_dropped') is not True
                or sample.get('native_event_node_checks') != events*5
                or sample.get('strict_transformed_checks') != events*4
                or not _digest(sample.get('metadata_sha256')) or len(sources) != 2
                or not all(isinstance(k, str) and _digest(v) for k, v in sources.items())
                or sum('final_state_hadrons' in Path(k).name for k in sources) != 1
                or sum('_observables_' in Path(k).name for k in sources) != 1
                or not np.isfinite(sigma) or sigma <= 0
                or difference.shape != (134,) or not np.isfinite(difference).all()
                or changed.shape != (134,) or changed.dtype.kind not in 'iu'
                or np.any(changed < 0) or np.any(changed > events)
                or np.any(difference[:nhadron] != 0) or np.any(changed[:nhadron] != 0)):
            raise ValueError('Missing or changed original source/baseline proof')
        for node in (0., .25, .5, .75, 1.):
            qa = sample.get('qa', {}).get(str(node), {})
            if qa.get('whole_event_fallback') != 0:
                raise ValueError('Missing zero-fallback bounded QA')
        for row in sample.get('unresolved_dr12_outside_fit', []):
            if any(200 <= jet[0] < 251 for jet in row['jets']):
                raise ValueError('Undefined selected dR12 was not fatal')


class BoundedBank(parent.FullBank):
    """Keep original count reconstruction, export, and projected noise unchanged."""

    def __init__(self, root=HERE, pp_source=None):
        self.root = Path(root)
        self.config = json.loads((self.root/'config.json').read_text())
        self.plan = json.loads((self.root/'PLAN.json').read_text())
        approved = json.loads((self.root/'APPROVAL.json').read_text())
        pins = _payload_pins(self.root, approved)
        self.features = feature_keys(self.config)
        self.ids = np.asarray(self.config['design_ids'], int)
        self.nodes = np.asarray(self.config['a_nodes'])
        self.shape = (len(self.ids), len(self.nodes), len(self.features))
        reference = json.loads((LEGACY_ROOT/'config.json').read_text())
        if (self.plan.get('method_policy') != METHOD_POLICY
                or self.shape != (40, 5, 170) or len(set(self.features)) != 170
                or self.features != feature_keys(reference)
                or self.config['design_ids'] != list(range(40))
                or any(type(i) is not int for i in self.config['design_ids'])
                or not np.array_equal(self.nodes, np.linspace(0, 1, 5))
                or self.config.get('max_events') is not None
                or self.plan.get('original_physics_designs') != list(range(40))
                or self.plan.get('a_nodes') != [0., .25, .5, .75, 1.]
                or self.plan.get('aa_chunks') != list(CHUNKS)
                or self.plan.get('pp_reused_chunks') != list(range(161))
                or self.plan.get('global_a') is not True or self.plan.get('no_new_events') is not True):
            raise ValueError('Changed approved bounded full-bank scope')
        reconstruction = _source_map(self.plan.get('reconstruction_sources'))
        names = [PurePosixPath(k).name for k in reconstruction]
        if (names.count('strict_first.py') != 1
                or names.count('atlas_dr12.py') != 1 or names.count('atlas_rg.py') != 1
                or sum(n.startswith('_rg_upstream') and n.endswith('.so') for n in names) != 1
                or sum(n.startswith('_rg_precision_candidate') and n.endswith('.so') for n in names) != 1):
            raise ValueError('Incomplete strict-native/policy source identity')
        for name in reconstruction.keys() & pins.keys():
            if reconstruction[name] != pins[name]:
                raise ValueError('Reconstruction identity contradicts payload source pin')
        coverage = [(t['tag'], tuple(t['chunks'])) for t in self.plan['tasks']]
        expected = {(tag, (chunk,)) for tag in ('aa05', 'aa510') for chunk in CHUNKS}
        if (len(coverage) != 40 or set(coverage) != expected
                or [t['id'] for t in self.plan['tasks']] != list(range(40))
                or any(type(t['id']) is not int for t in self.plan['tasks'])):
            raise ValueError('Incomplete or duplicate AA task coverage')
        nhadron = sum(len(b['edges'])-1 for b in self.config['layout'] if '__hadron__' in b['name'])
        self.inputs, method_sources = [], None
        for task in self.plan['tasks']:
            directory = self.root/'download'/f'task_{task["id"]}'
            info = json.loads((directory/'CHECKPOINT.json').read_text())
            if (info.get('status') != 'COMPLETE' or info.get('task') != task
                    or info.get('plan_sha256') != sha(self.root/'PLAN.json')
                    or info.get('config_sha256') != sha(self.root/'config.json')
                    or len(info.get('records', [])) != 1
                    or info.get('method_policy') != METHOD_POLICY
                    or info.get('sources_unchanged') is not True
                    or info.get('global_a') is not True or info.get('no_new_events') is not True):
                raise ValueError('Changed/incomplete bounded downloaded task')
            sources = _source_map(info.get('method_sources'))
            if (not set(METHOD_SOURCES) <= sources.keys()
                    or any(pins.get(k) != v for k, v in sources.items())
                    or (method_sources is not None and sources != method_sources)
                    or info.get('collector_sha256') != pins[COLLECTOR_SOURCE]
                    or ('transport_sha256' in info and info['transport_sha256'] != pins[TRANSPORT_SOURCE])
                    or any(k in info for k in ('conservation_sha256', 'conservation_recovery',
                                               'projection_source_sha256', 'projection_adapter_sha256'))):
                raise ValueError('Bounded method/collector sources differ from approved payload')
            method_sources = sources
            if info.get('reconstruction_sources') != reconstruction:
                raise ValueError('Strict reconstruction sources differ from approved plan')
            _native_identity(info, reconstruction)
            record = info['records'][0]
            events = 5630 if task['tag'] == 'aa05' else 4370
            name = f'chunk_{task["chunks"][0]:04d}.npz'
            if (record.get('filename') != name or record.get('chunk') != task['chunks'][0]
                    or record.get('events') != events or not _digest(record.get('sha256'))
                    or not _digest(record.get('paired_fingerprint_sha256'))):
                raise ValueError('Unexpected count path/event coverage')
            raw = directory/name
            if sha(raw) != record['sha256'] or record.get('bytes') != raw.stat().st_size:
                raise ValueError('Changed compressed count checksum/size')
            _sample_proofs(record, events, nhadron)
            _strict_audit(info, task, events)
            self.inputs.append((task, raw, info))
        pp_source = LEGACY_ROOT/'pp_reuse' if pp_source is None else Path(pp_source)
        proof_path = pp_source/'PROOF.json'
        if (self.plan.get('pp_reuse_proof_sha256') != PP_PROOF_SHA256
                or sha(proof_path) != PP_PROOF_SHA256):
            raise ValueError('Shared pp proof differs from fixed approved reference')
        proof = json.loads(proof_path.read_text())
        self.pp_path = pp_source/'pp.npz'
        if (proof.get('complete_pp_chunks') != 161 or proof.get('chunks') != list(range(161))
                or proof.get('raw_total_and_outer_reproduced') is not True
                or proof.get('output_sha256') != PP_SHA256 or sha(self.pp_path) != PP_SHA256):
            raise ValueError('Changed full-pp reuse proof')
        with np.load(self.pp_path, allow_pickle=False) as pp:
            self.pp = {k: pp[k].copy() for k in pp.files}
        if (self.pp['mean'].shape != (170,) or self.pp['covariance'].shape != (170, 170)
                or not np.isfinite(self.pp['mean']).all() or np.any(self.pp['mean'] <= 0)
                or not np.isfinite(self.pp['covariance']).all()):
            raise ValueError('Invalid fixed shared pp moments')
        self._controls = {p: sha(p) for p in (self.root/'PLAN.json', self.root/'config.json',
            self.root/'APPROVAL.json', self.root/'payload.tar.gz', proof_path, self.pp_path)}

    def raw_blocks(self, batch=256):
        # The parent decoder rejects all negative entries. Native hadron counts
        # are signed; only jets and the two normalization columns are unsigned.
        nonnegative = np.array(['__hadron__' not in name for name, _, _ in self.features]
                               + [True, True])
        for task, path, info in self.inputs:
            with np.load(path) as z:
                base, delta = z['baseline_counts'], z['delta_counts']
                identities = z['fingerprints']
                np.testing.assert_array_equal(z['design_ids'], self.ids)
                np.testing.assert_array_equal(z['a_nodes'], self.nodes)
            events = 5630 if task['tag'] == 'aa05' else 4370
            if (base.shape != (events, 40, 172) or delta.shape != (events, 40, 4, 172)
                    or base.dtype != np.dtype('int16') or delta.dtype != np.dtype('int16')
                    or identities.ndim != 2 or identities.shape != (events, 5)
                    or not np.isfinite(identities).all()
                    or len(np.unique(identities[:, 0])) != events):
                raise ValueError('Incomplete/reordered event, design or node coverage')
            fingerprint = hashlib.sha256(identities.astype('<f8').tobytes()).hexdigest()
            if fingerprint != info['records'][0]['paired_fingerprint_sha256']:
                raise ValueError('Changed paired event identities/weights')
            weights = identities[:, 1]
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError('Invalid event weights')
            centrality = 0 if task['tag'] == 'aa05' else 1
            for start in range(0, events, batch):
                stop = min(start+batch, events)
                original = base[start:stop].astype(np.int32)
                values = np.concatenate([original[:, :, None],
                    original[:, :, None]+delta[start:stop].astype(np.int32)], axis=2)
                np.testing.assert_array_equal(values[..., -1], 1)
                if np.any(values[..., nonnegative] < 0):
                    raise ValueError('Negative jet or normalization count')
                yield centrality, values*weights[start:stop, None, None, None]

    def input_identity(self):
        for path, expected in self._controls.items():
            if sha(path) != expected:
                raise ValueError('Changed validated bounded input: '+str(path))
        identity = super().input_identity()
        identity.update({str(path.resolve()): value for path, value in self._controls.items()})
        for path in (Path(__file__), LEGACY_ROOT.parent/'normalization.py'):
            identity['adapter:'+path.name] = sha(path)
        return identity


FullBank = BoundedBank
