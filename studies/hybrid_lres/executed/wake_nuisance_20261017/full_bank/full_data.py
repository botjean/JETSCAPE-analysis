"""Validated, lossless full-bank export and re-iterable measured-noise factors."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from ratio_factors import feature_keys, ratio_maps, aa_factors, NoiseAccumulator

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, indent=2)+'\n')
    pending.replace(path)


class FullBank:
    def __init__(self, root=HERE, pp_source=None):
        self.root = Path(root)
        self.config = json.loads((self.root/'config.json').read_text())
        self.plan = json.loads((self.root/'PLAN.json').read_text())
        approved = json.loads((self.root/'APPROVAL.json').read_text())
        if not approved['approved'] or sha(self.root/'payload.tar.gz') != approved['approved_payload_sha256']:
            raise ValueError('Unapproved or changed source payload')
        with tarfile.open(self.root/'payload.tar.gz') as archive:
            text = archive.extractfile('runtime/SOURCE_SHA256SUMS').read().decode()
        pins = dict(line.split('  ', 1)[::-1] for line in text.splitlines())
        prefix = 'wake_nuisance_20261017/full_bank/'
        for name in ('PLAN.json', 'config.json'):
            if sha(self.root/name) != pins[prefix+name]:
                raise ValueError('Local inputs differ from approved payload: '+name)
        self.features = feature_keys(self.config)
        self.ids = np.asarray(self.config['design_ids'], int)
        self.nodes = np.asarray(self.config['a_nodes'])
        self.shape = (len(self.ids), len(self.nodes), len(self.features))
        if (self.shape != (40, 5, 170) or len(set(self.features)) != 170
                or not np.array_equal(self.ids, np.arange(40))
                or self.config['design_ids'] != list(range(40))
                or not np.array_equal(self.nodes, np.linspace(0, 1, 5))):
            raise ValueError('Changed approved full-bank scope')
        coverage = [(t['tag'], tuple(t['chunks'])) for t in self.plan['tasks']]
        expected = {(tag, (chunk,)) for tag in ('aa05', 'aa510')
                    for chunk in self.plan['aa_chunks']}
        if (len(coverage) != 40 or len(expected) != 40 or set(coverage) != expected
                or [t['id'] for t in self.plan['tasks']] != list(range(40))):
            raise ValueError('Incomplete or duplicate AA task coverage')
        self.inputs = []
        for task in self.plan['tasks']:
            directory = self.root/'download'/f'task_{task["id"]}'
            path = directory/'CHECKPOINT.json'
            info = json.loads(path.read_text())
            if (info['status'] != 'COMPLETE' or info['task'] != task
                    or info['plan_sha256'] != sha(self.root/'PLAN.json')
                    or info['config_sha256'] != sha(self.root/'config.json')
                    or len(info['records']) != 1):
                raise ValueError('Changed/incomplete downloaded task')
            if info['collector_sha256'] != pins[prefix+'collect_full.py']:
                raise ValueError('Collector source differs from approved payload')
            if info['conservation_sha256'] != pins[prefix+'conservation.py']:
                raise ValueError('Conservation source differs from approved payload')
            if info['transport_sha256'] != pins['wake_paper_morph_20261017/transport.py']:
                raise ValueError('Transport source differs from approved payload')
            if 'projection_policy' in self.plan:
                for field, source in (('projection_source_sha256', 'multistart_projection.py'),
                                      ('projection_adapter_sha256', 'audited_projection.py')):
                    if info.get(field) != pins[prefix+source]:
                        raise ValueError('Projection source differs from approved payload')
            record = info['records'][0]
            name = f'chunk_{task["chunks"][0]:04d}.npz'
            if (record['filename'] != name or record['chunk'] != task['chunks'][0]
                    or record['events'] != (5630 if task['tag'] == 'aa05' else 4370)
                    or [x['design_id'] for x in record['samples']] != self.ids.tolist()):
                raise ValueError('Unexpected count path/design coverage')
            raw = directory/name
            if sha(raw) != record['sha256']:
                raise ValueError('Changed compressed count checksum')
            self.inputs.append((task, raw, info))
        pp_source = self.root/'pp_reuse' if pp_source is None else Path(pp_source)
        if sha(pp_source/'PROOF.json') != self.plan['pp_reuse_proof_sha256']:
            raise ValueError('Shared pp proof differs from approved plan')
        proof = json.loads((pp_source/'PROOF.json').read_text())
        self.pp_path = pp_source/'pp.npz'
        if proof['complete_pp_chunks'] != 161 or proof['output_sha256'] != sha(self.pp_path):
            raise ValueError('Changed full-pp reuse proof')
        with np.load(self.pp_path) as pp:
            self.pp = {k: pp[k].copy() for k in pp.files}

    def input_identity(self):
        identity = {str(p.relative_to(self.root)): sha(p)
                    for p in (self.root/'PLAN.json', self.root/'config.json',
                              self.root/'APPROVAL.json')}
        identity['shared_pp:'+str(self.pp_path.resolve())] = sha(self.pp_path)
        for _, path, info in self.inputs:
            if sha(path) != info['records'][0]['sha256']:
                raise ValueError('Changed compressed counts after validation')
            identity[str(path.relative_to(self.root))] = info['records'][0]['sha256']
            checkpoint = path.parent/'CHECKPOINT.json'
            if json.loads(checkpoint.read_text()) != info:
                raise ValueError('Changed checkpoint after validation')
            identity[str(checkpoint.relative_to(self.root))] = sha(checkpoint)
        for path in (Path(__file__), HERE/'ratio_factors.py'):
            identity['adapter:'+path.name] = sha(path)
        return identity

    def raw_blocks(self, batch=256):
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
                if np.any(values < 0):
                    raise ValueError('Negative observable count')
                yield centrality, values*weights[start:stop, None, None, None]

    def summarize(self):
        out = self.root/'export'
        target = out/'means.npz'
        identity = self.input_identity()
        marker = out/'COMPLETE.json'
        if marker.exists():
            proof = json.loads(marker.read_text())
            if proof['inputs'] != identity or proof['output_sha256'] != sha(target):
                raise ValueError('Changed completed local export; do not overwrite')
            with np.load(target) as z:
                result = {k: z[k].copy() for k in z.files}
            self.maps = ratio_maps(result['raw_totals'], self.config,
                                   result['cross_sections'], self.pp['mean'])
            return result
        if target.exists():
            raise ValueError('Uncommitted local export; inspect, do not overwrite')
        totals = np.zeros((2, 40, 5, 172))
        squares = np.zeros_like(totals)
        for i, values in self.raw_blocks():
            totals[i] += values.sum(0)
            squares[i] += (values*values).sum(0)
        sigma = np.full((2, 40, 1, 1), np.nan)
        for task, _, info in self.inputs:
            i = 0 if task['tag'] == 'aa05' else 1
            values = np.asarray([s['fixed_full_cross_section'] for s in info['records'][0]['samples']])
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('Invalid fixed full-production cross sections')
            if np.isfinite(sigma[i]).all():
                np.testing.assert_array_equal(sigma[i, :, 0, 0], values)
            sigma[i, :, 0, 0] = values
        self.maps = ratio_maps(totals, self.config, sigma, self.pp['mean'])
        if np.any(self.maps['mean'] <= 0):
            raise ValueError('Empty AA prediction bins; do not fit an emulator')
        variance = self.maps['pp_derivative']**2*np.diag(self.pp['covariance'])
        for factors in self.factor_blocks():
            variance += (factors*factors).sum(0).reshape(self.shape)
        result = dict(raw_totals=totals, raw_sumw2=squares, cross_sections=sigma,
            central_values=self.maps['mean'], measured_variance=variance,
            pp_mean=self.pp['mean'], pp_covariance=self.pp['covariance'],
            design_ids=self.ids, a_nodes=self.nodes, features=np.asarray(self.features, str))
        out.mkdir(exist_ok=True)
        pending = target.with_name('means.pending.npz')
        np.savez_compressed(pending, **result)
        pending.replace(target)
        save_json(marker, dict(status='PAIRED_COUNTS_EXPORTED_NOT_EMULATOR_CERTIFIED',
            inputs=identity, output_sha256=sha(target), events_per_design=200000,
            old_reference_sampling_identical=False,
            note='Exact first-order ratio event factors remain available; no covariance diagonalization'))
        return result

    def factor_blocks(self, permutation=None, batch=128):
        if not hasattr(self, 'maps'):
            self.summarize()
        for i, values in self.raw_blocks(batch):
            factors = aa_factors(values, i, self.maps).reshape(len(values), 200, 170)
            yield factors if permutation is None else factors[..., permutation]

    def project_noise(self, retained, omitted, training, targets=(), permutation=None):
        training, targets = np.asarray(training), np.asarray(targets)
        for rows in (training, targets):
            if (rows.ndim != 1 or (rows.size and not np.issubdtype(rows.dtype, np.integer))
                    or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= 200)):
                raise ValueError('Invalid flattened sample IDs')
        if not len(training) or np.intersect1d(training, targets).size:
            raise ValueError('Empty training set or overlapping noisy targets')
        if permutation is not None:
            permutation = np.asarray(permutation)
            if not np.array_equal(np.sort(permutation), np.arange(170)):
                raise ValueError('Feature mapping must be a complete permutation')
        if (retained.ndim != 2 or omitted.ndim != 2
                or retained.shape[1] != 170 or omitted.shape[1] != 170):
            raise ValueError('Misaligned noise projection maps')
        if not hasattr(self, 'maps'):
            self.summarize()
        acc = NoiseAccumulator(retained, omitted, training, targets, 170)
        for factors in self.factor_blocks(permutation):
            acc.add(factors)
        derivative = self.maps['pp_derivative'].reshape(200, 170)
        covariance = self.pp['covariance']
        if permutation is not None:
            derivative = derivative[:, permutation]
            covariance = covariance[np.ix_(permutation, permutation)]
        acc.add_pp(derivative, covariance)
        return acc.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    args = parser.parse_args()
    bank = FullBank(args.root, pp_source=HERE/'pp_reuse')
    result = bank.summarize()
    print('FULL_WAKE_LOCAL_EXPORT_COMPLETE', result['central_values'].shape, flush=True)
