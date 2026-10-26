"""Paired one-design wake sensitivity. No new events, emulator or inference fit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path[:0] = [str(HERE), str(WORK / 'JETSCAPE-analysis-atlas-dr12')]
from jetscape_analysis.analysis import atlas_dr12 as dr
from jetscape_analysis.analysis import atlas_rg as rg
import _wake_native as native
from deform import MODES, coordinates, deform, delta_r, four_vectors

REFERENCE = WORK / 'atlas_rg_inference_20261015/recovery_20261016/precision_continuation/timeout_completion'
TABLES = REFERENCE / 'local_inference/tables/Data'
SETTINGS = dict(seed=20261017, sweeps=2, fraction=.5, width=.4, hard_scale=1., association_radius=3.)
PT_EDGES = np.r_[0, np.geomspace(.05, 15, 45)]
DR_EDGES = np.linspace(0, 3, 46)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def specs():
    rows = []
    for path in sorted(TABLES.glob('Data__*.dat')):
        name = path.stem.removeprefix('Data__')
        energy, system, kind, obs, setting, centrality = name.split('__')
        include = (centrality == '0-10' and ((obs in ('pt_alice', 'pt_atlas', 'pt_cms')
            and setting in ('R0.2', 'R0.3', 'R0.4')) or obs.startswith('rg_atlas')
            or obs == 'dr12_atlas' or (obs == 'tg_alice' and setting == 'R0.2_zcut0.2_beta0')))
        if not include:
            continue
        table = np.loadtxt(path, ndmin=2)
        rows.append(dict(name=name, observable=obs, setting=setting,
            edges=np.r_[table[:, 0], table[-1, 1]].tolist(), data=table[:, 2].tolist()))
    assert len(rows) == 11
    return rows


def reconstruction(event, system, blocks):
    rows = dr.particle_rows(event, system)
    full = {radius: native.inclusive(rows, radius) for radius in (.2, .3, .4)}
    theta = np.asarray(native.theta(rows)).reshape(-1, 3)
    large = np.asarray(dr._atlas_dr12.reconstruct(rows, groomer='lund')['jets']).reshape(-1, 4)
    # Exactly the current upstream signed-prong reconstruction; no relaxed guard.
    groomed = rg._upstream_backend().reconstruct(rows, negative_marker=-1, ca_radius=1000,
        zero_energy_scale=.0001, grooming_policy='signed_prongs', require_all_constituents=True)
    groomed = np.asarray(groomed['jets']).reshape(-1, 4)
    result = []
    for block in blocks:
        obs, setting = block['observable'], block['setting']
        if obs.startswith('pt_'):
            radius = float(setting[1:])
            jets = np.asarray(full[radius]['jets']).reshape(-1, 7)
            if obs == 'pt_alice':
                mask = (np.abs(jets[:, 1]) < .7-radius) & (jets[:, 0] > 40) & (jets[:, 0] < 140)
                mask &= jets[:, 4] > (5 if radius == .2 else 7)
            elif obs == 'pt_atlas':
                mask = (np.abs(jets[:, 2]) < 2.8) & (jets[:, 0] > 50) & (jets[:, 0] < 1000)
            else:
                mask = (np.abs(jets[:, 1]) < 2) & (jets[:, 0] > 200) & (jets[:, 0] < 1000)
            values = jets[mask, 0]
        elif obs == 'tg_alice':
            selected = theta[(theta[:, 0] > 60) & (theta[:, 0] < 80)]
            values = selected[selected[:, 2] > 0, 1]
        elif obs == 'dr12_atlas':
            values = large[(large[:, 0] >= 200) & (large[:, 0] < 251), 1]
        else:
            index = int(setting.split('_pt')[1])
            low, high = rg.PT_BINS[index]
            mask = (groomed[:, 0] >= low) & (groomed[:, 0] < high) & (groomed[:, 3] > 0)
            values = groomed[mask, 1]
            values = values[(values > 0) & (values < .4)]
        # np.histogram includes the last edge; these tables have ROOT [low,high) bins.
        values = np.asarray(values)
        values = values[(values >= block['edges'][0]) & (values < block['edges'][-1])]
        result.extend(np.histogram(values, block['edges'])[0])
    # Inclusive-jet denominator, including failed grooming and theta overflow.
    result.append(np.count_nonzero((theta[:, 0] > 60) & (theta[:, 0] < 80)))
    result.append(1.)
    return np.asarray(result, float), full


def wake_metrics(event, axes, wake_indices):
    v = four_vectors(event)
    wake = v[wake_indices]
    if not len(wake):
        return np.zeros(len(PT_EDGES)-1), np.zeros(len(DR_EDGES)-1), np.zeros(9)
    pt = coordinates(wake)[0]
    distance = delta_r(wake[:, None], axes[None]).min(1) if len(axes) else np.full(len(wake), np.inf)
    energy = wake[:, 3]
    return (np.histogram(pt, PT_EDGES, weights=energy)[0],
            np.histogram(distance, DR_EDGES, weights=energy)[0],
            np.array([energy.sum(), energy[pt > 2].sum(), energy[pt > 4].sum(),
                      energy[distance < .2].sum(), energy[distance < .4].sum(),
                      len(wake), pt.sum(), np.count_nonzero(pt > 10), pt.max(initial=0)]))


def sample(path, system, blocks, limit, output):
    width = sum(len(b['edges'])-1 for b in blocks) + 2
    n_modes = len(MODES) if system == 'PbPb' else 1
    source = sha(path)
    identity = dict(input_sha256=source, max_events=limit, settings=SETTINGS,
        sources={str(p): sha(p) for p in [Path(__file__), HERE/'deform.py', HERE/'native.cc',
            HERE/'build.json', Path(dr.__file__), Path(dr._atlas_dr12.__file__),
            Path(rg._upstream_backend().__file__)]})
    if output.with_suffix('.json').exists():
        meta = json.loads(output.with_suffix('.json').read_text())
        if meta['identity'] != identity or meta['npz_sha256'] != sha(output):
            raise ValueError('Changed cached sample; use a new output directory')
        return meta
    total, outer = np.zeros(n_modes*width), np.zeros((n_modes*width, n_modes*width))
    pt_hist = np.zeros((n_modes, len(PT_EDGES)-1))
    dr_hist = np.zeros((n_modes, len(DR_EDGES)-1))
    metrics = np.zeros((n_modes, 9))
    qa = {mode: Counter() for mode in MODES[:n_modes]}
    maxqa = {mode: dict(p4=0., mass2=0.) for mode in MODES[:n_modes]}
    sigma = None
    sigma_count = 0
    started = time.monotonic()
    n = 0
    for event in dr.read_events(path):
        if limit and n >= limit:
            break
        n += 1
        weight = float(event['event_weight'])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Invalid event weight')
        current = float(event['cross_section'])
        if sigma is not None and not np.isclose(current, sigma, rtol=1e-6):
            raise ValueError('Changing cross section')
        sigma, sigma_count = current, sigma_count+1
        # Fixed, wake-independent shower axes, identical for every variation.
        shower = dict(event, **{key: np.asarray(event[key])[np.asarray(event['status']) == 0]
                                for key in ('px', 'py', 'pz', 'E', 'particle_ID', 'status')})
        axes = np.asarray(native.axes(dr.particle_rows(shower, system))).reshape(-1, 2)
        wake_indices = np.flatnonzero(np.asarray(event['status']) == 1)
        if len(wake_indices) and len(axes):
            wake_indices = wake_indices[delta_r(four_vectors(event)[wake_indices, None], axes[None]).min(1)
                                        < SETTINGS['association_radius']]
        else:
            wake_indices = np.array([], dtype=int)
        vectors = []
        for i, mode in enumerate(MODES[:n_modes]):
            modified, checks = deform(event, axes, mode, **SETTINGS)
            counts, full = reconstruction(modified, system, blocks)
            vectors.append(counts)
            ph, dh, met = wake_metrics(modified, axes, wake_indices)
            pt_hist[i] += weight*ph
            dr_hist[i] += weight*dh
            metrics[i, :8] += weight*met[:8]
            metrics[i, 8] = max(metrics[i, 8], met[8])
            for key in ('accepted_pairs', 'skipped_non_timelike_pairs', 'assigned_wake_particles'):
                qa[mode][key] += checks[key]
            p = four_vectors(modified)
            positive = np.asarray(modified['status']) == 1
            charged = np.isin(np.abs(np.asarray(modified['particle_ID'])), list(dr.CHARGED_PIDS))
            pt = np.hypot(p[:, 0], p[:, 1])
            qa[mode]['charged_positive_wake_pT_above10_weighted'] += weight*np.count_nonzero(positive & charged & (pt > 10))
            qa[mode]['charged_positive_wake_pT_above30_weighted'] += weight*np.count_nonzero(positive & charged & (pt > 30))
            maxqa[mode]['p4'] = max(maxqa[mode]['p4'], checks['max_four_momentum_residual_GeV'])
            maxqa[mode]['mass2'] = max(maxqa[mode]['mass2'], checks['max_mass_squared_residual_GeV2'])
        vector = weight*np.concatenate(vectors)
        total += vector
        outer += np.outer(vector, vector)
        if n % 1000 == 0:
            print('WAKE_SAMPLE', path.name, system, n, round(time.monotonic()-started, 1), flush=True)
    if not n:
        raise ValueError('Empty sample')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, total=total, outer=outer, pt_hist=pt_hist, dr_hist=dr_hist, metrics=metrics)
    meta = dict(identity=identity, path=str(path), system=system, events=n,
        cross_section=sigma, cross_section_entries=sigma_count,
        sum_weights=total[width-1], width=width, modes=list(MODES[:n_modes]),
        seconds=time.monotonic()-started, qa={key: dict(v) for key, v in qa.items()},
        maximum_conservation_residuals=maxqa, npz_sha256=sha(output))
    save(output.with_suffix('.json'), meta)
    return meta


def spectrum(total, blocks):
    """Cross sections/shape and exact first-order derivatives in joint mode space."""
    width = sum(len(b['edges'])-1 for b in blocks) + 2
    n_modes = len(total)//width
    result = np.zeros((n_modes, width-2))
    jac = np.zeros((n_modes*(width-2), len(total)))
    for mode in range(n_modes):
        values = total[mode*width:(mode+1)*width]
        start = 0
        for block in blocks:
            end = start+len(block['edges'])-1
            denominator = width-2 if block['observable'] == 'tg_alice' else width-1
            if values[denominator] <= 0:
                raise ValueError('No selected theta jets or no event weights')
            result[mode, start:end] = values[start:end]/values[denominator]
            rows = mode*(width-2)+np.arange(start, end)
            jac[rows, mode*width+np.arange(start, end)] = 1/values[denominator]
            jac[rows, mode*width+denominator] = -values[start:end]/values[denominator]**2
            start = end
    return result, jac


def summarize(output, blocks, metas, *, design_id=21, physical_pi_T_Lres=3.27045, kappa_sc=.4142):
    loaded = []
    for tag in ('aa05', 'aa510', 'pp'):
        with np.load(output/(tag+'.npz')) as archive:
            loaded.append({key: archive[key].copy() for key in archive.files})
    a, b, pp = loaded
    # RAA/theta legacy uses pooled centralities; dr12 and rg use separately normalized equal components.
    mean, gradient = spectrum(a['total']+b['total'], blocks)
    components = []
    for sample_data, meta in zip((a, b), metas[:2]):
        values, jac = spectrum(sample_data['total'], blocks)
        components.append((values*meta['cross_section'], jac*meta['cross_section'], sample_data['outer']))
    f = mean.shape[1]
    start = 0
    sigma = np.mean([m['cross_section'] for m in metas[:2]])
    factors = np.ones(f)
    separate = []
    for block in blocks:
        stop = start+len(block['edges'])-1
        if block['observable'] != 'tg_alice':
            factors[start:stop] = sigma
        if block['observable'] == 'dr12_atlas' or block['observable'].startswith('rg_atlas'):
            separate.extend(range(start, stop))
        start = stop
    mean *= factors
    # Rebuild combined Jacobians so old/new mixed-feature cross-covariances stay present.
    ja = gradient*np.tile(factors, len(MODES))[:, None]
    jb = ja.copy()
    for mode in range(len(MODES)):
        idx = mode*f+np.asarray(separate)
        mean[mode, separate] = .5*(components[0][0][mode, separate]+components[1][0][mode, separate])
        ja[idx] = .5*components[0][1][idx]
        jb[idx] = .5*components[1][1][idx]
    covariance = ja@a['outer']@ja.T + jb@b['outer']@jb.T
    pp_mean, pp_jac = spectrum(pp['total'], blocks)
    pp_factor = np.ones(f)
    start = 0
    for block in blocks:
        stop = start+len(block['edges'])-1
        if block['observable'] != 'tg_alice':
            pp_factor[start:stop] = metas[2]['cross_section']
        start = stop
    pp_mean = pp_mean[0]*pp_factor
    pp_jac = pp_jac*pp_factor[:, None]
    pp_cov = pp_jac@pp['outer']@pp_jac.T
    response = np.full_like(mean, np.nan)
    response_error = np.full_like(mean, np.nan)
    ratio = np.full_like(mean, np.nan)
    ratio_error = np.full_like(mean, np.nan)
    for mode in range(len(MODES)):
        valid = mean[0] > 0
        response[mode, valid] = mean[mode, valid]/mean[0, valid]
        for j in np.flatnonzero(valid):
            v = np.zeros(len(MODES)*f)
            v[mode*f+j] += 1/mean[0, j]
            v[j] -= mean[mode, j]/mean[0, j]**2
            response_error[mode, j] = np.sqrt(max(0., v@covariance@v))
        valid = pp_mean > 0
        ratio[mode, valid] = mean[mode, valid]/pp_mean[valid]
        variance = np.diag(covariance)[mode*f:(mode+1)*f]
        ratio_error[mode, valid] = np.sqrt(np.maximum(0., variance[valid]/pp_mean[valid]**2
            + mean[mode, valid]**2*np.diag(pp_cov)[valid]/pp_mean[valid]**4))
    # First-order event-Poisson covariance, including all-event/shape denominators.
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-16)
    scale = np.max(np.diag(covariance))
    minimum = float(np.linalg.eigvalsh(covariance).min())
    if minimum < -1e-10*max(scale, 1e-30):
        raise ValueError('Non-PSD propagated simulation covariance')
    np.savez_compressed(output/'comparison.npz', aa_mean=mean, aa_covariance=covariance,
        pp_mean=pp_mean, pp_covariance=pp_cov, ratio=ratio, ratio_error=ratio_error,
        response=response, response_error=response_error,
        pt_hist=a['pt_hist']+b['pt_hist'], dr_hist=a['dr_hist']+b['dr_hist'],
        metrics=np.column_stack(((a['metrics']+b['metrics'])[:, :8],
            np.maximum(a['metrics'][:, 8], b['metrics'][:, 8]))),
        pt_edges=PT_EDGES, dr_edges=DR_EDGES)
    report = dict(status='PAIRED_SENSITIVITY_NOT_THEORY_ERROR_CALIBRATION', design_id=design_id,
        physical_pi_T_Lres=physical_pi_T_Lres, kappa_sc=kappa_sc, settings=SETTINGS, modes=list(MODES), blocks=blocks,
        samples=metas, covariance_min_eigenvalue=minimum, independent_pp_AA=True,
        wake_profile_selection='Fixed original positive-wake association within DeltaR<3 of shower-only R=.4 jets above20GeV',
        covariance='Paired event-Poisson; first-order ratio/normalization propagation; fixed cross sections',
        caveats=['Not Efficient Wake or full-shower implementation of arXiv:2609.10679',
            'Only positive wake deformed; multiplicities, species, shower and negative wake fixed',
            'Fixed shower-axis association and stress scales are assumptions, not calibrated priors',
            'One chunk per centrality and one pp chunk; sparse bins cannot establish small effects',
            'Legacy particle content and current conditional rg bin edges retained',
            'Correct signed recombination/positive leading-track selector in the paired reconstruction; not a legacy-bug emulation',
            'No posterior/inference change, no uncertainty inflation or emulator refit'])
    save(output/'RESULTS.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-events', type=int, default=0)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    blocks = specs()
    sources = [('aa05', HERE/'input/aa05_design21.parquet', 'PbPb'),
               ('aa510', HERE/'input/aa510_design21.parquet', 'PbPb'),
               ('pp', WORK/'cluster-access/pi0-pilot-inputs/HYBRID_pp_0000_final_state_hadrons_00.parquet', 'pp')]
    metas = [sample(path, system, blocks, args.max_events, args.output/(tag+'.npz'))
             for tag, path, system in sources]
    summarize(args.output, blocks, metas)
    print('WAKE_STUDY_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
