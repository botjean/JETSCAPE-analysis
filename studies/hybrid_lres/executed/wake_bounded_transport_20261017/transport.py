"""Bounded paper-inspired wake sensitivity; not a hydrodynamic wake model.

Keep one global a. Propose the existing calibrated pT/angle/population changes,
then conserve each assigned sign budget with pT no greater than its proposal.
No particle is created, no hard recoil is added, and no failed event is dropped.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK/'wake_nuisance_20261017'))
from global_a import scalar_a, description, load_frozen
from bounded_projection import choose_budgets, project
from thinning import conditioned_indices

REFERENCE = load_frozen('bounded_transport_reference', 'transport.py')
PARTICLE_KEYS = REFERENCE.PARTICLE_KEYS
CHARGED_PIDS = (11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334)


def longitudinal_boost(v, rapidity):
    v = np.asarray(v, dtype=np.longdouble)
    plus = (v[..., 3]+v[..., 2])*np.exp(np.longdouble(rapidity))
    minus = (v[..., 3]-v[..., 2])*np.exp(-np.longdouble(rapidity))
    result = v.copy()
    result[..., 2] = (plus-minus)/2
    result[..., 3] = (plus+minus)/2
    return result


def deform(event, a, *, seed=20261018, observer=None):
    a = scalar_a(a)
    qa = dict(baseline_identity=int(a == 0), no_axis_events=0, degenerate_events=0,
        frozen_spacelike=0, frozen_zero_mt=0, population_restored=0, restored_rows=0,
        original_positive=0, original_negative=0, selected_positive=0, selected_negative=0,
        deleted_positive=0, deleted_negative=0, requested_negative_fraction=1.,
        actual_negative_fraction=1., budget_relaxed=0, signed_p4_residual=0.,
        mass2_residual=0., sign_budget_residual=0., unsigned_energy_ratio=1.,
        correction_rms_radians=0., maximum_sign_correction=0.,
        maximum_charged_wake_pt=0., pt_envelope_ratio=0., pt_damping_positive=1.,
        pt_damping_negative=1., sign_corrections={}, whole_event_fallback=0)
    if a == 0:
        return event, qa
    info = description(a)
    par, amplitudes = info['parameters'], info['amplitudes']
    hp, hn = par['positive_pt_scale'], par['negative_pt_scale']
    cp, cy = par['positive_phi_scale'], par['positive_y_scale']
    if hp < 1 or hn < 1:
        raise ValueError('The restoration certificate requires h_s >= 1')
    requested = amplitudes[1]*hn
    if not 0 < requested <= 1:
        raise ValueError('Invalid requested negative energy fraction')
    qa['requested_negative_fraction'] = requested
    original = REFERENCE.four_vectors(event)
    status, pid = np.asarray(event['status']), np.asarray(event['particle_ID'])
    wake = np.flatnonzero(np.isin(status, (1, 2)))
    hard = np.flatnonzero(status == -2)
    if not len(wake) or not len(hard):
        qa['no_axis_events'] = 1
        return event, qa
    masses2 = REFERENCE.mass2(original)
    qa['frozen_spacelike'] = int(np.count_nonzero(masses2[wake] < 0))
    wake = wake[masses2[wake] >= 0]
    old_pt = np.hypot(original[:, 0], original[:, 1])
    zero = (old_pt[wake] == 0) & (masses2[wake] == 0)
    qa['frozen_zero_mt'] = int(zero.sum())
    wake = wake[~zero]
    if len(wake) < 8:
        qa['degenerate_events'] = 1
        return event, qa
    qa['original_positive'] = int(np.count_nonzero(status[wake] == 1))
    qa['original_negative'] = int(np.count_nonzero(status[wake] == 2))
    identity = np.asarray([event.get(k, 0.) for k in
        ('event_ID', 'pt_hat', 'vertex_x', 'vertex_y')], dtype='<f8')
    rng = np.random.default_rng(np.random.SeedSequence([seed,
        int.from_bytes(hashlib.sha256(identity.tobytes()).digest()[:8], 'little')]))
    selected = []
    for s, amplitude in zip((1, 2), amplitudes):
        rows = wake[status[wake] == s]
        selected.extend(rows[conditioned_indices(pid[rows], amplitude, rng)])
    selected = np.sort(selected).astype(int)
    unsigned = original[wake].astype(np.longdouble).sum(0)
    frame_y = .5*np.log((unsigned[3]+unsigned[2])/(unsigned[3]-unsigned[2]))
    working = longitudinal_boost(original[wake], -frame_y)
    old_budgets = [working[status[wake] == s].sum(0) for s in (1, 2)]

    def budget_for(indices):
        signs = np.where(status[indices] == 1, 1, -1)
        caps = old_pt[indices]*np.where(signs > 0, hp, hn)
        return choose_budgets(*old_budgets, caps, masses2[indices], signs, requested), caps, signs

    match, caps, signs = budget_for(selected)
    if match is None:
        # Retain the protected species and restore removed neutral/pair rows;
        # do not demand a harder recoil or pretend a failed event was simulated.
        qa['population_restored'] = 1
        qa['restored_rows'] = len(wake)-len(selected)
        selected = wake.copy()
        match, caps, signs = budget_for(selected)
    if match is None:
        raise RuntimeError('Full original population failed its bounded f=1 feasibility certificate')
    fraction, targets = match
    qa.update(actual_negative_fraction=float(fraction), budget_relaxed=int(fraction > requested+1e-10),
        selected_positive=int((signs > 0).sum()), selected_negative=int((signs < 0).sum()))
    qa['deleted_positive'] = qa['original_positive']-qa['selected_positive']
    qa['deleted_negative'] = qa['original_negative']-qa['selected_negative']
    _, y, phi = REFERENCE.coordinates(original[selected])
    _, hard_y, hard_phi = REFERENCE.coordinates(original[hard])
    delta = (phi[:, None]-hard_phi[None]+np.pi) % (2*np.pi)-np.pi
    labels = np.argmin((y[:, None]-hard_y[None])**2+delta**2, axis=1)
    positive = signs > 0
    proposed_y = np.where(positive, hard_y[labels]+cy*(y-hard_y[labels]), y)-float(frame_y)
    proposed_phi = np.where(positive, hard_phi[labels]+REFERENCE.angle(delta[np.arange(len(labels)), labels], cp), phi)
    result_work = np.empty((len(selected), 4))
    correction2 = 0.
    for s, target in zip((1, -1), targets):
        rows = signs == s
        if not rows.any():
            np.testing.assert_array_equal(target, np.zeros(4))
            continue
        four, checks = project(target, caps[rows], masses2[selected][rows], proposed_y[rows], proposed_phi[rows])
        result_work[rows] = four
        qa['sign_corrections'][str(s)] = checks
        qa['pt_damping_positive' if s == 1 else 'pt_damping_negative'] = checks['pt_damping']
        correction2 += rows.sum()*checks['correction_rms_radians']**2
        qa['maximum_sign_correction'] = max(qa['maximum_sign_correction'], checks['correction_rms_radians'])
        qa['sign_budget_residual'] = max(qa['sign_budget_residual'], float(abs(four.sum(0)-target).max()))
    qa['correction_rms_radians'] = float(np.sqrt(correction2/len(selected)))
    projected = longitudinal_boost(result_work, frame_y).astype(float)
    targets_lab = longitudinal_boost(np.array(targets), frame_y)
    for s, target, old in zip((1, -1), targets_lab, old_budgets):
        value = projected[signs == s].sum(0)
        np.testing.assert_allclose(value, target, atol=2e-7, rtol=2e-10)
        # f<=1 reduces each original unsigned sign energy in every frame.
        original_sign = original[wake[status[wake] == (1 if s == 1 else 2)]].sum(0)
        if value[3] > original_sign[3]+2e-7+2e-10*abs(original_sign[3]):
            raise RuntimeError('An unsigned sign energy increased')
    expected = original[wake].astype(np.longdouble)*np.where(status[wake] == 1, 1, -1)[:, None]
    residual = (projected.astype(np.longdouble)*signs[:, None]).sum(0)-expected.sum(0)
    np.testing.assert_allclose((projected*signs[:, None]).sum(0), expected.sum(0), atol=2e-7, rtol=2e-10)
    qa['signed_p4_residual'] = float(abs(residual).max())
    np.testing.assert_allclose(REFERENCE.mass2(projected), masses2[selected], atol=2e-7, rtol=1e-7)
    qa['mass2_residual'] = float(abs(REFERENCE.mass2(projected)-masses2[selected]).max(initial=0))
    actual_pt = np.hypot(projected[:, 0], projected[:, 1])
    if np.any(actual_pt > caps*(1+1e-12)+1e-12):
        raise RuntimeError('A particle exceeds its paper-scaled original pT')
    qa['pt_envelope_ratio'] = float(np.max(np.divide(actual_pt, caps, out=np.zeros_like(caps), where=caps > 0), initial=0))
    _, final_y, _ = REFERENCE.coordinates(projected)
    if not np.isfinite(projected).all() or abs(final_y).max() > 30:
        raise RuntimeError('Final float64 representation/rapidity guard failed')
    alive = np.ones(len(original), bool)
    alive[np.setdiff1d(wake, selected)] = False
    indices = np.flatnonzero(alive)
    after = original[indices].copy()
    after[np.searchsorted(indices, selected)] = projected
    unchanged = ~np.isin(indices, selected)
    np.testing.assert_array_equal(after[unchanged], original[indices[unchanged]])
    for s in (1, 2):
        for species in np.unique(abs(pid)):
            if species in (22, 111, 221, 331):
                continue
            before = np.count_nonzero((status == s) & (pid == species))-np.count_nonzero((status == s) & (pid == -species))
            final = np.count_nonzero((status[indices] == s) & (pid[indices] == species))-np.count_nonzero((status[indices] == s) & (pid[indices] == -species))
            if before != final:
                raise RuntimeError('The bounded map changed a species charge')
    qa['unsigned_energy_ratio'] = float(projected[:, 3].sum()/original[wake, 3].sum())
    charged = np.isin(abs(pid[selected]), CHARGED_PIDS)
    qa['maximum_charged_wake_pt'] = float(actual_pt[charged].max(initial=0))
    result = dict(event, **{key: np.asarray(event[key])[indices] for key in PARTICLE_KEYS})
    result.update({key: after[:, i] for i, key in enumerate(('px', 'py', 'pz', 'E'))})
    if observer is not None:
        mt = np.sqrt(masses2[selected]+caps**2)
        proposal = np.column_stack((caps*np.cos(proposed_phi), caps*np.sin(proposed_phi),
                                   mt*np.sinh(proposed_y), mt*np.cosh(proposed_y)))
        frozen = np.setdiff1d(np.flatnonzero(np.isin(status, (1, 2))), wake)
        proposal_lab = longitudinal_boost(proposal, frame_y).astype(float)
        observer(dict(proposal_four=np.concatenate((proposal_lab, original[frozen])),
            proposal_status=np.r_[status[selected], status[frozen]],
            selected_original_indices=selected.copy(), pt_caps=caps.copy(),
            targets_lab=np.array(targets_lab, float)))
    return result, qa
