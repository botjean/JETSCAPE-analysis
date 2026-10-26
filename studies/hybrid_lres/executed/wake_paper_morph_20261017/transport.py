"""Paper-fitted signed wake transport with an explicit event-level projection.

No parent/deposition ancestry is available. Nearest outgoing hard-parton axes
are an explicit transfer assumption, NOT recovered wake source identities.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'wake_sensitivity_20261017'))
from deform import boost, coordinates, four_vectors, mass2
from calibrate import Model, angle

PARTICLE_KEYS = ('px', 'py', 'pz', 'E', 'particle_ID', 'status')
MODES = ('baseline', 'paper10', 'paper50_half', 'paper50', 'paper50_shuffled')


def parameters(mode):
    records = json.loads((HERE/'CALIBRATION.json').read_text())
    energy = 10 if mode == 'paper10' else 50
    record = next(item for item in records if item['energy'] == energy)
    fraction = .5 if mode == 'paper50_half' else 1.
    par = np.exp(fraction*np.log(list(record['parameters'].values())))
    amplitudes = Model(record['temperature'], record['deposition']).amplitudes(par)
    return par, amplitudes


PARAMETERS = None


def balanced_indices(pid, amplitude, rng):
    """Thin neutral self-conjugates or particle/antiparticle pairs, never charge.

    The protected species-number differences cause a recorded finite-event
    deviation from the continuum amplitudes, rather than changing charges.
    """
    if not 0 < amplitude <= 1:
        raise ValueError('This pilot only supports thinning, not particle creation')
    keep = np.ones(len(pid), bool)
    for absolute in np.unique(np.abs(pid)):
        positive = np.flatnonzero(pid == absolute)
        negative = np.flatnonzero(pid == -absolute)
        if absolute in (22, 111, 221, 331):
            for ids in (positive, negative):
                keep[ids] = rng.random(len(ids)) < amplitude
        else:
            # Treat signed species differences as conserved charges, including baryons.
            positive, negative = rng.permutation(positive), rng.permutation(negative)
            count = min(len(positive), len(negative))
            remove = rng.random(count) >= amplitude
            keep[positive[:count][remove]] = False
            keep[negative[:count][remove]] = False
    return np.flatnonzero(keep)


def project_signed(proposal, masses2, signs, target, *, tolerance=2e-12, iterations=160):
    """Restore signed four-momentum at FIXED proposed pT and particle masses.

    Only phi and rapidity change. Use a longitudinal reference boost, which
    cannot change transverse momenta. The unrestricted spatial projection was
    REJECTED: its mass-shell conservation alone allowed unphysical hard tails.
    """
    masses2, signs, target = map(np.asarray, (masses2, signs, target))
    if not len(proposal) or np.any(masses2 < 0):
        raise ValueError('Empty or spacelike projection input')
    pt, rapidity, phi = coordinates(proposal)
    mt = np.sqrt(pt**2+masses2)
    original_coordinates = np.column_stack((phi, rapidity))
    current = original_coordinates.copy()
    scale = max(1., np.max(np.abs(target)))
    if np.hypot(*target[:2]) >= np.sum(pt)*(1-1e-12):
        raise RuntimeError('Fixed-pT transverse momentum constraint is infeasible')

    def state(coordinates_):
        phi_, y_ = coordinates_.T
        if np.max(np.abs(y_)) > 30:
            return None, np.full(4, np.inf)
        four = np.column_stack((pt*np.cos(phi_), pt*np.sin(phi_), mt*np.sinh(y_), mt*np.cosh(y_)))
        residual = (four*signs[:, None]).sum(0)-target
        return four, residual

    _, initial_residual = state(current)
    for iteration in range(iterations):
        four, residual = state(current)
        if np.max(np.abs(residual)) <= tolerance*scale:
            change = current-original_coordinates
            change[:, 0] = (change[:, 0]+np.pi) % (2*np.pi)-np.pi
            correction = np.sqrt(np.mean(np.sum(change**2, axis=1)))
            return four, dict(iterations=iteration, residual=float(np.max(np.abs(residual))),
                              correction_rms_radians=float(correction),
                              proposal_residual=float(np.max(np.abs(initial_residual))))
        jac = np.zeros((4, len(pt), 2))
        jac[0, :, 0] = -signs*four[:, 1]
        jac[1, :, 0] = signs*four[:, 0]
        jac[2, :, 1] = signs*four[:, 3]
        jac[3, :, 1] = signs*four[:, 2]
        normal = np.einsum('aij,bij->ab', jac, jac)
        multiplier = np.linalg.lstsq(normal, residual, rcond=1e-13)[0]
        step = -np.einsum('aij,a->ij', jac, multiplier)
        fraction = min(1., .5/max(np.max(np.abs(step)), 1e-30))
        norm = np.linalg.norm(residual)
        for _ in range(22):
            candidate = current+fraction*step
            if np.linalg.norm(state(candidate)[1]) < norm:
                current = candidate
                break
            fraction *= .5
        else:
            raise RuntimeError('Signed conservation projection has no descending step')
    raise RuntimeError('Signed conservation projection did not converge')


def deform(event, mode, *, seed=20261018):
    if mode not in MODES:
        raise ValueError(mode)
    original = four_vectors(event)
    status, pid = np.asarray(event['status']), np.asarray(event['particle_ID'])
    qa = dict(deleted_positive=0, deleted_negative=0, frozen_spacelike=0,
              no_axis_events=0, degenerate_events=0, projection_iterations=0,
              correction_rms_radians=0., signed_p4_residual=0., mass2_residual=0.,
              proposal_residual=0., conditional_resampling_draws=0)
    if mode == 'baseline':
        return event, qa
    global PARAMETERS
    if PARAMETERS is None:
        PARAMETERS = {m: parameters(m) for m in MODES[1:]}
    par, amplitudes = PARAMETERS[mode]
    hp, hn, cp, cy = par
    wake = np.flatnonzero(np.isin(status, [1, 2]))
    hard = np.flatnonzero(status == -2)
    if not len(wake) or not len(hard):
        qa['no_axis_events'] = 1
        return event, qa
    invariant = mass2(original)
    valid = invariant[wake] >= 0
    qa['frozen_spacelike'] = int(np.count_nonzero(~valid))
    wake = wake[valid]
    if len(wake) < 8:
        qa['degenerate_events'] = 1
        return event, qa
    identity = np.asarray([event.get(k, 0.) for k in ('event_ID', 'pt_hat', 'vertex_x', 'vertex_y')], dtype='<f8')
    key = int.from_bytes(hashlib.sha256(identity.tobytes()).digest()[:8], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed, key]))
    old_signs = np.where(status[wake] == 1, 1., -1.)
    target_lab = (original[wake]*old_signs[:, None]).sum(0)
    target_pt = np.hypot(*target_lab[:2])
    for attempt in range(128):
        selected = []
        for s, amplitude in zip((1, 2), amplitudes):
            ids = wake[status[wake] == s]
            retained = ids[balanced_indices(pid[ids], amplitude, rng)]
            selected.extend(retained)
            qa['deleted_positive' if s == 1 else 'deleted_negative'] = int(len(ids)-len(retained))
        selected = np.sort(selected).astype(int)
        if len(selected) < 3:
            continue
        # Conditional finite-event resampling, not creation of a hard recoil.
        # Reject populations whose fixed-pT polygon cannot carry the target.
        trial_pt = np.hypot(original[selected, 0], original[selected, 1])*np.where(status[selected] == 1, hp, hn)
        if target_pt >= trial_pt.sum()*(1-1e-9) or target_pt < max(0., 2*trial_pt.max()-trial_pt.sum()):
            continue
        if np.any(status[wake] == 2) and not np.any(status[selected] == 2):
            continue
        if not np.any(status[selected] == 1):
            continue
        if not np.any(status[selected] == 2):
            minimum_longitudinal_mass = np.sqrt(trial_pt**2+invariant[selected]).sum()
            target_longitudinal_mass2 = target_lab[3]**2-target_lab[2]**2
            if target_longitudinal_mass2 <= minimum_longitudinal_mass**2:
                continue
        qa['conditional_resampling_draws'] = attempt+1
        break
    else:
        raise RuntimeError(f'{mode} event {event.get("event_ID")}: no feasible finite population in128 conditional draws')
    proposal = original[selected].copy()
    pt, y, phi = coordinates(proposal)
    _, axis_y, axis_phi = coordinates(original[hard])
    delta_phi = (phi[:, None]-axis_phi[None]+np.pi) % (2*np.pi)-np.pi
    label = np.argmin((y[:, None]-axis_y[None])**2+delta_phi**2, axis=1)
    positive = status[selected] == 1
    new_pt = pt*np.where(positive, hp, hn)
    new_y = np.where(positive, axis_y[label]+cy*(y-axis_y[label]), y)
    new_phi = np.where(positive, axis_phi[label]+angle(delta_phi[np.arange(len(label)), label], cp), phi)
    if mode == 'paper50_shuffled':
        # Explicit alternate unknown pT/angle coupling, not another paper prediction.
        for s in (1, 2):
            for species in np.unique(pid[selected]):
                for source in np.unique(label):
                    # Do not exchange pT across opposite outgoing partons or
                    # widely separated rapidities; those are different wakes.
                    bands = np.floor((y-axis_y[label])/.5).astype(int)
                    for band in np.unique(bands):
                        ids = np.flatnonzero((status[selected] == s) & (pid[selected] == species)
                                             & (label == source) & (bands == band))
                        new_pt[ids] = rng.permutation(new_pt[ids])
    mt = np.sqrt(invariant[selected]+new_pt**2)
    proposal = np.column_stack((new_pt*np.cos(new_phi), new_pt*np.sin(new_phi),
                                mt*np.sinh(new_y), mt*np.cosh(new_y)))
    # A LONGITUDINAL reference boost conditions forward events without ever
    # changing pT. A full transverse boost would defeat the hard-tail safeguard.
    unsigned = original[wake].sum(0)
    beta = np.array([0., 0., unsigned[2]/unsigned[3]])
    before_rest = boost(original[wake], -beta)
    signs_before = np.where(status[wake] == 1, 1., -1.)
    target = (before_rest*signs_before[:, None]).sum(0)
    proposal_rest = boost(proposal, -beta)
    signs = np.where(status[selected] == 1, 1., -1.)
    try:
        projected, checks = project_signed(proposal_rest, invariant[selected], signs, target)
    except RuntimeError as error:
        raise RuntimeError(f'{mode} event {event.get("event_ID")}: {error}') from error
    projected = boost(projected, beta)
    alive = np.ones(len(original), bool)
    alive[np.setdiff1d(wake, selected)] = False
    ids = np.flatnonzero(alive)
    after = original[ids].copy()
    position = np.searchsorted(ids, selected)
    after[position] = projected
    signed_before = (original[wake]*signs_before[:, None]).sum(0)
    signed_after = (projected*signs[:, None]).sum(0)
    residual = float(np.max(np.abs(signed_after-signed_before)))
    np.testing.assert_allclose(signed_after, signed_before, atol=2e-7, rtol=2e-10)
    np.testing.assert_array_equal(after[~np.isin(status[ids], [1, 2])], original[ids[~np.isin(status[ids], [1, 2])]])
    for s in (1, 2):
        # Every signed species imbalance, hence electric charge/baryon number, survives.
        for species in np.unique(np.abs(pid)):
            before = np.count_nonzero((status == s) & (pid == species))-np.count_nonzero((status == s) & (pid == -species))
            aft = np.count_nonzero((status[ids] == s) & (pid[ids] == species))-np.count_nonzero((status[ids] == s) & (pid[ids] == -species))
            if species not in (22, 111, 221, 331) and before != aft:
                raise RuntimeError('Changed signed species charge')
    residual_mass = float(np.max(np.abs(mass2(projected)-invariant[selected])))
    np.testing.assert_allclose(mass2(projected), invariant[selected], atol=2e-7, rtol=1e-7)
    result = dict(event, **{k: np.asarray(event[k])[ids] for k in PARTICLE_KEYS})
    result.update({key: after[:, i] for i, key in enumerate(('px', 'py', 'pz', 'E'))})
    np.testing.assert_allclose(np.hypot(projected[:, 0], projected[:, 1]), new_pt, rtol=1e-12, atol=1e-12)
    qa.update(projection_iterations=checks['iterations'], correction_rms_radians=checks['correction_rms_radians'],
              signed_p4_residual=residual, mass2_residual=residual_mass,
              proposal_residual=checks['proposal_residual'])
    return result, qa
