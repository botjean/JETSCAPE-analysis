"""All-positive fixed-pT projections, with constructive feasible starts.

There is no cross-sign optimization. A positive energy sum bounds every
particle's energy. The analytic start is only a numerical construction; the
original fixed-pT/mass/four-momentum checks still validate the answer.
"""
import numpy as np
from scipy.optimize import brentq


def wrapped(x):
    return (x+np.pi) % (2*np.pi)-np.pi


def polygon_angles(pt, preferred, target):
    pt, preferred, target = np.asarray(pt), np.asarray(preferred), np.asarray(target)
    order = np.argsort(-pt, kind='stable')
    lengths, phase = pt[order], preferred[order]
    tail = np.cumsum(lengths[::-1])[::-1]
    original = np.cumsum((lengths*np.exp(1j*phase))[::-1])[::-1]
    angles = np.empty(len(pt))
    vector = complex(*target)
    for i, length in enumerate(lengths):
        radius = abs(vector)
        if i == len(pt)-1:
            angles[i] = np.angle(vector) if radius else phase[i]
            break
        rest_max = tail[i+1]
        rest_min = max(0., 2*lengths[i+1]-rest_max)
        low = max(rest_min, abs(radius-length))
        high = min(rest_max, radius+length)
        if low > high+1e-10*max(1., tail[0]):
            raise RuntimeError('Per-sign transverse polygon is infeasible')
        rest = np.clip(abs(original[i+1]), min(low, high), high)
        if radius == 0:
            angle = phase[i]
        elif length == 0:
            angle = phase[i]
        else:
            cosine = np.clip((radius*radius+length*length-rest*rest)/(2*radius*length), -1., 1.)
            deviation = np.arccos(cosine)
            options = np.angle(vector)+np.array([-deviation, deviation])
            angle = options[np.argmin(abs(wrapped(options-phase[i])))]
        angles[i] = angle
        vector -= length*np.exp(1j*angle)
    result = np.empty(len(pt))
    result[order] = angles
    return result


def rapidity_seed(mt, preferred, target):
    mt = np.asarray(mt, dtype=np.longdouble)
    target = np.asarray(target, dtype=np.longdouble)
    mass2 = (target[3]-target[2])*(target[3]+target[2])
    if target[3] <= 0 or mass2 <= 0:
        raise RuntimeError('Non-future per-sign longitudinal target')
    mass = np.sqrt(mass2)
    minimum = mt.sum()
    margin = 128*np.finfo(float).eps*max(1., float(target[3]), float(minimum))
    if mass < minimum-margin:
        raise RuntimeError('Per-sign longitudinal mass is infeasible')
    y = np.asarray(preferred, dtype=np.longdouble)
    y = y-y.mean()
    if mass <= minimum+margin:
        spread = np.zeros(len(mt), dtype=np.longdouble)
    else:
        if len(mt) < 2:
            raise RuntimeError('Singleton cannot change its longitudinal mass')
        if np.ptp(y) < 1e-12:
            y = np.zeros(len(mt), dtype=np.longdouble)
            y[0] = 1.
            y -= y.mean()

        def group_mass(scale):
            v = scale*y
            if abs(v).max() > 60:
                return np.inf
            return float(np.sqrt((mt*np.exp(v)).sum()*(mt*np.exp(-v)).sum()))

        high = 1.
        while group_mass(high) < mass:
            high *= 2
            if high > 2**20:
                raise RuntimeError('Cannot bracket per-sign rapidity seed')
        scale = brentq(lambda value: group_mass(value)-float(mass), 0., high, xtol=1e-14)
        spread = scale*y
    # Match the well-resolved light-cone component directly. Near a minimum-mass
    # forward target, its tiny rounded mass deficit must not shift both E and pz.
    if target[2] >= 0:
        shift = np.log(target[3]+target[2])-np.log((mt*np.exp(spread)).sum())
    else:
        shift = np.log((mt*np.exp(-spread)).sum())-np.log(target[3]-target[2])
    result = np.asarray(spread+shift, float)
    if abs(result).max() > 30:
        raise RuntimeError('Constructive sign projection exceeds the original rapidity guard')
    return result


def project_positive(proposal, masses2, target, original_solver, coordinates):
    pt, y, phi = coordinates(proposal)
    mt = np.sqrt(pt**2+masses2)
    candidates, errors = [], []
    signs = np.ones(len(pt))
    for label in ('proposal', 'constructive'):
        try:
            seed = proposal
            if label == 'constructive':
                angles = polygon_angles(pt, phi, np.asarray(target)[:2])
                rapidity = rapidity_seed(mt, y, target)
                seed = np.column_stack((pt*np.cos(angles), pt*np.sin(angles),
                                        mt*np.sinh(rapidity), mt*np.cosh(rapidity)))
            if label == 'constructive':
                # Exact polygon endpoints are physically valid, although the
                # old Newton precheck excludes them to avoid a singular step.
                # A constructive answer needs no step: verify the same p4
                # tolerance directly, rather than weakening the Newton solver.
                four = seed
                residual = float(abs(four.sum(0)-target).max())
                if residual > 2e-12*max(1., float(abs(target).max())):
                    raise RuntimeError('Constructive state misses original conservation tolerance')
                checks = dict(iterations=0, residual=residual)
            else:
                four, checks = original_solver(seed, masses2, signs, target)
            _, final_y, final_phi = coordinates(four)
            correction = float(np.sqrt(np.mean(wrapped(final_phi-phi)**2+(final_y-y)**2)))
            np.testing.assert_allclose(four.sum(0), target, atol=2e-8, rtol=2e-12)
            np.testing.assert_allclose(np.hypot(four[:, 0], four[:, 1]), pt, atol=1e-12, rtol=1e-12)
            if np.any(four[:, 3] < 0) or four[:, 3].max() > target[3]+2e-8:
                raise RuntimeError('Per-sign energy bound violated')
            checks.update(correction_rms_radians=correction, seed=label)
            candidates.append((correction, four, checks))
        except RuntimeError as error:
            errors.append(label+': '+str(error))
    if not candidates:
        raise RuntimeError('Independent sign projection failed: '+'; '.join(errors))
    _, four, checks = min(candidates, key=lambda row: row[0])
    return four, checks
