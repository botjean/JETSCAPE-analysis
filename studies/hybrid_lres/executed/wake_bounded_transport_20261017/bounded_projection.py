"""Per-sign conservation with a paper-derived transverse-momentum envelope.

Only downward changes of the proposed pT are allowed. The remaining transverse
polygon and longitudinal mass constraints are solved independently. No rest-
frame rescaling can turn longitudinal energy into unbounded transverse pT.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

LEGACY = Path(__file__).resolve().parent.parent/'wake_sign_separated_20261017/projection.py'
spec = importlib.util.spec_from_file_location('verified_positive_geometry', LEGACY)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
LD = np.longdouble


def longitudinal_mass(target):
    target = np.asarray(target, dtype=LD)
    if target[3] < abs(target[2]):
        return None
    return np.sqrt((target[3]-target[2])*(target[3]+target[2]))


def bounds(target, caps, masses2):
    target, caps, masses2 = [np.asarray(x, dtype=LD) for x in (target, caps, masses2)]
    if (target.shape != (4,) or caps.ndim != 1 or not len(caps)
            or caps.shape != masses2.shape or np.any(caps < 0) or np.any(masses2 < 0)
            or not np.isfinite(np.r_[target, caps, masses2]).all()):
        raise ValueError('Invalid bounded sign-projection inputs')
    radius = np.hypot(*target[:2])
    total = caps.sum()
    lm = longitudinal_mass(target)
    if lm is None or target[3] <= 0:
        return None
    low = radius/total if total > 0 else LD(0)
    if total == 0 and radius > 0:
        return None
    minimum_polygon = max(LD(0), 2*caps.max(initial=0)-total)
    high = min(LD(1), radius/minimum_polygon) if minimum_polygon > 0 else LD(1)
    tolerance = 32*np.finfo(float).eps
    if low > high+tolerance:
        return None
    low = min(low, high)
    mt = lambda damping: np.sqrt(masses2+(damping*caps)**2).sum()
    # Use the small, physical longitudinal mass as the error scale. Scaling a
    # polynomial tolerance by large cancelling lab energies accepts bad budgets.
    margin = tolerance*max(LD(1), lm)
    active = (masses2+(high*caps)**2) > 0
    if np.count_nonzero(active) < 2:
        if not active.any():
            return None
        # A singleton's native float64 invariant mass is its equality test.
        # Comparing separately rounded square roots in extended precision can
        # reject the unchanged float32-source particle itself. Use exactly the
        # downstream mass guard; this is not extra internal rapidity freedom.
        native = np.asarray(target, float)
        native_mass2 = native[3]**2-(native[:3]**2).sum()
        required = float(masses2[active][0])
        if abs(native_mass2-required) > 2e-7+1e-7*abs(required):
            return None
        return float(low), float(high)
    if mt(low) > lm+margin:
        return None
    if mt(high) > lm:
        if mt(low) >= lm-margin:
            # This closed endpoint is already valid at the existing numerical
            # precision. Converting it to float64 can reverse a tiny root sign.
            high = low
        else:
            high = LD(brentq(lambda d: float(mt(d)-lm), float(low), float(high), xtol=1e-14))
    return float(low), float(max(low, high))


def choose_budgets(positive, negative, caps, masses2, signs, requested):
    positive, negative = np.asarray(positive, dtype=LD), np.asarray(negative, dtype=LD)
    caps, masses2, signs = np.asarray(caps), np.asarray(masses2), np.asarray(signs)
    fraction = np.asarray(requested)
    if (fraction.ndim != 0 or fraction.dtype.kind not in 'fiu'
            or not np.isfinite(fraction) or not 0 < float(fraction) <= 1
            or positive.shape != (4,) or negative.shape != (4,)
            or caps.ndim != 1 or caps.shape != masses2.shape or caps.shape != signs.shape
            or not np.isin(signs, (1, -1)).all() or np.any(caps < 0) or np.any(masses2 < 0)
            or not np.isfinite(np.r_[positive, negative, caps, masses2]).all()):
        raise ValueError('Invalid matched-budget inputs or requested fraction')
    difference = positive-negative
    force_original_fraction = (not np.any(positive) and np.any(negative)) or any(
        np.count_nonzero(signs == s) == 1 for s in (1, -1))
    requested = 1. if force_original_fraction else float(requested)

    def state(fraction):
        targets = (difference+LD(fraction)*negative, LD(fraction)*negative)
        for s, target in zip((1, -1), targets):
            rows = signs == s
            if not rows.any():
                if np.any(target != 0):
                    return None
            elif bounds(target, caps[rows], masses2[rows]) is None:
                return None
        return targets

    targets = state(requested)
    if targets is not None:
        return requested, targets
    if state(1.) is None:
        return None
    # The feasible target set is convex along this affine budget line. Starting
    # at its certified f=1 endpoint, bisection finds the nearest allowed fraction.
    low, high = requested, 1.
    for _ in range(42):
        middle = (low+high)/2
        if state(middle) is None:
            low = middle
        else:
            high = middle
    return high, state(high)


def cloud_mass(mt, y):
    y = np.asarray(y, dtype=LD)
    centered = y-y.mean()
    return np.sqrt((mt*np.exp(centered)).sum()*(mt*np.exp(-centered)).sum())


def choose_pt(target, caps, masses2, proposed_y):
    interval = bounds(target, caps, masses2)
    if interval is None:
        raise RuntimeError('No bounded transverse configuration for this sign budget')
    low, high = interval
    lm = longitudinal_mass(target)
    caps, masses2 = np.asarray(caps, dtype=LD), np.asarray(masses2, dtype=LD)

    def invariant(damping):
        return cloud_mass(np.sqrt(masses2+(LD(damping)*caps)**2), proposed_y)

    # Preserve the proposed rapidity spread when possible, rather than maximize
    # pT until every rapidity is forced onto a degenerate common value.
    if invariant(low) >= lm:
        damping = low
    elif invariant(high) <= lm:
        damping = high
    else:
        damping = brentq(lambda d: float(invariant(d)-lm), low, high, xtol=1e-14)
    return np.asarray(LD(damping)*caps, float), float(damping)


def match_phi(pt, initial, target):
    phi = np.asarray(initial, float).copy()
    tolerance = 2e-12*max(1., float(np.hypot(*target)), float(pt.sum()))
    for _ in range(100):
        residual = np.array([(pt*np.cos(phi)).sum(), (pt*np.sin(phi)).sum()])-target
        if max(abs(residual)) <= tolerance:
            return phi
        jacobian = np.array([-pt*np.sin(phi), pt*np.cos(phi)])
        multiplier = np.linalg.lstsq(jacobian@jacobian.T, np.asarray(residual, float), rcond=1e-13)[0]
        step = -jacobian.T@multiplier
        fraction = min(1., .5/max(abs(step).max(), 1e-30))
        old_norm = np.linalg.norm(residual)
        for _ in range(22):
            trial = phi+fraction*step
            value = np.array([(pt*np.cos(trial)).sum(), (pt*np.sin(trial)).sum()])-target
            if np.linalg.norm(value) < old_norm:
                phi = trial
                break
            fraction *= .5
        else:
            break
    return None


def polish_phi(pt, phi, preferred, target):
    cost = lambda value: np.mean(geometry.wrapped(value-preferred)**2)
    old_cost = cost(phi)
    accepted = 0
    for _ in range(16):
        gradient = geometry.wrapped(phi-preferred)
        jacobian = np.array([-pt*np.sin(phi), pt*np.cos(phi)])
        normal = jacobian@jacobian.T
        tangent = gradient-jacobian.T@np.linalg.lstsq(normal, jacobian@gradient, rcond=1e-13)[0]
        if np.linalg.norm(tangent) < 1e-7:
            break
        step = min(1., .25/max(abs(tangent).max(), 1e-30))
        for _ in range(8):
            candidate = match_phi(pt, phi-step*tangent, target)
            if candidate is not None and cost(candidate) < old_cost-1e-10:
                phi, old_cost = candidate, cost(candidate)
                accepted += 1
                break
            step *= .5
        else:
            break
    return phi, accepted


def construct_phi(pt, preferred, target):
    try:
        return geometry.polygon_angles(pt, preferred, target)
    except RuntimeError:
        # Sub-ULP sides cannot absorb a roundoff remainder from a long polygon.
        # Fix only their azimuths at the proposal, subtract their FULL vectors,
        # and construct the resolved polygon. No pT or particle is removed.
        small = pt <= 64*np.finfo(float).eps*pt.sum()
        if not small.any() or small.all():
            raise
        fixed = np.array([(pt[small]*np.cos(preferred[small])).sum(),
                          (pt[small]*np.sin(preferred[small])).sum()])
        result = preferred.copy()
        result[~small] = geometry.polygon_angles(pt[~small], preferred[~small], target-fixed)
        actual = np.array([(pt*np.cos(result)).sum(), (pt*np.sin(result)).sum()])
        if abs(actual-target).max() > 2e-12*max(1., float(pt.sum()), float(np.hypot(*target))):
            raise RuntimeError('Resolved polygon fails the unchanged full-vector conservation check')
        return result


def project(target, caps, masses2, proposed_y, proposed_phi):
    proposed_y, proposed_phi = np.asarray(proposed_y, float), np.asarray(proposed_phi, float)
    if (proposed_y.shape != np.asarray(caps).shape or proposed_phi.shape != np.asarray(caps).shape
            or not np.isfinite(np.r_[proposed_y, proposed_phi]).all()):
        raise ValueError('Invalid proposed angular coordinates')
    pt, damping = choose_pt(target, caps, masses2, proposed_y)
    target = np.asarray(target, dtype=LD)
    if len(pt) == 1:
        result = np.array([target], float)
        np.testing.assert_allclose(np.hypot(result[:, 0], result[:, 1]), pt, atol=2e-8, rtol=2e-12)
        np.testing.assert_allclose(result[:, 3]**2-(result[:, :3]**2).sum(1), masses2, atol=2e-7, rtol=1e-7)
        dp = geometry.wrapped(np.arctan2(result[:, 1], result[:, 0])-proposed_phi)
        dy = np.arcsinh(result[:, 2]/np.sqrt(np.asarray(masses2)+pt**2))-proposed_y
        return result, dict(pt_damping=damping, phi_correction=float(abs(dp[0])),
            y_correction=float(abs(dy[0])), correction_rms_radians=float(np.hypot(dp[0], dy[0])))
    preferred = proposed_phi.copy()
    # Closest linearized transverse corrections, with the constructive polygon
    # as a certified boundary/failure seed, not an unrestricted momentum kick.
    candidates = []
    initial = match_phi(pt, preferred, target[:2])
    if initial is not None:
        candidates.append(initial)
    constructor_available = True
    try:
        constructed = construct_phi(pt, preferred, np.asarray(target[:2], float))
        candidates.append(constructed)
    except RuntimeError:
        # An optional analytic start must not invalidate an already conserved
        # Newton solution. All final p4/mass/envelope guards still run below.
        constructor_available = False
        if not candidates:
            raise
    phi = min(candidates, key=lambda v: np.mean(geometry.wrapped(v-preferred)**2))
    phi, polish_steps = polish_phi(pt, phi, preferred, target[:2])
    mt = np.sqrt(np.asarray(masses2, dtype=LD)+pt.astype(LD)**2)
    y = geometry.rapidity_seed(mt, proposed_y, target)
    result = np.column_stack((pt*np.cos(phi), pt*np.sin(phi), mt*np.sinh(y.astype(LD)), mt*np.cosh(y.astype(LD)))).astype(float)
    np.testing.assert_allclose(result.sum(0), target, atol=2e-8, rtol=2e-12)
    np.testing.assert_allclose(result[:, 3]**2-(result[:, :3]**2).sum(1), masses2, atol=2e-7, rtol=1e-7)
    np.testing.assert_allclose(np.hypot(result[:, 0], result[:, 1]), pt, atol=1e-12, rtol=1e-12)
    if np.any(pt > np.asarray(caps)*(1+1e-12)+1e-12) or np.any(result[:, 3] < 0):
        raise RuntimeError('Bounded projection violated its positive-energy/pT contract')
    dp, dy = geometry.wrapped(phi-preferred), y-proposed_y
    return result, dict(pt_damping=damping, phi_correction=float(np.sqrt(np.mean(dp**2))),
        y_correction=float(np.sqrt(np.mean(dy**2))), correction_rms_radians=float(np.sqrt(np.mean(dp**2+dy**2))),
        phi_polish_steps=polish_steps, analytic_constructor_available=constructor_available)
