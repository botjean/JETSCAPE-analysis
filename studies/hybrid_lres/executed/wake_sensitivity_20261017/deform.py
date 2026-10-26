"""Conserving positive-wake stress test, NOT an Efficient Wake implementation."""
from __future__ import annotations

import hashlib
import numpy as np

MODES = ("baseline", "harder", "narrower", "combined")
MOMENTUM_KEYS = ("px", "py", "pz", "E")


def four_vectors(event):
    return np.column_stack([event[key] for key in MOMENTUM_KEYS]).astype(float)


def mass2(v):
    return v[..., 3] ** 2 - np.sum(v[..., :3] ** 2, axis=-1)


def boost(v, beta):
    """Active Lorentz boost; beta broadcasts against the leading dimensions."""
    v, beta = np.asarray(v, float), np.asarray(beta, float)
    b2 = np.sum(beta * beta, axis=-1)
    if np.any(b2 >= 1):
        raise ValueError("Non-timelike boost")
    gamma = 1 / np.sqrt(1 - b2)
    dot = np.sum(v[..., :3] * beta, axis=-1)
    # gamma^2/(gamma+1) is (gamma-1)/beta^2, including beta=0.
    factor = gamma * v[..., 3] + gamma**2 / (gamma + 1) * dot
    return np.concatenate((v[..., :3] + factor[..., None] * beta,
                           (gamma * (v[..., 3] + dot))[..., None]), axis=-1)


def coordinates(v):
    pt = np.hypot(v[..., 0], v[..., 1])
    # Use rapidity, with finite fallback only for zero-transverse-momentum rows.
    transverse_mass = np.sqrt(np.maximum(v[..., 3] ** 2 - v[..., 2] ** 2, 1e-30))
    return pt, np.arcsinh(v[..., 2] / transverse_mass), np.arctan2(v[..., 1], v[..., 0])


def delta_r(v, axis):
    _, y, phi = coordinates(v)
    dphi = (phi - axis[..., 1] + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(y - axis[..., 0], dphi)


def objective(v, axis, mode, width=.4, hard_scale=1.):
    pt, _, _ = coordinates(v)
    angular = 1 / (1 + (delta_r(v, axis) / width) ** 2)
    hard = pt / (pt + hard_scale)
    factor = hard if mode == "harder" else angular if mode == "narrower" else angular * hard
    return np.sum(v[..., 3] * factor, axis=-1)


def scatter_pairs(pairs, axes, mode, rng, fraction=.5, width=.4, hard_scale=1.):
    """Choose a two-body rotation in the pair rest frame, conserving both masses."""
    pairs = np.asarray(pairs, float)
    axes = np.asarray(axes, float)
    output = pairs.copy()
    total = pairs.sum(axis=1)
    s = mass2(total)
    masses = mass2(pairs)
    valid = (s > 1e-10) & np.all(masses >= 0, axis=1) & (total[:, 3] > 0)
    selected = np.flatnonzero(valid)
    if not len(selected):
        return output, 0, int(len(pairs))
    p, axis, ss, mm = pairs[selected], axes[selected], s[selected], masses[selected]
    momentum = total[selected, :3]
    energy = total[selected, 3]
    beta = momentum / energy[:, None]
    root = np.sqrt(ss)
    e1 = (ss + mm[:, 0] - mm[:, 1]) / (2 * root)
    magnitude = np.sqrt(np.maximum(e1**2 - mm[:, 0], 0))
    # Use the pair invariant mass instead of subtracting 1-beta^2 in a boost.
    # That subtraction loses precision for nearly lightlike, forward pairs.
    dot_first = np.sum(momentum*p[:, 0, :3], axis=1)
    original_p = p[:, 0, :3] + momentum*(dot_first/(root*(energy+root)) - p[:, 0, 3]/root)[:, None]
    original = original_p / np.maximum(magnitude[:, None], 1e-30)
    # Aberrate a lightlike vector along the fixed shower axis into this rest frame.
    direction = np.column_stack((np.cos(axis[:, 1]) / np.cosh(axis[:, 0]),
                                 np.sin(axis[:, 1]) / np.cosh(axis[:, 0]), np.tanh(axis[:, 0]),
                                 np.ones(len(axis))))
    dot_axis = np.sum(momentum*direction[:, :3], axis=1)
    toward = direction[:, :3] + momentum*(dot_axis/(root*(energy+root))-1/root)[:, None]
    toward /= np.linalg.norm(toward, axis=1)[:, None]
    speed = np.linalg.norm(beta, axis=1)
    along = beta / np.maximum(speed[:, None], 1e-30)
    random = rng.normal(size=(len(p), 4, 3))
    random /= np.linalg.norm(random, axis=2)[..., None]
    targets = np.concatenate((toward[:, None], -toward[:, None], along[:, None],
                               -along[:, None], random), axis=1)
    directions = (1 - fraction) * original[:, None] + fraction * targets
    norms = np.linalg.norm(directions, axis=2)
    directions /= np.maximum(norms[..., None], 1e-30)
    directions[norms < 1e-12] = original[np.where(norms < 1e-12)[0]]
    dot_direction = np.sum(momentum[:, None, :]*directions, axis=2)
    coefficient = (e1[:, None] + magnitude[:, None]*dot_direction/(energy+root)[:, None])/root[:, None]
    first_p = magnitude[:, None, None]*directions + momentum[:, None, :]*coefficient[..., None]
    first_e = (energy[:, None]*e1[:, None] + magnitude[:, None]*dot_direction)/root[:, None]
    first = np.concatenate((first_p, first_e[..., None]), axis=2)
    # The second body is the exact recoil, not a separately rounded second boost.
    second = total[selected, None, :]-first
    trials = np.stack((first, second), axis=2)
    scores = objective(trials, axis[:, None, None, :], mode, width, hard_scale)
    best = scores.argmax(axis=1)
    proposal = trials[np.arange(len(p)), best]
    old_score = objective(p, axis[:, None, :], mode, width, hard_scale)
    accept = scores[np.arange(len(p)), best] > old_score + 1e-12
    output[selected[accept]] = proposal[accept]
    np.testing.assert_allclose(output.sum(1), total, rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(mass2(output), masses, rtol=1e-8, atol=1e-9)
    return output, int(accept.sum()), int((~valid).sum())


def deform(event, axes, mode, *, seed=20261017, sweeps=2, fraction=.5,
           width=.4, hard_scale=1., association_radius=3.):
    if mode not in MODES:
        raise ValueError("Unknown stress mode")
    if not (0 < fraction < 1 and width > 0 and hard_scale > 0 and association_radius > 0
            and type(sweeps) is int and sweeps >= 0):
        raise ValueError("Invalid deformation settings")
    before = four_vectors(event)
    after = before.copy()
    axes = np.asarray(axes, float).reshape(-1, 2)
    if not np.isfinite(before).all() or not np.isfinite(axes).all():
        raise ValueError("Nonfinite event/axes")
    status = np.asarray(event["status"])
    wake = np.flatnonzero(status == 1)
    qa = dict(accepted_pairs=0, skipped_non_timelike_pairs=0, assigned_wake_particles=0)
    if mode != "baseline" and len(wake) and len(axes):
        distances = delta_r(before[wake, None], axes[None])
        labels = distances.argmin(axis=1)
        assigned = distances.min(axis=1) < association_radius
        qa["assigned_wake_particles"] = int(assigned.sum())
        identity = np.asarray([event.get(k, 0) for k in
                               ("event_ID", "pt_hat", "vertex_x", "vertex_y")], dtype="<f8")
        key = int.from_bytes(hashlib.sha256(identity.tobytes()).digest()[:8], "little")
        rng = np.random.default_rng(np.random.SeedSequence([seed, key]))
        for _ in range(sweeps):
            for label, axis in enumerate(axes):
                ids = rng.permutation(wake[assigned & (labels == label)])
                ids = ids[:len(ids) // 2 * 2].reshape(-1, 2)
                if not len(ids):
                    continue
                proposed, accepted, skipped = scatter_pairs(after[ids], np.repeat(axis[None], len(ids), axis=0),
                    mode, rng, fraction, width, hard_scale)
                after[ids] = proposed
                qa["accepted_pairs"] += accepted
                qa["skipped_non_timelike_pairs"] += skipped
    np.testing.assert_array_equal(after[status != 1], before[status != 1])
    for s in (0, 1, 2):
        np.testing.assert_allclose(after[status == s].sum(0), before[status == s].sum(0), rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(mass2(after), mass2(before), rtol=1e-8, atol=1e-8)
    qa["max_four_momentum_residual_GeV"] = float(np.max(np.abs((after - before).sum(0))))
    qa["max_mass_squared_residual_GeV2"] = float(np.max(np.abs(mass2(after) - mass2(before)), initial=0))
    result = dict(event, **{key: after[:, i] for i, key in enumerate(MOMENTUM_KEYS)})
    return result, qa
