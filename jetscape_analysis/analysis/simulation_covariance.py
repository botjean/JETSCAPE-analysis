"""Reusable event-count moments and ratio covariance propagation.

Callers supply aligned counts from the same event, including empty events, and
are responsible for event identities, source provenance and observable selection.
No saved-STAT extraction adapter, file join or command-line producer is supplied.
"""

from __future__ import annotations

import numpy as np


class EventMoments:
    """Joint event-Poisson statistics; each update is one caller-aligned event."""

    def __init__(self, n_features):
        self.total = np.zeros(n_features + 1)
        self.outer = np.zeros((n_features + 1, n_features + 1))
        self.events = 0

    def update(self, counts, weight):
        counts = np.asarray(counts, dtype=float)
        if counts.shape != self.total[:-1].shape or not np.isfinite(counts).all():
            raise ValueError("Invalid event contribution vector")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Invalid event weight")
        # The final component retains fluctuations of the all-event denominator.
        vector = weight * np.r_[counts, 1.0]
        self.total += vector
        self.outer += np.outer(vector, vector)
        self.events += 1


def histogram_counts(values, edges):
    """ROOT [low, high) bins; also return underflow + visible normalization."""
    values = np.asarray([] if values is None else values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite, already selected scalar observables")
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Invalid histogram edges")
    counts = np.histogram(values[(values >= edges[0]) & (values < edges[-1])], edges)[0]
    return counts, int(np.count_nonzero(values < edges[-1]))


def ratio_map(total, numerators, denominators, factors=1.0):
    """Values and exact Jacobian for scaled ratios of linear sufficient stats."""
    total = np.asarray(total, dtype=float)
    a, b = np.asarray(numerators, float), np.asarray(denominators, float)
    if a.ndim != 2 or b.shape != a.shape or a.shape[1] != len(total):
        raise ValueError("Incompatible ratio maps")
    num, den = a @ total, b @ total
    if not np.isfinite(total).all() or np.any(den <= 0):
        raise ValueError("Nonfinite statistics or nonpositive denominator")
    scale = np.broadcast_to(factors, num.shape)
    if not np.isfinite(scale).all():
        raise ValueError("Nonfinite ratio scale")
    jac = scale[:, None] * (a / den[:, None] - num[:, None] * b / den[:, None]**2)
    return scale * num / den, jac


def ratio_bank(aa, aa_cov, pp, pp_cov):
    """Full design x feature covariance for independent AA and one shared pp.

    Inputs must already include within-sample normalization and bin/observable
    cross-covariance. No assumption that different observable blocks are diagonal.
    """
    aa, aa_cov, pp, pp_cov = map(np.asarray, (aa, aa_cov, pp, pp_cov))
    if aa.ndim != 2 or pp.shape != aa.shape[1:]:
        raise ValueError("Incompatible spectra")
    n, f = aa.shape
    if aa_cov.shape != (n, f, f) or pp_cov.shape != (f, f):
        raise ValueError("Incompatible covariance dimensions")
    if not all(np.isfinite(v).all() for v in (aa, aa_cov, pp, pp_cov)) or np.any(pp <= 0):
        raise ValueError("Invalid spectra or empty pp bin")
    derivative = -aa / pp**2
    covariance = np.einsum("ib,bc,jc->ibjc", derivative, pp_cov, derivative)
    for i in range(n):
        covariance[i, :, i, :] += aa_cov[i] / np.outer(pp, pp)
    return aa / pp, covariance
