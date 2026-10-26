"""Charge-balanced Bernoulli thinning conditioned on retaining two rows.

The conditioning changes the finite-population sampling law explicitly; it does
not introduce a fitted parameter. No projection or event reconstruction is used.
"""
import numpy as np


SELF_CONJUGATE = (22, 111, 221, 331)


def conditioned_indices(pid, amplitude, rng):
    """Return sorted retained row indices, preserving each species imbalance.

    Self-conjugate rows are size-one Bernoulli units. Other species are paired
    with their antiparticles; unmatched rows are protected. For original groups
    of size >=2, sample that law conditional on retaining >=2 rows, without a
    redraw limit. Empty and singleton groups remain themselves.

    ``pid`` must be a one-dimensional array of nonzero integer IDs, with both
    signs representable in int64. ``amplitude`` is a finite real scalar in (0, 1].
    ``rng`` uses the NumPy Generator random/permutation interface.
    """
    amplitude_array = np.asarray(amplitude)
    if (amplitude_array.ndim != 0 or amplitude_array.dtype.kind not in "fiu"):
        raise ValueError("Amplitude must be a finite real scalar in (0, 1]")
    value = float(amplitude_array)
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Amplitude must be a finite real scalar in (0, 1]")

    pid = np.asarray(pid)
    if pid.ndim != 1:
        raise ValueError("Particle IDs must be a one-dimensional integer array")
    if not pid.size:
        return np.empty(0, dtype=np.intp)
    if (pid.dtype.kind not in "iu"
            or (pid.dtype.kind == "u" and np.any(pid > np.iinfo(np.int64).max))):
        raise ValueError("Particle IDs must be nonzero integers representable with both signs in int64")
    pid = pid.astype(np.int64, copy=False)
    if np.any(pid == 0) or np.any(pid == np.iinfo(np.int64).min):
        raise ValueError("Particle IDs must be nonzero integers representable with both signs in int64")
    if len(pid) == 1:
        return np.array([0], dtype=np.intp)

    units, protected = [], []
    for species in np.unique(np.abs(pid)):
        positive, negative = [np.flatnonzero(pid == sign * species) for sign in (1, -1)]
        if species in SELF_CONJUGATE:
            units.extend([[int(i)] for i in np.r_[positive, negative]])
        else:
            positive, negative = rng.permutation(positive), rng.permutation(negative)
            count = min(len(positive), len(negative))
            units.extend([list(pair) for pair in zip(positive[:count], negative[:count])])
            protected.extend(np.r_[positive[count:], negative[count:]].tolist())
    threshold = max(0, 2 - len(protected))
    # The conditioning event can have probability O(amplitude**2). Extended
    # arithmetic avoids underflow even for subnormal float64 amplitudes.
    probability_keep = np.longdouble(value)
    suffix = np.ones((len(units) + 1, threshold + 1), dtype=np.longdouble)
    suffix[-1, 1:] = 0
    for i in range(len(units) - 1, -1, -1):
        for need in range(1, threshold + 1):
            suffix[i, need] = ((1 - probability_keep) * suffix[i + 1, need]
                + probability_keep * suffix[i + 1, max(0, need - len(units[i]))])
    selected = protected.copy()
    for i, unit in enumerate(units):
        probability = (probability_keep * suffix[i + 1, max(0, threshold - len(unit))]
                       / suffix[i, threshold])
        if rng.random() < probability:
            selected.extend(unit)
            threshold = max(0, threshold - len(unit))
    if len(selected) < 2:
        raise RuntimeError("Conditional balanced thinning failed its retained-row contract")
    return np.sort(selected).astype(np.intp)
