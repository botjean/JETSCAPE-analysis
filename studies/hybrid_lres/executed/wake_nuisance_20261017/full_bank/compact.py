"""Lossless event-count storage, preserving every measured noise cross term."""
import numpy as np


def pack(counts):
    counts = np.asarray(counts)
    if counts.ndim != 4 or not np.isfinite(counts).all():
        raise ValueError('Expected event,design,a,feature integer counts')
    if not np.issubdtype(counts.dtype, np.integer) and np.any(counts != np.rint(counts)):
        raise ValueError('Refuse to round noninteger counts')
    base = counts[:, :, 0]
    limits = np.iinfo(np.int16)
    if counts.size and (counts.min() < limits.min or counts.max() > limits.max):
        raise ValueError('Signed int16 count range exceeded; do not truncate')
    delta = np.empty(counts[:, :, 1:].shape, np.int16)
    base32 = base.astype(np.int32)
    for node in range(1, counts.shape[2]):
        difference = counts[:, :, node].astype(np.int32)-base32
        if difference.size and (difference.min() < limits.min or difference.max() > limits.max):
            raise ValueError('Signed int16 count range exceeded; do not truncate')
        delta[:, :, node-1] = difference
    return dict(baseline_counts=base.astype(np.int16, copy=False), delta_counts=delta)


def unpack(archive):
    base = np.asarray(archive['baseline_counts'], dtype=np.int32)
    delta = np.asarray(archive['delta_counts'], dtype=np.int32)
    return np.concatenate((base[:, :, None], base[:, :, None]+delta), axis=2)


def weighted(archive, features=None):
    counts = unpack(archive)
    if features is not None:
        counts = counts[..., features]
    weights = np.asarray(archive['fingerprints'])[:, 1]
    return counts.astype(float)*weights[:, None, None, None]


def blocked_split(physics_design_ids, held_out):
    """Never place another a node of a held-out physics design in training."""
    ids = np.asarray(physics_design_ids)
    test = np.isin(ids, held_out)
    if not test.any() or test.all():
        raise ValueError('Split needs both training and held-out physical designs')
    return np.flatnonzero(~test), np.flatnonzero(test)
