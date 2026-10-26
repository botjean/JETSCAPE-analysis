"""Retry only the old numerical scale-limit failure along the identical ray.

This changes the root-search coordinate, not momenta, targets or tolerances.
Successful legacy calls return directly, including their exact array object.
"""
from copy import deepcopy

import numpy as np

SCALE_LIMIT_ERROR = 'Cannot bracket per-sign rapidity seed'
POLICY = 'same-ray-rapidity-rescale-on-scale-limit-v1'


class SeedRescaler:
    def __init__(self, original, context=None):
        if isinstance(original, SeedRescaler):
            raise ValueError('Rapidity seed rescaling is already installed')
        if not callable(original) or (context is not None and not callable(context)):
            raise TypeError('Expected solver and optional context callback')
        self.original, self.context, self.repairs = original, context, []

    def __call__(self, mt, preferred, target):
        try:
            return self.original(mt, preferred, target)
        except RuntimeError as error:
            if str(error) != SCALE_LIMIT_ERROR:
                raise
            direction = np.asarray(preferred, dtype=np.longdouble)
            weights = np.asarray(mt, dtype=np.longdouble)
            if (direction.ndim != 1 or direction.shape != weights.shape
                    or not np.isfinite(direction).all() or not np.isfinite(weights).all()
                    or np.any(weights < 0)):
                raise
            direction = direction-direction.mean()
            active = weights > 0
            # Do not replace a legacy degenerate-direction construction or invent
            # freedom carried only by zero-transverse-mass rows.
            if (len(direction) < 2 or np.ptp(direction) < 1e-12
                    or np.count_nonzero(active) < 2 or np.ptp(direction[active]) == 0):
                raise
            norm = abs(direction).max()
            if not np.isfinite(norm) or norm <= 0:
                raise
            event = None if self.context is None else deepcopy(self.context())
            # The original solver still checks feasibility and the |y|<=30 guard.
            # Its caller retains every p4, mass-shell, pT-cap and energy check.
            result = self.original(mt, direction/norm, target)
            self.repairs.append(dict(policy=POLICY, event=event, particles=len(weights),
                coordinate_scale=float(norm), old_preferred_spread=float(np.ptp(direction)),
                final_rapidity_min=float(np.min(result)), final_rapidity_max=float(np.max(result))))
            return result


def install(geometry, context=None):
    wrapper = SeedRescaler(geometry.rapidity_seed, context=context)
    geometry.rapidity_seed = wrapper
    return wrapper
