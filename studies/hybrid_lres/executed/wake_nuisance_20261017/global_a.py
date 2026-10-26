"""One global wake deformation; reuse, never edit, the reviewed transport."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
FROZEN = WORK / 'wake_paper_morph_20261017'
PARAMETER_NAMES = ('positive_pt_scale', 'negative_pt_scale',
                   'positive_phi_scale', 'positive_y_scale')


def scalar_a(value):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'fiu':
        raise ValueError('a must be one global real scalar, not an array or boolean')
    result = float(array)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('This table supports only 0 <= a <= 1; no extrapolation')
    return result


def load_frozen(name, filename):
    sys.path[:0] = [str(FROZEN), str(WORK / 'wake_sensitivity_20261017')]
    spec = importlib.util.spec_from_file_location(name, FROZEN / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=16)
def _transport(a):
    # Each fixed a gets an isolated module cache, not mutable shared parameters.
    module = load_frozen('wake_global_a_' + float(a).hex(), 'transport.py')
    record = next(r for r in json.loads((FROZEN / 'CALIBRATION.json').read_text())
                  if r['energy'] == 50)
    endpoint = np.array([record['parameters'][key] for key in PARAMETER_NAMES])
    parameters = np.exp(a * np.log(endpoint))
    amplitudes = module.Model(record['temperature'], record['deposition']).amplitudes(parameters)
    if np.any(amplitudes <= 0) or np.any(amplitudes > 1 + 1e-12):
        raise ValueError('The frozen thinning algorithm cannot implement these amplitudes')
    module.PARAMETERS = {'paper50': (parameters, amplitudes)}
    return module


def deform(event, a, *, seed=20261018):
    a = scalar_a(a)
    module = _transport(a)
    return module.deform(event, 'baseline' if a == 0 else 'paper50', seed=seed)


def description(a):
    a = scalar_a(a)
    parameters, amplitudes = _transport(a).PARAMETERS['paper50']
    return dict(a=a, global_across_designs_and_observables=True,
                parameters=dict(zip(PARAMETER_NAMES, parameters.tolist())),
                amplitudes=amplitudes.tolist(), benchmark_parton_energy_GeV=50)
