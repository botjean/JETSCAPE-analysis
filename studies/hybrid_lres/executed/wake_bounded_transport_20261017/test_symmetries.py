"""The sensitivity map must not depend on a laboratory azimuth or beam frame."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transport import deform, longitudinal_boost, REFERENCE

spec = importlib.util.spec_from_file_location('bounded_synthetic_fixture', HERE/'test_transport.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def with_vectors(event, vectors):
    return dict(event, **dict(zip(('px', 'py', 'pz', 'E'), vectors.T)))


@pytest.mark.parametrize('angle', [.7, -2., 2.9])
def test_rotation_about_beam_with_real_calibration(angle):
    event = fixture.synthetic_event()
    original, _ = deform(event, .5)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    vector = REFERENCE.four_vectors(event)
    vector[:, :2] = vector[:, :2]@rotation.T
    result, _ = deform(with_vectors(event, vector), .5)
    expected = REFERENCE.four_vectors(original)
    expected[:, :2] = expected[:, :2]@rotation.T
    np.testing.assert_array_equal(result['particle_ID'], original['particle_ID'])
    np.testing.assert_allclose(REFERENCE.four_vectors(result), expected, atol=2e-7, rtol=2e-10)


@pytest.mark.parametrize('rapidity', [-1.2, .8, 2.])
def test_common_longitudinal_boost_with_real_calibration(rapidity):
    event = fixture.synthetic_event()
    original, _ = deform(event, .5)
    vector = longitudinal_boost(REFERENCE.four_vectors(event), rapidity).astype(float)
    result, _ = deform(with_vectors(event, vector), .5)
    expected = longitudinal_boost(REFERENCE.four_vectors(original), rapidity)
    np.testing.assert_array_equal(result['particle_ID'], original['particle_ID'])
    np.testing.assert_allclose(REFERENCE.four_vectors(result), expected, atol=2e-7, rtol=2e-10)
