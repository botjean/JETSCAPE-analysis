"""Regression tests for the full-population endpoint certificate."""
import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("certificate_projection_test_subject", HERE / "bounded_projection.py")
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)

SPEC = importlib.util.spec_from_file_location("certificate_transport_test_subject", HERE / "transport.py")
transport = importlib.util.module_from_spec(SPEC)
previous_path = sys.path.copy()
sys.path.insert(0, str(HERE))
try:
    SPEC.loader.exec_module(transport)
finally:
    sys.path[:] = previous_path


def original_wake():
    pt = np.array([.4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8])
    energy = np.sqrt(pt ** 2 + .13957 ** 2)
    return np.column_stack((pt, np.zeros(8), np.zeros(8), energy))


def test_full_original_certificate_survives_lower_endpoint_float_conversion():
    original = original_wake()
    masses2 = original[:, 3] ** 2 - (original[:, :3] ** 2).sum(1)
    target = original.astype(np.longdouble).sum(0)
    # The original state passes the final representation/conservation guards.
    np.testing.assert_allclose(original.sum(0), target, atol=2e-7, rtol=2e-10)
    np.testing.assert_allclose(original[:, 3] ** 2 - (original[:, :3] ** 2).sum(1),
                               masses2, atol=2e-7, rtol=1e-7)
    interval = projection.bounds(target, 1.25 * original[:, 0], masses2)
    assert interval is not None
    assert interval[0] - 2e-12 <= .8 <= interval[1] + 2e-12


def test_transport_real_calibration_a_half_collinear_eight_row_population():
    vectors = np.vstack(([10., 0., 0., 10.], original_wake()))
    event = dict(zip(("px", "py", "pz", "E"), vectors.T))
    event.update(status=np.r_[-2, np.ones(8, int)],
                 particle_ID=np.array([21] + [211, -211] * 4),
                 event_ID=123, pt_hat=40., vertex_x=0., vertex_y=0., event_weight=1.)
    result, qa = transport.deform(event, .5)
    assert qa["whole_event_fallback"] == 0
    assert np.isfinite(transport.REFERENCE.four_vectors(result)).all()
