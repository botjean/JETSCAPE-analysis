"""Independent-review rounded-source singleton regression, no native backend."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bounded_projection as projection
from transport import deform, REFERENCE


def test_unchanged_float32_source_singleton_is_accepted_by_native_mass_equality():
    q = np.array([.8, 0., 2*np.sinh(5.935), 2*np.cosh(5.935)], dtype=np.float32).astype(float)
    mass2 = q[3]**2-(q[:3]**2).sum()
    caps = np.array([1.868*np.hypot(q[0], q[1])])
    interval = projection.bounds(q, caps, np.array([mass2]))
    assert interval is not None
    result, _ = projection.project(q, caps, [mass2], [5.], [.2])
    np.testing.assert_array_equal(result[0], q)


def test_float32_singleton_with_real_calibration_and_other_transformed_sign():
    q = np.array([.8, 0., 2*np.sinh(5.935), 2*np.cosh(5.935)], dtype=np.float32).astype(float)
    pt = np.array([.4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8])
    positive = np.column_stack((pt, pt*0, pt*0, np.sqrt(pt**2+.13957**2)))
    vectors = np.vstack(([10., 0., 0., 10.], positive, q))
    event = dict(zip(('px', 'py', 'pz', 'E'), vectors.T))
    event.update(status=np.r_[-2, np.ones(8, int), 2],
        particle_ID=np.array([21]+[211, -211]*4+[211]), event_ID=321,
        pt_hat=40., vertex_x=.1, vertex_y=.2, event_weight=1.)
    result, qa = deform(event, .5)
    assert qa['whole_event_fallback'] == 0
    assert qa['actual_negative_fraction'] == 1.
    assert qa['pt_envelope_ratio'] <= 1+1e-12
    final = REFERENCE.four_vectors(result)
    np.testing.assert_allclose(final[np.asarray(result['status']) == 2][0], q, atol=2e-7, rtol=2e-10)
