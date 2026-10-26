"""Small local native smoke tests; the full historical policy suite is preserved."""
import math

import pytest

native = pytest.importorskip("_wake_dr12_diagnostic", reason="Explicit optional hardest build required")


def row(pt, phi=0.):
    return [pt * math.cos(phi), pt * math.sin(phi), 0., pt, 1., 1., 1.]


def test_regular_two_prong_kt_split():
    result = native.reconstruct([row(150), row(80, .4)])
    assert len(result["jets"]) == 1
    assert result["jets"][0][1] == pytest.approx(.4)
    assert result["jets"][0][2] == pytest.approx(32.)
    assert not result["unresolved"]


@pytest.mark.parametrize("options", [dict(ghost_pt=0), dict(groomer="bad"), dict(legacy_marker=1)])
def test_unchanged_native_guards(options):
    with pytest.raises(ValueError):
        native.reconstruct([], **options)
