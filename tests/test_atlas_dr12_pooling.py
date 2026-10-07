"""Independent formula checks for legacy normalization/errors and centrality mixing."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("jetscape_analysis.analysis._atlas_dr12")
from jetscape_analysis.analysis import atlas_dr12 as analysis


@pytest.fixture
def config():
    c = analysis.load_config(Path(__file__).parents[1] / "config/atlas_dr12_5020.yaml")
    c["observables"] = {"dr12_atlas": {"column": 1, "edges": [0, 1]}}
    return c


def event(sigma=10, accepted=True, weight=1, two_jets=False):
    angles = [0, .4, 3, 3.4] if two_jets else [0, .4]
    pt = [150, 80] * (2 if two_jets else 1)
    if not accepted:
        angles, pt = [], []
    return dict(px=(np.array(pt)*np.cos(angles)).tolist(), py=(np.array(pt)*np.sin(angles)).tolist(),
                pz=[0]*len(pt), E=pt, particle_ID=[211]*len(pt), status=[0]*len(pt),
                event_weight=weight, cross_section=sigma)


def sample(tmp_path, config, name, events, centrality=None, identity=None):
    data, info = analysis.analyze_events(events, config, "PbPb" if centrality else "pp",
                                       centrality, "all", design_id=0 if centrality else None,
                                       normalization_id=identity)
    info["sources"] = [name]
    p = tmp_path / f"{name}.npz"
    analysis.save_sample(p, data, info)
    return p


def test_legacy_per_jet_errors_default_and_event_option(tmp_path, config):
    e = event(weight=2, two_jets=True)
    p = sample(tmp_path, config, "legacy", [e, event(accepted=False, weight=3)])
    y, c, _, _ = analysis.spectrum([p], "dr12_atlas", "pp")
    np.testing.assert_allclose(y, [8])  # 10 / (2+3) * (2*2)
    np.testing.assert_allclose(c, [[32]])  # (10/5)^2 * (2 jets * 2^2)
    config["mc_error"] = "event"
    p = sample(tmp_path, config, "event", [e, event(accepted=False, weight=3)])
    ye, ce, _, _ = analysis.spectrum([p], "dr12_atlas", "pp")
    np.testing.assert_array_equal(ye, y)
    np.testing.assert_allclose(ce, [[64]])


def test_different_cross_sections_match_legacy_entry_mean(tmp_path, config):
    a = sample(tmp_path, config, "a", [event(8), event(8, accepted=False)])
    b = sample(tmp_path, config, "b", [event(12)] * 4)
    p = tmp_path / "merged.npz"
    analysis.merge_samples([a, b], p)
    y, c, _, _ = analysis.spectrum([p], "dr12_atlas", "pp")
    np.testing.assert_allclose(y, [((8+12)/2) / 6 * 5])
    np.testing.assert_allclose(c, [[(10/6)**2 * 5]])


def test_split_merge_and_merge_tree_invariant(tmp_path, config):
    events = [event(8, weight=2), event(8, accepted=False, weight=3), event(8)]
    whole = sample(tmp_path, config, "whole", events, identity="original_file_A")
    chunks = [sample(tmp_path, config, f"a{i}", [e], identity="original_file_A") for i, e in enumerate(events)]
    other = sample(tmp_path, config, "b", [event(12)], identity="original_file_B")
    direct, split, staged = [tmp_path / f"{name}.npz" for name in ("direct", "split", "staged")]
    analysis.merge_samples([whole, other], direct)
    analysis.merge_samples(chunks, split)
    analysis.merge_samples([other, split], staged)
    for i in (0, 1):
        np.testing.assert_allclose(analysis.spectrum([direct], "dr12_atlas", "pp")[i],
                                   analysis.spectrum([staged], "dr12_atlas", "pp")[i])
    _, info = analysis.load_sample(staged)
    assert info["cross_section"] == 10
    assert len(info["cross_section_entries"]) == 2


def test_reused_identity_requires_consistent_sigma(tmp_path, config):
    a = sample(tmp_path, config, "a", [event(8)], identity="same_file")
    b = sample(tmp_path, config, "b", [event(12)], identity="same_file")
    with pytest.raises(ValueError, match="normalization_id"):
        analysis.merge_samples([a, b], tmp_path / "bad.npz")


def components(tmp_path, config):
    a = sample(tmp_path, config, "aa05", [event(8), event(8, accepted=False)], [0, 5])
    b = sample(tmp_path, config, "aa510", [event(12)] + [event(12, accepted=False)]*3, [5, 10])
    return [dict(centrality=[0, 5], weight=.5, samples=[a]),
            dict(centrality=[5, 10], weight=.5, samples=[b])]


def test_equal_mixture_of_normalized_spectra_not_counts(tmp_path, config):
    parts = components(tmp_path, config)
    y, c, _, _ = analysis.centrality_spectrum(parts, "dr12_atlas", 0)
    np.testing.assert_allclose(y, [3.5])
    np.testing.assert_allclose(c, [[.25*(4**2+3**2)]])
    pp = sample(tmp_path, config, "pp", [event(), event()])
    yp, cp, _, _ = analysis.spectrum([pp], "dr12_atlas", "pp")
    ratio, cov = analysis.ratio(y, c, yp, cp)
    np.testing.assert_allclose(ratio, [.35])
    # Shared pp is propagated ONCE, after mixing AA; not as independent pp copies.
    np.testing.assert_allclose(cov, [[.12375]])


def test_centrality_mixture_required_and_bad_weights_rejected(tmp_path, config):
    parts = components(tmp_path, config)
    with pytest.raises(ValueError, match="centrality"):
        analysis.spectrum(parts[0]["samples"], "dr12_atlas", "PbPb", 0)
    with pytest.raises(ValueError, match="centrality"):
        analysis.merge_samples(parts[0]["samples"] + parts[1]["samples"], tmp_path / "bad.npz")
    for weight in (0, -.1, .6, float("nan")):
        bad = deepcopy(parts)
        bad[0]["weight"] = weight
        with pytest.raises(ValueError, match="weights"):
            analysis.centrality_spectrum(bad, "dr12_atlas", 0)
    bad = [parts[0], deepcopy(parts[0])]
    with pytest.raises(ValueError, match="share events"):
        analysis.centrality_spectrum(bad, "dr12_atlas", 0)


def test_explicit_centrality_export_values_and_errors(tmp_path, config):
    parts = components(tmp_path, config)
    pp = sample(tmp_path, config, "pp", [event(), event()])
    data = tmp_path / "Data__5020__PbPb__inclusive_jet__dr12_atlas__R1.0_pt200-251_trk4_zcut0.15_beta0__0-10.dat"
    np.savetxt(data, [[0, 1, .35, .1]])
    design = tmp_path / "Design__toy.dat"
    design.write_text("# Design point indices (row index): 0\n0.5 0.5\n")
    for c in parts:
        c["samples"] = [p.name for p in c["samples"]]
    manifest = dict(observable="dr12_atlas", pp=[pp.name],
                    design_points={0: dict(centrality_components=parts)},
                    data_table=data.name, design_table=design.name)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    out = tmp_path / "tables"
    analysis.export_tables(path, out)
    assert np.loadtxt(next((out / "Prediction").glob("*values.dat"))) == pytest.approx(.35)
    assert np.loadtxt(next((out / "Prediction").glob("*errors.dat"))) == pytest.approx(np.sqrt(.12375))
    assert (out / "Design" / design.name).read_bytes() == design.read_bytes()
