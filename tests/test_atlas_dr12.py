"""Real FastJet tests, with analytic toy events (not physics production results)."""

from copy import deepcopy
import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("particle", reason="Install requirements-atlas-dr12.txt for optional analysis tests")
native = pytest.importorskip("jetscape_analysis.analysis._atlas_dr12", reason="Run scripts/build_atlas_dr12.py")
from jetscape_analysis.analysis import atlas_dr12 as analysis


def row(pt, phi=0.0, eta=0.0, sign=1, calo=1, track=1, mass=0.0):
    return [
        pt * math.cos(phi),
        pt * math.sin(phi),
        pt * math.sinh(eta),
        math.sqrt((pt * math.cosh(eta)) ** 2 + mass**2),
        sign,
        calo,
        track,
    ]


def event(rows, weight=1.0, sigma=10.0):
    return dict(
        px=[r[0] for r in rows],
        py=[r[1] for r in rows],
        pz=[r[2] for r in rows],
        E=[r[3] for r in rows],
        particle_ID=[211 if r[6] else 22 for r in rows],
        status=[2 if r[4] < 0 else 0 for r in rows],
        event_weight=weight,
        cross_section=sigma,
    )


@pytest.fixture
def config():
    return analysis.load_config(Path(__file__).parents[1] / "config/atlas_dr12_5020.yaml")


def test_two_prong_analytic():
    result = native.reconstruct([row(150), row(80, 0.4)])
    assert len(result["jets"]) == 1
    pt, dr, scale, fraction = result["jets"][0]
    assert pt == pytest.approx(math.hypot(150 + 80 * math.cos(0.4), 80 * math.sin(0.4)))
    assert dr == pytest.approx(0.4)
    assert scale == pytest.approx(32.0)
    assert fraction == pytest.approx(230 / pt)


def test_small_jet_threshold_and_reclustering():
    # Both small jets must pass 35, even if the sum before that cut exceeds 200.
    assert not native.reconstruct([row(185), row(30, 0.5)])["jets"]
    # Direct R=1 clustering of all particles would accept the first event.
    assert native.reconstruct([row(185), row(40, 0.5)])["jets"]


def test_kt_tree_not_cambridge_aachen():
    rows = [row(160), row(55, 0.1), row(15, 0.4)]
    result = native.reconstruct(rows)["jets"][0]
    # kt merges 55 and 15 first; CA merges 160 and 55, then drops the 15.
    merged_phi = math.atan2(rows[1][1] + rows[2][1], rows[1][0] + rows[2][0])
    assert result[1] == pytest.approx(merged_phi)
    assert result[1] != pytest.approx(0.1)


def test_grooming_failure_and_no_duplicate_subtraction():
    assert not native.reconstruct([row(225)])["jets"]
    assert not native.reconstruct([row(220), row(10, 0.1)])["jets"]
    result = native.reconstruct([row(160), row(80, 0.4), row(10, 0, sign=-1)])["jets"][0]
    assert result[0] == pytest.approx(math.hypot(150 + 80 * math.cos(0.4), 80 * math.sin(0.4)))
    assert result[2] == pytest.approx(32.0)


def test_negative_composite_retains_sign():
    # The two nearby holes cluster first. Losing their sign turns subtraction into addition.
    result = native.reconstruct([row(400, 0.05), row(110), row(110)], 1e-9)
    assert result["cutflow"]["fiducial_jets"] == 1
    result = native.reconstruct([row(400, 0.05), row(110, sign=-1), row(110, sign=-1)])
    assert result["cutflow"]["fiducial_jets"] == 0
    assert not result["jets"]


def test_negative_branch_is_not_a_soft_drop_prong():
    result = native.reconstruct([row(150), row(80, 0.05), row(80, 0.9, sign=-1, calo=0)], groomer="signed_prongs")
    assert result["cutflow"]["negative_splits_skipped"] == 1
    assert result["jets"][0][1] == pytest.approx(0.05)


@pytest.mark.parametrize("groomer", ["signed_prongs", "softdrop_reference"])
def test_upstream_groomers_agree_for_positive_prongs(groomer):
    for rows in ([row(150), row(80, .4)], [row(160), row(55, .1), row(15, .4)], [row(225)]):
        expected = native.reconstruct(rows)["jets"]
        actual = native.reconstruct(rows, groomer=groomer)["jets"]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_heppy_formula_with_negative_marker_agrees():
    rows = [row(400, .05), row(110, sign=-1), row(110, sign=-1)]
    reference = native.reconstruct(rows, recombiner_mode="legacy_reference", legacy_marker=-1)
    assert reference == native.reconstruct(rows)
    # Explicitly demonstrate non-equivalence of the original default marker 0.
    lost_sign = native.reconstruct(rows, recombiner_mode="legacy_reference", legacy_marker=0)
    assert lost_sign["cutflow"]["fiducial_jets"] != reference["cutflow"]["fiducial_jets"]


def test_grooming_negative_prong_policy_is_not_claimed_equivalent():
    rows = [row(150), row(80, .05), row(80, .9, sign=-1, calo=0)]
    signed = native.reconstruct(rows, groomer="signed_prongs")["jets"][0]
    legacy = native.reconstruct(rows, groomer="lund")["jets"][0]
    assert signed[1] == pytest.approx(.05)
    assert legacy[1] != pytest.approx(signed[1])


def test_azimuth_rotation_and_input_order_invariance():
    base = [row(150, 0.1), row(80, 0.5), row(5, 0.3, sign=-1)]
    rotated = [row(5, 3.3, sign=-1), row(80, 3.5), row(150, 3.1)]
    np.testing.assert_allclose(
        native.reconstruct(base)["jets"], native.reconstruct(rotated)["jets"], atol=1e-12, rtol=1e-12
    )


def test_ghost_association_unique_and_pt_invariant():
    rows = [row(170), row(65, 0.2), row(170, 1.25), row(65, 1.45), row(8, 0.8, calo=0)]
    a = native.reconstruct(rows, 1e-12)["jets"]
    b = native.reconstruct(rows, 1e-7)["jets"]
    assert len(a) == len(b) == 2
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    # The extra track lies within R=1 of both axes but must be assigned only once.
    assert sum(j[0] * j[3] for j in a) == pytest.approx(478.0)


def test_track_pt_and_charged_fraction_cuts():
    # A 4 GeV particle contributes to the calorimetric jet, not the track tree.
    assert not native.reconstruct([row(220), row(4, 0.1)])["jets"]
    assert not native.reconstruct([row(200, track=0), row(20, 0.1), row(20, -0.1)])["jets"]
    assert native.reconstruct([row(160, track=0), row(40, 0.1), row(40, -0.1)])["jets"]


def test_rapidity_not_pseudorapidity():
    a, b = row(150, eta=0.1, mass=40), row(80, 0.4, eta=0.4, mass=30)
    result = native.reconstruct([a, b])["jets"][0]
    y = lambda r: 0.5 * math.log((r[3] + r[2]) / (r[3] - r[2]))
    assert result[1] == pytest.approx(math.hypot(y(a) - y(b), 0.4))
    assert result[1] != pytest.approx(0.5)


def test_fiducial_eta():
    assert not native.reconstruct([row(150, eta=1.4), row(80, 0.4, eta=1.4)])["jets"]


def test_particle_selection_and_unknown_status():
    e = event([row(150), row(80, 0.4), row(500), row(500)])
    e["particle_ID"][-2:] = [13, 12]
    rows = analysis.particle_rows(e, "pp")
    assert rows[-2][5:] == [0, 1]  # muon is a track, not calorimeter energy
    assert rows[-1][5:] == [0, 0]
    e["status"][-1] = -2
    assert len(analysis.particle_rows(e, "pp")) == 3
    e["status"][-1] = -1
    with pytest.raises(ValueError, match="status"):
        analysis.particle_rows(e, "pp")
    assert analysis.particle_properties(111) == (True, False)


def test_charged_selection_matches_legacy_hybrid():
    for pid in analysis.CHARGED_PIDS:
        assert analysis.particle_properties(pid)[1]
        assert analysis.particle_properties(-pid)[1]
    assert not analysis.particle_properties(411)[1]  # not in legacy charged list


@pytest.mark.parametrize("status", [0, 1, 2])
def test_pi0_kept_by_default_and_decay_preserves_sign(status):
    e = event([row(20, 0.3, mass=analysis.PI0_MASS, track=0)])
    e["particle_ID"] = [111]
    e["status"] = [status]
    kept = analysis.particle_rows(e, "PbPb")
    assert len(kept) == 1
    np.testing.assert_allclose(kept[0][:4], [e[k][0] for k in ("px", "py", "pz", "E")])
    qa = {}
    photons = np.asarray(analysis.particle_rows(e, "PbPb", "decay_gg", 10, qa))
    assert photons.shape == (2, 7)
    np.testing.assert_allclose(photons[:, :4].sum(axis=0), kept[0][:4], atol=1e-12)
    np.testing.assert_allclose(np.sum(photons[:, :3]**2, axis=1), photons[:, 3]**2, atol=1e-12)
    assert np.all(photons[:, 4] == (-1 if status == 2 else 1))
    assert np.all(photons[:, 5] == 1) and np.all(photons[:, 6] == 0)
    assert qa["pi0_decayed"] == 1


def test_pi0_decay_reproducible_and_pp_still_excludes_wake():
    e = event([row(20, mass=analysis.PI0_MASS), row(5, mass=analysis.PI0_MASS, sign=-1)])
    e["particle_ID"] = [111, 111]
    a = analysis.particle_rows(e, "pp", "decay_gg", 9)
    np.testing.assert_array_equal(a, analysis.particle_rows(e, "pp", "decay_gg", 9))
    assert len(a) == 2
    assert not np.allclose(a, analysis.particle_rows(e, "pp", "decay_gg", 10))


def test_pi0_float32_projection_is_reported_and_bad_input_rejected():
    p = np.array([1000, 200, 50], dtype=np.float32)
    energy = float(np.float32(np.sqrt(np.dot(p.astype(float), p) + analysis.PI0_MASS**2)))
    photons, delta = analysis.decay_pi0_to_photons(p, energy, np.random.default_rng(1))
    assert sum(e for _, e in photons) == pytest.approx(energy + delta)
    np.testing.assert_allclose(sum(p for p, _ in photons), p)
    with pytest.raises(ValueError, match="mass-shell"):
        analysis.decay_pi0_to_photons([1, 0, 0], 2, np.random.default_rng(1))


def test_pi0_mode_provenance_and_config_isolation(config):
    e = event([row(150), row(80, 0.4), row(2, mass=analysis.PI0_MASS, track=0)])
    e["particle_ID"][-1] = 111
    config["pi0_treatment"] = "decay_gg"
    _, info = analysis.analyze_events([e], config, "pp", None, "all")
    assert info["particle_qa"]["pi0_decayed"] == 1
    config["pi0_treatment"] = "keep"
    assert info["config"]["pi0_treatment"] == "decay_gg"


def test_pp_excludes_wake():
    e = event([row(150), row(80, 0.4), row(10, sign=-1)])
    assert len(analysis.particle_rows(e, "pp")) == 2
    assert len(analysis.particle_rows(e, "PbPb")) == 3


def test_all_event_normalization_and_event_level_errors(config):
    # Two jets in one event share an event weight, so variance is (2w)^2, not 2w^2.
    e = event([row(150), row(80, 0.4), row(150, 3), row(80, 3.4)], weight=2)
    arrays, info = analysis.analyze_events([e, event([], weight=3)], config, "pp", None, "all")
    assert info["weight_sum"] == 5
    assert arrays["dr12_atlas_sumw"].sum() == 4
    assert arrays["dr12_atlas_event_outer"].sum() == 16
    assert info["cutflow"]["events"] == 2


def test_explicit_centrality(config):
    with pytest.raises(ValueError, match="centrality"):
        analysis.analyze_events([], config, "PbPb", None, "all")
    e = event([])
    e["centrality"] = 35
    with pytest.raises(ValueError, match="contradicts"):
        analysis.analyze_events([e], config, "PbPb", [0, 10], "all", design_id=0)


def test_ratio_absolute_yields_and_errors():
    aa, caa, pp, cpp = np.array([4.0, 2.0]), np.diag([1.0, 1.0]), np.array([8.0, 10.0]), np.diag([2.0, 3.0])
    r, c = analysis.ratio(aa, caa, pp, cpp)
    np.testing.assert_allclose(r, [0.5, 0.2])
    np.testing.assert_allclose(np.diag(c), np.diag(caa) / pp**2 + aa**2 * np.diag(cpp) / pp**4)
    with pytest.raises(ValueError, match="MC bin"):
        analysis.ratio(np.zeros(2), caa, pp, cpp)


def test_ascii_parser_to_native_analysis(tmp_path, config):
    content = "# event 0\nweight 1 cross 10 X 0 Y 0 cross_err 0.1 pthat 250\n"
    for r in [row(150), row(80, 0.4)]:
        content += f"{r[0]} {r[1]} {r[2]} 0.13957 211 0\n"
    content += "end\n# event 1\nweight 3 cross 10 X 0 Y 0 cross_err 0.1 pthat 250\n1 0 0 0.13957 211 0\nend\n"
    path = tmp_path / "hybrid.out"
    path.write_text(content)
    arrays, info = analysis.analyze_events(analysis.read_events(path), config, "pp", None, "all")
    assert info["weight_sum"] == 4
    assert info["cutflow"]["events"] == 2
    assert arrays["dr12_atlas_sumw"].sum() == 1


def test_parquet_reader(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    e = event([row(150), row(80, 0.4)])
    path = tmp_path / "hybrid.parquet"
    pq.write_table(pa.Table.from_pylist([e]), path)
    assert list(analysis.read_events(path)) == [e]


def test_merge_and_export_in_existing_format(tmp_path, config):
    config = deepcopy(config)
    config["observables"] = {"dr12_atlas": {"column": 1, "edges": [0.0, 1.0]}}
    e = event([row(150), row(80, 0.4)])
    for name, system, events in (
        ("pp", "pp", [e, e]),
        ("aa1", "PbPb", [e, event([])]),
        ("aa2", "PbPb", [e, event([])]),
    ):
        arrays, info = analysis.analyze_events(
            events,
            config,
            system,
            [0, 10] if system == "PbPb" else None,
            "all",
            design_id=0 if system == "PbPb" else None,
        )
        info["sources"] = [name]
        analysis.save_sample(tmp_path / f"{name}.npz", arrays, info)
    analysis.merge_samples([tmp_path / "aa1.npz", tmp_path / "aa2.npz"], tmp_path / "aa.npz")
    arrays, info = analysis.load_sample(tmp_path / "aa.npz")
    assert info["weight_sum"] == 4
    data = tmp_path / "Data__5020__PbPb__inclusive_jet__dr12_atlas__R1.0_pt200-251_trk4_zcut0.15_beta0__0-10.dat"
    np.savetxt(data, [[0, 1, 0.5, 0.1]], header="Label xmin xmax y stat")
    design = tmp_path / "Design__Lres-E-loss.dat"
    design.write_text("# Design point indices (row index): 0\n0.5 0.5\n")
    manifest = dict(
        observable="dr12_atlas",
        pp=["pp.npz"],
        design_points={0: ["aa.npz"]},
        data_table=data.name,
        design_table=design.name,
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    output = tmp_path / "tables"
    analysis.export_tables(path, output)
    prediction = next((output / "Prediction").glob("*values.dat"))
    assert np.loadtxt(prediction) == pytest.approx(0.5)
    assert "design_point0" in prediction.read_text()
    assert (output / "Design" / design.name).read_bytes() == design.read_bytes()
    with pytest.raises(ValueError, match="Repeated"):
        analysis.spectrum([tmp_path / "aa1.npz", tmp_path / "aa2.npz"], "dr12_atlas", "PbPb", design_id=0)
    with pytest.raises(ValueError, match="design_id"):
        analysis.spectrum([tmp_path / "aa.npz"], "dr12_atlas", "PbPb", design_id=1)


def test_exact_bins_match_extracted_data(config):
    # Run when the user's separately audited archive extraction is present.
    if not os.environ.get("ATLAS_DR12_DATA_DIR"):
        pytest.skip("Set ATLAS_DR12_DATA_DIR to independently extracted experimental tables")
    folder = Path(os.environ["ATLAS_DR12_DATA_DIR"])
    for name, block in config["observables"].items():
        table = np.loadtxt(next(folder.glob(f"Data*__{name}__*__0-10.dat")))
        np.testing.assert_allclose(table[:, 0], block["edges"][:-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(table[:, 1], block["edges"][1:], rtol=0, atol=1e-12)


def test_config_rejects_mislabeled_observable(config):
    config["observables"]["dr12_atlas"]["column"] = 2
    with pytest.raises(ValueError, match="name/column"):
        analysis.validate_config(config)


def test_design_id_required_before_reading_events(config):
    with pytest.raises(ValueError, match="design_id"):
        analysis.analyze_events([], config, "PbPb", [0, 10], "all")


def test_ratio_rejects_nonfinite():
    with pytest.raises(ValueError, match="Nonfinite"):
        analysis.ratio(np.array([np.nan]), np.eye(1), np.ones(1), np.eye(1))


def test_native_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Nonfinite"):
        native.reconstruct([row(float("nan"))])
    with pytest.raises(ValueError, match="ghost_pt"):
        native.reconstruct([], ghost_pt=0)
