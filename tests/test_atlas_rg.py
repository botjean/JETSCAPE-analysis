"""Real FastJet reconstruction and exact weighted-ratio/covariance fixtures."""

import json
import math
from copy import deepcopy

import numpy as np
import pytest
import yaml

pytest.importorskip("jetscape_analysis.analysis._atlas_rg", reason="Run scripts/build_atlas_rg.py")
from jetscape_analysis.analysis import _atlas_rg as native
from jetscape_analysis.analysis import atlas_rg as analysis

from tests.test_atlas_dr12 import event, row


@pytest.fixture
def config():
    # Synthetic binning ONLY: deliberately not claimed to be published ATLAS edges.
    return dict(
        sqrt_s=5020,
        centrality=[0, 10],
        particle_level="atlas_30ps",
        groomer="signed_prongs",
        rg_edges=[0, 0.05, 0.15, 0.4],
        binning_status="verified",
        binning_source="synthetic test fixture",
    )


def test_two_prong_absolute_angle_and_ungroomed_pt():
    a, b = row(150), row(50, 0.2, track=0)
    pt, rg, zg, tagged = native.reconstruct([a, b])["jets"][0]
    assert pt == pytest.approx(math.hypot(150 + 50 * math.cos(0.2), 50 * math.sin(0.2)))
    assert rg == pytest.approx(0.2)  # NOT .2/.4, and the neutral prong is retained.
    assert zg == pytest.approx(0.25) and tagged == 1


def test_ca_not_kt_tree():
    values = native.reconstruct([row(160), row(55, 0.1), row(15, 0.35)])["jets"][0]
    assert values[1] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "rows",
    [
        [row(160), row(55, 0.1), row(15, 0.35)],
        [row(150), row(50, 0.2)],
        [row(225)],
        [row(175), row(20, 0.1)],
    ],
)
def test_positive_particles_match_two_upstream_oracles(rows):
    expected = native.reconstruct(rows)["jets"]
    for groomer in ("lund_reference", "softdrop_reference"):
        np.testing.assert_allclose(native.reconstruct(rows, groomer)["jets"], expected, atol=1e-12)


def test_softdrop_strict_threshold_and_no_particle_pt_cut():
    assert native.reconstruct([row(160), row(40, 0.1)])["jets"][0][3] == 0
    assert native.reconstruct([row(160), row(40.001, 0.1)])["jets"][0][3] == 1
    # Many individually sub-4-GeV particles form a genuine second hard prong.
    rows = [row(150)] + [row(3, 0.2) for _ in range(20)]
    assert native.reconstruct(rows)["jets"][0][3] == 1


def test_failed_tag_not_first_positive_bin(config):
    counts, qa = analysis.count_event(event([row(175)]), config, "pp")
    assert counts.sum() == 0 and qa["pt0_untagged"] == 1
    assert native.reconstruct([row(175)])["jets"][0][1:] == [0.0, 0.0, 0.0]


def test_prepared_row_hook_preserves_event_counter(config):
    e = event([row(150), row(50, 0.1), row(175, 3)])
    expected, expected_qa = analysis.count_event(e, config, "pp")
    actual, actual_qa = analysis.count_rows(analysis.rows_for_event(e, "pp", config), config)
    np.testing.assert_array_equal(actual, expected)
    assert actual_qa == expected_qa


def test_signed_energy_no_duplicate_subtraction_and_negative_prong():
    rows = [row(150), row(70, 0.05), row(80, 0.35, sign=-1)]
    result = native.reconstruct(rows)
    assert not result["jets"]  # The signed ungroomed jet is below 158 GeV.
    rows[0] = row(250)
    result = native.reconstruct(rows)
    assert result["jets"][0][1] == pytest.approx(0.05)
    assert result["cutflow"]["negative_splits_skipped"] == 1
    with pytest.raises(ValueError, match="positive-particle-only"):
        native.reconstruct(rows, "softdrop_reference")


def test_negative_composites_never_turn_into_positive_energy():
    assert not native.reconstruct([row(350, 0.05), row(110, sign=-1), row(110, sign=-1)])["jets"]
    assert native.reconstruct([row(350, 0.05), row(110), row(110)])["jets"]
    assert native.reconstruct([row(200, sign=-1)])["cutflow"]["negative_jets"] == 1


def test_rapidity_cut_not_eta():
    # eta=2.2, but a massive jet's y is inside 2.1.
    assert native.reconstruct([row(200, eta=2.2, mass=200)])["jets"]
    assert not native.reconstruct([row(200, eta=2.2)])["jets"]


def test_angle_uses_rapidity_not_eta():
    a, b = row(150, eta=0.1, mass=40), row(70, 0.2, eta=0.3, mass=30)

    def y(r):
        return 0.5 * math.log((r[3] + r[2]) / (r[3] - r[2]))

    rg = native.reconstruct([a, b])["jets"][0][1]
    assert rg == pytest.approx(math.hypot(y(a) - y(b), 0.2))
    assert rg != pytest.approx(math.hypot(0.2, 0.2))


def test_rotation_and_permutation():
    a = [row(250, 0.1), row(70, 0.2), row(10, 0.35, sign=-1)]
    b = [row(10, 3.35, sign=-1), row(70, 3.2), row(250, 3.1)]
    np.testing.assert_allclose(native.reconstruct(a)["jets"], native.reconstruct(b)["jets"], atol=1e-12)


def test_measurement_inputs_reject_pi0_instead_of_losing_energy(config):
    e = event([row(200, track=0)])
    e["particle_ID"] = [111]
    with pytest.raises(ValueError, match="30 ps"):
        analysis.count_event(e, config, "pp")
    config["particle_level"] = "legacy_diagnostic"
    assert analysis.count_event(e, config, "pp")[1]["fiducial_jets"] == 1


def test_muon_neutrino_veto_and_pp_wake_policy(config):
    e = event([row(150), row(70, 0.2), row(500), row(500), row(40, sign=-1)])
    e["particle_ID"][2:4] = [13, 12]
    pp, _ = analysis.count_event(e, config, "pp")
    assert pp.reshape(3, -1)[1].sum() == 1  # A 219-GeV jet, neither lepton added.
    aa, _ = analysis.count_event(e, config, "PbPb")
    assert aa.reshape(3, -1)[0].sum() == 1  # The hole lowers it to the first pT slice.


def test_pt_slices_are_ungroomed(config):
    # Wide soft branch fails SD, but must not move the original 206-GeV jet to pt0.
    counts, _ = analysis.count_event(event([row(185), row(25, 0.3)]), config, "pp")
    assert counts.sum() == 0
    result = native.reconstruct([row(185), row(25, 0.3)])["jets"][0]
    assert 200 < result[0] < 315 and not result[3]


def test_moments_include_all_event_denominator_and_event_bin_correlations(config):
    e = event([row(150), row(50, 0.1), row(150, 3), row(50, 3.1)], weight=2, sigma=10)
    arrays, info = analysis.analyze([[e, event([], weight=3)]], [None], config, "pp", input_provenance="fixture")
    assert arrays["total"][-1] == 5
    means, covariance = analysis.spectrum(arrays, info)
    assert means.sum() == pytest.approx(8)
    # y=10*4/5; derivatives evaluated on the two weighted event vectors.
    assert covariance.sum() == pytest.approx((10 * (4 / 5 - 4 * 2 / 25)) ** 2 + (10 * (-4 * 3 / 25)) ** 2)


def test_paired_designs_have_measured_cross_covariance(config):
    e = event([row(150), row(50, 0.1)])
    for v in (e,):
        v["pt_hat"] = 80
        v["event_ID"] = 0
    empty = event([])
    empty["pt_hat"] = 70
    empty["event_ID"] = 1
    arrays, info = analysis.analyze(
        [[e, empty], [deepcopy(e), deepcopy(empty)]],
        [2, 0],
        config,
        "PbPb",
        [0, 10],
        "explicit-shared-events",
        "fixture",
    )
    means, covariance = analysis.spectrum(arrays, info)
    np.testing.assert_array_equal(means[0], means[1])
    np.testing.assert_allclose(covariance[0, :, 1, :], covariance[0, :, 0, :])
    assert covariance[0, :, 1, :].sum() > 0
    with pytest.raises(ValueError, match="pairing"):
        analysis.analyze([[e], [e]], [0, 1], config, "PbPb", [0, 10], input_provenance="fixture")
    bad = deepcopy(e)
    bad["pt_hat"] = 81
    with pytest.raises(ValueError, match="alignment"):
        analysis.analyze([[e], [bad]], [0, 1], config, "PbPb", [0, 10], "cohort", "fixture")


def test_full_ratio_covariance_independent_jacobian_oracle():
    rng = np.random.default_rng(21)
    aa, pp = np.array([[4.0, 8.0], [5.0, 9.0]]), np.array([10.0, 12.0])
    a, b = rng.normal(size=(4, 6)), rng.normal(size=(2, 5))
    caa, cpp = a @ a.T, b @ b.T
    mean, cov = analysis.ratio_bank(aa, caa.reshape(2, 2, 2, 2), pp, cpp)
    jac = np.zeros((4, 6))
    for i in range(2):
        for j in range(2):
            jac[2 * i + j, 2 * i + j] = 1 / pp[j]
            jac[2 * i + j, 4 + j] = -aa[i, j] / pp[j] ** 2
    full = np.zeros((6, 6))
    full[:4, :4] = caa
    full[4:, 4:] = cpp
    np.testing.assert_allclose(cov.reshape(4, 4), jac @ full @ jac.T)
    np.testing.assert_allclose(mean, aa / pp)
    assert cov[0, 0, 1, 0] != 0
    with pytest.raises(ValueError, match="nonpositive"):
        analysis.ratio_bank(aa, caa.reshape(2, 2, 2, 2), [0, 12], cpp)


def test_same_weights_and_pthat_do_not_allow_swapped_event_ids(config):
    first = event([row(150), row(50, 0.1)])
    second = event([])
    first.update(event_ID=0, pt_hat=80)
    second.update(event_ID=1, pt_hat=80)
    with pytest.raises(ValueError, match="identities"):
        analysis.analyze([[first, second], [second, first]], [0, 1], config, "PbPb", [0, 10], "cohort", "fixture")
    missing = deepcopy(first)
    del missing["event_ID"]
    with pytest.raises(ValueError, match="identities"):
        analysis.analyze([[missing], [missing]], [0, 1], config, "PbPb", [0, 10], "cohort", "fixture")


def test_invalid_input_four_vectors_and_nonfinite_centrality(config):
    a, b = row(150), row(80, 0.1)
    a[3], b[3] = 75, 40
    with pytest.raises(ValueError, match="Spacelike"):
        native.reconstruct([a, b])
    bad = event([])
    bad["centrality"] = float("nan")
    with pytest.raises(ValueError, match="centrality"):
        analysis.analyze([[bad]], [0], config, "PbPb", [0, 10], input_provenance="fixture")


def test_pt_boundaries_and_strict_positive_rg(config, monkeypatch):
    jets = [[158, 0.1, 0.25, 1], [200, 0.1, 0.25, 1], [315, 0.1, 0.25, 1], [501, 0.1, 0.25, 1], [175, 0, 0.25, 1]]
    monkeypatch.setattr(native, "reconstruct", lambda *args: dict(jets=jets, cutflow={}))
    counts, qa = analysis.count_event(event([]), config, "pp")
    np.testing.assert_array_equal(counts.reshape(3, -1).sum(axis=1), [0, 1, 1])
    assert qa["pt0_underflow"] == 1


def test_malformed_moment_shape_and_normalization_rejected(config):
    arrays, info = analysis.analyze([[event([])]], [None], config, "pp", input_provenance="fixture")
    bad = deepcopy(arrays)
    bad["jet_sumw2"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="contract"):
        analysis.spectrum(bad, info)
    for key, value in (
        ("cross_sections", []),
        ("normalization", "minimum_bias"),
        ("covariance", "independent_per_jet"),
    ):
        corrupted = dict(info, **{key: value})
        with pytest.raises(ValueError, match="contract"):
            analysis.spectrum(arrays, corrupted)


def make_sample(path, config, ids, system, sigma, source):
    # Every synthetic feature occurs once, followed by one empty event.
    n, f = len(ids), 3 * (len(config["rg_edges"]) - 1)
    vectors = []
    for k in range(f):
        vector = np.zeros((n, f + 1))
        vector[:, k] = 1
        vector[:, -1] = 1
        vectors.append(vector.ravel())
    vector = np.zeros((n, f + 1))
    vector[:, -1] = 1
    vectors.append(vector.ravel())
    matrix = np.array(vectors)
    arrays = dict(total=matrix.sum(axis=0), outer=matrix.T @ matrix, jet_sumw2=np.ones((n, f)))
    info = dict(
        version=analysis.VERSION,
        config=config,
        system=system,
        design_ids=ids,
        centrality=[0, 10] if system == "PbPb" else None,
        input_provenance="fixture",
        cross_sections=[sigma] * n if np.ndim(sigma) == 0 else list(sigma),
        source_sha256=[source],
        pairing_id=source,
        reconstruction_sha256=analysis.reconstruction_hashes(),
        normalization="per_binary_hard_process",
        covariance="joint_event_poisson_with_denominators",
    )
    analysis.save(path, arrays, info)


def test_export_roundtrip_absolute_raa_order_edges_and_covariance(tmp_path, config):
    make_sample(tmp_path / "pp.npz", config, [None], "pp", 10.0, "pp-source")
    make_sample(tmp_path / "aa.npz", config, [2, 0], "PbPb", [5.0, 7.0], "aa-source")
    design = tmp_path / "Design__synthetic.dat"
    design.write_text("# Version 1\n# Design point indices: 0 2\n.1 .2\n.3 .4\n")
    paths = []
    for i in range(3):
        path = tmp_path / f"Data__5020__PbPb__inclusive_jet__rg_atlas__R0.4_pt{i}__0-10.dat"
        values = np.tile([0.5, 0.01, 0.01, 0.02, 0.02, 0.0013, 0.0013, 0.008, 0.008], (3, 1))
        np.savetxt(
            path,
            np.c_[config["rg_edges"][:-1], config["rg_edges"][1:], values],
            header="Version 1.0\nSystem PbPb5020\nCentrality 0 10\n"
            "Label xmin xmax y stat,low stat,high sys,low sys,high "
            "sys,taa,low sys,taa,high sys,luminosity,low sys,luminosity,high",
        )
        paths.append(path.name)
    manifest = dict(
        pp="pp.npz", aa_components=[dict(sample="aa.npz", weight=1.0)], design_table=design.name, data_tables=paths
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    output = tmp_path / "export"
    analysis.export_tables(path, output)
    predictions = sorted((output / "Prediction").glob("*__values.dat"))
    assert len(predictions) == 3
    for table in predictions:
        # ID0 has 0.7, ID2 has 0.5, despite the opposite extraction order.
        np.testing.assert_allclose(np.loadtxt(table), np.tile([0.7, 0.5], (3, 1)))
        assert table.read_text().splitlines()[3] == "# design_point0 design_point2"
    with np.load(output / "QA/rg_simulation_covariance.npz") as data:
        assert data["covariance"].shape == (2, 9, 2, 9)
        np.testing.assert_allclose(data["central_values"], np.array([[0.7] * 9, [0.5] * 9]))
        assert data["covariance"][0, 0, 0, 0] / data["covariance"][1, 0, 1, 0] == pytest.approx(1.96)
    pp_arrays, pp_info = analysis.load(tmp_path / "pp.npz")
    pp_info["config"]["binning_status"] = "unverified"
    analysis.save(tmp_path / "unverified.npz", pp_arrays, pp_info)
    manifest["pp"] = "unverified.npz"
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="verified"):
        analysis.export_tables(path, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
    manifest["pp"] = "pp.npz"
    aa_arrays, aa_info = analysis.load(tmp_path / "aa.npz")
    for index, change in enumerate(
        (
            dict(pairing_id=None),
            dict(source_sha256=[]),
            dict(normalization="minimum_bias"),
            dict(reconstruction_sha256={"wrong": "build"}),
            dict(pairing_id="pp-source"),
        )
    ):
        damaged = dict(aa_info, **change)
        sample = tmp_path / f"bad{index}.npz"
        analysis.save(sample, aa_arrays, damaged)
        manifest["aa_components"] = [dict(sample=sample.name, weight=1.0)]
        path.write_text(yaml.safe_dump(manifest))
        with pytest.raises(ValueError):
            analysis.export_tables(path, tmp_path / f"bad-export{index}")
        assert not (tmp_path / f"bad-export{index}").exists()
    manifest["aa_components"] = [dict(sample="aa.npz", weight=1.0)]
    path.write_text(yaml.safe_dump(manifest))
    design.write_text("# Version 1\n# Design point indices: 0 2\nnan .2\n.3 .4\n")
    with pytest.raises(ValueError, match="Design"):
        analysis.export_tables(path, tmp_path / "nan-design")
    assert not (tmp_path / "nan-design").exists()


def test_roundoff_diagonal_never_serializes_nan(tmp_path, config, monkeypatch):
    test_export_roundtrip_absolute_raa_order_edges_and_covariance(tmp_path, config)
    design = tmp_path / "Design__synthetic.dat"
    design.write_bytes((tmp_path / "export/Design" / design.name).read_bytes())
    original = analysis.ratio_bank

    def inject_roundoff(*args):
        means, covariance = original(*args)
        flat = covariance.reshape(means.size, means.size)
        flat[0, :] = 0
        flat[:, 0] = 0
        flat[0, 0] = -1e-34
        return means, covariance

    monkeypatch.setattr(analysis, "ratio_bank", inject_roundoff)
    output = tmp_path / "roundoff"
    analysis.export_tables(tmp_path / "manifest.yaml", output)
    for table in (output / "Prediction").glob("*__errors.dat"):
        assert np.isfinite(np.loadtxt(table)).all()
    report = json.loads((output / "QA/manifest.json").read_text())
    assert len(report["roundoff_diagonal_clips"]) == 1
    with np.load(output / "QA/rg_simulation_covariance.npz") as data:
        assert np.min(np.diag(data["covariance"].reshape(18, 18))) == 0


def test_shared_centrality_cohort_is_not_assumed_independent(tmp_path, config):
    test_export_roundtrip_absolute_raa_order_edges_and_covariance(tmp_path, config)
    arrays, info = analysis.load(tmp_path / "aa.npz")
    for name, centrality in (("first", [0, 5]), ("second", [5, 10])):
        copied = dict(info, centrality=centrality, source_sha256=[name])
        analysis.save(tmp_path / f"{name}.npz", arrays, copied)
    path = tmp_path / "manifest.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["aa_components"] = [dict(sample="first.npz", weight=0.5), dict(sample="second.npz", weight=0.5)]
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="cross-centrality cohort"):
        analysis.export_tables(path, tmp_path / "not-independent")
    assert not (tmp_path / "not-independent").exists()


@pytest.mark.parametrize(
    "change",
    [dict(particle_level="legacy_diagnostic"), dict(binning_status="inferred"), dict(groomer="lund_reference")],
)
def test_production_diagnostic_gates(config, change):
    config.update(change)
    with pytest.raises(ValueError, match="Production"):
        analysis.validate_config(config, production=True)
