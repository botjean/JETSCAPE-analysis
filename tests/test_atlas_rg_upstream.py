"""The opt-in backend uses upstream libraries, with explicit Hybrid policies."""
import math
from copy import deepcopy

import numpy as np
import pytest

native = pytest.importorskip("jetscape_analysis.analysis._rg_upstream")
from jetscape_analysis.analysis import atlas_rg as analysis
from tests.test_atlas_dr12 import event, row


def reconstruct(rows, scale=.0001):
    return native.reconstruct(rows, -1, 1000., scale, "signed_prongs", True)


@pytest.fixture
def config():
    return dict(sqrt_s=5020, centrality=[0,10], particle_level="atlas_30ps",
                groomer="upstream_signed_prongs", negative_marker=-1,
                reclustering_radius=1000, require_all_constituents=True,
                zero_energy_geometry_scale_GeV=.0001, rg_edges=[0,.05,.15,.4],
                binning_status="verified", binning_source="SYNTHETIC TEST ONLY")


def test_full_neutral_jet_and_absolute_angle():
    result = reconstruct([row(150), row(50,.2,track=0)])
    assert result["jets"][0][1:] == pytest.approx([.2,.25,1])
    assert result["reclustering"][0]["all_constituents_retained"]
    assert result["geometry_interventions"] == 0


def test_actual_upstream_default_marker_can_lose_a_negative_composite():
    rows = [row(350,.05), row(110,sign=-1), row(110,sign=-1)]
    assert not reconstruct(rows)["jets"]
    assert native.reconstruct(rows, 0, 1000.)["jets"]


def test_preserve_signed_prong_policy_instead_of_silently_adopting_lund():
    rows = [row(250), row(70,.05), row(80,.35,sign=-1)]
    assert reconstruct(rows)["jets"][0][1] == pytest.approx(.05)
    assert native.reconstruct(rows, -1, 1000.)["jets"][0][1] > .3


def test_strict_softdrop_threshold_and_failed_tag():
    assert reconstruct([row(160),row(40,.1)])["jets"][0][1:] == [0.,0.,0.]
    assert reconstruct([row(160),row(40.001,.1)])["jets"][0][3] == 1
    assert reconstruct([row(200)])["jets"][0][1:] == [0.,0.,0.]


def test_rapidity_acceptance_and_ungroomed_threshold():
    assert reconstruct([row(200,eta=2.2,mass=200)])["jets"]
    assert not reconstruct([row(200,eta=2.2)])["jets"]
    assert not reconstruct([row(158)])["jets"]


def test_large_spacelike_cancellation_is_not_repaired_or_dropped():
    mass = 15*math.sqrt(math.cosh(.25)**2-math.cosh(.2)**2)
    rows = [row(200),row(15,.1,eta=.2,mass=mass),row(15,.1,eta=.25,sign=-1)]
    for scale in (0., .00002, .0001, .001):
        with pytest.raises(RuntimeError, match="lost jet constituents"):
            reconstruct(rows, scale)


def test_rotation_and_permutation_without_numerical_intervention():
    a = [row(250,.1),row(70,.2),row(10,.35,sign=-1)]
    b = [row(10,3.35,sign=-1),row(70,3.2),row(250,3.1)]
    np.testing.assert_allclose(reconstruct(a)["jets"],reconstruct(b)["jets"],atol=1e-12)


def test_existing_event_counter_and_moment_propagation_use_the_new_backend(config):
    e = event([row(150),row(50,.1)],weight=2,sigma=10)
    empty = event([],weight=3,sigma=10)
    arrays, info = analysis.analyze([[e,empty]],[None],config,"pp",input_provenance="fixture")
    means, covariance = analysis.spectrum(arrays,info)
    oldconfig = dict(config,groomer="signed_prongs")
    old, old_info = analysis.analyze([[e,empty]],[None],oldconfig,"pp",input_provenance="fixture")
    old_means, old_covariance = analysis.spectrum(old,old_info)
    np.testing.assert_array_equal(means,old_means)
    np.testing.assert_array_equal(covariance,old_covariance)
    assert any("_rg_upstream" in name for name in info["reconstruction_sha256"])
    assert not any("_rg_upstream" in name for name in old_info["reconstruction_sha256"])


@pytest.mark.parametrize("key,value",[("negative_marker",0),("reclustering_radius",.4),
    ("require_all_constituents",False),("zero_energy_geometry_scale_GeV",float("nan")),
    ("zero_energy_geometry_scale_GeV",True),("zero_energy_geometry_scale_GeV",.01)])
def test_backend_does_not_silently_use_unsafe_settings(config,key,value):
    config[key] = value
    with pytest.raises(ValueError,match="explicit signed/preserving"):
        analysis.validate_config(config)


def test_paired_design_covariance_remains_joint(config):
    e = event([row(150),row(50,.1)])
    e.update(event_ID=0,pt_hat=80)
    empty = event([])
    empty.update(event_ID=1,pt_hat=70)
    arrays,info = analysis.analyze([[e,empty],[deepcopy(e),deepcopy(empty)]],
        [0,21],config,"PbPb",[0,10],"test-cohort","fixture")
    means,covariance = analysis.spectrum(arrays,info)
    np.testing.assert_array_equal(means[0],means[1])
    np.testing.assert_array_equal(covariance[0,:,0,:],covariance[0,:,1,:])


def test_runtime_fastjet_dependencies_are_checked_and_fingerprinted(config,monkeypatch):
    hashes = analysis.reconstruction_hashes(config)
    assert "libfastjet.so" in hashes and "libfastjettools.so" in hashes
    monkeypatch.setattr(analysis,"_upstream_runtime_libraries",lambda:{})
    with pytest.raises(ValueError,match="Loaded FastJet libraries differ"):
        analysis.reconstruction_hashes(config)
