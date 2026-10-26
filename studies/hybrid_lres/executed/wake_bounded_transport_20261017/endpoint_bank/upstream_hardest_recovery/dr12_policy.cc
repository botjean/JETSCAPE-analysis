// Particle-level reconstruction for arXiv:2504.04805, sections 4--5.
// This deliberately declusters the kt tree, not a Cambridge/Aachen tree.
#include <fastjet/ClusterSequence.hh>
#include <fastjet/tools/Recluster.hh>
#include <fastjet/contrib/LundGenerator.hh>
#include <fastjet/contrib/SoftDrop.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using fastjet::PseudoJet;

// The sign convention is that of heppy's fjext::NegativeEnergyRecombiner.
// Unlike its default ui=0, -1 retains the sign of negative-energy composites.
class SignedRecombiner : public fastjet::JetDefinition::Recombiner {
 public:
  std::string description() const override { return "signed E-scheme, negative marker -1"; }
  void recombine(const PseudoJet &a, const PseudoJet &b, PseudoJet &out) const override {
    const int sa = a.user_index() < 0 ? -1 : 1;
    const int sb = b.user_index() < 0 ? -1 : 1;
    const double e = sa * a.E() + sb * b.E();
    const int sign = e < 0 ? -1 : 1;
    out.reset(sign * (sa*a.px() + sb*b.px()), sign * (sa*a.py() + sb*b.py()),
              sign * (sa*a.pz() + sb*b.pz()), sign * e);
    out.set_user_index(sign < 0 ? -1 : 0);
  }
};

// Independent reference matching heppy fjext/fjtools.cxx (Tachibana adaptation).
// Diagnostic only: marker 0 reproduces its default, -1 preserves composite signs.
class LegacyRecombinerReference : public fastjet::JetDefinition::Recombiner {
 public:
  explicit LegacyRecombinerReference(int marker) : marker_(marker) {}
  std::string description() const override { return "legacy heppy recombination reference"; }
  void recombine(const PseudoJet &a, const PseudoJet &b, PseudoJet &out) const override {
    int ca = a.user_index() < 0 ? -1 : 1;
    int cb = b.user_index() < 0 ? -1 : 1;
    out = ca*a + cb*b;
    if (out.E() < 0) {
      out.set_user_index(marker_);
      out.reset_momentum(-out.px(), -out.py(), -out.pz(), -out.E());
    } else {
      out.set_user_index(0);
    }
  }
 private:
  int marker_;
};

// Each input row: px, py, pz, E, sign (+1/-1), calorimeter-input, track-input.
py::dict reconstruct(const std::vector<std::array<double, 7>> &rows,
                     double ghost_pt = 1e-9,
                     std::string recombiner_mode = "signed",
                     std::string groomer = "lund", int legacy_marker = -1) {
  if (!std::isfinite(ghost_pt) || ghost_pt <= 0 || ghost_pt > 1e-6)
    throw std::invalid_argument("ghost_pt must be positive and <= 1e-6 GeV");
  if (recombiner_mode != "signed" && recombiner_mode != "legacy_reference")
    throw std::invalid_argument("Unknown recombiner mode");
  if (groomer != "signed_prongs" && groomer != "lund" && groomer != "softdrop_reference")
    throw std::invalid_argument("Unknown groomer mode");
  if (legacy_marker != -1 && legacy_marker != 0)
    throw std::invalid_argument("Reference marker must be -1 or 0");
  SignedRecombiner signed_recombiner;
  LegacyRecombinerReference legacy_recombiner(legacy_marker);
  const fastjet::JetDefinition::Recombiner *recombiner = recombiner_mode == "signed"
      ? static_cast<const fastjet::JetDefinition::Recombiner *>(&signed_recombiner)
      : static_cast<const fastjet::JetDefinition::Recombiner *>(&legacy_recombiner);
  std::vector<PseudoJet> particles, tracks;
  std::map<std::string, int> counts{{"small_jets", 0}, {"negative_small_jets", 0},
    {"large_jets", 0}, {"fiducial_jets", 0}, {"charged_fraction_pass", 0},
    {"grooming_failed", 0}, {"negative_splits_skipped", 0}, {"accepted_jets", 0}};
  for (const auto &r : rows) {
    for (double x : r) if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite particle");
    if (r[3] < 0 || (r[4] != 1 && r[4] != -1))
      throw std::invalid_argument("Inputs need positive energy and an explicit +/-1 sign");
    PseudoJet p(r[0], r[1], r[2], r[3]);
    p.set_user_index(r[4] < 0 ? -1 : 0);
    if (r[5] != 0) particles.push_back(p);
    if (r[6] != 0 && p.pt() > 4 && std::abs(p.eta()) < 2.5) tracks.push_back(p);
  }
  fastjet::JetDefinition small_def(fastjet::antikt_algorithm, 0.2);
  small_def.set_recombiner(recombiner);
  fastjet::ClusterSequence small_cs(particles, small_def);
  std::vector<PseudoJet> large_inputs;
  for (const auto &p : small_cs.inclusive_jets()) {
    if (p.user_index() < 0) { ++counts["negative_small_jets"]; continue; }
    if (p.pt() > 35 && std::abs(p.eta()) < 3) {
      // Strip the old clustering structure; only small-jet four-vectors enter.
      PseudoJet q(p.px(), p.py(), p.pz(), p.E());
      q.set_user_index(0);
      large_inputs.push_back(q);
      ++counts["small_jets"];
    }
  }
  for (size_t i = 0; i < tracks.size(); ++i) {
    PseudoJet ghost = tracks[i] * (ghost_pt / tracks[i].pt());
    ghost.set_user_index(static_cast<int>(i) + 1);
    large_inputs.push_back(ghost);
  }
  fastjet::JetDefinition large_def(fastjet::antikt_algorithm, 1.0);
  fastjet::ClusterSequence large_cs(large_inputs, large_def);
  // pt, Delta R12, sqrt(d12), signed charged scalar-pT fraction.
  std::vector<std::array<double, 4>> output;
  std::vector<std::array<double, 4>> unresolved;
  py::list remnant_audit;
  for (const auto &large : fastjet::sorted_by_pt(large_cs.inclusive_jets())) {
    PseudoJet jet(0, 0, 0, 0);
    std::vector<PseudoJet> associated;
    double charged_pt = 0;
    int n_small = 0;
    for (const auto &c : large.constituents()) {
      if (c.user_index() == 0) { jet += c; ++n_small; }
      else {
        const auto &track = tracks.at(c.user_index() - 1);
        associated.push_back(track);
        charged_pt += (track.user_index() < 0 ? -1 : 1) * track.pt();
      }
    }
    if (n_small == 0) continue;
    ++counts["large_jets"];
    // Exclude ghost momenta even at floating-point threshold boundaries.
    if (jet.pt() <= 200 || std::abs(jet.eta()) >= 1.3) continue;
    ++counts["fiducial_jets"];
    const double fraction = charged_pt / jet.pt();
    if (fraction <= 0.2) continue;
    ++counts["charged_fraction_pass"];
    fastjet::JetDefinition kt_def(fastjet::kt_algorithm, 2.5);
    kt_def.set_recombiner(recombiner);
    fastjet::ClusterSequence kt_cs(associated, kt_def);
    auto kt_jets = kt_cs.inclusive_jets();
    if (kt_jets.empty()) {
      int n_negative = 0;
      for (const auto &q : kt_jets) if(q.user_index()<0) ++n_negative;
      unresolved.push_back({jet.pt(), fraction, static_cast<double>(kt_jets.size()), static_cast<double>(n_negative)});
      continue;
    }
    // Use FastJet's own selection on this existing sequence, without reclustering.
    fastjet::Recluster selector(kt_def, false, fastjet::Recluster::keep_only_hardest);
    PseudoJet current = selector.generate_output_jet(kt_jets, false), a, b;
    if (kt_jets.size() > 1) {
      py::dict audit;
      audit["jet_pt"] = jet.pt();
      audit["fraction"] = fraction;
      audit["selected_root_pt"] = current.pt();
      audit["selected_root_sign"] = current.user_index() < 0 ? -1 : 1;
      audit["selected_root_p4"] = std::array<double, 4>{current.px(), current.py(), current.pz(), current.E()};
      audit["selected_root_m2"] = current.m2();
      py::list remnants;
      int equal_pt_roots = 0;
      for (const auto &q : kt_jets) {
        if (q.pt2() == current.pt2()) ++equal_pt_roots;
        if (q.cluster_hist_index() == current.cluster_hist_index()) continue;
        remnants.append(std::array<double, 5>{q.px(), q.py(), q.pz(), q.E(),
                                               q.user_index() < 0 ? -1. : 1.});
      }
      audit["equal_pt_roots"] = equal_pt_roots;
      audit["remnants"] = remnants;
      remnant_audit.append(audit);
    }
    bool accepted = false;
    if (groomer == "lund") {
      // Same first-passing-Lund-splitting rule as heppy GroomerShop::soft_drop.
      fastjet::contrib::LundGenerator lund(kt_def);
      for (const auto &split : lund.result(current)) {
        if (split.z() > 0.15) {
          output.push_back({jet.pt(), split.Delta(), split.kt(), fraction});
          accepted = true;
          break;
        }
      }
    } else if (groomer == "softdrop_reference") {
      fastjet::contrib::SoftDrop sd(0, 0.15, 2.5);
      sd.set_reclustering(false);  // The measurement requires the existing kt tree.
      sd.set_tagging_mode();
      PseudoJet groomed = sd(current);
      if (groomed != 0 && groomed.has_parents(a, b)) {
        const double dr = a.delta_R(b);
        output.push_back({jet.pt(), dr, std::min(a.pt(), b.pt())*dr, fraction});
        accepted = true;
      }
    }
    if (groomer != "signed_prongs") {
      if (accepted) ++counts["accepted_jets"];
      else ++counts["grooming_failed"];
      continue;
    }
    while (current.has_parents(a, b)) {
      // A negative wake branch is not a physical Soft-Drop prong.
      if (a.user_index() < 0 || b.user_index() < 0) {
        ++counts["negative_splits_skipped"];
        if (a.user_index() < 0 && b.user_index() < 0) break;
        current = a.user_index() < 0 ? b : a;
        continue;
      }
      const double denominator = a.pt() + b.pt();
      if (denominator > 0 && std::min(a.pt(), b.pt()) / denominator > 0.15) {
        const double dr = a.delta_R(b);
        output.push_back({jet.pt(), dr, std::min(a.pt(), b.pt()) * dr, fraction});
        ++counts["accepted_jets"];
        accepted = true;
        break;
      }
      current = a.pt() >= b.pt() ? a : b;
    }
    if (!accepted) ++counts["grooming_failed"];
  }
  py::dict result;
  result["jets"] = output;
  result["cutflow"] = counts;
  result["unresolved"] = unresolved;
  result["remnant_audit"] = remnant_audit;
  return result;
}

PYBIND11_MODULE(_wake_dr12_diagnostic, m) {
  m.doc() = "Native FastJet ATLAS charged kt-splitting reconstruction";
  m.def("reconstruct", &reconstruct, py::arg("particles"), py::arg("ghost_pt") = 1e-9,
        py::arg("recombiner_mode") = "signed", py::arg("groomer") = "lund",
        py::arg("legacy_marker") = -1);
}
