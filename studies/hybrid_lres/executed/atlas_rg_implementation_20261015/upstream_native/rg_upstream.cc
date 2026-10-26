// Diagnostic adapter: calls the unmodified heppy classes used by STAT.
#include <fastjet/ClusterSequence.hh>
#include <fastjet/tools/Recluster.hh>
#include "GroomerShop.hh"
#include "fjtools.cxx"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using fastjet::PseudoJet;

class GeometryRecombiner : public FJTools::NegativeEnergyRecombiner {
 public:
  GeometryRecombiner(int marker, double scale)
      : FJTools::NegativeEnergyRecombiner(marker), scale_(scale) {}
  mutable size_t interventions = 0;
  void recombine(const PseudoJet& a, const PseudoJet& b, PseudoJet& out) const override {
    FJTools::NegativeEnergyRecombiner::recombine(a, b, out);
    if (scale_ > 0 && (a.user_index() < 0) != (b.user_index() < 0)
        && out.E() == 0 && std::hypot(out.px(), out.py(), out.pz()) <= scale_
        && !std::isfinite(out.rap())) {
      const PseudoJet direction = a + b;
      if (!std::isfinite(direction.rap()))
        throw std::runtime_error("Undefined parent geometry for zero-energy cancellation");
      // Assign only the undefined angular cache; never alter signed momentum.
      out.set_cached_rap_phi(direction.rap(), direction.phi());
      ++interventions;
    }
  }
 private:
  double scale_;
};

py::dict reconstruct(const std::vector<std::array<double, 7>>& rows,
                     int negative_marker = -1, double ca_radius = .4,
                     double zero_energy_scale = 0., std::string grooming_policy = "upstream_lund",
                     bool require_all_constituents = false) {
  if (negative_marker != -1 && negative_marker != 0)
    throw std::invalid_argument("Use -1 (signed) or 0 (upstream default control)");
  if (!std::isfinite(zero_energy_scale) || zero_energy_scale < 0 || zero_energy_scale > .001)
    throw std::invalid_argument("Explicit zero-energy scale must be within [0,.001] GeV");
  if (!std::isfinite(ca_radius) || ca_radius <= 0 || ca_radius > 1000.)
    throw std::invalid_argument("Invalid C/A radius");
  if (grooming_policy != "upstream_lund" && grooming_policy != "signed_prongs")
    throw std::invalid_argument("Declare upstream_lund or signed_prongs");
  FJTools::NegativeEnergyRecombiner recombiner(negative_marker);
  GeometryRecombiner ca_recombiner(negative_marker, zero_energy_scale);
  fastjet::JetDefinition anti(fastjet::antikt_algorithm, .4);
  fastjet::JetDefinition ca(fastjet::cambridge_algorithm, ca_radius);
  anti.set_recombiner(&recombiner);
  ca.set_recombiner(&ca_recombiner);
  std::vector<PseudoJet> particles;
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    for (double value : r)
      if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite particle");
    if (r[3] <= 0 || (r[4] != 1 && r[4] != -1) || (r[5] != 0 && r[5] != 1))
      throw std::invalid_argument("Invalid particle energy, sign or calorimeter mask");
    if (std::hypot(r[0], r[1], r[2]) - r[3] > 1e-5 * std::max(1., r[3]))
      throw std::invalid_argument("Spacelike input beyond float32 tolerance");
    if (!r[5]) continue;
    PseudoJet p(r[0], r[1], r[2], r[3]);
    p.set_user_index((r[4] < 0 ? -1 : 1) * static_cast<int>(i + 1));
    particles.push_back(p);
  }
  fastjet::ClusterSequence sequence(particles, anti);
  std::vector<std::array<double, 4>> output;
  py::list details;
  std::map<std::string, int> counts{{"jets", 0}, {"negative_jets", 0},
      {"fiducial_jets", 0}, {"tagged", 0}, {"untagged", 0}};
  for (const auto& jet : fastjet::sorted_by_pt(sequence.inclusive_jets())) {
    ++counts["jets"];
    if (jet.user_index() < 0) { ++counts["negative_jets"]; continue; }
    if (jet.pt() <= 158 || std::abs(jet.rap()) >= 2.1) continue;
    ++counts["fiducial_jets"];
    // Keep the owner alive while inspecting splitting constituent histories.
    fastjet::contrib::GroomerShop shop(jet, ca);
    const auto* split = shop.soft_drop(0, .2, .4);
    bool tagged = split->z() > .2;
    bool selected_negative_prong = tagged &&
        (split->harder().user_index() < 0 || split->softer().user_index() < 0);
    double rg = tagged ? split->Delta() : 0., zg = tagged ? split->z() : 0.;
    // Audit Recluster's actual output, not just the selected SD prongs.
    fastjet::Recluster recluster(ca);
    const auto root = recluster(jet);
    const auto input = jet.constituents();
    const auto kept = root.constituents();
    std::vector<int> ids_in, ids_kept;
    for (const auto& p : input) ids_in.push_back(p.user_index());
    for (const auto& p : kept) ids_kept.push_back(p.user_index());
    std::sort(ids_in.begin(), ids_in.end());
    std::sort(ids_kept.begin(), ids_kept.end());
    if (require_all_constituents && ids_in != ids_kept)
      throw std::runtime_error("Upstream C/A reclustering lost jet constituents");
    if (grooming_policy == "signed_prongs") {
      // Retain the existing Hybrid background prescription, not Lund's unsigned walk.
      tagged = false; selected_negative_prong = false; rg = zg = 0.;
      PseudoJet current = root, a, b;
      while (current.has_parents(a, b)) {
        if (a.user_index() < 0 || b.user_index() < 0) {
          ++counts["negative_splits_skipped"];
          if (a.user_index() < 0 && b.user_index() < 0) break;
          current = a.user_index() < 0 ? b : a;
          continue;
        }
        const double sum = a.pt() + b.pt();
        if (sum > 0 && std::min(a.pt(), b.pt()) / sum > .2) {
          rg = a.delta_R(b); zg = std::min(a.pt(), b.pt()) / sum;
          tagged = true; break;
        }
        current = a.pt() >= b.pt() ? a : b;
      }
    }
    output.push_back({jet.pt(), rg, zg, tagged ? 1. : 0.});
    ++counts[tagged ? "tagged" : "untagged"];
    py::dict detail;
    detail["input_ids"] = ids_in;
    detail["reclustered_ids"] = ids_kept;
    detail["all_constituents_retained"] = ids_in == ids_kept;
    detail["selected_negative_prong"] = selected_negative_prong;
    detail["jet_four_vector"] = std::array<double,4>{jet.px(), jet.py(), jet.pz(), jet.E()};
    detail["reclustered_four_vector"] = std::array<double,4>{root.px(), root.py(), root.pz(), root.E()};
    details.append(detail);
  }
  py::dict result;
  result["jets"] = output;
  result["cutflow"] = counts;
  result["reclustering"] = details;
  result["geometry_interventions"] = ca_recombiner.interventions;
  return result;
}

PYBIND11_MODULE(_rg_upstream, m) {
  m.def("reconstruct", &reconstruct, py::arg("rows"),
        py::arg("negative_marker") = -1, py::arg("ca_radius") = .4,
        py::arg("zero_energy_scale") = 0., py::arg("grooming_policy") = "upstream_lund",
        py::arg("require_all_constituents") = false);
}
