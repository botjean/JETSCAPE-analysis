// Particle-level full-jet Soft-Drop radius, arXiv:2211.11470, section 4.
#include <fastjet/ClusterSequence.hh>
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

namespace {
class SignedRecombiner : public fastjet::JetDefinition::Recombiner {
 public:
  std::string description() const override { return "signed E-scheme, negative marker -1"; }
  void recombine(const PseudoJet &a, const PseudoJet &b, PseudoJet &out) const override {
    const int sa = a.user_index() < 0 ? -1 : 1;
    const int sb = b.user_index() < 0 ? -1 : 1;
    const double energy = sa*a.E() + sb*b.E();
    const int sign = energy < 0 ? -1 : 1;
    out.reset(sign*(sa*a.px() + sb*b.px()), sign*(sa*a.py() + sb*b.py()),
              sign*(sa*a.pz() + sb*b.pz()), sign*energy);
    out.set_user_index(sign < 0 ? -1 : 0);
  }
};
}  // namespace

// Rows use the existing dR12 adapter: px,py,pz,E,sign,calorimeter-input,track-input.
py::dict reconstruct_rg(const std::vector<std::array<double, 7>> &rows,
                        std::string groomer = "signed_prongs") {
  if (groomer != "signed_prongs" && groomer != "lund_reference"
      && groomer != "softdrop_reference")
    throw std::invalid_argument("Unknown groomer");
  SignedRecombiner recombiner;
  fastjet::JetDefinition antikt(fastjet::antikt_algorithm, 0.4);
  antikt.set_recombiner(&recombiner);
  fastjet::JetDefinition ca(fastjet::cambridge_algorithm,
                          fastjet::JetDefinition::max_allowable_R);
  ca.set_recombiner(&recombiner);
  std::vector<PseudoJet> particles;
  bool has_holes = false;
  for (const auto &r : rows) {
    for (double x : r) if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite particle");
    if (r[3] <= 0 || (r[4] != 1 && r[4] != -1)
        || (r[5] != 0 && r[5] != 1))
      throw std::invalid_argument("Invalid particle energy, sign or calorimeter mask");
    const double momentum = std::hypot(r[0], r[1], r[2]);
    if (momentum-r[3] > 1e-5*std::max(1.0, r[3]))
      throw std::invalid_argument("Spacelike input particle exceeds float32 rounding tolerance");
    if (!r[5]) continue;
    PseudoJet p(r[0], r[1], r[2], r[3]);
    p.set_user_index(r[4] < 0 ? -1 : 0);
    has_holes |= r[4] < 0;
    particles.push_back(p);
  }
  if (has_holes && groomer == "softdrop_reference")
    throw std::invalid_argument("The independent SoftDrop oracle is positive-particle-only");
  fastjet::ClusterSequence clustering(particles, antikt);
  std::map<std::string, int> counts{{"jets", 0}, {"negative_jets", 0},
      {"fiducial_jets", 0}, {"tagged", 0}, {"untagged", 0},
      {"negative_splits_skipped", 0}};
  // Ungroomed pT, absolute rg (not rg/R), zg, and explicit tagging flag.
  std::vector<std::array<double, 4>> output;
  for (const auto &jet : fastjet::sorted_by_pt(clustering.inclusive_jets())) {
    ++counts["jets"];
    if (jet.user_index() < 0) { ++counts["negative_jets"]; continue; }
    if (jet.pt() <= 158 || std::abs(jet.rap()) >= 2.1) continue;
    ++counts["fiducial_jets"];
    fastjet::ClusterSequence recluster(jet.constituents(), ca);
    const auto trees = recluster.inclusive_jets();
    if (trees.size() != 1) throw std::runtime_error("Jet did not form one C/A tree");
    double rg = 0, zg = 0;
    bool tagged = false;
    PseudoJet current = trees[0], a, b;
    if (groomer == "signed_prongs") {
      while (current.has_parents(a, b)) {
        // A subtraction branch is not a positive-energy physical SD prong.
        if (a.user_index() < 0 || b.user_index() < 0) {
          ++counts["negative_splits_skipped"];
          if (a.user_index() < 0 && b.user_index() < 0) break;
          current = a.user_index() < 0 ? b : a;
          continue;
        }
        const double sum = a.pt() + b.pt();
        if (sum > 0 && std::min(a.pt(), b.pt())/sum > 0.2) {
          rg = a.delta_R(b);
          zg = std::min(a.pt(), b.pt())/sum;
          tagged = true;
          break;
        }
        current = a.pt() >= b.pt() ? a : b;
      }
    } else if (groomer == "lund_reference") {
      fastjet::contrib::LundGenerator lund(ca);
      for (const auto &split : lund.result(current)) {
        if (split.z() > 0.2) {
          rg = split.Delta(); zg = split.z(); tagged = true; break;
        }
      }
    } else {
      fastjet::contrib::SoftDrop sd(0, 0.2, 0.4);
      sd.set_reclustering(false);
      sd.set_tagging_mode();
      const PseudoJet groomed = sd(current);
      if (groomed != 0 && groomed.has_parents(a, b)) {
        rg = a.delta_R(b); zg = std::min(a.pt(), b.pt())/(a.pt()+b.pt()); tagged = true;
      }
    }
    ++counts[tagged ? "tagged" : "untagged"];
    output.push_back({jet.pt(), rg, zg, tagged ? 1.0 : 0.0});
  }
  py::dict result;
  result["jets"] = output;
  result["cutflow"] = counts;
  return result;
}

PYBIND11_MODULE(_atlas_rg, m) {
  m.doc() = "ATLAS full-jet Soft-Drop radius, arXiv:2211.11470";
  m.def("reconstruct", &reconstruct_rg, py::arg("particles"),
        py::arg("groomer") = "signed_prongs");
}
