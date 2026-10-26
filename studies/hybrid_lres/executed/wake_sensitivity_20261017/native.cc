// Isolated selectors using the existing signed FastJet and GroomerShop sources.
#include <fastjet/ClusterSequence.hh>
#include "GroomerShop.hh"
#include "fjtools.cxx"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <vector>

namespace py = pybind11;
using fastjet::PseudoJet;
using Row = std::array<double, 7>;

std::vector<PseudoJet> particles(const std::vector<Row>& rows, bool charged=false, bool calo=false) {
  std::vector<PseudoJet> result;
  for (size_t i=0; i<rows.size(); ++i) {
    const auto& r=rows[i];
    for (double value:r) if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite particle");
    if (r[3] <= 0 || (r[4] != 1 && r[4] != -1)) throw std::invalid_argument("Invalid signed particle");
    if ((charged && !r[6]) || (calo && !r[5])) continue;
    PseudoJet p(r[0],r[1],r[2],r[3]);
    p.set_user_index((r[4]<0?-1:1)*static_cast<int>(i+1));
    result.push_back(p);
  }
  return result;
}

std::vector<std::array<double,2>> axes(const std::vector<Row>& rows) {
  auto input=particles(rows,false,true);
  fastjet::ClusterSequence cs(input,fastjet::JetDefinition(fastjet::antikt_algorithm,.4));
  std::vector<std::array<double,2>> result;
  for (const auto& jet:fastjet::sorted_by_pt(cs.inclusive_jets(20.)))
    if (std::abs(jet.rap())<3.) result.push_back({jet.rap(),jet.phi_std()});
  return result;
}

py::dict inclusive(const std::vector<Row>& rows, double radius, bool charged=false, int marker=-1) {
  auto input=particles(rows,charged);
  FJTools::NegativeEnergyRecombiner recombiner(marker);
  fastjet::JetDefinition def(fastjet::antikt_algorithm,radius);
  def.set_recombiner(&recombiner);
  fastjet::ClusterSequence cs(input,def);
  std::vector<std::array<double,7>> result;
  std::vector<std::vector<int>> ids;
  for (const auto& jet:fastjet::sorted_by_pt(cs.inclusive_jets(20.))) {
    if (jet.user_index()<0) continue;
    double leading=0.;
    std::vector<int> constituents;
    for (const auto& p:jet.constituents()) {
      const int index=std::abs(p.user_index())-1;
      constituents.push_back(index);
      if (p.user_index()>0 && rows.at(index)[6]) leading=std::max(leading,p.pt());
    }
    result.push_back({jet.pt(),jet.eta(),jet.rap(),jet.phi_std(),leading,jet.E(),jet.m()});
    ids.push_back(constituents);
  }
  py::dict out;
  out["jets"]=result;
  out["constituents"]=ids;
  return out;
}

std::vector<std::array<double,3>> theta(const std::vector<Row>& rows) {
  auto input=particles(rows,true);
  FJTools::NegativeEnergyRecombiner recombiner(-1);
  fastjet::JetDefinition anti(fastjet::antikt_algorithm,.2), ca(fastjet::cambridge_algorithm,.2);
  anti.set_recombiner(&recombiner); ca.set_recombiner(&recombiner);
  fastjet::ClusterSequence cs(input,anti);
  std::vector<std::array<double,3>> result;
  for (const auto& jet:fastjet::sorted_by_pt(cs.inclusive_jets(20.))) {
    if (jet.user_index()<0 || std::abs(jet.eta())>=.7) continue;
    fastjet::contrib::GroomerShop shop(jet,ca);
    const auto* split=shop.soft_drop(0.,.2,.2);
    const bool tagged=split->z()>.2;
    result.push_back({jet.pt(),tagged?split->Delta()/.2:-1.,tagged?1.:0.});
  }
  return result;
}

PYBIND11_MODULE(_wake_native,m) {
  m.def("axes",&axes);
  m.def("inclusive",&inclusive,py::arg("rows"),py::arg("radius"),py::arg("charged")=false,
        py::arg("marker")=-1);
  m.def("theta",&theta);
}
