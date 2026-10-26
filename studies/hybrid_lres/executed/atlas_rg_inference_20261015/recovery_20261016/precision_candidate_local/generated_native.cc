// Local-only candidate. The physics reconstruction body is taken from pinned source.
#include <fastjet/ClusterSequence.hh>
#include <fastjet/tools/Recluster.hh>
#include "GroomerShop.hh"
#include "fjtools.cxx"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py=pybind11;
using fastjet::PseudoJet;
using Triple=std::array<double,3>;

double up(double value) {
  if (!std::isfinite(value) || value<0)
    throw std::invalid_argument("Nonfinite or negative numerical bound");
  const double result=std::nextafter(value,std::numeric_limits<double>::infinity());
  if (!std::isfinite(result)) throw std::invalid_argument("Numerical bound overflow");
  return result;
}

double down(double value) {
  return std::max(0.,std::nextafter(value,-std::numeric_limits<double>::infinity()));
}

double add_up(double a,double b) {
  if (a==0) return b;
  if (b==0) return a;
  return up(a+b);
}

template<class T> double rounding_radius(T value) {
  const auto infinity=std::numeric_limits<T>::infinity();
  const auto lower=std::nextafter(value,-infinity),upper=std::nextafter(value,infinity);
  double gap=0;
  if (std::isfinite(lower)) gap=std::max(gap,static_cast<double>(value)-lower);
  if (std::isfinite(upper)) gap=std::max(gap,static_cast<double>(upper)-value);
  return up(gap/2);
}

double decimal_radius(double magnitude,int digits) {
  if (magnitude==0) return 0;
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream<<std::scientific<<std::setprecision(digits-1)<<magnitude;
  const auto token=stream.str();
  const int exponent=std::stoi(token.substr(token.find('e')+1))-digits+1;
  double quantum=1;
  // Outward rounding at every step, including subnormal/zero storage cells.
  for (int j=0;j<std::abs(exponent);++j)
    quantum=up(exponent<0 ? quantum/10 : quantum*10);
  return up(quantum/2);
}

struct Bounds : public PseudoJet::UserInfoBase {
  Triple radius;
  size_t leaves;
  bool enabled;
  Bounds(Triple r,size_t n,bool e) : radius(r),leaves(n),enabled(e) {}
};

void validate_precision(int digits,bool single) {
  if (digits!=0 && digits!=6)
    throw std::invalid_argument("Declare six-digit decimal source or zero decimal model");
  if ((digits || single) && std::fegetround()!=FE_TONEAREST)
    throw std::invalid_argument("Precision model requires round-to-nearest arithmetic");
}

Triple leaf_bounds(const PseudoJet& p,int digits,bool single) {
  validate_precision(digits,single);
  const Triple values{p.px(),p.py(),p.pz()};
  Triple result{};
  for (size_t j=0;j<3;++j) {
    const double value=values[j];
    if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite source momentum");
    if (single && (!std::isfinite(static_cast<float>(value)) ||
        static_cast<double>(static_cast<float>(value))!=value))
      throw std::invalid_argument("Momentum does not match declared float32 storage");
    double bound=single ? rounding_radius(static_cast<float>(value))
                        : digits ? rounding_radius(value) : 0;
    if (digits && value!=0) {
      std::ostringstream stream;
      stream.imbue(std::locale::classic());
      stream<<std::scientific<<std::setprecision(digits-1)<<value;
      const auto token=stream.str();
      const auto decimal=std::strtold(token.c_str(),nullptr);
      if (single ? static_cast<float>(decimal)!=static_cast<float>(value)
                 : static_cast<double>(decimal)!=value)
        throw std::invalid_argument("Momentum does not round-trip through declared decimal precision");
    }
    // Include every possible decimal token in the storage cell, not only its centre.
    if (digits) bound=add_up(bound,decimal_radius(add_up(std::abs(value),bound),digits));
    result[j]=bound;
  }
  return result;
}

void attach_precision(PseudoJet& p,size_t index,size_t count,int digits,bool single,
                      const std::vector<Triple>& supplied) {
  validate_precision(digits,single);
  if (!supplied.empty() && (supplied.size()!=count || digits || single))
    throw std::invalid_argument("Explicit transported bounds require all rows and no new source model");
  if (!supplied.empty() && std::fegetround()!=FE_TONEAREST)
    throw std::invalid_argument("Transported bounds require round-to-nearest arithmetic");
  Triple radius=supplied.empty() ? leaf_bounds(p,digits,single) : supplied.at(index);
  for (double value:radius)
    if (!std::isfinite(value) || value<0) throw std::invalid_argument("Invalid transported numerical bound");
  p.set_user_info(new Bounds(radius,1,digits || single || !supplied.empty()));
}

void validate_bound_request(size_t count,int digits,bool single,const std::vector<Triple>& supplied) {
  validate_precision(digits,single);
  if (!supplied.empty() && (supplied.size()!=count || digits || single))
    throw std::invalid_argument("Explicit transported bounds require all rows and no new source model");
  for (const auto& radius:supplied)
    for (double value:radius)
      if (!std::isfinite(value) || value<0) throw std::invalid_argument("Invalid transported numerical bound");
  if (!supplied.empty() && std::fegetround()!=FE_TONEAREST)
    throw std::invalid_argument("Transported bounds require round-to-nearest arithmetic");
}

struct NormInterval {
  double lower_squared=0,upper_squared=0;
};

NormInterval norm_interval(const Triple& momentum,const Triple& radius) {
  NormInterval interval;
  for (size_t j=0;j<3;++j) {
    if (std::abs(momentum[j])<=radius[j]) continue;
    const double value=std::abs(momentum[j])-radius[j];
    const double lo=down(value),hi=up(value);
    interval.lower_squared=down(interval.lower_squared+down(lo*lo));
    interval.upper_squared=add_up(interval.upper_squared,up(hi*hi));
  }
  return interval;
}

py::dict norm_test(const Triple& momentum,const Triple& radius,double scale) {
  if (!std::isfinite(scale) || scale<=0 || scale>.001)
    throw std::invalid_argument("Invalid test scale");
  for (size_t j=0;j<3;++j)
    if (!std::isfinite(momentum[j]) || !std::isfinite(radius[j]) || radius[j]<0)
      throw std::invalid_argument("Invalid norm interval");
  const auto interval=norm_interval(momentum,radius);
  const double original=std::hypot(momentum[0],momentum[1],momentum[2]);
  const bool old_accept=original<=scale;
  const bool certified=interval.upper_squared<=down(scale*scale);
  py::dict result;
  result["lower_squared"]=interval.lower_squared;
  result["upper_squared"]=interval.upper_squared;
  result["accepted"]=old_accept || certified;
  result["ambiguous_rejected"]=!old_accept && !certified && interval.lower_squared<=up(scale*scale);
  return result;
}

class PrecisionGeometryRecombiner : public FJTools::NegativeEnergyRecombiner {
 public:
  PrecisionGeometryRecombiner(int marker,double scale)
    : FJTools::NegativeEnergyRecombiner(marker),scale_(scale) {}
  mutable size_t interventions=0,precision_interventions=0;
  mutable py::list decisions;
  void recombine(const PseudoJet& a,const PseudoJet& b,PseudoJet& out) const override {
    if (!a.has_user_info<Bounds>() || !b.has_user_info<Bounds>())
      throw std::runtime_error("Numerical bounds lost before recombination");
    const auto& ua=a.user_info<Bounds>(); const auto& ub=b.user_info<Bounds>();
    if (ua.enabled!=ub.enabled) throw std::runtime_error("Mixed precision contracts in one tree");
    FJTools::NegativeEnergyRecombiner::recombine(a,b,out);
    const Triple momentum{out.px(),out.py(),out.pz()};
    Triple radius{};
    for (size_t j=0;j<3;++j)
      radius[j]=add_up(add_up(ua.radius[j],ub.radius[j]),
          ua.enabled ? rounding_radius(momentum[j]) : 0);
    out.set_user_info(new Bounds(radius,ua.leaves+ub.leaves,ua.enabled));
    if (!(scale_>0 && (a.user_index()<0)!=(b.user_index()<0) && out.E()==0 && !std::isfinite(out.rap())))
      return;
    const double actual=std::hypot(out.px(),out.py(),out.pz());
    const auto interval=norm_interval(momentum,radius);
    const bool accepted=actual<=scale_ || (ua.enabled && interval.upper_squared<=down(scale_*scale_));
    py::dict decision;
    decision["residual"]=momentum; decision["radius"]=radius;
    decision["parent_radii"]=std::array<Triple,2>{ua.radius,ub.radius};
    decision["leaf_count"]=ua.leaves+ub.leaves;
    decision["norm"]=actual;
    decision["lower_norm"]=down(std::sqrt(interval.lower_squared));
    decision["upper_norm"]=up(std::sqrt(interval.upper_squared));
    decision["ambiguous_rejected"]=!accepted && interval.lower_squared<=up(scale_*scale_);
    decision["accepted"]=accepted; decision["nominal_scale"]=scale_;
    decision["precision_enabled"]=ua.enabled;
    decisions.append(decision);
    if (!accepted) return;
    const PseudoJet direction=a+b;
    if (!std::isfinite(direction.rap()))
      throw std::runtime_error("Undefined parent geometry for zero-energy cancellation");
    const auto original=std::array<double,4>{out.px(),out.py(),out.pz(),out.E()};
    out.set_cached_rap_phi(direction.rap(),direction.phi());
    if (original!=std::array<double,4>{out.px(),out.py(),out.pz(),out.E()})
      throw std::runtime_error("Angular cache changed a four-vector");
    ++interventions;
    if (actual>scale_) ++precision_interventions;
  }
 private:
  double scale_;
};

py::dict reconstruct(const std::vector<std::array<double, 7>>& rows,
                     int negative_marker = -1, double ca_radius = .4,
                     double zero_energy_scale = 0., std::string grooming_policy = "upstream_lund",
                     bool require_all_constituents = false, int source_decimal_digits = 0, bool source_float32 = false, const std::vector<Triple>& input_bounds = {}) {
  validate_bound_request(rows.size(),source_decimal_digits,source_float32,input_bounds);
  if (negative_marker != -1 && negative_marker != 0)
    throw std::invalid_argument("Use -1 (signed) or 0 (upstream default control)");
  if (!std::isfinite(zero_energy_scale) || zero_energy_scale < 0 || zero_energy_scale > .001)
    throw std::invalid_argument("Explicit zero-energy scale must be within [0,.001] GeV");
  if (!std::isfinite(ca_radius) || ca_radius <= 0 || ca_radius > 1000.)
    throw std::invalid_argument("Invalid C/A radius");
  if (grooming_policy != "upstream_lund" && grooming_policy != "signed_prongs")
    throw std::invalid_argument("Declare upstream_lund or signed_prongs");
  FJTools::NegativeEnergyRecombiner recombiner(negative_marker);
  PrecisionGeometryRecombiner ca_recombiner(negative_marker, zero_energy_scale);
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
    attach_precision(p,i,rows.size(),source_decimal_digits,source_float32,input_bounds);
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
  result["precision_boundary_interventions"] = ca_recombiner.precision_interventions;
  result["precision_decisions"] = ca_recombiner.decisions;
  return result;
}



std::vector<Triple> source_bounds(const std::vector<std::array<double,7>>& rows,int digits,bool single) {
  std::vector<Triple> result;
  for (const auto& r:rows) result.push_back(leaf_bounds(PseudoJet(r[0],r[1],r[2],r[3]),digits,single));
  return result;
}

py::dict combine_test(const std::vector<std::array<double,7>>& rows,
                     const std::vector<std::array<size_t,2>>& plan,int digits,bool single,
                     const std::vector<Triple>& supplied,double scale) {
  std::vector<PseudoJet> nodes;
  for (size_t i=0;i<rows.size();++i) {
    const auto& r=rows[i];
    PseudoJet p(r[0],r[1],r[2],r[3]);
    p.set_user_index(r[4]<0 ? -static_cast<int>(i+1) : static_cast<int>(i+1));
    attach_precision(p,i,rows.size(),digits,single,supplied);
    nodes.push_back(p);
  }
  PrecisionGeometryRecombiner recombiner(-1,scale);
  py::list reports;
  for (const auto& pair:plan) {
    if (pair[0]>=nodes.size() || pair[1]>=nodes.size() || pair[0]==pair[1])
      throw std::invalid_argument("Invalid test merge plan");
    PseudoJet out;
    recombiner.recombine(nodes[pair[0]],nodes[pair[1]],out);
    nodes.push_back(out);
    py::dict report;
    report["p4"]=std::array<double,4>{out.px(),out.py(),out.pz(),out.E()};
    report["radius"]=out.user_info<Bounds>().radius;
    report["leaf_count"]=out.user_info<Bounds>().leaves;
    report["sign"]=out.user_index()<0 ? -1 : 1;
    report["rapidity_finite"]=std::isfinite(out.rap());
    reports.append(report);
  }
  py::dict result;
  result["nodes"]=reports; result["decisions"]=recombiner.decisions;
  result["geometry_interventions"]=recombiner.interventions;
  result["precision_boundary_interventions"]=recombiner.precision_interventions;
  return result;
}

PYBIND11_MODULE(_rg_precision_candidate,m) {
  m.def("reconstruct",&reconstruct,py::arg("rows"),py::arg("negative_marker")=-1,
      py::arg("ca_radius")=.4,py::arg("zero_energy_scale")=0.,
      py::arg("grooming_policy")="upstream_lund",py::arg("require_all_constituents")=false,
      py::arg("source_decimal_digits")=0,py::arg("source_float32")=false,
      py::arg("input_bounds")=std::vector<Triple>{});
  m.def("source_bounds",&source_bounds,py::arg("rows"),py::arg("digits"),py::arg("single"));
  m.def("_test_norm",&norm_test,py::arg("momentum"),py::arg("radius"),py::arg("scale")=.0001);
  m.def("_test_combine",&combine_test,py::arg("rows"),py::arg("plan"),py::arg("digits")=0,
      py::arg("single")=false,py::arg("input_bounds")=std::vector<Triple>{},py::arg("scale")=.0001);
}
