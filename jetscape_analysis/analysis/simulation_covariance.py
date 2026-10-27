"""Joint event statistics for saved STAT observables and ATLAS dR12.

This is an opt-in sidecar producer. It does not modify observable reconstruction
or existing tables. All-event weights and pThat must match the original particle
file row by row; matching only the number of accepted events is insufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np


class EventMoments:
    """Sufficient statistics for joint event-Poisson uncertainty propagation."""

    def __init__(self, n_features):
        self.total = np.zeros(n_features + 1)
        self.outer = np.zeros((n_features + 1, n_features + 1))
        self.events = 0

    def update(self, counts, weight):
        counts = np.asarray(counts, dtype=float)
        if counts.shape != self.total[:-1].shape or not np.isfinite(counts).all():
            raise ValueError("Invalid event contribution vector")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Invalid event weight")
        # The final component retains fluctuations of the all-event denominator.
        vector = weight * np.r_[counts, 1.0]
        self.total += vector
        self.outer += np.outer(vector, vector)
        self.events += 1


def histogram_counts(values, edges):
    """ROOT [low, high) bins; also return underflow + visible normalization."""
    values = np.asarray([] if values is None else values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite, already selected scalar observables")
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Invalid histogram edges")
    counts = np.histogram(values[(values >= edges[0]) & (values < edges[-1])], edges)[0]
    return counts, int(np.count_nonzero(values < edges[-1]))


def ratio_map(total, numerators, denominators, factors=1.0):
    """Values and exact Jacobian for scaled ratios of linear sufficient stats."""
    total = np.asarray(total, dtype=float)
    a, b = np.asarray(numerators, float), np.asarray(denominators, float)
    if a.ndim != 2 or b.shape != a.shape or a.shape[1] != len(total):
        raise ValueError("Incompatible ratio maps")
    num, den = a @ total, b @ total
    if not np.isfinite(total).all() or np.any(den <= 0):
        raise ValueError("Nonfinite statistics or nonpositive denominator")
    scale = np.broadcast_to(factors, num.shape)
    if not np.isfinite(scale).all():
        raise ValueError("Nonfinite ratio scale")
    jac = scale[:, None] * (a / den[:, None] - num[:, None] * b / den[:, None]**2)
    return scale * num / den, jac


def ratio_bank(aa, aa_cov, pp, pp_cov):
    """Full design x feature covariance for independent AA and one shared pp.

    Inputs must already include within-sample normalization and bin/observable
    cross-covariance. No assumption that different observable blocks are diagonal.
    """
    aa, aa_cov, pp, pp_cov = map(np.asarray, (aa, aa_cov, pp, pp_cov))
    if aa.ndim != 2 or pp.shape != aa.shape[1:]:
        raise ValueError("Incompatible spectra")
    n, f = aa.shape
    if aa_cov.shape != (n, f, f) or pp_cov.shape != (f, f):
        raise ValueError("Incompatible covariance dimensions")
    if not all(np.isfinite(v).all() for v in (aa, aa_cov, pp, pp_cov)) or np.any(pp <= 0):
        raise ValueError("Invalid spectra or empty pp bin")
    derivative = -aa / pp**2
    covariance = np.einsum("ib,bc,jc->ibjc", derivative, pp_cov, derivative)
    for i in range(n):
        covariance[i, :, i, :] += aa_cov[i] / np.outer(pp, pp)
    return aa / pp, covariance


def validate_alignment(observable, event):
    for saved, raw in (("event_weight_all", "event_weight"), ("pt_hat", "pt_hat")):
        a, b = observable.get(saved), event.get(raw)
        if a is None or b is None or not np.isfinite(a) or a != b:
            raise ValueError(f"Event alignment failed for {saved}; do not guess a join")
    accepted_weight = observable.get("event_weight")
    if accepted_weight is not None and accepted_weight != event["event_weight"]:
        raise ValueError("Accepted-event weight differs from all-event weight")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def extract(observables, particles, cross_section, layout, system, centrality,
            dr12_config, max_events=None):
    import pyarrow.parquet as pq
    from jetscape_analysis.analysis import atlas_dr12 as dr

    dr.validate_config(dr12_config)
    if system not in ("pp", "PbPb") or (system == "PbPb") != (centrality is not None):
        raise ValueError("Specify an actual PbPb centrality, none for pp")
    if centrality is not None and not 0 <= centrality[0] < centrality[1] <= 10:
        raise ValueError("Expected an actual centrality interval within 0--10")
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive")
    obs_file, raw_file = pq.ParquetFile(observables), pq.ParquetFile(particles)
    if obs_file.metadata.num_rows != raw_file.metadata.num_rows:
        raise ValueError("Different event counts: cannot align saved observables")
    metadata_rows = pq.read_table(cross_section).to_pylist()
    if len(metadata_rows) != 1:
        raise ValueError("Expected one normalization record")
    metadata = metadata_rows[0]
    if not np.isfinite(metadata['cross_section']) or metadata['cross_section'] <= 0:
        raise ValueError('Invalid recorded generator cross section')
    if not np.isfinite(metadata['weight_sum']) or metadata['weight_sum'] <= 0:
        raise ValueError('Invalid recorded all-event weight sum')
    if int(metadata["n_events"]) != raw_file.metadata.num_rows:
        raise ValueError("Normalization record has a different event count")
    features, active, columns = [], [], {"event_weight", "event_weight_all", "pt_hat"}
    norms = []
    for spec in layout:
        edges = np.asarray(spec["edges"], float)
        histogram_counts([], edges)
        start = len(features)
        features.extend((spec["name"], float(a), float(b)) for a, b in zip(edges[:-1], edges[1:]))
        column = spec.get("column_pp" if system == "pp" else "column_aa")
        enabled = system == "pp" or (spec["centrality"][0] <= centrality[0] < centrality[1] <= spec["centrality"][1])
        active.append((spec, slice(start, len(features)), column, enabled))
        if column:
            columns.add(column)
        if spec.get("self_normalized"):
            norms.append(spec["name"])
    if len(set(features)) != len(features):
        raise ValueError("Duplicate observable/bin identities")
    schema = set(obs_file.schema_arrow.names)
    if not {"event_weight_all", "pt_hat", "event_weight"} <= schema:
        raise ValueError("Saved observables lack mandatory all-event alignment fields")
    missing = sorted(columns - schema)
    # Legacy writers omit an entire observable column when no event filled it.
    obs_iter = (row for batch in obs_file.iter_batches(columns=sorted(columns & schema), batch_size=1024)
                for row in batch.to_pylist())
    raw_iter = dr.read_events(Path(particles))
    pairs = itertools.zip_longest(obs_iter, raw_iter)
    if max_events is not None:
        pairs = itertools.islice(pairs, max_events)
    moments = EventMoments(len(features) + len(norms))
    jet_sumw2 = np.zeros(len(features))
    start_time = time.monotonic()
    sentinel = object()
    for row, event in pairs:
        if row is None or event is None:
            raise ValueError("Event streams ended at different positions")
        validate_alignment(row, event)
        if not np.isclose(event["cross_section"], metadata["cross_section"], rtol=1e-6, atol=0):
            raise ValueError("Particle/observable cross sections differ")
        if centrality is not None and not centrality[0] <= event["centrality"] < centrality[1]:
            raise ValueError("Particle centrality contradicts requested sample")
        vector = np.zeros(len(moments.total) - 1)
        dr_values = sentinel
        for spec, target, column, enabled in active:
            if not enabled:
                continue
            if spec.get("source") == "dr12":
                if dr_values is sentinel:
                    result = dr._atlas_dr12.reconstruct(dr.particle_rows(
                        event, system, dr12_config.get("pi0_treatment", "keep"),
                        dr12_config.get("pi0_decay_seed", 0)), groomer=dr12_config.get("grooming", "lund"))
                    low, high = dr12_config["jet_pt"]
                    dr_values = [j[1] for j in result["jets"] if low <= j[0] < high]
                values = dr_values
            else:
                values = row.get(column)
            counts, norm = histogram_counts(values, spec["edges"])
            vector[target] = counts
            if spec.get("self_normalized"):
                vector[len(features) + norms.index(spec["name"])] = norm
        weight = event["event_weight"]
        moments.update(vector, weight)
        jet_sumw2 += weight**2 * vector[:len(features)]
        if moments.events % 10000 == 0:
            print(f"{system}: {moments.events} events, {time.monotonic()-start_time:.1f}s", flush=True)
    partial = moments.events != raw_file.metadata.num_rows
    if not partial:
        np.testing.assert_allclose(moments.total[-1], metadata["weight_sum"], rtol=1e-12)
    report = dict(system=system, centrality=centrality, events=moments.events, partial=partial,
                  seconds=time.monotonic()-start_time, features=features, normalization_features=norms,
                  layout=layout, dr12_config=dr12_config, cross_section_metadata=metadata,
                  dr12_version=dr.VERSION, dr12_source_sha256=digest(dr.__file__),
                  missing_zero_columns=missing,
                  sources={str(Path(p).resolve()): digest(p) for p in (observables, particles, cross_section)},
                  method="Joint weighted event-Poisson sufficient statistics, including all-event and theta normalization denominators. Generator cross sections retained, not independently resampled.")
    return dict(total=moments.total, outer=moments.outer, jet_sumw2=jet_sumw2), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observables", "particles", "cross-section", "layout", "dr12-config", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--system", choices=("pp", "PbPb"), required=True)
    parser.add_argument("--centrality", type=int, nargs=2)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Refusing to overwrite prior covariance output")
    from jetscape_analysis.analysis import atlas_dr12 as dr
    arrays, report = extract(args.observables, args.particles, args.cross_section,
                             json.loads(args.layout.read_text()), args.system, args.centrality,
                             dr.load_config(args.dr12_config), args.max_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("events", "partial", "seconds", "missing_zero_columns")}), flush=True)


if __name__ == "__main__":
    main()
