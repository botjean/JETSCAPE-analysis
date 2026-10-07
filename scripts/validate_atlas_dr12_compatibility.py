"""Read-only local event comparisons against heppy-formula/Lund/SoftDrop references."""

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path

import numpy as np

from jetscape_analysis.analysis import atlas_dr12 as analysis


def compare(path, system, limit, config):
    modes = {
        "signed_custom": dict(groomer="signed_prongs"),
        "heppy_marker_minus1_custom": dict(recombiner_mode="legacy_reference", legacy_marker=-1, groomer="signed_prongs"),
        "signed_lund": dict(groomer="lund"),
        "signed_softdrop": dict(groomer="softdrop_reference"),
        "heppy_marker0_lund": dict(recombiner_mode="legacy_reference", legacy_marker=0, groomer="lund"),
    }
    totals = {name: Counter() for name in modes}
    mismatches = {name: [] for name in modes}
    mismatch_counts = Counter()
    hist = {name: {o: np.zeros(len(b["edges"])-1) for o, b in config["observables"].items()} for name in modes}
    events = 0
    for i, event in enumerate(islice(analysis.read_events(path), limit)):
        rows = analysis.particle_rows(event, system)
        results = {}
        for mode, kwargs in modes.items():
            result = analysis._atlas_dr12.reconstruct(rows, **kwargs)
            jets = [j for j in result["jets"] if config["jet_pt"][0] <= j[0] < config["jet_pt"][1]]
            results[mode] = jets
            totals[mode].update(result["cutflow"])
            totals[mode]["selected_jets"] += len(jets)
            for o, b in config["observables"].items():
                values = np.array([j[b["column"]] for j in jets])
                edges = np.asarray(b["edges"])
                hist[mode][o] += event["event_weight"] * np.histogram(values[(values>=edges[0]) & (values<edges[-1])], edges)[0]
        reference = results["signed_custom"]
        for mode, jets in results.items():
            if len(jets) != len(reference) or not np.allclose(jets, reference, atol=1e-10, rtol=1e-10):
                mismatch_counts[mode] += 1
                if len(mismatches[mode]) < 10:
                    mismatches[mode].append(dict(event_index=i, reference=reference, variant=jets))
        events += 1
        if events % 1000 == 0:
            print(f"{path.name}: {events} events, differences {dict(mismatch_counts)}", flush=True)
    return dict(input=str(path.resolve()), system=system, events=events,
                cutflows={k: dict(v) for k, v in totals.items()},
                mismatched_events={k: mismatch_counts[k] for k in modes},
                examples=mismatches,
                histogram_relative_changes={k: {o: float(h.sum()/hist["signed_custom"][o].sum()-1)
                    if hist["signed_custom"][o].sum()>0 else None for o,h in hs.items()} for k,hs in hist.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True, help="Local directory of the three approved pilot files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-events", type=int, default=5000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = analysis.load_config(Path("config/atlas_dr12_5020.yaml"))
    report = dict(version=analysis.VERSION, reference_scope="Same ATLAS kt-tree definition. Heppy recombiner formula and first-passing Lund split verified against cluster source; not the old full container binary.", samples={})
    for name, filename, system in (
        ("pp", "HYBRID_pp_0000_final_state_hadrons_00.parquet", "pp"),
        ("aa05", "HYBRID_PbPb_0000_final_state_hadrons_00.parquet", "PbPb"),
        ("aa510", "HYBRID_PbPb_5-10_0000_final_state_hadrons_00.parquet", "PbPb"),
    ):
        report["samples"][name] = compare(args.inputs/filename, system, args.max_events, config)
        (args.output/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
