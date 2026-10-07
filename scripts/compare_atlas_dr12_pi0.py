"""Paired retain-pi0 versus pi0->gamma gamma check; never exports production tables."""

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jetscape_analysis.analysis import atlas_dr12 as analysis


def inspect_sample(path, system, config, limit, seed):
    cutflows = {mode: Counter() for mode in ("keep", "decay_gg")}
    histograms = {}
    for observable, block in config["observables"].items():
        bins = len(block["edges"]) - 1
        histograms[observable] = {k: np.zeros(bins) for k in ("keep", "decay_gg", "paired_difference_sumw2")}
    weight_sum = 0.0
    events = changed_events = 0
    sigma = None
    qa = {}
    for event in islice(analysis.read_events(path), limit):
        weight = float(event["event_weight"])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Invalid event weight")
        current = float(event["cross_section"])
        if sigma is None:
            sigma = current
        if not np.isclose(current, sigma, rtol=1e-6, atol=0):
            raise ValueError("Cross section changed inside this file")
        weight_sum += weight
        events += 1
        counts = {}
        jets_by_mode = {}
        for mode in cutflows:
            rows = analysis.particle_rows(event, system, mode, seed, qa if mode == "decay_gg" else None)
            result = analysis._atlas_dr12.reconstruct(rows, groomer=config.get("grooming", "lund"))
            jets = [j for j in result["jets"] if config["jet_pt"][0] <= j[0] < config["jet_pt"][1]]
            jets_by_mode[mode] = jets
            cutflows[mode].update(result["cutflow"])
            cutflows[mode]["jets_in_pt_interval"] += len(jets)
            for observable, block in config["observables"].items():
                edges = np.asarray(block["edges"])
                values = np.array([j[block["column"]] for j in jets])
                counts[mode, observable] = np.histogram(values[(values >= edges[0]) & (values < edges[-1])], edges)[0]
                histograms[observable][mode] += weight * counts[mode, observable]
        if len(jets_by_mode["keep"]) != len(jets_by_mode["decay_gg"]):
            changed_events += 1
        for observable in histograms:
            delta = weight * (counts["decay_gg", observable] - counts["keep", observable])
            histograms[observable]["paired_difference_sumw2"] += delta**2
        if events % 1000 == 0:
            print(f"{path.name}: {events} paired events", flush=True)
    if not weight_sum > 0 or sigma is None or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Empty/invalid normalization")
    factor = sigma / weight_sum
    result = {}
    for observable, h in histograms.items():
        keep, decay = factor * h["keep"], factor * h["decay_gg"]
        result[observable] = dict(
            keep=keep.tolist(), decay_gg=decay.tolist(),
            paired_difference_error=(factor * np.sqrt(h["paired_difference_sumw2"])).tolist(),
            integrated_relative_change=float(decay.sum() / keep.sum() - 1) if keep.sum() > 0 else None,
        )
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(input=str(path.resolve()), sha256=digest, system=system, events=events,
                weight_sum=weight_sum, cross_section=sigma, seed=seed,
                changed_selected_jet_count_events=changed_events,
                cutflows={m: dict(c) for m, c in cutflows.items()},
                particle_qa=qa, histograms=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp", type=Path, required=True)
    parser.add_argument("--aa05", type=Path, required=True)
    parser.add_argument("--aa510", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/atlas_dr12_5020.yaml"))
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_events <= 0 or args.seed < 0:
        parser.error("Require positive max-events and nonnegative seed")
    args.output.mkdir(parents=True, exist_ok=False)
    config = analysis.load_config(args.config)
    samples = {}
    for name, path, system in (("pp", args.pp, "pp"), ("PbPb 0-5%", args.aa05, "PbPb"), ("PbPb 5-10%", args.aa510, "PbPb")):
        samples[name] = inspect_sample(path, system, config, args.max_events, args.seed)
        (args.output / "summary.json").write_text(json.dumps(dict(
            version=analysis.VERSION, config=config, samples=samples,
            note="Paired small-sample sensitivity only. Forced two-photon decay, not a full branching-ratio model. Errors are paired event-Poisson at fixed normalization; no inference or production tables."
        ), indent=2) + "\n")
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "legend.fontsize": 14})
    for observable, block in config["observables"].items():
        edges = np.asarray(block["edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout="constrained", height_ratios=[2, 1])
        for column, (name, sample) in enumerate(samples.items()):
            h = sample["histograms"][observable]
            keep, decay = np.asarray(h["keep"]), np.asarray(h["decay_gg"])
            top, bottom = axes[:, column]
            top.stairs(keep / widths, edges, color="#007D80", linewidth=2.3, label=r"Retain $\pi^0$")
            top.stairs(decay / widths, edges, color="#D26926", linewidth=2.3, linestyle="--", label=r"$\pi^0\to\gamma\gamma$")
            top.set_title(name)
            top.legend(frameon=False)
            mask = keep > 0
            bottom.axhline(0, color="0.5", linewidth=1)
            bottom.errorbar(centers[mask], 100*(decay[mask]-keep[mask])/keep[mask],
                            yerr=100*np.asarray(h["paired_difference_error"])[mask]/keep[mask],
                            fmt="o", color="#D26926", markersize=4, capsize=2)
            bottom.set_xlabel(r"$\Delta R_{12}$" if observable == "dr12_atlas" else r"$\sqrt{d_{12}}$ [GeV]")
            for ax in (top, bottom):
                ax.spines[["top", "right"]].set_visible(False)
                if observable == "sqrt_d12_atlas":
                    ax.set_xscale("log")
            if np.any(keep > 0) or np.any(decay > 0):
                top.set_yscale("log")
        axes[0, 0].set_ylabel(r"$d\sigma/dx$ [mb / unit $x$]")
        axes[1, 0].set_ylabel("Change [%]")
        for extension in ("png", "pdf"):
            fig.savefig(args.output / f"pi0_sensitivity_{observable}.{extension}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
