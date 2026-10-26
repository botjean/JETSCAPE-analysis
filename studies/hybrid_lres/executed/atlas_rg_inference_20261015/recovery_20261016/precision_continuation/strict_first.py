"""Keep successful pinned reconstruction unchanged; repair one explicit failure."""
from __future__ import annotations

from collections import Counter


RETENTION_ERROR = "Upstream C/A reclustering lost jet constituents"


class StrictFirst:
    def __init__(self, strict, precision):
        self.strict = strict
        self.precision = precision
        self.audit = Counter()
        self.events = []
        self.context = None

    def reconstruct(self, rows, *args, **kwargs):
        # Successful calls return the original object, without a second fit/pass.
        try:
            result = self.strict.reconstruct(rows, *args, **kwargs)
        except RuntimeError as error:
            if str(error) != RETENTION_ERROR:
                raise
            if args or kwargs != dict(negative_marker=-1, ca_radius=1000,
                    zero_energy_scale=.0001, grooming_policy="signed_prongs",
                    require_all_constituents=True):
                raise ValueError("Precision recovery requires the exact frozen configuration") from error
            self.audit["retention_failures"] += 1
            result = self.precision.reconstruct(rows, **kwargs,
                source_decimal_digits=6, source_float32=True)
            if result.get("precision_boundary_interventions", 0) <= 0:
                raise RuntimeError("Recovery did not exercise the approved precision boundary") from error
            details = result.get("reclustering", [])
            if not details or not all(d["all_constituents_retained"] for d in details):
                raise RuntimeError("Recovered reclustering has incomplete constituents") from error
            self.audit["recovered_events"] += 1
            self.audit["precision_boundary_interventions"] += result["precision_boundary_interventions"]
            self.events.append(dict(event=self.context, trigger=RETENTION_ERROR,
                decisions=result.get("precision_decisions", [])))
            return result
        self.audit["unchanged_successes"] += 1
        return result


def install(rg, precision, policy_path):
    """Install only in the isolated collector process, not the analysis checkout."""
    from pathlib import Path
    import hashlib

    strict = rg._upstream_backend()
    before = rg.reconstruction_hashes
    wrapper = StrictFirst(strict, precision)
    paths = [Path(policy_path), Path(precision.__file__),
             Path(precision.__file__).with_name("precision_build.json")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def reconstruction_hashes(config=None):
        # The original function checks the actual strict backend and runtime.
        return {**before(config), **hashes}

    # Keep the strict loader for original fingerprinting; intercept only count_rows.
    original = rg.count_rows

    def count_rows(rows, config):
        previous = rg._upstream_backend
        rg._upstream_backend = lambda: wrapper
        try:
            return original(rows, config)
        finally:
            rg._upstream_backend = previous

    rg.count_rows = count_rows
    original_event = rg.count_event

    def count_event(event, config, system):
        wrapper.context = dict(system=system,
            **{k: float(event[k]) for k in ("event_ID", "event_weight", "pt_hat", "vertex_x", "vertex_y") if k in event})
        try:
            return original_event(event, config, system)
        finally:
            wrapper.context = None

    rg.count_event = count_event
    rg.reconstruction_hashes = reconstruction_hashes
    return wrapper
