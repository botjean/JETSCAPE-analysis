"""Gate-only ratio/Jacobian algebra, preserving the reference centrality rules."""
import numpy as np


def blocks(config):
    hadron = [dict(b, observable=b['name'].split('__')[3])
              for b in config['layout'] if '__hadron__' in b['name']]
    return hadron + config['jet_blocks']


def aa_spectrum(first, second, config, cross_sections):
    """Return AA spectra and event-level first-order noise factors.

    Paired design/a/bin columns remain together within each event row. The
    centralities are independent samples, so their noise rows concatenate.
    """
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 4 or second.ndim != 4 or first.shape[1:] != second.shape[1:]:
        raise ValueError('Expected matching event,design,a,counts-and-normalizations axes')
    d, a, width = first.shape[1:]
    f = width-2
    totals = [first.sum(0), second.sum(0)]
    combined = totals[0]+totals[1]
    mean = np.full((d, a, f), np.nan)
    factors = [np.zeros((len(x), d, a, f)) for x in (first, second)]
    valid = np.zeros((d, a, f), bool)
    offset = 0
    for block in blocks(config):
        stop = offset+len(block['edges'])-1
        target = slice(offset, stop)
        obs = block['observable']
        is_hadron = '__hadron__' in block['name']
        is_separate = obs == 'dr12_atlas' or obs.startswith('rg_atlas')
        norm = f if obs == 'tg_alice' else f+1
        if is_hadron:
            index = 0 if block['centrality'] == [0, 5] else 1
            if block['centrality'] not in ([0, 5], [5, 10]):
                raise ValueError('Unexpected hadron centrality')
            pieces = [(index, totals[index], cross_sections[index])]
        elif is_separate:
            pieces = [(i, totals[i], .5*cross_sections[i]) for i in (0, 1)]
        else:
            scale = 1. if obs == 'tg_alice' else np.mean(cross_sections)
            pieces = [(i, combined, scale) for i in (0, 1)]
        ready = np.ones((d, a), bool)
        for _, total, _ in pieces:
            ready &= total[..., norm] > 0
        valid[..., target] = ready[..., None]
        mean[..., target] = 0.
        for i, total, scale in pieces:
            den = total[..., norm]
            safe = np.where(ready, den, 1.)
            numerator = total[..., target]
            # Pooled values appear once, but the derivative acts on both samples.
            if is_hadron or is_separate or i == 0:
                mean[..., target] += scale*numerator/safe[..., None]
            x = (first, second)[i]
            factors[i][..., target] = scale*(x[..., target]/safe[None, ..., None]
                -x[..., norm, None]*numerator[None, ...]/safe[None, ..., None]**2)
        mean[..., target] = np.where(ready[..., None], mean[..., target], np.nan)
        offset = stop
    if offset != f:
        raise ValueError('Counts do not match feature layout')
    return mean, np.concatenate(factors, axis=0), valid


def pp_spectrum(events, config, cross_section):
    events = np.asarray(events)
    if events.ndim != 3 or events.shape[1] != 1:
        raise ValueError('Shared pp has one undeformed node')
    x = events[:, 0]
    total = x.sum(0)
    f = x.shape[1]-2
    mean, noise = np.full(f, np.nan), np.zeros((len(x), f))
    offset = 0
    for block in blocks(config):
        stop = offset+len(block['edges'])-1
        target = slice(offset, stop)
        shape = block['observable'] == 'tg_alice'
        den = total[f if shape else f+1]
        if den > 0:
            scale = 1. if shape else cross_section
            mean[target] = scale*total[target]/den
            noise[:, target] = scale*(x[:, target]/den
                -x[:, f if shape else f+1, None]*total[None, target]/den**2)
        offset = stop
    return mean, noise
