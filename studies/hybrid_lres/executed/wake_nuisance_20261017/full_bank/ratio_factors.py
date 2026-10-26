"""Stream the established ratio Jacobians, preserving all paired event columns."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalization import blocks


def feature_keys(config):
    return [(b['name'], float(lo), float(hi)) for b in blocks(config)
            for lo, hi in zip(b['edges'][:-1], b['edges'][1:])]


def ratio_maps(totals, config, cross_sections, pp_mean):
    """Return AA/pp means and sparse numerator/normalization derivatives.

    Shapes: totals[centrality,design,a,raw feature]; derivatives have one entry
    per output bin. Shared pp noise is added separately, not repeated per event.
    """
    totals, pp = np.asarray(totals), np.asarray(pp_mean)
    if totals.ndim != 4 or totals.shape[0] != 2:
        raise ValueError('Need both complete paired AA cohorts')
    shape = totals.shape[1:-1]+(totals.shape[-1]-2,)
    if pp.shape != shape[-1:] or np.any(pp <= 0):
        raise ValueError('Invalid shared pp reference')
    mean = np.zeros(shape)
    numerator = np.zeros((2,)+shape)
    normalization = np.zeros_like(numerator)
    norm_indices = np.empty(shape[-1], int)
    offset = 0
    for b in blocks(config):
        stop = offset+len(b['edges'])-1
        sl = slice(offset, stop)
        obs = b['observable']
        norm = shape[-1] if obs == 'tg_alice' else shape[-1]+1
        norm_indices[sl] = norm
        hadron = '__hadron__' in b['name']
        separate = obs == 'dr12_atlas' or obs.startswith('rg_atlas')
        if hadron:
            if b['centrality'] not in ([0, 5], [5, 10]):
                raise ValueError('Unapproved hadron centrality')
            i = 0 if b['centrality'] == [0, 5] else 1
            pieces = [(i, totals[i], cross_sections[i], True)]
        elif separate:
            pieces = [(i, totals[i], .5*cross_sections[i], True) for i in (0, 1)]
        else:
            scale = 1. if obs == 'tg_alice' else np.mean(cross_sections, axis=0)
            pieces = [(i, totals.sum(0), scale, i == 0) for i in (0, 1)]
        for i, total, scale, add_mean in pieces:
            den = total[..., norm, None]
            if np.any(den <= 0):
                raise ValueError('Empty ratio normalization')
            coefficient = scale/den/pp[sl]
            numerator[i, ..., sl] = coefficient
            normalization[i, ..., sl] = -coefficient*total[..., sl]/den
            if add_mean:
                mean[..., sl] += scale*total[..., sl]/den/pp[sl]
        offset = stop
    if offset != shape[-1] or not np.isfinite(mean).all():
        raise ValueError('Incomplete or nonfinite prediction layout')
    return dict(mean=mean, numerator=numerator, normalization=normalization,
                norm_indices=norm_indices, pp_derivative=-mean/pp)


def aa_factors(events, centrality, maps):
    """Events are already weighted; paired designs and a nodes remain together."""
    events = np.asarray(events)
    f = maps['mean'].shape[-1]
    return (events[..., :f]*maps['numerator'][centrality]
            + events[..., maps['norm_indices']]*maps['normalization'][centrality])


def projected_pp_jacobian(derivative, retained, omitted, training, targets=()):
    """Map the same pp covariance once into training and noisy-target columns."""
    n = len(training)
    train = derivative[training]
    rr = (train.T[None, :, :]*retained[:, :, None]).transpose(0, 2, 1).reshape(-1, train.shape[1])
    oo = omitted*train.mean(0)
    jac = np.vstack([rr, oo])
    target = derivative[np.asarray(targets, int)]
    return jac, target


class NoiseAccumulator:
    """Exact train Gram, target/train cross blocks and per-target full covariance.

    Target/target blocks between different parameter points are not needed by
    single-point validation. They remain recoverable from the unchanged factors.
    """
    def __init__(self, retained, omitted, training, targets, features):
        self.retained, self.omitted = retained, omitted
        self.training, self.targets = np.asarray(training, int), np.asarray(targets, int)
        self.width = len(retained)*len(training)+len(omitted)
        self.train = np.zeros((self.width, self.width))
        self.cross = np.zeros((len(targets), features, self.width))
        self.variance = np.zeros((len(targets), features, features))

    def add(self, factors):
        training = factors[:, self.training]
        pc = np.einsum('enf,pf->epn', training, self.retained, optimize=True).reshape(len(factors), -1)
        oo = np.einsum('ef,of->eo', training.mean(1), self.omitted, optimize=True)
        columns = np.concatenate([pc, oo], axis=1)
        self.train += columns.T@columns
        if len(self.targets):
            target = factors[:, self.targets]
            self.cross += np.einsum('emf,ew->mfw', target, columns, optimize=True)
            self.variance += np.einsum('emf,emg->mfg', target, target, optimize=True)

    def add_pp(self, derivative, covariance):
        jac, target = projected_pp_jacobian(derivative, self.retained, self.omitted,
                                           self.training, self.targets)
        self.train += jac@covariance@jac.T
        if len(self.targets):
            self.cross += target[:, :, None]*(covariance@jac.T)[None, :, :]
            self.variance += target[:, :, None]*covariance[None, :, :]*target[:, None, :]

    def result(self):
        return (self.train+self.train.T)/2, dict(cross=self.cross,
            variance=(self.variance+self.variance.swapaxes(1, 2))/2)
