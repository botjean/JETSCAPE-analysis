"""Conserving signed transport fitted to Fig. 6; not Efficient Wake hydrodynamics.

The reference is the MASSLESS limit of Eq. (4.7), arXiv:1609.05842.
Positive and negative refer to its JOINT signed density, not the sign of
one-dimensional marginals. Number normalizations are solved together from
signed energy and x-momentum conservation at every candidate transport.
"""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent


def quadrature(n, low, high):
    x, w = leggauss(n)
    return (low+high)/2+(high-low)/2*x, w*(high-low)/2


def angle(phi, scale):
    return 2*np.arctan2(scale*np.sin(phi/2), np.cos(phi/2))


class Model:
    def __init__(self, temperature=.145, deposition=1., order=64):
        self.temperature, self.deposition = temperature, deposition
        self.y, self.wy = quadrature(order, -8., 8.)
        self.phi, self.wp = quadrature(order*2, -np.pi, np.pi)
        y, phi = self.y[:, None], self.phi[None, :]
        a = (np.cos(phi)+np.cosh(y)/3)*np.cosh(y)
        pref = deposition/(32*np.pi*temperature**5)
        # Integrate pT^3 and pT^4 analytically over [0,infinity).
        self.number = pref*a*6*(temperature/np.cosh(y))**4
        self.first_pt = pref*a*24*(temperature/np.cosh(y))**5
        self.weights = self.wy[:, None]*self.wp[None, :]
        self.target = np.array([np.sum(self.first_pt*np.cosh(y)*self.weights),
                                np.sum(self.first_pt*np.cos(phi)*self.weights)])

    def amplitudes(self, parameters):
        hp, hn, cp, cy = parameters
        y, phi = self.y[:, None], self.phi[None, :]
        positive = np.maximum(self.first_pt, 0)*self.weights
        negative = np.maximum(-self.first_pt, 0)*self.weights
        matrix = np.array([[np.sum(positive*hp*np.cosh(cy*y)),
                            -np.sum(negative*hn*np.cosh(y))],
                           [np.sum(positive*hp*np.cos(angle(phi, cp))),
                            -np.sum(negative*hn*np.cos(phi))]])
        return np.linalg.solve(matrix, self.target)

    def _integrated(self, y, phi, *, first_pt=False):
        t, dep = self.temperature, self.deposition
        power, factor = (5, 24) if first_pt else (4, 6)
        return dep/(32*np.pi*t**5)*(np.cos(phi)+np.cosh(y)/3)*np.cosh(y)*factor*(t/np.cosh(y))**power

    def marginals(self, coordinates, parameters=(1., 1., 1., 1.)):
        hp, hn, cp, cy = parameters
        ap, an = self.amplitudes(parameters)
        out = {}
        for obs, x in coordinates.items():
            if obs == 'pt':
                y, phi = self.y[:, None], self.phi[None, :]
                a = (np.cos(phi)+np.cosh(y)/3)*np.cosh(y)
                pref = self.deposition/(32*np.pi*self.temperature**5)
                plus, minus = np.maximum(a, 0), np.maximum(-a, 0)
                # Angle/rapidity transport integrates out of the pT marginal.
                out[obs] = np.array([np.sum(pref*(ap*plus*(pt/hp)**3/hp*np.exp(-pt/hp*np.cosh(y)/self.temperature)
                    -an*minus*(pt/hn)**3/hn*np.exp(-pt/hn*np.cosh(y)/self.temperature))*self.weights) for pt in x])
            elif obs == 'phi':
                old_phi = angle(x, 1/cp)
                jac = (np.cos(x/2)**2+np.sin(x/2)**2/cp**2)*cp
                pos = np.maximum(self._integrated(self.y[:, None], old_phi[None, :]), 0)
                neg = np.maximum(-self._integrated(self.y[:, None], x[None, :]), 0)
                out[obs] = ap*np.sum(pos*self.wy[:, None], axis=0)/jac-an*np.sum(neg*self.wy[:, None], axis=0)
            elif obs == 'y':
                pos = np.maximum(self._integrated(x[:, None]/cy, self.phi[None, :]), 0)
                neg = np.maximum(-self._integrated(x[:, None], self.phi[None, :]), 0)
                out[obs] = ap*np.sum(pos*self.wp[None, :], axis=1)/cy-an*np.sum(neg*self.wp[None, :], axis=1)
            else:
                raise ValueError(obs)
        return out


def fit(energy, curves):
    from scipy.optimize import least_squares
    coords = {obs: curves[f'{energy}_{obs}_old'][:, 0] for obs in ('pt', 'phi', 'y')}
    old = {obs: curves[f'{energy}_{obs}_old'][:, 1] for obs in coords}
    norms = {obs: max(np.max(np.abs(old[obs])), 1e-12) for obs in coords}

    def old_residual(z):
        model = Model(np.exp(z[0]), np.exp(z[1]))
        pred = model.marginals(coords)
        return np.concatenate([(pred[obs]-old[obs])/norms[obs]/np.sqrt(len(old[obs])) for obs in coords])

    initial = least_squares(old_residual, np.log([.145, energy*.75]),
                            bounds=(np.log([.08, .01]), np.log([.3, energy*2])),
                            xtol=1e-9, ftol=1e-9, gtol=1e-9)
    old_rmse = float(np.sqrt(np.mean(old_residual(initial.x)**2)))
    model = Model(*np.exp(initial.x))
    coords = {obs: curves[f'{energy}_{obs}_efficient'][:, 0] for obs in coords}
    target = {obs: curves[f'{energy}_{obs}_efficient'][:, 1] for obs in coords}
    norms = {obs: max(np.max(np.abs(target[obs])), 1e-12) for obs in coords}

    def residual(z):
        par = np.exp(z)
        amps = model.amplitudes(par)
        pred = model.marginals(coords, par)
        values = [(pred[obs]-target[obs])/norms[obs]/np.sqrt(len(target[obs])) for obs in coords]
        return np.r_[np.concatenate(values), np.minimum(amps, 0)*10]

    candidates = []
    for start in ([2.5, 1., .5, .6], [3.5, 1.5, .3, .5], [2., .8, .7, .7]):
        result = least_squares(residual, np.log(start), bounds=(np.log([1., .5, .1, .15]),
            np.log([6., 3., 1., 1.])), max_nfev=150, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        candidates.append(result)
    result = min(candidates, key=lambda item: item.cost)
    parameters = np.exp(result.x)
    amplitudes = model.amplitudes(parameters)
    if not result.success or np.any(amplitudes <= 0):
        raise RuntimeError('Nonphysical or failed calibration')
    pred = model.marginals(coords, parameters)
    report = dict(energy=energy, temperature=model.temperature, deposition=model.deposition,
        parameters=dict(zip(('positive_pt_scale', 'negative_pt_scale', 'positive_phi_scale', 'positive_y_scale'), parameters)),
        amplitudes=dict(zip(('positive', 'negative'), amplitudes)), optimizer_success=bool(result.success),
        old_shape_rmse=old_rmse,
        normalized_rmse={obs: float(np.sqrt(np.mean((pred[obs]-target[obs])**2))/norms[obs]) for obs in coords},
        target_signed_energy_Px=model.target.tolist(),
        caveat='Fit to signed massless single-parton marginals; species, joint structure and shower transfer remain assumptions')
    return report


def main():
    with np.load(HERE/'paper/figure6_curves.npz') as z:
        curves = {k: z[k] for k in z.files}
    reports = [fit(energy, curves) for energy in (10, 50)]
    (HERE/'CALIBRATION.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__':
    main()
