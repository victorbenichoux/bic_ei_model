import numpy as np
from scipy.optimize import minimize
from sympy import Symbol, lambdify, erf, sqrt, pi, exp, diff
import scipy.special as spec

'''
Fitting of the EI model of the BIC. 
author: victorbenichoux
'''

def get_sympy_functions():
    '''
    This function returns different intermediate Sympy functions used to compute the fit jacobian (for the lags).

    :return:
    '''
    ITD = Symbol('itd')
    sigma = Symbol('sigma')
    w = Symbol('w')
    x = Symbol('x')
    rL = 1. / 2 * (erf((w - ITD) / (sqrt(2) * sigma)) - erf(-ITD / (sqrt(2) * sigma)))
    rR = rL.subs(ITD, -ITD)
    r = (rL + rR) / (rL + rR).subs(ITD, 0)

    dr_dsigma = diff(r, sigma)
    dr_dw = diff(r, w)

    phi = 1. / (sqrt(2 * pi)) * exp(-1. / 2 * x ** 2)
    deltaL = 3. / 2 * ITD + sigma * (phi.subs(x, -ITD / sigma) - phi.subs(x, (w - ITD) / sigma)) / rL
    deltaR = deltaL.subs(ITD, -ITD)
    delta = (deltaL * rL + deltaR * rR) / (rL + rR)
    delta = delta.simplify()
    ddelta_dsigma = diff(delta, sigma)
    ddelta_dw = diff(delta, w)

    deltaL_fun = lambdify((sigma, w, ITD), deltaL, modules=[np, spec])
    delta_fun = lambdify((sigma, w, ITD), delta, modules=[np, spec])
    ddelta_dsigma_fun = lambdify((sigma, w, ITD), ddelta_dsigma, modules=[np, spec])
    ddelta_dw_fun = lambdify((sigma, w, ITD), ddelta_dw, modules=[np, spec])

    r_fun = lambdify((sigma, w, ITD), r, modules=[np, spec])
    rL_fun = lambdify((sigma, w, ITD), rL, modules=[np, spec])

    dr_dsigma_fun = lambdify((sigma, w, ITD), dr_dsigma, modules=[np, spec])
    dr_dw_fun = lambdify((sigma, w, ITD), dr_dw, modules=[np, spec])

    return delta_fun, ddelta_dsigma_fun, ddelta_dw_fun, r_fun, dr_dsigma_fun, dr_dw_fun, rL_fun, deltaL_fun


delta_fun, ddelta_dsigma_fun, ddelta_dw_fun, r_fun, dr_dsigma_fun, dr_dw_fun, rL_fun, deltaL_fun = get_sympy_functions()


def f_amps_sympy(p, amps, fititds, return_pred=False):
    '''
    Returns the sum of squared difference between the current fitted amplitudes (with parameters p) and the data

    :param p:
    :param amps:
    :param fititds:
    :param return_pred:
    :return:
    '''
    sigma, w, a, b = p[:]

    rbar = r_fun(sigma, w, fititds)
    pred_amp = a * rbar + b
    if return_pred:
        return pred_amp

    grad_pred_amp = np.zeros((len(amps), 4))
    # dpred_amp/dsigma
    grad_pred_amp[:, 0] = a * dr_dsigma_fun(sigma, w, fititds)
    # dpred_amp/dw
    grad_pred_amp[:, 1] = a * dr_dw_fun(sigma, w, fititds)
    # dpred_amp/da
    grad_pred_amp[:, 2] = rbar
    # dpred_amp/db
    grad_pred_amp[:, 3] = 1.

    return np.mean((pred_amp - amps) ** 2), np.sum(grad_pred_amp * 2 * (pred_amp - amps).reshape((len(amps), 1)),
                                                   axis=0)


def f_lags(p, lags, fititds, return_pred=False):
    '''
    Returns the sum of squared difference between the current fitted lags (with parameters p) and the data

    :param p:
    :param lags:
    :param fititds:
    :param return_pred:
    :return:
    '''
    sigma, w, offset = p[:]

    pred_lags = delta_fun(sigma, w, fititds) - offset
    if return_pred:
        return pred_lags

    grad_pred_lags = np.zeros((len(fititds), 3))

    # dpred_lags/dsigma
    grad_pred_lags[:, 0] = ddelta_dsigma_fun(sigma, w, fititds)

    # dpred_lags/dw
    grad_pred_lags[:, 1] = ddelta_dw_fun(sigma, w, fititds)

    # dpred_lags/doffset
    grad_pred_lags[:, 2] = -1.

    return np.sum((pred_lags - lags) ** 2), np.sum(grad_pred_lags * 2 * (pred_lags - lags).reshape((len(lags), 1)),
                                                   axis=0)


def f_amplags(p, amps, lags, fititds, return_pred=False):
    '''
    This function returns the sum of squared difference between the current fitted values (with parameters p) and the data.
    This is the quantity that has to be minimized. It also returns the evaluation of the gradient.

    :param p: parameter values
    :param amps: values of the BIC amplitude
    :param lags: values of the BIC lags
    :param fititds: ITD points
    :param return_pred: Only returns amps/lags if True
    :return:
    '''
    sigma, w, a, b, offset = p

    tmp = f_amps_sympy([sigma, w, a, b], amps, fititds, return_pred=return_pred)
    tmp1 = f_lags([sigma, w, offset], lags, fititds, return_pred=return_pred)
    if return_pred:
        return tmp, tmp1
    lags_ssq, lags_grad = tmp1
    amps_ssq, amps_grad = tmp

    grad_out = np.zeros(5)
    grad_out[0] = lags_grad[0] + amps_grad[0]
    grad_out[1] = lags_grad[1] + amps_grad[1]
    grad_out[2] = amps_grad[2]
    grad_out[3] = amps_grad[3]
    grad_out[4] = lags_grad[2]

    ssq = lags_ssq + amps_ssq

    return ssq, grad_out


def fit_ei_model(itds, amps, lags,
                 params_init=[0.5, .5, 1., 1., 0.],
                 fit_bounds=[(0.05, 2.), ] * 4 + [(None, None)]):
    '''
    This function fits the amplitudes and the lags of a measured BIC vs. ITD trace.

    :param itds: a (N,) array of ITD points where the data was gathered
    :param amps: a (N,) array of BIC amplitude measurements
    :param lags: a (N,) array of BIC lag measurements
    :param params_init: The initial array of (sigma, w, a, b, offset) parameters
    :param fit_bounds: Bounds imposed on the parameters during the fitting procedure.
    :return: The optimal parameters and r-squared values of the fit result: sigma, w, a, b, offset, rsq_amps, rsq_lags
    '''
    popt_amplag = minimize(f_amplags, params_init, jac=True, args=(amps, lags, itds),
                           bounds=fit_bounds,
                           method='L-BFGS-B')

    sigma, w, a, b, offset = popt_amplag.x
    pred_amp, pred_lag = \
        f_amplags(popt_amplag.x, amps, lags, itds, return_pred=True)

    rsq_amps = 1 - (np.sum((amps - pred_amp) ** 2) / np.sum((amps - np.mean(amps)) ** 2))
    rsq_lags = 1 - (np.sum((lags - pred_lag) ** 2) / np.sum((lags - np.mean(lags)) ** 2))

    return sigma, w, a, b, offset, rsq_amps, rsq_lags


if __name__ == '__main__':
    '''
    In this example, the model is fitted to Human BIC data digitized from (Riedel and Kollmeier, 2006).
    '''
    amps = np.array([1., 0.97, 0.91, 0.84, 0.77, 0.71, 0.65, 0.61, 0.55, 0.5, 0.43])
    lags = np.array([0., 0.05849037, 0.12989312, 0.21203919, 0.29849556,
                     0.40000556, 0.49720525, 0.58796266, 0.67225925, 0.77161873,
                     0.84947303])
    itds = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

    # The fitting procedure per se
    sigma, w, a, b, offset, rsq_amps, rsq_lags = fit_ei_model(itds, amps, lags)

    print rsq_lags, rsq_amps
