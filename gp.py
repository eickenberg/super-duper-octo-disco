"""Gaussian process for hrf estimaton (sandbox)
This implementation is based on scikit learn and Michael's implementation

"""
# TODO add hrf as a mean for the gp
# TODO add more kernels

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_X_y, check_random_state
# from sklearn.metrics import pairwise_kernels
from scipy.sparse import coo_matrix
from nistats.experimental_paradigm import check_paradigm
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import fmin_cobyla
from scipy.linalg import cholesky, LinAlgError
from nistats.hemodynamic_models import spm_hrf, glover_hrf
import warnings

# from joblib import delayed, Parallel
# MACHINE_EPS = np.finfo(np.double).eps


def _rbf_kernel(X, Y, gamma=1., tau=1.):
    X, Y = map(np.atleast_1d, (X, Y))
    diff_squared = (X.reshape(-1, 1) - Y.reshape(-1)) ** 2

    return np.exp(-diff_squared / gamma) * tau


def _get_design_from_hrf_measures(hrf_measures, beta_indices):
    event_names = np.unique(np.concatenate(beta_indices)).astype('int')

    design = np.zeros([len(beta_indices), len(event_names)])
    pointer = 0

    for beta_ind, row in zip(beta_indices, design):
        measures = hrf_measures[pointer:pointer + len(beta_ind)]
        for i, name in enumerate(event_names):
            row[i] = measures[beta_ind == name].sum()
        pointer += len(beta_ind)
    return design


def _get_hrf_measurements(paradigm, hrf_length=32., t_r=2, time_offset=10):
    """This function:
    Parameters
    ----------
    paradigm : s
    hrf_length : float
    t_r : float
    time_offset : float

    Returns
    -------
    hrf_measurement_points : list of list
    visible events : s
    alphas : s
    beta_indices : s
    unique_events : s
    """
    names, onsets, durations, modulation = check_paradigm(paradigm)
    frame_times = np.arange(0, onsets.max() + time_offset, t_r)

    time_differences = frame_times[:, np.newaxis] - onsets
    scope_masks = (time_differences > 0) & (time_differences < hrf_length)
    belong_to_measurement, which_event = np.where(scope_masks)

    unique_events, event_type_indices = np.unique(names, return_inverse=True)

    hrf_measurement_points = [list() for _ in range(len(frame_times))]
    alphas = [list() for _ in range(len(frame_times))]
    beta_indices = [list() for _ in range(len(frame_times))]
    visible_events = [list() for _ in range(len(frame_times))]

    for frame_id, event_id in zip(belong_to_measurement, which_event):
        hrf_measurement_points[frame_id].append(time_differences[frame_id,
                                                                 event_id])
        alphas[frame_id].append(modulation[event_id])
        beta_indices[frame_id].append(event_type_indices[event_id])
        visible_events[frame_id].append(event_id)

    return (hrf_measurement_points, visible_events, alphas, beta_indices,
            unique_events)


def _alpha_weighted_kernel(hrf_measurement_points, alphas,
                           evaluation_points=None, gamma=1., tau=1.,
                           return_eval_cov=False):
    """This function computes the kernel matrix of all measurement points,
    potentially redundantly per measurement points, just to be sure we
    identify things correctly afterwards.

    If evaluation_points is set to None, then the original
    hrf_measurement_points will be used
    """
    hrf_measurement_points = np.concatenate(hrf_measurement_points)

    if evaluation_points is None:
        evaluation_points = hrf_measurement_points

    alphas = np.concatenate(alphas)
    alpha_weight = alphas[:, np.newaxis] * alphas

    # XXX Call a general kernel
    K = _rbf_kernel(hrf_measurement_points, hrf_measurement_points,
                           gamma=gamma, tau=tau)
    K_cross = _rbf_kernel(evaluation_points, hrf_measurement_points,
                              gamma=gamma, tau=tau)

    pre_cov = K * alpha_weight
    pre_cross_cov = K_cross * alphas

    if return_eval_cov:
        K_22 = _rbf_kernel(evaluation_points, evaluation_points,
                           gamma=gamma, tau=tau)
        return pre_cov, pre_cross_cov, K_22

    return pre_cov, pre_cross_cov


def _get_hrf_values_from_betas(ys, beta_values, alpha_weighted_cov,
                               alpha_weighted_cross_cov, beta_indices,
                               sigma_noise, K_22=None):

    col_coordinates = np.concatenate(
        [i * np.ones(len(beta_ind)) for i, beta_ind in enumerate(beta_indices)])

    all_betas = beta_values[np.concatenate(beta_indices).astype('int')]

    row_coordinates = np.arange(len(all_betas))
    collapser = coo_matrix((all_betas, (row_coordinates, col_coordinates)),
    shape = (len(all_betas), len(beta_indices))).tocsc()

    K = collapser.T.dot(collapser.T.dot(alpha_weighted_cov).T).T
    K_cross = collapser.T.dot(alpha_weighted_cross_cov.T).T  # again

    K_reg = K + np.eye(K.shape[0]) * sigma_noise ** 2

    inv_K_reg = np.linalg.inv(K_reg)
    alpha = inv_K_reg.dot(ys)

    mu_bar = K_cross.dot(inv_K_reg.dot(ys))

    if K_22 is not None:
        var_bar = np.diag(K_22) - np.einsum(
            'ij, ji -> i', K_cross, np.dot(inv_K_reg, K_cross.T))
        return (mu_bar, var_bar),  (alpha, K_reg, inv_K_reg, K_cross)

    return mu_bar, (alpha, K_reg, inv_K_reg, K_cross)


# XXX change the name of this function
def _get_data(ys, beta_values, beta_indices, hrf_measurement_points, alphas,
              evaluation_points=None, gamma=1., tau=1., sigma_noise=0.0001):
    """Find the HRF given the measurements
    """
    # weighted kernels
    alpha_weighted_cov, alpha_weighted_cross_cov, K_22 = \
        _alpha_weighted_kernel(hrf_measurement_points, alphas,
                               evaluation_points=evaluation_points,
                               gamma=gamma, tau=tau, return_eval_cov=True)

    (mu, var), (alpha, K_reg, inv_K_reg, K_cross) =  _get_hrf_values_from_betas(
        ys, beta_values, alpha_weighted_cov, alpha_weighted_cross_cov,
        beta_indices, sigma_noise=sigma_noise, K_22=K_22)

    return (mu, var), (alpha, K_reg, inv_K_reg, K_cross)


def _get_betas_and_hrf(ys, betas, pre_cov, pre_cross_cov, beta_indices,
                       sigma_noise, n_iter=10, K_22=None):
    """Alternate optimization: Find HRF, build a new design matrix and repeat
    """
    all_hrf_values = []
    all_hrf_var = []
    all_designs = []
    all_betas = []
    for i in range(n_iter):
        values, _ = _get_hrf_values_from_betas(ys, betas, pre_cov,
                                               pre_cross_cov, beta_indices,
                                               sigma_noise, K_22=K_22)
        hrf_values, hrf_var = values
        design = _get_design_from_hrf_measures(hrf_values, beta_indices)
        # Least squares estimation
        betas = np.linalg.pinv(design).dot(ys)

        all_hrf_values.append(hrf_values)
        all_hrf_var.append(hrf_var)
        all_designs.append(design)
        all_betas.append(betas)

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return (betas, hrf_values, hrf_var, all_hrf_values, all_hrf_var, all_designs,
            all_betas)


## Fitness function ###########################################################
def get_loglikelihood(ys, alpha, K_reg, inv_K_reg):
    (sign, logdet) = np.linalg.slogdet(K_reg)
    print logdet
    loglikelihood = -0.5 * ys.dot(alpha) \
        -0.5 * (sign * logdet) - (ys.shape[0] / 2) * np.log(2*np.pi)
    return loglikelihood


### loglikelihood
def f(params, *args):
    gamma, sigma_noise, tau = params
    (ys, beta_values, beta_indices, hrf_measurement_points, alphas,
     evaluation_points) = args

    _, (alpha, K_reg, inv_K_reg, K_cross) = _get_data(
        ys, beta_values, beta_indices, hrf_measurement_points, alphas,
        evaluation_points, gamma=gamma, tau=tau, sigma_noise=sigma_noise)

    return - get_loglikelihood(ys, alpha, K_reg, inv_K_reg)


# adding a positivity constrain
def constr0(params, *args):
    return params[0]


def constr1(params, *args):
    return params[1]


def constr2(params, *args):
    return params[2]


def get_hrf_fit(ys, hrf_measurement_points, visible_events, alphas, beta_indices,
                initial_beta, unique_events, gamma_0, tau_0, sigma_noise_0,
                evaluation_points=None, max_iter=20, n_iter=20):

    betas = initial_beta.copy()
    # Finding the parameters ##################################################
    # Maximizing the log-likelihood (gradient based optimization)
    args = (ys, betas, beta_indices, hrf_measurement_points, alphas,
            evaluation_points)
    params = fmin_cobyla(f, [gamma_0, sigma_noise_0, tau_0],
                         [constr0, constr1, constr2], args=args,
                         maxfun=max_iter, rhoend=1e-7)

    gamma_, sigma_noise_, tau_ = params

    pre_cov, pre_cross_cov, K_22 = _alpha_weighted_kernel(
        hrf_measurement_points, alphas, evaluation_points=evaluation_points,
        gamma=gamma_, tau=tau_, return_eval_cov=True)

    (betas, hrf_values, hrf_var, all_hrf_values, all_hrf_var, all_designs, all_betas) = \
        _get_betas_and_hrf(ys, betas, pre_cov, pre_cross_cov, beta_indices,
                           sigma_noise_, n_iter=n_iter, K_22=K_22)

    return (betas, (hrf_measurement_points, hrf_values, hrf_var), all_hrf_values,
            all_designs, all_betas, params)


class SuperDuperGP(BaseEstimator):

    def __init__(self, hrf_length=32., t_r=2, time_offset=10,
                 modulation=None, sigma_noise_0=0.001, tau_0=1., gamma_0=1.,
                 copy=True, fmin_max_iter=10, n_iter=10, hrf_model=None,
                 normalize_y=True):
        self.t_r = t_r
        self.hrf_length = hrf_length
        self.modulation = modulation
        self.time_offset = time_offset
        self.sigma_noise_0 = sigma_noise_0
        self.gamma_0 = gamma_0
        self.tau_0 = tau_0
        self.copy = copy
        self.fmin_max_iter = fmin_max_iter
        self.n_iter = n_iter
        self.hrf_model = hrf_model
        self.normalize_y = normalize_y

    def fit(self, ys, paradigm, initial_beta=None):

        ys = np.atleast_1d(ys)
        if self.normalize_y:
            self.y_train_mean = np.mean(ys, axis=0)
            ys = ys - self.y_train_mean
        else:
            self.y_train_mean = np.zeros(1)

        hrf_measurement_points, visible_events, alphas, beta_indices, unique_events = \
            _get_hrf_measurements(paradigm, hrf_length=self.hrf_length,
                                  t_r=self.t_r, time_offset=self.time_offset)
        if initial_beta is None:
            initial_beta = np.ones(len(unique_events))

        output = get_hrf_fit(ys, hrf_measurement_points, visible_events,
                             alphas, beta_indices, initial_beta, unique_events,
                             gamma_0=self.gamma_0, tau_0=self.tau_0,
                             max_iter=self.fmin_max_iter, n_iter=self.n_iter,
                             sigma_noise_0=self.sigma_noise_0)

        hrf_measurement_points = np.concatenate(output[1][0])
        order = np.argsort(hrf_measurement_points)

        hrf_var = output[1][2][order]
        hx, hy = hrf_measurement_points[order], output[1][1][order]

        self.params_ = output[-1]
        self.hrf_measurement_points_ = hrf_measurement_points

        return hx, hy, hrf_var


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data_generator import generate_spikes_time_series

    plt.close('all')

    seed = 42
    rng = check_random_state(seed)

    # Generate simulated data
    n_events = 200
    n_blank_events = 50
    event_spacing = 6
    t_r = 2
    jitter_min, jitter_max = -1, 1
    event_types = ['evt_1', 'evt_2', 'evt_3', 'evt_4', 'evt_5', 'evt_6']
    sigma_noise = .1

    paradigm, design, modulation, measurement_time = \
        generate_spikes_time_series(n_events=n_events,
                                    n_blank_events=n_blank_events,
                                    event_spacing=event_spacing, t_r=t_r,
                                    return_jitter=True, jitter_min=jitter_min,
                                    jitter_max=jitter_max,
                                    event_types=event_types, period_cut=64,
                                    time_offset=10, modulation=None, seed=seed)
    ###########################################################################
    # GP parameters
    hrf_length = 24
    gamma_0 = 1.
    tau_0 = 1.
    sigma_noise_0 = 0.01
    time_offset = 10
    fmin_max_iter = 20
    n_iter = 10

    gp = SuperDuperGP(hrf_length=hrf_length, modulation=modulation,
                      gamma_0=gamma_0, fmin_max_iter=fmin_max_iter, tau_0=tau_0,
                      sigma_noise_0=sigma_noise_0, time_offset=time_offset,
                      n_iter=n_iter)

    design = design[event_types].values  # forget about drifts for the moment
    beta = rng.randn(len(event_types))

    ys = design.dot(beta) + rng.randn(design.shape[0]) * sigma_noise ** 2

    hx, hy, hrf_var = gp.fit(ys, paradigm)

    plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                     hy + 1.96 * np.sqrt(hrf_var), alpha=0.1)
    plt.plot(hx, hy)
    plt.show()


# # This is just temporal
# def _get_hrf_model(hrf_model, hrf_length):
#     if hrf_model is None:
#         hrf_0 = 0
#     elif hrf_model == 'spm':
#         hrf_0 = spm_hrf(1., 1., time_length=hrf_length)
#     elif hrf_model == 'glover':
#         hrf_0 = glover_hrf(1., 1., time_length=hrf_length)
#     else:
#         hrf_0 = 0
#         warnings.warn("The HRF model is not recognized, setting it to None")
#     return hrf_0


