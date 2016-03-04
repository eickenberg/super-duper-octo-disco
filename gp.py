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
from scipy.linalg import (cholesky, LinAlgError, solve, lstsq, cho_solve,
                          solve_triangular)
from nistats.hemodynamic_models import spm_hrf, glover_hrf
import warnings

# from joblib import delayed, Parallel
# MACHINE_EPS = np.finfo(np.double).eps


def _kernel(X, Y, theta=[1., 1.]):
    gamma, tau = theta
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
                           evaluation_points=None, theta=[1., 1.],
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

    K = _kernel(hrf_measurement_points, hrf_measurement_points, theta=theta)
    K_cross = _kernel(evaluation_points, hrf_measurement_points, theta=theta)

    pre_cov = K * alpha_weight
    pre_cross_cov = K_cross * alphas

    if return_eval_cov:
        K_22 = _kernel(evaluation_points, evaluation_points, theta=theta)
        return pre_cov, pre_cross_cov, K_22

    return pre_cov, pre_cross_cov


def _get_gp_kernels(beta_values, alpha_weighted_cov,
                    alpha_weighted_cross_cov, beta_indices):

    col_coordinates = np.concatenate(
        [i * np.ones(len(beta_ind)) for i, beta_ind in enumerate(beta_indices)])

    all_betas = beta_values[np.concatenate(beta_indices).astype('int')]

    row_coordinates = np.arange(len(all_betas))
    collapser = coo_matrix((all_betas, (row_coordinates, col_coordinates)),
    shape = (len(all_betas), len(beta_indices))).tocsc()

    K = collapser.T.dot(collapser.T.dot(alpha_weighted_cov).T).T
    K_cross = collapser.T.dot(alpha_weighted_cross_cov.T).T  # again

    return K, K_cross


def _get_hrf_values_from_betas(ys, beta_values, alpha_weighted_cov,
                               alpha_weighted_cross_cov, beta_indices,
                               sigma_noise, K_22=None):

    K, K_cross =  _get_gp_kernels(beta_values, alpha_weighted_cov,
                                  alpha_weighted_cross_cov, beta_indices)

    # Adding noise to the diagonal (Ridge)
    K[np.diag_indices_from(K)] += sigma_noise ** 2

    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), ys)
    mu_bar = K_cross.dot(alpha)

    if K_22 is not None:
        L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        var_bar = np.diag(K_22) - np.einsum("ki,kj,ij->k", K_cross, K_cross,
                                            K_inv)

        check_negative_var = var_bar < 0.
        if np.any(check_negative_var):
            var_bar[check_negative_var] = 0.

        return mu_bar, var_bar

    return mu_bar




# # XXX change the name of this function
# def _get_data(ys, beta_values, beta_indices, hrf_measurement_points, alphas,
#               evaluation_points=None, theta=[1., 1.], sigma_noise=0.0001):
#     """Find the HRF given the measurements
#     """
#     # weighted kernels
#     alpha_weighted_cov, alpha_weighted_cross_cov, K_22 = \
#         _alpha_weighted_kernel(hrf_measurement_points, alphas,
#                                evaluation_points=evaluation_points,
#                                theta=theta, return_eval_cov=True)
#     #
#     (mu, var), (alpha, K_reg, inv_K_reg, K_cross) =  _get_hrf_values_from_betas(
#         ys, beta_values, alpha_weighted_cov, alpha_weighted_cross_cov,
#         beta_indices, sigma_noise=sigma_noise, K_22=K_22)

#     return (mu, var), (alpha, K_reg, inv_K_reg, K_cross)


def _get_betas_and_hrf(ys, betas, pre_cov, pre_cross_cov, beta_indices,
                       sigma_noise, n_iter=10, K_22=None):
    """Alternate optimization: Find HRF, build a new design matrix and repeat
    """
    all_hrf_values = []
    all_hrf_var = []
    all_designs = []
    all_betas = []
    for i in range(n_iter):
        hrf_values, hrf_var = _get_hrf_values_from_betas(ys, betas, pre_cov,
                                                         pre_cross_cov,
                                                         beta_indices,
                                                         sigma_noise, K_22=K_22)
        design = _get_design_from_hrf_measures(hrf_values, beta_indices)
        # Least squares estimation
        betas = np.linalg.pinv(design).dot(ys)

        all_hrf_values.append(hrf_values)
        all_hrf_var.append(hrf_var)
        all_designs.append(design)
        all_betas.append(betas)

    return (betas, hrf_values, hrf_var, all_hrf_values, all_hrf_var, all_designs,
            all_betas)


## Fitness function ###########################################################
# ### loglikelihood
# def f(params, *args):
#     # theta = params[:-1]
#     # sigma_noise = params[-1]

#     theta = params

#     (ys, beta_values, beta_indices, hrf_measurement_points, alphas,
#      evaluation_points, sigma_noise) = args

#     _, (alpha, K_reg, inv_K_reg, K_cross) = _get_data(
#         ys, beta_values, beta_indices, hrf_measurement_points, alphas,
#         evaluation_points, theta=theta, sigma_noise=sigma_noise)

#     return - get_loglikelihood(ys, alpha, K_reg, inv_K_reg)


# # adding a positivity constrain
# def constr0(params, *args):
#     return params[0]


# def constr1(params, *args):
#     return params[1]


def get_hrf_fit(ys, hrf_measurement_points, visible_events, alphas, beta_indices,
                initial_beta, unique_events, theta_0, sigma_noise_0,
                optimize=False, evaluation_points=None, max_iter=20, n_iter=20):

    betas = initial_beta.copy()

    # # Finding the parameters ##################################################
    # if optimize:
    #     # Maximizing the log-likelihood (gradient based optimization)
    #     args = (ys, betas, beta_indices, hrf_measurement_points, alphas,
    #             evaluation_points, sigma_noise)
    #     # theta_0.append(sigma_noise_0)
    #     params = fmin_cobyla(f, theta_0,
    #                         [constr0, constr1], args=args,
    #                         maxfun=max_iter, rhoend=1e-7)
    #     theta_ = params
    #     sigma_noise_ = sigma_noise
    #     # theta_ = params[:-1]
    #     # sigma_noise_ = params[-1]
    # else:
    theta_ = theta_0
    sigma_noise_ = sigma_noise_0

    pre_cov, pre_cross_cov, K_22 = _alpha_weighted_kernel(
        hrf_measurement_points, alphas, evaluation_points=evaluation_points,
        theta=theta_, return_eval_cov=True)

    (betas, hrf_values, hrf_var, all_hrf_values, all_hrf_var, all_designs, all_betas) = \
        _get_betas_and_hrf(ys, betas, pre_cov, pre_cross_cov, beta_indices,
                           sigma_noise_, n_iter=n_iter, K_22=K_22)

    return (betas, (hrf_measurement_points, hrf_values, hrf_var), all_hrf_values,
            all_designs, all_betas)


class SuperDuperGP(BaseEstimator):

    def __init__(self, hrf_length=32., t_r=2, time_offset=10,
                 modulation=None, sigma_noise_0=0.001, theta_0=[1., 1.],
                 copy=True, fmin_max_iter=10, n_iter=10, hrf_model=None,
                 normalize_y=False, optimize=False):
        self.t_r = t_r
        self.hrf_length = hrf_length
        self.modulation = modulation
        self.time_offset = time_offset
        self.sigma_noise_0 = sigma_noise_0
        self.theta_0 = theta_0
        self.copy = copy
        self.fmin_max_iter = fmin_max_iter
        self.n_iter = n_iter
        self.hrf_model = hrf_model
        self.normalize_y = normalize_y
        self.optimize = optimize

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
                             theta_0=self.theta_0, optimize=self.optimize,
                             max_iter=self.fmin_max_iter, n_iter=self.n_iter,
                             sigma_noise_0=self.sigma_noise_0)

        hrf_measurement_points = np.concatenate(output[1][0])
        order = np.argsort(hrf_measurement_points)

        hrf_var = output[1][2][order]
        hx, hy = hrf_measurement_points[order], output[1][1][order]

        # self.params_ = output[-1]
        # self.hrf_measurement_points_ = hrf_measurement_points

        return hx, hy, hrf_var


    def predict(self, paradigm):
        """
        """
        check_is_fitted(self, "params_")

        pass

    def scorer(self, paradigm, ys):
        """Please put here the scorer
        """
        pass



    def get_loglikelihood(self, ys, kernel, sigma_noise=0.001, theta=None, eval_gradient=None):
        """
        ys : array-like
        kernel: kernel function
        sigma_noise: float
        theta: list of kernel's parameters
        """
        # evaluate the kernel on the training data
        #  XXX This is not working, we need to change the parameters of the kernel
        #  Please wait a minute
        K = kernel

        # Adding noise to the diagonal (Ridge)
        K[np.diag_indices_from(K)] += sigma_noise ** 2

        try:
            L = cholesky(kernel, lower=True)
            alpha = cho_solve((L, True), ys)

        except LinAlgError:
            loglikelihood = -np.inf
            return loglikelihood


        loglikelihood_dims = -0.5 * np.einsum("ik, jk->k", ys, alpha)
        loglikelihood_dims -= np.log(np.diag(L)).sum()
        loglikelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        loglikelihood = loglikelihood_dims.sum(-1)

        # if eval_gradient:
        #     tmp = np.einsum("ik, jk->ijk", alpha, alpha)
        #     tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]

        #     loglikelihood_gradient_dims = \
        #         0.5 * np.einsum("ijl, ijk->kl", tmp, K_gradient)
        #     loglikelihood_gradient = loglikelihood_gradient_dims.sum(-1)

        return loglikelihood


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
    sigma_noise = .01

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
    theta_0 = [10., 1.]
    sigma_noise_0 = sigma_noise
    time_offset = 10
    fmin_max_iter = 20
    n_iter = 10

    gp = SuperDuperGP(hrf_length=hrf_length, modulation=modulation,
                      theta_0=theta_0, fmin_max_iter=fmin_max_iter,
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


