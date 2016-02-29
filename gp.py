"""Gaussian process for hrf estimaton (sandbox)
This implementation is based on scikit learn and Michael's implementation

"""

# TODO add hrf as a mean for the gp
# TODO add hyperparameter optimization

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_random_state
# from sklearn.utils.validation import check_is_fitted
from scipy.sparse import coo_matrix
from nistats.experimental_paradigm import check_paradigm
# from joblib import delayed, Parallel



def _rbf_kernel(X, Y, gamma=1., tau=1.):
    X, Y = map(np.atleast_1d, (X, Y))
    diff_squared = (X.reshape(-1, 1) - Y.reshape(-1)) ** 2

    return tau * np.exp(-diff_squared / gamma)


def _der_rbf_kernel(X, Y, gamma=1., tau=1.):
    X, Y = map(np.atleast_1d, (X, Y))
    diff_squared = (X.reshape(-1, 1) - Y.reshape(-1)) ** 2

    return tau * np.exp(-diff_squared / gamma) * (diff_squared / gamma ** 2)


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
                           evaluation_points=None, gamma=1.):
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

    Sigma_22 = _rbf_kernel(hrf_measurement_points, hrf_measurement_points,
                           gamma=gamma)
    Sigma_cross = _rbf_kernel(evaluation_points, hrf_measurement_points,
                              gamma=gamma)

    pre_cov = Sigma_22 * alpha_weight
    pre_cross_cov = Sigma_cross * alphas

    return pre_cov, pre_cross_cov


def _der_alpha_weighted_kernel(hrf_measurement_points, alphas,
                               evaluation_points=None, gamma=1.):

    hrf_measurement_points = np.concatenate(hrf_measurement_points)

    if evaluation_points is None:
        evaluation_points = hrf_measurement_points

    alphas = np.concatenate(alphas)
    alpha_weight = alphas[:, np.newaxis] * alphas

    der_Sigma_22 = _der_rbf_kernel(hrf_measurement_points,
                                   hrf_measurement_points, gamma=gamma)

    der_cov = der_Sigma_22 * alpha_weight

    return der_cov


def _der_marginal_likelihood(y, beta_values, beta_indices,
                             hrf_measurement_points, alphas,
                             evaluation_points=None, gamma=1.,
                             noise_level=0.0001):
    # weighted kernel
    alpha_weighted_cov, _ = _alpha_weighted_kernel(
        hrf_measurement_points, alphas, evaluation_points=evaluation_points,
        gamma=gamma)
    # derivate
    der_alpha_weighted_cov = _der_alpha_weighted_kernel(
        hrf_measurement_points, alphas, evaluation_points=evaluation_points,
        gamma=gamma)

    col_coordinates = np.concatenate(
        [i * np.ones(len(beta_ind)) for i, beta_ind in enumerate(beta_indices)])

    all_betas = beta_values[np.concatenate(beta_indices).astype('int')]

    row_coordinates = np.arange(len(all_betas))
    collapser = coo_matrix((all_betas, (row_coordinates, col_coordinates)),
    shape = (len(all_betas), len(beta_indices))).tocsc()

    K = collapser.T.dot(collapser.T.dot(alpha_weighted_cov).T).T
    der_K = collapser.T.dot(collapser.T.dot(der_alpha_weighted_cov).T).T

    inv_reg_K = np.linalg.inv(K + np.eye(K.shape[0]) * noise_level)
    alpha = inv_reg_K.dot(y)

    # optimize noise level, GCV

    grad = 0.5 * np.trace((np.dot(alpha.reshape(-1, 1),
                                  alpha.reshape(1, -1)) - inv_reg_K).dot(der_K))

    return grad


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


def _get_hrf_values_from_betas(y, beta_values, alpha_weighted_kernel_cov,
                               alpha_weighted_kernel_cross_cov,
                               beta_indices, noise_level, Sigma_22=None):

    col_coordinates = np.concatenate(
        [i * np.ones(len(beta_ind)) for i, beta_ind in enumerate(beta_indices)])

    all_betas = beta_values[np.concatenate(beta_indices).astype('int')]

    row_coordinates = np.arange(len(all_betas))
    collapser = coo_matrix((all_betas, (row_coordinates, col_coordinates)),
    shape = (len(all_betas), len(beta_indices))).tocsc()
    cov = collapser.T.dot(collapser.T.dot(
        alpha_weighted_kernel_cov).T).T   # awful, I know. sparse matrix problem

    cross_cov = collapser.T.dot(alpha_weighted_kernel_cross_cov.T).T  # again
    inv_reg_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * noise_level)

    mu_bar = cross_cov.dot(inv_reg_cov.dot(y))

    if Sigma_22 is not None:
        var_bar = np.diag(Sigma_22) - np.einsum(
            'ij, ji -> i', cross_cov, np.dot(inv_reg_cov, cross_cov.T))
        return mu_bar, var_bar

    return mu_bar


def get_hrf_gp(ys, evaluation_points, initial_beta, paradigm, hrf_length, t_r,
               time_offset, gamma, max_iter, noise_level, n_iter=100,
               step_size=0.05, verbose=True):

    hrf_measurement_points, visible_events, alphas, beta_indices, unique_events = \
        _get_hrf_measurements(paradigm, hrf_length=hrf_length, t_r=t_r,
                              time_offset=time_offset)
    if initial_beta is None:
        initial_beta = np.ones(len(unique_events))

    betas = initial_beta.copy()

    # Maximizing the log-likelihoo (gradient based optimization)
    gamma_ = gamma
    for i in range(n_iter):
        grad = _der_marginal_likelihood(ys, betas, beta_indices,
                                        hrf_measurement_points, alphas,
                                        evaluation_points=evaluation_points,
                                        gamma=gamma_, noise_level=noise_level)
        gamma_ += step_size * grad
        gamma_ = np.abs(gamma_)

        if verbose:
            print "iter: %s gamma: %s" % (i, gamma_)

    pre_cov, pre_cross_cov = \
        _alpha_weighted_kernel(hrf_measurement_points, alphas,
                               evaluation_points=evaluation_points,
                               gamma=gamma_)
    all_hrf_values = []
    all_designs = []
    all_betas = []

    for i in range(max_iter):
        hrf_values = _get_hrf_values_from_betas(ys, betas, pre_cov,
                                                pre_cross_cov, beta_indices,
                                                noise_level)
        design = _get_design_from_hrf_measures(hrf_values, beta_indices)
        # Least squares estimation
        betas = np.linalg.pinv(design).dot(ys)

        all_hrf_values.append(hrf_values)
        all_designs.append(design)
        all_betas.append(betas)

    return (betas, (hrf_measurement_points, hrf_values), all_hrf_values,
            all_designs, all_betas)


class SuperDuperGP(BaseEstimator):

    def __init__(self, paradigm, hrf_length=32., t_r=2, time_offset=10,
                 modulation=None, noise_level=0, tau=1., gamma=1., copy=True,
                 max_iter=10, boundary_conditions=True):
        self.paradigm = paradigm
        self.t_r = t_r
        self.hrf_length = hrf_length
        self.modulation = modulation
        self.time_offset = time_offset
        self.noise_level = noise_level
        self.gamma = gamma
        self.copy = copy
        self.tau = tau
        self.max_iter = max_iter

    def fit(self, ys, evaluation_points=None, initial_beta=None):

        output = get_hrf_gp(ys, evaluation_points=evaluation_points,
                            initial_beta=initial_beta, paradigm=self.paradigm,
                            hrf_length=self.hrf_length, t_r=self.t_r,
                            noise_level=self.noise_level, gamma=self.gamma,
                            time_offset=self.time_offset,
                            max_iter=self.max_iter)

        hrf_measurement_points = np.concatenate(output[1][0])
        order = np.argsort(hrf_measurement_points)
        hx, hy = hrf_measurement_points[order], output[1][1][order]

        return hx, hy


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
    noise_level = .05

    paradigm, design, modulation, measurement_time = \
        generate_spikes_time_series(n_events=n_events,
                                    n_blank_events=n_blank_events,
                                    event_spacing=event_spacing, t_r=t_r,
                                    return_jitter=True, jitter_min=jitter_min,
                                    jitter_max=jitter_max,
                                    event_types=event_types, period_cut=64,
                                    time_offset=10, modulation=None, seed=seed)
    # GP parameters
    hrf_length = 24
    gamma = 1.
    time_offset = 10
    max_iter = 10

    gp = SuperDuperGP(paradigm, hrf_length=hrf_length, modulation=modulation,
                      gamma=gamma, max_iter=max_iter,
                      noise_level=noise_level, time_offset=time_offset)

    design = design[event_types].values  # forget about drifts for the moment
    beta = rng.randn(len(event_types))

    ys = design.dot(beta) + rng.randn(design.shape[0]) * noise_level

    hx, hy = gp.fit(ys)

    plt.plot(hx, hy)
    plt.show()


