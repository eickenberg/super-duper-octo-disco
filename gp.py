"""Gaussian process for hrf estimaton (sandbox)
This implementation is based on scikit learn and Michael's implementation

"""

# TODO add more kernels
# TODO add hrf as a men for the gp
# TODO add hyperparameter optimization

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import coo_matrix
from nistats.experimental_paradigm import check_paradigm
from joblib import delayed, Parallel


def _get_design_from_hrf_measures(hrf_measures, beta_indices):

    event_names = np.unique(np.concatenate(beta_indices)).astype(np.int)

    design = np.zeros([len(beta_indices), len(event_names)])
    pointer = 0

    for beta_ind, row in zip(beta_indices, design):
        measures = hrf_measures[pointer:pointer + len(beta_ind)]
        for i, name in enumerate(event_names):
            row[i] = measures[beta_ind == name].sum()

        pointer += len(beta_ind)
    return design



# TODO add a function with more kernels
def _get_kernel(X, Y=None, kernel='linear', gamma=1., degree=0, coef0=0, tau=1.):

    X = np.atleast_1d(X)

    if callable(kernel):
        params = kernel_params or {}
    else:
        params = {"gamma": gamma, "degree": degree, "coef0": coef0}

    if Y is None:
        diff_squared = (X.reshape(-1, 1) - X.reshape(-1)) ** 2
    else:
        diff_squared = (X.reshape(-1, 1) - Y.reshape(-1)) ** 2

    return tau * np.exp(-diff_squared / gamma)



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


def _get_alpha_weighted_kernel(hrf_measurement_points, alphas,
                               evaluation_points=None, kernel='linear',
                               gamma=1., coef0=0, degree=1,
                               boundary_conditions=True):
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

    Sigma_22 = _get_kernel(hrf_measurement_points, kernel=kernel, gamma=gamma,
                           degree=degree, coef0=coef0)
    Sigma_cross = _get_kernel(evaluation_points, hrf_measurement_points,
                              kernel=kernel, gamma=gamma, degree=degree,
                              coef0=coef0)

    pre_cov = Sigma_22 * alpha_weight
    pre_cross_cov = Sigma_cross * alphas

    return pre_cov, pre_cross_cov


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
               time_offset, kernel, gamma, coef0, degree, max_iter, noise_level,
               boundary_conditions):

    hrf_measurement_points, visible_events, alphas, beta_indices, unique_events = \
        _get_hrf_measurements(paradigm, hrf_length=hrf_length, t_r=t_r,
                              time_offset=time_offset)
    if initial_beta is None:
        initial_beta = np.ones(len(unique_events))

    betas = initial_beta.copy()
    pre_cov, pre_cross_cov = \
        _get_alpha_weighted_kernel(hrf_measurement_points, alphas,
                                   evaluation_points=evaluation_points,
                                   kernel=kernel, gamma=gamma, coef0=coef0,
                                   degree=degree,
                                   boundary_conditions=boundary_conditions)
    all_hrf_values = []
    all_designs = []
    all_betas = []

    for i in range(max_iter):
        # print betas
        hrf_values = _get_hrf_values_from_betas(ys, betas, pre_cov,
                                                pre_cross_cov, beta_indices,
                                                noise_level)
        design = _get_design_from_hrf_measures(hrf_values, beta_indices)
        # Least squares estimation
        betas = np.linalg.pinv(design).dot(ys)

        all_hrf_values.append(hrf_values)
        all_designs.append(design)
        all_betas.append(betas)

    return betas, (hrf_measurement_points, hrf_values), all_hrf_values, all_designs, all_betas


class SuperDuperGP(BaseEstimator):

    def __init__(self, paradigm, hrf_length=32., t_r=2, time_offset=10,
                 modulation=None, noise_level=0, tau=1., kernel="linear",
                 gamma=1., degree=2, coef0=0, kernel_params=None, copy=True,
                 max_iter=10, boundary_conditions=True):
        self.paradigm = paradigm
        self.t_r = t_r
        self.hrf_length = hrf_length
        self.modulation = modulation
        self.time_offset = time_offset
        self.noise_level = noise_level
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.boundary_conditions = boundary_conditions
        self.copy = copy
        self.tau = tau
        self.max_iter = max_iter


    def fit(self, ys, evaluation_points=None, initial_beta=None):

        output = get_hrf_gp(ys, evaluation_points=evaluation_points,
                            initial_beta=initial_beta, paradigm=self.paradigm,
                            hrf_length=self.hrf_length, t_r=self.t_r,
                            noise_level=self.noise_level, kernel=self.kernel,
                            coef0=self.coef0, degree=self.degree, gamma=self.gamma,
                            time_offset=self.time_offset, max_iter=self.max_iter,
                            boundary_conditions=self.boundary_conditions)
        return output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data_generator import generate_spikes_time_series

    seed = 42
    rng = check_random_state(42)

    n_events = 200
    n_blank_events = 50
    event_spacing = 6
    t_r = 2
    jitter_min, jitter_max = -1, 1
    event_types = ['evt_1', 'evt_2', 'evt_3', 'evt_4', 'evt_5', 'evt_6']
    noise_level = .01

    paradigm, design, modulation, measurement_time = \
        generate_spikes_time_series(n_events=n_events,
                                    n_blank_events=n_blank_events,
                                    event_spacing=event_spacing, t_r=t_r,
                                    return_jitter=True, jitter_min=jitter_min,
                                    jitter_max=jitter_max,
                                    event_types=event_types, period_cut=64,
                                    time_offset=10, modulation=None, seed=seed)

    hrf_length = 24
    gamma = 10.
    time_offset = 10
    max_iter = 10
    boundary_conditions = True

    gp = SuperDuperGP(paradigm, hrf_length=hrf_length, modulation=modulation,
                      kernel='rbf', gamma=gamma, max_iter=max_iter,
                      noise_level=noise_level, time_offset=time_offset,
                      boundary_conditions=boundary_conditions)

    design = design[event_types].values  # forget about drifts for the moment
    beta = rng.randn(len(event_types))

    ys = design.dot(beta) + rng.randn(design.shape[0]) * noise_level

    output = gp.fit(ys)

    hrf_measurement_points = np.concatenate(output[1][0])
    order = np.argsort(hrf_measurement_points)
    hx, hy = hrf_measurement_points[order], output[1][1][order]
    plt.plot(hx, hy)
    # plt.axis([0, 32, -.02, .05])

