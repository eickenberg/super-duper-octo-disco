"""Gaussian process for hrf estimaton (sandbox)
This implementation is based on scikit learn and Michael's implementation

"""
# TODO add hrf as a mean for the gp
# TODO add more kernels
# TODO add hyperparameter optimization

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state
from nistats.experimental_paradigm import check_paradigm
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import (cholesky, cho_solve, solve_triangular, LinAlgError)
from nistats.hemodynamic_models import spm_hrf, glover_hrf, _gamma_difference_hrf
from hrf import bezier_hrf, physio_hrf
import warnings
from gp_kernels import HRFKernel
from operator import itemgetter
from scipy.interpolate import interp1d


###############################################################################
# HRF utils
###############################################################################
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


def _get_hrf_measurements(paradigm, hrf_length=32., t_r=2, time_offset=10,
                          zeros_extremes=False):
    """This function:
    Parameters
    ----------
    paradigm : paradigm type
    hrf_length : float
    t_r : float
    time_offset : float
    zeros_extremes : bool

    Returns
    -------
    hrf_measurement_points : list of list
    visible events : list of list
    etas : list of list
    beta_indices : list of list
    unique_events : array-like
    """
    names, onsets, durations, modulation = check_paradigm(paradigm)
    frame_times = np.arange(0, onsets.max() + time_offset, t_r)

    time_differences = frame_times[:, np.newaxis] - onsets
    scope_masks = (time_differences > 0) & (time_differences < hrf_length)
    belong_to_measurement, which_event = np.where(scope_masks)

    unique_events, event_type_indices = np.unique(names, return_inverse=True)

    if zeros_extremes:
        lft = len(frame_times) + 2
    else:
        lft = len(frame_times)
    hrf_measurement_points = [list() for _ in range(lft)]
    etas = [list() for _ in range(lft)]
    beta_indices = [list() for _ in range(lft)]
    visible_events = [list() for _ in range(lft)]

    for frame_id, event_id in zip(belong_to_measurement, which_event):
        hrf_measurement_points[frame_id].append(time_differences[frame_id,
                                                                 event_id])
        etas[frame_id].append(modulation[event_id])
        beta_indices[frame_id].append(event_type_indices[event_id])
        visible_events[frame_id].append(event_id)

    if zeros_extremes:
        # we add first and last point of the hrf
        hrf_measurement_points[frame_id + 1].append(0.)
        etas[frame_id + 1].append(modulation[event_id])
        beta_indices[frame_id + 1].append(event_type_indices[event_id])
        visible_events[frame_id + 1].append(event_id)
        hrf_measurement_points[frame_id + 2].append(hrf_length)
        etas[frame_id + 2].append(modulation[event_id])
        beta_indices[frame_id + 2].append(event_type_indices[event_id])
        visible_events[frame_id + 2].append(event_id)

    return (hrf_measurement_points, visible_events, etas, beta_indices,
            unique_events)


# XXX this function should generate the HRF with a fixed number of samples
# for instance, if we hace ys.shape[0] = 2000, then the size of the HRF should
# be the same, giving us the possibility to evaluate it on every single point.
def _get_hrf_model(hrf_model=None, hrf_length=25., dt=1., normalize=False):
    """Returns HRF created with model hrf_model. If hrf_model is None,
    then a vector of 0 is returned

    Parameters
    ----------
    hrf_model: str
    hrf_length: float
    dt: float
    normalize: bool

    Returns
    -------
    hrf_0: hrf
    """
    if hrf_model == 'glover':
        hrf_0 = glover_hrf(tr=1., oversampling=1./dt, time_length=hrf_length)
    elif hrf_model == 'spm':
        hrf_0 = spm_hrf(tr=1., oversampling=1./dt, time_length=hrf_length)
    elif hrf_model == 'gamma':
        hrf_0 = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length,
                                      onset=0., delay=6, undershoot=16., dispersion=1.,
                                      u_dispersion=1., ratio=0.167)
    elif hrf_model == 'bezier':
        # Bezier curves. We can indicate where is the undershoot and the peak etc
        hrf_0 = bezier_hrf(hrf_length=hrf_length, dt=dt, pic=[6,1], picw=2,
                           ushoot=[15,-0.2], ushootw=3, normalize=normalize)
    elif hrf_model == 'physio':
        # Balloon model. By default uses the parameters of Khalidov11
        hrf_0 = physio_hrf(hrf_length=hrf_length, dt=dt, normalize=normalize)
    else:
        # Mean 0 if no hrf_model is specified
        hrf_0 = np.zeros(hrf_length/dt)
        warnings.warn("The HRF model is not recognized, setting it to None")
    if normalize and hrf_model is not None:
        hrf_0 = hrf_0 / np.linalg.norm(hrf_0)
    return hrf_0


###############################################################################
#
###############################################################################
class SuperDuperGP(BaseEstimator, RegressorMixin):
    """
    """
    def __init__(self, hrf_length=32., t_r=2, time_offset=10, kernel=None,
                 modulation=None, sigma_noise=0.001, gamma=1.,
                 fmin_max_iter=10, n_iter=10,
                 normalize_y=False, optimize=False, return_var=True,
                 random_state=None, n_restarts_optimizer=3,
                 zeros_extremes=False, f_mean=None, verbose=True):
        self.t_r = t_r
        self.hrf_length = hrf_length
        self.modulation = modulation
        self.time_offset = time_offset
        self.sigma_noise = sigma_noise
        self.gamma = gamma
        self.fmin_max_iter = fmin_max_iter
        self.n_iter = n_iter
        self.f_mean = f_mean
        self.normalize_y = normalize_y
        self.optimize = optimize
        self.kernel = kernel
        self.return_var = return_var
        self.random_state = random_state
        self.zeros_extremes = zeros_extremes
        self.verbose = verbose
        self.n_restarts_optimizer = n_restarts_optimizer

    def _get_hrf_values_from_betas(self, ys, beta_values, beta_indices, etas,
                                   pre_cov, pre_cross_cov, pre_mu_n,
                                   mu_m, K_22):
        """This function returns the HRF estimation given information about
        beta (i.e. beta_values, beta_indices)

        Rasmussen and Williams. Varying the hyperparameters (Alg. 2.1)
        """
        # Updating the parameters of the kernel
        kernel = self.hrf_kernel.clone_with_params(**dict(
            beta_values=beta_values, beta_indices=beta_indices, etas=etas))

        # Getting the new kernel evaluation
        K, K_cross, mu_n = kernel._fit_hrf_kernel(eta_weighted_cov=pre_cov,
            eta_weighted_cross_cov=pre_cross_cov, eta_weighted_mean=pre_mu_n)

        # Adding noise to the diagonal (Ridge)
        indx, indy = np.diag_indices_from(K)
        if self.zeros_extremes:
            K[indx[:-2], indy[:-2]] += self.sigma_noise ** 2
        else:
            K[indx, indy] += self.sigma_noise ** 2

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), ys - mu_n) # a.k.a. dual coef
        mu_bar = K_cross.dot(alpha) + mu_m

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

    def _fit(self, ys, hrf_measurement_points, visible_events, etas,
             beta_indices, initial_beta, unique_events, f_mean=None,
             evaluation_points=None):
        """This function performs an alternate optimization.
        i) Finds HRF given the betas
        ii) Finds the betas given the HRF estimation, we build a new design
        matrix repeat until reach the number of iterations, n_iter
        """
        beta_values = initial_beta.copy()

        kernel = self.hrf_kernel.clone_with_params(**dict(
            beta_values=beta_values, beta_indices=beta_indices, etas=etas))

        # Getting eta weighted matrices
        pre_cov, pre_cross_cov, pre_mean_n, pre_mean_m, \
        K_22 = kernel._eta_weighted_kernel(
                    hrf_measurement_points, f_mean, evaluation_points)

        all_hrf_values = []
        all_hrf_var = []
        all_designs = []
        all_betas = []
        for i in range(self.n_iter):
            if self.verbose:
                print "iter: %s" % i

            hrf_values, hrf_var = self._get_hrf_values_from_betas(
                ys, beta_values, beta_indices, etas, pre_cov, pre_cross_cov,
                pre_mean_n, pre_mean_m, K_22=K_22)

            design = _get_design_from_hrf_measures(hrf_values, beta_indices)
            # Least squares estimation
            beta_values = np.linalg.pinv(design).dot(ys)

            all_hrf_values.append(hrf_values)
            all_hrf_var.append(hrf_var)
            all_designs.append(design)
            all_betas.append(beta_values)

        return (beta_values, (hrf_measurement_points, hrf_values, hrf_var),
                all_hrf_values, all_designs, all_betas)

    def fit(self, ys, paradigm, initial_beta=None):

        rng = check_random_state(self.random_state)

        ys = np.atleast_1d(ys)
        if self.normalize_y:
            self.y_train_mean = np.mean(ys, axis=0)
            ys = ys - self.y_train_mean
        else:
            self.y_train_mean = np.zeros(1)

        self.y_train = ys

        if self.zeros_extremes:
            ys = np.append(ys, np.array([0., 0.]))

        # Get paradigm data
        hrf_measurement_points, visible_events, etas, beta_indices, unique_events = \
            _get_hrf_measurements(paradigm, hrf_length=self.hrf_length,
                                  t_r=self.t_r, time_offset=self.time_offset,
                                  zeros_extremes=self.zeros_extremes)
        if initial_beta is None:
            initial_beta = np.ones(len(unique_events))

        # Just to be able to use Kernels class
        hrf_measurement_points = np.concatenate(hrf_measurement_points)
        self.hrf_measurement_points = hrf_measurement_points[:, np.newaxis]
        etas = np.concatenate(etas)

        # Initialize the kernel
        self.hrf_kernel = HRFKernel(kernel=self.kernel, gamma=self.gamma,
                                    return_eval_cov=self.return_var)

        # Maximizing the log-likelihood (gradient based optimization)
        if self.optimize:
            self.hrf_kernel.set_params(**dict(
                beta_values=initial_beta, beta_indices=beta_indices, etas=etas))

            def obj_func(theta):
                return -self.log_marginal_likelihood(theta)

            optima = [(self._constrained_optimization(
                obj_func, self.hrf_kernel.theta, self.hrf_kernel.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                bounds = self.hrf_kernel.bounds
                for i in range(self.n_restarts_optimizer):
                    theta_initial = rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(self._constrained_optimization(
                        obj_func, theta_initial, bounds))
            # Select the best result
            lm_values = list(map(itemgetter(1), optima))
            self.theta_ = optima[np.argmin(lm_values)][0]
            self.hrf_kernel.theta = self.theta_
            self.log_marginal_likelihood_value_ = -np.min(lm_values)

        output = self._fit(ys,
                           hrf_measurement_points=self.hrf_measurement_points,
                           visible_events=visible_events, etas=etas,
                           beta_indices=beta_indices, initial_beta=initial_beta,
                           unique_events=unique_events, f_mean=self.f_mean)

        hrf_measurement_points = np.concatenate(output[1][0])
        order = np.argsort(hrf_measurement_points)

        hrf_var = output[1][2][order]
        hx, hy = hrf_measurement_points[order], output[1][1][order]

        return hx, hy, hrf_var


    def predict(self, paradigm):
        """
        """
        pass

    def transform(self, paradimg):
        """
        """
        pass

    def scorer(self, paradigm, ys):
        """Please put here the scorer
        """
        pass

    def log_marginal_likelihood(self, theta):
    # , eval_gradient=None):
        """This functions return the marginal log-likelihood

        Rasmussen and Williams, model selection(E.q. 5.8)

        Parameters
        ----------
        ys : array-like
        kernel: kernel function
        sigma_noise: float
        theta: list of kernel's parameters

        Returns
        -------
        loglikelihood: float

        see Rasmussen and Williams book, model selection in regression. Eq. 5.8

        """
        kernel = self.hrf_kernel.clone_with_theta(theta)
        K, K_cross, mu_n, mu_m, _ = kernel(self.hrf_measurement_points)

        # Adding noise to the diagonal (Ridge)
        indx, indy = np.diag_indices_from(K)
        if self.zeros_extremes:
            K[indx[:-2], indy[:-2]] += sigma_noise ** 2
        else:
            K[indx, indy] += sigma_noise ** 2

        try:
            L = cholesky(K, lower=True)

        except LinAlgError:
            return -np.inf

        y_train = self.y_train
        y_train = y_train[:, np.newaxis]
        alpha = cho_solve((L, True), ys) # a.k.a. dual coef

        # alpha = alpha[:, np.newaxis]
        loglikelihood_dims = -0.5 * np.einsum("ik,jk->k", y_train, alpha)
        # loglikelihood_dims = -0.5 * y_train.T.dot(alpha) # data fit
        # model compkexity
        loglikelihood_dims -= np.log(np.diag(L)).sum()
        # normalization constant
        loglikelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # sum over all dim (sklearn)
        loglikelihood = loglikelihood_dims.sum(-1)

        print kernel.theta[0], loglikelihood
        return loglikelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        theta_opt, func_min, convergence_dict = fmin_l_bfgs_b(
            obj_func, initial_theta, maxiter=self.fmin_max_iter, bounds=bounds,
            approx_grad=True)
        if convergence_dict["warnflag"] != 0:
            warnings.warn("something happended!: %s " % convergence_dict)

        return theta_opt, func_min


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data_generator import generate_spikes_time_series

    plt.close('all')

    seed = 42
    rng = check_random_state(seed)
    ###########################################################################
    # Generate simulated data
    n_events = 200
    n_blank_events = 50
    event_spacing = 6
    t_r = 2
    jitter_min, jitter_max = -1, 1
    event_types = ['evt_1', 'evt_2', 'evt_3', 'evt_4', 'evt_5', 'evt_6']
    sigma_noise = .01

    hrf_model = 'glover'
    hrf_length = 32
    dt = 0.1
    x_0 = np.arange(0, hrf_length + dt, dt)
    hrf_0 = _get_hrf_model(hrf_model, hrf_length=hrf_length + dt,
                           dt=dt, normalize=True)
    f_hrf = interp1d(x_0, hrf_0)

    paradigm, design, modulation, measurement_time = \
        generate_spikes_time_series(n_events=n_events,
                                    n_blank_events=n_blank_events,
                                    event_spacing=event_spacing, t_r=t_r,
                                    return_jitter=True, jitter_min=jitter_min,
                                    jitter_max=jitter_max,
                                    f_hrf=f_hrf, hrf_length=hrf_length,
                                    event_types=event_types, period_cut=64,
                                    time_offset=10, modulation=None, seed=seed)
    ###########################################################################
    # GP parameters
    hrf_length = 32
    time_offset = 10
    gamma = 10.0
    fmin_max_iter = 10
    n_restarts_optimizer = 5
    n_iter = 10
    normalize_y = False
    optimize = False
    sigma_noise = 0.01
    zeros_extremes = True

    # Mean function of GP set to a certain HRF model
    hrf_model = 'glover'
    dt = 0.1
    x_0 = np.arange(0, hrf_length + dt, dt)
    hrf_0 = _get_hrf_model(hrf_model, hrf_length=hrf_length + dt,
                           dt=dt, normalize=True)
    f_hrf = interp1d(x_0, hrf_0)

    gp = SuperDuperGP(hrf_length=hrf_length, modulation=modulation,
                      gamma=gamma, fmin_max_iter=fmin_max_iter,
                      sigma_noise=sigma_noise, time_offset=time_offset,
                      n_iter=n_iter, normalize_y=normalize_y, verbose=True,
                      optimize=optimize,
                      n_restarts_optimizer=n_restarts_optimizer,
                      zeros_extremes=zeros_extremes, f_mean=f_hrf)

    design = design[event_types].values  # forget about drifts for the moment
    beta = rng.randn(len(event_types))

    ys = design.dot(beta) + rng.randn(design.shape[0]) * sigma_noise ** 2

    hx, hy, hrf_var = gp.fit(ys, paradigm)

    plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                     hy + 1.96 * np.sqrt(hrf_var), alpha=0.1)
    plt.plot(hx, hy)
    # plt.axis([0, hrf_length, -0.02, 0.025])
    # plt.axis('tight')
    plt.show()
