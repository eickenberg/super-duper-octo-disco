"""Gaussian process for hrf estimaton (sandbox)
This implementation is based on scikit learn and Michael's implementation

"""
# TODO add more kernels
# TODO finish the scorer and add test

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state
from nistats.experimental_paradigm import check_paradigm
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import (cholesky, cho_solve, solve_triangular, LinAlgError)
import warnings
from gp_kernels import HRFKernel
from operator import itemgetter
from scipy.interpolate import interp1d
from data_generator import (make_design_matrix_hrf, _get_hrf_model,
                            _get_design_from_hrf_measures,
                            _get_hrf_measurements)
from nistats.design_matrix import (_make_drift)

###############################################################################
#
###############################################################################
class SuperDuperGP(BaseEstimator, RegressorMixin):
    """
    """
    def __init__(self, hrf_length=32., t_r=2, time_offset=10, kernel=None,
                 sigma_noise=0.001, gamma=1., fmin_max_iter=10, n_iter=10,
                 drift_order=1, period_cut=64, normalize_y=False, optimize=False,
                 return_var=True, random_state=None, n_restarts_optimizer=3,
                 oversampling=16, drift_model='cosine', zeros_extremes=False,
                 f_mean=None, min_onset=-24, verbose=True, modulation=None,
                 order=1, remove_difts=True):
        self.t_r = t_r
        self.hrf_length = hrf_length
        self.time_offset = time_offset
        self.period_cut = period_cut
        self.oversampling = oversampling
        self.sigma_noise = sigma_noise
        self.gamma = gamma
        self.fmin_max_iter = fmin_max_iter
        self.n_iter = n_iter
        self.f_mean = f_mean
        self.drift_order = drift_order
        self.drift_model = drift_model
        self.min_onset = min_onset
        self.normalize_y = normalize_y
        self.optimize = optimize
        self.kernel = kernel
        self.return_var = return_var
        self.random_state = random_state
        self.zeros_extremes = zeros_extremes
        self.verbose = verbose
        self.n_restarts_optimizer = n_restarts_optimizer
        self.modulation = modulation
        self.order = order
        self.remove_difts = remove_difts

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

        try:
            L = cholesky(K, lower=True)
        except LinAlgError:
            # loglikelihood = -np.inf # XXX using a large number instead
            loglikelihood = -1e6
            if K_22 is not None:
                return (loglikelihood, mu_m, np.zeros_like(mu_m))
            else:
                return loglikelihood, mu_m

        if ys.ndim==2 and mu_n.ndim==1:
            mu_n = mu_n[:, np.newaxis]
            mu_m = mu_m[:, np.newaxis]

        fs = ys - mu_n
        alpha = cho_solve((L, True), fs)        # K^-1 (ys - mu_n)
        mu_bar = K_cross.dot(alpha) + mu_m

        data_fit = -0.5 * fs.T.dot(alpha)
        model_complexity = -np.log(np.diag(L)).sum()
        normal_const = -0.5 * K.shape[0] * np.log(2 * np.pi)
        loglikelihood_dims = data_fit + model_complexity + normal_const
        loglikelihood = loglikelihood_dims.sum(-1)

        if K_22 is not None:
            L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
            K_inv = L_inv.dot(L_inv.T)
            var_bar = np.diag(K_22) - np.einsum("ki,kj,ij->k", K_cross, K_cross,
                                                K_inv)
            check_negative_var = var_bar < 0.
            if np.any(check_negative_var):
                var_bar[check_negative_var] = 0.

            return loglikelihood, mu_bar, var_bar
        return loglikelihood, mu_bar

    def _fit(self, theta):
        """This function performs an alternate optimization.
        i) Finds HRF given the betas
        ii) Finds the betas given the HRF estimation, we build a new design
        matrix repeat until reach the number of iterations, n_iter
        """
        beta_values = self.initial_beta_.copy()

        kernel = self.hrf_kernel.clone_with_params(**dict(
            beta_values=beta_values, beta_indices=self.beta_indices_,
            etas=self.etas_))

        index =  np.isnan(theta)
        if any(index):
            theta = self.hrf_kernel.bounds[:, 0]

        kernel.theta = theta

        # Getting eta weighted matrices
        pre_cov, pre_cross_cov, pre_mean_n, pre_mean_m, K_22 = \
            kernel._eta_weighted_kernel(
                self.hrf_measurement_points,
                evaluation_points=self.evaluation_points, f_mean=self.f_mean)

        all_hrf_values = []
        all_hrf_var = []
        all_designs = []
        all_betas = []
        for i in range(self.n_iter):
            if self.verbose:
                print "iter: %s" % i

            loglikelihood, hrf_values, hrf_var = \
                self._get_hrf_values_from_betas(
                    self.y_train, beta_values, self.beta_indices_, self.etas_,
                    pre_cov, pre_cross_cov, pre_mean_n, pre_mean_m, K_22=K_22)

            design = _get_design_from_hrf_measures(hrf_values,
                                                   self.beta_indices_)
            # Least squares estimation
            beta_values = np.linalg.pinv(design).dot(self.y_train)

            all_hrf_values.append(hrf_values)
            all_hrf_var.append(hrf_var)
            all_designs.append(design)
            all_betas.append(beta_values)

            if self.verbose:
                print loglikelihood, self.sigma_noise

        residual_norm_squared = ((self.y_train - design.dot(beta_values)) ** 2).sum()
        sigma_squared_resid = \
            residual_norm_squared / (design.shape[0] - design.shape[1])

        # XXX this is going to be removed, only if we can split the data
        self.sigma_noise = np.sqrt(sigma_squared_resid)

        return np.float64(loglikelihood), \
            (beta_values, (self.hrf_measurement_points, hrf_values, hrf_var),
             (residual_norm_squared, sigma_squared_resid),
             all_hrf_values, all_designs, all_betas)


    def fit(self, ys, paradigm, initial_beta=None):

        rng = check_random_state(self.random_state)

        ys = np.atleast_1d(ys)
        if self.normalize_y:
            self.y_train_mean = np.mean(ys, axis=0)
            ys = ys - self.y_train_mean
        else:
            self.y_train_mean = np.zeros(1)

        # Removing the drifts
        if self.remove_difts:
            names, onsets, durations, modulation = check_paradigm(paradigm)

            frame_times = np.arange(0, onsets.max() + self.time_offset, self.t_r)
            drifts, dnames = _make_drift(self.drift_model, frame_times,
                                         self.order, self.period_cut)
            ys -= drifts.dot(np.linalg.pinv(drifts).dot(ys))

        if self.zeros_extremes:
            if ys.ndim==2:
                ys = np.append(ys, np.zeros((2, ys.shape[1])), axis=0)
            else:
                ys = np.append(ys, np.zeros((2,)), axis=0)

        self.y_train = ys

        # Get paradigm data
        hrf_measurement_points, visible_events, etas, beta_indices, unique_events = \
            _get_hrf_measurements(paradigm, hrf_length=self.hrf_length,
                                  t_r=self.t_r, time_offset=self.time_offset,
                                  zeros_extremes=self.zeros_extremes,
                                  frame_times=frame_times)
        if initial_beta is None:
            initial_beta = np.ones(len(unique_events))

        # Just to be able to use Kernels class
        hrf_measurement_points = np.concatenate(hrf_measurement_points)
        self.hrf_measurement_points = hrf_measurement_points[:, np.newaxis]
        self.evaluation_points = None
        etas = np.concatenate(etas)

        # Initialize the kernel
        self.hrf_kernel = HRFKernel(kernel=self.kernel, gamma=self.gamma,
                                    return_eval_cov=self.return_var)
        self.hrf_kernel.set_params(**dict(
            beta_values=initial_beta, beta_indices=beta_indices, etas=etas))

        self.visible_events_ = visible_events
        self.unique_events_ = unique_events
        self.etas_ = etas
        self.beta_indices_ = beta_indices
        self.initial_beta_ = initial_beta

        # Maximizing the log-likelihood (gradient based optimization)
        self.f_mean_ = self.f_mean
        self.f_mean = None
        if self.optimize:

            def obj_func(theta):
                print theta
                return -self._fit(theta)[0]

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
            # add logic to deal with nan and -inf
            lm_values = list(map(itemgetter(1), optima))
            self.theta_ = optima[np.argmin(lm_values)][0]
            self.hrf_kernel.theta = self.theta_
            self.log_marginal_likelihood_value_ = -np.min(lm_values)
            # Refit the model
            self.f_mean = self.f_mean_
            loglikelihood, output = self._fit(self.theta_)
        else:
            loglikelihood, output = self._fit(self.hrf_kernel.theta)

        hrf_measurement_points = np.concatenate(output[1][0])
        order = np.argsort(hrf_measurement_points)

        hrf_var = output[1][2][order]
        hx, hy = hrf_measurement_points[order], output[1][1][order]

        residual_norm_squared = output[2][0]
        sigma_squared_resid = output[2][1]

        self.hx_ = hx
        self.hrf_ = hy
        self.hrf_var_ = hrf_var
        return (hx, hy, hrf_var, residual_norm_squared, sigma_squared_resid)

    def predict(self, ys, paradigm):
        """
        """
        check_is_fitted(self, "hrf_")
        names, onsets, durations, modulation = check_paradigm(paradigm)
        frame_times = np.arange(0, onsets.max() + self.time_offset, self.t_r)
        f_hrf = interp1d(self.hx_, self.hrf_)

        dm = make_design_matrix_hrf(frame_times, paradigm,
                                    hrf_length=self.hrf_length,
                                    t_r=self.t_r, time_offset=self.time_offset,
                                    drift_model=self.drift_model,
                                    period_cut=self.period_cut,
                                    drift_order=self.drift_order,
                                    f_hrf=f_hrf)
        # Least squares estimation
        beta_values = np.linalg.pinv(dm.values).dot(ys)
        ys_fit = dm.values.dot(beta_values)
        # ress
        ress = ys - ys_fit

        return ys_fit, dm, beta_values, ress

    def scorer(self, ys_true, ys_test, paradigm):
        """Please put here the scorer

        Parameters
        ----------
        ys_true: array-like, the signal without noise
        ys_test: array-like, noisy signal used to learn the hrf
        paradigm: dataframe

        """
        # ys_fit, _, _, _ = self.predict(ys_test, paradigm)
        # # Measure the norm or something
        # import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        theta_opt, func_min, convergence_dict = fmin_l_bfgs_b(
            obj_func, initial_theta, maxfun=self.fmin_max_iter, bounds=bounds,
            approx_grad=True)
        if convergence_dict["warnflag"] != 0:
            warnings.warn("something happended!: %s " % convergence_dict)

        if self.verbose:
            print func_min

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
    gamma = 10.
    fmin_max_iter = 10
    n_restarts_optimizer = 0
    n_iter = 3
    normalize_y = False
    optimize = True
    sigma_noise = .1
    zeros_extremes = True

    # Mean function of GP set to a certain HRF model
    hrf_model = 'glover'
    dt = 0.1
    x_0 = np.arange(0, hrf_length + dt, dt)
    hrf_0 = _get_hrf_model(hrf_model, hrf_length=hrf_length + dt,
                           dt=dt, normalize=True)
    f_hrf = interp1d(x_0, hrf_0)
    # f_hrf = None

    gp = SuperDuperGP(hrf_length=hrf_length, gamma=gamma,
                      fmin_max_iter=fmin_max_iter, sigma_noise=sigma_noise,
                      time_offset=time_offset, n_iter=n_iter,
                      normalize_y=normalize_y, verbose=True, optimize=optimize,
                      n_restarts_optimizer=n_restarts_optimizer,
                      zeros_extremes=zeros_extremes, f_mean=f_hrf)

    design = design[event_types].values  # forget about drifts for the moment
    beta = rng.randn(len(event_types))

    ys = design.dot(beta)
    noise = rng.randn(design.shape[0])
    scale_factor = np.linalg.norm(ys) / np.linalg.norm(noise)
    ys_acquired = ys + noise * scale_factor * sigma_noise

    (hx, hy, hrf_var,
     resid_norm_sq,
     sigma_sq_resid) = gp.fit(ys_acquired, paradigm)


    hy *= np.sign(hy[np.argmax(np.abs(hy))]) / np.abs(hy).max()
    hrf_0 /= hrf_0.max()

    ys_pred, _, _, _ = gp.predict(ys, paradigm)

    plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                     hy + 1.96 * np.sqrt(hrf_var), alpha=0.1)
    plt.plot(hx, hy)
    plt.plot(x_0, hrf_0)

    # plt.axis([0, hrf_length, -0.02, 0.025])
    # plt.axis('tight')
    # plt.show()
