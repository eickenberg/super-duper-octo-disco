"""Kernels: Gaussian process for hrf estimaton (sandbox)
"""
# TODO add more kernels

import numpy as np
from sklearn.base import clone
from scipy.sparse import coo_matrix
# from scipy.optimize import fmin_cobyla
from sklearn.gaussian_process.kernels import (Kernel, RBF,
                                              StationaryKernelMixin)
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter
from scipy.interpolate import interp1d


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
        import warnings
        warnings.warn("The HRF model is not recognized, setting it to None")
    return hrf_0

###############################################################################
# Kernel utils
###############################################################################
class HRFKernel(StationaryKernelMixin, Kernel):
    """This is just a class based on sklearn.
    Here, we add our utils to find the kernel when working with a gaussian
    process of linear combinations.

    Parameters
    ----------
    beta_values: array-like, shape number of event types
    beta_indices: list of list
    etas: array_like, amplitude of the event
    """
    def __init__(self, gamma=10., gamma_bounds=(1e-5, 1e5), kernel=None,
                 beta_values=None, beta_indices=None, etas=None,
                 return_eval_cov=True):
        self.return_eval_cov = return_eval_cov
        self.beta_values = beta_values
        self.beta_indices = beta_indices
        self.etas = etas
        self.kernel = kernel
        self.gamma = float(gamma)
        self.hyperparameter_gamma = Hyperparameter("gamma", "numeric",
                                                   gamma_bounds)

    def _eta_weighted_kernel(self, hrf_measurement_points,
                             evaluation_points=None):
        """This function computes the kernel matrix of all measurement points,
        potentially redundantly per measurement points, just to be sure we
        identify things correctly afterwards.

        If evaluation_points is set to None, then the original
        hrf_measurement_points will be used

        Parameters
        ----------
        hrf_measurement_points: list of list
        evaluation_points: list of list

        Returns
        -------
        pre_cov: array-like
        pre_cross_cov: array-like (optional)
        """
        if self.kernel is None:
            self.kernel_ = ConstantKernel(1., constant_value_bounds="fixed") \
                * RBF(self.gamma, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        if evaluation_points is None:
            evaluation_points = hrf_measurement_points

        etas = self.etas
        eta_weight = etas[:, np.newaxis] * etas

        K = self.kernel_(hrf_measurement_points)
        K_cross = self.kernel_(evaluation_points, hrf_measurement_points)

        eta_weighted_cov = K * eta_weight
        eta_weighted_cross_cov = K_cross * etas

        dt = 0.01
        hrf_length = 24.
        hrf_model = None
        x_0 = np.arange(0, hrf_length + dt, dt)
        hrf_0 = _get_hrf_model(self.hrf_model, hrf_length=hrf_length + dt,
                               dt=dt, normalize=False)
        f = interp1d(x_0, hrf_0)
        pre_mean_n = f(hrf_measurement_points).squeeze() * etas
        pre_mean_m = f(hrf_measurement_points).squeeze() * etas

        if self.return_eval_cov:
            K_22 = self.kernel_(evaluation_points)
            return eta_weighted_cov, eta_weighted_cross_cov, pre_mean_n, pre_mean_m, K_22
        return eta_weighted_cov, eta_weighted_cross_cov, pre_mean_n, pre_mean_m

    def _fit_hrf_kernel(self, eta_weighted_cov, eta_weighted_cross_cov,
                        eta_weighted_mean):
        """
        """
        beta_values = self.beta_values.copy()
        beta_indices = self.beta_indices

        col_coordinates = np.concatenate(
            [i * np.ones(len(beta_ind)) for i, beta_ind in enumerate(beta_indices)])

        all_betas = beta_values[np.concatenate(beta_indices).astype('int')]

        row_coordinates = np.arange(len(all_betas))
        collapser = coo_matrix((all_betas, (row_coordinates, col_coordinates)),
        shape = (len(all_betas), len(beta_indices))).tocsc()

        K = collapser.T.dot(collapser.T.dot(eta_weighted_cov).T).T
        K_cross = collapser.T.dot(eta_weighted_cross_cov.T).T  # again
        mu_n = collapser.T.dot(eta_weighted_mean).T

        return K, K_cross, mu_n

    def __call__(self, hrf_measurement_points, evaluation_points=None):
        """
        """
        if evaluation_points is None:
            evaluation_points = hrf_measurement_points

        if self.return_eval_cov:
            eta_weighted_cov, eta_weighted_cross_cov, \
            eta_weighted_mean_n, mu_m, K_22 = \
                self._eta_weighted_kernel(hrf_measurement_points,
                                          evaluation_points)
            K, K_cross, mu_n = self._fit_hrf_kernel(eta_weighted_cov,
                                              eta_weighted_cross_cov,
                                              eta_weighted_mean_n)
            return K, K_cross, mu_n, mu_m, K_22
        else:
            eta_weighted_cov, eta_weighted_cross_cov, \
            eta_weighted_mean_n, mu_m = \
                self._eta_weighted_kernel(hrf_measurement_points,
                                          evaluation_points)
            K, K_cross, mu_n = self._fit_hrf_kernel(eta_weighted_cov,
                                              eta_weighted_cross_cov,
                                              eta_weighted_mean_n)
            return K, K_cross, mu_n, mu_m

    def clone_with_params(self, **params):
        cloned = clone(self)
        cloned.set_params(**params)
        return cloned

    # XXX
    def diag(X):
        """Returns the diagonal of K(X, X)
        """


