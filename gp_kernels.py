"""Kernels: Gaussian process for hrf estimaton (sandbox)
"""
# TODO add more kernels

import numpy as np
from sklearn.base import clone
from scipy.sparse import coo_matrix
# from scipy.optimize import fmin_cobyla
from sklearn.gaussian_process.kernels import (Kernel, RBF, DotProduct,
                                              StationaryKernelMixin)
from sklearn.gaussian_process.kernels import (ConstantKernel, Hyperparameter,
                                              Exponentiation, Product)
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, cdist, squareform


###############################################################################
# Kernel utils
###############################################################################
# XXX this is just a test
class Kernel_hrf(StationaryKernelMixin, Kernel):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 sigma_0=1., sigma_0_bounds=(1e-5, 1e5)):
        self.length_scale = float(length_scale)
        self.sigma_0 = float(sigma_0)
        self.length_scale_bounds = length_scale_bounds
        self.sigma_0_bounds = sigma_0_bounds
        self.hyperparameter_length_scale = Hyperparameter("length_scale",
                                                          "numeric",
                                                          length_scale_bounds)
        self.hyperparameter_sigma_0 = Hyperparameter("sigma_0", "numeric",
                                                     sigma_0_bounds)

    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        if Y is None:
            inner = np.inner(X, X) + self.sigma_0 / self.length_scale
            modulation = self.length_scale / inner

            dists = pdist(X / self.length_scale, metric='sqeuclidean')
            K = -0.5 * dists
            K = modulation * squareform(K)
            K = np.exp(K)
            np.fill_diagonal(K, 1)
        else:
            inner = np.inner(X, Y) + self.sigma_0 / self.length_scale
            modulation = self.length_scale / inner

            dists = modulation * cdist(X / self.length_scale,
                                       Y / self.length_scale,
                                       metric='sqeuclidean')
            K = np.exp(-0.5 * dists)

        return K

    def diag(X):
        pass




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
                 beta_values=None, beta_indices=None, etas=None, sigma_0=1.,
                 sigma_0_bounds=(1e-5, 1e5), return_eval_cov=True,
                 eval_gradient=False):
        self.return_eval_cov = return_eval_cov
        self.beta_values = beta_values
        self.beta_indices = beta_indices
        self.etas = etas
        self.kernel = kernel
        self.gamma = float(gamma)
        self.sigma_0 = float(sigma_0)
        self.hyperparameter_sigma_0 = Hyperparameter("sigma_0", "numeric",
                                                     sigma_0_bounds)
        self.hyperparameter_gamma = Hyperparameter("gamma", "numeric",
                                                   gamma_bounds)

    def _eta_weighted_kernel(self, hrf_measurement_points,
                             evaluation_points=None, f_mean=None):
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
            self.kernel_ = Product(ConstantKernel(self.sigma_0, constant_value_bounds="fixed"),
                                   RBF(self.gamma, length_scale_bounds="fixed"))
            # self.kernel_ = Kernel_hrf(self.gamma, length_scale_bounds='fixed',
            #                           sigma_0=self.sigma_0,
            #                           sigma_0_bounds='fixed')
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

        if f_mean is not None:
            pre_mean_n = f_mean(hrf_measurement_points).squeeze() * etas
            pre_mean_m = f_mean(evaluation_points).squeeze() * etas
        else:
            pre_mean_n = np.zeros_like(etas)
            pre_mean_m = np.zeros_like(etas)

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

    def __call__(self, hrf_measurement_points, evaluation_points=None,
                 f_mean=None):
        """
        """
        if self.return_eval_cov:
            eta_weighted_cov, eta_weighted_cross_cov, \
            eta_weighted_mean_n, mu_m, K_22 = \
                self._eta_weighted_kernel(hrf_measurement_points,
                                          evaluation_points, f_mean)
            K, K_cross, mu_n = self._fit_hrf_kernel(eta_weighted_cov,
                                              eta_weighted_cross_cov,
                                              eta_weighted_mean_n)
            return K, K_cross, mu_n, mu_m, K_22
        else:
            eta_weighted_cov, eta_weighted_cross_cov, \
            eta_weighted_mean_n, mu_m = \
                self._eta_weighted_kernel(hrf_measurement_points,
                                          evaluation_points, f_mean)
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
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')

    xs = np.arange(0, 4 * np.pi, 0.01)
    xs = xs[:, np.newaxis]
    xs = xs.repeat(xs.shape[0], axis=1)
    kernel = Kernel_hrf(length_scale=1, sigma_0=0.01)

    fig, axx = plt.subplots(1, 2)
    sim = kernel(xs, xs)
    axx[0].imshow(sim)
    axx[0].axis('off')
    plt.hot()
    sim = kernel(xs)
    axx[1].imshow(sim)
    axx[1].axis('off')
    plt.hot()
    plt.show()
