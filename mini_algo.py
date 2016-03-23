import numpy as np

from sklearn.base import clone
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
standard_kernel = ConstantKernel(1.) * RBF(length_scale=2.)

from nistats.hemodynamic_models import glover_hrf
from scipy.interpolate import interp1d
hrf = glover_hrf(1., 16., time_length=33)
glover = interp1d(np.linspace(0, 33., len(hrf), endpoint=False), hrf)
def zero_hrf(x):
    return np.zeros_like(x)

from gp import _get_hrf_measurements, _get_design_from_hrf_measures
from scipy.sparse import coo_matrix, eye, block_diag


def get_hrf_measurement_covariance(hrf_measurement_points, kernel,
                                   extra_points=None,
                                   eval_gradient=False):

    X = np.concatenate(hrf_measurement_points)
    if extra_points is not None:
        X = np.concatenate([X, extra_points])
    return kernel(X[:, np.newaxis], eval_gradient=eval_gradient)


def get_collapser_Zbeta(beta_values, modulation,
                        beta_indices,
                        n_extra_points=0):

    col_coordinates = np.concatenate(
        [i * np.ones(len(beta_ind))
         for i, beta_ind in enumerate(beta_indices)])

    all_betas = beta_values[np.concatenate(beta_indices).astype('int')]
    all_alphas = np.concatenate(modulation)
    row_coordinates = np.arange(len(all_betas))
    collapser = coo_matrix((all_betas * all_alphas,
                            (row_coordinates, col_coordinates)),
                           shape=(len(all_betas),
                                  len(beta_indices)))
    collapser = block_diag([collapser,
                            eye(n_extra_points, n_extra_points)]).tocsc()
    return collapser


def get_hrf_measures(y, betas, hrf_kernel_matrix,
                     mean_vector,
                     sigma_squared,
                     alphas, beta_indices,
                     extra_values=None,
                     n_extra_points=None,
                     return_loglikelihood=False,
                     hrf_kernel_matrix_gradients=None,
                     return_loglikelihood_gradient=False,
                     return_loo_error=False):

    if n_extra_points is None:
        n_extra_points = hrf_kernel_matrix.shape[0] - len(y)
    if extra_values is None:
        extra_values = np.zeros(n_extra_points)

    y = np.concatenate([y, extra_values])
    Zcollapser = get_collapser_Zbeta(betas, alphas,
                                     beta_indices,
                                     n_extra_points=n_extra_points)
    measurement_covariance = Zcollapser.T.dot(
        Zcollapser.T.dot(hrf_kernel_matrix.T).T)
    measurement_mean = Zcollapser.T.dot(mean_vector)
    cross_covariance = Zcollapser.T.dot(hrf_kernel_matrix.T).T

    noisy_measurement_covariance = measurement_covariance.copy()
    noisy_measurement_covariance[
        :len(y) - n_extra_points,
        :len(y) - n_extra_points] += np.eye(
            len(y) - n_extra_points) * sigma_squared

    G = np.linalg.inv(noisy_measurement_covariance)
    dual_coef = G.dot(y - measurement_mean)
    hrf_measurements = mean_vector + cross_covariance.dot(dual_coef)

    outputs = [hrf_measurements]

    if return_loglikelihood:
        data_fit = -.5 * np.dot(y - measurement_mean, dual_coef)
        ev = np.linalg.eigvalsh(noisy_measurement_covariance)
        complexity = -.5 * np.sum(np.log(ev))
        loglikelihood = data_fit + complexity - y.shape[0] / 2. * np.log(2 * np.pi)
        outputs.append(loglikelihood)
    if return_loglikelihood_gradient and hrf_kernel_matrix_gradients is not None:
        dual_minus_G = dual_coef[:, np.newaxis] * dual_coef - G
        measurement_kernel_gradients = np.concatenate([Zcollapser.T.dot(
            Zcollapser.T.dot(grad_mat).T)[..., np.newaxis] 
            for grad_mat in hrf_kernel_matrix_gradients.T], axis=2)
        ll_gradient = .5 * (dual_minus_G[..., np.newaxis] *
                            measurement_kernel_gradients).sum(0).sum(0)
        outputs.append(ll_gradient)
    if return_loo_error:
        diagG = np.diag(G)
        looe = dual_coef / diagG
        outputs.append(looe)

    return outputs


def alternating_optimization(paradigm, y, hrf_length=32.,
                             t_r=None, time_offset=None,
                             frame_times=None,
                             kernel=standard_kernel,
                             mean=glover,
                             sigma_squared=1.,
                             initial_beta=None,
                             modulation=None,
                             n_alternations=20,
                             rescale_hrf=False,
                             optimize_kernel=False,
                             optimize_sigma_squared=False,
                             n_iter_optimize=None,
                             clamp_endpoints=True):
    if frame_times is None:
        if t_r is None or time_offset is None:
            raise ValueError(
                'If frametimes not specified, then set t_r and time_offset')

    kernel = clone(kernel)
    # if rescale_hrf:
    #     scale_kernel = ConstantKernel(1.)
    #     kernel = scale_kernel * kernel
    #     scale_kernel = kernel.k1

    (hrf_measurement_points,
     visible_events,
     alphas, beta_indices,
     unique_events) = _get_hrf_measurements(paradigm, modulation=modulation,
                                            hrf_length=hrf_length, t_r=t_r,
                                            time_offset=time_offset,
                                            zeros_extremes=False,
                                            frame_times=frame_times)
    if initial_beta is None:
        betas = np.ones(len(unique_events))
    else:
        betas = initial_beta.copy()

    if clamp_endpoints:
        extra_points = np.array([0., hrf_length * .99])
    else:
        extra_points = None

    hrf_kernel_matrix, gradients = get_hrf_measurement_covariance(
        hrf_measurement_points, kernel,
        extra_points=extra_points, eval_gradient=True)

    mean_vector = mean(np.concatenate(hrf_measurement_points))
    if extra_points is not None:
        mean_vector = np.concatenate([mean_vector, mean(extra_points)])

    all_residual_norms = []
    all_hrf_measures = []
    all_lls = []
    all_gradients = []
    all_looe = []
    all_thetas = []
    all_sigmas_squared = []
    step_size = .01
    for alternating_iter in range(n_alternations):
        hrf_measures_, ll, ll_gradient, looe = get_hrf_measures(y, betas, hrf_kernel_matrix,
                                                                mean_vector, sigma_squared,
                                                                alphas, beta_indices,
                                                                n_extra_points=(
                                                                    len(extra_points) if extra_points is not None else 0),
                                                                return_loglikelihood=True,
                                                                hrf_kernel_matrix_gradients=gradients,
                                                                return_loglikelihood_gradient=True,
                                                                return_loo_error=True)
        hrf_measures = (hrf_measures_[:-len(extra_points)]
                                      if extra_points is not None else None)
        if rescale_hrf:
            hrf_size = np.abs(hrf_measures).max()
            hrf_measures /= hrf_size
#            scale_kernel.theta /= hrf_size
        design = _get_design_from_hrf_measures(
            hrf_measures,
            beta_indices)
        betas = np.linalg.pinv(design).dot(y)
        residual_norm_squared = ((y - design.dot(betas)) ** 2).sum()
        if optimize_sigma_squared:
            sigma_squared = residual_norm_squared / (design.shape[0] - design.shape[1])
        all_sigmas_squared.append(sigma_squared)
        all_residual_norms.append(residual_norm_squared)
        all_hrf_measures.append(hrf_measures)
        all_lls.append(ll)
        all_gradients.append(ll_gradient)
        all_looe.append(looe)
        all_thetas.append(kernel.theta.copy())
        if optimize_kernel:
            if len(all_looe) > 1:
                if (all_looe[-1] ** 2).sum() > (all_looe[-2] ** 2).sum():
                    # revert theta step
                    kernel.theta = np.log(np.maximum(np.exp(kernel.theta) - step_size * ll_gradient, 1e-4))
                    step_size *= .5
                    print('Gradient step too large, reverting. Step size {}'.format(step_size))

            kernel.theta = np.log(np.maximum(np.exp(kernel.theta) + step_size * ll_gradient, 1e-4))
            if np.isnan(kernel.theta).any(): stop
            hrf_kernel_matrix, gradients = get_hrf_measurement_covariance(
                hrf_measurement_points, kernel,
                extra_points=extra_points, eval_gradient=True)
            if np.isnan(hrf_kernel_matrix).any():
                stop


    return betas, (hrf_measurement_points, hrf_measures), all_residual_norms, all_hrf_measures, all_lls, all_gradients, all_looe, all_thetas, all_sigmas_squared


if __name__ == "__main__":
    from data_generator import (
        generate_spikes_time_series as generate_experiment)
    n_events=100
    n_blank_events=25
    event_spacing=6
    t_r=2
    hrf_length=32.
    event_types=['ev1', 'ev2']
    jitter_min=-1
    jitter_max=1
    time_offset = 20
    modulation=None
    seed = 42

    from nistats.hemodynamic_models import _gamma_difference_hrf
    simulation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
                                           time_length=hrf_length + 1,
                                           undershoot=16., delay=11.)
    xs = np.linspace(0., hrf_length + 1, len(simulation_hrf), endpoint=False)
    f_sim_hrf = interp1d(xs, simulation_hrf)
    from data_generator import make_design_matrix_hrf


    paradigm, design_, modulation, measurement_times = generate_experiment(
        n_events=n_events,
        n_blank_events=n_blank_events,
        event_spacing=event_spacing,
        t_r=t_r, hrf_length=hrf_length,
        event_types=event_types,
        jitter_min=jitter_min,
        jitter_max=jitter_max,
        time_offset=time_offset,
        modulation=modulation,
        return_jitter=True,
        seed=seed)

    design_ = make_design_matrix_hrf(measurement_times, paradigm, f_hrf=f_sim_hrf)

    design = design_[event_types].values
    rng = np.random.RandomState(seed)
    beta = rng.randn(len(event_types))
    y_clean = design.dot(beta)
    noise  = rng.randn(len(y_clean))
    noise /= np.linalg.norm(noise)
    y_noisy = y_clean + np.linalg.norm(y_clean) * noise * 2

    kernel = RBF()

    (betas, (hrf_measurement_points, hrf_measures),
     residuals, hrfs, lls, grads, looes, thetas, sigmas_squared) = alternating_optimization(
         paradigm, y_noisy,
         hrf_length,
         frame_times=measurement_times,
         modulation=modulation,
         mean=zero_hrf,
         n_alternations=50,
         sigma_squared=4,
         rescale_hrf=True,
         optimize_kernel=True,
         optimize_sigma_squared=False)


