import os
import os.path as op
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from scipy.interpolate import interp1d
from data_generator import generate_spikes_time_series
from gp import SuperDuperGP, _get_hrf_model
from nistats.hemodynamic_models import spm_hrf, glover_hrf, _gamma_difference_hrf
from hrf import bezier_hrf, physio_hrf


seed = 42
rng = check_random_state(seed)


# Generate simulated data
n_events = 200
n_blank_events = 50
event_spacing = 6
t_r = 2
jitter_min, jitter_max = -1, 1
event_types = ['evt_1', 'evt_2', 'evt_3', 'evt_4', 'evt_5', 'evt_6']

# GP parameters
time_offset = 10
gamma = 10.0
fmin_max_iter = 10
n_restarts_optimizer = 5
n_iter = 3
normalize_y = False
optimize = False
zeros_extremes = True

# HRF related params
hrf_length = 25
dt = 0.1
x_0 = np.arange(0, hrf_length  + dt, dt)
hrf_ushoot = 16.
peak_range_sim = np.arange(3, 9)
peak_range = np.arange(3, 9)


# Initialization of matrix of residuals
norm_resid = np.zeros((len(peak_range), len(peak_range)))
i = 0

# For different noise levels, we run everything
for sigma_noise in np.array([0.01]): #, 0.001, 0.1, 1.]):
    print 'sigma_noise = ', sigma_noise

    plt.figure(figsize=(12, 8))
    i = 0

    for isim, hrf_peak_sim in enumerate(peak_range_sim):
        print 'hrf simulated peaks in ', hrf_peak_sim

        # Simulate with different hrf peaks
        hrf_sim = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length + dt,
                                      onset=0., delay=hrf_peak_sim, undershoot=hrf_ushoot,
                                      dispersion=1., u_dispersion=1., ratio=0.167)
        f_hrf_sim = interp1d(x_0, hrf_sim)
        paradigm, design, modulation, measurement_time = \
            generate_spikes_time_series(n_events=n_events, n_blank_events=n_blank_events,
                event_spacing=event_spacing, t_r=t_r, return_jitter=True, jitter_min=jitter_min,
                jitter_max=jitter_max, f_hrf=f_hrf_sim, hrf_length=hrf_length, modulation=None,
                event_types=event_types, period_cut=64, time_offset=10, seed=seed)
        design = design[event_types].values  # forget about drifts for the moment
        beta = rng.randn(len(event_types))
        ys = design.dot(beta)
        noise = rng.randn(design.shape[0])
        scale_factor = np.linalg.norm(ys) / np.linalg.norm(noise)
        ys_acquired = ys + noise * scale_factor * sigma_noise

        snr = np.linalg.norm(ys_acquired) / sigma_noise
        snr_db = 20 * (np.log10(np.linalg.norm(ys_acquired) / sigma_noise))
        print 'SNR = ', snr.mean()
        print 'SNR = ', snr_db.mean(), ' dB'


        for iest, hrf_peak_est in enumerate(peak_range):

            # Estimate using GP mean with a different peak
            hrf_est = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length + dt,
                                          onset=0., delay=hrf_peak_est, undershoot=hrf_ushoot,
                                          dispersion=1., u_dispersion=1., ratio=0.167)
            f_hrf = interp1d(x_0, hrf_est)

            # Estimation with 1 hrf
            gp = SuperDuperGP(hrf_length=hrf_length, t_r=t_r, oversampling=1./dt, modulation=modulation,
                              gamma=gamma, fmin_max_iter=fmin_max_iter, sigma_noise=sigma_noise,
                              time_offset=time_offset, n_iter=n_iter, normalize_y=normalize_y,
                              verbose=True, optimize=optimize, n_restarts_optimizer=n_restarts_optimizer,
                              zeros_extremes=zeros_extremes, f_mean=f_hrf)
            (hx, hy, hrf_var,
             resid_norm_sq,
             sigma_sq_resid) = gp.fit(ys_acquired, paradigm)

            print 'resid_norm_sq = ', resid_norm_sq
            print 'sigma_sq_resid = ', sigma_sq_resid
            #glm = FirstLevelGLM(mask=mask_img, t_r=t_r, standardize=True, noise_model='ols')
            #glm.fit(niimgs, design)
            norm_resid[isim, iest] = resid_norm_sq


    # Plot for each noise level a matrix showing the residual
    fig_folder = 'images'
    if not op.exists(fig_folder): os.makedirs(fig_folder)
    fig_name = op.join(fig_folder, 'sdgp_residual_norm_sigma' + str(sigma_noise))
    plt.matshow(norm_resid, cmap=plt.cm.Blues)
    plt.xticks(np.arange(peak_range.shape[0]), peak_range)
    plt.yticks(np.arange(peak_range_sim.shape[0]), peak_range_sim)
    plt.xlabel('estimation hrf peak')
    plt.ylabel('simulation hrf peak')
    plt.colorbar()
    plt.title('GLM residual norm')
    plt.savefig(fig_name + '.eps', format='eps')
    plt.show()


