import numpy as np
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
sigma_noise = .01
hrf_length = 32
dt = 0.1
x_0 = np.arange(0, hrf_length  + dt, dt)


# GP parameters
time_offset = 10
gamma = 10.0
fmin_max_iter = 10
n_restarts_optimizer = 5
n_iter = 2
normalize_y = False
optimize = False
sigma_noise = 0.01
zeros_extremes = True



hrf_ushoot = 16.
plt.figure(figsize=(12, 12))
i = 0

for hrf_peak in xrange(3, 9):


    # Simulate with different hrf peaks
    hrf_sim = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length + dt,
                                  onset=0., delay=hrf_peak, undershoot=hrf_ushoot,
                                  dispersion=1., u_dispersion=1., ratio=0.167)
    f_hrf_sim = interp1d(x_0, hrf_sim)

    paradigm, design, modulation, measurement_time = \
        generate_spikes_time_series(n_events=n_events,
                                    n_blank_events=n_blank_events,
                                    event_spacing=event_spacing, t_r=t_r,
                                    return_jitter=True, jitter_min=jitter_min,
                                    jitter_max=jitter_max,
                                    f_hrf=f_hrf_sim, hrf_length=hrf_length,
                                    event_types=event_types, period_cut=64,
                                    time_offset=10, modulation=None, seed=seed)


    # Mean function of GP set to a certain HRF model
    hrf_model = 'glover'
    hrf_0 = _get_hrf_model(hrf_model, hrf_length=hrf_length + dt,
                           dt=dt, normalize=True)
    f_hrf = interp1d(x_0, hrf_0)


    # Estimation with 1 hrf
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


    # Plotting
    plt.subplot(2, 3, i + 1)
    i += 1
    plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                     hy + 1.96 * np.sqrt(hrf_var), alpha=0.1)
    plt.plot(hx, hy, 'b', label='estimated HRF')
    plt.plot(x_0, hrf_sim, 'r--', label='simulated HRF')
    plt.title('hrf peak ' + str(hrf_peak))
    plt.xlabel('time (sec.)')
    plt.axis('tight')
    #plt.legend()

plt.show()


# check norm of the residuals
