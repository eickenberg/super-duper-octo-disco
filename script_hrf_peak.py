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

import seaborn as sns
from matplotlib import rc
rc('axes', labelsize=32)
rc('xtick', labelsize=32)
rc('ytick', labelsize=32)
rc('legend', fontsize=32)
rc('axes', titlesize=32)
rc('lines', linewidth=1)
# rc('figure', figsize=(18, 10))
rc('text', usetex=False)
rc('font', family='sans-serif')
rc('mathtext', default='regular')

from matplotlib.ticker import FuncFormatter
def add_s(x, pos):
    return '%s s' %s
formatter = FuncFormatter(add_s)



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
hrf_ushoot = 16.


# GP parameters
time_offset = 10
gamma = 1.
fmin_max_iter = 20
n_restarts_optimizer = 0
n_iter = 3
normalize_y = False
optimize = True
zeros_extremes = True

range_peak = np.arange(2, 8)
range_peak = np.array([3, 8])
sigma_noise = 0.01

for sigma_noise in np.array([0.01]):
    if len(range_peak)==2:
        plt.figure(figsize=(8, 4))
    else:
        plt.figure(figsize=(12, 8))
    i = 0

    for hrf_peak in range_peak:


        # Simulate with different hrf peaks
        hrf_sim = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length+dt,
                                    onset=0., delay=hrf_peak, undershoot=hrf_ushoot,
                                    dispersion=1., u_dispersion=1., ratio=0.167)
        f_hrf_sim = interp1d(x_0, hrf_sim)

        paradigm, design, modulation, measurement_time = \
            generate_spikes_time_series(n_events=n_events, n_blank_events=n_blank_events,
                            event_spacing=event_spacing, t_r=t_r, return_jitter=True,
                            jitter_min=jitter_min, jitter_max=jitter_max, f_hrf=f_hrf_sim,
                            hrf_length=hrf_length, event_types=event_types, period_cut=64,
                            time_offset=10, modulation=None, seed=seed)
        design = design[event_types].values  # forget about drifts for the moment
        beta = rng.randn(len(event_types))
        ys = design.dot(beta)
        noise = rng.randn(design.shape[0])
        scale_factor = np.linalg.norm(ys) / np.linalg.norm(noise)
        ys_acquired = ys + noise * scale_factor * sigma_noise

        snr = 20 * (np.log10(np.linalg.norm(ys_acquired) / np.linalg.norm(ys - ys_acquired)))
        print 'SNR = ', snr, ' dB'


        # Estimation with 1 hrf. Uses glover as mean GP
        hrf_model = 'glover'
        hrf_0 = _get_hrf_model(hrf_model, hrf_length=hrf_length + dt,
                               dt=dt, normalize=True)
        f_hrf = interp1d(x_0, hrf_0)
        gp = SuperDuperGP(hrf_length=hrf_length, t_r=t_r, oversampling=1./dt, gamma=gamma,
                    modulation=modulation, fmin_max_iter=fmin_max_iter, sigma_noise=1.,
                    time_offset=time_offset, n_iter=n_iter, normalize_y=normalize_y, verbose=True,
                    optimize=optimize, n_restarts_optimizer=n_restarts_optimizer,
                    zeros_extremes=zeros_extremes, f_mean=f_hrf)
        (hx, hy, hrf_var,
         resid_norm_sq,
         sigma_sq_resid) = gp.fit(ys_acquired, paradigm)
        print 'residual norm square = ', resid_norm_sq

        hy *= np.sign(hy[np.argmax(np.abs(hy))]) / np.abs(hy).max()
        hrf_0 /= hrf_0.max()
        hrf_sim /= hrf_sim.max()

        # Plotting each HRF simulated vs estimated
        if len(range_peak)==5 or len(range_peak)==6:
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
        elif len(range_peak)==3 or len(range_peak)==4:
            plt.subplot(2, 2, i + 1)
            plt.tight_layout()
        elif len(range_peak)==2:
            ax = plt.subplot(1, 2, i + 1)
            ax.tight_layout()
        else:
            plt.figure()
        i += 1
        if np.abs(hy.max())>np.abs(hy.min()):
            nm = hy.max()
        else:
            nm = hy.min()
        ax.fill_between(hx, (hy - 1.96 * np.sqrt(hrf_var))/nm,
                         (hy + 1.96 * np.sqrt(hrf_var))/nm, alpha=0.1)
        ax.plot(hx, hy/nm, 'b', label='estimated HRF')
        ax.plot(x_0, hrf_sim/hrf_sim.max(), 'r--', label='simulated HRF')
        ax.plot(x_0, hrf_0/hrf_0.max(), 'k-', label='GP mean')
        #plt.title('hrf peak ' + str(hrf_peak))
        ax.xlabel('time')
        ax.xaxis.set_major_formatter(formatter)
        ax.axis('tight')
        if len(range_peak)==1:
            plt.legend()

    # Save one image per noise level, with different HRFs
    fig_folder = 'images'
    if not op.exists(fig_folder): os.makedirs(fig_folder)
    fig_name = op.join(fig_folder, \
        'results_GP_simulation_diff_hrf_peak_sigma' + str(sigma_noise) + '_gamma' + str(gamma))
    plt.tight_layout()
    plt.savefig(fig_name + '.png', format='png')
    plt.savefig(fig_name + '.pdf', format='pdf')
    plt.show()

