import os
import os.path as op
import numpy as np
import nibabel as nb
from sklearn.utils import check_random_state
from scipy.interpolate import interp1d
from nistats.glm import FirstLevelGLM
from nilearn.input_data import NiftiMasker
from nilearn.plotting import find_cuts
import matplotlib
import matplotlib.pyplot as plt

from nistats.hemodynamic_models import (spm_hrf, glover_hrf,
                                        _gamma_difference_hrf)
from hrf import bezier_hrf, physio_hrf
from data_generator import (generate_spikes_time_series,
                            generate_fmri)
from gp import SuperDuperGP, _get_hrf_model


seed = 42
rng = check_random_state(seed)
fig_folder = 'images'

# Generate simulated data
hrf_length = 25
dt = 0.1
x_0 = np.arange(0, hrf_length  + dt, dt)

n_x, n_y, n_z = 20, 20, 20
event_types = ['ev1', 'ev2']
n_events = 100
n_blank_events = 50
event_spacing = 6
jitter_min, jitter_max = -1, 1
t_r = 2
smoothing_fwhm = 1
sigma = 2
#sigma_noise = 0.000001
threshold = 0.7
period_cut = 512
drift_order = 1

mask_img = nb.Nifti1Image(np.ones((n_x, n_y, n_z)), affine=np.eye(4))
masker = NiftiMasker(mask_img=mask_img)
masker.fit()


#HRF peak
peak_range_sim = np.arange(3, 11)
peak_range = np.arange(3, 11)
hrf_ushoot = 16.

norm_resid = np.zeros((len(peak_range), len(peak_range)))
i = 0

for sigma_noise in np.array([0.1, 0.001, 0.00001, 0.0000001]):

    for isim, hrf_peak_sim in enumerate(peak_range_sim):

        # Simulate with different hrf peaks
        hrf_sim = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length + dt,
                                      onset=0., delay=hrf_peak_sim, undershoot=hrf_ushoot,
                                      dispersion=1., u_dispersion=1., ratio=0.167)
        f_hrf_sim = interp1d(x_0, hrf_sim)
        plt.plot(hrf_sim)
        plt.show()

        fmri, paradigm, design_sim, masks = generate_fmri(
            n_x=n_x, n_y=n_y, n_z=n_y, modulation=None, n_events=n_events,
            event_types=event_types, n_blank_events=n_blank_events,
            event_spacing=event_spacing, t_r=t_r, smoothing_fwhm=smoothing_fwhm,
            sigma=sigma, sigma_noise=sigma_noise, threshold=threshold, seed=seed,
            f_hrf=f_hrf_sim, hrf_length=hrf_length,
            period_cut=period_cut, drift_order=drift_order)

        fmri = fmri / fmri.mean() * 100
        niimgs = nb.Nifti1Image(fmri, affine=np.eye(4))

        for iest, hrf_peak in enumerate(peak_range):

            # GLM using HRF with a different peak
            hrf_est = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length + dt,
                                          onset=0., delay=hrf_peak, undershoot=hrf_ushoot,
                                          dispersion=hrf_width, u_dispersion=1., ratio=0.167)
            f_hrf_est = interp1d(x_0, hrf_est)

            _, design, _, _ = generate_spikes_time_series(
                n_events=n_events, n_blank_events=n_blank_events,
                event_spacing=event_spacing, t_r=t_r, event_types=event_types,
                return_jitter=True, jitter_min=jitter_min, jitter_max=jitter_max,
                period_cut=period_cut, drift_order=drift_order, time_offset=10,
                modulation=None, seed=seed, f_hrf=f_hrf_est, hrf_length=hrf_length)

            # Testing with a GLM
            glm = FirstLevelGLM(mask=mask_img, t_r=t_r, standardize=True,
                                noise_model='ols')
            glm.fit(niimgs, design)
            #print 'n_timepoints, n_voxels: ', glm.results_[0][0].norm_resid.shape
            #print glm.results_[0][0].resid
            #print glm.results_[0][0].logL
            print glm.results_[0][0].norm_resid.mean()
            norm_resid[isim, iest] = np.linalg.norm(glm.results_[0][0].resid, axis=0).mean()


    if not op.exists(fig_folder): os.makedirs(fig_folder)
    fig_name = op.join(fig_folder, 'glm_residual_norm_sigma' + str(sigma_noise))
    plt.matshow(norm_resid, cmap=plt.cm.Blues)
    plt.xticks(np.arange(peak_range.shape[0]), peak_range)
    plt.yticks(np.arange(peak_range_sim.shape[0]), peak_range_sim)
    plt.xlabel('estimation hrf peak')
    plt.ylabel('simulation hrf peak')
    plt.colorbar()
    plt.title('GLM residual norm')
    plt.savefig(fig_name + '.eps', format='eps')
    plt.show()

