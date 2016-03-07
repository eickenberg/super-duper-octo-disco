import numpy as np
import nibabel as nb
from sklearn.utils import check_random_state
from scipy.interpolate import interp1d
from nistats.glm import FirstLevelGLM
from nilearn.input_data import NiftiMasker
from nilearn.plotting import find_cuts
import matplotlib.pyplot as plt

from nistats.hemodynamic_models import spm_hrf, glover_hrf, _gamma_difference_hrf
from hrf import bezier_hrf, physio_hrf
from data_generator import (generate_spikes_time_series,
                            generate_fmri)
from gp import SuperDuperGP, _get_hrf_model


seed = 42
rng = check_random_state(seed)


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
sigma_noise = 0.01
threshold = 0.7
seed = 42

mask_img = nb.Nifti1Image(np.ones((n_x, n_y, n_z)), affine=np.eye(4))
masker = NiftiMasker(mask_img=mask_img)
masker.fit()


hrf_ushoot = 16.
i = 0

for hrf_peak_sim in xrange(3, 9):


    # Simulate with different hrf peaks
    hrf_sim = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length + dt,
                                  onset=0., delay=hrf_peak_sim, undershoot=hrf_ushoot,
                                  dispersion=1., u_dispersion=1., ratio=0.167)
    f_hrf_sim = interp1d(x_0, hrf_sim)


    fmri, paradigm, design_sim, masks = generate_fmri(
        n_x=n_x, n_y=n_y, n_z=n_y, modulation=None, n_events=n_events,
        event_types=event_types, n_blank_events=n_blank_events,
        event_spacing=event_spacing, t_r=t_r, smoothing_fwhm=smoothing_fwhm,
        sigma=sigma, sigma_noise=sigma_noise, threshold=threshold, seed=seed,
        f_hrf=f_hrf_sim, hrf_length=hrf_length)

    niimgs = nb.Nifti1Image(fmri, affine=np.eye(4))


    for hrf_peak in xrange(3, 9):

        # GLM using HRF with a different peak
        hrf_est = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length + dt,
                                      onset=0., delay=hrf_peak, undershoot=hrf_ushoot,
                                      dispersion=1., u_dispersion=1., ratio=0.167)
        f_hrf_est = interp1d(x_0, hrf_est)

        _, design, _, _ = generate_spikes_time_series(
            n_events=n_events, n_blank_events=n_blank_events,
            event_spacing=event_spacing, t_r=t_r, event_types=event_types,
            return_jitter=True, jitter_min=jitter_min, jitter_max=jitter_max,
            period_cut=64, time_offset=10, modulation=None,
            seed=seed, f_hrf=f_hrf_est, hrf_length=hrf_length)


        # Testing with a GLM
        glm = FirstLevelGLM(mask=mask_img, t_r=t_r, standardize=True,
                            noise_model='ols')
        glm.fit(niimgs, design)


# Check the norm of the residuals!!
# HOW?
