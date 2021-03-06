import os.path as op
import nibabel as nb
import numpy as np

from gp import SuperDuperGP, _get_hrf_model
from nilearn.input_data import NiftiMasker
from nistats.glm import FirstLevelGLM
from nistats import experimental_paradigm, design_matrix
from scipy.interpolate import interp1d

folder = 'data_example'
bold_fn = op.join(folder, 's444wuaAINSI_002_EVep2dbolds005a001.nii')
mask_fn = op.join(folder, 'visual_small_mask_dilated.nii')
paradigm_fn = op.join(folder, 'onsets.csv')

# Load data and parameters
niimgs = nb.load(bold_fn)
ys = niimgs.get_data()
print ys.shape
n_scans = ys.shape[3]
#n_scans = 144
t_r = 3.
mask_img = nb.load(mask_fn)
masker = NiftiMasker(mask_img=mask_img, detrend=True)
ys = masker.fit_transform(niimgs)
print ys.shape

# Create design matrix
frametimes = np.arange(0, n_scans * t_r, t_r)
paradigm = experimental_paradigm.paradigm_from_csv(paradigm_fn)
dm = design_matrix.make_design_matrix(frametimes, paradigm=paradigm)
modulation = np.array(paradigm)[:, 4]

# Define HRF of mean GP
hrf_length = 32
dt = 0.1
x_0 = np.arange(0, hrf_length  + dt, dt)
hrf_ushoot = 16.
hrf_model = 'glover'
hrf_0 = _get_hrf_model(hrf_model, hrf_length=hrf_length + dt,
                       dt=dt, normalize=True)
f_hrf = interp1d(x_0, hrf_0)

# GP parameters
time_offset = 10
gamma = 10.
fmin_max_iter = 20
n_restarts_optimizer = 10
n_iter = 3
normalize_y = False
optimize = True
zeros_extremes = True


# Estimation
gp = SuperDuperGP(hrf_length=hrf_length, t_r=t_r, oversampling=1./dt, gamma=gamma,
            modulation=modulation, fmin_max_iter=fmin_max_iter, sigma_noise=1.,
            time_offset=time_offset, n_iter=n_iter, normalize_y=normalize_y, verbose=True,
            optimize=optimize, n_restarts_optimizer=n_restarts_optimizer,
            zeros_extremes=zeros_extremes, f_mean=f_hrf)
(hx, hy, hrf_var, resid_norm_sq, sigma_sq_resid) = gp.fit(ys, paradigm)
hy *= np.sign(hy[np.argmax(np.abs(hy))]) / np.abs(hy).max()
hrf_0 /= hrf_0.max()
print 'residual norm square = ', resid_norm_sq

# Testing with a GLM
glm = FirstLevelGLM(mask=mask_img, t_r=t_r, standardize=True, noise_model='ols')
glm.fit(niimgs, dm)
norm_resid = (np.linalg.norm(glm.results_[0][0].resid, axis=0)**2).mean()
ys_pred_glm = glm.results_[0][0].predicted[:, 0]



# Predict GP
# XXX: Do we need to predict for GLM???
ys_pred, matrix, betas, resid = gp.predict(ys, paradigm)


# Plot HRF
import matplotlib.pyplot as plt
plt.figure(1)
plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                 hy + 1.96 * np.sqrt(hrf_var), alpha=0.1)
plt.plot(hx, hy, label='estimated HRF')
plt.plot(x_0, hrf_0, label='glover HRF')
plt.axis('tight')
plt.legend()

# Plot predicted signal
plt.figure(2)
plt.plot(ys, 'r', label='acquired')
plt.plot(ys_pred, 'b', label='predicted GP')
nm = np.abs([ys_pred_glm.max(), ys_pred_glm.min()]).max()
plt.plot(ys_pred_glm/nm, 'g', label='predicted GLM')
plt.axis('tight')
plt.legend()
plt.show()
