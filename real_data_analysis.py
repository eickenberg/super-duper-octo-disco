import os.path as op
import nibabel as nb
import numpy as np

from gp import SuperDuperGP, _get_hrf_model
from nistats import experimental_paradigm, design_matrix
from scipy.interpolate import interp1d


#folder = 'AINSI_002'
#bold_fn = op.join(folder, 'Preprocessed', 's444wuaAINSI 002 EVep2dbolds005a001.nii')
#mask_fn = op.join(folder, 'Masks', 'visual_small_mask_dilated.nii')
#voxel_fn = op.join(folder, 'Preprocessed', 'voxel92812_min.npy')

folder = 'data_example'
#voxel_fn = op.join(folder, 'voxel92812_min.npy')
#voxel_fn = op.join(folder, 'voxel95930_min2.npy')
#voxel_fn = op.join(folder, 'voxel54246_min3.npy')
#voxel_fn = op.join(folder, 'voxel3624_min6.npy')
voxel_fn = op.join(folder, 'audio_voxel272.npy')
voxel_fn = op.join(folder, 'visual_voxel81.npy')
paradigm_fn = op.join(folder, 'onsets.csv')

# Load data and parameters
#niimgs = nb.load(bold_fn)
#bold_data = niimgs.get_data()
#n_scans = bold_data.shape[3]
n_scans = 144
t_r = 3.
ys = np.load(voxel_fn)

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
print 'residual norm square = ', resid_norm_sq

hy *= np.sign(hy[np.argmax(np.abs(hy))]) / np.abs(hy).max()
hrf_0 /= hrf_0.max()

import matplotlib.pyplot as plt
plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                 hy + 1.96 * np.sqrt(hrf_var), alpha=0.1)
plt.plot(hx, hy)
plt.plot(x_0, hrf_0)
