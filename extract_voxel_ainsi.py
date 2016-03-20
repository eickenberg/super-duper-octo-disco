import os.path as op
import nibabel as nb
import numpy as np
from nilearn.input_data import NiftiMasker
from nistats import experimental_paradigm, design_matrix
from nistats.glm import FirstLevelGLM
from nilearn.input_data import NiftiMasker


# File names and paths. Start from AINSI folder.
paradigm_fn = 'onsets.csv'
bold_fn = op.join('AINSI_002', 'Preprocessed', 's444wuaAINSI 002 EVep2dbolds005a001.nii')
mask_file = op.join('AINSI_002', 'Masks', 'rlabels_Neuromorphometrics.nii'

# Load data and parameters
t_r = 3.
niimgs = nb.load(bold_fn)
bold_data = niimgs.get_data()
n_scans = bold_data.shape[3]

# Create design matrix
frametimes = np.arange(0, n_scans * t_r, t_r)
paradigm = experimental_paradigm.paradigm_from_csv(paradigm_fn)
dm = design_matrix.make_design_matrix(frametimes, paradigm=paradigm)

# Create masker from epi and detrended
masker2 = NiftiMasker(mask_strategy='epi', detrend=True)
niimgs_detrended = masker2.fit_transform(niimgs)

# GLM analysis
glm_results = glm.session_glm(niimgs_detrended, dm, noise_model='ols')

# Extract norm_resid results
glm_reg_res = glm_results[1][0]
mean_resid = glm_reg_res.norm_resid.mean(0)

# Choose voxel with lower norm_resid
ind_min = mean_resid.argmin()
ts1 = niimgs_detrended[:, ind_min]
np.save('voxel%d.npy' % ind_min, ts1)


