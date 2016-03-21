import os.path as op
import nibabel as nb
import numpy as np
import scipy

from nilearn.input_data import NiftiMasker
from nistats import experimental_paradigm, design_matrix
#from nistats.glm import FirstLevelGLM
from nilearn.input_data import NiftiMasker
from nistats import glm


# File names and paths. Start from AINSI folder.
paradigm_fn = 'onsets.csv'
bold_fn = op.join('AINSI_002', 'Preprocessed', 's444wuaAINSI 002 EVep2dbolds005a001.nii')
mask_file0 = op.join('AINSI_002', 'Masks', 'rlabels_Neuromorphometrics.nii')
name='visual'
mask_file = op.join('AINSI_002', 'Masks', 'visual_small_mask.nii')
name='audio'
mask_file = op.join('AINSI_002', 'Masks', 'audio_mask.nii')

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
mask_img = nb.load(mask_file)
masker2 = NiftiMasker(mask_img=mask_img, detrend=True)
#masker2 = NiftiMasker(mask_strategy='epi', detrend=True)
niimgs_detrended = masker2.fit_transform(niimgs)
# GLM analysis
glm_results = glm.session_glm(niimgs_detrended, dm, noise_model='ols')
# Extract norm_resid results
glm_reg_res = glm_results[1][0]
mean_resid = glm_reg_res.norm_resid.mean(0)
mean_resid0 = (glm_reg_res.resid**2).sum(0)

# Choose voxel with lower norm_resid
ind_min = mean_resid.argmin()
ts1 = niimgs_detrended[:, ind_min]
np.save(name + '_voxel%d.npy' % ind_min, ts1)

indices_vec = np.zeros_like(niimgs_detrended[0, :])
indices_vec[ind_min] = 1.
image = masker2.inverse_transform(indices_vec)
nb.save(image, name + '_voxel%d_mask.nii' % ind_min)
masker = NiftiMasker(mask_img=image, detrend=True)
niimgs_detrended = masker.fit_transform(niimgs)
mean_voxels = niimgs_detrended.mean(1)
np.save(name + '_voxel%d_mean%dv.npy' % (ind_min, niimgs_detrended.shape[1]), mean_voxels)
glm_results = glm.session_glm(niimgs_detrended, dm, noise_model='ols')
glm_reg_res = glm_results[1][0]
mean_resid = (glm_reg_res.resid**2).sum(0)
print mean_resid

data = image.get_data()
datad = scipy.ndimage.binary_dilation(data).astype(data.dtype)
imaged = nb.Nifti1Image(datad, affine=image.get_affine())
nb.save(imaged, name + '_voxel%d_mask_dilated.nii' % ind_min)
masker = NiftiMasker(mask_img=imaged, detrend=True)
niimgs_detrended = masker.fit_transform(niimgs)
mean_voxels = niimgs_detrended.mean(1)
np.save(name + '_voxel%d_mean%dv.npy' % (ind_min, niimgs_detrended.shape[1]), mean_voxels)
glm_results = glm.session_glm(niimgs_detrended, dm, noise_model='ols')
glm_reg_res = glm_results[1][0]
mean_resid = (glm_reg_res.resid**2).sum(0)
print mean_resid

datad2 = scipy.ndimage.binary_dilation(datad).astype(datad.dtype)
imaged2 = nb.Nifti1Image(datad2, affine=image.get_affine())
nb.save(imaged2, name + '_voxel%d_mask_dilated2.nii' % ind_min)
masker = NiftiMasker(mask_img=imaged2, detrend=True)
niimgs_detrended = masker.fit_transform(niimgs)
mean_voxels = niimgs_detrended.mean(1)
np.save(name + '_voxel%d_mean%dv.npy' % (ind_min, niimgs_detrended.shape[1]), mean_voxels)
glm_results = glm.session_glm(niimgs_detrended, dm, noise_model='ols')
glm_reg_res = glm_results[1][0]
mean_resid = (glm_reg_res.resid**2).sum(0)
print mean_resid

datad3 = scipy.ndimage.binary_dilation(datad2).astype(datad.dtype)
imaged3 = nb.Nifti1Image(datad3, affine=image.get_affine())
nb.save(imaged3, name + '_voxel%d_mask_dilated3.nii' % ind_min)
masker = NiftiMasker(mask_img=imaged3, detrend=True)
niimgs_detrended = masker.fit_transform(niimgs)
mean_voxels = niimgs_detrended.mean(1)
np.save(name + '_voxel%d_mean%dv.npy' % (ind_min, niimgs_detrended.shape[1]), mean_voxels)
glm_results = glm.session_glm(niimgs_detrended, dm, noise_model='ols')
glm_reg_res = glm_results[1][0]
mean_resid = (glm_reg_res.resid**2).sum(0)
print mean_resid


# Sort voxels
#sorted_ind = np.sort(mean_resid)
#tsm = niimgs_detrended[:, sorted_ind[:20]].mean(1)
#np.save('visual_20voxel_mean.npy', ts1)



