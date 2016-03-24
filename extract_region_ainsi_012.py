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
#folder = 'data_example2'
folder = 'data_ainsi012'
bold_fn1 = op.join(folder, 's444rwra1201_WIP_EPI-Rennes_SENSE_4D.nii')
# Acquired with 2x2x4 voxel size
bold_fn2 = op.join(folder, 's444rwra1701_WIP_EPI-Ainsi_SENSE_4D.nii')
# Acquired with 3x3x3 voxel size

folder0 = 'data_example'
#bold_fn = op.join(folder0, 's444wuaAINSI_002_EVep2dbolds005a001.nii')
paradigm_fn = op.join(folder0, 'onsets.csv')
names = ['visualG', 'visualD', 'audioG', 'audioD', 'motorG', 'motorD']

#Rest, calculaudio, calculvideo, clicDaudio, clicDvideo, clicGaudio,
#clicGvideo  phraseaudio  phrasevideo
contrasts = [np.array([-1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]),
			 np.array([-1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]),
			 np.array([-1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
			 np.array([-1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
			 np.array([ 0, 0, 0,-1,-1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
			 np.array([ 0, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0])]

# Load data and parameters
t_r = 3.
niimgs1 = nb.load(bold_fn1)
niimgs2 = nb.load(bold_fn2)
bold_data1 = niimgs1.get_data()
bold_data2 = niimgs2.get_data()
print bold_data1.shape
print bold_data2.shape
n_scans = bold_data1.shape[3]

# Create design matrix
frametimes = np.arange(0, n_scans * t_r, t_r)
paradigm = experimental_paradigm.paradigm_from_csv(paradigm_fn)
dm = design_matrix.make_design_matrix(frametimes, paradigm=paradigm)


for i in xrange(6):
	print names[i]

	mask_file = op.join(folder, 'parcel_' + names[i] + '.nii')

	# Create masker from epi and detrended
	mask_img = nb.load(mask_file)
	masker2 = NiftiMasker(mask_img=mask_img, detrend=True)
	niimgs_detrended = masker2.fit_transform(niimgs1)
	niimgs_masked = masker2.inverse_transform(niimgs_detrended)
	niimgs_detrended_r2 = masker2.fit_transform(niimgs2)
	niimgs_masked_r2 = masker2.inverse_transform(niimgs_detrended_r2)
	np.save(op.join(folder, names[i] + '_mean_timeseries_r1.npy'), niimgs_detrended.mean(1))
	np.save(op.join(folder, names[i] + '_mean_timeseries_r2.npy'), niimgs_detrended_r2.mean(1))

	# GLM analysis
	glm_results = glm.session_glm(niimgs_detrended, dm, noise_model='ols')
	labels = glm_results[0]
	reg_results = glm_results[1]
	contrast_map = glm.compute_contrast(labels, reg_results, contrasts[i])
	indexes = contrast_map.p_value()<0.05
	sum_significant_voxels = niimgs_detrended[:, indexes].mean(1)
	np.save(op.join(folder, names[i] + '_mean_pvalues_005.npy'), sum_significant_voxels)

	inds_mins = np.argsort(contrast_map.p_value())
	ind_min = contrast_map.p_value().argmin()
	v1 = niimgs_detrended[:, ind_min]
	v2 = niimgs_detrended_r2[:, ind_min]
	np.save(op.join(folder, names[i] + '_voxelmin%d.npy' % ind_min), v1)
	np.save(op.join(folder, names[i] + '_voxelmin%d_r2.npy' % ind_min), v2)

	indices_vec = np.zeros_like(niimgs_detrended[0, :])
	indices_vec[ind_min] = 1.
	image = masker2.inverse_transform(indices_vec)
	nb.save(image, op.join(folder, names[i] + '_voxel%d_mask.nii' % ind_min))
	masker = NiftiMasker(mask_img=image, detrend=True)
	niimgs_detrended0 = masker.fit_transform(niimgs_masked)
	niimgs_detrended0_r2 = masker.fit_transform(niimgs_masked_r2)
	mean_voxels = niimgs_detrended0.mean(1)
	mean_voxels2 = niimgs_detrended0_r2.mean(1)
	np.save(op.join(folder, names[i] + '_voxel%d_mean%dv.npy' % (ind_min, niimgs_detrended0.shape[1])), mean_voxels)
	np.save(op.join(folder, names[i] + '_voxel%d_mean%dv_r2.npy' % (ind_min, niimgs_detrended0.shape[1])), mean_voxels2)

	data = image.get_data()
	datad = scipy.ndimage.binary_dilation(data).astype(data.dtype)
	imaged = nb.Nifti1Image(datad, affine=image.get_affine())
	nb.save(imaged, op.join(folder, names[i] + '_voxel%d_mask_dilated.nii' % ind_min))
	masker = NiftiMasker(mask_img=imaged, detrend=True)
	niimgs_detrended1 = masker.fit_transform(niimgs_masked)
	niimgs_detrended1_r2 = masker.fit_transform(niimgs_masked_r2)
	niimgs_detrended2 = niimgs_detrended1[:, (np.sum(np.abs(niimgs_detrended1), 0) > 0)]
	niimgs_detrended2_r2 = niimgs_detrended1_r2[:, (np.sum(np.abs(niimgs_detrended1), 0) > 0)]
	values = ((np.corrcoef(niimgs_detrended2.T)>0.7).mean(1)>0.5)
	print niimgs_detrended2[:, values].shape
	mean_voxels = np.mean(niimgs_detrended2[:, values], 1)
	print mean_voxels.shape
	mean_voxels2 = np.mean(niimgs_detrended2_r2[:, values], 1)
	np.save(op.join(folder, names[i] + '_voxel%d_mean%dv.npy' % \
						(ind_min, values.sum())), mean_voxels)
	np.save(op.join(folder, names[i] + '_voxel%d_mean%dv_r2.npy' % \
						(ind_min, values.sum())), mean_voxels)

	values = ((np.corrcoef(niimgs_detrended2.T)>0.7).mean(1)>0.5)
	print values
	#print np.corrcoef(niimgs_detrended2.T)>0.7
	#print np.corrcoef(niimgs_detrended2.T)
	#print (np.corrcoef(niimgs_detrended2.T)>0.7).mean(1)

	"""
	datad = scipy.ndimage.binary_dilation(datad).astype(data.dtype)
	imaged = nb.Nifti1Image(datad, affine=image.get_affine())
	nb.save(imaged, names[i] + '_voxel%d_mask_dilated.nii' % ind_min)
	masker = NiftiMasker(mask_img=imaged, detrend=True)
	niimgs_detrended1 = masker.fit_transform(niimgs_masked)
	niimgs_detrended2 = niimgs_detrended1[:, (np.sum(np.abs(niimgs_detrended1), 0) > 0)]
	mean_voxels = np.mean(niimgs_detrended2, 1)
	np.save(names[i] + '_voxel%d_mean%dv.npy' % (ind_min, niimgs_detrended2.shape[1]), mean_voxels)
	print np.corrcoef(niimgs_detrended2.T)
	print np.corrcoef(niimgs_detrended2.T)>0.8
	"""

	"""
	indices_vec = np.zeros_like(niimgs_detrended[0, :])
	indices_vec[inds_mins[:1]] = 1.
	image = masker2.inverse_transform(indices_vec)
	#masker = NiftiMasker(mask_img=image, detrend=True)
	#niimgs_detrended0 = masker.fit_transform(niimgs_masked)
	data = image.get_data()
	#print (data>0).sum()
	datad = scipy.ndimage.morphology.binary_dilation(data, iterations=2).astype(data.dtype)
	#print (datad>0).sum()
	imaged = nb.Nifti1Image(datad, affine=image.get_affine())
	masker = NiftiMasker(mask_img=imaged, detrend=True)
	niimgs_detrended1 = masker.fit_transform(niimgs_masked)
	aux = (np.sum(np.abs(niimgs_detrended1), 0) > 0)
	aa = niimgs_detrended1[:, aux].sum(0) / niimgs_detrended1[:, aux].sum(0)
	niimgs_masked1 = masker.inverse_transform(niimgs_detrended1)
	nb.save(niimgs_masked1, 'after_dilating_' + names[i] + '_masked.nii')
	print niimgs_detrended1.shape
	niimgs_detrended2 = niimgs_detrended1[:, (np.sum(np.abs(niimgs_detrended1), 0) > 0)]
	print niimgs_detrended2.shape
	mean_voxels = np.mean(niimgs_detrended2, 1)
	np.save(names[i] + '_mean_timeseries_small3.npy', mean_voxels)
	"""




