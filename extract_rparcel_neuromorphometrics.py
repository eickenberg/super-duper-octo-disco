import os.path as op
import nibabel as nb
import numpy as np
import scipy

from nilearn.input_data import NiftiMasker
from nistats import experimental_paradigm, design_matrix
from nilearn.input_data import NiftiMasker
from nistats import glm


# File names and paths. Start from AINSI folder.
paradigm_fn = 'onsets.csv'
#bold_fn = op.join('AINSI_002', 'Preprocessed', 's444wuaAINSI_002_EVep2dbolds005a001.nii')
mask_file0 = op.join('AINSI_012', 'Masks', 'rlabels_Neuromorphometrics.nii')
bold_fn = op.join('AINSI_012', 'Preprocessed', 's444rwra1201_WIP_EPI-Rennes_SENSE_4D.nii')
# Acquired with 2x2x4 voxel size
#bold_fn = op.join(folder, 's444rwra1701_WIP_EPI-Ainsi_SENSE_4D.nii')
# Acquired with 3x3x3 voxel size

folderr = 'AINSI_012'

mask_nb = nb.load(mask_file0)
mask = mask_nb.get_data()

mask1 = np.zeros_like(mask)
mask1[mask==201] = 1
print (mask1>0).sum()
mask_nb0 = nb.Nifti1Image(mask1, affine=mask_nb.get_affine())
nb.save(mask_nb0, op.join(folderr,'parcel_audioD.nii'))

mask1 = np.zeros_like(mask)
mask1[mask==200] = 1
print (mask1>0).sum()
mask_nb0 = nb.Nifti1Image(mask1, affine=mask_nb.get_affine())
nb.save(mask_nb0, op.join(folderr,'parcel_audioG.nii'))

mask1 = np.zeros_like(mask)
mask1[mask==157] = 1
print (mask1>0).sum()
mask_nb0 = nb.Nifti1Image(mask1, affine=mask_nb.get_affine())
nb.save(mask_nb0, op.join(folderr,'parcel_visualD.nii'))

mask1 = np.zeros_like(mask)
mask1[mask==156] = 1
print (mask1>0).sum()
mask_nb0 = nb.Nifti1Image(mask1, affine=mask_nb.get_affine())
nb.save(mask_nb0, op.join(folderr,'parcel_visualG.nii'))

mask1 = np.zeros_like(mask)
mask1[mask==177] = 1
print (mask1>0).sum()
mask_nb0 = nb.Nifti1Image(mask1, affine=mask_nb.get_affine())
nb.save(mask_nb0, op.join(folderr,'parcel_motorD.nii'))

mask1 = np.zeros_like(mask)
mask1[mask==176] = 1
print (mask1>0).sum()
mask_nb0 = nb.Nifti1Image(mask1, affine=mask_nb.get_affine())
nb.save(mask_nb0, op.join(folderr,'parcel_motorG.nii'))


