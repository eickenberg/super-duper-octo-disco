import os.path as op
import nibabel as nb
import numpy as np

import matplotlib
matplotlib.use('Agg')

from gp import SuperDuperGP, _get_hrf_model
from nilearn.input_data import NiftiMasker
from nistats.glm import FirstLevelGLM
from nistats import experimental_paradigm, design_matrix
from scipy.interpolate import interp1d

folder = 'data_example2'
folder0 = 'data_example'

studies = [
    'audioD_voxel743_mean1v', 'audioD_voxel743_mean5v',
    'audioG_voxel145_mean1v', 'audioG_voxel145_mean7v',
    'visualD_voxel201_mean1v','visualD_voxel201_mean6v',
    'visualG_voxel46_mean1v', 'visualG_voxel46_mean5v',
    'motorD_voxel638_mean1v', 'motorD_voxel638_mean6v',
    'motorG_voxel805_mean1v', 'motorG_voxel805_mean6v'
    ]
studies = ['motorD_mean_timeseries', 'motorG_mean_timeseries']
#'audioD_mean_timeseries', 'audioG_mean_timeseries'
#'visualD_mean_timeseries','visualG_mean_timeseries',
#'motorD_mean_timeseries', 'motorG_mean_timeseries'

# STUDY: audioD_mean_timeseries | corr glm: 0.70 | corr gp: 0.86
# STUDY: audioG_mean_timeseries | corr glm: 0.78 | corr gp: 0.88
# STUDY: visualD_mean_timeseries | corr glm: 0.77 | corr gp: 0.81
# STUDY: visualG_mean_timeseries | corr glm: 0.75 | corr gp: 0.81
# STUDY: motorD_mean_timeseries | corr glm: 0.74 | corr gp: 0.71
# STUDY: motorG_mean_timeseries | corr glm: 0.74 | corr gp: 0.74

studies = ['audioD_mean_timeseries', 'audioG_mean_timeseries',
           'visualD_mean_timeseries','visualG_mean_timeseries',
           'motorD_mean_timeseries', 'motorG_mean_timeseries']
regions = ['audio right', 'audio left', 'visual right',
		   'visual left', 'motor right', 'motor right']

#studies = ['audioD_mean_timeseries_small', 'audioG_mean_timeseries_small',
#           'visualD_mean_timeseries_small','visualG_mean_timeseries_small',
#           'motorD_mean_timeseries_small', 'motorG_mean_timeseries_small']

# Define HRF of mean GP
hrf_length = 32
dt = 0.1
x_0 = np.arange(0, hrf_length  + dt, dt)
hrf_ushoot = 16.
hrf_model = 'glover'
hrf_0 = _get_hrf_model(hrf_model, hrf_length=hrf_length + dt,
                    			  dt=dt, normalize=True)
f_hrf = interp1d(x_0, hrf_0)

# This is just a flag to be able to use the same script for the plotting
if False:
    for study in studies:
        voxel_fn = op.join(folder, study + '.npy')
        # Paradigm file
        paradigm_fn = op.join(folder0, 'onsets.csv')
        ########################################################################
        # Load data and parameters
        n_scans = 144
        t_r = 3.
        ys = np.load(voxel_fn)

        # Create design matrix
        frametimes = np.arange(0, n_scans * t_r, t_r)
        paradigm = experimental_paradigm.paradigm_from_csv(paradigm_fn)
        dm = design_matrix.make_design_matrix(frametimes, paradigm=paradigm)
        modulation = np.array(paradigm)[:, 4]

        # GP parameters
        time_offset = 10
        gamma = 10.
        fmin_max_iter = 50
        n_restarts_optimizer = 10
        n_iter = 3
        normalize_y = False
        optimize = True
        zeros_extremes = True

        # Estimation
        gp = SuperDuperGP(hrf_length=hrf_length, t_r=t_r, oversampling=1./dt,
                          gamma=gamma, modulation=modulation,
                          fmin_max_iter=fmin_max_iter, sigma_noise=1.0,
                          time_offset=time_offset, n_iter=n_iter,
                          normalize_y=normalize_y, verbose=True,
                          optimize=optimize,
                          n_restarts_optimizer=n_restarts_optimizer,
                          zeros_extremes=zeros_extremes, f_mean=f_hrf)

        (hx, hy, hrf_var, resid_norm_sq, sigma_sq_resid) = gp.fit(ys, paradigm)

        print 'residual norm square = ', resid_norm_sq

        # Testing with a GLM
        mask_img = nb.Nifti1Image(np.ones((2, 2, 2)), affine=np.eye(4))
        masker = NiftiMasker(mask_img=mask_img)
        masker.fit()
        ys2 = np.ones((2, 2, 2, ys.shape[0])) * ys[np.newaxis, np.newaxis, np.newaxis, :]
        niimgs = nb.Nifti1Image(ys2, affine=np.eye(4))
        glm = FirstLevelGLM(mask=mask_img, t_r=t_r, standardize=True, noise_model='ols')
        glm.fit(niimgs, dm)
        norm_resid = (np.linalg.norm(glm.results_[0][0].resid, axis=0)**2).mean()
        ys_pred_glm = glm.results_[0][0].predicted[:, 0]

        # Predict GP
        # XXX: Do we need to predict for GLM???
        ys_pred, matrix, betas, resid = gp.predict(ys, paradigm)

        corr_gp = np.corrcoef(ys_pred, ys)[1, 0]
        corr_glm = np.corrcoef(ys_pred_glm, ys)[1, 0]

        print "corr glm: %s, corr gp: %s" % (corr_glm, corr_gp)

        data = {}
        data['ys'] = ys
        data['study'] = study
        data['ys_pred_glm'] = ys_pred_glm
        data['ys_pred_gp'] = ys_pred
        data['corr_glm'] = corr_glm
        data['corr_gp'] = corr_gp
        data['hy'] = hy
        data['hx'] = hx
        data['hrf_var'] = hrf_var
        data['norm_resid_gp'] = resid_norm_sq
        data['norm_resid_glm'] = norm_resid

        np.save(op.join(folder, study + '_data.npy'), data)


if True:
    import matplotlib.pyplot as plt
    # reading the data and plotting
    for study in studies:
        data = np.load(op.join(folder, study + '_data.npy')).item()

        ys = data['ys']
        ys_pred_glm = data['ys_pred_glm']
        ys_pred = data['ys_pred_gp']
        corr_glm = data['corr_glm']
        corr_gp = data['corr_gp']
        hy = data['hy']
        hx = data['hx']
        hrf_var = data['hrf_var']

        print "STUDY: %s | corr glm: %.2f | corr gp: %.2f" % (study, corr_glm,
                                                              corr_gp)

        hy *= np.sign(hy[np.argmax(np.abs(hy))]) / np.abs(hy).max()
        hrf_0 /= hrf_0.max()

        # Plot HRF
        plt.figure(1)
        plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                        hy + 1.96 * np.sqrt(hrf_var), alpha=0.1)
        plt.plot(hx, hy, label='estimated HRF')
        plt.plot(x_0, hrf_0, label='glover HRF')
        plt.axis('tight')
        plt.legend()
        fname = op.join(folder, study + "_hrf")
        plt.savefig(fname + ".png")
        plt.savefig(fname + ".pdf")
        plt.savefig(fname + ".svg")
        plt.close()

        # Plot predicted signal
        plt.figure(2)
        nm = np.abs([ys.max(), ys.min()]).max()
        plt.plot(ys/nm, 'r', label='acquired')
        nm = np.abs([ys_pred.max(), ys_pred.min()]).max()
        plt.plot(ys_pred/nm, 'b', label='predicted GP')
        nm = np.abs([ys_pred_glm.max(), ys_pred_glm.min()]).max()
        plt.plot(ys_pred_glm/nm, 'g', label='predicted GLM')
        plt.axis('tight')
        plt.legend()
        plt.show()
        fname = op.join(folder, study + "_fitting")
        plt.savefig(fname + ".png")
        plt.savefig(fname + ".pdf")
        plt.savefig(fname + ".svg")
        plt.close()

        # import pdb; pdb.set_trace()  # XXX BREAKPOINT


