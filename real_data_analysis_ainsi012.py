import os.path as op
import nibabel as nb
import numpy as np

import matplotlib
matplotlib.use('Agg')

from gp import SuperDuperGP, _get_hrf_model
from nilearn.input_data import NiftiMasker
from nistats.glm import FirstLevelGLM
from nistats.design_matrix import _make_drift
from nistats import experimental_paradigm, design_matrix
from scipy.interpolate import interp1d

folder = 'data_ainsi012'
folder0 = 'data_example'

"""studies = [ 'audioD_mean_timeseries', 'audioG_mean_timeseries',
            'visualD_mean_timeseries','visualG_mean_timeseries',
            'motorD_mean_timeseries', 'motorG_mean_timeseries' ]
"""

n_vox = 1
studies = [ 'audioD_voxel193_mean1v', 'audioG_voxel166_mean1v',
            'visualD_voxel18_mean1v', 'visualG_voxel111_mean1v',
            'motorD_voxel481_mean1v', 'motorG_voxel277_mean1v' ]
"""
n_vox = 6
studies = [ 'audioD_voxel193_mean6v', 'audioG_voxel166_mean5v',
            'visualD_voxel2_mean5v', 'visualG_voxel111_mean4v',
            'motorD_voxel481_mean5v', 'motorG_voxel277_mean5v' ]
"""

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
if True:
    for study in studies:
        voxel_fn = op.join(folder, study + '.npy')
        voxel_r2_fn = op.join(folder, study + '_r2.npy')
        # Paradigm file
        paradigm_fn = op.join(folder0, 'onsets.csv')
        ########################################################################
        # Load data and parameters
        t_r = 3.
        ys_ = np.load(voxel_fn)
        ysr2_ = np.load(voxel_r2_fn)
        n_scans = ys.shape[0]

        # Create design matrix
        frametimes = np.arange(0, n_scans * t_r, t_r)
        paradigm = experimental_paradigm.paradigm_from_csv(paradigm_fn)
        dm = design_matrix.make_design_matrix(frametimes, paradigm=paradigm)
        modulation = np.array(paradigm)[:, 4]

        drifts = _make_drift('cosine', frame_times)
        ys = ys_ - drifts.dot(np.linalg.pinv(drifts).dot(ys_))
        ysr2 = ysr2_ - drifts.dot(np.linalg.pinv(drifts).dot(ysr2_))

        # GP parameters
        time_offset = 6
        gamma = 4
        fmin_max_iter = 50
        n_restarts_optimizer = 10
        n_iter = 10
        normalize_y = False
        optimize = False
        zeros_extremes = True

        # Estimation HRF on 1 run
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

        # make the design matrix using estimated hrf
        f_hrf_est = interp1d(hx, hy)

        from data_generator import make_design_matrix_hrf
        dm2 = make_design_matrix_hrf(frametimes, paradigm=paradigm,
        	hrf_length=hrf_length, t_r=t_r, time_offset=time_offset,
        	f_hrf=f_hrf_est)


        # Predict using the betas with GP
        ys_pred, matrix, betas, resid = gp.predict(ysr2, paradigm)


        # Prepare data for GLM
        mask_img = nb.Nifti1Image(np.ones((2, 2, 2)), affine=np.eye(4))
        masker = NiftiMasker(mask_img=mask_img)
        masker.fit()
        ys2 = np.ones((2, 2, 2, ys.shape[0])) * ysr2[np.newaxis, np.newaxis, np.newaxis, :]
        niimgs = nb.Nifti1Image(ys2, affine=np.eye(4))

        # Re-estimate with a GLM on run 2 with dm
        glm = FirstLevelGLM(mask=mask_img, t_r=t_r, standardize=True, noise_model='ols')
        glm.fit(niimgs, dm)
        norm_resid1 = (np.linalg.norm(glm.results_[0][0].resid, axis=0)**2).mean()
        ys_pred_glm1 = glm.results_[0][0].predicted[:, 0]

        # Re-estimate with a GLM on run 2 with dm2
        glm.fit(niimgs, dm2)
        norm_resid2 = (np.linalg.norm(glm.results_[0][0].resid, axis=0)**2).mean()
        ys_pred_glm2 = glm.results_[0][0].predicted[:, 0]

        betaglm = np.linalg.pinv(dm).dot(ys)
        ys_pred_train = dm.dot(betaglm)
        resid =(ys_pred_train - ys) ** 2

        corr_gp = np.corrcoef(ys_pred, ysr2)[1, 0]
        corr_glm1 = np.corrcoef(ys_pred_glm1, ysr2)[1, 0]
        corr_glm2 = np.corrcoef(ys_pred_glm2, ysr2)[1, 0]
        corr_glm0 = np.corrcoef(ys_pred_train, ys)[1, 0]

        print "corr glm1: %s, corr glm2: %s, corr gp: %s, corr glm0: %s" % (corr_glm1, corr_glm2, corr_gp, corr_glm0)

        data = {}
        data['ys'] = ys
        data['study'] = study
        data['ys_pred_train'] = ys_pred_train
        data['ys_pred_glm1'] = ys_pred_glm1
        data['ys_pred_glm2'] = ys_pred_glm2
        data['ys_pred_gp'] = ys_pred
        data['corr_glm0'] = corr_glm0
        data['corr_glm1'] = corr_glm1
        data['corr_glm2'] = corr_glm2
        data['corr_gp'] = corr_gp
        data['hy'] = hy
        data['hx'] = hx
        data['hrf_var'] = hrf_var
        data['norm_resid_gp'] = resid_norm_sq
        data['norm_resid_glm1'] = norm_resid1
        data['norm_resid_glm2'] = norm_resid2

        np.save(op.join(folder, study + '_data.npy'), data)


if True:
    #folder = 'results'
    regions = ['audio right', 'audio left', 'visual right',
        'visual left', 'motor right', 'motor left']
    colors = ['b', 'c', 'r', 'm', 'g', 'lime']

    import matplotlib.pyplot as plt
    # reading the data and plotting
    for istudy, study in enumerate(studies):
        data = np.load(op.join(folder, study + '_data.npy')).item()

        ys = data['ys']
        ys_glm0 = data['ys_pred_train']
        ys_pred_glm1 = data['ys_pred_glm1']
        ys_pred_glm2 = data['ys_pred_glm2']
        ys_pred = data['ys_pred_gp']
        corr_glm0 = data['corr_glm0']
        corr_glm1 = data['corr_glm1']
        corr_glm2 = data['corr_glm2']
        corr_gp = data['corr_gp']
        hy = data['hy']
        hx = data['hx']
        hrf_var = data['hrf_var']

        print "STUDY: %s | corr glm: %.2f | corr glm_h: %.2f | corr gp: %.2f | corr glm_train: %.2f" % \
        				(study, corr_glm1, corr_glm2, corr_gp, corr_glm0)

        hy *= np.sign(hy[np.argmax(np.abs(hy))]) / np.abs(hy).max()
        hrf_0 /= hrf_0.max()

        # Plot HRF
        plt.figure(1)
        #if regions[istudy]=='visual right':
        #	hy = - hy
        plt.fill_between(hx, hy - 1.96 * np.sqrt(hrf_var),
                        hy + 1.96 * np.sqrt(hrf_var),
                        color=colors[istudy], alpha=0.1)
        #plt.plot(hx, hy, label='estimated HRF')
        plt.plot(hx, hy, color=colors[istudy], label='HRF ' + regions[istudy])

        plt.hold('on')

    plt.plot(x_0, hrf_0, color='k', label='glover HRF')
    plt.axis('tight')
    plt.legend()
    fname = op.join(folder, "hrf_regions_joint_small_%dvoxels_gamma%d" % (n_vox, gamma))
    #fname = op.join(folder, "hrf_regions_joint_small_20voxels")
    plt.savefig(fname + ".png")
    plt.savefig(fname + ".pdf")
    plt.savefig(fname + ".svg")
    plt.close()



