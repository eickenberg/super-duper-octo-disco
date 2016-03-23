import numpy as np
import pandas as pd
import nibabel as nb
from sklearn.utils import check_random_state
from nilearn.input_data import NiftiMasker
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from nistats.experimental_paradigm import check_paradigm
from nistats.design_matrix import (make_design_matrix, full_rank, _make_drift)
from nistats.hemodynamic_models import (spm_hrf, glover_hrf, _resample_regressor,
                                        _gamma_difference_hrf)
from hrf import bezier_hrf, physio_hrf
from paradigm import _sample_condition


# XXX putting this here, just because now we are calling
# make_design_matrix_hrf, which is here
###############################################################################
# HRF utils
###############################################################################
def _get_hrf_measurements(paradigm, modulation=None, hrf_length=32., t_r=2,
                          time_offset=10, zeros_extremes=False,
                          frame_times=None):
    """This function:
    Parameters
    ----------
    paradigm : paradigm type
    hrf_length : float
    t_r : float
    time_offset : float
    zeros_extremes : bool

    Returns
    -------
    hrf_measurement_points : list of list
    visible events : list of list
    etas : list of list
    beta_indices : list of list
    unique_events : array-like
    """
    if modulation is None:
        names, onsets, durations, modulation = check_paradigm(paradigm)
    else:
       names, onsets, durations, _ = check_paradigm(paradigm)

    if frame_times is None:
        frame_times = np.arange(0, onsets.max() + time_offset, t_r)

    time_differences = frame_times[:, np.newaxis] - onsets
    scope_masks = (time_differences > 0) & (time_differences < hrf_length)
    belong_to_measurement, which_event = np.where(scope_masks)

    unique_events, event_type_indices = np.unique(names, return_inverse=True)

    if zeros_extremes:
        lft = len(frame_times) + 2
    else:
        lft = len(frame_times)
    hrf_measurement_points = [list() for _ in range(lft)]
    etas = [list() for _ in range(lft)]
    beta_indices = [list() for _ in range(lft)]
    visible_events = [list() for _ in range(lft)]

    for frame_id, event_id in zip(belong_to_measurement, which_event):
        hrf_measurement_points[frame_id].append(time_differences[frame_id,
                                                                 event_id])
        etas[frame_id].append(modulation[event_id])
        beta_indices[frame_id].append(event_type_indices[event_id])
        visible_events[frame_id].append(event_id)

    if zeros_extremes:
        # we add first and last point of the hrf
        hrf_measurement_points[frame_id + 1].append(0.)
        etas[frame_id + 1].append(modulation[event_id])
        beta_indices[frame_id + 1].append(event_type_indices[event_id])
        visible_events[frame_id + 1].append(event_id)
        hrf_measurement_points[frame_id + 2].append(hrf_length)
        etas[frame_id + 2].append(modulation[event_id])
        beta_indices[frame_id + 2].append(event_type_indices[event_id])
        visible_events[frame_id + 2].append(event_id)

    return (hrf_measurement_points, visible_events, etas, beta_indices,
            unique_events)


def _get_design_from_hrf_measures(hrf_measures, beta_indices):
    event_names = np.unique(np.concatenate(beta_indices)).astype('int')

    design = np.zeros([len(beta_indices), len(event_names)])
    pointer = 0

    for beta_ind, row in zip(beta_indices, design):
        measures = hrf_measures[pointer:pointer + len(beta_ind)]
        for i, name in enumerate(event_names):
            row[i] = measures[beta_ind == name].sum()

        pointer += len(beta_ind)
    return design


def _get_hrf_model(hrf_model=None, hrf_length=25., dt=1., normalize=False):
    """Returns HRF created with model hrf_model. If hrf_model is None,
    then a vector of 0 is returned

    Parameters
    ----------
    hrf_model: str
    hrf_length: float
    dt: float
    normalize: bool

    Returns
    -------
    hrf_0: hrf
    """
    if hrf_model == 'glover':
        hrf_0 = glover_hrf(tr=1., oversampling=1./dt, time_length=hrf_length)
    elif hrf_model == 'spm':
        hrf_0 = spm_hrf(tr=1., oversampling=1./dt, time_length=hrf_length)
    elif hrf_model == 'gamma':
        hrf_0 = _gamma_difference_hrf(1., oversampling=1./dt, time_length=hrf_length,
                                      onset=0., delay=6, undershoot=16., dispersion=1.,
                                      u_dispersion=1., ratio=0.167)
    elif hrf_model == 'bezier':
        # Bezier curves. We can indicate where is the undershoot and the peak etc
        hrf_0 = bezier_hrf(hrf_length=hrf_length, dt=dt, pic=[6,1], picw=2,
                           ushoot=[15,-0.2], ushootw=3, normalize=normalize)
    elif hrf_model == 'physio':
        # Balloon model. By default uses the parameters of Khalidov11
        hrf_0 = physio_hrf(hrf_length=hrf_length, dt=dt, normalize=normalize)
    else:
        # Mean 0 if no hrf_model is specified
        hrf_0 = np.zeros(hrf_length/dt)
        warnings.warn("The HRF model is not recognized, setting it to None")
    if normalize and hrf_model is not None:
        hrf_0 = hrf_0 / np.linalg.norm(hrf_0)
    return hrf_0


###############################################################################
def generate_mask_condition(n_x=10, n_y=10, n_z=10, sigma=1., threshold=0.5,
                            seed=None):
    """

    Parameters
    ----------
    n_x : int
    n_y : int
    n_z : int
    sigma : float
    threshold : float [0, 1]
    seed : int

    Returns
    -------
    mask_img : bool array of shape (n_x, n_y, n_z)
    """
    rng = check_random_state(seed)

    image = rng.rand(n_x, n_y, n_z)
    for k in [0, 1, 2]:
        gaussian_filter1d(image, sigma=sigma, output=image, axis=k)
    max_img, min_img = image.max(), image.min()
    image -= min_img
    image /= max_img - min_img
    mask_img = image > threshold
    return mask_img


def make_design_matrix_hrf(
    frame_times, paradigm=None, hrf_length=32., t_r=2., time_offset=10,
    drift_model='cosine', period_cut=128, drift_order=1, fir_delays=[0],
    add_regs=None, add_reg_names=None, min_onset=-24, f_hrf=None):
    """Generate a design matrix from the input parameters

    Parameters
    ----------
    frame_times : array of shape (n_frames,)
        The timing of the scans in seconds.

    paradigm : DataFrame instance, optional
        Description of the experimental paradigm.

    drift_model : string, optional
        Specifies the desired drift model,
        It can be 'polynomial', 'cosine' or 'blank'.

    period_cut : float, optional
        Cut period of the low-pass filter in seconds.

    drift_order : int, optional
        Order of the drift model (in case it is polynomial).

    fir_delays : array of shape(n_onsets) or list, optional,
        In case of FIR design, yields the array of delays used in the FIR
        model.

    add_regs : array of shape(n_frames, n_add_reg), optional
        additional user-supplied regressors

    add_reg_names : list of (n_add_reg,) strings, optional
        If None, while n_add_reg > 0, these will be termed
        'reg_%i', i = 0..n_add_reg - 1

    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.

    Returns
    -------
    design_matrix : DataFrame instance,
        holding the computed design matrix
    """
    # check arguments
    # check that additional regressor specification is correct
    n_add_regs = 0
    if add_regs is not None:
        if add_regs.shape[0] == np.size(add_regs):
            add_regs = np.reshape(add_regs, (np.size(add_regs), 1))
        n_add_regs = add_regs.shape[1]
        assert add_regs.shape[0] == np.size(frame_times), ValueError(
            'incorrect specification of additional regressors: '
            'length of regressors provided: %s, number of ' +
            'time-frames: %s' % (add_regs.shape[0], np.size(frame_times)))

    # check that additional regressor names are well specified
    if add_reg_names is None:
        add_reg_names = ['reg%d' % k for k in range(n_add_regs)]
    elif len(add_reg_names) != n_add_regs:
        raise ValueError(
            'Incorrect number of additional regressor names was provided'
            '(%s provided, %s expected) % (len(add_reg_names),'
            'n_add_regs)')

    # computation of the matrix
    names = []
    matrix = None

    # step 1: paradigm-related regressors
    if paradigm is not None:
        # create the condition-related regressors
        names0, _, _, _ = check_paradigm(paradigm)
        names = np.append(names, np.unique(names0))
        hrf_measurement_points, _, _, beta_indices, _ = \
                    _get_hrf_measurements(paradigm, hrf_length=hrf_length,
                                          t_r=t_r, time_offset=time_offset, frame_times=frame_times)
        hrf_measurement_points = np.concatenate(hrf_measurement_points)
        hrf_measures = f_hrf(hrf_measurement_points)
        matrix = _get_design_from_hrf_measures(hrf_measures, beta_indices)
        #matrix, names = _convolve_regressors(
        #    paradigm, hrf_model.lower(), frame_times, fir_delays, min_onset)

    # step 2: additional regressors
    if add_regs is not None:
        # add user-supplied regressors and corresponding names
        if matrix is not None:
            matrix = np.hstack((matrix, add_regs))
        else:
            matrix = add_regs
        names = np.append(names, add_reg_names)

    # step 3: drifts
    drift, dnames = _make_drift(drift_model.lower(), frame_times, drift_order,
                                period_cut)

    if matrix is not None:
        matrix = np.hstack((matrix, drift))
    else:
        matrix = drift

    names = np.append(names, dnames)

    # step 4: Force the design matrix to be full rank at working precision
    matrix, _ = full_rank(matrix)

    design_matrix = pd.DataFrame(
        matrix, columns=list(names), index=frame_times)
    return design_matrix


def generate_spikes_time_series(n_events=200, n_blank_events=50,
                                event_spacing=6, t_r=2, hrf_length=32.,
                                event_types=['ev1', 'ev2'], period_cut=64,
                                jitter_min=-1, jitter_max=1, drift_order=1,
                                return_jitter=False, time_offset=10,
                                modulation=None, seed=None, f_hrf=None):
    """Voxel-level activations

    Parameters
    ----------
    n_events
    n_blank_events
    event_spacing
    t_r
    event_types
    period_cut
    jitter_min
    jitter_max
    return_jitter
    time_offset
    modulation
    seed


    Returns
    -------
    paradigm
    design
    modulation
    measurement_times
    """

    rng = check_random_state(seed)
    event_types = np.array(event_types)

    all_times = (1. + np.arange(n_events + n_blank_events)) * event_spacing
    non_blank_events = rng.permutation(len(all_times))[:n_events]
    onsets = np.sort(all_times[non_blank_events])

    names = event_types[rng.permutation(n_events) % len(event_types)]
    measurement_times = np.arange(0., onsets.max() + time_offset, t_r)

    if modulation is None:
        modulation = np.ones_like(onsets)

    # Jittered paradigm
    if return_jitter:
        onsets += rng.uniform(jitter_min, jitter_max, len(onsets))

    paradigm = pd.DataFrame.from_dict(dict(onset=onsets, name=names))

    if f_hrf is None:
        design = make_design_matrix(measurement_times, paradigm=paradigm,
                                    period_cut=period_cut,
                                    drift_order=drift_order)
    else:
        design = make_design_matrix_hrf(measurement_times, paradigm=paradigm,
                                        period_cut=period_cut,
                                        drift_order=drift_order,
                                        hrf_length=hrf_length,
                                        t_r=t_r, time_offset=time_offset,
                                        f_hrf=f_hrf)

    return paradigm, design, modulation, measurement_times


def generate_fmri(n_x, n_y, n_z, modulation=None, betas=None, n_events=200,
                  n_blank_events=50, event_spacing=6, t_r=2, hrf_length=25.,
                  smoothing_fwhm=2, event_types=['ev1', 'ev2'], drift_order=1,
                  period_cut=64, time_offset=10, sigma_noise=0.001, sigma=None,
                  threshold=None, seed=None, f_hrf=None):
    """

    Parameters
    ----------
    n_x
    n_y
    n_z
    modulation
    betas
    n_events
    n_blank_events
    event_spacing
    t_r
    smoothing_fwhm
    event_types
    period_cut
    time_offset
    sigma_noise
    sigma
    threshold
    seed


    Returns
    -------
    fmri_timeseries
    paradigm
    design
    images
    """

    rng = check_random_state(seed)
    event_types = np.array(event_types)

    smoothing_fwhm = (smoothing_fwhm, ) * 3
    sigma_ratio = np.sqrt(8 * np.log(2))
    sigma_smoothing = smoothing_fwhm / sigma_ratio

    paradigm, design, modulation, measurement_times = generate_spikes_time_series(
        n_events=n_events, n_blank_events=n_blank_events,
        event_spacing=event_spacing, t_r=t_r, event_types=event_types,
        period_cut=period_cut, time_offset=time_offset, modulation=modulation,
        seed=seed, f_hrf=f_hrf, hrf_length=hrf_length, drift_order=drift_order)

    n_volumes, n_regressors = design.shape

    if betas is None:
        betas = rng.rand(n_regressors)

    fmri_timeseries = np.zeros((n_x, n_y, n_y, n_volumes))
    # Generate one image per condition
    masks = {}
    for i, condition in enumerate(event_types):
        masks[condition] = generate_mask_condition(n_x, n_y, n_z, sigma=sigma,
                                                   threshold=threshold,
                                                   seed=seed+i)

    # Assign a temporal series to each image
    for i, condition in enumerate(event_types):
        ind = np.where(event_types != condition)[0]
        events = event_types[ind]
        cond_design = design.copy()
        cond_design[events] = 0
        fmri_timeseries[masks[condition], :] = cond_design.dot(betas)

        for k, s in enumerate(sigma_smoothing):
            gaussian_filter1d(fmri_timeseries, sigma=s,
                              output=fmri_timeseries, axis=k)

        noise = rng.randn(n_x, n_y, n_z, n_volumes)
        scale_factor = (np.linalg.norm(fmri_timeseries[masks[condition], :], axis=1) \
                        / np.linalg.norm(noise[masks[condition], :], axis=1)).mean()
        fmri_timeseries += sigma_noise * noise * scale_factor

    return fmri_timeseries, paradigm, design, masks


if __name__ == "__main__":
    from nistats.glm import FirstLevelGLM
    from nilearn.plotting import find_cuts
    import matplotlib.pyplot as plt

    plt.close('all')

    n_x, n_y, n_z = 20, 20, 20
    event_types = ['ev1', 'ev2']
    n_events = 100
    n_blank_events = 50
    event_spacing = 6
    t_r = 2
    smoothing_fwhm = 1
    sigma = 2
    sigma_noise = 0.01
    threshold = 0.7
    seed = 42

    mask_img = nb.Nifti1Image(np.ones((n_x, n_y, n_z)), affine=np.eye(4))
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()

    fmri, paradigm, design, masks = generate_fmri(
        n_x=n_x, n_y=n_y, n_z=n_y, modulation=None, n_events=n_events,
        event_types=event_types, n_blank_events=n_blank_events,
        event_spacing=event_spacing, t_r=t_r, smoothing_fwhm=smoothing_fwhm,
        sigma=sigma, sigma_noise=sigma_noise, threshold=threshold, seed=seed)

    niimgs = nb.Nifti1Image(fmri, affine=np.eye(4))
    # Testing with a GLM
    glm = FirstLevelGLM(mask=mask_img, t_r=t_r, standardize=True,
                        noise_model='ols')
    glm.fit(niimgs, design)

    contrast_matrix = np.eye(design.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design.columns)])

    z_maps = {}
    for condition_id in event_types:
        z_maps[condition_id] = glm.transform(contrasts[condition_id],
                                             contrast_name=condition_id,
                                             output_z=True, output_stat=False,
                                             output_effects=False,
                                             output_variance=False)

    fig, axx = plt.subplots(nrows=len(event_types), ncols=2, figsize=(8, 8))

    for i, ((cond_id, mask), (condition_id, z_map)) in enumerate(
        zip(masks.items(), z_maps.items())):

        img_z_map = z_map[0].get_data()
        niimg = nb.Nifti1Image(mask.astype('int'), affine=np.eye(4))
        cuts = find_cuts.find_cut_slices(niimg)
        axx[i, 0].imshow(mask[..., cuts[0]])
        axx[i, 1].imshow(img_z_map[..., cuts[0]])
        axx[i, 1].set_title('z map: %s' % condition_id)
        axx[i, 0].set_title('ground truth: %s' % condition_id)
        axx[i, 0].axis('off')
        axx[i, 1].axis('off')

    plt.hot()
    plt.show()








