import numpy as np
import pandas as pd
import nibabel as nb
from sklearn.utils import check_random_state
from nilearn.input_data import NiftiMasker
from scipy.ndimage.filters import gaussian_filter1d
from nistats.design_matrix import make_design_matrix


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


def generate_spikes_time_series(n_events=200, n_blank_events=50,
                                event_spacing=6, t_r=2,
                                event_types=['ev1', 'ev2'], period_cut=64,
                                jitter_min=-1, jitter_max=1,
                                return_jitter=False, time_offset=10,
                                modulation=None, seed=None):
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
    design = make_design_matrix(measurement_times, paradigm=paradigm,
                                period_cut=period_cut)

    return paradigm, design, modulation, measurement_times


def generate_fmri(n_x, n_y, n_z, modulation=None, betas=None, n_events=200,
                  n_blank_events=50, event_spacing=6, t_r=2,
                  smoothing_fwhm=2, event_types=['ev1', 'ev2'],
                  period_cut=64, time_offset=10, sigma_noise=0.001, sigma=None,
                  threshold=None, seed=None):
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
        seed=seed)

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

        fmri_timeseries += sigma_noise * rng.randn(n_x, n_y, n_z, n_volumes)

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








