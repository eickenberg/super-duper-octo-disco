import numpy as np
import pandas

from scipy.interpolate import interp1d

from data_generator import  generate_spikes_time_series as generate_experiment
from data_generator import make_design_matrix_hrf
from nistats.hemodynamic_models import _gamma_difference_hrf

from sklearn.gaussian_process.kernels import RBF

n_events=90
n_blank_events=15
event_spacing=6
t_r=2
hrf_length=32.
event_types=['ev1', 'ev2']
jitter_min=-1
jitter_max=1
time_offset = 20
modulation=None


n_runs = 2
hrf_peak_locations = np.array([3, 4, 5, 6, 7, 8])


paradigms, _, _, measurement_times = list(zip(*[
    generate_experiment(n_events=n_events,
                        n_blank_events=n_blank_events,
                        event_spacing=event_spacing,
                        t_r=t_r, hrf_length=hrf_length,
                        event_types=event_types,
                        jitter_min=jitter_min,
                        jitter_max=jitter_max,
                        return_jitter=True,
                        time_offset=time_offset,
                        modulation=modulation,
                        seed=seed) for seed in np.arange(n_runs)]))


rng = np.random.RandomState(42)

noise_levels = np.array([0., .1, 1., 2., 5., 10.])

beta = rng.randn(len(event_types))

n_new_betas = 4

new_betas = rng.randn(len(event_types), n_new_betas) # these are to see how well the hrf spans the space
frame_times_run = np.arange(0, paradigms[0]['onset'].max() + time_offset, t_r)

n_noises = 2
noise_vectors = [rng.randn(len(frame_times_run), n_noises) for _ in range(n_runs)]
noise_vectors = [noise_vector / np.linalg.norm(noise_vector, axis=0)
                 for noise_vector in noise_vectors]

gammas = [1., 2., 4.]
def zero_mean(x):
    return np.zeros_like(x)

from mini_algo import alternating_optimization


def get_values(simulation_peak, estimation_peak, held_out_index,
               noise_level, noise_vector_list,
               paradigms=paradigms, frame_times_run=frame_times_run,
               beta=beta, new_betas=new_betas):
    simulation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
                                           time_length=hrf_length,
                                           undershoot=16., delay=simulation_peak)
    xs = np.linspace(0., hrf_length + 1, len(simulation_hrf), endpoint=False)
    f_sim_hrf = interp1d(xs, simulation_hrf)
    shifted_paradigms =  [paradigm.copy() 
                          for i, paradigm in enumerate(paradigms)
                          if i != held_out_index]
    shifted_frame_times = []
    offset = 0
    # shift paradigms to concatenate them
    for paradigm in shifted_paradigms:
        paradigm_length = paradigm['onset'].max()
        paradigm['onset'] += offset
        shifted_frame_times.append(frame_times_run + offset)
        offset += paradigm_length + time_offset
    shifted_frame_times = np.concatenate(shifted_frame_times)

    train_paradigm = pandas.concat(shifted_paradigms)
    test_paradigm = paradigms[held_out_index]
    train_noise = np.concatenate(
        [noise for i, noise in enumerate(noise_vector_list)
         if i != held_out_index])
    scaled_train_noise = train_noise[:, np.newaxis] * noise_level 

    #test_noise = noise_vectors[held_out_index]

    # design matrix dataframes
    train_design_gen_df = make_design_matrix_hrf(shifted_frame_times,
                                                 train_paradigm, f_hrf=f_sim_hrf)
    test_design_gen_df = make_design_matrix_hrf(frame_times_run,
                                                test_paradigm, f_hrf=f_sim_hrf)
    # design matrix without drifts
    train_design_gen = train_design_gen_df[event_types].values
    test_design_gen = test_design_gen_df[event_types].values
    y_train_clean = train_design_gen.dot(beta)
    y_train_norm = np.linalg.norm(y_train_clean) ** 2
    y_train_noisy = y_train_clean[:, np.newaxis] + np.linalg.norm(y_train_clean) * scaled_train_noise
    y_train_noisy_norm = np.linalg.norm(y_train_noisy, axis=0) ** 2
    #train_signal_norm[i_sim, held_out_index, :] = y_train_noisy_norm

    y_test = test_design_gen.dot(beta)
    y_test_new = test_design_gen.dot(new_betas)

    y_test_norm = np.linalg.norm(y_test) ** 2
    y_test_new_norm = np.linalg.norm(y_test_new, axis=0) ** 2

    #test_signal_norm[i_sim, held_out_index, :] = y_test_norm
    #new_test_signal_norm[i_sim, held_out_index, :] = y_test_new_norm

    beta_hat_gen = np.linalg.pinv(train_design_gen).dot(y_train_noisy)
    train_gen_resid = np.linalg.norm(y_train_noisy -
                                     train_design_gen.dot(beta_hat_gen), axis=0) ** 2
    #train_gen_train_gen[i_sim, held_out_index, :] = train_gen_resid
    y_pred_gen = test_design_gen.dot(beta_hat_gen)
    test_gen_resid = np.linalg.norm(y_test[:, np.newaxis] - y_pred_gen) ** 2
    #train_gen_test_gen[i_sim, held_out_index, :] = test_gen_resid

    #print("Generation peak {} Estimation peak {} Fold {}".format(simulation_peak, estimation_peak, held_out_index))
    estimation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
                                           time_length=hrf_length,
                                           undershoot=16, delay=estimation_peak)
    f_est_hrf = interp1d(xs, estimation_hrf)
    # design matrix dataframes
    train_design_est_df = make_design_matrix_hrf(shifted_frame_times,
                                                 train_paradigm, f_hrf=f_est_hrf)
    test_design_est_df = make_design_matrix_hrf(frame_times_run,
                                                test_paradigm, f_hrf=f_est_hrf)
    # design matrix without drifts
    train_design_est = train_design_est_df[event_types].values
    test_design_est = test_design_est_df[event_types].values

    beta_hat_est = np.linalg.pinv(train_design_est).dot(y_train_noisy)
    train_est_resid = np.linalg.norm(y_train_noisy -
                                     train_design_est.dot(beta_hat_est), axis=0) ** 2
    #train_gen_train_est[i_sim, i_est, held_out_index, :] = train_est_resid
    y_pred_est = test_design_est.dot(beta_hat_est)

    test_est_resid = np.linalg.norm(y_test[:, np.newaxis] - y_pred_est, axis=0) ** 2
    #train_gen_test_est[i_sim, i_est, held_out_index, :] = test_est_resid

    y_test_squashed = test_design_est.dot(np.linalg.pinv(test_design_est).dot(y_test))
    test_squashed_resid = np.linalg.norm(y_test - y_test_squashed, axis=0) ** 2
    #test_est_test_est[i_sim, i_est, held_out_index, :] = test_squashed_resid

    # now for some crazy hrf fitting
    output = alternating_optimization(
        train_paradigm, y_train_noisy,
        hrf_length,
        frame_times=shifted_frame_times,
        mean=f_est_hrf,
        n_alternations=10,
        sigma_squared=1,
        rescale_hrf=False,
        optimize_kernel=True,
        optimize_sigma_squared=False)

    (betas, (hrf_measurement_points, hrf_measures),
     residuals,
     hrfs, lls, grads, looes, thetas, sigmas_squared) = output

    hrf_measurement_points = np.concatenate(hrf_measurement_points)
    order = np.argsort(hrf_measurement_points)
    hrf_measurement_points = hrf_measurement_points[order]
    hrf_measures = hrf_measures[order]
    extra_points = np.array([0., -.1, hrf_length, hrf_length + 1.])
    extra_values = np.zeros_like(extra_points)
    hrf_func = interp1d(np.concatenate([hrf_measurement_points, extra_points]),
                        np.concatenate([hrf_measures, extra_points]))
    fitted_train_resid = residuals[-1]
    fitted_train_design_ = make_design_matrix_hrf(frame_times_run, train_paradigm,
                                                f_hrf=hrf_func)
    fitted_test_design_ = make_design_matrix_hrf(frame_times_run, test_paradigm,
                                                f_hrf=hrf_func)
    fitted_train_design = fitted_train_design_[event_types].values
    fitted_test_design = fitted_test_design_[event_types].values
    ftd_sparsity = np.abs(fitted_test_design).sum(axis=0) / np.sqrt((fitted_test_design ** 2).sum(axis=0))
    if (ftd_sparsity < 2.).any():
        #print('Spike at sim {} est {} ho {} noise {}'.format(simulation_peak, estimation_peak, held_out_index, noise_level))
        #print('Removing strongest entry')
        spiking = ftd_sparsity < 2.
        d = fitted_test_design[:, spiking]
        location = np.abs(d).argmax(0)
        d[location, np.arange(len(location))] = .5 * (d[location - 1, np.arange(len(location))] +
                                                      d[location + 1, np.arange(len(location))])
        fitted_test_design[:, spiking] = d
    reest_betas = np.linalg.pinv(fitted_train_design).dot(y_train_noisy)
#    fitted_test_pred = fitted_test_design.dot(reest_betas)
    fitted_test_pred = fitted_test_design.dot(betas)
    fitted_test_resid = np.linalg.norm(y_test - fitted_test_pred) ** 2

    fitted_test_squashed_pred = fitted_test_design.dot(
        np.linalg.pinv(fitted_test_design).dot(y_test_new))
    fitted_test_squashed_resid = np.linalg.norm(
        y_test_new - fitted_test_squashed_pred, axis=0) ** 2


    # now do exactly the same thing again for 0 mean ...
    # I know it is redundant
    output = alternating_optimization(
        train_paradigm, y_train_noisy,
        hrf_length,
        frame_times=shifted_frame_times,
        mean=zero_mean,
        n_alternations=10,
        sigma_squared=1,
        rescale_hrf=False,
        optimize_kernel=True,
        optimize_sigma_squared=False)

    (zm_betas, (zm_hrf_measurement_points, zm_hrf_measures),
     zm_residuals,
     zm_hrfs, zm_lls, zm_grads, zm_looes, zm_thetas, zm_sigmas_squared) = output

    zm_hrf_measurement_points = np.concatenate(zm_hrf_measurement_points)
    order = np.argsort(zm_hrf_measurement_points)
    zm_hrf_measurement_points = zm_hrf_measurement_points[order]
    zm_hrf_measures = zm_hrf_measures[order]
    extra_points = np.array([0., -.1, hrf_length, hrf_length + 1.])
    extra_values = np.zeros_like(extra_points)
    zm_hrf_func = interp1d(np.concatenate([zm_hrf_measurement_points, extra_points]),
                        np.concatenate([zm_hrf_measures, extra_points]))
    zm_fitted_train_resid = zm_residuals[-1]
    zm_fitted_train_design_ = make_design_matrix_hrf(frame_times_run, train_paradigm,
                                                f_hrf=zm_hrf_func)
    zm_fitted_test_design_ = make_design_matrix_hrf(frame_times_run, test_paradigm,
                                                f_hrf=zm_hrf_func)
    zm_fitted_train_design = zm_fitted_train_design_[event_types].values
    zm_fitted_test_design = zm_fitted_test_design_[event_types].values
    ftd_sparsity = np.abs(zm_fitted_test_design).sum(axis=0) / np.sqrt((zm_fitted_test_design ** 2).sum(axis=0))
    if (ftd_sparsity < 2.).any():
        #print('Spike at sim {} est {} ho {} noise {}'.format(simulation_peak, estimation_peak, held_out_index, noise_level))
        #print('Removing strongest entry')
        spiking = ftd_sparsity < 2.
        d = zm_fitted_test_design[:, spiking]
        location = np.abs(d).argmax(0)
        d[location, np.arange(len(location))] = .5 * (d[location - 1, np.arange(len(location))] +
                                                      d[location + 1, np.arange(len(location))])
        zm_fitted_test_design[:, spiking] = d
    zm_reest_betas = np.linalg.pinv(zm_fitted_train_design).dot(y_train_noisy)
#    zm_fitted_test_pred = zm_fitted_test_design.dot(zm_reest_betas)
    zm_fitted_test_pred = zm_fitted_test_design.dot(zm_betas)
    zm_fitted_test_resid = np.linalg.norm(y_test - zm_fitted_test_pred) ** 2

    zm_fitted_test_squashed_pred = zm_fitted_test_design.dot(
        np.linalg.pinv(zm_fitted_test_design).dot(y_test_new))
    zm_fitted_test_squashed_resid = np.linalg.norm(
        y_test_new - zm_fitted_test_squashed_pred, axis=0) ** 2


    return (train_paradigm, test_paradigm,
            beta, betas, zm_betas, reest_betas, zm_reest_betas, 
            y_train_noisy, y_test, y_train_norm, y_train_noisy_norm, y_test_norm, y_test_new_norm,
            train_gen_resid, test_gen_resid, train_est_resid, test_est_resid,
            test_squashed_resid, fitted_train_resid, fitted_test_resid,
            fitted_test_squashed_resid, hrf_measurement_points, hrf_measures,
            zm_fitted_train_resid, zm_fitted_test_resid,
            zm_fitted_test_squashed_resid, zm_hrf_measurement_points, zm_hrf_measures,
            train_design_gen, test_design_gen, train_design_est, test_design_est,
            fitted_train_design, fitted_test_design, zm_fitted_train_design, zm_fitted_test_design)


from sklearn.externals.joblib import Parallel, delayed, Memory

mem = Memory(cachedir=None)
mem_get_values = mem.cache(get_values)

from itertools import product

parameters = product(hrf_peak_locations, hrf_peak_locations,
                     range(n_runs), noise_levels, range(n_noises))

print('Starting Parallel for {} parameter settings'.format(len(hrf_peak_locations) ** 2 * n_runs * len(noise_levels) * n_noises))

results = Parallel(n_jobs=48)(delayed(mem_get_values)(
    simulation_peak, estimation_peak, held_out_index, noise_level,
    [nois[:, i] for nois in noise_vectors])
    for simulation_peak, estimation_peak,
                              held_out_index, noise_level, i in parameters)


# def get_values(simulation_peak, estimation_peak, held_out_index,
#                noise_level, noise_vector_list,

def reshaper(x):
    return x.reshape(len(hrf_peak_locations), len(hrf_peak_locations),
                     n_runs, len(noise_levels), -1)

(beta, betas, zm_betas, reest_betas, zm_reest_betas, 
y_train_noisy, y_test, y_train_norms, y_train_noisy_norms, y_test_norms, y_test_new_norms,
 train_gen_resids, test_gen_resids, train_est_resids, test_est_resids,
 test_squashed_resids,fitted_train_resids, fitted_test_resids,
 fitted_test_squashed_resids,
 hrf_measurement_points, hrf_measures, zm_fitted_train_resid, zm_fitted_test_resid,
 zm_fitted_test_squashed_resid, zm_hrf_measurement_points, zm_hrf_measures,
 train_design_gen, test_design_gen, train_design_est, test_design_est,
 fitted_train_design, fitted_test_design, zm_fitted_train_design, zm_fitted_test_design
) = map(reshaper, map(np.array, list(zip(*results))[2:]))


train_paradigms, test_paradigms = list(zip(*results))[:2]




# train_gen_train_est = np.zeros(
#     (len(hrf_peak_locations),
#      len(hrf_peak_locations),
#      n_runs,
#      n_noises * len(noise_levels)))
# train_gen_train_gen = np.zeros(
#     (len(hrf_peak_locations),
#      n_runs,
#      n_noises * len(noise_levels)))

# train_gen_test_est = np.zeros(
#     (len(hrf_peak_locations),
#      len(hrf_peak_locations),
#      n_runs,
#      n_noises * len(noise_levels)))
# train_gen_test_gen = np.zeros(
#     (len(hrf_peak_locations),
#      n_runs,
#      n_noises * len(noise_levels)))

# test_est_test_est = np.zeros(
#     (len(hrf_peak_locations),
#      len(hrf_peak_locations),
#      n_runs,
#      n_new_betas))

# train_signal_norm = np.zeros(
#     (len(hrf_peak_locations),
#      n_runs,
#      n_noises * len(noise_levels)))
# test_signal_norm = np.zeros(
#     (len(hrf_peak_locations),
#      n_runs,
#      n_noises * len(noise_levels)))
# new_test_signal_norm = np.zeros(
#     (len(hrf_peak_locations),
#      n_runs,
#      n_new_betas))




# for i_sim, simulation_peak in enumerate(hrf_peak_locations):
#     simulation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
#                                            time_length=hrf_length + 1,
#                                            undershoot=16., delay=simulation_peak)
#     xs = np.linspace(0., hrf_length + 1, len(simulation_hrf), endpoint=False)
#     f_sim_hrf = interp1d(xs, simulation_hrf)
#     for held_out_index in range(n_runs):
#         shifted_paradigms =  [paradigm.copy() 
#                               for i, paradigm in enumerate(paradigms)
#                               if i != held_out_index]
#         shifted_frame_times = []
#         offset = 0
#         # shift paradigms to concatenate them
#         for paradigm in shifted_paradigms:
#             paradigm_length = paradigm['onset'].max()
#             paradigm['onset'] += offset
#             shifted_frame_times.append(frame_times_run + offset)
#             offset += paradigm_length + time_offset
#         shifted_frame_times = np.concatenate(shifted_frame_times)

#         train_paradigm = pandas.concat(shifted_paradigms)
#         test_paradigm = paradigms[held_out_index]
#         train_noise = np.concatenate(
#             [noise for i, noise in enumerate(noise_vectors)
#              if i != held_out_index], axis=0)
#         scaled_train_noise = (train_noise[:, np.newaxis] *
#                               noise_levels[np.newaxis, :, np.newaxis]
#                           ).reshape(train_noise.shape[0], -1)
#         #test_noise = noise_vectors[held_out_index]

#         # design matrix dataframes
#         train_design_gen_df = make_design_matrix_hrf(shifted_frame_times,
#                                                      train_paradigm, f_hrf=f_sim_hrf)
#         test_design_gen_df = make_design_matrix_hrf(frame_times_run,
#                                                      test_paradigm, f_hrf=f_sim_hrf)
#         # design matrix without drifts
#         train_design_gen = train_design_gen_df[event_types].values
#         test_design_gen = test_design_gen_df[event_types].values
#         y_train_clean = train_design_gen.dot(beta)
#         y_train_norm = np.linalg.norm(y_train_clean) ** 2
#         y_train_noisy = y_train_clean[:, np.newaxis] + np.linalg.norm(y_train_clean) * scaled_train_noise
#         y_train_noisy_norm = np.linalg.norm(y_train_noisy, axis=0) ** 2
#         train_signal_norm[i_sim, held_out_index, :] = y_train_noisy_norm

#         y_test = test_design_gen.dot(beta)
#         y_test_new = test_design_gen.dot(new_betas)

#         y_test_norm = np.linalg.norm(y_test) ** 2
#         y_test_new_norm = np.linalg.norm(y_test_new, axis=0) ** 2

#         test_signal_norm[i_sim, held_out_index, :] = y_test_norm
#         new_test_signal_norm[i_sim, held_out_index, :] = y_test_new_norm

#         beta_hat_gen = np.linalg.pinv(train_design_gen).dot(y_train_noisy)
#         train_gen_resid = np.linalg.norm(y_train_noisy -
#                                          train_design_gen.dot(beta_hat_gen), axis=0) ** 2
#         train_gen_train_gen[i_sim, held_out_index, :] = train_gen_resid
#         y_pred_gen = test_design_gen.dot(beta_hat_gen)
#         test_gen_resid = np.linalg.norm(y_test[:, np.newaxis] - y_pred_gen) ** 2
#         train_gen_test_gen[i_sim, held_out_index, :] = test_gen_resid
#         for i_est, estimation_peak in enumerate(hrf_peak_locations):
#             #print("Generation peak {} Estimation peak {} Fold {}".format(simulation_peak, estimation_peak, held_out_index))
#             estimation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
#                                                    time_length=hrf_length + 1,
#                                                    undershoot=16, delay=estimation_peak)
#             f_est_hrf = interp1d(xs, estimation_hrf)
#             # design matrix dataframes
#             train_design_est_df = make_design_matrix_hrf(shifted_frame_times,
#                                                          train_paradigm, f_hrf=f_est_hrf)
#             test_design_est_df = make_design_matrix_hrf(frame_times_run,
#                                                          test_paradigm, f_hrf=f_est_hrf)
#             # design matrix without drifts
#             train_design_est = train_design_est_df[event_types].values
#             test_design_est = test_design_est_df[event_types].values

#             beta_hat_est = np.linalg.pinv(train_design_est).dot(y_train_noisy)
#             train_est_resid = np.linalg.norm(y_train_noisy -
#                                              train_design_est.dot(beta_hat_est), axis=0) ** 2
#             train_gen_train_est[i_sim, i_est, held_out_index, :] = train_est_resid
#             y_pred_est = test_design_est.dot(beta_hat_est)

#             test_est_resid = np.linalg.norm(y_test[:, np.newaxis] - y_pred_est, axis=0) ** 2
#             train_gen_test_est[i_sim, i_est, held_out_index, :] = test_est_resid

#             y_test_squashed = test_design_est.dot(np.linalg.pinv(test_design_est).dot(y_test))
#             test_squashed_resid = np.linalg.norm(y_test - y_test_squashed, axis=0) ** 2
#             test_est_test_est[i_sim, i_est, held_out_index, :] = test_squashed_resid

