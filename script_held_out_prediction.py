import numpy as np
import pandas

from scipy.interpolate import interp1d

from data_generator import  generate_spikes_time_series as generate_experiment
from data_generator import make_design_matrix_hrf
from nistats.hemodynamic_models import _gamma_difference_hrf

n_events=100
n_blank_events=25
event_spacing=6
t_r=2
hrf_length=32.
event_types=['ev1', 'ev2']
jitter_min=-1
jitter_max=1
time_offset = 20
modulation=None


n_runs = 3
hrf_peak_locations = np.arange(3, 10)


paradigms, _, _, measurement_times = list(zip(*[
    generate_experiment(n_events=n_events,
                        n_blank_events=n_blank_events,
                        event_spacing=event_spacing,
                        t_r=t_r, hrf_length=hrf_length,
                        event_types=event_types,
                        jitter_min=jitter_min,
                        jitter_max=jitter_max,
                        time_offset=time_offset,
                        modulation=modulation,
                        seed=seed) for seed in np.arange(n_runs)]))


rng = np.random.RandomState(42)

noise_levels = np.array([1., .1, 2., .01, 10.])

beta = rng.randn(len(event_types))

n_new_betas = 4

new_betas = rng.randn(len(event_types), n_new_betas) # these are to see how well the hrf spans the space
frame_times_run = np.arange(0, paradigms[0]['onset'].max() + time_offset, t_r)

n_noises = 4
noise_vectors = [rng.randn(len(frame_times), n_noises) for _ in range(n_runs)]
noise_vectors = [noise_vector / np.linalg.norm(noise_vector, axis=0)
                 for noise_vector in noise_vectors]



for simulation_peak in hrf_peak_locations:
    simulation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
                                           time_length=hrf_length,
                                           undershoot=16., delay=simulation_peak)
    xs = np.linspace(0., hrf_length, len(estimation_hrf), endpoint=False)
    f_sim_hrf = interp1d(xs, simulation_hrf)
    for held_out_index in range(n_runs):
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
            [noise for i, noise in enumerate(noise_vectors)
             if i != held_out_index], axis=0)
        scaled_train_noise = (train_noise[:, np.newaxis] *
                              noise_levels[np.newaxis, :, np.newaxis]
                          ).reshape(train_noise.shape[0], -1)
        #test_noise = noise_vectors[held_out_index]

        # design matrix dataframes
        train_design_gen_df = make_design_matrix_hrf(shifted_frame_times,
                                                     train_paradigm, f_hrf=f_sim_hrf)
        # design matrix without drifts
        train_design_gen = train_design_gen_df[event_types].values
        y_train_clean = train_design_gen.dot(beta)
        y_train_norm = np.linalg.norm(y_train_clean) ** 2
        y_train_noisy = y_train_clean + np.linalg.norm(y_train_clean) * scaled_train_noise
        y_train_noisy_norm = np.linalg.norm(y_train_noisy, axis=0) ** 2

        y_test = test_design_gen.dot(beta)
        y_test_new = test_design_gen.dot(beta_new)

        y_test_norm = np.linalg.norm(y_test) ** 2
        y_test_new_norm = np.linalg.norm(y_test_new, axis=0) ** 2

        beta_hat_gen = np.linalg.pinv(train_design_gen).dot(y_train_noisy)
        train_gen_resid = np.linalg.norm(y_train_noisy -
                                         train_design_gen.dot(beta_hat_gen), axis=0) ** 2
        y_pred_gen = test_design_gen.dot(beta_hat_gen)
        test_gen_resid = np.linalg.norm(y_test - y_pred_gen) ** 2
        for estimation_peak in hrf_peak_locations:

            estimation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
                                                   time_length=hrf_length,
                                                   undershoot=16, delay=estimation_peak)
            f_est_hrf = interp1d(xs, estimation_hrf)
            # design matrix dataframes
            train_design_est_df = make_design_matrix_hrf(shifted_frame_times,
                                                         train_paradigm, f_hrf=f_est_hrf)
            # design matrix without drifts
            train_design_est = train_design_est_df[event_types].values

            beta_hat_est = np.linalg.pinv(train_design_est).dot(y_train_noisy)
            train_est_resid = np.linalg.norm(y_train_noisy -
                                             train_design_est.dot(beta_hat_est), axis=0) ** 2
            y_pred_est = test_design_est.dot(beta_hat_est)

            test_est_resid = np.linalg.norm(y_test - y_pred_test) ** 2

            y_test_squashed = test_design_est.dot(np.linalg.pinv(test_design_est).dot(y_test))
            
