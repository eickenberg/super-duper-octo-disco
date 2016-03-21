import numpy as np
import pandas

from scipy.interpolation import interp1d

from data_generator import  generate_spikes_time_series as generate_experiment
from data_generator import make_design_matrix_hrf
from nistats.hemodynamic_models import _gamma_difference_hrf

n_events=200
n_blank_events=50
event_spacing=6
t_r=2
hrf_length=32.
event_types=['ev1', 'ev2']
jitter_min=-1
jitter_max=1
time_offset = 20
modulation=None


n_runs = 5
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
noise_vectors = [rng.randn(len(paradigm)) for paradigm in paradigms]
noise_vectors = [noise_vector / np.linalg.norm(noise_vector)
                 for noise_vector in noise_vectors]


for simulation_peak in hrf_peak_locations:
    for estimation_peak in hrf_peak_locations:

        estimation_hrf = _gamma_difference_hrf(tr=1., oversampling=20,
                                               time_length=hrf_length,
                                               undershoot=16, delay=estimation_peak)
        simulation_hrf = _gamme_difference_hrf(tr=1., oversampling=20,
                                               time_length=hrf_length,
                                               undershoot=16., delay=simulation_peak)
        xs = np.linspace(0. hrf_length, len(estimation_hrf), endpoint=False)
        f_est_hrf = interp1d(xs, estimation_hrf)
        f_sim_hrf = interp1d(xs, simulation_hrf)

        for held_out_index in range(n_runs):
            train_paradigm = pandas.concat(
                [paradigm for i, paradigm in enumerate(paradigms)
                 if i != held_out_index])
            train_noise = np.concatenate(
                [noise for i, noise in enumerate(noise_vectors)
                 if i != held_out_index])

            train_design = make_design_matrix_hrf(train_paradigm, )
