import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

f = np.load('data.npz')

noise_levels = f['noise_levels']
hrf_peak_locations = f['hrf_peak_locations']

for i in range(len(noise_levels)):
    plt.figure(figsize=(4,4))
    plt.plot(f['test_prediction_residuals_glm'].mean(-1).mean(2)[..., i], 'r')
    plt.plot(f['test_prediction_residuals_gp'].mean(-1).mean(2)[..., i], 'g')
    plt.plot(f['test_prediction_residuals_gp_zero'].mean(-1).mean(2)[:, 0, ..., i], 'b', lw=2)
    plt.axis([0, 5, -.0001, max(.002, f['test_prediction_residuals_gp'].mean(-1).mean(2)[..., i].max())])
    plt.xticks(range(len(hrf_peak_locations)), hrf_peak_locations)
    plt.xlabel('HRF peak location\nfor generation')
    plt.ylabel('Residual error')
    plt.title('noise {:1.2f}'.format(noise_levels[i]))
    file_base = './resid_test_prediction_noise_{:1.2f}'.format(noise_levels[i])
    plt.savefig(file_base + ".png")
    plt.savefig(file_base + ".pdf")
    plt.savefig(file_base + ".svg")


for i in range(len(noise_levels)):
    plt.figure(figsize=(4,4))
    plt.plot(f['test_projection_residuals_gp'].mean(-1).mean(2)[..., i], 'r')
    plt.plot(f['test_projection_residuals_gp'].mean(-1).mean(2)[..., i], 'g')
    plt.plot(f['test_projection_residuals_gp_zero'].mean(-1).mean(2)[:, 0, ..., i], 'b', lw=2)
    plt.axis([0, 5, -.0001, max(.002, f['test_projection_residuals_gp'].mean(-1).mean(2)[..., i].max())])
    plt.xticks(range(len(hrf_peak_locations)), hrf_peak_locations)
    plt.xlabel('True HRF peak location')
    plt.ylabel('Residual error')
    plt.title('noise {:1.2f}'.format(noise_levels[i]))
    file_base = './resid_test_projection_noise_{:1.2f}'.format(noise_levels[i])
    plt.savefig(file_base + ".png")
    plt.savefig(file_base + ".pdf")
    plt.savefig(file_base + ".svg")





for i in range(len(noise_levels)):
    plt.figure(figsize=(4,4))
    plt.plot(f['train_prediction_residuals_glm'].mean(-1).mean(2)[..., i], 'r')
    plt.plot(f['train_prediction_residuals_gp'].mean(-1).mean(2)[..., i], 'g')
    plt.plot(f['train_prediction_residuals_gp_zero'].mean(-1).mean(2)[:, 0, ..., i], 'b', lw=2)
    plt.axis([0, 5, -.0001, max(.002, f['train_prediction_residuals_gp'].mean(-1).mean(2)[..., i].max())])
    plt.xticks(range(len(hrf_peak_locations)), hrf_peak_locations)
    plt.xlabel('HRF peak location\nfor generation')
    plt.ylabel('Residual error')
    plt.title('noise {:1.2f}'.format(noise_levels[i]))
    file_base = './resid_train_prediction_noise_{:1.2f}'.format(noise_levels[i])
    plt.savefig(file_base + ".png")
    plt.savefig(file_base + ".pdf")
    plt.savefig(file_base + ".svg")
