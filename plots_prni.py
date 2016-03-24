import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

sns.set_style('whitegrid',)
# basic settings
rc('axes', labelsize=32)
rc('xtick', labelsize=32)
rc('ytick', labelsize=32)
rc('legend', fontsize=32)
rc('lines', linewidth=3)
rc('axes', titlesize=32)
rc('text', usetex=False)
rc('font', family='sans-serif')
rc('mathtext', default='regular')

plt.close('all')

def add_s(x, pos):
    return '%is' % x

###############################################################################
# Figure 1
###############################################################################

folder = 'results_figure1'
range_peak = np.array([3, 8])
for sigma_noise in np.array([0.01, 2.]):
    plt.figure(figsize=(8, 4))
    i = 0
    for hrf_peak in range_peak:
        hx, hy, hrf_var, _, _ = np.load(op.join(folder, 'hrf_sigmanoise%f_hrfpeak%d.npz' % (sigma_noise, hrf_peak)))
        plt.subplot(1, 2, i + 1)
        i += 1
        plt.tight_layout()
        if np.abs(hy.max())>np.abs(hy.min()):
            nm = hy.max()
        else:
            nm = hy.min()
        plt.fill_between(hx, (hy - 1.96 * np.sqrt(hrf_var))/nm,
                         (hy + 1.96 * np.sqrt(hrf_var))/nm, alpha=0.1)
        plt.plot(hx, hy/nm, 'b', label='estimated HRF')
        plt.plot(x_0, hrf_sim/hrf_sim.max(), 'r--', label='simulated HRF')
        plt.plot(x_0, hrf_0/hrf_0.max(), 'k-', label='GP mean')
        plt.xlabel('time')
        plt.axis('tight')

    # Save one image per noise level, with different HRFs
    fig_folder = 'images'
    if not op.exists(fig_folder): os.makedirs(fig_folder)
    fig_name = op.join(fig_folder, \
        'results_GP_simulation_diff_hrf_peak_sigma' + str(sigma_noise) + '_gamma' + str(gamma))
    plt.tight_layout()
    plt.savefig(fig_name + '.pdf', format='pdf')
    plt.show()

###############################################################################
# Figure 2
###############################################################################

###############################################################################
# Figure 3
###############################################################################

#A
def k(xs, ys, gamma=1.):
    xs, ys = map(np.atleast_1d, (xs, ys))
    diffs_squared = (xs.reshape(-1, 1) - ys.reshape(-1)) ** 2
    return np.exp(-diffs_squared / gamma)

sim1 = k(xs, xs, gamma=.1)
sim2 = k(xs, xs, gamma=1.)
sim3 = k(xs, xs, gamma=10.)
plt.subplot(1, 3, 1)
plt.imshow(sim1)
plt.title('$\gamma$=.1')
plt.axis('off')
plt.hot()
plt.subplot(1, 3, 2)
plt.imshow(sim2)
plt.title('$\gamma$=1.')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(sim3)
plt.title('$\gamma$=10.')
plt.axis('off')

#B

plt.figure(figsize=(12, 4))
i = 0

for gamma in np.array([0.1, 1., 10.]):

    hx, hy, hrf_var, _, _ = np.load(op.join(folder, 'hrf_sigmanoise%f_gamma%f.npz' % (sigma_noise, gamma)))
    plt.subplot(1, 3, i + 1)
    i += 1
    plt.tight_layout()
    if np.abs(hy.max())>np.abs(hy.min()):
        nm = hy.max()
    else:
        nm = hy.min()
    plt.fill_between(hx, (hy - 1.96 * np.sqrt(hrf_var))/nm,
                     (hy + 1.96 * np.sqrt(hrf_var))/nm, alpha=0.1)
    plt.plot(hx, hy/nm, 'b', label='estimated HRF')
    plt.plot(x_0, hrf_sim/hrf_sim.max(), 'r--', label='simulated HRF')
    plt.xlabel('time (sec.)')
    plt.axis('tight')

# Save one image per noise level, with different HRFs
fig_folder = 'images'
if not op.exists(fig_folder): os.makedirs(fig_folder)
fig_name = op.join(fig_folder, \
    'resultsG_GP_simulation_diff_hrf_peak_sigma' + str(sigma_noise) + '_gamma' + str(gamma))
plt.tight_layout()
plt.savefig(fig_name + '.pdf', format='pdf')
plt.show()


###############################################################################
# Figure 4
###############################################################################
# loading the data
f = np.load('diagrams_synthetic/data.npz')

noise_levels = f['noise_levels']
hrf_peak_location = f['hrf_peak_locations']

train_pred_resid_glm = f['train_prediction_residuals_glm']
train_pred_resid_gp = f['train_prediction_residuals_gp']
train_pred_resid_gp_zero = f['train_prediction_residuals_gp_zero']

test_pred_resid_glm = f['test_prediction_residuals_glm']
test_pred_resid_gp = f['test_prediction_residuals_gp']
test_pred_resid_gp_zero = f['test_prediction_residuals_gp_zero']

test_proj_resid_glm = f['test_projection_residuals_glm']
test_proj_resid_gp = f['test_projection_residuals_gp']
test_proj_resid_gp_zero = f['test_projection_residuals_gp_zero']

noise_levels_index = [1, 2, 3]
fig, axx = plt.subplots(nrows=2, ncols=3, figsize=(15, 7), sharey=True,
                        sharex=True)

formatter = FuncFormatter(add_s)
for col, j in enumerate(noise_levels_index):
    axx[0, col].plot(test_pred_resid_glm.mean(axis=-1).mean(axis=2)[..., j], 'r')
    axx[0, col].plot(test_pred_resid_gp.mean(axis=-1).mean(axis=2)[..., j], 'g')
    axx[0, col].plot(test_pred_resid_gp_zero.mean(axis=-1).mean(axis=2)[:, 0, ..., j], 'b')
    axx[0, col].set_title('noise = %s' % j)

    axx[1, col].plot(test_proj_resid_glm.mean(axis=-1).mean(axis=2)[..., j], 'r')
    axx[1, col].plot(test_proj_resid_gp.mean(axis=-1).mean(axis=2)[..., j], 'g')
    axx[1, col].plot(test_proj_resid_gp_zero.mean(axis=-1).mean(axis=2)[:, 0, ..., j], 'b')

    axx[0, col].set_title('noise = %s' % j)
    axx[1, col].set_ylim([0, 0.002])
    axx[0, col].set_ylim([0, 0.002])
    plt.setp(axx[1, col], xticks=range(hrf_peak_location.shape[0]),
             xticklabels=hrf_peak_location)
    axx[1, col].xaxis.set_major_formatter(formatter)

axx[0, 0].set_ylabel('Residual error')
axx[1, 0].set_ylabel('Residual error')

l_glm = Rectangle((0, 0), 1, 1, fc='r', alpha=0.8, edgecolor='.8')
l_gp = Rectangle((0, 0), 1, 1, fc='g', alpha=0.8, edgecolor='.8')
l_gp_zero = Rectangle((0, 0), 1, 1, fc='b', alpha=0.8, edgecolor='.8')

lgd = plt.legend([l_glm, l_gp, l_gp_zero], ['GLM', 'GP', 'zero-mean GP'],
                 ncol=3, loc=(-2., -1))

fig.savefig('super_duper_diagram.pdf', bbox_inches='tight')
