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

###############################################################################
# Figure 2
###############################################################################

###############################################################################
# Figure 3
###############################################################################

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
fig, axx = plt.subplots(nrows=2, ncols=3, figsize=(15, 7),
                        sharey=True,
                        # sharex=True
                        )

# 1- residnorm / signal norm

# note: the mean(axis=2) is the folds
# mean(axis=-1) for different noise initializations
signal_norm = f['test_norm_squared']

score_test_pred_resid_glm = 1. - test_pred_resid_glm / signal_norm
score_test_pred_resid_gp = 1. - test_pred_resid_gp / signal_norm
score_test_pred_resid_gp_zero = 1. - test_pred_resid_gp_zero / signal_norm

# score_test_proj_resid_glm = 1. - test_proj_resid_glm / signal_norm
# score_test_proj_resid_gp = 1. - test_proj_resid_gp / signal_norm
# score_test_proj_resid_gp_zero = 1. - test_proj_resid_gp_zero / signal_norm

score_pred_glm = score_test_pred_resid_glm.mean(axis=-1).mean(axis=2)
score_pred_gp = score_test_pred_resid_gp.mean(axis=-1).mean(axis=2)
score_pred_gp_zero = score_test_pred_resid_gp_zero.mean(axis=-1).mean(axis=2)

# score_proj_glm = score_test_proj_resid_glm.mean(axis=-1).mean(axis=2)
# score_proj_gp = score_test_proj_resid_gp.mean(axis=-1).mean(axis=2)
# score_proj_gp_zero = score_test_proj_resid_gp_zero.mean(axis=-1).mean(axis=2)

formatter = FuncFormatter(add_s)
for col, j in enumerate(noise_levels_index):
    axx[0, col].plot(score_pred_glm[..., j], 'r', linewidth=2)
    axx[0, col].plot(score_pred_gp[..., j], 'g')
    axx[0, col].plot(score_pred_gp_zero[:, 0, ..., j], 'b')
    axx[0, col].set_title('noise = %s' % j)

    # axx[1, col].plot(score_proj_glm[..., j], 'r')
    # axx[1, col].plot(score_proj_gp[..., j], 'g')
    # axx[1, col].plot(score_proj_gp_zero[:, 0, ..., j], 'b')

    axx[0, col].set_title('noise = %s' % noise_levels[j])
    axx[1, col].set_ylim([0, 1.])
    # axx[0, col].set_ylim([1e-6, 1e-3])
    # plt.setp(axx[1, col], xticks=range(hrf_peak_location.shape[0]),
    #          xticklabels=hrf_peak_location)
    # axx[1, col].xaxis.set_major_formatter(formatter)
    # axx[1, col].set_xlabel('HRF peak location \n for generation')
    # axx[0, col].set_yscale('log')

axx[0, 0].set_ylabel('Residual error')
axx[1, 0].set_ylabel('Residual error')

l_glm = Rectangle((0, 0), 1, 1, fc='r', alpha=0.8, edgecolor='.8')
l_gp = Rectangle((0, 0), 1, 1, fc='g', alpha=0.8, edgecolor='.8')
l_gp_zero = Rectangle((0, 0), 1, 1, fc='b', alpha=0.8, edgecolor='.8')

lgd = plt.legend([l_glm, l_gp, l_gp_zero], ['GLM', 'GP', 'zero-mean GP'],
                 ncol=3, loc=(-2.5, -1))


fig.savefig('images/super_duper_diagram.pdf', bbox_inches='tight')
fig.savefig('images/super_duper_diagram.png', bbox_inches='tight')
fig.savefig('images/super_duper_diagram.svg', bbox_inches='tight')
