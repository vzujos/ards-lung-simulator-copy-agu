import matplotlib.pyplot as plt
import numpy as np

calibrated_parameters = {"c_tissue":[1.8377,2.3214,1.6903,2.0923,2.0082],
                         "k_cw":[0.0477,0.0508,0.0398,0.0023,0.0860,],
                         "k_d":[0.0070,0.0361,0.0003,0.0058,0.0669,],
                         "alpha":[1.0462,1.0325,1.0458, 1.0358,1.0376],
                         "gamma_insp":[0.2996,0.1506,0.3900,0.2944,0.2958],
                         "gamma_exp":[0.3609,0.2853,0.7320,0.4476,0.3099]}

# Make axes[2] and axes[3] wider
fig, axes = plt.subplots(
    ncols=4,
    figsize=(8,6),
    gridspec_kw={'width_ratios':[1,1,2.0,2.0]},
    constrained_layout=True,
    dpi=300
)


use_boxplots = True
use_scatters = False


bar_alpha = 0.8
bar_color = 'tab:blue'
lw = 2.0

textsize=15
xs = np.zeros(5)
color = "tab:blue"

if use_boxplots:
    boxes = []
    boxes += [axes[0].boxplot(calibrated_parameters["c_tissue"],positions=[0],vert=True,showfliers=False, patch_artist=True)]
if use_scatters:
    axes[0].scatter(xs,calibrated_parameters["c_tissue"],color=color)
axes[0].set_xticks([0])
axes[0].set_xticklabels(["$C_{tissue}$"],size=textsize)
axes[0].set_ylabel("Tissue stiffness (kPa)",size=textsize)
axes[0].set_ylim((0.0,2.5))
yticks = [0.0, 0.5,1.0,1.5,2.0,2.5]
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(["%.1f"%ylab for ylab in yticks],size=textsize-2)

if use_boxplots:
    boxes += [axes[1].boxplot(calibrated_parameters["alpha"],positions=[0],vert=True,showfliers=False, patch_artist=True)]
if use_scatters:
    axes[1].scatter(xs,calibrated_parameters["alpha"],color=color)
axes[1].set_xticks([0])
axes[1].set_xticklabels(["$\\alpha$"],size=textsize)
y0,y1 = axes[1].get_ylim()
axes[1].set_ylim((1.0,y1))
axes[1].set_ylabel("Pre-strain factor (-)",size=textsize)
yticks = [1.0, 1.02, 1.04,1.06]
axes[1].set_yticks(yticks)
axes[1].set_yticklabels(["%.2f"%ylab for ylab in yticks],size=textsize-2)


if use_boxplots:
    boxes += [axes[2].boxplot(calibrated_parameters["k_cw"],positions=[0],vert=True,showfliers=False, patch_artist=True)]
    boxes += [axes[2].boxplot(calibrated_parameters["k_d"],positions=[1],vert=True,showfliers=False, patch_artist=True)]
if use_scatters:
    axes[2].scatter(xs,calibrated_parameters["k_cw"],color=color)
    axes[2].scatter(xs+1,calibrated_parameters["k_d"],color=color)
    
axes[2].set_xticks([0,1])
axes[2].set_xticklabels(["$K_{cw}$","$K_{d}$"],size=textsize)
axes[2].set_xlim((-0.5,1.5))
axes[2].set_ylabel("Stiffness (kPa/mm)",size=textsize)
axes[2].set_ylim((0.0,0.10))
yticks = [0.0, 0.02,0.04,0.06,0.08,0.10]
axes[2].set_yticks(yticks)
axes[2].set_yticklabels(["%.2f"%ylab for ylab in yticks],size=textsize-2)


if use_boxplots:
    boxes += [axes[3].boxplot(calibrated_parameters["gamma_insp"],positions=[0],vert=True,showfliers=False, patch_artist=True)]
    boxes += [axes[3].boxplot(calibrated_parameters["gamma_exp"],positions=[1],vert=True,showfliers=False, patch_artist=True)]
if use_scatters:
    axes[3].scatter(xs,calibrated_parameters["gamma_insp"],color=color)
    axes[3].scatter(xs+1,calibrated_parameters["gamma_exp"],color=color)

axes[3].set_xticks([0,1])
axes[3].set_xticklabels(["$\gamma_{insp}$","$\gamma_{exp}$"],size=textsize)
axes[3].set_xlim((-0.5,1.5))
axes[3].set_ylabel("Gamma factor (-)",size=textsize)
axes[3].set_ylim((0.0,0.80))
yticks=[0.0, 0.20,0.40,0.60, 0.80]
axes[3].set_yticks(yticks)
axes[3].set_yticklabels(["%.1f"%ylab for ylab in yticks],size=textsize-2)

if use_boxplots:
    for box in boxes:
        box['medians'][0].set_color('k')
        box['medians'][0].set_alpha(1.0)
        box['medians'][0].set_linewidth(lw+0.5)
        box['boxes'][0].set_alpha(bar_alpha)
        box['boxes'][0].set_color(bar_color)
        box['whiskers'][0].set_linewidth(lw)
        box['whiskers'][1].set_linewidth(lw)
        box['caps'][0].set_linewidth(lw)
        box['caps'][1].set_linewidth(lw)




for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
