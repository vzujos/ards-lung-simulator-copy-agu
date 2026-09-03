# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:06:55 2026

@author: angus
"""

import meshio as io
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as sw
from scipy.stats import pearsonr, linregress
import pickle

# %%

bin_and_number = {'NAT':0,
                  'PAT':1,
                  'AT':2,
                  'HIT':3,}

def IsoVolumetricSegmentation(direction, Mvec, xyz, nROI):

	V = np.sum(Mvec)
	dv = V/nROI
	d = np.ravel(xyz*direction)
	sum_mass = np.cumsum(Mvec[np.argsort(d)])
	lim_index = np.array([np.abs(sum_mass-i).argmin() for i in
						  np.linspace(dv, V, nROI)])
	id_roi = np.ones(len(Mvec))*(nROI-1)
	for i, j in enumerate(lim_index):
		if i == 0:
			id_roi[:j] = np.ones(j)*i
		else:
			id_roi[lim_index[i-1]:j] = np.ones(j-lim_index[i-1])*i

	id_roi_output = [None]*len(id_roi)
	for i, j in zip(id_roi, np.argsort(d)):
		id_roi_output[j] = int(i)

	return [sum_mass, np.sort(d)], np.array(id_roi_output)

def pvalue_to_stars(p):
    """
    Translate a p-value into a significance annotation.

    Parameters
    ----------
    p : float
        p-value

    Returns
    -------
    str
        Significance string: "", "(*)", "(**)", or "(***)"
    """
    if p < 0.001:
        return "(***)"
    elif p < 0.01:
        return "(**)"
    elif p < 0.05:
        return "(*)"
    else:
        return ""


def build_sorted_data_per_subject(data, subject, compartments, nrois=10):
    """
    Build a canonical per-subject data dictionary.

    Structure:
    type -> state -> compartment / phi_mean -> np.ndarray (nrois,)
    """

    sorted_data = {
        "image": {},
        "mesh": {}
    }

    # ==========================
    # IMAGE-BASED (CT)
    # ==========================
    image_src = data["regional-image-hist"][subject]

    for state in ["Exp", "Insp"]:
        sorted_data["image"][state] = {}

        # categorical fractions
        for comp in compartments:
            vals = np.asarray(image_src[state][comp], dtype=float)
            sorted_data["image"][state][comp] = vals[:nrois]

        # continuous aeration (phi_mean)
        Tvals = np.array(data["regional-image-hist"][subject][state]["phi_mean"],
            dtype=float)
        sorted_data["image"][state]["phi_mean"] = Tvals

    # ==========================
    # MESH-BASED
    # ==========================
    mesh_src = data["regional-mesh-hist"][subject]

    for state in ["Exp", "Insp-Simulation", "Insp-Experiment"]:
        sorted_data["mesh"][state] = {}

        # categorical fractions
        for comp in compartments:
            bin_id = bin_and_number[comp]
            sorted_data["mesh"][state][comp] = np.array(
                [mesh_src[state][roi][bin_id] for roi in range(nrois)],
                dtype=float
            )

        # continuous porosity
        sorted_data["mesh"][state]["phi_mean"] = np.array(
            [mesh_src[state][roi]["phi_mean"] for roi in range(nrois)],
            dtype=float
        )
        
    # ==========================
    # MESH-BASED (but handled separately)
    # ==========================
    mesh_src = data["regional-mesh-fields"][subject]

    for state in ["Simulation", "Experiment"]:
        sorted_data["mesh"][state] = {}

        # continuous porosity
        sorted_data["mesh"][state]["j"] = np.array(
            [mesh_src[state][roi]["j"] for roi in range(nrois)],
            dtype=float
        )
        sorted_data["mesh"][state]["dgf"] = np.array(
            [mesh_src[state][roi]["dgf"] for roi in range(nrois)],
            dtype=float
        )

    return sorted_data

# %% Target 

subject = 3
mesh_type = 'medium-fine'

mroot = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/'%(subject, mesh_type)

sim_mesh = io.read(mroot+"sim_anim_ready.vtu")
exp_mesh = io.read(mroot+"reg_anim_ready.vtu")


directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}

direction = 'VD'
nrois = 10
M = sim_mesh.point_data['Mass']


_, ids = IsoVolumetricSegmentation(directions[direction], M, sim_mesh.points, nrois)


# %%

# Analyze sim mesh

xdata = sim_mesh.point_data['End-Inspiratory Porosity']
ydata = sim_mesh.point_data['End-Expiratory Porosity']
plt.scatter(xdata, ydata, 
            marker='x', alpha=0.15)

# %% 

new_ei_porosity = sim_mesh.point_data['End-Inspiratory Porosity']*sim_mesh.point_data['Jacobian Forward']

xdata = sim_mesh.point_data['End-Inspiratory Porosity'] 
ydata = new_ei_porosity

fig,ax = plt.subplots(figsize=(4,4),dpi=200)
ax.scatter(xdata, ydata, marker='x', alpha=0.25)
ax.plot(np.linspace(0,1,3),np.linspace(0,1,3),ls='--',color='k')
ax.set_xlabel("Spatial porosity (-)")
ax.set_ylabel("Material porosity (-)")

# Material porosity is consistently higher thant the spatial porosity. We should keep
# the spatial (Eulerian) porosity (for inspiration state) as otherwise, we would have
# issues in the delta porosity.

# %% Recompute the 'End-Inspiratory Porosity' and compare with stored value

Jf = sim_mesh.point_data['Jacobian Forward']
xdata = (Jf-1 +  sim_mesh.point_data['End-Expiratory Porosity'])/Jf
ydata = sim_mesh.point_data['End-Inspiratory Porosity'] 

fig,ax = plt.subplots(figsize=(4,4),dpi=200)
ax.scatter(xdata, ydata, marker='x', alpha=0.25)
ax.plot(np.linspace(0,1,3),np.linspace(0,1,3),ls='--',color='k')
ax.set_xlabel("Spatial porosity [Recomputed] (-)")
ax.set_ylabel("Spatial porosity [Stored] (-)")

# %% Relationship between forward jacobian and an alternative way to delta porosity

Jf = sim_mesh.point_data['Jacobian Forward']
xdata = (Jf-1 +  sim_mesh.point_data['End-Expiratory Porosity']) - sim_mesh.point_data['End-Expiratory Porosity']
ydata = Jf

fig,ax = plt.subplots(figsize=(4,4),dpi=200)
ax.scatter(xdata, ydata, marker='x', alpha=0.25)
ax.set_xlabel("Delta porosity (-)")
ax.set_ylabel("Jacobian Forward (-)")

# %% 'Traditional' delta porosity against jacobian forward 

ydata = sim_mesh.point_data['Delta Porosity']
xdata = sim_mesh.point_data['Jacobian Forward']

xdat = []; ydat = [];

# Determine roiwise-values for xdata and ydata
for roi in range(nrois):
    mask = ids==roi
    roimass = M[mask]
    xst = sw.DescrStatsW(xdata[mask],weights=roimass)
    yst = sw.DescrStatsW(ydata[mask],weights=roimass)
    xdat += [xst.mean]; ydat += [yst.mean]

xdat = np.array(xdat); ydat = np.array(ydat)

# Determine correlation coefficient
r, p = pearsonr(xdat, ydat)

# Determine a linear regression
slope, intercept, r_val, p_val, stderr = linregress(xdat, ydat)

# Generate a plot
fig, axes = plt.subplots(ncols=2,figsize=(6,4),dpi=200)

ax = axes[1]

# Scatter plot
ax.scatter(xdat,ydat)
x0,x1 = ax.get_xlim(); dx=x1-x0
y0,y1 = ax.get_ylim(); dy=y1-y0

# Place labels
ax.set_xlabel("Jacobian Forward (-)")
ax.set_ylabel("Delta Porosity (-)")

# If the linear regression has a SS p-value, plot the line
if p_val<0.05:
    xfit = np.linspace(xdat.min(), xdat.max(), 100)
    yfit = slope * xfit + intercept
    ax.plot(xfit, yfit, color='r', alpha=0.50, ls='--')

# Pearson correlation coefficient
ax.text(x0+0.05*dx, y1-0.05*dy, "r=%.2f"%r+pvalue_to_stars(p),ha='left')


ax = axes[0]

ydata = sim_mesh.point_data['End-Inspiratory Porosity']
xdata = sim_mesh.point_data['End-Expiratory Porosity']

xdat = []; ydat = [];

# Determine roiwise-values for xdata and ydata
for roi in range(nrois):
    mask = ids==roi
    roimass = M[mask]
    xst = sw.DescrStatsW(xdata[mask],weights=roimass)
    yst = sw.DescrStatsW(ydata[mask],weights=roimass)
    xdat += [xst.mean]; ydat += [yst.mean]

xdat = np.array(xdat); ydat = np.array(ydat)
ax.scatter(xdat,ydat, marker='x')
ax.set_ylabel("End-Inspiratory Porosity")
ax.set_xlabel("End-Expiratory Porosity")
x0,x1 = ax.get_xlim()
ax.set_ylim((0,1.2))
ax.plot(np.linspace(x0,x1,3),np.linspace(x0,x1,3),label='Identity line', color='r',alpha=0.5, ls='--')
ax.axhline(1.0, color="k",ls='--',alpha=0.35,label='Porosity limit')
ax.legend(loc='lower right')
plt.tight_layout()



# %% 'Convenient' delta porosity against jacobian forward 

ydata = sim_mesh.point_data['End-Inspiratory Porosity']*sim_mesh.point_data['Jacobian Forward'] - sim_mesh.point_data['End-Expiratory Porosity']
xdata = sim_mesh.point_data['Jacobian Forward']

xdat = []; ydat = [];

# Determine roiwise-values for xdata and ydata
for roi in range(nrois):
    mask = ids==roi
    roimass = M[mask]
    xst = sw.DescrStatsW(xdata[mask],weights=roimass)
    yst = sw.DescrStatsW(ydata[mask],weights=roimass)
    xdat += [xst.mean]; ydat += [yst.mean]

xdat = np.array(xdat); ydat = np.array(ydat)

# Determine correlation coefficient
r, p = pearsonr(xdat, ydat)

# Determine a linear regression
slope, intercept, r_val, p_val, stderr = linregress(xdat, ydat)

# Generate a plot
fig, axes = plt.subplots(ncols=2, figsize=(6,4),dpi=200)

ax = axes[1]
# Scatter plot
ax.scatter(xdat,ydat,marker='x')
x0,x1 = ax.get_xlim(); dx=x1-x0
y0,y1 = ax.get_ylim(); dy=y1-y0

# Place labels
ax.set_xlabel("Jacobian Forward (-)")
ax.set_ylabel("Delta Porosity (-)")

# If the linear regression has a SS p-value, plot the line
if p_val<0.05:
    xfit = np.linspace(xdat.min(), xdat.max(), 100)
    yfit = slope * xfit + intercept
    ax.plot(xfit, yfit, color='r', alpha=0.50, ls='--',label="Identity line")

ax.legend(loc='lower right')
# Pearson correlation coefficient
ax.text(x0+0.05*dx, y1-0.05*dy, "r=%.2f"%r+pvalue_to_stars(p),ha='left')

ax = axes[0]

ydata = sim_mesh.point_data['End-Inspiratory Porosity']*sim_mesh.point_data['Jacobian Forward']
xdata = sim_mesh.point_data['End-Expiratory Porosity']

xdat = []; ydat = [];

# Determine roiwise-values for xdata and ydata
for roi in range(nrois):
    mask = ids==roi
    roimass = M[mask]
    xst = sw.DescrStatsW(xdata[mask],weights=roimass)
    yst = sw.DescrStatsW(ydata[mask],weights=roimass)
    xdat += [xst.mean]; ydat += [yst.mean]

xdat = np.array(xdat); ydat = np.array(ydat)
ax.scatter(xdat,ydat, marker='x')
ax.set_ylabel("End-Inspiratory Porosity")
ax.set_xlabel("End-Expiratory Porosity")
x0,x1 = ax.get_xlim()
ax.set_ylim((0,1.2))
ax.plot(np.linspace(x0,x1,3),np.linspace(x0,x1,3),label='Identity line', color='r',alpha=0.5, ls='--')
ax.axhline(1.0, color="k",ls='--',alpha=0.35,label='Porosity limit')
ax.legend(loc='lower right')
plt.tight_layout()

# %%
with open('./groupwise_intensity_analysis.pkl', 'rb') as f:
    data = pickle.load(f)

sdata = {s:build_sorted_data_per_subject(data=data,
                                         subject=s,
                                         compartments=["NAT","PAT","AT","HIT"],
                                         nrois=10) for s in [subject]}

# %%

ydata = sim_mesh.point_data['Delta Porosity']

ydat = [];

# Determine roiwise-values for xdata and ydata
for roi in range(nrois):
    mask = ids==roi
    roimass = M[mask]
    yst = sw.DescrStatsW(ydata[mask],weights=roimass)
    ydat += [yst.mean]


xdat =  sdata[3]['mesh']['Experiment']['dgf']

plt.scatter(xdat,ydat)
l = np.linspace(0.03,0.16)
plt.plot(l,l, ls='--',alpha=0.3)
plt.xlabel("")

# %% 

ydata = sim_mesh.point_data['Jacobian Forward']

ydat = [];

# Determine roiwise-values for xdata and ydata
for roi in range(nrois):
    mask = ids==roi
    roimass = M[mask]
    yst = sw.DescrStatsW(ydata[mask],weights=roimass)
    ydat += [yst.mean]


xdat =  sdata[3]['mesh']['Experiment']['j']

plt.scatter(xdat,ydat)
#l = np.linspace(0.03,0.16)
#plt.plot(l,l, ls='--',alpha=0.3)
plt.xlabel("")