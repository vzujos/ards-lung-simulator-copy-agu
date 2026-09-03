# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:16:34 2025

@author: angus
"""

import os
import meshio as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
# Declare paths
root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/%s/"
case_a = "Calibrated-PIG5-ARDSnet-7"
case_b = "Calibrated-PIG5-ARDSnet-1"
tgt = "/post/full_0.750000000000.vtu"

# Load meshes
mesh_a = io.read(root%case_a+tgt)
mesh_b = io.read(root%case_b+tgt)

# Path to a reference mesh where to extract initial porosity information
ref_geometry = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH-8/FEniCS-default/Porosity_Visualization.vtu"

# Path to the registration results
reg_geometry = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH-8/reg_anim_ready.vtu"


# Out temp
temp = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/temp.vtu"

# %%

# Extract data
u_a = mesh_a.point_data['u']
u_b = mesh_b.point_data['u']

phi_a = mesh_a.point_data['Eulerian Porosity']
phi_b = mesh_b.point_data['Eulerian Porosity']

out_mesh = io.Mesh(points=mesh_a.points,cells=mesh_a.cells_dict,
                   point_data={"u_a":u_a,"u_b":u_b,
                               "phi_a":phi_a,
                               "phi_b":phi_b,
                               "diff_u":u_b-u_a,
                               "diff_phi":phi_b-phi_a})

out_mesh.write(temp)

# %%

ref_geom = io.read(ref_geometry)
ee_porosity = ref_geom.point_data['Porosity_EE']

# %%
reg_geom = io.read(reg_geometry)
ei_porosity = reg_geom.point_data['new_ei_porosity']

# %% Silly plot to see some associations

nat_mask = ee_porosity < 0.10
pat_mask = np.logical_and(ee_porosity>0.10, ee_porosity<0.5)
alt_mask = ee_porosity > 0.5

diff_phi = phi_b-phi_a
diff_u = np.linalg.norm(u_b-u_a,axis=1)

fig, axes = plt.subplots(nrows=3,figsize=(6,12),dpi=150)
alpha=0.15
ax=axes[0]
ax.scatter(ee_porosity[nat_mask],diff_phi[nat_mask], alpha=alpha,color="r")
ax.scatter(ee_porosity[pat_mask],diff_phi[pat_mask], alpha=alpha, color="g")
ax.scatter(ee_porosity[alt_mask],diff_phi[alt_mask], alpha=alpha, color="b")

ax.set_ylabel("Porosity difference (-)")
ax.set_xticks([])
ax.set_xticklabels([])

ax=axes[1]
ax.scatter(ee_porosity[nat_mask],diff_u[nat_mask], alpha=alpha,color="r")
ax.scatter(ee_porosity[pat_mask],diff_phi[pat_mask],alpha=alpha, color="g")
ax.scatter(ee_porosity[alt_mask],diff_phi[alt_mask],alpha=alpha, color="b")

ax.set_ylabel("Displacement difference (-)")
ax.set_xlabel("End-expiratory porosity (-)")
ax.set_xticks(np.linspace(0,1,6))
ax.set_xticklabels(["%.1f"%l for l in np.linspace(0,1,6)])

ax=axes[2]
ax.scatter(diff_phi[nat_mask],diff_u[nat_mask], alpha=alpha,color="r")
ax.scatter(diff_phi[pat_mask],diff_u[pat_mask],alpha=alpha, color="g")
ax.scatter(diff_phi[alt_mask],diff_u[alt_mask],alpha=alpha, color="b")

ax.set_ylabel("Displacement difference (-)")
ax.set_xlabel("Porosity difference (-)")
#ax.set_xticks(np.linspace(0,1,6))
#ax.set_xticklabels(["%.1f"%l for l in np.linspace(0,1,6)])
plt.tight_layout()

# %%
fig, axes = plt.subplots(ncols=2,dpi=250, figsize=(8,4))
alpha=0.10
s = 12

handles, labels = plt.gca().get_legend_handles_labels()

nat_lab = Line2D([0], [0], label='NAT', marker='o', markersize=s, 
         markeredgecolor='r', markerfacecolor='r', linestyle='',alpha=alpha)

pat_lab = Line2D([0], [0], label='PAT', marker='o', markersize=s, 
         markeredgecolor='g', markerfacecolor='g', linestyle='',alpha=alpha)

alt_lab = Line2D([0], [0], label='AT & HIT', marker='o', markersize=s, 
         markeredgecolor='b', markerfacecolor='b', linestyle='',alpha=alpha)

line = Line2D([0], [0], label='Identity line', color='k', ls="--")

handles.extend([nat_lab, pat_lab, alt_lab, line])

ax = axes[0]
ax.scatter(ee_porosity[nat_mask],phi_b[nat_mask],s=s, alpha=alpha,color="r")
ax.scatter(ee_porosity[pat_mask],phi_b[pat_mask],s=s,alpha=alpha, color="g")
ax.scatter(ee_porosity[alt_mask],phi_b[alt_mask],s=s,alpha=alpha, color="b")

ax.plot(np.linspace(0,1,5),np.linspace(0,1,5),color="k",ls="--",alpha=1.0)
ax.legend(handles=handles)
ax.set_title("Simulated results")

xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.fill_between(xlim,0,0.1,alpha=0.1,color='k')
ax.fill_between((0.0,0.1),ylim[0],ylim[1],alpha=0.1,color='k')

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_ylabel("End-inspiratory porosity (-)")
ax.set_xlabel("End-expiratory porosity (-)")

ax = axes[1]
ax.scatter(ee_porosity[nat_mask],ei_porosity[nat_mask],s=s, alpha=alpha,color="r")
ax.scatter(ee_porosity[pat_mask],ei_porosity[pat_mask],s=s,alpha=alpha, color="g")
ax.scatter(ee_porosity[alt_mask],ei_porosity[alt_mask],s=s,alpha=alpha, color="b")

ax.plot(np.linspace(0,1,5),np.linspace(0,1,5),color="k",ls="--",alpha=1.0)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.fill_between(xlim,0,0.1,alpha=0.1,color='k')
ax.fill_between((0.0,0.1),ylim[0],ylim[1],alpha=0.1,color='k')
ax.set_title("Registration-derived results")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_ylabel("End-inspiratory porosity (-)")
ax.set_xlabel("End-expiratory porosity (-)")
plt.tight_layout()

# %%

fig, ax = plt.subplots(ncols=1,dpi=500, figsize=(4,4))
alpha=0.10
s = 12

handles, labels = plt.gca().get_legend_handles_labels()

nat_lab = Line2D([0], [0], label='(EE) NAT', marker='o', markersize=s, 
         markeredgecolor='r', markerfacecolor='r', linestyle='',alpha=alpha)

pat_lab = Line2D([0], [0], label='(EE) PAT', marker='o', markersize=s, 
         markeredgecolor='g', markerfacecolor='g', linestyle='',alpha=alpha)

alt_lab = Line2D([0], [0], label='(EE) AT & HIT', marker='o', markersize=s, 
         markeredgecolor='b', markerfacecolor='b', linestyle='',alpha=alpha)

line = Line2D([0], [0], label='Identity line', color='k', ls="--")

handles.extend([nat_lab, pat_lab, alt_lab, line])

ax.scatter(phi_b[nat_mask],ei_porosity[nat_mask],s=s, alpha=alpha,color="r")
ax.scatter(phi_b[pat_mask],ei_porosity[pat_mask],s=s,alpha=alpha, color="g")
ax.scatter(phi_b[alt_mask],ei_porosity[alt_mask],s=s,alpha=alpha, color="b")

ax.plot(np.linspace(0,1,5),np.linspace(0,1,5),color="k",ls="--",alpha=1.0)
#ax.legend(handles=handles)


xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.fill_between(xlim,0,0.1,alpha=0.1,color='k')
ax.fill_between((0.0,0.1),ylim[0],ylim[1],alpha=0.1,color='k')

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel("EI theoretical porosity (-)")
ax.set_ylabel("EI predicted porosity (-)")

q50 = np.quantile(np.abs(phi_b-ei_porosity),0.50)
q25 = np.quantile(np.abs(phi_b-ei_porosity),0.25)
q75 = np.quantile(np.abs(phi_b-ei_porosity),0.75)
iqr = q75-q25

ax.text(0.025,0.975,"%.2f(%.2f-%.2f)"%(q50,q25,q75),size=8)

# %% Comparison between voxels

if False:
    
    import nibabel as nib
    root = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/"
    ref_img = root+"NIFTI/Exp.nii.gz"
    ref_msk = root+"NIFTI/NEW_Mask_Exp.nii.gz"
    res_img = root+"REGISTRATION/Resampled_Insp.nii.gz"
    
    ref = nib.load(ref_img)
    msk = nib.load(ref_msk)
    res = nib.load(res_img)
    
    refdata = np.array(ref.dataobj)[np.array(msk.dataobj)==1]
    resdata = np.array(res.dataobj)[np.array(msk.dataobj)==1]
    
    # For PIG5-ARDSnet, 63.3% of the points in EI have lower intensities (more air) than
    # in EE. 