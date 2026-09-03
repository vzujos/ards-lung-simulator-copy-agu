# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import meshio as io
from scipy.stats import pearsonr

# %%

bin_and_number = {'NAT':0,
                  'PAT':1,
                  'AT':2,
                  'HIT':3,}

renamer = {"NAT":"Non-aerated tissue",
           "PAT":"Poorly-aerated tissue",
           "AT":"Normally-aerated tissue",
           "HIT":"Hyperinflated tissue"}

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
compartments = ["NAT","PAT","AT","HIT"]

# %%

AERATION_BINS = {
    "HIT": (-1100, -900.0),
    "AT":  (-900.0,  -500.0),
    "PAT": (-500.0,  -100.0),
    "NAT": (-100.0,   200.0),
}

def classify_aeration(hu_values: np.ndarray) -> dict:
    """
    Classify HU values into aeration compartments.
    Returns voxel counts per class, including OOB.
    """
    counts = {}
    used = np.zeros(hu_values.shape, dtype=bool)

    for name, (lo, hi) in AERATION_BINS.items():
        mask = (hu_values > lo) & (hu_values <= hi)
        counts[name] = int(np.count_nonzero(mask))
        used |= mask

    counts["OOB"] = int(np.count_nonzero(~used))
    counts["TOTAL"] = int(hu_values.size)

    return counts


def analyze_whole_lung(nib_ct, nib_seg) -> dict:
    """
    Whole-lung aeration analysis.
    """
    ct = np.asarray(nib_ct.dataobj)
    seg = np.asarray(nib_seg.dataobj).astype(bool)

    lung_voxels = ct[seg]

    return classify_aeration(lung_voxels)

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

def retrieve_regional_histogram(mesh_path,
                                direction,
                                weights=None,
                                nrois=10,
                                verbose=False,
                                deformed_state=True,
                                bins=[0.0,0.1,0.5,0.9,1.0],
                                porosity_field='Eulerian Porosity',
                                displacement_name = 'u'):
   
    mesh = io.read(mesh_path)
    
    if deformed_state: # Add deformation field to the mesh points
        xyz = mesh.points
        u = mesh.point_data[displacement_name]
        xyz += u
    else: # Use raw point data
        xyz = mesh.points
        
    # Retrieve porosity field
    porosity = mesh.point_data[porosity_field] 


    # Define the directions in use
    dirs = {"BA" : np.asmatrix([0.,0.,1.]).T, # BA tested; Direction checks out 
            "VD" : np.asmatrix([0.,1.,0.]).T, # VD tested; Direction checks out
            "RL" : np.asmatrix([1.,0.,0.]).T}

    # Dummy; Node mass should be used. How do I compute it?
    # TODO: Use the actual mass
    if weights is None:
        w = np.ones(xyz.shape[0])
    else:
        w = weights

    _, id_roi = IsoVolumetricSegmentation(dirs[direction],w,xyz,nrois)

    counter = {roi:{i:None for i in range(6)} for roi in range(nrois)}
    gen_binner = {roi:{i:None for i in [1,2,3,4]} for roi in range(nrois)}


    namer = {0:"OOB-NAT", # Out-of-bounds non-aerated tissue
             1:"NAT", # Non-aerated tissue
             2:"PAT", # Poorly aerated tissue
             3:" AT", # Normally aerated tissue
             4:"HIT", # Hyperinflated tissue
             5:"OOB-HIT"} # Out-of-bounds hypperinflated tissue

    for roi in range(nrois):
        
        roimask = id_roi == roi
        roi_porosity = porosity[roimask]
        N = len(roi_porosity)
        
        digit = np.digitize(roi_porosity,bins)

        for bin_ in range(6):
            counter[roi][bin_] = np.count_nonzero(digit==bin_)

        for i in [1,2,3,4]:
            if i == 1:
                gen_binner[roi][i-1] = (counter[roi][0]+counter[roi][1])/N
                if verbose: print("NAT: %.1f"%((counter[roi][0]+counter[roi][1])/N*100)+"%")
            elif i == 4:
                gen_binner[roi][i-1] = (counter[roi][4]+counter[roi][5])/N
                if verbose: print("HIT: %.1f"%((counter[roi][4]+counter[roi][5])/N*100)+"%")
            else:
                gen_binner[roi][i-1] = counter[roi][i]/N
                if verbose: print("%s: %.1f"%(namer[i],counter[roi][i]*100/N)+"%")

    return gen_binner, counter


def find_dorsal_ventral_bounds(seg: np.ndarray):
    """
    Returns (dorsal_index, ventral_index) where lung segmentation exists.
    """
    sy = seg.shape[1]

    dorsal = next(i for i in range(sy) if np.count_nonzero(seg[:, i, :]))
    ventral = next(i for i in range(sy) if np.count_nonzero(seg[:, sy - i - 1, :]))

    return dorsal, sy - ventral - 1


def analyze_isovolumetric_rois(
    nib_ct,
    nib_seg,
    N: int,
) -> dict:
    """
    Ventro–dorsal isovolumetric aeration analysis.
    Returns a list of ROI dictionaries.
    """
    ct = np.asarray(nib_ct.dataobj)
    seg = np.asarray(nib_seg.dataobj).astype(bool)

    total_voxels = int(np.count_nonzero(seg))
    target_per_roi = int(np.round(total_voxels / N))

    dorsal, ventral = find_dorsal_ventral_bounds(seg)

    rois = []
    voxel_buffer = []

    for y in range(dorsal, ventral + 1):
        slice_mask = seg[:, y, :]
        if not np.any(slice_mask):
            continue

        voxel_buffer.extend(ct[:, y, :][slice_mask])

        if len(voxel_buffer) >= target_per_roi:
            rois.append(classify_aeration(np.array(voxel_buffer)))
            voxel_buffer = []

    if voxel_buffer:
        rois.append(classify_aeration(np.array(voxel_buffer)))

    return {
        "N_requested": N,
        "N_actual": len(rois),
        "ROIs": rois
    }

def analyze_extreme_compartments(nib_ct, nib_seg, bins=100):
    """
    Analyze intensity distributions of extreme aeration compartments:
      - Non-aerated: I > -100 HU
      - Hyperinflated: I < -900 HU

    Parameters
    ----------
    nib_ct : nibabel.Nifti1Image
        CT image in HU
    nib_seg : nibabel.Nifti1Image
        Binary lung segmentation
    bins : int
        Number of bins for histograms
    """

    # Load data
    ct = np.asarray(nib_ct.dataobj)
    seg = np.asarray(nib_seg.dataobj).astype(bool)

    lung_voxels = ct[seg]

    compartments = {
        "Non-aerated (I > -100 HU)": lung_voxels[lung_voxels > -100],
        "Hyperinflated (I < -900 HU)": lung_voxels[lung_voxels < -900],
    }

    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    # --- Plot histograms ---
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(compartments),
        figsize=(6 * len(compartments), 4)
    )

    if len(compartments) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, compartments.items()):
        
        ax.hist(values, bins=bins)
        ax.set_title(name)
        ax.set_xlabel("Intensity [HU]")
        ax.set_ylabel("Voxel count")
        ax.grid(alpha=0.3)
        
        if name == "Non-aerated (I > -100 HU)":
            ax.set_xlim((-100,600))
        else:
            ax.set_xlim((-1500,-900))

    plt.tight_layout()
    plt.show()

    # --- Quantile statistics ---
    print("\nExtreme compartment intensity statistics")
    print("=" * 50)

    for name, values in compartments.items():

        if values.size == 0:
            print(f"\n{name}")
            print("  No voxels found.")
            continue

        qs = np.quantile(values, quantiles)
        q_dict = dict(zip(quantiles, qs))
        iqr = q_dict[0.75] - q_dict[0.25]

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Voxel count: {values.size}")
        print(f"q01 : {q_dict[0.01]:8.1f} HU")
        print(f"q05 : {q_dict[0.05]:8.1f} HU")
        print(f"q25 : {q_dict[0.25]:8.1f} HU")
        print(f"q50 : {q_dict[0.50]:8.1f} HU (median)")
        print(f"q75 : {q_dict[0.75]:8.1f} HU")
        print(f"q95 : {q_dict[0.95]:8.1f} HU")
        print(f"q99 : {q_dict[0.99]:8.1f} HU")
        print(f"IQR : {iqr:8.1f} HU")


# %% Declare targets

subject = 3
root = 'D:/CORNELL-NEWGEO/PIG%i/ARDSnet/NIFTI/'%subject

paths = {"Exp-CT":"Exp_median.nii.gz",
         "Exp-Seg":"NEW_Mask_Exp.nii.gz",
         "Insp-CT":"Insp_median.nii.gz",
         "Insp-Seg":"NEW_Mask_Insp.nii.gz"}

states = ["Exp","Insp"]

# Empty dictionary to handle result data
analyzer ={}

for state in states: 
    # Load images
    nib_ct = nib.load(root+paths["%s-CT"%state])
    nib_seg = nib.load(root+paths["%s-Seg"%state])
    # Analyze and store data
    analyzer.update({state:analyze_whole_lung(nib_ct,nib_seg)})
    # Add other relevant information
    voxel_volume = np.abs(np.prod([nib_ct.affine[i,i] for i in range(3)]))
    analyzer.update({"info":{"volume":voxel_volume}})

# %%

ee_nvoxnat = analyzer['Exp']['NAT']
ei_nvoxnat = analyzer['Insp']['NAT']
d_nvoxnat = ee_nvoxnat - ei_nvoxnat

print("Volume at EE (ml): %6.1f"%(analyzer["Exp"]["TOTAL"]*voxel_volume*1e-3))
print("Volume at EI (ml): %6.1f"%(analyzer["Insp"]["TOTAL"]*voxel_volume*1e-3))
print("Delta volume (mL): %6.1f"%((analyzer["Insp"]["TOTAL"]-analyzer["Exp"]["TOTAL"])*voxel_volume*1e-3))
print()
print("Number of NAT voxels at EE: %i"%ee_nvoxnat)
print("Number of NAT voxels at EI: %i"%ei_nvoxnat)
print("Estimated recruited voxels: %i"%d_nvoxnat)
print()
print("Volume of NAT voxels at EE (mL): %5.1f"%(ee_nvoxnat*voxel_volume*1e-3))
print("Volume of NAT voxels at EI (mL): %5.1f"%(ei_nvoxnat*voxel_volume*1e-3))
print("Estimated recruited volume (mL): %5.1f"%(d_nvoxnat*voxel_volume*1e-3))
print()
print("Recruited lung volume over tidal volume ratio: %.2f"%(d_nvoxnat/(analyzer["Insp"]["TOTAL"]-analyzer["Exp"]["TOTAL"])))

# %% Determine image-based regional aeration histograms

root = 'D:/CORNELL-NEWGEO/PIG%i/ARDSnet/NIFTI/'%subject

paths = {"Exp-CT":"Exp_median.nii.gz",
         "Exp-Seg":"NEW_Mask_Exp.nii.gz",
         "Insp-CT":"Insp_median.nii.gz",
         "Insp-Seg":"NEW_Mask_Insp.nii.gz"}

states = ["Exp","Insp"]

# Empty dictionary to handle result data
analyzer ={}

for state in states: 
    
    # Load images
    nib_ct = nib.load(root+paths["%s-CT"%state])
    nib_seg = nib.load(root+paths["%s-Seg"%state])
    
    # Analyze and store data
    analyzer.update({state:analyze_isovolumetric_rois(nib_ct,nib_seg,10)})
    
    # Add other relevant information
    voxel_volume = np.abs(np.prod([nib_ct.affine[i,i] for i in range(3)]))
    analyzer.update({"info":{"volume":voxel_volume}})

# %% Transform to required format

categories = ["NAT", "PAT", "AT", "HIT"]
states = ["Exp", "Insp"]

# number of ROIs (defensive)
n_rois = min(len(analyzer["Exp"]["ROIs"]),
             len(analyzer["Insp"]["ROIs"]))

roi_ids = np.arange(1, n_rois + 1)

# Containers
counts = {state: {cat: np.zeros(n_rois) for cat in categories} for state in states}
fractions = {state: {cat: np.zeros(n_rois) for cat in categories} for state in states}

for state in states:
    for i in range(n_rois):
        roi = analyzer[state]["ROIs"][i]
        total = roi["TOTAL"]

        for cat in categories:
            counts[state][cat][i] = roi[cat]
            fractions[state][cat][i] = roi[cat] / total
        
# %% Generate a subjectwise plot for the intensity-based aeration histogram

fig, axes = plt.subplots(nrows=1, ncols=len(categories),
                         figsize=(3*len(categories), 4),)

colors = {"Exp": "royalblue", "Insp": "navy"}
markers = {"Exp": "o", "Insp": "s"}

for c, (ax,cat) in enumerate(zip(axes,categories)):

    for state in states:
        
        # Generate markers and lines
        ax.scatter(fractions[state][cat],
                   roi_ids,
                   color=colors[state],
                   marker=markers[state],
                   label=state)
        
        ax.plot(fractions[state][cat],
                roi_ids,
                color=colors[state],
                alpha=0.3)
    
    # Place titles on top and fix axes
    ax.set_title(renamer[cat])
    
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    
    if cat == 'NAT':
        ax.set_ylabel("ROI ID")
        ax.set_yticks(roi_ids)
        ax.set_yticklabels(roi_ids)
        ax.text(-0.24, 0.3, 'Ventral',weight='bold')
        ax.text(-0.24, 10.8, 'Dorsal',weight='bold')

    else:
        ax.set_yticks([])

    ax.set_xlabel("Fraction [-]")

# Single global legend
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc="upper right", ncol=2)

fig.suptitle(
    f"PIG{subject} – ARDSnet \n Ventro–dorsal isovolumetric ROIs",
    y=1.01, weight='bold'
)

plt.tight_layout()

# %% Analyze the extreme compartments (NAT and HIT)

analyze_extreme_compartments(nib_ct, nib_seg)

# %% Determine mesh-based histograms


meshes_and_subjects = {4:"medium",
                       2:"medium-fine",
                       3:"medium-fine",
                       5:"medium-fine",
                       6:"medium-fine"}

# Retrieve the subject-wise mesh values for the analysis
mesh_type = meshes_and_subjects[subject]
mroot = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/'%(subject, mesh_type)
nrois=10
directions = {'BA':np.asmatrix([0.,0.,1.]).T,
              'VD':np.asmatrix([0.,1.,0.]).T,}
direction='VD'

reg_mesh = mroot+'reg_anim_ready.vtu'
sim_mesh = mroot+'sim_anim_ready.vtu'

rmesh = io.read(reg_mesh); smesh = io.read(sim_mesh)
mass = rmesh.point_data['Mass']


mesh_handle = {}

for name, mesh_path in zip(['Simulation','Experiment'], [sim_mesh,reg_mesh]):
      
    
    porosity_field = 'End-Inspiratory Porosity'
            
    hist, count = retrieve_regional_histogram(mesh_path,direction,
                                       weights=mass, verbose=False,
                                       deformed_state=True,
                                       porosity_field=porosity_field,
                                       nrois=nrois,
                                       displacement_name='DispField')
        
    mesh_handle.update({name:{'hist':hist,
                             'count':count} } )

hist, count = retrieve_regional_histogram(mesh_path,direction,
                                           weights=mass, verbose=False,
                                           deformed_state=False,
                                           porosity_field='End-Expiratory Porosity',
                                           nrois=nrois,
                                           displacement_name='DispField')
            
mesh_handle.update({'Baseline':{'hist':hist,
                                'count':count} } )



# %% Generate comparison plots for the histogram-binned lung data

# Generate data structures
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(6,6),dpi=200)

# For every axes, a compartment plot
for ax,compartment in zip(axes.flatten(),compartments):
   
    # Retrieve data
    xdata = fractions['Insp'][compartment]
    ydata_exp = np.array([mesh_handle['Experiment']['hist'][i][bin_and_number[compartment]] for i in range(10)])
    ydata_sim = np.array([mesh_handle['Simulation']['hist'][i][bin_and_number[compartment]] for i in range(10)])
    
    # Generate scattered points
    ax.scatter(xdata,ydata_exp, label="Experiment",color="royalblue",alpha=0.75,marker='x')
    ax.scatter(xdata,ydata_sim, label="Simulation",color="navy",alpha=0.75,marker='+')
   
    # Generate identity line
    line = np.linspace(0,1,5)
    ax.plot(line,line, ls='--',alpha=0.5,color='k')
    
    # Pearson correlation: Exp
    r_exp, p_exp = pearsonr(xdata, ydata_exp)
    
    # Pearson correlation: Simulation
    r_sim, p_sim = pearsonr(xdata, ydata_sim)
    
    # Labels and the such
    ax.set_xlabel("Fraction\n(Image-based histogram)")
    ax.set_ylabel("Fraction\n(Mesh-based histogram)")
    ax.set_title(renamer[compartment])
    label_sim = "$r_{sim}$"+f"={r_sim:.2f} {pvalue_to_stars(p_sim)}"
    label_exp = "$r_{exp}$"+f"={r_exp:.2f} {pvalue_to_stars(p_exp)}"
    ax.text(0.025,0.9,label_sim,size=8)
    ax.text(0.025,0.82,label_exp,size=8)
    
    # Place legend at the first plot
    if compartment=='NAT':
        ax.legend()
    
    plt.suptitle('End-inspiration aeration compartments',weight='bold')
    
    plt.tight_layout()
    
# %% For the end-expiratory information, a comparison between image-based and mesh-based histograms

# Generate data structures
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(6,6),dpi=200)

# A plot for each compartment
for ax,compartment in zip(axes.flatten(),compartments):

    # Retrieve data
    xdata = fractions['Exp'][compartment]
    ydata = np.array([mesh_handle['Baseline']['hist'][i][bin_and_number[compartment]] for i in range(10)])
    
    # Scatter datapoints
    ax.scatter(xdata,ydata,marker='x',color='navy')
    
    # Plot identity line
    line = np.linspace(0,1,5)
    ax.plot(line,line, ls='--',alpha=0.5,color='k')
    
    # Pearson correlation: Baseline (Expiration)
    r, p = pearsonr(xdata, ydata)
    
    # Place captions and the such
    ax.set_xlabel("Fraction\n(Image-based histogram)")
    ax.set_ylabel("Fraction\n(Mesh-based histogram)")
    ax.set_title(renamer[compartment])
    label = f"r={r:.2f} {pvalue_to_stars(p)}"
    ax.text(0.05,0.9,label)
    
plt.suptitle("End-expiration aeration compartments", weight='bold')
plt.tight_layout()