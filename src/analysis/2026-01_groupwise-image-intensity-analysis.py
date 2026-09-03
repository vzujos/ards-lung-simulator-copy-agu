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
from scipy.stats import pearsonr, linregress
import pickle
from scipy.ndimage import median_filter


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
                                displacement_name='u'):
   
    mesh = io.read(mesh_path)
    
    if deformed_state:
        xyz = mesh.points.copy()
        u = mesh.point_data[displacement_name]
        xyz += u
    else:
        xyz = mesh.points.copy()
        
    porosity = mesh.point_data[porosity_field]

    dirs = {
        "BA": np.asmatrix([0., 0., 1.]).T,
        "VD": np.asmatrix([0., 1., 0.]).T,
        "RL": np.asmatrix([1., 0., 0.]).T
    }

    if weights is None:
        w = np.ones(xyz.shape[0])
    else:
        w = weights

    _, id_roi = IsoVolumetricSegmentation(dirs[direction], w, xyz, nrois)

    counter = {roi: {i: None for i in range(6)} for roi in range(nrois)}
    gen_binner = {roi: {i: None for i in [0, 1, 2, 3]} for roi in range(nrois)}

    for roi in range(nrois):

        roimask = id_roi == roi
        roi_porosity = porosity[roimask]
        roi_weights = w[roimask]

        N = roi_porosity.size

        # ---------- weighted mean porosity ----------
        if np.sum(roi_weights) > 0:
            phi_mean = float(
                np.sum(roi_weights * roi_porosity) / np.sum(roi_weights)
            )
        else:
            phi_mean = np.nan

        # ---------- histogram binning ----------
        digit = np.digitize(roi_porosity, bins)

        for bin_ in range(6):
            counter[roi][bin_] = np.count_nonzero(digit == bin_)

        # NAT
        gen_binner[roi][0] = (counter[roi][0] + counter[roi][1]) / N
        # PAT
        gen_binner[roi][1] = counter[roi][2] / N
        # AT
        gen_binner[roi][2] = counter[roi][3] / N
        # HIT
        gen_binner[roi][3] = (counter[roi][4] + counter[roi][5]) / N

        # ---------- attach mean porosity (non-destructive) ----------
        gen_binner[roi]["phi_mean"] = phi_mean

        if verbose:
            print(f"ROI {roi}: φ̄ = {phi_mean:.3f}")

    return gen_binner, counter

def retrieve_regional_fields(mesh_path,
                             direction,
                             weights=None,
                             nrois=10,
                             verbose=False,
                             jacobian_field='Jacobian',
                             dgf_field='Delta Porosity'):
   
    mesh = io.read(mesh_path)
    xyz = mesh.points.copy()
        
    jacobian = mesh.point_data[jacobian_field]
    dgf = mesh.point_data[dgf_field]

    dirs = {
        "BA": np.asmatrix([0., 0., 1.]).T,
        "VD": np.asmatrix([0., 1., 0.]).T,
        "RL": np.asmatrix([1., 0., 0.]).T
    }

    if weights is None:
        w = np.ones(xyz.shape[0])
    else:
        w = weights

    _, id_roi = IsoVolumetricSegmentation(dirs[direction], w, xyz, nrois)

    holder = {roi: {i:None for i in ["j","dgf"]} for roi in range(nrois)}

    for roi in range(nrois):

        roimask = id_roi == roi
        roi_jacobian = jacobian[roimask]
        roi_dgf = dgf[roimask]
        roi_weights = w[roimask]

        # ---------- weighted mean porosity ----------
        if np.sum(roi_weights) > 0:
            j_mean = float(np.sum(roi_weights * roi_jacobian) / np.sum(roi_weights))
            dgf_mean = float(np.sum(roi_weights * roi_dgf) / np.sum(roi_weights))

        else:
            j_mean = np.nan
            dgf_mean = np.nan

        # ---------- attach mean porosity (non-destructive) ----------
        holder[roi]["j"] = j_mean
        holder[roi]["dgf"] = dgf_mean

    return holder

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

    Returns
    -------
    dict with keys:
        - N_requested
        - N_actual
        - ROIs : list of dicts, each containing:
            * aeration compartment counts (NAT, PAT, AT, HIT, OOB, TOTAL)
            * phi_mean : mean aeration content of the ROI
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
            voxels = np.array(voxel_buffer)

            # ---- Aeration content computation ----
            T = -voxels / 1000.0
            T[T < 0.0] = 0.0
            T[T > 1.0] = 1.0
            phi_mean = float(np.mean(T))

            roi_data = classify_aeration(voxels)
            roi_data["phi_mean"] = phi_mean

            rois.append(roi_data)
            voxel_buffer = []

    # Handle remainder voxels
    if voxel_buffer:
        voxels = np.array(voxel_buffer)

        T = -voxels / 1000.0
        T[T < 0.0] = 0.0
        T[T > 1.0] = 1.0
        phi_mean = float(np.mean(T))

        roi_data = classify_aeration(voxels)
        roi_data["phi_mean"] = phi_mean

        rois.append(roi_data)

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


# %%



def apply_median_filter_nifti(
    input_nii_path,
    output_dir=None,
    kernel_size=3,
    suffix="_median"
):
    """
    Apply a 3D median filter to a NIfTI image and save the result,
    preserving affine and header.

    Parameters
    ----------
    input_nii_path : str
        Path to input NIfTI file (.nii or .nii.gz)
    output_dir : str or None
        Directory where the filtered image will be saved.
        If None, saves next to the input file.
    kernel_size : int or tuple
        Median filter size (default: 3 → 3x3x3)
    suffix : str
        Suffix added to the output filename (before .nii/.nii.gz)

    Returns
    -------
    output_nii_path : str
        Path to the saved filtered NIfTI file
    """

    # Load NIfTI
    nii = nib.load(input_nii_path)
    data = nii.get_fdata(dtype=np.float32)

    # Apply median filter
    filtered_data = median_filter(data, size=kernel_size)

    # Create output filename
    base = os.path.basename(input_nii_path)
    if base.endswith(".nii.gz"):
        name = base[:-7]
        ext = ".nii.gz"
    elif base.endswith(".nii"):
        name = base[:-4]
        ext = ".nii"
    else:
        raise ValueError("Input file must be .nii or .nii.gz")

    output_name = f"{name}{suffix}{ext}"

    if output_dir is None:
        output_dir = os.path.dirname(input_nii_path)

    output_nii_path = os.path.join(output_dir, output_name)

    # Create new NIfTI (reuse affine and header)
    new_nii = nib.Nifti1Image(
        filtered_data,
        affine=nii.affine,
        header=nii.header
    )

    # Save
    nib.save(new_nii, output_nii_path)

    return output_nii_path


# %% Declare targets

PROCESS_DATA = False

subjects = [2,3,4,5,6]

paths = {"Exp-CT":"Exp_median.nii.gz",
         "Exp-Seg":"NEW_Mask_Exp.nii.gz",
         "Insp-CT":"Insp_median.nii.gz",
         "Insp-Seg":"NEW_Mask_Insp.nii.gz"}

states = ["Exp","Insp"]
categories = ["NAT", "PAT", "AT", "HIT"]

meshes_and_subjects = {4:"medium",
                       2:"medium-fine",
                       3:"medium-fine",
                       5:"medium-fine",
                       6:"medium-fine"}

# Big data manager
data = {"global-image-hist":{},
                "regional-image-hist":{},
                "regional-image-count":{},
                "regional-mesh-hist":{},
                "regional-mesh-count":{},
                "regional-mesh-fields":{}}


nrois=10
directions = {'BA':np.asmatrix([0.,0.,1.]).T,
              'VD':np.asmatrix([0.,1.,0.]).T,}

direction='VD'

if PROCESS_DATA:
    
    # For every subject
    for subject in subjects:
        
        root = 'D:/CORNELL-NEWGEO/PIG%i/ARDSnet/NIFTI/'%subject
        mesh_type = meshes_and_subjects[subject]
        mroot = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/'%(subject, mesh_type)

        # Compute global histogram data
        global_analyzer ={}
    
        for state in states: 
            # Load images
            nib_ct = nib.load(root+paths["%s-CT"%state])
            nib_seg = nib.load(root+paths["%s-Seg"%state])
            # Analyze and store data
            global_analyzer.update({state:analyze_whole_lung(nib_ct,nib_seg)})
            # Add other relevant information
            voxel_volume = np.abs(np.prod([nib_ct.affine[i,i] for i in range(3)]))
            global_analyzer.update({"info":{"volume":voxel_volume}})
    
        # Add subject-wise data to the data manager
        data['global-image-hist'].update({subject:global_analyzer})
        
        # Determine image-based regional aeration histograms
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
         
        # number of ROIs (defensive)
        n_rois = min(len(analyzer["Exp"]["ROIs"]),
                     len(analyzer["Insp"]["ROIs"]))
        
        # Label for the ROIs
        roi_ids = np.arange(1, n_rois + 1)
        
        # Containers
        counts = {state: {cat: np.zeros(n_rois) for cat in categories} for state in states}
        fractions = {state: {cat: np.zeros(n_rois) for cat in (categories+["phi_mean"])} for state in states}
        
        # For the inspiratory and expiratory images
        for state in states:
            for i in range(n_rois):
                roi = analyzer[state]["ROIs"][i]
                total = roi["TOTAL"]
        
                for cat in categories:
                    counts[state][cat][i] = roi[cat]
                    fractions[state][cat][i] = roi[cat] / total
                fractions[state]["phi_mean"][i] = roi["phi_mean"]

        
        data['regional-image-hist'].update({subject:fractions.copy()})
        data['regional-image-count'].update({subject:counts.copy()})

        # Determine mesh-based histograms
        reg_mesh = mroot+'reg_anim_ready.vtu'
        sim_mesh = mroot+'sim_anim_ready.vtu'
        
        # Load meshes and retrieve mass
        rmesh = io.read(reg_mesh); smesh = io.read(sim_mesh)
        mass = rmesh.point_data['Mass']
    
        # Handle for mesh-related data
        mesh_handle = {}
        
        for name, mesh_path in zip(['Simulation','Experiment'], [sim_mesh,reg_mesh]):
              
            
            porosity_field = 'End-Inspiratory Porosity'
                    
            hist, count = retrieve_regional_histogram(mesh_path,direction,
                                               weights=mass, verbose=False,
                                               deformed_state=True,
                                               porosity_field=porosity_field,
                                               nrois=nrois,
                                               displacement_name='DispField')
                
            mesh_handle.update({name:{'hist':hist.copy(),
                                     'count':count.copy()} } )
               
        hist, count = retrieve_regional_histogram(mesh_path,direction,
                                                   weights=mass, verbose=False,
                                                   deformed_state=False,
                                                   porosity_field='End-Expiratory Porosity',
                                                   nrois=nrois,
                                                   displacement_name='DispField')
        
        
        # Determine regional J and DGF for the meshes
        holder_sim = retrieve_regional_fields(sim_mesh,
                             direction,
                             weights=mass,
                             nrois=10,
                             verbose=False,
                             jacobian_field='Jacobian Forward',
                             dgf_field='Delta Porosity')
            
        holder_exp = retrieve_regional_fields(reg_mesh,
                             direction,
                             weights=mass,
                             nrois=10,
                             jacobian_field='Jacobian',
                             dgf_field='Delta Porosity')
        
                    
        mesh_handle.update({'Baseline':{'hist':hist.copy(),
                                        'count':count.copy()} } )
    
        data['regional-mesh-hist'].update({subject:{'Exp':mesh_handle['Baseline']['hist'],
                                           'Insp-Simulation':mesh_handle['Simulation']['hist'],
                                           'Insp-Experiment':mesh_handle['Experiment']['hist'],}},)
        
        data['regional-mesh-count'].update({subject:{'Exp':mesh_handle['Baseline']['count'],
                                            'Insp-Simulation':mesh_handle['Simulation']['count'],
                                            'Insp-Experiment':mesh_handle['Experiment']['count'],},})
        
        data['regional-mesh-fields'].update({subject:{"Simulation":holder_sim,
                                                      "Experiment":holder_exp}})
    
 
# %%
    
    with open('./groupwise_intensity_analysis.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    
    with open('./groupwise_intensity_analysis.pkl', 'rb') as f:
        data = pickle.load(f)

# %%

sdata = {s:build_sorted_data_per_subject(data=data,
                                         subject=s,
                                         compartments=["NAT","PAT","AT","HIT"],
                                         nrois=10) for s in subjects}

# %% Agreement in aeration compartments in end-expiration mesh and images.

fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(6,6),dpi=200)

xtype, xstate = ('mesh','Insp-Experiment')
ytype, ystate = ('mesh','Insp-Simulation')

labeler = {'image':'Image-based aeration',
           'mesh':'Mesh-based aeration',
           'Insp':'(End-inspiratory pause)',
           'Insp-Simulation':'(Simulation)',
           'Insp-Experiment':'(Experiment)',
           'Exp':'(End-expiratory pause)'}

for ax,compartment in zip(axes.flatten(),compartments):
    
    xdata = np.hstack([sdata[s][xtype][xstate][compartment] for s in subjects])
    ydata = np.hstack([sdata[s][ytype][ystate][compartment] for s in subjects])
    
    ax.scatter(xdata,ydata,marker='x',alpha=0.5)
    
    ax.set_xlim((0,1)); ax.set_ylim((0,1))
    
    ax.plot(np.linspace(0,1),np.linspace(0,1),color='k',alpha=0.25,ls='--')
    
    ax.set_title(renamer[compartment])
    
    ax.set_xlabel(labeler[xtype]+"\n"+labeler[xstate])
    ax.set_ylabel(labeler[ytype]+"\n"+labeler[ystate])


    diff = xdata - ydata
    mae = np.mean(np.abs(diff))
    ax.text(0.05,0.90,"MAE: %.3f"%mae,size=10)

plt.suptitle('Aggregated ROI information',weight='bold')
plt.tight_layout()

# %% Agreement between simulated and experimental mesh-based aerations

fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(4,4),dpi=200)

xtype, xstate = ('mesh','Insp-Simulation')
ytype, ystate = ('mesh','Insp-Experiment')

labeler = {'image':'Image-based aeration',
           'mesh':'Mesh-based aeration',
           'Insp':'(End-inspiratory pause)',
           'Insp-Simulation':'(Simulation)',
           'Insp-Experiment':'(Experiment)',
           'Exp':'(End-expiratory pause)'}
    
xdata = np.hstack([sdata[s][xtype][xstate]["phi_mean"] for s in subjects])
ydata = np.hstack([sdata[s][ytype][ystate]["phi_mean"] for s in subjects])
    
ax.scatter(xdata,ydata,marker='x',alpha=0.5)
ax.set_xlim((0,1)); ax.set_ylim((0,1))
    
ax.plot(np.linspace(0,1),np.linspace(0,1),color='k',alpha=0.25,ls='--')
    
ax.set_title("Regional aeration correlation")
    
ax.set_xlabel(labeler[xtype]+"\n"+labeler[xstate])
ax.set_ylabel(labeler[ytype]+"\n"+labeler[ystate])

r, p = pearsonr(xdata, ydata)
slope, intercept, r_val, p_val, stderr = linregress(xdata, ydata)

    
xfit = np.linspace(xdata.min(), xdata.max(), 100)
yfit = slope * xfit + intercept
    
if p_val<0.05:
    ax.plot(xfit, yfit, color='r', alpha=0.50, ls='--')


plt.suptitle('Aggregated ROI information',weight='bold')
plt.tight_layout()

# %% Subject-wise comparison between image-based and mesh based aeration in end-expiration

fig, axes = plt.subplots(ncols=5,nrows=1, figsize=(12,3),dpi=300)

xtype, xstate = ('image','Exp')
ytype, ystate = ('mesh','Exp')

labeler = {'image':'Image-based aeration',
           'mesh':'Mesh-based aeration',
           'Insp':'(End-inspiratory pause)',
           'Insp-Simulation':'(Simulation)',
           'Insp-Experiment':'(Experiment)',
           'Exp':'(End-expiratory pause)'}

for s, ax in zip(subjects,axes.flatten()):
    
    xdata = np.hstack([sdata[s][xtype][xstate]["phi_mean"]])
    ydata = np.hstack([sdata[s][ytype][ystate]["phi_mean"]])
        
    ax.scatter(xdata,ydata,marker='x',alpha=0.75,color='navy')
    ax.set_xlim((0,1)); ax.set_ylim((0,1))
        
    ax.plot(np.linspace(0,1),np.linspace(0,1),color='k',alpha=0.25,ls='--')
        
    ax.set_title("PIG %i"%s,weight='bold')
        
    ax.set_xlabel(labeler[xtype]+"\n"+labeler[xstate])
    if s == 2:
        ax.set_ylabel(labeler[ytype]+"\n"+labeler[ystate])
    else:
        ax.set_yticks([])
    
    ax.set_xticks(np.linspace(0,1,3))

    diff = xdata - ydata
    mae = np.mean(np.abs(diff))
    ax.text(0.05,0.90,"MAE: %.3f"%mae,size=10)

plt.suptitle('Subject-wise ROI information - End-expiratory aeration',weight='bold')
plt.tight_layout()

# %% Group-wise comparison between image-based and mesh based end-expiratory aeration

fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(5,5),dpi=300)

xtype, xstate = ('image','Exp')
ytype, ystate = ('mesh','Exp')

labeler = {'image':'Image-based aeration',
           'mesh':'Mesh-based aeration',
           'Insp':'(End-inspiratory pause)',
           'Insp-Simulation':'(Simulation)',
           'Insp-Experiment':'(Experiment)',
           'Exp':'(End-expiratory pause)'}

if True:
    
    xdata = np.hstack([sdata[s][xtype][xstate]["phi_mean"] for s in subjects])
    ydata = np.hstack([sdata[s][ytype][ystate]["phi_mean"] for s in subjects])
        
    ax.scatter(xdata,ydata,marker='x',alpha=0.75,color='navy')
    ax.set_xlim((0,1)); ax.set_ylim((0,1))
        
    ax.plot(np.linspace(0,1),np.linspace(0,1),color='k',alpha=0.25,ls='--')
        
    ax.set_title("Group-wise ROI information",weight='bold')
        
    ax.set_xlabel(labeler[xtype]+"\n"+labeler[xstate])
    ax.set_ylabel(labeler[ytype]+"\n"+labeler[ystate])
    
    ax.set_xticks(np.linspace(0,1,3))

    diff = xdata - ydata
    mae = np.mean(np.abs(diff))
    ax.text(0.05,0.90,"MAE: %.3f"%mae,size=10)

plt.suptitle('End-expiratory aeration',weight='bold')
plt.tight_layout()

# %% Subject-wise comparison between image-based and mesh based aeration in end-inspiration

fig, axes = plt.subplots(ncols=5,nrows=1, figsize=(12,3),dpi=300)

xtype, xstate = ('image','Insp')
ytype1, ystate1 = ('mesh','Insp-Experiment')
ytype2, ystate2 = ('mesh','Insp-Simulation')

labeler = {'image':'Image-based aeration',
           'mesh':'Mesh-based aeration',
           'Insp':'(End-inspiratory pause)',
           'Insp-Simulation':'(Simulation)',
           'Insp-Experiment':'(Experiment)',
           'Exp':'(End-expiratory pause)'}

for s, ax in zip(subjects,axes.flatten()):
    
    xdata = np.hstack([sdata[s][xtype][xstate]["phi_mean"]])
    ydata1 = np.hstack([sdata[s][ytype1][ystate1]["phi_mean"]])
    ydata2 = np.hstack([sdata[s][ytype2][ystate2]["phi_mean"]])
        
    ax.scatter(xdata,ydata1,marker='+',alpha=0.75,color='navy',label="Experiment")
    ax.scatter(xdata,ydata2,marker='x',alpha=0.75,color='royalblue',label="Simulation")

    ax.set_xlim((0,1)); ax.set_ylim((0,1))
        
    ax.plot(np.linspace(0,1),np.linspace(0,1),color='k',alpha=0.25,ls='--')
        
    ax.set_title("PIG %i"%s,weight='bold')
        
    ax.set_xlabel(labeler[xtype]+"\n"+labeler[xstate])
    if s == 2:
        ax.set_ylabel(labeler[ytype])
    else:
        ax.set_yticks([])
    
    ax.set_xticks(np.linspace(0,1,3))
    
    for d,(typ, ydata) in enumerate(zip(["Exp","Sim"],[ydata1, ydata2])):
        diff = xdata - ydata
        mae = np.mean(np.abs(diff))
        ax.text(0.05,0.90-d*0.10,"$MAE_{%s}$:"%typ+"%.3f"%mae,size=10)

axes[0].legend(loc='lower right')


plt.suptitle('Subject-wise ROI information - End-inspiratory aeration',weight='bold')
plt.tight_layout()

# %% Comparison between image- and mesh-based end-inspiratory aeration 

fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(5,5),dpi=300)

xtype, xstate = ('image','Insp')
ytype1, ystate1 = ('mesh','Insp-Experiment')
ytype2, ystate2 = ('mesh','Insp-Simulation')

labeler = {'image':'Image-based aeration',
           'mesh':'Mesh-based aeration',
           'Insp':'(End-inspiratory pause)',
           'Insp-Simulation':'(Simulation)',
           'Insp-Experiment':'(Experiment)',
           'Exp':'(End-expiratory pause)'}

    
xdata = np.hstack([sdata[s][xtype][xstate]["phi_mean"] for s in subjects])
ydata1 = np.hstack([sdata[s][ytype1][ystate1]["phi_mean"] for s in subjects])
ydata2 = np.hstack([sdata[s][ytype2][ystate2]["phi_mean"] for s in subjects])

r1, p1 = pearsonr(xdata, ydata1)
r2, p2 = pearsonr(xdata, ydata2)
        
ax.scatter(xdata,ydata1,marker='+',alpha=0.75,color='navy',label="Experiment")
ax.scatter(xdata,ydata2,marker='x',alpha=0.75,color='royalblue',label="Simulation")

ax.set_xlim((0,1)); ax.set_ylim((0,1))
        
ax.plot(np.linspace(0,1),np.linspace(0,1),color='k',alpha=0.25,ls='--')
        
ax.set_title("Group-wise ROI information",weight='bold')
        
ax.set_xlabel(labeler[xtype]+"\n"+labeler[xstate])
ax.set_ylabel(labeler[ytype]+"\n"+labeler['Insp'])
    
ax.set_xticks(np.linspace(0,1,3))
    
for d,(typ, ydata) in enumerate(zip(["Exp","Sim"],[ydata1, ydata2])):
    diff = xdata - ydata
    mae = np.mean(np.abs(diff))
    ax.text(0.05,0.90-d*0.05,"$MAE_{%s}$:"%typ+"%.3f"%mae,size=10)

ax.legend(loc='lower right') 

plt.suptitle('End-inspiratory aeration',weight='bold')
plt.tight_layout()

# %% Comparison between and mesh-based end-inspiratory aeration 

fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(5,5),dpi=300)

xtype, xstate = ('mesh','Insp-Experiment')
ytype, ystate = ('mesh','Insp-Simulation')

labeler = {'image':'Image-based aeration',
           'mesh':'Mesh-based aeration',
           'Insp':'(End-inspiratory pause)',
           'Insp-Simulation':'(Simulation)',
           'Insp-Experiment':'(Experiment)',
           'Exp':'(End-expiratory pause)'}

    
xdata = np.hstack([sdata[s][xtype][xstate]["phi_mean"] for s in subjects])
ydata = np.hstack([sdata[s][ytype][ystate]["phi_mean"] for s in subjects])

r, p = pearsonr(xdata, ydata)
        
ax.scatter(xdata,ydata,marker='+',alpha=0.75,color='navy',label="Experiment")
#ax.scatter(xdata,ydata2,marker='x',alpha=0.75,color='royalblue',label="Simulation")

ax.set_xlim((0,1)); ax.set_ylim((0,1))
        
ax.plot(np.linspace(0,1),np.linspace(0,1),color='k',alpha=0.25,ls='--')
        
ax.set_title("Group-wise ROI information",weight='bold')
        
ax.set_xlabel(labeler[xtype]+"\n"+labeler[xstate])
ax.set_ylabel(labeler[ytype]+"\n"+labeler['Insp'])
    
ax.set_xticks(np.linspace(0,1,3))
    
#for d,(typ, ydata) in enumerate(zip(["Exp","Sim"],[ydata1, ydata2])):
#    diff = xdata - ydata
#    mae = np.mean(np.abs(diff))
#    ax.text(0.05,0.90-d*0.05,"$MAE_{%s}$:"%typ+"%.3f"%mae,size=10)

ax.legend(loc='lower right') 

plt.suptitle('End-inspiratory aeration',weight='bold')
plt.tight_layout()

# %% Associate recruitment error to error in the regional aeration estimation

# Associate recruitment to error (?)
image_nat_insp = np.hstack([sdata[s]['image']['Insp']['NAT'] for s in subjects])
image_nat_exp = np.hstack([sdata[s]['image']['Exp']['NAT'] for s in subjects])

# Reduction of the NAT compartment
image_dnat = image_nat_exp - image_nat_insp

# Error in the aeration prediction (Experiment/Simulation - Image )
aeration_image_insp = np.hstack([sdata[s]['image']['Insp']["phi_mean"] for s in subjects])
aeration_mesh_simulation = np.hstack([sdata[s]['mesh']['Insp-Simulation']["phi_mean"] for s in subjects])
aeration_mesh_experiment = np.hstack([sdata[s]['mesh']['Insp-Experiment']["phi_mean"] for s in subjects])
aeration_error_simulation = aeration_image_insp - aeration_mesh_simulation
aeration_error_experiment = aeration_image_insp - aeration_mesh_experiment

# Scatter plot
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(6,3),dpi=300)

ydatas = [aeration_error_simulation,aeration_error_experiment]
labels = ["Simulation", "Experiment"]
colors = ["royalblue","navy"]
markers = ['x','+']

for e, (ax,ydata,label,color,marker) in enumerate(zip(axes,ydatas,labels,colors,markers)):
    
    ydata=np.abs(ydata)
    
    r, p = pearsonr(image_dnat, ydata)
    slope, intercept, r_val, p_val, stderr = linregress(image_dnat, ydata)

    
    xfit = np.linspace(image_dnat.min(), image_dnat.max(), 100)
    yfit = slope * xfit + intercept
    
    if p_val<0.05:
        ax.plot(xfit, yfit, color=color, alpha=0.50, ls='--')

    ax.scatter(image_dnat, ydata, marker=marker,color=color,label=label, alpha=0.5)
    ax.set_xlabel("Recruitment (-)\nReduction of the NAT compartment", size=8)
    ax.set_ylabel("Aeration MAE (-)")
    ax.set_title(label, weight='bold')
    ax.set_ylim((0,0.08))
    ax.set_xlim((0,0.3))
    
    ax.text(0.02, 0.075, "r=%.2f"%r+pvalue_to_stars(p),ha='left')
    
plt.tight_layout()

# %% Associate hyperinflation to error in the regional aeration estimation

# Associate hyperinflation to aeration error
image_hit_insp = np.hstack([sdata[s]['image']['Insp']['HIT'] for s in subjects])
image_hit_exp = np.hstack([sdata[s]['image']['Exp']['HIT'] for s in subjects])

# Change of the HIT compartment
image_dhit = image_hit_insp - image_hit_exp

# Error in the aeration prediction (Experiment/Simulation - Image )
aeration_image_insp = np.hstack([sdata[s]['image']['Insp']["phi_mean"] for s in subjects])
aeration_mesh_simulation = np.hstack([sdata[s]['mesh']['Insp-Simulation']["phi_mean"] for s in subjects])
aeration_mesh_experiment = np.hstack([sdata[s]['mesh']['Insp-Experiment']["phi_mean"] for s in subjects])
aeration_error_simulation = aeration_image_insp - aeration_mesh_simulation
aeration_error_experiment = aeration_image_insp - aeration_mesh_experiment

# Scatter plot
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(6,3),dpi=200)

ydatas = [aeration_error_simulation,aeration_error_experiment]
labels = ["Simulation", "Experiment"]
colors = ["royalblue","navy"]
markers = ['x','+']

for e, (ax,ydata,label,color,marker) in enumerate(zip(axes,ydatas,labels,colors,markers)):
    
    ydata=np.abs(ydata)
    
    r, p = pearsonr(image_dhit, ydata)
    slope, intercept, r_val, p_val, stderr = linregress(image_dhit, ydata)

    
    xfit = np.linspace(image_dhit.min(), image_dhit.max(), 100)
    yfit = slope * xfit + intercept
    
    if p_val<0.05:
        ax.plot(xfit, yfit, color=color, alpha=0.50, ls='--')

    ax.scatter(image_dhit, ydata, marker=marker,color=color,label=label, alpha=0.5)
    ax.set_xlabel("Hyperinflation (-)\nChange in the HIT compartment", size=8)
    ax.set_ylabel("Aeration MAE (-)")
    ax.set_title(label, weight='bold')
    ax.set_ylim((0,0.08))
    
    ax.text(0.00, 0.075, "r=%.2f"%r+pvalue_to_stars(p),ha='left')
    
plt.tight_layout()

# %% Compare simulation/experiment Jacobian and DGF

# Retrieve jacobian information
mesh_j_sim = np.hstack([sdata[s]['mesh']['Simulation']['j'] for s in subjects])
mesh_j_exp = np.hstack([sdata[s]['mesh']['Experiment']['j'] for s in subjects])

# Retrieve delta gas fraction information
mesh_dgf_sim = np.hstack([sdata[s]['mesh']['Simulation']['dgf'] for s in subjects])
mesh_dgf_exp = np.hstack([sdata[s]['mesh']['Experiment']['dgf'] for s in subjects])

# Scatter plot
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(6,3),dpi=300)

ydatas = [mesh_j_sim ,mesh_j_exp]
xdatas = [mesh_dgf_sim, mesh_dgf_exp]
labels = ["Simulation", "Experiment"]
colors = ["royalblue","navy"]
markers = ['x','+']

for e, (ax,ydata,xdata,label,color,marker) in enumerate(zip(axes,ydatas,xdatas,labels,colors,markers)):
    
    ydata=np.abs(ydata)
    
    r, p = pearsonr(xdata, ydata)
    slope, intercept, r_val, p_val, stderr = linregress(xdata, ydata)

    
    xfit = np.linspace(xdata.min(), xdata.max(), 100)
    yfit = slope * xfit + intercept
    
    if p_val<0.05:
        ax.plot(xfit, yfit, color=color, alpha=0.50, ls='--')

    ax.scatter(xdata, ydata, marker=marker,color=color,label=label, alpha=0.5)
    ax.set_xlabel("Delta gas fraction (-)\n (Change of aeration)", size=8)
    ax.set_ylabel("Jacobian (-)",size=8)
    ax.set_title(label, weight='bold')
   
    ax.set_ylim((1.0,1.4))
    ax.set_xlim(0.,0.15)
    
    ax.text(0.01, 1.35, "r=%.2f"%r+pvalue_to_stars(p),ha='left')
    
plt.tight_layout()


# %% Compare simulation/experiment Jacobian and DGF

# Retrieve jacobian information
mesh_j_sim = np.hstack([sdata[s]['mesh']['Simulation']['j'] for s in subjects])
mesh_j_exp = np.hstack([sdata[s]['mesh']['Experiment']['j'] for s in subjects])

# Retrieve delta gas fraction information
mesh_dgf_sim = np.hstack([sdata[s]['mesh']['Simulation']['dgf'] for s in subjects])
mesh_dgf_exp = np.hstack([sdata[s]['mesh']['Experiment']['dgf'] for s in subjects])

xtype = 'image'; xstate='Exp'
xdata = np.hstack([sdata[s][xtype][xstate]["phi_mean"] for s in subjects])


# Scatter plot
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(6,6),dpi=200)

ydatas = [mesh_j_sim ,mesh_j_exp,mesh_dgf_sim, mesh_dgf_exp]
xdatas = [xdata]*4
labels = ["Simulation", "Experiment"]
colors = ["royalblue","navy"]
markers = ['x','+']

for e, (ax,ydata,xdata) in enumerate(zip(axes.flatten(),ydatas,xdatas)):
    
    ydata=np.abs(ydata)
    
    r, p = pearsonr(xdata, ydata)
    slope, intercept, r_val, p_val, stderr = linregress(xdata, ydata)

    
    xfit = np.linspace(xdata.min(), xdata.max(), 100)
    yfit = slope * xfit + intercept
    
    if p_val<0.05:
        ax.plot(xfit, yfit, color=color, alpha=0.50, ls='--')

    ax.scatter(xdata, ydata, marker=marker,color=color,label=label, alpha=0.5)
    ax.set_xlabel("End-expiratory aeration (-)", size=8)
    ax.set_title(label, weight='bold')
    
    ax.set_xlim(0.,1.0)

    if e in [0,1]:
        ax.set_ylim((1.0,1.4))
        ax.set_ylabel('Jacobian (-)',size=10)
    else:
        ax.set_ylim((0.0,0.15))
        ax.set_ylabel("Delta Gas Fraction (-)",size=10)
    
    
    if e%2 != 0:
        ax.set_title('Experiment',weight='bold')
    else:
        ax.set_title('Simulation',weight='bold')
    
    x0,x1 = ax.get_xlim(); dx = x1-x0;     y0,y1 = ax.get_ylim(); dy = y1-y0; 

    ax.text(x0+0.02*dx, y1-0.08*dy, "r=%.2f"%r+pvalue_to_stars(p),ha='left')
    
plt.tight_layout()

# %% Evaluate if jacobian and DGF mismatch are related to the recruitment

# Retrieve jacobian information
mesh_j_sim = np.hstack([sdata[s]['mesh']['Simulation']['j'] for s in subjects])
mesh_j_exp = np.hstack([sdata[s]['mesh']['Experiment']['j'] for s in subjects])
j_mismatch = np.abs(mesh_j_sim - mesh_j_exp)

# Retrieve delta gas fraction information
mesh_dgf_sim = np.hstack([sdata[s]['mesh']['Simulation']['dgf'] for s in subjects])
mesh_dgf_exp = np.hstack([sdata[s]['mesh']['Experiment']['dgf'] for s in subjects])
dgf_mismatch = np.abs(mesh_dgf_sim - mesh_dgf_exp)

# Error in the aeration prediction (Experiment/Simulation - Image )
aeration_image_insp = np.hstack([sdata[s]['image']['Insp']["phi_mean"] for s in subjects])
aeration_mesh_simulation = np.hstack([sdata[s]['mesh']['Insp-Simulation']["phi_mean"] for s in subjects])
aeration_mesh_experiment = np.hstack([sdata[s]['mesh']['Insp-Experiment']["phi_mean"] for s in subjects])
aeration_error_simulation = aeration_image_insp - aeration_mesh_simulation
aeration_error_experiment = aeration_image_insp - aeration_mesh_experiment

# Scatter plot
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(6,4),dpi=200)

ydatas = [j_mismatch ,dgf_mismatch]
xdatas = [image_dnat]*2
labels = ["Jacobian mismatch", "DGF mismatch"]
colors = ["royalblue","navy"]
markers = ['x','+']

for e, (ax,ydata,xdata,label,color,marker) in enumerate(zip(axes,ydatas,xdatas,labels,colors,markers)):
    
    ydata=np.abs(ydata)
    
    r, p = pearsonr(xdata, ydata)
    slope, intercept, r_val, p_val, stderr = linregress(xdata, ydata)

    
    xfit = np.linspace(xdata.min(), xdata.max(), 100)
    yfit = slope * xfit + intercept
    
    if p_val<0.05:
        ax.plot(xfit, yfit, color=color, alpha=0.50, ls='--')

    ax.scatter(xdata, ydata, marker=marker,color=color,label=label, alpha=0.5)
    
    ax.set_xlabel("Recruitment (-)\n (Change of NAT compartment)", size=8)
    
    if e == 0:
        ax.set_ylabel("Jacobian (-)")
        ax.set_ylim((0,0.25))
    else:
        ax.set_ylabel("Delta Gas Fraction mismatch (-)")
        ax.set_ylim((0,0.08))

    ax.set_title(label, weight='bold')

plt.tight_layout()

# %% Delta Gas Fraction Experimental vs Simulation

# Retrieve delta gas fraction information
xdata = np.hstack([sdata[s]['mesh']['Experiment']['dgf'] for s in subjects])
ydata = np.hstack([sdata[s]['mesh']['Simulation']['dgf'] for s in subjects])

# Scatter plot
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)

ydatas = [mesh_j_sim ,mesh_j_exp]
xdatas = [mesh_dgf_sim, mesh_dgf_exp]
labels = ["Simulation", "Experiment"]
colors = ["royalblue","navy"]
markers = ['x','+']

    
r, p = pearsonr(xdata, ydata)
slope, intercept, r_val, p_val, stderr = linregress(xdata, ydata)

    
xfit = np.linspace(xdata.min(), xdata.max(), 100)
yfit = slope * xfit + intercept
    
if p_val<0.05:
    ax.plot(xfit, yfit, color=color, alpha=0.50, ls='--')

ax.scatter(xdata, ydata, marker=marker,color=color,label=label, alpha=0.5)
ax.set_xlabel("Delta gas fraction (-)\n(Experiment)", size=8)
ax.set_ylabel("Delta gas fraction (-)\n(Simulation)", size=8)
ax.set_title("Delta gas fraction agreement", weight='bold')
ticks = np.linspace(0.,0.16,5)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
   # ax.set_ylim((1.0,1.4))
 #   ax.set_xlim(0.,0.15)
 
ax.plot(ticks,ticks, ls='--',color='r',alpha=0.5)
    
ax.text(0.01, 0.145, "r=%.2f"%r+pvalue_to_stars(p),ha='left')
    
plt.tight_layout()

# %% Recruitability and DGF assessment

s = 3
dnat_data = sdata[s]['image']['Exp']['NAT']-sdata[s]['image']['Insp']['NAT']
exp_dgf_data = sdata[s]['mesh']['Experiment']['dgf']
sim_dgf_data = sdata[s]['mesh']['Simulation']['dgf']

phi_data = sdata[s]['image']['Exp']['phi_mean']

roi_id = np.linspace(1,10,10)

fig, ax = plt.subplots()

ax.scatter(exp_dgf_data,roi_id, color='royalblue', marker='+', label="Experiment")
ax.scatter(sim_dgf_data,roi_id, color='navy', marker='x', label="Simulation")
ax.set_xlabel('Delta Gas Fraction (-)')
ax.invert_yaxis()
ax.set_ylabel("ROI ID")
ax.set_xticks(np.linspace(0.,0.20,5))

x0,x1 = ax.get_xlim(); dx = x1-x0
y0,y1 = ax.get_ylim(); dy = y1-y0

ax.set_yticks(roi_id)
ax.text(x0-0.05*dx,y1,'Ventral',size=8,ha='right')
ax.text(x0-0.05*dx,y0,'Dorsal',size=8,ha='right')


ax2 = ax.twiny()

ax2.plot(dnat_data, roi_id, alpha=0.5,color='r', ls='--')
ax2.scatter(dnat_data, roi_id, alpha=0.5,color='r', marker='o')

ax2.set_xlabel("Recruitment index (-)")
ax2.set_xlim(0,0.20)
ax2.set_xticks(np.linspace(0.,0.30,7))
ax.legend()

# %%

s = 3
dnat_data = np.hstack([sdata[s]['image']['Exp']['NAT']-sdata[s]['image']['Insp']['NAT'] for s in subjects])
exp_dgf_data = np.hstack([sdata[s]['mesh']['Experiment']['dgf'] for s in subjects])
sim_dgf_data = np.hstack([sdata[s]['mesh']['Simulation']['dgf'] for s in subjects])

ae_dgf_data = np.abs(sim_dgf_data-exp_dgf_data)

phi_data = sdata[s]['image']['Exp']['phi_mean']

roi_id = np.linspace(1,10,10)

fig, ax = plt.subplots(dpi=250)

ax.scatter(dnat_data,ae_dgf_data, color='royalblue', marker='+')
r, p = pearsonr(dnat_data, ae_dgf_data)
ax.text(0.002, 0.055, "r=%.2f"%r+pvalue_to_stars(p),ha='left')

ax.set_xlabel('Recruitment (-)')
ax.set_ylabel('Absolute error in Delta \n Gas Fraction prediction(-)')

# %% Recruitability and J assessment

s = 3
dnat_data = sdata[s]['mesh']['Exp']['NAT']-sdata[s]['mesh']['Insp-Experiment']['NAT']
exp_dgf_data = sdata[s]['mesh']['Experiment']['j']
sim_dgf_data = sdata[s]['mesh']['Simulation']['j']

phi_data = sdata[s]['image']['Exp']['phi_mean']

roi_id = np.linspace(1,10,10)

fig, ax = plt.subplots()

ax.scatter(exp_dgf_data,roi_id, color='royalblue', marker='+', label="Experiment")
ax.scatter(sim_dgf_data,roi_id, color='navy', marker='x', label="Simulation")
ax.plot(exp_dgf_data,roi_id, color='royalblue', alpha=0.1,)
ax.plot(sim_dgf_data,roi_id, color='navy', alpha=0.1)
ax.set_xlabel('Jacobian (-)')
ax.invert_yaxis()
ax.set_ylabel("ROI ID")
ax.set_xticks(np.linspace(1.0,1.5,6))

x0,x1 = ax.get_xlim(); dx = x1-x0
y0,y1 = ax.get_ylim(); dy = y1-y0

ax.set_yticks(roi_id)
ax.text(x0-0.05*dx,y1,'Ventral',size=8,ha='right')
ax.text(x0-0.05*dx,y0,'Dorsal',size=8,ha='right')


ax2 = ax.twiny()

ax2.plot(dnat_data, roi_id, alpha=0.5,color='r', ls='--')
ax2.scatter(dnat_data, roi_id, alpha=0.5,color='r', marker='o')

ax2.set_xlabel("Recruitment index (-)")
ax2.set_xlim(0,0.20)
ax2.set_xticks(np.linspace(0.,0.30,7))
ax.legend()