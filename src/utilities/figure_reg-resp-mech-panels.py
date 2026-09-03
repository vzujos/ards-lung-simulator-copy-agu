# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:52:34 2025

@author: angus
"""

import os
import matplotlib.pyplot as plt
import meshio as io
import numpy as np
import statsmodels.stats.weightstats as sw
from scipy.stats import linregress
from matplotlib.lines import Line2D

#%config InlineBackend.figure_format='svg'
plt.rcParams['font.family'] = 'Helvetica'

parameters = {"PIG2-ARDSnet":{"Tsyr":0.435,
                           "Tpausa":0.560,
                           "Texp":1.870,
                           "Pplat":21.7,
                           "PEEP":11.0,
                           "Ppeak":26.8,
                           "Signal path":'model2/signals/PIG2-ARDSnet.npz',
                           "Target volume":0.3527,
                           "EELV":1936.33,
                           },
           "PIG3-ARDSnet":{"Tsyr":0.455,
                           "Tpausa":0.435,
                           "Texp":1.100,
                           "Pplat":26.3,
                           "PEEP":11.3 ,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG3-ARDSnet.npz',
                           "Target volume":0.3870,
                           "EELV":1745.23 ,
                           },
           "PIG4-ARDSnet":{"Tsyr":0.48,
                           "Tpausa":0.28,
                           "Texp":1.73,
                           "Pplat":18.5,
                           "PEEP": 10.7,
                           "Ppeak":24.8,
                           "Signal path":'model2/signals/PIG4-ARDSnet.npz',
                           "Target volume":0.411,
                           "EELV":2407.44,
                           },
           "PIG5-ARDSnet":{"Tsyr":0.375,
                           "Tpausa":0.375,
                           "Texp":1.25,
                           "Pplat":20.6,
                           "PEEP": 10.8,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG5-ARDSnet.npz',
                           "Target volume":0.4011,
                           "EELV":2072.99,
                           },
           "PIG6-ARDSnet":{"Tsyr":0.540,
                           "Tpausa":0.265,
                           "Texp":1.340,
                           "Pplat":22.7,
                           "PEEP": 10.7,
                           "Ppeak":25.7,
                           "Signal path":'model2/signals/PIG6-ARDSnet.npz',
                           "Target volume":0.2995,
                           "EELV":2314.06,
                           }
           }

# %% Define some useful functions


def get_farthest_points_from_line(x, y, slope, intercept, N=5):
    """
    Returns the indices of the N points with the largest perpendicular
    (normal) distance to the line y = slope*x + intercept.
    """
    # Normal distance from point (x_i, y_i) to line y = m*x + b
    # Formula: |m*x_i - y_i + b| / sqrt(m^2 + 1)
    distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)

    # Get indices of top-N largest distances
    idx_sorted = np.argsort(distances)[::-1]  # descending
    idx_farthest = idx_sorted[:N]

    return idx_farthest, distances[idx_farthest]

def scatter_and_bland_altman(field, data_manager, key_exp, key_sim, figsize=(12, 4), 
                             dpi=200,
                             colors=['tab:blue', 'tab:orange', 'tab:green'],
                             rename={'VS': 'Volumetric Strain (%)',
                                     'Jacobian':'Jacobian (-)',
                                     'DGF': 'Delta Gas Fraction (-)',
                                     'HIT': 'Hyperinsufflated Tissue (-)',
                                     'AT':'Normally-Aerated Tissue (-)',
                                     'PAT':'Poorly-Aerated Tissue (-)',
                                     'NAT':'Non-Aerated Tissue (-)',
                                     'Raw':"Airway Resistance (cmH2O/L/s)",
                                     'Crs':"Respiratory System Compliance (cmH2O/L)",
                                     'PEEP':"PEEP (cmH2O)"},
                             bbox_to_anchor=(1.00,1.0)):
    """
    Creates side-by-side plots:
      (1) Experiment vs Simulation scatter with linear regression fit.
      (2) Bland–Altman plot showing agreement between Experiment and Simulation.
      
    Parameters
    ----------
    field : str
        Key for the variable to analyze ('VS', 'DGF', etc.).
    data_manager : dict
        Contains simulation and experimental data under:
        data_manager['reg'][field]['mean'], data_manager['sim'][field]['mean'], and data_manager['pos'].
    figsize : tuple, optional
        Figure size (width, height). Default = (12, 4).
    dpi : int, optional
        Figure resolution. Default = 200.
    colors : list, optional
        RGB or named colors for regions [Ventral, Central, Dorsal].
    rename : dict, optional
        Mapping from data field names to display names.
    """

    # --- Extract data ---
    ids = np.array(data_manager['pos'])
    xdata = np.array(data_manager[key_exp][field]['mean'])  # Experimental values
    ydata = np.array(data_manager[key_sim][field]['mean'])  # Simulation values
    nrois = len(np.unique(ids))

    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=dpi,constrained_layout=True)

    # =====================================================
    # (1) Scatter plot: Simulation vs Experiment
    # =====================================================
    ax = axes[1]

    # Plot points by region
    for i in range(nrois):
        mask = ids == i
        if i < 3:
            color = colors[0]
        elif i < 7:
            color = colors[1]
        else:
            color = colors[2]
        ax.scatter(xdata[mask], ydata[mask], alpha=0.5, color=color)

    # Plot y = x reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='k', ls='--', alpha=0.25)

    # Linear regression
    slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
    
    text = "("
    if p_value<0.05: 
        text += "*"
        if p_value<0.01:
            text += "*"
            if p_value<0.001:
                text += "*)"
            else:
                text += ")"   
        else:
            text += ")"   
    else:
        text = ""
    
    
    x_fit = np.linspace(min(xdata), max(xdata), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='red', lw=1.5,
            label='r = %.2f '%r_value+text)

    get_farthest_points_from_line(xdata,ydata,1.0,0.0)

    # Labels & formatting
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Simulation")
    ax.set_title("Scatter Plot — "+rename.get(field, field))
    ax.legend(frameon=False, loc='best', fontsize=fs-1)

    # =====================================================
    # (2) Bland–Altman plot
    # =====================================================
    ax = axes[0]

    # Compute Bland–Altman components
    mean_values = (xdata + ydata) / 2
    diff_values = ydata - xdata
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Plot region-colored points
    for i in range(nrois):
        mask = ids == i
        if i < 3:
            color = colors[0]
        elif i < 7:
            color = colors[1]
        else:
            color = colors[2]
        ax.scatter(mean_values[mask], diff_values[mask], alpha=0.5, color=color)

    # Add mean and LoA lines
    ax.axhline(mean_diff, color='red', linestyle='-', label=f'Mean bias = {mean_diff:.2f}')
    ax.axhline(loa_upper, color='gray', linestyle='--', label='±1.96 SD ')
    ax.axhline(loa_lower, color='gray', linestyle='--')
    
    
    y0, y1 = ax.get_ylim()
    ymax = np.max(np.abs([y0,y1]))
    ax.set_ylim((-ymax,ymax))
    
    # Labels & formatting
    ax.set_xlabel('Mean of Experiment and Simulation')
    ax.set_ylabel('Simulation − Experiment')
    ax.set_title(f'Bland–Altman Plot — {rename.get(field, field)}')
    ax.grid(False, linestyle=':', alpha=0.5)

    # Legend for statistical lines
    legend_stats = ax.legend(
        frameon=False,
        fontsize=9,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0
    )

    # Legend for region coloring (placed outside)
    region_handles = [
        Line2D([], [], color=colors[0], marker='o', linestyle='None', label='Ventral'),
        Line2D([], [], color=colors[1], marker='o', linestyle='None', label='Central'),
        Line2D([], [], color=colors[2], marker='o', linestyle='None', label='Dorsal')
    ]
    
    legend_regions = ax.legend(
        handles=region_handles,
        title='Region',
        frameon=False,
        fontsize=8,
        loc='lower left',
        bbox_to_anchor=(1.02, 0.00),
        borderaxespad=0.0
    )
    ax.add_artist(legend_stats)
    #plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.show()


def pvalue_stars(p_value):
    """
    Returns significance stars in parentheses based on p-value.
    Matches the original nested logic exactly.
    """
    if p_value >= 0.05:
        return ""

    stars = "*"
    if p_value < 0.01:
        stars += "*"
    if p_value < 0.001:
        stars += "*"

    return f"({stars})"

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
                                displacement_name = 'DispField'):
   
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
    dirs = {"BA" : np.mat([0.,0.,1.]).T, # BA tested; Direction checks out 
            "VD" : np.mat([0.,1.,0.]).T, # VD tested; Direction checks out
            "RL" : np.mat([1.,0.,0.]).T}

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


def extract_roi_data(subject, mesh_type, case_path, tag="", nrois=10, ):
    
    tags = ["Sim: PIG%i/%s"%(subject, mesh_type)+tag, "Exp: %s"%mesh_type]

    
    # Create data structure
    data_manager = {kind:{field:{'mean':[],'std':[]} for field in ['VS','Jacobian','DGF','NAT','PAT','AT','HIT',
                                                                   'PEEP','Crs','Raw','EE-porosity','EI-porosity']} 
                                                                   for kind in tags+['Expiration-histogram']}

    data_manager.update({'pos':[]})
    for kind in tags:
        data_manager[kind].update({'hist':[],'count':[]})
        
    # Build code
    code = 'PIG%i-ARDSnet'%subject
    
    # Path to the signal data
    sim_signals = case_path + "/Signals/"
    
    # Path to wherever the meshes are 
    mesh_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/'%(subject, mesh_type)
    
    # Names for the registration-derived mesh and the simulated mesh
    reg_mesh = mesh_path + 'reg_anim_ready.vtu'
   # sim_mesh = case_path +'/post/full_%.12f.vtu'%Tinsp
    sim_mesh = mesh_path + 'sim_anim_ready.vtu'
   
    # Load both meshes
    rmesh = io.read(reg_mesh); smesh = io.read(sim_mesh)
    # Retrieve mass
    mass = smesh.point_data['Mass']
    
    # Analyze regionally
    _, ids = IsoVolumetricSegmentation(directions[direction], mass, smesh.points, nrois)
    
    # Tags
    
    for mesh,kind in zip([smesh,rmesh],tags):
        
        if kind[:3] != 'Exp':
            j = mesh.point_data['Jacobian Forward']
        else:
            j = mesh.point_data['Jacobian']
        
        eiporosity = mesh.point_data['End-Inspiratory Porosity']
        eeporosity = mesh.point_data['End-Expiratory Porosity']
        dgf = mesh.point_data['Delta Porosity']
        
        for i in range(nrois):
            
            # Generate a mask based on ROI id
            mask = ids==i
            # Mask all the fields data to the current ROI only
            local_eip = eiporosity[mask]
            local_eep = eeporosity[mask]
            local_dgf = dgf[mask]
            local_mass = mass[mask]
            local_j = j[mask]
            
            # Determine weighted descriptive statistics
            j_stat = sw.DescrStatsW(data=local_j,weights=local_mass)
            eip_stat = sw.DescrStatsW(data=local_eip,weights=local_mass)
            eep_stat = sw.DescrStatsW(data=local_eep,weights=local_mass)
            dgf_stat = sw.DescrStatsW(data=local_dgf,weights=local_mass)
            
            # Assign the values to the respective data managers
            data_manager[kind]['Jacobian']['mean'] += [j_stat.mean]
            data_manager[kind]['Jacobian']['std'] += [j_stat.std]
            
            data_manager[kind]['EI-porosity']['mean'] += [eip_stat.mean]
            data_manager[kind]['EI-porosity']['std'] += [eip_stat.std]
            
            data_manager[kind]['EE-porosity']['mean'] += [eep_stat.mean]
            data_manager[kind]['EE-porosity']['std'] += [eep_stat.std]
            
            data_manager[kind]['DGF']['mean'] += [dgf_stat.mean]
            data_manager[kind]['DGF']['std'] += [dgf_stat.std]
            
            # Store position once; Arbitrarily use 'Simulation'
            if kind[:3] == 'Sim':
                data_manager['pos'] += [i]
    
    # Now generate aeration histograms based on the inspiratory state
    for kind,path in zip(tags,[sim_mesh,reg_mesh]):
        
        # Select the field
        porosity_field = 'End-Inspiratory Porosity'
        # Apply histogram-retrieving functoin
        page = retrieve_regional_histogram(path,direction,
                                            weights=mass, verbose=False,
                                            deformed_state=True,
                                            porosity_field=porosity_field,
                                            nrois=nrois,
                                            displacement_name='DispField')
        
        # Store the raw data 
        data_manager[kind]['hist'] += [page[0]]
        data_manager[kind]['count'] += [page[1]]    
        
        # Compartamentalize data and store accordingly
        for i in range(nrois):
            data_manager[kind]['NAT']['mean'] += [page[0][i][0]]
            data_manager[kind]['PAT']['mean'] += [page[0][i][1]]
            data_manager[kind]['AT']['mean'] += [page[0][i][2]]
            data_manager[kind]['HIT']['mean'] += [page[0][i][3]]
            
    # Now generate aeration histograms based on the expiratory state
    for kind,path in zip(['Expiration-histogram'],[sim_mesh]):
        
        # Select the field
        porosity_field = 'End-Expiratory Porosity'
        # Apply histogram-retrieving functoin
        page = retrieve_regional_histogram(path,direction,
                                            weights=mass, verbose=False,
                                            deformed_state=False,
                                            porosity_field=porosity_field,
                                            nrois=nrois,
                                            displacement_name='DispField')
        
        # Compartamentalize data and store accordingly
        for i in range(nrois):
            data_manager[kind]['NAT']['mean'] += [page[0][i][0]]
            data_manager[kind]['PAT']['mean'] += [page[0][i][1]]
            data_manager[kind]['AT']['mean'] += [page[0][i][2]]
            data_manager[kind]['HIT']['mean'] += [page[0][i][3]]
    
    # Now determine regression data
    # Load the simulated signals
    t = np.load(sim_signals+"effectivetimes.npy").flatten()
    p = np.load(sim_signals+"presionestodas.npy").flatten()*10.1972
    q = np.load(sim_signals+"fluxes.npy").flatten()
    v = np.load(sim_signals+"volumenes.npy").flatten()
    v -= v[0]
    ones = np.ones_like(v)
    p = p.reshape((p.shape[0],1))
            
    # Compute the single compartment equation from the simulation data
    X = np.vstack([v,q,ones]).T
    Amid = np.linalg.inv(X.T@X)
    A = Amid@X.T@p
            
    # Compute the values for the simulation
    E_rs = A[0]
    C_rs = 1/E_rs*1000 # mL/cmH2O
    R_rs = A[1]
    PEEP = A[2]
            
    data_manager[tags[0]]['Raw']['mean'] += [R_rs]
    data_manager[tags[0]]['Crs']['mean'] += [C_rs]
    data_manager[tags[0]]['PEEP']['mean'] += [PEEP]            
    
    # Experimental signals
    exp_signals = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz"%(subject,subject)
    signals = np.load(exp_signals)
    p = signals['pressure']
    q = -signals['flow']
    v = signals['volume']
    ones = np.ones_like(v)
    p = p.reshape((p.shape[0],1))
    
    X = np.vstack([v,q,ones]).T
    Amid = np.linalg.inv(X.T@X)
    A = Amid@X.T@p
    
    # Compute the values for the simulation
    E_rs = A[0]
    C_rs = 1/E_rs*1000 # mL/cmH2O
    R_rs = A[1]
    PEEP = A[2]
        
    data_manager[tags[1]]['Raw']['mean'] += [R_rs]
    data_manager[tags[1]]['Crs']['mean'] += [C_rs]
    data_manager[tags[1]]['PEEP']['mean'] += [PEEP]

    return data_manager

# %% Datanase construction

# Path to files
root_path = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/'

cases = ['PIG2-mf-per-2','PIG3-mf-per-3','PIG4-m-per','PIG5-m-per','PIG6-mf-per',]


subjects = [2,3,4,5,6]

nrois=10
directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}

direction = 'VD'
alpha=0.5
dms = {}; mesh_types = {}
for subj,case in zip(subjects,cases):
    
    mtype = case.split('-')[1]
    if mtype == 'mf':
        mesh_type = 'medium-fine'
    elif mtype == 'm':
        mesh_type = 'medium'
    
    mesh_types.update({subj:mesh_type})
    # Extract data
    dm = extract_roi_data(subj, mesh_type, root_path+case,  tag="", nrois=10)

    dms.update({subj:dm})

# %% Subjectwise jacobian, end-inspiratory aeration and delta gas fraction

# Not in the paper

if True:
    
    alpha=0.5
    
    fig, axes = plt.subplots(ncols=3,nrows=len(subjects),figsize=(12,9),dpi=300)
    
    for e,field in enumerate(['EI-porosity','Jacobian', 'DGF']):
        
        for s,subj in enumerate(subjects):
            
            ax = axes[s,e]
            # Porosity is spatially correlated to Ventro-Dorsal position.
            # The further dorsal, the lowest the porosity, which is known.
            
            dm = dms[subj]
            
            mesh_type = mesh_types[subj]
            
            xdata = np.array(dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean'])
          #  xdata2 = np.array(dm2['Sim: PIG5/medium-fine'][field]['mean'])
            rxdata = np.array(dm['Exp: %s'%mesh_type][field]['mean'])
            
            ydata = np.array(dm['pos'])
                
            ax.scatter(xdata,ydata, label="Simulation",alpha=alpha)
            ax.plot(rxdata,ydata,label='Experiment', color='k',ls='--',alpha=0.4)
            ax.scatter(rxdata,ydata, color='k',marker='x')
    
        
        for s in range(len(subjects)):
            ax = axes[s,e]
            
            yticks = np.array([0,9])
            ax.invert_yaxis()
            ax.set_yticks(yticks,yticks+1)
    
            if field == 'EI-porosity':
                ax.set_xlabel("End-inspiratory porosity (-)")
                ax.set_xlim((0,1))
                
            elif field == 'Jacobian':
                ax.set_xlabel("Jacobian (-)")
                ax.set_xlim((1.,1.5))
        
            elif field == 'DGF':
                ax.set_xlabel("Delta Gas Fraction (-)")
                ax.set_xlim((0.00,0.15))    
        
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if e>0:
                ax.set_ylabel("")
                ax.set_yticks([])
            else:
                x0,x1 = ax.get_xlim(); dx = x1-x0
        
                xtext = x0-0.25*dx
                ax.text(xtext,0,'Ventral',weight='bold')
                ax.text(xtext,9.5,'Dorsal',weight='bold')
        plt.tight_layout()
    

# %% Subject-wise aeration compartments

renamer = {'NAT':'Non-aerated tissue',
               'AT':'Normally-aerated tissue',
               'PAT':'Poorly-aerated tissue',
               'HIT':'Hyperinflated tissue'}
    
if True:
    
    ydata = np.linspace(0,9,10)
    labels = ['Simulation', 'Experimental']
    
    fig,axes = plt.subplots(ncols=4,nrows=len(subjects),figsize=(12,9),dpi=300)
    

    
    for s, subj in enumerate(subjects):
        
    
        dm = dms[subj]
        mesh_type = mesh_types[subj]
    
        for e,field in enumerate(['NAT','PAT','AT','HIT']):
        
            
            ax = axes[s,e]    
            xdata = np.array(dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean'])
            rxdata = np.array(dm['Exp: %s'%mesh_type][field]['mean'])
        
            for xdata,label in zip([xdata, rxdata],labels):
                
                if label != 'Experimental':
                    ax.scatter(xdata,ydata,alpha=alpha, label=label)
                else:
                    ax.plot(rxdata,ydata,label='Experiment', color='k',ls='--',alpha=0.4)
                    ax.scatter(rxdata,ydata, color='k',marker='x')       
                    
            ax.set_xlim((0.0,1.0))
            ax.set_ylim((-0.2,9.2))
            ax.set_yticklabels(["%i"%(f+1) for f in np.arange(nrois)])
                
            if e in [0]:
                
                ax.set_ylabel('ROI ID')
                yticks = np.array([0,9])
    
                ax.set_yticks(yticks,yticks+1
                              )
                
                ax.text(-0.34,0.0,'Ventral',weight='bold')
                ax.text(-0.34,8.5,'Dorsal',weight='bold')
                
            else:
                ax.set_yticks([])
                
            if s == 0: ax.set_title(renamer[field])    
            ax.set_xlabel('Fraction (-)')
            ax.set_xticks(np.linspace(0.0,1.0,6))
        
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()
            
        ax.legend()
        plt.tight_layout()

# %
# %% Aeration compartments
  
fields = ['Jacobian','DGF','NAT','PAT','AT','HIT','EI-porosity','EE-porosity',"Raw","Crs","PEEP"]

gdm = {kind:{field:{'mean':[]} for field in fields} for kind in ["Sim","Exp"]}
gdm.update({'pos':[]})

for subj in subjects:
    
    mesh_type = mesh_types[subj]
    
    for field in fields:
        
        gdm['Sim'][field]['mean'] += dms[subj]['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean']
        gdm['Exp'][field]['mean'] += dms[subj]['Exp: %s'%mesh_type][field]['mean']

    gdm['pos'] += dms[subj]['pos']

#

rename = {'VS': 'Volumetric Strain (%)',
        'Jacobian':'Jacobian (-)',
        'DGF': 'Delta Gas Fraction (-)',
        'HIT': 'Hyperinsufflated Tissue (-)',
        'AT':'Normally-Aerated Tissue (-)',
        'PAT':'Poorly-Aerated Tissue (-)',
        'NAT':'Non-Aerated Tissue (-)',
        'Raw':"Airway Resistance (cmH2O/L/s)",
        'Crs':"Respiratory System Compliance (cmH2O/L)",
        'PEEP':"PEEP (cmH2O)",
        'EI-porosity':"End-Inspiratory Ventilation (-)"}

keys = {(0,0):'(a)',(0,1):'(b)',
        (1,0):'(c)',(1,1):'(d)',
        (2,0):'(e)',(2,1):'(f)',
        (3,0):'(g)',(3,1):'(h)'}


# Consolidated regional aeration compartments; 
# Bland-Altman plot | Scatter plot Journal figure
if True:

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(3.5*2,2.5*4),dpi=300)
    
    fig.subplots_adjust(
        left=0.10,
        right=0.80,   # leave room for external legends
        bottom=0.08,
        top=0.95,
        wspace=0.35,  # space between columns
        hspace=0.50   # space between rows
    )
    
    
    
    key_exp='Exp'; key_sim='Sim'; data_manager = gdm
    
    colors = ['slategrey','royalblue','midnightblue']; markers = ['.','+','x']
    ids = np.array(data_manager['pos'])
    nrois = len(np.unique(ids))
    fs=11  # Font size
    
    ticks = np.linspace(0,1,6)
    ticklabels = ["%.1f"%t for t in ticks]
    
    for ai,field in enumerate(['NAT','PAT','AT','HIT']):
    
        # --- Extract data ---
        xdata = np.array(data_manager[key_exp][field]['mean'])  # Experimental values
        ydata = np.array(data_manager[key_sim][field]['mean'])  # Simulation values
    
        # =====================================================
        # (1) Scatter plot: Simulation vs Experiment
        # =====================================================
        ax = axes[ai,1]
    
        # Plot points by region
        for i in range(nrois):
            mask = ids == i
            if i < 3:
                color = colors[0]; marker = markers[0]
            elif i < 7:
                color = colors[1]; marker = markers[1]
            else:
                color = colors[2]; marker = markers[2]
            ax.scatter(xdata[mask], ydata[mask], alpha=0.75, color=color,marker=marker)
    
        # Plot y = x reference line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, color='k', ls='--', alpha=0.25)
    
        # Linear regression
        slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
        
        # Generate a text that symbolizes p-value
        text = pvalue_stars(p_value)
        
        # Generate ticks
        ax.set_xlim((0,1)); ax.set_ylim((0,1))
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels,size=fs-1)
        ax.set_yticks(ticks); ax.set_yticklabels(ticklabels,size=fs-1)
    
        
        x_fit = np.linspace(min(xdata), max(xdata), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='k', lw=1.5,label='r = %.2f '%r_value+text)
    
        get_farthest_points_from_line(xdata,ydata,1.0,0.0)
    
        # Labels & formatting
        ax.set_xlabel("Experiment",size=fs-1)
        ax.set_ylabel("Simulation",size=fs-1)
        ax.legend(frameon=False, loc='best', fontsize=fs-2)
        ax.set_title(renamer[field],size=fs,weight='bold')
    
        # Regression legend
        legend_reg = ax.legend(
            frameon=False,
            loc='upper left',
            fontsize=fs-2
        )
        
        # Region legend
        region_handles = [
            Line2D([], [], color=colors[0], marker=markers[0], linestyle='None', label='Ventral'),
            Line2D([], [], color=colors[1], marker=markers[1], linestyle='None', label='Central'),
            Line2D([], [], color=colors[2], marker=markers[2], linestyle='None', label='Dorsal')
        ]
        
        legend_regions = ax.legend(
            handles=region_handles,
            title='Region',
            frameon=False,
            fontsize=fs-2,
            loc='lower right'
        )
        
        # Re-add regression legend
        ax.add_artist(legend_reg)
    
        # =====================================================
        # (2) Bland–Altman plot
        # =====================================================
        ax = axes[ai,0]
    
        # Compute Bland–Altman components
        mean_values = (xdata + ydata) / 2
        diff_values = ydata - xdata
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
    
        # Plot region-colored points
        for i in range(nrois):
            mask = ids == i
            if i < 3:
                color = colors[0]; marker = markers[0]
            elif i < 7:
                color = colors[1]; marker = markers[1]
            else:
                color = colors[2]; marker = markers[2]
            
            ax.scatter(mean_values[mask], diff_values[mask], 
                       alpha=0.85, color=color, marker=marker)
    
        # Add mean and LoA lines
        ax.axhline(mean_diff, color='k', linestyle='-', label=f'Mean bias = {mean_diff:.2f}')
        ax.axhline(loa_upper, color='gray', linestyle='--')
        ax.axhline(loa_lower, color='gray', linestyle='--')
        
        # Ticks and similar
        ax.set_ylim((-0.30,0.20)); ax.set_xlim((0.0,1.0))
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels,size=fs-1)
        yticks = np.linspace(-0.3,0.2,6); yticklabels = ["%.1f"%t for t in yticks]
        ax.set_yticks(yticks); ax.set_yticklabels(yticklabels,size=fs-1)
        
        # Labels & formatting
        ax.set_xlabel('Mean of Experiment and Simulation',size=fs-1)
        ax.set_ylabel('Simulation − Experiment',size=fs-1)
        ax.set_title(renamer[field],size=fs,weight='bold')
       # ax.grid(False, linestyle=':', alpha=0.5)
    
        # Legend for statistical lines
        legend_stats = ax.legend(
            frameon=False,
            fontsize=fs-2,
            loc='lower right',
        )    
        ax.add_artist(legend_stats)
    
    for i in range(4):
        for j in range(2):
            ax = axes[i,j]
            x0,x1 = ax.get_xlim(); dx=x1-x0
            y0,y1 = ax.get_ylim(); dy=y1-y0
            ax.text(x0-0.03*dx,y1+0.05*dy,keys[(i,j)], size=fs+2,weight='bold')
        
    plt.savefig('./figures/reg-aer-comp-panel.pdf',dpi=300, bbox_inches='tight')



# %% EI gas fraction and Delta Gas Fraction for the group values

# In paper! 

if True:

    renamer = {'DGF':'Delta gas fraction',
               'EI-porosity':'End-inspiratory\ngas fraction'}
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.5*2,2.5*2),dpi=300)
    
    fig.subplots_adjust(
        left=0.10,
        right=0.80,   # leave room for external legends
        bottom=0.08,
        top=0.95,
        wspace=0.35,  # space between columns
        hspace=0.50   # space between rows
    )
    
    
    key_exp='Exp'; key_sim='Sim'; data_manager = gdm
    
    colors = ['slategrey','royalblue','midnightblue']; markers = ['.','+','x']
    ids = np.array(data_manager['pos'])
    nrois = len(np.unique(ids))
    fs = 11 # Font size
    
    ticks = np.linspace(0,1,6)
    ticklabels = ["%.1f"%t for t in ticks]
    
    for ai,field in enumerate(['EI-porosity','DGF']):
    
        # --- Extract data ---
        xdata = np.array(data_manager[key_exp][field]['mean'])  # Experimental values
        ydata = np.array(data_manager[key_sim][field]['mean'])  # Simulation values
    
        # =====================================================
        # (1) Scatter plot: Simulation vs Experiment
        # =====================================================
        ax = axes[ai,1]
    
        # Plot points by region
        for i in range(nrois):
            mask = ids == i
            if i < 3:
                color = colors[0]; marker = markers[0]
            elif i < 7:
                color = colors[1]; marker = markers[1]
            else:
                color = colors[2]; marker = markers[2]
            ax.scatter(xdata[mask], ydata[mask], alpha=0.75, color=color,marker=marker)
    
        # Plot y = x reference line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, color='k', ls='--', alpha=0.25)
    
        # Linear regression
        slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
        
        # Generate a text that symbolizes p-value
        text = pvalue_stars(p_value)
        
        # Generate ticks
        if field == 'EI-porosity':
            ax.set_xlim((0,1)); ax.set_ylim((0,1))
            ax.set_xticks(ticks); ax.set_xticklabels(ticklabels,size=fs-1)
            ax.set_yticks(ticks); ax.set_yticklabels(ticklabels,size=fs-1)
        else:
            ax.set_xlim((0,0.20)); ax.set_ylim((0,0.20))
            ticks = np.linspace(0,0.20,5); ticklabels = ["%.2f"%t for t in ticks]
            ax.set_xticks(ticks); ax.set_xticklabels(ticklabels,size=fs-1)
            ax.set_yticks(ticks); ax.set_yticklabels(ticklabels,size=fs-1)
            
        x_fit = np.linspace(min(xdata), max(xdata), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='k', lw=1.5,label='r = %.2f '%r_value+text)
    
        get_farthest_points_from_line(xdata,ydata,1.0,0.0)
    
        # Labels & formatting
        ax.set_xlabel("Experiment",size=fs-1)
        ax.set_ylabel("Simulation",size=fs-1)
        ax.legend(frameon=False, loc='best', fontsize=fs-1)
        ax.set_title(renamer[field],size=fs,weight='bold')
    
        # Regression legend
        legend_reg = ax.legend(
            frameon=False,
            loc='upper left',
            fontsize=fs-2
        )
        
        # Region legend
        region_handles = [
            Line2D([], [], color=colors[0], marker=markers[0], linestyle='None', label='Ventral'),
            Line2D([], [], color=colors[1], marker=markers[1], linestyle='None', label='Central'),
            Line2D([], [], color=colors[2], marker=markers[2], linestyle='None', label='Dorsal')
        ]
        
        legend_regions = ax.legend(
            handles=region_handles,
            title='Region',
            frameon=False,
            fontsize=8,
            loc='lower right'
        )
        
        # Re-add regression legend
        ax.add_artist(legend_reg)
    
        # =====================================================
        # (2) Bland–Altman plot
        # =====================================================
        ax = axes[ai,0]
    
        # Compute Bland–Altman components
        mean_values = (xdata + ydata) / 2
        diff_values = ydata - xdata
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
    
        # Plot region-colored points
        for i in range(nrois):
            mask = ids == i
            if i < 3:
                color = colors[0]; marker = markers[0]
            elif i < 7:
                color = colors[1]; marker = markers[1]
            else:
                color = colors[2]; marker = markers[2]
            
            ax.scatter(mean_values[mask], diff_values[mask], 
                       alpha=0.75, color=color, marker=marker)
    
        # Add mean and LoA lines
        ax.axhline(mean_diff, color='k', linestyle='-', label=f'Mean bias = {mean_diff:.2f}')
        ax.axhline(loa_upper, color='gray', linestyle='--')
        ax.axhline(loa_lower, color='gray', linestyle='--')
        
        # Ticks and similar
        if field == 'EI-porosity':
            ax.set_ylim((-0.08,0.08)); ax.set_xlim((0.0,1.0))
            ax.set_xticks(ticks); ax.set_xticklabels(ticklabels,size=fs-1)
            yticks = np.linspace(-0.08,0.08,5); yticklabels = ["%.2f"%t for t in yticks]
            ax.set_yticks(yticks); ax.set_yticklabels(yticklabels,size=fs-1)
        else:
            ax.set_ylim((-0.05,0.05)); ax.set_xlim((0.0,0.20))
            ax.set_xticks(np.linspace(0.,0.20,5)); 
            ax.set_xticklabels(["%.2f"%t for t in np.linspace(0.,0.20,5)],size=fs-1)
            yticks = np.linspace(-0.05,0.05,5); yticklabels = ["%.2f"%t for t in yticks]
            ax.set_yticks(yticks); ax.set_yticklabels(yticklabels,size=fs-1)
        
        # Labels & formatting
        ax.set_xlabel('Mean of Experiment and Simulation',size=fs-1)
        ax.set_ylabel('Simulation − Experiment',size=fs-1)
        ax.set_title(renamer[field],size=fs,weight='bold')
#        ax.grid(False, linestyle=':', alpha=0.5)
    
        # Legend for statistical lines
        legend_stats = ax.legend(
            frameon=False,
            fontsize=fs-2,
            loc='upper right',
        )    
        ax.add_artist(legend_stats)
    
    for i in range(2):
        for j in range(2):
            ax = axes[i,j]
            x0,x1 = ax.get_xlim(); dx=x1-x0
            y0,y1 = ax.get_ylim(); dy=y1-y0
            #ax.text(x0-0.1*dx,y1+0.08*dy,keys[(i,j)], size=fs+2,weight='bold')
            ax.text(x0-0.035*dx,y1+0.045*dy,keys[(i,j)], size=fs+2,weight='bold')

    plt.savefig('./figures/reg-ventilation-panel.pdf',dpi=300, bbox_inches='tight')

# %% 

if True:
    
    
    subj = 4
    alpha=0.5; fs = 11
    
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(3,3*2),dpi=300)
    
    for e,field in enumerate(['EI-porosity','DGF']):
        
        for s,subj in enumerate([subj]):
            
            ax = axes[e]
            # Porosity is spatially correlated to Ventro-Dorsal position.
            # The further dorsal, the lowest the porosity, which is known.
            
            dm = dms[subj]
            
            mesh_type = mesh_types[subj]
            
            xdata = np.array(dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean'])
            rxdata = np.array(dm['Exp: %s'%mesh_type][field]['mean'])
            
            ydata = np.array(dm['pos'])
                
            ax.scatter(xdata,ydata, label="Simulation",alpha=alpha, color='royalblue')
            ax.scatter(rxdata,ydata, color='k',marker='x',label='Experiment')
            
            
            if field == 'EI-porosity' and False:
                xdata_ee_sim = np.array(dm['Sim: PIG%i/%s'%(subj,mesh_type)]['EE-porosity']['mean'])
                xdata_ee_exp = np.array(dm['Exp: %s'%mesh_type]['EE-porosity']['mean'])

                ax.scatter(xdata_ee_sim,ydata, label="EE Simulation",alpha=alpha,color='r',marker='+')
                ax.scatter(xdata_ee_exp,ydata, label="EE Experiment",alpha=alpha,color='g',marker='+')

        
        for s in range(len([5])):
            ax = axes[e]
            
            yticks = np.linspace(0,9,10)
            yticklabels = ["%i"%t for t in yticks+1]
            ax.invert_yaxis()
            ax.set_yticks(yticks,yticklabels, size=fs-1)
    
            if field == 'EI-porosity':
                ax.set_xlabel("End-inspiratory gas fraction (-)",size=fs-1)
                ax.set_xlim((0,1))
                ax.set_xticks(np.linspace(0,1,6),["%.1f"%t for t in np.linspace(0,1,6)],size=fs-1)
            elif field == 'DGF':
                ax.set_xlabel("Delta gas fraction (-)",size=fs-1)
                ax.set_xlim((0.00,0.10))   
                ax.set_xticks(np.linspace(0,0.1,6),
                              ["%.2f"%t for t in np.linspace(0,0.1,6)],size=fs-1)
            
            if field=='EI-porosity':
                ax.legend(frameon=False, fontsize=fs-1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if e==0 or True:
                x0,x1 = ax.get_xlim(); dx = x1-x0
                xtext = x0-0.25*dx
                ax.text(xtext,0.5,'V',weight='bold')
                ax.text(xtext,9.0,'D',weight='bold')
            
            ax.set_ylabel("ROI ID", size=fs-1)
    
        
    for txt,ax in zip(["(b)","(c)"],axes):
        x0,x1=ax.get_xlim();dx=x1-x0
        y0,y1=ax.get_ylim();dy=y1-y0
        ax.text(x0-0.15*dx,y1+0.08*dy,txt,size=fs+2,weight='bold')
    
    plt.tight_layout()
    plt.savefig('./figures/representative-subject.pdf',dpi=300, bbox_inches='tight')
    
# %% Global aeration histogram

subjects = [2,3,4,5,6]

nrois=1
directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}

direction = 'VD'
alpha=0.5
dms = {}; mesh_types = {}
for subj,case in zip(subjects,cases):
    
    mtype = case.split('-')[1]
    if mtype == 'mf':
        mesh_type = 'medium-fine'
    elif mtype == 'm':
        mesh_type = 'medium'
    
    mesh_types.update({subj:mesh_type})
    # Extract data
    dm = extract_roi_data(subj, mesh_type, root_path+case,  tag="", nrois=nrois)

    dms.update({subj:dm})
    

# %%
# global aeration compartments

if False:
    
    dynamic_limits=True
        
    fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(6,6),dpi=300)
    
    renamer = {'NAT':'Non-aerated tissue',
               'AT':'Normally-aerated tissue',
               'PAT':'Poorly-aerated tissue',
               'HIT':'Hyperinflated tissue'}
    
    axes=axes.flatten()
    
    for e,field in enumerate(['NAT','PAT','AT','HIT']):
        
        # Select the axis for the aeration compartment
        ax = axes[e]   
        # Create empty arrays to manage data
        sim_xdata = []; exp_xdata = []
            
        for s, subj in enumerate(subjects):
            
            # Point towards subject
            dm = dms[subj]; mesh_type = mesh_types[subj]
    
            # Build up data array
            sim_xdata += [dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean']]
            exp_xdata += [dm['Exp: %s'%mesh_type][field]['mean']]
        
                
        # Scatter plot for xdata and ydata
        xdata = np.array(exp_xdata).flatten(); ydata = np.array(sim_xdata).flatten()
        ax.scatter(xdata,ydata,alpha=alpha, marker='x', color='royalblue')
        
        # Fix the ranges
        if dynamic_limits:
            x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
            z0 = np.floor(min([x0, y0])/0.1)*0.1 
            z1 = np.ceil(max([x1,y1])/0.1)*0.1
            ax.set_xlim((z0,z1)); ax.set_ylim((z0,z1))
            zs = np.linspace(z0,z1)

        else:
            ax.set_xlim((0.0,1.0)); ax.set_ylim((0.0,1.0))
            zs = np.linspace(0.0,1.0)
            
        if e == 2:
            ticks= np.linspace(z0,z1,num=4)
        else:
            ticks = np.linspace(z0,z1,num=3)
        
        ax.set_xticks(ticks);        ax.set_yticks(ticks)

        # Scatter plot
        slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
        x_fit = np.linspace(min(xdata), max(xdata), 100)
        y_fit = slope * x_fit + intercept
    
        # Place text for p-value
        text = "("
        if p_value<0.05: 
            text += "*"
            if p_value<0.01:
                text += "*"
                if p_value<0.001:
                    text += "*)"
                else:
                    text += ")"   
            else:
                text += ")"   
        else:
            text = ""
    
        ax.plot(x_fit, y_fit, color='k', lw=1.5,
                        label='r = %.2f ' % r_value + text)


        ax.plot(zs,zs,color='gray',alpha=0.5,ls='--',label='Identity')

        #if e in [0,2]:
        ax.set_ylabel('Simulation\nFraction (-)')
        #else:
         #   ax.set_yticks([])
                
        ax.set_title(renamer[field], weight='bold')   
        ax.set_xlabel('Experiment\nFraction (-)')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.invert_yaxis()
            
        ax.legend()
        
    for ax,txt in zip(axes.flatten(),['(a)','(b)','(c)','(d)','(e)','(f)']):
        x0,x1=ax.get_xlim();dx=x1-x0
        y0,y1=ax.get_ylim();dy=y1-y0;
        ax.text(x0-0.150*dx,y1+0.075*dy,txt,size=12,weight='bold')
            
    
    plt.tight_layout()

    plt.savefig('./figures/PLUG-global-aeration-hist.pdf',dpi=300, bbox_inches='tight')
    
# %
# %% Aeration compartments
  

if False:
    
    dynamic_limits=True
        
    fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(6,6),dpi=300)
    
    renamer = {'NAT':'Non-aerated tissue',
               'AT':'Normally-aerated tissue',
               'PAT':'Poorly-aerated tissue',
               'HIT':'Hyperinflated tissue'}
    
    axes=axes.flatten()
    
    for e,field in enumerate(['NAT','PAT','AT','HIT']):
        
        # Select the axis for the aeration compartment
        ax = axes[e]   
        # Create empty arrays to manage data
        sim_xdata = []; exp_xdata = []
            
        for s, subj in enumerate(subjects):
            
            # Point towards subject
            dm = dms[subj]; mesh_type = mesh_types[subj]
    
            # Build up data array
            sim_xdata += [dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean']]
            exp_xdata += [dm['Exp: %s'%mesh_type][field]['mean']]
        
                
        # Scatter plot for xdata and ydata
        xdata = np.array(exp_xdata).flatten(); ydata = np.array(sim_xdata).flatten()
        ax.scatter(xdata,ydata,alpha=alpha, marker='x', color='royalblue')
        
        # Fix the ranges
        if dynamic_limits:
            x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
            z0 = np.floor(min([x0, y0])/0.1)*0.1 
            z1 = np.ceil(max([x1,y1])/0.1)*0.1
            ax.set_xlim((z0,z1)); ax.set_ylim((z0,z1))
            zs = np.linspace(z0,z1)

        else:
            ax.set_xlim((0.0,1.0)); ax.set_ylim((0.0,1.0))
            zs = np.linspace(0.0,1.0)
            
        if e == 2:
            ticks= np.linspace(z0,z1,num=4)
        else:
            ticks = np.linspace(z0,z1,num=3)
        
        ax.set_xticks(ticks);        ax.set_yticks(ticks)

        # Scatter plot
        slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
        x_fit = np.linspace(min(xdata), max(xdata), 100)
        y_fit = slope * x_fit + intercept
    
        # Place text for p-value
        text = "("
        if p_value<0.05: 
            text += "*"
            if p_value<0.01:
                text += "*"
                if p_value<0.001:
                    text += "*)"
                else:
                    text += ")"   
            else:
                text += ")"   
        else:
            text = ""
    
        ax.plot(x_fit, y_fit, color='k', lw=1.5,
                        label='r = %.2f ' % r_value + text)


        ax.plot(zs,zs,color='gray',alpha=0.5,ls='--',label='Identity')

        #if e in [0,2]:
        ax.set_ylabel('Simulation\nFraction (-)')
        #else:
         #   ax.set_yticks([])
                
        ax.set_title(renamer[field], weight='bold')   
        ax.set_xlabel('Experiment\nFraction (-)')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.invert_yaxis()
            
        ax.legend()
        
    for ax,txt in zip(axes.flatten(),['(a)','(b)','(c)','(d)','(e)','(f)']):
        x0,x1=ax.get_xlim();dx=x1-x0
        y0,y1=ax.get_ylim();dy=y1-y0;
        ax.text(x0-0.150*dx,y1+0.075*dy,txt,size=12,weight='bold')
            
    
    plt.tight_layout()

    plt.savefig('./figures/PLUG-global-aeration-hist.pdf',dpi=300, bbox_inches='tight')
    
# %% BOXPLOT
import matplotlib.patches as mpatches

if False:
    
        
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(5,4),dpi=300)
    
    renamer = {'NAT':'Non-aerated tissue',
               'AT':'Normally-aerated tissue',
               'PAT':'Poorly-aerated tissue',
               'HIT':'Hyperinflated tissue'}

    def summarize(arr):
        """Return median and IQR (p25, p75)."""
        arr = np.array(arr)
        med = np.median(arr)
        q25 = np.percentile(arr,25)
        q75 = np.percentile(arr,75)
        return med, q25, q75
    
    lw=1.5
    bar_alpha=0.5
    colors = {0:'gray',
              1:'royalblue',
              2:'navy'}
    
    legends = {0:'(Exp.) Baseline',1:"(Insp.) Experiment",2:"(Insp.) Simulation"}    
    
    for e,field in enumerate(['NAT','PAT','AT','HIT']):
        
        # Create empty arrays to manage data
        sim_xdata = []; exp_xdata = []; baseline_xdata = []
            
        for s, subj in enumerate(subjects):
            
            # Point towards subject
            dm = dms[subj]; mesh_type = mesh_types[subj]
    
            # Build up data array
            baseline_xdata += [dm['Expiration-histogram'][field]['mean']]
            sim_xdata += [dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean']]
            exp_xdata += [dm['Exp: %s'%mesh_type][field]['mean']]
        
                
        # Scatter plot for xdata and ydata
        data_e = np.array(exp_xdata).flatten(); data_s = np.array(sim_xdata).flatten()
        data_b = np.array(baseline_xdata).flatten()

        for i,data in enumerate([data_b,data_e, data_s]):
            med,q25,q75 = summarize(data)
            pos = e+(i-1)*0.20
            xerr = np.array((med-q25,q75-med)).reshape((2,1))
            box = ax.boxplot(data,positions=[pos],vert=False,showfliers=False, patch_artist=True)
            box['medians'][0].set_color('k')
            box['medians'][0].set_alpha(1.0)
            box['medians'][0].set_linewidth(lw)
            box['boxes'][0].set_alpha(bar_alpha)
            box['boxes'][0].set_color(colors[i])
            box['whiskers'][0].set_linewidth(lw)
            box['whiskers'][1].set_linewidth(lw)
            box['caps'][0].set_linewidth(lw)
            box['caps'][1].set_linewidth(lw)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.invert_yaxis()
            

    # Legend handles
    legend_handles = [
        mpatches.Patch(color=colors[0], alpha=bar_alpha, label=legends[0]),
        mpatches.Patch(color=colors[1], alpha=bar_alpha, label=legends[1]),
        mpatches.Patch(color=colors[2], alpha=bar_alpha, label=legends[2])
    ]
    
    ax.legend(handles=legend_handles, loc='best',fontsize=8)
        
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Non-aerated\ntissue',
                        'Poorly aerated\ntissue',
                        'Normally aerated\ntissue',
                        'Hyperinflated\ntissue'])
    ax.set_title('Global aeration compartments',weight='bold')
    ax.set_xlabel('Fraction (-)')
    ax.set_xlim((0,1))
    plt.tight_layout()
    
    plt.savefig('./figures/PLUG-global-aeration-boxplot.pdf',dpi=300, bbox_inches='tight')


# %% Single subject stacked bars
import matplotlib.patches as mpatches

if False:
    
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(5,3),dpi=300)
    
    renamer = {'NAT':'Non-aerated tissue',
               'AT':'Normally-aerated tissue',
               'PAT':'Poorly-aerated tissue',
               'HIT':'Hyperinflated tissue'}

    def summarize(arr):
        """Return median and IQR (p25, p75)."""
        arr = np.array(arr)
        med = np.median(arr)
        q25 = np.percentile(arr,25)
        q75 = np.percentile(arr,75)
        return med, q25, q75
    
    lw=1.5
    bar_alpha=0.5
    colors = {0:'gray',
              1:'royalblue',
              2:'navy'}
    
    legends = {0:'(Exp.)\nBaseline',1:"(Insp.)\nExperiment",2:"(Insp.)\nSimulation"}    
    
    sim_xdata = []; exp_xdata = []; baseline_xdata = []
    
    width = 0.1
    dx = 0.15
    
    for e,field in enumerate(['NAT','PAT','AT','HIT']):
                    
        for s, subj in enumerate([5]):
            
            # Point towards subject
            dm = dms[subj]; mesh_type = mesh_types[subj]
    
            # Build up data array
            baseline_xdata += [dm['Expiration-histogram'][field]['mean']]
            sim_xdata += [dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean']]
            exp_xdata += [dm['Exp: %s'%mesh_type][field]['mean']]
        
    for e,binner in enumerate([baseline_xdata, sim_xdata, exp_xdata]):
        binner = np.array(binner).flatten()
        ax.barh((e-1)*dx, binner[0],height=width, color = "w",edgecolor='k') 
        ax.barh((e-1)*dx, binner[1],height=width,left=binner[0], color = "k", alpha = 0.25, edgecolor="k")
        ax.barh((e-1)*dx, binner[2],height=width,left=(binner[0]+binner[1]), color="k", alpha = 0.5, edgecolor="k")
        ax.barh((e-1)*dx, binner[3],height=width,left=(binner[0]+binner[1]+binner[2]), color = "k", edgecolor="k", alpha = 0.999)
        h1= -0.20+dx*e; h2=-0.10+dx*e
        ax.plot([0,1],[h2,h2],color='k',lw=1.0)
        ax.plot([0,1],[h1,h1],color='k',lw=1.0)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_yticks([-0.15, 0.0, 0.15])
    ax.set_yticklabels([legends[i] for i in range(3)])

    ax.set_title('Global aeration compartments',weight='bold')
    ax.set_xlabel('Fraction (-)')

    # Legend handles
    legend_handles = [
        mpatches.Patch(facecolor='w',edgecolor='k', alpha=1.0, label='NAT'),
        mpatches.Patch(color='k', alpha=0.25, label='PAT'),
        mpatches.Patch(color='k', alpha=0.5, label='AT'),
        mpatches.Patch(color='k', alpha=0.99, label='HIT')
    ]
    
    ax.legend(handles=legend_handles, loc='upper right',fontsize=8,
    bbox_to_anchor=(1.2, 1.0),  # x > 1 moves it outside right
    frameon=True
)
    
    plt.tight_layout()
    
    plt.savefig('./figures/PLUG-global-aeration-horbars.pdf',dpi=300, bbox_inches='tight')

