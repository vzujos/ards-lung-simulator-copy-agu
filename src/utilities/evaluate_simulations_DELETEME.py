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
                                                                   for kind in tags}

    data_manager.update({'pos':[]})
    for kind in tags:
        data_manager[kind].update({'hist':[],'count':[]})
        
    # Build code
    code = 'PIG%i-ARDSnet'%subject
    # Retrieve time data
    Tsyr = parameters[code]['Tsyr']
    Tpausa = parameters[code]['Tpausa']
    Tinsp = Tsyr+Tpausa
    
    # Path to the signal data
    sim_signals = case_path + "/Signals/"
    
    # Path to wherever the meshes are 
    mesh_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/'%(subject, mesh_type)
    
    # Names for the registration-derived mesh and the simulated mesh
    reg_mesh = mesh_path + 'reg_anim_ready.vtu'
    sim_mesh = case_path +'/post/full_%.12f.vtu'%Tinsp
    
    # Load both meshes
    rmesh = io.read(reg_mesh); smesh = io.read(sim_mesh)
    # Retrieve mass
    mass = np.ones_like(smesh.point_data['Pressure'])
    
    # Analyze regionally
    _, ids = IsoVolumetricSegmentation(directions[direction], mass, smesh.points, nrois)
    
    # Tags
    
    for mesh,kind in zip([smesh,rmesh],tags):
        
        if kind[:3] != 'Exp':
            j = mesh.point_data['Jacobian Forward']
            eiporosity = mesh.point_data['Eulerian Porosity']

        else:
            j = mesh.point_data['Jacobian']
            eiporosity = mesh.point_data['End-Inspiratory Porosity']

        dgf = mesh.point_data['Delta Porosity']
        
        for i in range(nrois):
            
            mask = ids==i
            local_eip = eiporosity[mask]
            local_dgf = dgf[mask]

            local_mass = mass[mask]
            local_j = j[mask]
            
            j_stat = sw.DescrStatsW(data=local_j,weights=local_mass)
            eip_stat = sw.DescrStatsW(data=local_eip,weights=local_mass)
            dgf_stat = sw.DescrStatsW(data=local_dgf,weights=local_mass)
      
            data_manager[kind]['Jacobian']['mean'] += [j_stat.mean]
            data_manager[kind]['Jacobian']['std'] += [j_stat.std]
            data_manager[kind]['EI-porosity']['mean'] += [eip_stat.mean]
            data_manager[kind]['EI-porosity']['std'] += [eip_stat.std]
            data_manager[kind]['DGF']['mean'] += [dgf_stat.mean]
            data_manager[kind]['DGF']['std'] += [dgf_stat.std]
            
            if kind[:3] == 'Sim':
                data_manager['pos'] += [i]

    for kind,path in zip(tags,[sim_mesh,reg_mesh]):
        
        
        if kind[:3] != 'Exp':
            porosity_field = 'Eulerian Porosity'
        else:
            porosity_field = 'End-Inspiratory Porosity'
            
        page = retrieve_regional_histogram(path,direction,
                                            weights=mass, verbose=False,
                                            deformed_state=True,
                                            porosity_field=porosity_field,
                                            nrois=nrois,
                                            displacement_name='u')
        
        data_manager[kind]['hist'] += [page[0]]
        data_manager[kind]['count'] += [page[1]]    

        for i in range(nrois):
            data_manager[kind]['NAT']['mean'] += [page[0][i][0]]
            data_manager[kind]['PAT']['mean'] += [page[0][i][1]]
            data_manager[kind]['AT']['mean'] += [page[0][i][2]]
            data_manager[kind]['HIT']['mean'] += [page[0][i][3]]

    
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
  #  t = signals['time']
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

# %%

# Path to files
root_path = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/'

cases = ['PIG5-mf-bir','PIG5-mf-per',]


subjects = [5,5]

nrois=10
directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}

direction = 'VD'

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




# %%

alpha=0.5

fig, axes = plt.subplots(ncols=1,nrows=3,figsize=(5,10),dpi=300)

for e,field in enumerate(['EI-porosity','Jacobian', 'DGF']):
    
    ax = axes[e]

    for s,subj in enumerate(subjects):
        
        # Porosity is spatially correlated to Ventro-Dorsal position.
        # The further dorsal, the lowest the porosity, which is known.
        
        dm = dms[subj]
        
        mesh_type = mesh_types[subj]
        
        xdata = np.array(dm['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean'])
      #  xdata2 = np.array(dm2['Sim: PIG5/medium-fine'][field]['mean'])
        rxdata = np.array(dm['Exp: %s'%mesh_type][field]['mean'])
        
        ydata = np.array(dm['pos'])
        
        if s == 0:
            tag = "Birzle's"
        else:
            tag = "Current"
        
        ax.scatter(xdata,ydata, label=tag,alpha=alpha)
        
        if s == 1:
            ax.plot(rxdata,ydata,label='Experiment', color='k',ls='--',alpha=0.4)
            ax.scatter(rxdata,ydata, color='k',marker='x')

    
    for s in range(len(subjects)):
        
        yticks = np.array(np.arange(10))
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
        
        if False:
            ax.set_ylabel("")
            ax.set_yticks([])
        else:
            x0,x1 = ax.get_xlim(); dx = x1-x0
    
            xtext = x0-0.25*dx
            ax.text(xtext,0,'Ventral',weight='bold')
            ax.text(xtext,9.5,'Dorsal',weight='bold')
    plt.tight_layout()


# %%

ydata = np.linspace(0,9,10)
labels = ['Simulation', 'Experimental']

fig,axes = plt.subplots(ncols=4,nrows=len(subjects),figsize=(12,9),dpi=300)

renamer = {'NAT':'Non-aerated tissue',
           'AT':'Normally-aerated tissue',
           'PAT':'Poorly-aerated tissue',
           'HIT':'Hyperinflated tissue'}


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

            ax.set_yticks(yticks,yticks+1)
            
            ax.text(-0.34,0.0,'Ventral',weight='bold')
            ax.text(-0.34,8.5,'Dorsal',weight='bold')
            
        else:
            ax.set_yticks([])
            
        ax.set_title(renamer[field])    
        ax.set_xlabel('Fraction (-)')
        ax.set_xticks(np.linspace(0.0,1.0,6))
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        
    ax.legend()
    plt.tight_layout()

# %%


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
        RGB or named colors for regions [Ventral, Medial, Dorsal].
    rename : dict, optional
        Mapping from data field names to display names.
    """

    # --- Extract data ---
    ids = np.array(data_manager['pos'])
    xdata = np.array(data_manager[key_exp][field]['mean'])  # Experimental values
    ydata = np.array(data_manager[key_sim][field]['mean'])  # Simulation values
    nrois = len(np.unique(ids))

    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)

    # =====================================================
    # (1) Scatter plot: Simulation vs Experiment
    # =====================================================
    ax = axes[0]

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
    ax.legend(frameon=False, loc='best', fontsize=10)

    # =====================================================
    # (2) Bland–Altman plot
    # =====================================================
    ax = axes[1]

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
    ax.axhline(mean_diff, color='red', linestyle='-', label=f'Mean diff = {mean_diff:.2f}')
    ax.axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD = {loa_upper:.2f}')
    ax.axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD = {loa_lower:.2f}')

    # Labels & formatting
    ax.set_xlabel('Mean of Experiment and Simulation')
    ax.set_ylabel('Simulation − Experiment')
    ax.set_title(f'Bland–Altman Plot — {rename.get(field, field)}')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Legend for statistical lines
    legend_stats = ax.legend(frameon=False, fontsize=10)

    # Legend for region coloring (placed outside)
    region_handles = [
        Line2D([], [], color=colors[0], marker='o', linestyle='None', label='Ventral'),
        Line2D([], [], color=colors[1], marker='o', linestyle='None', label='Medial'),
        Line2D([], [], color=colors[2], marker='o', linestyle='None', label='Dorsal')
    ]
    legend_regions = ax.legend(handles=region_handles, title='Region',
                               frameon=False, bbox_to_anchor=bbox_to_anchor,
                               fontsize=8)

    ax.add_artist(legend_stats)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # space for legend on right
    plt.show()
    
# %%
fields = ['Jacobian','DGF','NAT','PAT','AT','HIT','EI-porosity',"Raw","Crs","PEEP"]

gdm = {kind:{field:{'mean':[]} for field in fields} for kind in ["Sim","Exp"]}
gdm.update({'pos':[]})

for subj in subjects:
    
    mesh_type = mesh_types[subj]
    
    for field in fields:
        
        gdm['Sim'][field]['mean'] += dms[subj]['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean']
        gdm['Exp'][field]['mean'] += dms[subj]['Exp: %s'%mesh_type][field]['mean']

    gdm['pos'] += dms[subj]['pos']

# %%

for field in ['Jacobian','DGF','NAT','PAT','AT','HIT','EI-porosity']:
    scatter_and_bland_altman(field, gdm,'Exp','Sim',
                         figsize=(12, 4), dpi=200,
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
                             bbox_to_anchor=(1.00,1.0))
    

# %%

rename = {'Raw':"Airway Resistance \n(cmH$_2$O/L/s)",
          'Crs':"Respiratory System Compliance \n(mL/cmH$_2$O)",
          'PEEP':"PEEP \n(cmH$_2$O)"}

bounds = {'Raw':(5,13),
          'PEEP':(9,12),
          'Crs':(20,60)
          }

fig,axes = plt.subplots(ncols=3, figsize=(9,3),dpi=300)

for field,ax in zip(["Raw", "Crs","PEEP"],axes):

    xdata = []
    ydata = [] 
    
    for subj in subjects:
        mesh_type = mesh_types[subj]
        xdata += list(dms[subj]['Exp: %s'%mesh_type][field]['mean'])
        ydata += list(dms[subj]['Sim: PIG%i/%s'%(subj,mesh_type)][field]['mean'])
    
    xdata = np.hstack(xdata); ydata = np.hstack(ydata)
    slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
    ax.scatter(xdata, ydata, marker='x', color='tab:blue')
    
    
    x_fit = np.linspace(min(xdata), max(xdata), 100)
    y_fit = slope * x_fit + intercept
    
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
    
    ax.plot(x_fit, y_fit, color='red', lw=1.5,
            label='r = %.2f '%r_value+text)


    ax.set_ylabel("Simulation")
    ax.set_xlabel("Experiment")
    ax.set_title(rename[field])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

    ax.set_xlim(bounds[field])
    ax.set_ylim(bounds[field])
    
    if field == 'Raw':
        ticks = [5,7,9,11,13]
    elif field == 'Crs':
        ticks = [20,30,40,50]
    else:
        ticks=[9,10,11,12]
        
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.legend()
    x0,x1 = ax.get_xlim(); dx = x1-x0
    y0,y1 = ax.get_ylim(); dy = y1-y0
    ax.plot([x0,x1],[y0,y1], ls='--',alpha=0.3, color='k')
    
    for s,subj in enumerate(subjects):
        ax.text(xdata[s]+dx*0.075,ydata[s],"%i"%subj,ha='center', va='center')
