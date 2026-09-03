# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:56:22 2025

@author: angus
"""

# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import meshio as io
import statsmodels.stats.weightstats as sw
from scipy.stats import linregress
from matplotlib.lines import Line2D

#%config InlineBackend.figure_format='svg'
plt.rcParams['font.family'] = 'Helvetica'

# %% Organizer

organizer = {'sim':{2:'PIG2',
                    3:'PIG3',
                    4:'PIG4-redo',
                    5:'PIG5-ma',
                    6:'PIG6'},
             }


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

def scatter_and_bland_altman(field, data_manager, figsize=(12, 4), dpi=200,
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
    xdata = np.array(data_manager['reg'][field]['mean'])  # Experimental values
    ydata = np.array(data_manager['sim'][field]['mean'])  # Simulation values
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
                               fontsize=8)#,loc='upper left', )

    ax.add_artist(legend_stats)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # space for legend on right
    plt.show()
    
def scatter_and_bland_altman_global(field, data_manager,tags, figsize=(12, 4), dpi=200,
                             colors=['tab:blue', 'tab:orange', 'tab:green'],
                             rename={'Raw':"AW Resistance (cmH2O/L/s)",
                                     'Crs':"RS Compliance (mL/cmH2O)",
                                     'PEEP':"PEEP (cmH2O)"}):
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
    xdata = np.array(data_manager['reg'][field]['mean'])  # Experimental values
    ydata = np.array(data_manager['sim'][field]['mean'])  # Simulation values

    fig, axes = plt.subplots(ncols=2, figsize=figsize, dpi=dpi)

    # =====================================================
    # (1) Scatter plot: Simulation vs Experiment
    # =====================================================
    ax = axes[0]

    # Plot points by region
    ax.scatter(xdata, ydata, alpha=0.5, color=colors[0])

    # Plot y = x reference line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='k', ls='--', alpha=0.25)

    # Linear regression
    if len(xdata)>3:
        slope, intercept, r_value, p_value, _ = linregress(xdata.flatten(), ydata.flatten())
        
        
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


    # Labels & formatting
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Simulation")
    ax.set_title("Scatter Plot — "+rename.get(field, field))
    ax.legend(frameon=False, loc='best', fontsize=8)
    
    y0,y1=ax.get_ylim();dy=y1-y0;ddy=dy/80
    x0,x1=ax.get_xlim();dx=x1-x0;ddx=dx/80
    
    for x,y,text in zip(xdata,ydata,tags):
        ax.text(x+ddx,y+ddy,text)
    
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
    ax.scatter(mean_values, diff_values, alpha=0.5, color=colors[0])

    # Add mean and LoA lines
    ax.axhline(mean_diff, color='red', linestyle='-', label=f'Mean diff = {mean_diff:.2f}')
    ax.axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD = {loa_upper:.2f}')
    ax.axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD = {loa_lower:.2f}')

    # Labels & formatting
    ax.set_xlabel('Mean of Experiment & Simulation')
    ax.set_ylabel('Simulation − Experiment')
    ax.set_title(f'Bland–Altman Plot — {rename.get(field, field)}')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Legend for statistical lines
    legend_stats = ax.legend(frameon=False, fontsize=8)

    ax.add_artist(legend_stats)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # space for legend on right
    plt.show()


# %% Declare paths 

# IDs
subjects = [5]

simulated_path = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/MFSIMS/'
nrois=10
directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}
direction = 'BA'

data_manager = {kind:{field:{'mean':[],'std':[]} for field in ['VS','Jacobian','DGF','NAT','PAT','AT','HIT',
                                                               'PEEP','Crs','Raw','EE-porosity','EI-porosity']} for kind in ['sim','reg']}
data_manager.update({'pos':[]})
for kind in ['sim','reg']:
    data_manager[kind].update({'hist':[],'count':[]})

for subject in subjects:
    
    # % Evaluate simulation length
    
    # Build code
    code = 'PIG%i-ARDSnet'%subject
    # Retrieve time data
    Tsyr = parameters[code]['Tsyr']
    Tpausa = parameters[code]['Tpausa']
    Texp = parameters[code]['Texp']
    
    # Path to the signal data
    sim_signals = simulated_path+organizer['sim'][subject]+"/output/Signals/"
    # Path to the time data
    sim_time_npy = sim_signals + 'effectivetimes.npy'
    sim_time = np.load(sim_time_npy)
    
    # Maximum simulated time
    sim_time_max = sim_time.max()
    
    Tinsp = Tsyr+Tpausa
    
    # Check if the inspiratory pause was captured
    if sim_time_max/Tinsp < 1.0:
        print("Simulation did not reach inspiratory pause")
        insp_pause_check = False
    else:
        print("Inspiratory pause reached!")
        insp_pause_check = True
        
    
    print("Maximum simulated time: %.3f (s)"%sim_time_max)    
    # Check how far onto the simulation we reached
    Tcycle = Tinsp+Texp
    print(" > Progress: %.1f"%(sim_time_max/Tcycle*100)+" %")

    # Path to wherever the meshes are 
    mesh_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%s/ARDSnet/medium-fine/'%subject
    
    # Names for the registration-derived mesh and the simulated mesh
    reg_mesh = mesh_path+'reg_anim_ready.vtu'
    sim_mesh = mesh_path+'sim_anim_ready.vtu'
    
    # Load both meshes
    rmesh = io.read(reg_mesh)
    smesh = io.read(sim_mesh)
    mass = rmesh.point_data['Mass']
    
    # Analyze regionally
    _, ids = IsoVolumetricSegmentation(directions[direction],mass,rmesh.points, nrois)
    
    for mesh,kind in zip([smesh, rmesh],['sim','reg']):
        
        if kind == 'reg':
            j = mesh.point_data['Jacobian']
        elif kind == 'sim':
            j = mesh.point_data['Jacobian Partial']
        else:
            raise Exception('Kind!')
            
        vs = mesh.point_data['VolStrain']
        dgf = mesh.point_data['Delta Porosity']
        mass = mesh.point_data['Mass']
        eeporosity = mesh.point_data['End-Expiratory Porosity']
        eiporosity = mesh.point_data['End-Inspiratory Porosity']
        
        for i in range(nrois):
            
            mask = ids==i
            local_vs = vs[mask]
            local_eep = eeporosity[mask]
            local_eip = eiporosity[mask]
            local_dgf = dgf[mask]

            local_mass = mass[mask]
            local_j = j[mask]
            
            j_stat = sw.DescrStatsW(data=local_j,weights=local_mass)
            vs_stat = sw.DescrStatsW(data=local_vs,weights=local_mass)
            eep_stat = sw.DescrStatsW(data=local_eep,weights=local_mass)
            eip_stat = sw.DescrStatsW(data=local_eip,weights=local_mass)
            dgf_stat = sw.DescrStatsW(data=local_dgf,weights=local_mass)
      
            data_manager[kind]['Jacobian']['mean'] += [j_stat.mean]
            data_manager[kind]['Jacobian']['std'] += [j_stat.std]
            data_manager[kind]['VS']['mean'] += [vs_stat.mean]
            data_manager[kind]['VS']['std'] += [vs_stat.std]
            data_manager[kind]['EE-porosity']['mean'] += [eep_stat.mean]
            data_manager[kind]['EE-porosity']['std'] += [eep_stat.std]
            data_manager[kind]['EI-porosity']['mean'] += [eip_stat.mean]
            data_manager[kind]['EI-porosity']['std'] += [eip_stat.std]
            data_manager[kind]['DGF']['mean'] += [dgf_stat.mean]
            data_manager[kind]['DGF']['std'] += [dgf_stat.std]
            
            if kind == 'sim':
                data_manager['pos'] += [10-i]
                
        
    for kind,path in zip(['reg','sim'],[reg_mesh, sim_mesh]):
        
        page = retrieve_regional_histogram(path,direction,
                                            weights=mass, verbose=False,
                                            deformed_state=True,
                                            porosity_field='End-Inspiratory Porosity',
                                            nrois=nrois,
                                            displacement_name='DispField')
        
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
    
    data_manager['sim']['Raw']['mean'] += [R_rs]
    data_manager['sim']['Crs']['mean'] += [C_rs]
    data_manager['sim']['PEEP']['mean'] += [PEEP]

    # Experimental signals
    exp_signals = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz"%(subject,subject)
    signals = np.load(exp_signals)
    t = signals['time']
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
    
    data_manager['reg']['Raw']['mean'] += [R_rs]
    data_manager['reg']['Crs']['mean'] += [C_rs]
    data_manager['reg']['PEEP']['mean'] += [PEEP]

# %%

#scatter_and_bland_altman('VS',data_manager,dpi=300)
scatter_and_bland_altman('Jacobian',data_manager,dpi=300)

scatter_and_bland_altman('DGF',data_manager,bbox_to_anchor=(1.24,1.0),dpi=300)
# %%
scatter_and_bland_altman('NAT',data_manager)
scatter_and_bland_altman('PAT',data_manager)
scatter_and_bland_altman('AT',data_manager,bbox_to_anchor=(1.24,1.0))
scatter_and_bland_altman('HIT',data_manager)

# %%

scatter_and_bland_altman_global('Raw',data_manager,subjects)
scatter_and_bland_altman_global('Crs',data_manager,subjects)
scatter_and_bland_altman_global('PEEP',data_manager,subjects)

# %%


def pval2txt(p_value):
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
    return text


# %%

# Plotting the error in the predicted volumetric strain against the end-expiratory porosity

# We see that the error in the estimation is associated
# with the local end-expiratory porosity

rxdata = np.array(data_manager['reg']['EE-porosity']['mean'])
rydata = np.array(data_manager['reg']['VS']['mean'])
sxdata = np.array(data_manager['sim']['EE-porosity']['mean'])
sydata = np.array(data_manager['sim']['VS']['mean'])

xdata = (rxdata + sxdata)/2
ydata = (rydata-sydata)


# Linear regression
slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
    
text = pval2txt(p_value)
    

x_fit = np.linspace(min(xdata), max(xdata), 100)
y_fit = slope * x_fit + intercept

fig,ax = plt.subplots(figsize=(4,4),dpi=300)

ax.plot(x_fit, y_fit, color='red', lw=1.5,
            label='r = %.2f '%r_value+text)

ax.scatter(xdata,ydata)
ax.set_ylabel("Volumetric Strain Difference ($VS_{exp}-VS_{sim}$)")
ax.axhline(0.0, color='k',alpha=0.25,ls='--')
ax.set_xlabel("End-expiratory porosity (-)")
ax.legend()
#plt.show()

# %%

rydata = np.array(data_manager['reg']['VS']['mean'])
sydata = np.array(data_manager['sim']['VS']['mean'])
ydata = (rydata-sydata)
xdata = np.array(data_manager['pos'])
# Linear regression
slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
txt = pval2txt(p_value)    




fig, ax = plt.subplots(dpi=300,figsize=(4,4))
x_fit = np.linspace(min(xdata), max(xdata), 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color='red', lw=1.5,
            label='r = %.2f '%r_value+txt)

ax.set_xlabel('ROI ID')
ax.scatter(xdata,ydata)
ax.set_ylabel("Volumetric Strain Difference ($VS_{exp}-VS_{sim}$)")
ax.set_xticks(np.arange(10)+1,np.arange(10)+1)
ax.text(0.5,-15,'Ventral')
ax.text(9.7,-15,'Dorsal')
ax.axhline(0.0, color='k',alpha=0.25,ls='--')

ax.legend()

# %%

# Porosity is spatially correlated to Ventro-Dorsal position.
# The further dorsal, the lowest the porosity, which is known.

rydata = np.array(data_manager['reg']['EE-porosity']['mean'])
sydata = np.array(data_manager['sim']['EE-porosity']['mean'])

xdata = np.array(data_manager['pos'])
ydata = (rydata + sydata)/2

# Linear regression
slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
txt = pval2txt(p_value)    

x_fit = np.linspace(min(xdata), max(xdata), 100)
y_fit = slope * x_fit + intercept

fig, ax = plt.subplots(figsize=(4,4),dpi=300)
ax.plot(x_fit, y_fit, color='red', lw=1.5,
            label='r = %.2f '%r_value+txt)

ax.set_xlabel('ROI ID')
ax.scatter(xdata,ydata)
ax.set_ylabel("End-expiratory porosity (-)")
ax.set_xticks(np.arange(10)+1,np.arange(10)+1)
ax.legend()
ax.text(0.5,0.35,'Ventral')
ax.text(9.7,0.35,'Dorsal')
#plt.show()

# %%

rxdata = np.array(data_manager['reg']['EE-porosity']['mean'])
rydata = np.array(data_manager['reg']['DGF']['mean'])
sxdata = np.array(data_manager['sim']['EE-porosity']['mean'])
sydata = np.array(data_manager['sim']['DGF']['mean'])


fig,axes = plt.subplots(ncols=2,figsize=(8,4),dpi=300)

for ax,(xdata,ydata),title in zip(axes,[(rxdata,rydata),(sxdata,sydata)],['Experiment','Simulation']):
  
    slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
    txt = pval2txt(p_value)    

    x_fit = np.linspace(min(xdata), max(xdata), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='red', lw=1.5,
                label='r = %.2f '%r_value+txt)
   
    ax.scatter(xdata,ydata,alpha=0.50)
    ax.set_title(title)

    ax.set_ylim((-0.025,0.10))
    ax.set_xlim((0.4,0.8))
    ax.legend()
    
    ax.axhline(0.0, ls='--',color='k',alpha=0.2)
    ax.fill_between([0.4,0.8],-0.025,0.0,color='silver',alpha=0.25)
    
    ax.set_xlabel('End-expiratory porosity (-)')
   # ax.set_xticks(np.linspace(0.40,0.80,5))
   # ax.set_xticklabels(["%.1f"%f for f in np.linspace(0.40,0.80,5)])
    
axes[0].set_ylabel('Delta Gas Fraction (-)')
#axes[0].set_yticks(np.linspace(0,40,5))
axes[1].set_yticks([])
plt.tight_layout()


# %%

rxdata = np.array(data_manager['reg']['EE-porosity']['mean'])
rydata = np.array(data_manager['reg']['VS']['mean'])
sxdata = np.array(data_manager['sim']['EE-porosity']['mean'])
sydata = np.array(data_manager['sim']['VS']['mean'])


fig,axes = plt.subplots(ncols=2,figsize=(8,4),dpi=300)

for ax,(xdata,ydata),title in zip(axes,[(rxdata,rydata),(sxdata,sydata)],['Experiment','Simulation']):
  
    slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
    txt = pval2txt(p_value)    

    x_fit = np.linspace(min(xdata), max(xdata), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='red', lw=1.5,
                label='r = %.2f '%r_value+txt)
   
    ax.scatter(xdata,ydata,alpha=0.50)
    ax.set_title(title)

    ax.set_ylim((0,40))
    ax.set_xlim((0.4,0.8))
    ax.legend()
    ax.set_xlabel('End-expiratory porosity (-)')
    ax.set_xticks(np.linspace(0.40,0.80,5))
    ax.set_xticklabels(["%.1f"%f for f in np.linspace(0.40,0.80,5)])
    
axes[0].set_ylabel('Volumetric strain (%)')
axes[0].set_yticks(np.linspace(0,40,5))
axes[1].set_yticks([])
plt.tight_layout()

# %%

rydata = np.array(data_manager['reg']['DGF']['mean'])
rxdata = np.array(data_manager['reg']['VS']['mean'])
sydata = np.array(data_manager['sim']['DGF']['mean'])
sxdata = np.array(data_manager['sim']['VS']['mean'])


fig,axes = plt.subplots(ncols=2,figsize=(8,4),dpi=300)

for ax,(xdata,ydata),title in zip(axes,[(rxdata,rydata),(sxdata,sydata)],['Experiment','Simulation']):
  
    slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
    txt = pval2txt(p_value)    

    x_fit = np.linspace(min(xdata), max(xdata), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='red', lw=1.5,
                label='r = %.2f '%r_value+txt)
   
    ax.scatter(xdata,ydata,alpha=0.50)
    ax.set_title(title)
    ax.axhline(0.0, ls='--',color='k',alpha=0.2)
    ax.fill_between([0,40],-0.025,0.0,color='silver',alpha=0.25)
    ax.set_xlim((0,40))
    ax.set_ylim((-0.025,0.125)) # dgf
    ax.legend()
    ax.set_xlabel('Volumetric Strain (%)')
    ax.set_xticks(np.linspace(0,40,5))
    ax.set_xticklabels(["%.0f"%f for f in np.linspace(40,80,5)])
    
axes[0].set_ylabel('Delta Gas Fraction')
#axes[0].set_yticks
axes[1].set_yticks([])
plt.tight_layout()

# %%

# Plotting the error in the predicted delta gas fraction against the end-expiratory porosity

# We see that the error in the estimation is associated
# with the local end-expiratory porosity

rxdata = np.array(data_manager['reg']['EE-porosity']['mean'])
rydata = np.array(data_manager['reg']['DGF']['mean'])
sxdata = np.array(data_manager['sim']['EE-porosity']['mean'])
sydata = np.array(data_manager['sim']['DGF']['mean'])

xdata = (rxdata + sxdata)/2
ydata = (rydata-sydata)


# Linear regression
slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
    
text = pval2txt(p_value)
    

x_fit = np.linspace(min(xdata), max(xdata), 100)
y_fit = slope * x_fit + intercept

fig,ax = plt.subplots(figsize=(4,4),dpi=300)

ax.plot(x_fit, y_fit, color='red', lw=1.5,
            label='r = %.2f '%r_value+text)

ax.scatter(xdata,ydata)
ax.set_ylabel("Delta Gas Fraction Difference ($\Delta GF_{exp}-\Delta GF_{sim}$)")
ax.axhline(0.0, color='k',alpha=0.25,ls='--')
ax.set_xlabel("End-expiratory porosity (-)")
ax.legend()
#plt.show()

# %%

