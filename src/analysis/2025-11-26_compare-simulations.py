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



# Path to files
root_path = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/'

cases = ['PIG5-mc-mediastinum-9']


subjects = [5]

nrois=10
directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}

direction = 'VD'
kinds = ['Constant','Variable']

data_manager = {kind:{field:{'mean':[],'std':[]} for field in ['VS','Jacobian','DGF','NAT','PAT','AT','HIT',
                                                               'PEEP','Crs','Raw','EE-porosity','EI-porosity']} 
                                                               for kind in (kinds+['reg'])}


data_manager.update({'pos':[]})
for kind in (kinds+['reg']):
    data_manager[kind].update({'hist':[],'count':[]})

for subject in subjects:
        
    # Build code
    code = 'PIG%i-ARDSnet'%subject
    # Retrieve time data
    Tsyr = parameters[code]['Tsyr']
    Tpausa = parameters[code]['Tpausa']
    Texp = parameters[code]['Texp']
    
    # Path to the signal data
    sim_signals_C = root_path + cases[0]+"/Signals/"
    sim_signals_V = root_path + cases[1]+"/Signals/"
    

    # Path to wherever the meshes are 
    mesh_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/medium-coarse/'
    
    # Names for the registration-derived mesh and the simulated mesh
    reg_mesh = mesh_path+'reg_anim_ready.vtu'
    sim_mesh_C = root_path + cases[0]+'/post/full_0.750000000000.vtu'
    sim_mesh_V = root_path + cases[1]+'/post/full_0.750000000000.vtu'

    
    # Load both meshes
    R_mesh = io.read(reg_mesh)
    C_mesh = io.read(sim_mesh_C)
    V_mesh = io.read(sim_mesh_V)
    mass = np.ones_like(C_mesh.point_data['Pressure'])
    
    # Analyze regionally
    _, ids = IsoVolumetricSegmentation(directions[direction],mass,C_mesh.points, nrois)
    
    for mesh,kind in zip([C_mesh,V_mesh,R_mesh],(kinds+['reg'])):
        
        if kind != 'reg':
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
            
            if kind == 'Constant':
                data_manager['pos'] += [i]

                
        
    for kind,path in zip(kinds+['reg'],[sim_mesh_C,sim_mesh_V,reg_mesh]):
        
        
        if kind != 'reg':
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

# %%

    if True:            
        
        fig, axes = plt.subplots(nrows=3, figsize=(5,5),dpi=200)
        
        
        for kind, sim_signals in zip(kinds,[sim_signals_C, sim_signals_V]):
        
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
            
            data_manager[kind]['Raw']['mean'] += [R_rs]
            data_manager[kind]['Crs']['mean'] += [C_rs]
            data_manager[kind]['PEEP']['mean'] += [PEEP]
            
            # Pressure
            ax = axes[0]
            ax.plot(t,p,label=kind,alpha=0.80)
            # Flux
            ax = axes[1]
            ax.plot(t,q,alpha=0.80)
            # Volume
            ax = axes[2]
            ax.plot(t,v,alpha=0.80)
            
    
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
        
        ax = axes[0]
        ax.plot(t,p,label='Experiment',alpha=0.80, ls="--",color="k")
        ax.set_ylabel("Pressure (cmH2O)")
        ax.set_ylim((0,40))
        ax.set_xticks([])
        ax.legend() 
            
        ax = axes[1]
        ax.plot(t,q,alpha=0.80,color="k",ls="--")
        ax.set_ylim(-1.25,1.25)
        ax.set_xticks([])
        ax.set_ylabel("Flux (L/s)")
            
        ax = axes[2]
        ax.plot(t,v,alpha=0.80,color="k",ls="--")
        ax.set_ylabel("Volume (L)")
        ax.set_ylim(0,0.50)
        ax.set_ylabel("Time (s)")
        ax.set_xticks([0.0, 0.5, 1.0,1.5, 2.0])
        plt.tight_layout()

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

rydata = np.array(data_manager['Constant']['Jacobian']['mean'])
sydata = np.array(data_manager['Variable']['Jacobian']['mean'])
ydata = (rydata-sydata)
xdata = np.array(data_manager['pos'])

# Linear regression
#slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)
#txt = pval2txt(p_value)    

"t"


fig, ax = plt.subplots(dpi=300,figsize=(4,4))


#ax.plot(x_fit, y_fit, color='red', lw=1.5,
#            label='r = %.2f '%r_value+txt)

ax.set_xlabel('ROI ID')
ax.scatter(xdata,ydata)
ax.set_ylabel("Jacobian difference ($J_{Exponential}-J_{Birzle}$)")
ax.set_xticks(np.arange(10)+1,np.arange(10)+1)
#ax.text(0.5,-0.12,'Ventral')
#ax.text(9.7,-0.12,'Dorsal')
ax.axhline(0.0, color='k',alpha=0.25,ls='--')

ax.legend()

# %%

fig, axes = plt.subplots(ncols=3,figsize=(12,4),dpi=300)

for e,(ax,field) in enumerate(zip(axes,['EI-porosity','Jacobian', 'DGF'])):
    
    # Porosity is spatially correlated to Ventro-Dorsal position.
    # The further dorsal, the lowest the porosity, which is known.
    
    c_xdata = np.array(data_manager['Constant'][field]['mean'])
    v_xdata = np.array(data_manager['Variable'][field]['mean'])
    r_xdata = np.array(data_manager['reg'][field]['mean'])
    
    ydata = np.array(data_manager['pos'])
    
    labels = ['Const. $C_{tissue}$', 'Var. $C_{tissue}$', 'Experimental']
    
    

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    alpha=0.5
    ax.set_ylabel('ROI ID')
    
    for xdata,label in zip([c_xdata,v_xdata], labels):
        ax.scatter(xdata,ydata, label=label,alpha=alpha)
    
    ax.plot(r_xdata,ydata,label='Experimental', color='k',ls='--')
    
    ax.invert_yaxis()
    ax.set_yticks(np.arange(10),np.arange(10)+1)

    if field == 'EI-porosity':
        ax.set_xlabel("End-inspiratory porosity (-)")
        ax.set_xlim((0,1))
        
    elif field == 'Jacobian':
        ax.set_xlabel("Jacobian (-)")
        ax.set_xlim((1.,1.5))

    elif field == 'DGF':
        ax.set_xlabel("Delta Gas Fraction (-)")
        ax.set_xlim((-0.10,0.20))    
    
    
    
    ax.legend()
    
    if e>0:
        ax.set_ylabel("")
        ax.set_yticks([])
    else:
        x0,x1 = ax.get_xlim(); dx = x1-x0

        xtext = x0-0.25*dx
        ax.text(xtext,0,'Ventral',weight='bold')
        ax.text(xtext,9.5,'Dorsal',weight='bold')
    plt.tight_layout()


# %%
ydata = np.array(data_manager['pos'])
labels = ['Const. $C_{tissue}$', 'Var. $C_{tissue}$', 'Experimental']

fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(8,6),dpi=300)
axes = axes.flatten()

renamer = {'NAT':'Non-aerated tissue',
           'AT':'Normally-aerated tissue',
           'PAT':'Poorly-aerated tissue',
           'HIT':'Hyperinflated tissue'}

for e,(ax,field) in enumerate(zip(axes, ['NAT','PAT','AT','HIT'])):
    
    c_xdata = np.array(data_manager['Constant'][field]['mean'])
    v_xdata = np.array(data_manager['Variable'][field]['mean'])
    r_xdata = np.array(data_manager['reg'][field]['mean'])

    for xdata,label in zip([c_xdata, v_xdata, r_xdata],labels):
        
        if label != 'Experimental':
            ax.scatter(xdata,ydata,alpha=alpha, label=label)
        else:
            ax.plot(xdata,ydata,ls='--',color='k', label=label)
            
    ax.set_xlim((0.0,1.0))
    ax.set_ylim((-0.2,9.2))
    ax.set_yticklabels(["%i"%(f+1) for f in np.arange(nrois)])
        
    if e in [0,2]:
        
        ax.set_ylabel('ROI ID')
        ax.set_yticks(np.arange(nrois))
        
        ax.text(-0.24,0.0,'Ventral',weight='bold')
        ax.text(-0.24,8.5,'Dorsal',weight='bold')
        
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