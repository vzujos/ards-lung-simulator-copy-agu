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
#root_path = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/'
root_path = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/GitHub/ards-lung-simulator/src/tests/inverse-problem-lab/problem-'

cases = ['16/inverse-analysis/','17/inverse-analysis/',]


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

subject = 5

for e,(kind,case) in enumerate(zip(kinds,cases)):
        
    # Build code
    code = 'PIG%i-ARDSnet'%subject
    # Retrieve time data
    Tsyr = parameters[code]['Tsyr']
    Tpausa = parameters[code]['Tpausa']
    Texp = parameters[code]['Texp']
    

    # Path to wherever the meshes are 
    mesh_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/medium/'
    
    # Names for the registration-derived mesh and the simulated mesh
    inv_mesh = root_path + case+'/Jacobian000000.vtu'

    
    # Load both meshes
    imesh = io.read(inv_mesh)
    mass = np.ones_like(imesh.point_data['Jacobian'])
    
    # Analyze regionally
    _, ids = IsoVolumetricSegmentation(directions[direction],mass,imesh.points, nrois)
    

    j = imesh.point_data['Jacobian']
        
    for i in range(nrois):
            
        mask = ids==i
        #local_eip = eiporosity[mask]
        #local_dgf = dgf[mask]

        local_mass = mass[mask]
        local_j = j[mask]
            
        j_stat = sw.DescrStatsW(data=local_j,weights=local_mass)
        #eip_stat = sw.DescrStatsW(data=local_eip,weights=local_mass)
        #dgf_stat = sw.DescrStatsW(data=local_dgf,weights=local_mass)
      
        data_manager[kind]['Jacobian']['mean'] += [j_stat.mean]
        data_manager[kind]['Jacobian']['std'] += [j_stat.std]
        #data_manager[kind]['EI-porosity']['mean'] += [eip_stat.mean]
        #data_manager[kind]['EI-porosity']['std'] += [eip_stat.std]
        #data_manager[kind]['DGF']['mean'] += [dgf_stat.mean]
        #data_manager[kind]['DGF']['std'] += [dgf_stat.std]
        
        if e==0:
            data_manager['pos'] += [i]

# %%


fig, ax = plt.subplots(ncols=1,figsize=(4,4),dpi=300)

#for e,(ax,field) in enumerate(zip(axes,['EI-porosity','Jacobian', 'DGF'])):
for e,(ax,field) in enumerate(zip([ax],['Jacobian',])):
    
    # Porosity is spatially correlated to Ventro-Dorsal position.
    # The further dorsal, the lowest the porosity, which is known.
    
    c_xdata = np.array(data_manager['Constant'][field]['mean'])
    v_xdata = np.array(data_manager['Variable'][field]['mean'])
   # r_xdata = np.array(data_manager['reg'][field]['mean'])
    
    ydata = np.array(data_manager['pos'])
    
    labels = ['Without', 'With', 'Experimental']
    
    

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    alpha=0.5
    ax.set_ylabel('ROI ID')
    
    for xdata,label in zip([c_xdata, v_xdata], labels):
        ax.scatter(xdata,ydata, label=label,alpha=alpha)
        ax.plot(xdata,ydata,alpha=alpha)
    
    ax.axvline(1.0, ls='--',color='r',alpha=0.25)
    #ax.plot(r_xdata,ydata,label='Experimental', color='k',ls='--')
    
    ax.invert_yaxis()
    ax.set_yticks(np.arange(10),np.arange(10)+1)

    if field == 'EI-porosity':
        ax.set_xlabel("End-inspiratory porosity (-)")
        ax.set_xlim((0,1))
        
    elif field == 'Jacobian':
        ax.set_xlabel("Jacobian (-)")
        ax.set_xlim((0.95,1.80))

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
  #  r_xdata = np.array(data_manager['reg'][field]['mean'])

   # for xdata,label in zip([c_xdata, v_xdata, r_xdata],labels):
    for xdata,label in zip([c_xdata, v_xdata],labels):
        
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