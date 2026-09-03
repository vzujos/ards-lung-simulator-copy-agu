# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 13:33:15 2025

@author: angus
"""
import os
import meshio as io
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt

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


def retrieve_regional_histogram(mesh,
                                direction,
                                weights=None,
                                nrois=10,
                                verbose=False,
                                deformed_state=True,
                                bins=[0.0,0.1,0.5,0.9,1.0],
                                porosity_field='Eulerian Porosity',
                                dispfield='u'):
       
    if deformed_state: # Add deformation field to the mesh points
        xyz = mesh.points
        u = mesh.point_data[dispfield]
        xyz += u
    else: # Use raw point data
        xyz = mesh.points
        
    # Retrieve porosity field
    porosity = mesh.point_data[porosity_field] 

    # Dummy; Node mass should be used. How do I [compute] it?
    # TODO: Use the actual mass
    if weights is None:
        w = np.ones(xyz.shape[0])
    else:
        w = weights

    _, id_roi = IsoVolumetricSegmentation(direction,w,xyz,nrois)

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
            if i == 1: # Non aerated
                gen_binner[roi][i] = (counter[roi][0]+counter[roi][1])/N
                if verbose: print("NAT: %.1f"%((counter[roi][0]+counter[roi][1])/N*100)+"%")
            elif i == 4: # Hyperaerated
                gen_binner[roi][i] = (counter[roi][4]+counter[roi][5])/N
                if verbose: print("HIT: %.1f"%((counter[roi][4]+counter[roi][5])/N*100)+"%")
            else:
                gen_binner[roi][i] = counter[roi][i]/N
                if verbose: print("%s: %.1f"%(namer[i],counter[roi][i]*100/N)+"%")

    return gen_binner, counter

# Compare the different simulations when activating/deactivating options at the model

if __name__ == '__main__':

    # Case identifier and setting up analysis
    subject = 5
    mesh_type = "medium"
    direction = 'VD'
    nrois = 10
    
    # Directions vector
    directions = {'BA':np.mat([0.,0.,1.]).T,
                  'VD':np.mat([0.,1.,0.]).T,}
    
    # Address of the registration-derived mesh (ground-truth)
    baseline_mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/experiment-mesh.vtu"%(subject,mesh_type)
    baseline_mesh = io.read(baseline_mesh_path)
    
    # Address of the simulated meshes under analysis
    simulated_mesh = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    simulated_mesh += "gravity-tests/birzle-grav/post/"
        
    if subject == 5:
        simulated_mesh_name = "full_0.750000000000.vtu"
    elif subject == 4:
        simulated_mesh_name = "full_0.760000000000.vtu"
    elif subject == 3:
        simulated_mesh_name = "full_0.890000000000.vtu"
    else:
        simulated_mesh_name = None
    
    # Tags
    tags = ["A","B","C","D"]
    
    # Targets
    target_root =  "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    target_meshes = [target_root+"PIG5-medium-birzle-%s/post/"%tag+simulated_mesh_name for tag in tags]
    
    # Retrieve mass from baseline_mesh
    mass = baseline_mesh.point_data['Mass']        
    
    # ROI analysis
    _, ids = IsoVolumetricSegmentation(directions[direction],mass,baseline_mesh.points, nrois)
    
    
    # Fields that are requested for every geometry
    transversal_fields = ["End-Inspiratory Porosity", "Volumetric Strain", "Delta Gas Fraction",
                          "HIT","AT","PAT","NAT"]
    
    # Fields that are requested only for simulations
    specific_fields = ["Pressure (cmH2O)", "VM", "VM_tissue", "QQint", "HYD","HYD_tissue"]
    
    # Generate the data manager
    data_manager = {tag:{field:[] for field in transversal_fields} for tag in tags+["Exp"]}
    for tag in tags:
        data_manager[tag].update({field:[] for field in specific_fields})
    
    # Generate regional histogram for the baseline geometry
    hist, _ = retrieve_regional_histogram(baseline_mesh,directions[direction], nrois=nrois,
                                          weights=mass, porosity_field='End-Inspiratory Porosity')
    
    # Fill the data manager with the relevant data
    for roi in range(nrois):
        data_manager['Exp']['NAT'] += [hist[roi][1]] # NAT
        data_manager['Exp']['PAT'] += [hist[roi][2]] # PAT
        data_manager['Exp']['AT']  += [hist[roi][3]] # AT
        data_manager['Exp']['HIT'] += [hist[roi][4]] # HIT
    
    # ROI by ROI weighted analysis
    for roi in range(nrois):
        # ROI mask; Masked mass
        mask = ids==roi; mmass = mass[mask]
        # End-inspiratory intensity
        st_ei = DescrStatsW(baseline_mesh.point_data['EI Intensity'][mask], weights=mmass)
        data_manager['Exp']['End-Inspiratory Porosity'] += [st_ei.mean]
        # Volumetric Strain
        st_vs = DescrStatsW(baseline_mesh.point_data['VolStrain'][mask], weights=mmass)
        data_manager['Exp']['Volumetric Strain'] += [st_vs.mean]
        # Delta Gas Fraction
        st_dgf = DescrStatsW(baseline_mesh.point_data['Delta Porosity'][mask], weights=mmass)
        data_manager['Exp']['Delta Gas Fraction'] += [st_dgf.mean]
    
    data_manager['Exp']['Volumetric Strain'] 
    
    # For every mesh
    for target_mesh_path,tag in zip(target_meshes,tags):
        
        # Load the target mesh
        target_mesh = io.read(target_mesh_path)

        # Generate histograms
        hist, _ = retrieve_regional_histogram(target_mesh,directions[direction], nrois=nrois,
                                              weights=mass, porosity_field='Eulerian Porosity')
    
        for roi in range(nrois):
            
            mask = ids==roi; mmass = mass[mask]

            data_manager[tag]['NAT'] += [hist[roi][1]] # NAT
            data_manager[tag]['PAT'] += [hist[roi][2]] # PAT
            data_manager[tag]['AT']  += [hist[roi][3]] # AT
            data_manager[tag]['HIT'] += [hist[roi][4]] # HIT
            
            mmass = mass[mask]
            
            st_ei = DescrStatsW(target_mesh.point_data['Eulerian Porosity'][mask], weights=mmass)
            data_manager[tag]['End-Inspiratory Porosity'] += [st_ei.mean]
            # Volumetric Strain
            vs = (target_mesh.point_data['Jacobian Partial'][mask]-1)*100
            st_vs = DescrStatsW(vs, weights=mmass)
            data_manager[tag]['Volumetric Strain'] += [st_vs.mean]
            # Delta Gas Fraction
            st_dgf = DescrStatsW(target_mesh.point_data['Delta Porosity'][mask], weights=mmass)
            data_manager[tag]['Delta Gas Fraction'] += [st_dgf.mean]
            # Pressure (cmH2O)            
            st_pres = DescrStatsW(target_mesh.point_data['Pressure (cmH2O)'][mask], weights=mmass)
            data_manager[tag]['Pressure (cmH2O)'] += [st_pres.mean]
            # VM
            st_vm = DescrStatsW(target_mesh.point_data['VM'][mask], weights=mmass)
            data_manager[tag]['VM'] += [st_vm.mean]
            # VM tissue
            st_vmt = DescrStatsW(target_mesh.point_data['VM_tissue'][mask], weights=mmass)
            data_manager[tag]['VM_tissue'] += [st_vmt.mean]
            # HYD
            st_hyd = DescrStatsW(target_mesh.point_data['HYD'][mask], weights=mmass)
            data_manager[tag]['HYD'] += [-st_hyd.mean*10.1972]
            # HYD tissue
            st_hydt = DescrStatsW(target_mesh.point_data['HYD_tissue'][mask], weights=mmass)
            data_manager[tag]['HYD_tissue'] += [-st_hydt.mean*10.1972]
                      
# %%
# Aeration compartment information

rename = {"NAT":"Non-aerated tissue",
          "PAT":"Poorly-aerated tissue",
          "AT":"Normally-aerated tissue",
          "HIT":"Hyperinflated tissue",
          "Delta Gas Fraction":"Delta Gas Fraction (-)",
          "Volumetric Strain":"Volumetric Strain (%)",
          "Pressure (cmH2O)":"Alveolar pressure \n($cmH_{2}O$)",
          "VM":"Von Misses stress \n(kPa)",
          "HYD":"Hydrostatic pressure  \n($cmH_{2}O$)",
          "VM_tissue":"(Tissue) \n Von Misses stress \n(kPa)",
          "HYD_tissue":"(Tissue) \nHydrostatic pressure \n($cmH_{2}O$)",
          "Exp":"Experiment",
          "A":"No gravity - $C_{tissue}=Const.$",
          "B":"Gravity - $C_{tissue}=Const.$",
          "C":"No gravity - $C_{tissue}=C_{tissue}(x)$",
          "D":"Gravity - $C_{tissue}=C_{tissue}(x)$",
          }

        
alph = 0.6

line_style = {"Exp":("k","--","*",1.0), # Color, Linestyle, marker,alpha
              "A":("tab:orange","-","s",alph),
              "B":("tab:green","-","s",alph),
              "C":("tab:purple","-","s",alph),
              "D":("tab:blue","-","s",alph)}

fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(7,7),dpi=200)
axes = axes.flatten()

ydata = np.arange(nrois)

for ax,field in zip(axes,['NAT','PAT','AT','HIT']):
    for tag in tags+["Exp"]:
        color, ls,m, alpha = line_style[tag]
        xdata = data_manager[tag][field]
        ax.plot(xdata,ydata, color=color,ls=ls, lw=1.0,alpha=alpha)
        ax.scatter(xdata,ydata,color=color,label=rename[tag], marker=m,s=10,alpha=alpha)    
    ax.set_xlim((0,1))
    ax.set_title(rename[field],size=10)
    ax.invert_yaxis()
    if field == "NAT":
        ax.legend(loc=1,prop={'size':8})
        
    if field in ['NAT','AT']:
        ax.set_yticks(np.linspace(0,9,10))
        ax.set_yticklabels(["%i"%n for n in np.linspace(0,9,10)+1])
        ax.text(-0.25,0,'Ventral')
        ax.text(-0.25,9.5,'Dorsal')
        ax.set_ylabel("ROI ID")
    else:
        ax.set_yticks([])
    ax.set_xlabel("Fraction")
plt.tight_layout()

# %%

# Other field information
        
fig,axes = plt.subplots(ncols=2,figsize=(10,4),dpi=200)
axes = axes.flatten()

ydata = np.arange(nrois)

for ax,field in zip(axes,['Delta Gas Fraction',"Volumetric Strain"]):
    for tag in tags+["Exp"]:
        color, ls,m, alpha = line_style[tag]
        xdata = data_manager[tag][field]
        ax.plot(xdata,ydata, color=color,ls=ls, lw=1.0,alpha=alpha)
        ax.scatter(xdata,ydata,color=color,label=rename[tag], marker=m,s=10,alpha=alpha)    
#    ax.set_xlim((0,1))
    ax.set_title(rename[field],size=10)
    ax.invert_yaxis()
  
    if field == "NAT":
        ax.legend(loc=1,prop={'size':8})
     
    if field == "Delta Gas Fraction":
        ax.set_yticks(np.linspace(0,9,10))
        ax.set_yticklabels(["%i"%n for n in np.linspace(0,9,10)+1])
        ax.text(-0.05,-0.3,'Ventral')
        ax.text(-0.05,9.6,'Dorsal')
        ax.set_ylabel("ROI ID")
        ax.set_xlabel(rename[field])        
        ax.set_xlim((-0.020,0.140))
        ax.set_xticks(np.linspace(-0.02,0.14,5))
    else:
        ax.set_xlabel(rename[field])
        ax.set_yticks([])
        ax.set_xlim((0,30))
        ax.legend(bbox_to_anchor=(1.05, 1))
 #   else:
    #    ax.set_yticks([])
        
    
plt.tight_layout()


# %%

# Other field information
        
fig,axes = plt.subplots(ncols=3,figsize=(10,4),dpi=200)
axes = axes.flatten()

ydata = np.arange(nrois)

for ax,field in zip(axes,["Pressure (cmH2O)","VM","HYD"]):
    for tag in tags:
        color, ls,m, alpha = line_style[tag]
        xdata = data_manager[tag][field]
        ax.plot(xdata,ydata, color=color,ls=ls, lw=1.0,alpha=alpha)
        ax.scatter(xdata,ydata,color=color,label=rename[tag], marker=m,s=10,alpha=alpha)    
#    ax.set_xlim((0,1))
  #  ax.set_title(rename[field],size=10)
    ax.invert_yaxis()
  
    if field == "Pressure (cmH2O)":
        ax.set_yticks(np.linspace(0,9,10))
        ax.set_yticklabels(["%i"%n for n in np.linspace(0,9,10)+1])
        ax.text(20.6,-0.3,'Ventral')
        ax.text(20.6,9.6,'Dorsal')
        ax.set_ylabel("ROI ID")
        ax.set_xlabel(rename[field])
    #    ax.set_xlim((-0.020,0.120))
    else:
        ax.set_xlabel(rename[field])
        ax.set_yticks([])
      #  ax.set_xlim((0,30))

ax.legend(bbox_to_anchor=(1.05, 1))
 #   else:
    #    ax.set_yticks([])
        

plt.tight_layout()

# %%
# Other field information
        
fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(8,6),dpi=200)
axes = axes.flatten()

ydata = np.arange(nrois)

for ax,field in zip(axes,["VM","VM_tissue","HYD","HYD_tissue"]):
    for tag in tags:
        color, ls,m, alpha = line_style[tag]
        xdata = data_manager[tag][field]
        ax.plot(xdata,ydata, color=color,ls=ls, lw=1.0,alpha=alpha)
        ax.scatter(xdata,ydata,color=color,label=rename[tag], marker=m,s=10,alpha=alpha)    
#    ax.set_xlim((0,1))
  #  ax.set_title(rename[field],size=10)
    ax.invert_yaxis()
  
    if field == "VM":
        ax.set_yticks(np.linspace(0,9,10))
        ax.set_yticklabels(["%i"%n for n in np.linspace(0,9,10)+1])
        ax.text(-0.4,-0.3,'Ventral')
        ax.text(-0.4,9.6,'Dorsal')
        ax.set_ylabel("ROI ID")
        ax.set_xlabel(rename[field])
    #    ax.set_xlim((-0.020,0.120))
    elif field == "HYD":
        ax.set_yticks(np.linspace(0,9,10))
        ax.set_xlim((-1,5))
        ax.set_xticks((np.linspace(-1,5,7)))
        ax.set_yticklabels(["%i"%n for n in np.linspace(0,9,10)+1])
        ax.text(-3.0,-0.5,'Ventral')
        ax.text(-3.0,9.9,'Dorsal')
        ax.set_ylabel("ROI ID")
        ax.set_xlabel(rename[field])
    else:
        ax.set_xlabel(rename[field])
        ax.set_yticks([])
      #  ax.set_xlim((0,30))

axes[1].legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()

# %%
# Other field information
        
fig,axes = plt.subplots(ncols=5,nrows=1,figsize=(15,4),dpi=200)
axes = axes.flatten()

ydata = np.arange(nrois)

for ax,field in zip(axes,["Pressure (cmH2O)","HYD","HYD_tissue","VM","VM_tissue",]):
    for tag in tags:
        color, ls,m, alpha = line_style[tag]
        xdata = data_manager[tag][field]
        ax.plot(xdata,ydata, color=color,ls=ls, lw=1.0,alpha=alpha)
        ax.scatter(xdata,ydata,color=color,label=rename[tag], marker=m,s=10,alpha=alpha)    
#    ax.set_xlim((0,1))
  #  ax.set_title(rename[field],size=10)
    ax.invert_yaxis()
  
    if field == "Pressure (cmH2O)":
        ax.set_yticks(np.linspace(0,9,10))
        ax.set_yticklabels(["%i"%n for n in np.linspace(0,9,10)+1])
        ax.text(18.6,-0.3,'Ventral')
        ax.text(18.6,9.6,'Dorsal')
        ax.set_ylabel("ROI ID")
        ax.set_xlabel(rename[field])
        
        ax.set_xlim((20,25))
        ax.set_xticks(np.linspace(20,25,6))
        ax.set_xlabel(rename[field])
    #    ax.set_xlim((-0.020,0.120))
    else:
        ax.set_xlabel(rename[field])
        ax.set_yticks([])
      #  ax.set_xlim((0,30))
      
    if field == 'HYD':
        ax.set_xlim((-1,6))
        ax.set_xticks(np.linspace(-1,6,8))
        ax.set_xlabel(rename[field])

    if field == 'HYD_tissue':
        ax.set_xlim((-24,-16))
        ax.set_xticks(np.linspace(-24,-16,5))
        ax.set_xlabel(rename[field])
        
    if field in ['VM','VM_tissue'] :
        ax.set_xlim((0.0,1.25))
        ax.set_xticks(np.linspace(0.0,1.25,6))
        ax.set_xlabel(rename[field])


axes[-1].legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()


# %%
# Other field information
        
fig,axes = plt.subplots(ncols=3,nrows=1,figsize=(11,4),dpi=200)
axes = axes.flatten()

ydata = np.arange(nrois)

for ax,field in zip(axes,["Pressure (cmH2O)","HYD","VM"]):
    for tag in tags:

        
        color, ls,m, alpha = line_style[tag]
        xdata = data_manager[tag][field]
        ax.plot(xdata,ydata, color=color,ls=ls, lw=1.0,alpha=alpha)
        ax.scatter(xdata,ydata,color=color,label=rename[tag], marker=m,s=10,alpha=alpha)    
#    ax.set_xlim((0,1))
  #  ax.set_title(rename[field],size=10)
    ax.invert_yaxis()
  
    if field == "Pressure (cmH2O)":
        ax.set_yticks(np.linspace(0,9,10))
        ax.set_yticklabels(["%i"%n for n in np.linspace(0,9,10)+1])
        ax.text(16.6,-0.3,'Ventral')
        ax.text(16.6,9.6,'Dorsal')
        ax.set_ylabel("ROI ID")
        ax.set_xlabel(rename[field])
        
        ax.set_xlim((20,30))
#        ax.set_xticks(np.linspace(20,25,6))
        ax.set_xlabel(rename[field])
    #    ax.set_xlim((-0.020,0.120))
    else:
        ax.set_xlabel(rename[field])
        ax.set_yticks([])
      #  ax.set_xlim((0,30))
      
    if field == 'HYD':
        ax.set_xlim((-2,10))
        ax.set_xticks(np.linspace(-2,10,7))
        ax.set_xlabel(rename[field])

    if field == 'HYD_tissue':
        ax.set_xlim((-24,-16))
        ax.set_xticks(np.linspace(-24,-16,5))
        ax.set_xlabel(rename[field])
        
    if field in ['VM','VM_tissue'] :
        ax.set_xlim((0.0,2.0))
        ax.set_xticks(np.linspace(0.0,2.0,5))
        ax.set_xlabel(rename[field])


axes[-1].legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()