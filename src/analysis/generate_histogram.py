# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:34:28 2024

@author: angus
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import meshio as io
import legacy.ROIAnalysis as ROI
from scipy.sparse import coo_matrix

def Tet4DefGradSmoothingOptiSPARSE(xyz, defmap, LM):
	"""
	This subroutine computes the assemble of mass vector and force vector (matrix)
	Input:
	  xyz		: array with node coordinates in reference configuration
	  defmap	 : array with nodal deformation mapping vector 
	  LM		 : connectivity matrix

	Output:
	  Mlumped   : Lumped mass matrix (in vector form)
	  P		 : Weighted deformation-gradient vector
	"""	
	me  = LM
	q   = xyz
	phi = defmap
	
	V = np.zeros((me.shape[0],4,4))
	
	qme   = q[me[:]]
	phime = phi[me[:]]
	
	ones = np.ones((me.shape[0],4))
	x = qme[:,:,0]
	y = qme[:,:,1]
	z = qme[:,:,2]
	
	T = np.column_stack((ones[:],x[:],y[:],z[:]))
	V[:,0,0]=T[:,0];V[:,0,1]=T[:,1];V[:,0,2]=T[:,2];V[:,0,3]=T[:,3];
	V[:,1,0]=T[:,4];V[:,1,1]=T[:,5];V[:,1,2]=T[:,6];V[:,1,3]=T[:,7];
	V[:,2,0]=T[:,8];V[:,2,1]=T[:,9];V[:,2,2]=T[:,10];V[:,2,3]=T[:,11]
	V[:,3,0]=T[:,12];V[:,3,1]=T[:,13];V[:,3,2]=T[:,14];V[:,3,3]=T[:,15]
	
	ve0=np.zeros(me.shape[0])
	ve0[:]=np.linalg.det(V[:])
	ve0=ve0/6.
	
	xdef=phime[:,:,0]
	ydef=phime[:,:,1]
	zdef=phime[:,:,2]
	
	T=np.column_stack((ones[:],xdef[:],ydef[:],zdef[:]))
	
	V[:,0,0]=T[:,0];V[:,0,1]=T[:,1];V[:,0,2]=T[:,2];V[:,0,3]=T[:,3];
	V[:,1,0]=T[:,4];V[:,1,1]=T[:,5];V[:,1,2]=T[:,6];V[:,1,3]=T[:,7];
	V[:,2,0]=T[:,8];V[:,2,1]=T[:,9];V[:,2,2]=T[:,10];V[:,2,3]=T[:,11]
	V[:,3,0]=T[:,12];V[:,3,1]=T[:,13];V[:,3,2]=T[:,14];V[:,3,3]=T[:,15]
	
	ve=np.zeros(me.shape[0])
	ve[:]=np.linalg.det(V[:])
	ve=ve/6.
	
	x=qme[:,:,0]
	y=qme[:,:,1]
	z=qme[:,:,2]
	
	a1=(y[:,3]-y[:,1])*(z[:,2]-z[:,1])-(y[:,2]-y[:,1])*(z[:,3]-z[:,1])	  #yd[3,1]*zd[2,1]-yd[2,1]*zd[3,1]
	a2=(y[:,2]-y[:,0])*(z[:,3]-z[:,2])-(y[:,2]-y[:,3])*(z[:,0]-z[:,2])	  #yd[2,0]*zd[3,2]-yd[2,3]*zd[0,2]
	a3=(y[:,1]-y[:,3])*(z[:,0]-z[:,3])-(y[:,0]-y[:,3])*(z[:,1]-z[:,3])	  #yd[1,3]*zd[0,3]-yd[0,3]*zd[1,3]
	a4=(y[:,0]-y[:,2])*(z[:,1]-z[:,0])-(y[:,0]-y[:,1])*(z[:,2]-z[:,0])	  #yd[0,2]*zd[1,0]-yd[0,1]*zd[2,0]
	
	b1=(x[:,2]-x[:,1])*(z[:,3]-z[:,1])-(x[:,3]-x[:,1])*(z[:,2]-z[:,1])	  #xd[2,1]*zd[3,1]-xd[3,1]*zd[2,1]
	b2=(x[:,3]-x[:,2])*(z[:,2]-z[:,0])-(x[:,0]-x[:,2])*(z[:,2]-z[:,3])	  #xd[3,2]*zd[2,0]-xd[0,2]*zd[2,3]
	b3=(x[:,0]-x[:,3])*(z[:,1]-z[:,3])-(x[:,1]-x[:,3])*(z[:,0]-z[:,3])	  #xd[0,3]*zd[1,3]-xd[1,3]*zd[0,3]
	b4=(x[:,1]-x[:,0])*(z[:,0]-z[:,2])-(x[:,2]-x[:,0])*(z[:,0]-z[:,1])	  #xd[1,0]*zd[0,2]-xd[2,0]*zd[0,1]
	
	c1=(x[:,3]-x[:,1])*(y[:,2]-y[:,1])-(x[:,2]-x[:,1])*(y[:,3]-y[:,1])	  #xd[3,1]*yd[2,1]-xd[2,1]*yd[3,1]
	c2=(x[:,2]-x[:,0])*(y[:,3]-y[:,2])-(x[:,2]-x[:,3])*(y[:,0]-y[:,2])	  #xd[2,0]*yd[3,2]-xd[2,3]*yd[0,2]
	c3=(x[:,1]-x[:,3])*(y[:,0]-y[:,3])-(x[:,0]-x[:,3])*(y[:,1]-y[:,3])	  #xd[1,3]*yd[0,3]-xd[0,3]*yd[1,3]
	c4=(x[:,0]-x[:,2])*(y[:,1]-y[:,0])-(x[:,0]-x[:,1])*(y[:,2]-y[:,0])	  #xd[0,2]*yd[1,0]-xd[0,1]*yd[2,0]
	
	
	DN=[[a1[:]/(6*ve0[:]),a2[:]/(6*ve0[:]),a3[:]/(6*ve0[:]),a4[:]/(6*ve0[:])],[b1[:]/(6*ve0[:]),b2[:]/(6*ve0[:]),b3[:]/(6*ve0[:]),b4[:]/(6*ve0[:])],[c1[:]/(6*ve0[:]),c2[:]/(6*ve0[:]),c3[:]/(6*ve0[:]),c4[:]/(6*ve0[:])]]		
	
	DN=np.array(DN).T
	
	F11=DN[:,0,0]*phime[:,0,0]+DN[:,1,0]*phime[:,1,0]+DN[:,2,0]*phime[:,2,0]+DN[:,3,0]*phime[:,3,0]
	F12=DN[:,0,1]*phime[:,0,0]+DN[:,1,1]*phime[:,1,0]+DN[:,2,1]*phime[:,2,0]+DN[:,3,1]*phime[:,3,0]
	F13=DN[:,0,2]*phime[:,0,0]+DN[:,1,2]*phime[:,1,0]+DN[:,2,2]*phime[:,2,0]+DN[:,3,2]*phime[:,3,0]
	
	F21=DN[:,0,0]*phime[:,0,1]+DN[:,1,0]*phime[:,1,1]+DN[:,2,0]*phime[:,2,1]+DN[:,3,0]*phime[:,3,1]
	F22=DN[:,0,1]*phime[:,0,1]+DN[:,1,1]*phime[:,1,1]+DN[:,2,1]*phime[:,2,1]+DN[:,3,1]*phime[:,3,1]
	F23=DN[:,0,2]*phime[:,0,1]+DN[:,1,2]*phime[:,1,1]+DN[:,2,2]*phime[:,2,1]+DN[:,3,2]*phime[:,3,1]
	
	F31=DN[:,0,0]*phime[:,0,2]+DN[:,1,0]*phime[:,1,2]+DN[:,2,0]*phime[:,2,2]+DN[:,3,0]*phime[:,3,2]
	F32=DN[:,0,1]*phime[:,0,2]+DN[:,1,1]*phime[:,1,2]+DN[:,2,1]*phime[:,2,2]+DN[:,3,1]*phime[:,3,2]
	F33=DN[:,0,2]*phime[:,0,2]+DN[:,1,2]*phime[:,1,2]+DN[:,2,2]*phime[:,2,2]+DN[:,3,2]*phime[:,3,2]
	
	
	F=np.zeros((me.shape[0],3,3))
	
	F[:,0,0]=F11[:]
	F[:,0,1]=F12[:]
	F[:,0,2]=F13[:]
	
	F[:,1,0]=F21[:]
	F[:,1,1]=F22[:]
	F[:,1,2]=F23[:]
	
	F[:,2,0]=F31[:]
	F[:,2,1]=F32[:]
	F[:,2,2]=F33[:]
	
	Ft=np.zeros((me.shape[0],3,3))
	
	Ft[:,0,0]=F11[:]
	Ft[:,0,1]=F21[:]
	Ft[:,0,2]=F31[:]
	
	Ft[:,1,0]=F12[:]
	Ft[:,1,1]=F22[:]
	Ft[:,1,2]=F32[:]
	
	Ft[:,2,0]=F13[:]
	Ft[:,2,1]=F23[:]
	Ft[:,2,2]=F33[:]
	
	
	Ig=me.reshape((me.shape[0]*4,))
	Mg=np.array([ve0[:],ve0[:],ve0[:],ve0[:]])*1./4.
	Mg=Mg.T
	Mg=Mg.reshape((me.shape[0]*4,))
	M=coo_matrix((Mg,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	Mlumped=np.array(M.diagonal())
	
	
	e11=np.array([ve0[:]*F11[:],ve0[:]*F11[:],ve0[:]*F11[:],ve0[:]*F11[:]])*1./4.
	e11=e11.T
	e11=e11.reshape((me.shape[0]*4,))
	E11=coo_matrix((e11,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E11=np.array(E11.diagonal())
	
	e12=np.array([ve0[:]*F12[:],ve0[:]*F12[:],ve0[:]*F12[:],ve0[:]*F12[:]])*1./4.
	e12=e12.T
	e12=e12.reshape((me.shape[0]*4,))
	E12=coo_matrix((e12,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E12=np.array(E12.diagonal())
	
	
	e13=np.array([ve0[:]*F13[:],ve0[:]*F13[:],ve0[:]*F13[:],ve0[:]*F13[:]])*1./4.
	e13=e13.T
	e13=e13.reshape((me.shape[0]*4,))
	E13=coo_matrix((e13,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E13=np.array(E13.diagonal())
	
	e21=np.array([ve0[:]*F21[:],ve0[:]*F21[:],ve0[:]*F21[:],ve0[:]*F21[:]])*1./4.
	e21=e21.T
	e21=e21.reshape((me.shape[0]*4,))
	E21=coo_matrix((e21,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E21=np.array(E21.diagonal())
	
	
	e22=np.array([ve0[:]*F22[:],ve0[:]*F22[:],ve0[:]*F22[:],ve0[:]*F22[:]])*1./4.
	e22=e22.T
	e22=e22.reshape((me.shape[0]*4,))
	E22=coo_matrix((e22,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E22=np.array(E22.diagonal())
	
	e23=np.array([ve0[:]*F23[:],ve0[:]*F23[:],ve0[:]*F23[:],ve0[:]*F23[:]])*1./4.
	e23=e23.T
	e23=e23.reshape((me.shape[0]*4,))
	E23=coo_matrix((e23,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E23=np.array(E23.diagonal())
	
	e31=np.array([ve0[:]*F31[:],ve0[:]*F31[:],ve0[:]*F31[:],ve0[:]*F31[:]])*1./4.
	e31=e31.T
	e31=e31.reshape((me.shape[0]*4,))
	E31=coo_matrix((e31,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E31=np.array(E31.diagonal())
	
	e32=np.array([ve0[:]*F32[:],ve0[:]*F32[:],ve0[:]*F32[:],ve0[:]*F32[:]])*1./4.
	e32=e32.T
	e32=e32.reshape((me.shape[0]*4,))
	E32=coo_matrix((e32,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E32=np.array(E32.diagonal())
	
	e33=np.array([ve0[:]*F33[:],ve0[:]*F33[:],ve0[:]*F33[:],ve0[:]*F33[:]])*1./4.
	e33=e33.T
	e33=e33.reshape((me.shape[0]*4,))
	E33=coo_matrix((e33,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E33=np.array(E33.diagonal())
	
	P=np.array([E11,E12,E13,E21,E22,E23,E31,E32,E33]).T
	
	return Mlumped, P, ve, ve0

def retrieve_global_histogram(mesh_path, 
                              bins=[0.0,0.1,0.5,0.9,1.0],
                              reference_state=True,
                              verbose=True):
    
    '''
    Generate a histogram for the whole lung mesh.
    '''
    
    # Read the mesh
    mesh = io.read(mesh_path)
    # Extract the Eulerian porosity
    porosity = mesh.point_data['Eulerian Porosity'] 
    N = len(porosity)
    # Clasify the points according to their intensity (porosity)
    digit = np.digitize(porosity,bins)
    # Generate a counter for each possible classification
    counter = {i:None for i in range(6)}
    # Organize data
    for bin_ in range(6):
        counter[bin_] = np.count_nonzero(digit==bin_)
    # Keep track of the names
    namer = {0:"OOB-NAT", # Out-of-bounds non-aerated tissue
             1:"NAT", # Non-aerated tissue
             2:"PAT", # Poorly aerated tissue
             3:" AT", # Normally aerated tissue
             4:"HIT", # Hyperinflated tissue
             5:"OOB-HIT"} # Out-of-bounds hypperinflated tissue
    
    # Keep the final bins
    binner = {i:None for i in range(4)}

    # Generate the final sorting on each bin
    for i in [1,2,3,4]:
        if i == 1:
            binner[i-1] = (counter[0]+counter[1])/N
            if verbose: print("NAT: %.1f"%((counter[0]+counter[1])/N*100)+"%")
        elif i == 4:
            binner[i-1] = (counter[4]+counter[5])/N
            if verbose: print("HIT: %.1f"%((counter[4]+counter[5])/N*100)+"%")
        else:
            binner[i-1] = counter[i]/N
            if verbose: print("%s: %.1f"%(namer[i],counter[i]*100/N)+"%")
    
    return binner, counter, digit



def retrieve_regional_histogram(mesh_path,
                                direction,
                                weights=None,
                                nrois=10,
                                verbose=False,
                                deformed_state=True,
                                bins=[0.0,0.1,0.5,0.9,1.0]):
   
    mesh = io.read(mesh_path)
    
    if deformed_state: # Add deformation field to the mesh points
        xyz = mesh.points
        u = mesh.point_data['u']
        xyz += u
    else: # Use raw point data
        xyz = mesh.points
        
    # Retrieve porosity field
    porosity = mesh.point_data['Eulerian Porosity'] 


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

    _, id_roi = ROI.IsoVolumetricSegmentation(dirs[direction],w,xyz,nrois)

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

# %%


plot_signals = True

results_root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
case =  "Calibrated-PIG5-ARDSnet-6" # Fine
#case = "CORNELL-PIG6-ARDSnet-calibrated-1" # Coarse

 
results_dir = results_root+case
post_dir = results_dir+"/post/"
post_files = os.listdir(post_dir)
if 'simulation.vtu.series' in post_files:
    post_files.remove('simulation.vtu.series')
post_files=sorted(post_files)


if os.path.isdir(results_dir):
    print("Results folder found!")
else:
    print("Failed to found the results folder! ")
    
# Examine the signals associated to the simulation to find an appropiate
# inspiratory pause point
pressure = np.load(results_dir+"/Signals/presionestodas.npy")
flow = np.load(results_dir+"/Signals/fluxes.npy")
time = np.load(results_dir+"/Signals/effectivetimes.npy")

# Bound it through peak peak pressure and minimal flow
peak_pos = np.argmax(pressure)
min_flow = np.argmin(flow)

# Generate a mask to sample
sampling_mask = np.zeros_like(pressure.flatten(),dtype=bool)
sampling_mask[peak_pos:min_flow] = True

# Determine the latest point with zero-ish flow
ids = np.arange(flow.shape[0])[sampling_mask]
candidates = ids[np.abs(flow[sampling_mask])<1e-6]
sampling_id = np.max(candidates)+1

# Generate a plot to see what happened
if plot_signals:
    
    fig,axes = plt.subplots(ncols=1,nrows=2,figsize=(6,6))
    ax=axes[0]
    ax.plot(time,pressure,color="k",alpha=0.2)
    ax.scatter(time[peak_pos],pressure[peak_pos], color='r')
    ax.scatter(time[min_flow],pressure[min_flow], color='b')
    ax.plot(time[sampling_mask], pressure[sampling_mask],color='g',alpha=0.6)
    ax.scatter(time[sampling_id],pressure[sampling_id], color='m',marker='d')
    ax.set_ylabel("Pressure")
    
    ax=axes[1]
    ax.plot(time,flow,color="k",alpha=0.2)
    ax.scatter(time[peak_pos],flow[peak_pos], color='r')
    ax.scatter(time[min_flow],flow[min_flow], color='b')
    ax.plot(time[sampling_mask], flow[sampling_mask],color='g',alpha=0.6)
    ax.scatter(time[sampling_id],flow[sampling_id], color='m',marker='d')
    ax.set_ylabel("Flow")
    ax.set_xlabel("Time (s)")
 
# Retrieve time data from post_files to compare with current times under evaluation
#func = lambda x: float(x.split("_")[1].split(".vtu")[0])
#post_times = list(map(func,post_files))[1:]


# %% Generate global histogram
       
end_expiratory_geometry = post_dir+"/"+post_files[0]
end_inspiratory_geometry = post_dir+"/"+post_files[sampling_id]

ee_bins,_,_ = retrieve_global_histogram(end_expiratory_geometry)
ei_bins,_,_ = retrieve_global_histogram(end_inspiratory_geometry)

# Generate legends
natp = mpatches.Patch(facecolor="w",edgecolor="k",label="NAT")
patp = mpatches.Patch(facecolor="k",edgecolor="k",alpha=0.25,label="PAT") 
atp = mpatches.Patch(facecolor="k",edgecolor="k",alpha=0.50,label="AT") 
hitp = mpatches.Patch(facecolor="k",edgecolor="k",label="HIT") 

# Create global figure
fig, ax = plt.subplots(figsize=(6,8),dpi=200)
width = 0.15
dx = 0.20
for e,binner in enumerate([ee_bins, ei_bins]):
    # Stack bars
    ax.bar(1+(e-0.5)*dx, binner[0],width=width, color = "w",edgecolor='k') 
    ax.bar(1+(e-0.5)*dx, binner[1],width=width,bottom=binner[0], color = "k", alpha = 0.25, edgecolor="k")
    ax.bar(1+(e-0.5)*dx, binner[2],width=width,bottom=binner[0]+binner[1], color="k", alpha = 0.5, edgecolor="k")
    ax.bar(1+(e-0.5)*dx, binner[3],width=width,bottom=binner[0]+binner[1]+binner[2], color = "k", edgecolor="k", alpha = 0.999)
    # Include text
    ax.text(1+(e-0.5)*dx, binner[0]*0.5, "%.1f"%(binner[0]*100)+"%",ha='center',va='center')
    ax.text(1+(e-0.5)*dx, binner[1]*0.5+binner[0], "%.1f"%(binner[1]*100)+"%",ha='center',va='center')
    ax.text(1+(e-0.5)*dx, binner[2]*0.5+binner[1]+binner[0], "%.1f"%(binner[2]*100)+"%",ha='center',va='center')
    ax.text(1+(e-0.5)*dx, binner[3]*0.5+binner[2]+binner[1]+binner[0], "%.1f"%(binner[3]*100)+"%",ha='center',va='center',color="w")

# Generate additional stuff for the plot
ax.set_title("Global histograms for simulated data")
ax.set_ylabel("Fraction (-)")
ax.set_xticks([0.90, 1.10])
ax.set_xticklabels(["End-expiration","End-inspiration"])
ax.set_xlim((0.8,1.3))
ax.legend(handles=[natp,patp,atp,hitp])

# %%

if False:
    
    '''generate a histogram from the image based data'''
    
    
    binner = {'Exp':[],'Insp':[]}
    for state in ['Exp','Insp']:
        for compartment in ['NAT','PAT','AT','HIT']:
            binner[state] += [global_histograms['PIG6']['ARDSnet'][state][compartment]]
    
    
    # Create global figure
    fig, ax = plt.subplots(figsize=(6,8),dpi=200)
    width = 0.15
    dx = 0.20
    for e,binner in enumerate([binner['Exp'], binner['Insp']]):
        # Stack bars
        ax.bar(1+(e-0.5)*dx, binner[0],width=width, color = "w",edgecolor='k') 
        ax.bar(1+(e-0.5)*dx, binner[1],width=width,bottom=binner[0], color = "k", alpha = 0.25, edgecolor="k")
        ax.bar(1+(e-0.5)*dx, binner[2],width=width,bottom=binner[0]+binner[1], color="k", alpha = 0.5, edgecolor="k")
        ax.bar(1+(e-0.5)*dx, binner[3],width=width,bottom=binner[0]+binner[1]+binner[2], color = "k", edgecolor="k", alpha = 0.999)
        # Include text
        ax.text(1+(e-0.5)*dx, binner[0]*0.5, "%.1f"%(binner[0]*100)+"%",ha='center',va='center')
        ax.text(1+(e-0.5)*dx, binner[1]*0.5+binner[0], "%.1f"%(binner[1]*100)+"%",ha='center',va='center')
        ax.text(1+(e-0.5)*dx, binner[2]*0.5+binner[1]+binner[0], "%.1f"%(binner[2]*100)+"%",ha='center',va='center')
        ax.text(1+(e-0.5)*dx, binner[3]*0.5+binner[2]+binner[1]+binner[0], "%.1f"%(binner[3]*100)+"%",ha='center',va='center',color="w")

    # Generate additional stuff for the plot
    ax.set_title("Global histograms for image data")
    ax.set_ylabel("Fraction (-)")
    ax.set_xticks([0.90, 1.10])
    ax.set_xticklabels(["End-expiration","End-inspiration"])
    ax.set_xlim((0.8,1.3))
    ax.legend(handles=[natp,patp,atp,hitp])

# %% Generate regional histograms

type_histogram = "0" # "expiration" or "inspiration"

if type_histogram == "0":
    geometry_path = post_dir+"/"+post_files[0]
    mesh = io.read(geometry_path)

    xyz = mesh.points
    defmap = xyz + mesh.point_data['u']
    elem = mesh.cells_dict['tetra']
    porosity = mesh.point_data['Eulerian Porosity'] 

    M,_,_,_ = Tet4DefGradSmoothingOptiSPARSE(xyz, defmap, elem)
    deformed_state=False
    
    
elif type_histogram == "inspiration":
    
    geometry_path = post_dir+"/"+post_files[sampling_id]
    
    mesh = io.read(geometry_path)

    xyz = mesh.points
    defmap = xyz + mesh.point_data['u']
    elem = mesh.cells_dict['tetra']
    porosity = mesh.point_data['Eulerian Porosity'] 
    deformed_state=True
    
    M,_,_,_ = Tet4DefGradSmoothingOptiSPARSE(defmap, defmap, elem)
    


# Define the directions in use
dirs = {"BA" : np.mat([0.,0.,1.]).T, # BA tested; Direction checks out 
        "VD" : np.mat([0.,1.,0.]).T, # VD tested; Direction checks out
        "RL" : np.mat([1.,0.,0.]).T}

direction="VD"
nrois=10
verbose=False

# Weights for the histogram generation
w = np.abs(M)


gen_binner, counter = retrieve_regional_histogram(geometry_path,direction,deformed_state=deformed_state,
                                                  weights=w)
    


fig, ax = plt.subplots(figsize=(8,6),dpi=200)
width = 0.8
textsize=8

for roi in range(nrois):
    
    binner = gen_binner[roi]
    
    ax.bar(roi+(e-0.5)*dx, binner[0],width=width, color = "w",edgecolor='k') 
    ax.bar(roi+(e-0.5)*dx, binner[1],width=width,bottom=binner[0], color = "k", alpha = 0.25, edgecolor="k")
    ax.bar(roi+(e-0.5)*dx, binner[2],width=width,bottom=binner[0]+binner[1], color="k", alpha = 0.5, edgecolor="k")
    ax.bar(roi+(e-0.5)*dx, binner[3],width=width,bottom=binner[0]+binner[1]+binner[2], color = "k", edgecolor="k", alpha = 0.999)
    # Include text
    if binner[0]>0.025:
        ax.text(roi+(e-0.5)*dx, binner[0]*0.5, "%.1f"%(binner[0]*100)+"%",
            ha='center',va='center',size=textsize)
    if binner[1]>0.025:
        ax.text(roi+(e-0.5)*dx, binner[1]*0.5+binner[0], "%.1f"%(binner[1]*100)+"%",
            ha='center',va='center',size=textsize)
    
    if binner[2]>0.025:
        ax.text(roi+(e-0.5)*dx, binner[2]*0.5+binner[1]+binner[0], "%.1f"%(binner[2]*100)+"%",
            ha='center',va='center',size=textsize)
    
    if binner[3]>0.025:
        ax.text(roi+(e-0.5)*dx, binner[3]*0.5+binner[2]+binner[1]+binner[0], "%.1f"%(binner[3]*100)+"%",
            ha='center',va='center',color="w",size=textsize)

ax.set_title("Regional histogram")
ax.set_ylabel("Fraction (-)")
ax.set_xticks(range(10))
ax.set_xticklabels(np.arange(1,11))
ax.text(-1,-0.05,direction[0])
ax.text(10,-0.05,direction[1])
ax.set_xlabel("ROI#")

# %%

if False:
    # Load previously computed histogram data
    root = "C:/Users/angus/Downloads/CORNELLU-PIGS-GROUPED/"
    data = np.load(root+"histograms.npz",allow_pickle=True)
    holder = data['arr_0'].item().get('BA')['PIG6_ARDSnet']
    
    
    img_binner = {state:{roi:[] for roi in range(nrois)} for state in ["E","I"]}
    
    for state in ["E","I"]:
        for roi in range(nrois):
            for binn in ["NAT","PAT","AT","HIT"]:
                img_binner[state][roi] += [holder[state][binn][roi]]



# %%

    fig, ax = plt.subplots(figsize=(8,6),dpi=200)
    width = 0.8
    textsize=8
    direction="BA"
    state="I"
    for roi in range(nrois):
        
        binner = img_binner[state][roi]
        
        ax.bar(roi+(e-0.5)*dx, binner[0],width=width, color = "w",edgecolor='k') 
        ax.bar(roi+(e-0.5)*dx, binner[1],width=width,bottom=binner[0], color = "k", alpha = 0.25, edgecolor="k")
        ax.bar(roi+(e-0.5)*dx, binner[2],width=width,bottom=binner[0]+binner[1], color="k", alpha = 0.5, edgecolor="k")
        ax.bar(roi+(e-0.5)*dx, binner[3],width=width,bottom=binner[0]+binner[1]+binner[2], color = "k", edgecolor="k", alpha = 0.999)
        # Include text
        if binner[0]>0.025:
            ax.text(roi+(e-0.5)*dx, binner[0]*0.5, "%.1f"%(binner[0]*100)+"%",
                ha='center',va='center',size=textsize)
        if binner[1]>0.025:
            ax.text(roi+(e-0.5)*dx, binner[1]*0.5+binner[0], "%.1f"%(binner[1]*100)+"%",
                ha='center',va='center',size=textsize)
        
        if binner[2]>0.025:
            ax.text(roi+(e-0.5)*dx, binner[2]*0.5+binner[1]+binner[0], "%.1f"%(binner[2]*100)+"%",
                ha='center',va='center',size=textsize)
        
        if binner[3]>0.025:
            ax.text(roi+(e-0.5)*dx, binner[3]*0.5+binner[2]+binner[1]+binner[0], "%.1f"%(binner[3]*100)+"%",
                ha='center',va='center',color="w",size=textsize)
    
    ax.set_title("Regional histogram (Naive method)")
    ax.set_ylabel("Fraction (-)")
    ax.set_xticks(range(10))
    ax.set_xticklabels(np.arange(1,11))
    ax.text(-1,-0.05,direction[0])
    ax.text(10,-0.05,direction[1])
    ax.set_xlabel("ROI#")
