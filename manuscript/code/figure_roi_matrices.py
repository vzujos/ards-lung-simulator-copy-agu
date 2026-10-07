# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:10:50 2026

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
import meshio as io
import scipy.stats as sst

# These two libraries are not optimized and they have been inherited from
# previous projects.

import ROIAnalysis as roi
import BidimLib as bdl
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%

def plot_matrix(fig, ax, mat, field, state, vmin, vmax, 
                cmap="jet",
                ncolors=10,fs = 11,
                asterisks_coords=[], 
                norm=None, flip_vd=False,
                title="", colorbar=True):
        
        d, v = mat.returnSortedData(field,state)
        d = d.reshape(mat.nrois_rl, mat.nrois_vd, mat.nrois_ba).astype(float)
        v = v.reshape(mat.nrois_rl, mat.nrois_vd, mat.nrois_ba).astype(float)
        
        if flip_vd:
            d = np.flip(d,axis=1)
            v = np.flip(v,axis=1)
            len_vd = d.shape[1]
        
        if len(asterisks_coords)>0:
            for key in asterisks_coords:
                d[key] = np.nan
        
        cmap = plt.get_cmap(cmap, ncolors)

        ax.matshow(d[0,:,:], cmap = cmap, vmin=vmin, vmax=vmax)
        ax.set_xticklabels([]); ax.set_xticks([])
        ax.set_yticklabels([]); ax.set_yticks([])  
    
        # after your imshow/matshow call
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5, alpha=0.4)
        
        # Hide the minor tick marks themselves
        ax.tick_params(which="minor", bottom=False, left=False)
            
        trans = ax.get_xaxis_transform()
        ax.annotate("", xy=(-0.95,0.85), xytext=(-0.95, 0.15), xycoords=trans, 
                      arrowprops=dict(arrowstyle="<|-|>")) # VD - left side

        ax.annotate("B", xy= (-0.5, -0.10), xycoords=trans, size=fs, weight = 'bold')
        ax.annotate("A", xy= (9.0, -0.10), xycoords=trans, size=fs, weight = 'bold') #5x5
        ax.annotate("", xy=(0.0,-0.05), xytext=(9.0, -0.05), xycoords=trans, 
                          arrowprops=dict(arrowstyle="<|-|>")) # BA - below 

        ax.annotate("D", xy= (-1.35, 0.05), xycoords=trans, size=fs, weight = 'bold')
        ax.annotate("V", xy= (-1.35, 0.9), xycoords=trans, size=fs, weight = 'bold')
        
        ax.tick_params(
            which="minor",
            bottom=False,
            top=False,
            left=False,
            right=False,
        )

        if title != "":
            ax.set_title(title)

        if len(asterisks_coords)>0:
            for (kk,y,x) in asterisks_coords:
                
                if flip_vd: y = len_vd-1-y

                ax.annotate("X", xy=(x,y),size=fs-2,weight='bold',alpha=0.25,ha='center',va='center')
            
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.25)
                
        if norm is None:
            im = ax.imshow(d[0,:,:], cmap=cmap)
            cb = fig.colorbar(im, cax=cax, orientation='vertical')
            im.set_clim(vmin,vmax)
        
        if not colorbar:
            cb.ax.set_visible(False)
            
        return im
    
    
def plot_matrix2(fig, ax, mat, field, state, vmin, vmax, 
                cmap="jet",
                ncolors=10,fs = 11,
                asterisks_coords=[], 
                norm=None, flip_vd=False,
                title="", colorbar=True):
        
        d, v = mat.returnSortedData(field,state)
        d = d.reshape(mat.nrois_rl, mat.nrois_vd, mat.nrois_ba).astype(float)
        v = v.reshape(mat.nrois_rl, mat.nrois_vd, mat.nrois_ba).astype(float)
        
        if flip_vd:
            d = np.flip(d,axis=1)
            v = np.flip(v,axis=1)
            len_vd = d.shape[1]
        
        if len(asterisks_coords)>0:
            for key in asterisks_coords:
                d[key] = np.nan
        
        cmap = plt.get_cmap(cmap, ncolors)

        im = ax.matshow(d[0,:,:], cmap = cmap, vmin=vmin, vmax=vmax)
        ax.set_xticklabels([]); ax.set_xticks([])
        ax.set_yticklabels([]); ax.set_yticks([])  
    
        # after your imshow/matshow call
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5, alpha=0.4)
        
        # Hide the minor tick marks themselves
        ax.tick_params(which="minor", bottom=False, left=False)
            
        trans = ax.get_xaxis_transform()
        ax.annotate("", xy=(-0.95,0.85), xytext=(-0.95, 0.15), xycoords=trans, 
                      arrowprops=dict(arrowstyle="<|-|>")) # VD - left side

        ax.annotate("B", xy= (-0.5, -0.10), xycoords=trans, size=fs, weight = 'bold')
        ax.annotate("A", xy= (9.0, -0.10), xycoords=trans, size=fs, weight = 'bold') #5x5
        ax.annotate("", xy=(0.0,-0.05), xytext=(9.0, -0.05), xycoords=trans, 
                          arrowprops=dict(arrowstyle="<|-|>")) # BA - below 

        ax.annotate("D", xy= (-1.35, 0.05), xycoords=trans, size=fs, weight = 'bold')
        ax.annotate("V", xy= (-1.35, 0.9), xycoords=trans, size=fs, weight = 'bold')
        
        ax.tick_params(
            which="minor",
            bottom=False,
            top=False,
            left=False,
            right=False,
        )

        if title != "":
            ax.set_title(title)

        if len(asterisks_coords)>0:
            for (kk,y,x) in asterisks_coords:      
                if flip_vd: y = len_vd-1-y
                ax.annotate("X", xy=(x,y),size=fs-2,weight='bold',alpha=0.25,ha='center',va='center')
            
        return im

if False:
    
    fs=11
    flip_vd = False
    ov = overall_validity.reshape((10,10))
    keys = []
    for i in range(10):
        for j in range(10):
            if not ov[i,j]:
                keys += [(0,i,j)]
    
    fig, axes = plt.subplots(ncols=3,dpi=300,figsize=(9,3))
    subject = 4
    field = 'dgf'; state = ""
    if field == 'dgf':
        vmin, vmax = (0.0,0.10)
        cmap='Blues'
        ncolors=5
    elif field == 'eigf':
        vmin, vmax = (0.0,1.0)
        cmap='Greys'
        ncolors=5
        
    for ax, case in zip(axes,['Experiment','Simulation']):
        mat = bounds[case].matrices[subject]
        plot_matrix(fig,ax, mat,field,state,vmin,vmax,cmap=cmap,
                    ncolors=ncolors, asterisks_coords=keys,title="DGF | "+case,
                    flip_vd=flip_vd)
    
    mat = bounds['Simulation'].matrices[subject]
    field = 'abs_error_dgf'
    
    if field == 'abs_error_dgf':
        vmin,vmax=(0,0.04)
        cmap='Reds'
        title='Absolute error'
        ncolors=4

    elif field == 'rel_error_dgf':
        vmin,vmax=(0,100)
        cmap='Reds'
        title='Relative error (%)'    
        ncolors=5

    plot_matrix(fig,axes[2], mat,field,state,vmin,vmax,cmap=cmap,
                ncolors=ncolors, asterisks_coords=keys,title="DGF | "+title,
                flip_vd=flip_vd)   
    
    
    for ax,txt in zip(axes.flatten(),['(d)','(e)','(f)']):
        x0,x1=ax.get_xlim();dx=x1-x0
        y0,y1=ax.get_ylim();dy=y1-y0
        ax.text(x0-0.12*dx,y1+0.08*dy,txt,size=fs+2,weight='bold')
        
    plt.tight_layout()
    
    plt.savefig('./figures/representative-subject2.pdf',dpi=300, bbox_inches='tight')
    
# %%


if False:
    
    fs=11
    flip_vd = False
    ov = overall_validity.reshape((10,10))
    keys = []
    for i in range(10):
        for j in range(10):
            if not ov[i,j]:
                keys += [(0,i,j)]
    
    ncols=3; nrows=4
    fig, big_axes = plt.subplots(ncols=ncols,nrows=nrows, dpi=300,figsize=(3*ncols,3*nrows))
    subject = 4
    state = ""

    for e,field in enumerate(['nat','pat','at','hit']): 
        vmin, vmax = (0,100)
        cmap='Blues'
        ncolors=5
    
        axes = big_axes[e,:].flatten()
        
        for ax, case in zip(axes,['Experiment','Simulation']):
            mat = bounds[case].matrices[subject]
            plot_matrix(fig,ax, mat,field,state,vmin,vmax,cmap=cmap,
                        ncolors=ncolors, asterisks_coords=keys,title=field.upper()+" | "+case,
                        flip_vd=flip_vd)
        
        mat = bounds['Simulation'].matrices[subject]
      
        error_field = 'abs_error_'+field
        
        if error_field.split('_')[0] == 'abs':
            vmin,vmax=(0,20)
            cmap='Reds'
            title='Absolute error'
            ncolors=4
    
        elif error_field.split('_')[0] == 'rel':
            vmin,vmax=(0,100)
            cmap='Reds'
            title='Relative error (%)'    
            ncolors=5
    
        plot_matrix(fig,axes[2], mat,error_field,state,vmin,vmax,cmap=cmap,
                    ncolors=ncolors, asterisks_coords=keys,title=field.upper()+" | "+title,
                    flip_vd=flip_vd)   
        
        
       # for ax,txt in zip(axes.flatten(),['(d)','(e)','(f)']):
       #     x0,x1=ax.get_xlim();dx=x1-x0
       #     y0,y1=ax.get_ylim();dy=y1-y0
       #     ax.text(x0-0.12*dx,y1+0.08*dy,txt,size=fs+2,weight='bold')
            
        plt.tight_layout()
    
    plt.savefig('./figures/representative-subject3.pdf',dpi=300, bbox_inches='tight')

# %%
# 6x2 panel

from matplotlib.lines import Line2D

rename = {'pat':"Percentage of PAT (%)",
          'nat':"Percentage of NAT (%)",
          'at':"Percentage of AT (%)",
          'hit':"Percentage of HIT (%)",
          'dgf':"Delta Gas Fraction (-)",
          'eigf':"End-Inspiratory Gas Fraction (-)"
          }


if False:
    
    fs=11
    flip_vd = False
    ov = overall_validity.reshape((10,10))
    keys = []
    for i in range(10):
        for j in range(10):
            if not ov[i,j]:
                keys += [(0,i,j)]
    
    ncols=6; nrows=2
    fig, big_axes = plt.subplots(ncols=ncols,nrows=nrows, dpi=300,figsize=(3*ncols,3*nrows))
    subject = 4
    state = ""

    for e,field in enumerate(['eigf','dgf','nat','pat','at','hit']): 
        
        axes = big_axes[:,e].flatten()
        
        for ax, case in zip(axes,['Experiment','Simulation']):
            
            if field == 'dgf':
                vmin, vmax = (0,0.10)
                cmap='Blues'
                ncolors=5
            elif field == 'eigf':
                vmin, vmax = (0,1)
                cmap='Greys'
                ncolors=5
            else:
                vmin, vmax = (0,100)
                cmap='Reds'
                ncolors=5
            
                
            mat = bounds[case].matrices[subject]
            plot_matrix(fig,ax, mat,field,state,vmin,vmax,cmap=cmap,
                        ncolors=ncolors, asterisks_coords=keys,title=rename[field]+"\n"+case,
                        flip_vd=flip_vd)
        
        mat = bounds['Simulation'].matrices[subject]
      
     
            
    for ax,txt in zip(big_axes.flatten(),['(a)','(b)','(c)','(d)','(e)','(f)']):
        x0,x1=ax.get_xlim();dx=x1-x0
        y0,y1=ax.get_ylim();dy=y1-y0
        ax.text(x0-0.12*dx,y1+0.04*dy,txt,size=fs+2,weight='bold')

    plt.tight_layout()

    if True:
        for i in range(5):
            # Draw a vertical separator between columns 2 and 3
            # (i.e. after column index 2)
            right_ax = big_axes[0, i+1]
            
           # x = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2
            x0 = right_ax.get_position().x0 
            x1 = right_ax.get_position().x1
            dx = x1-x0
            x = x0-0.12*dx
            
            fig.add_artist(Line2D(
                [x, x], [0.05, 0.95],
                transform=fig.transFigure,
                color='black',
                linewidth=1, 
                alpha=.5
            ))    
        
     
    if False: plt.savefig('./figures/representative-subject4.pdf',dpi=300, bbox_inches='tight')

# %%

# 4x3 Panel

from matplotlib.lines import Line2D

rename = {'pat':"Percentage of PAT (%)",
          'nat':"Percentage of NAT (%)",
          'at':"Percentage of AT (%)",
          'hit':"Percentage of HIT (%)",
          'dgf':"Delta Gas Fraction (-)",
          'eigf':"End-Inspiratory Gas Fraction (-)"
          }


if True:
    
    
    targets = ['eigf','dgf','nat','pat','at','hit']
    
    fs=11
    flip_vd = False
    ov = overall_validity.reshape((10,10))
    keys = []
    for i in range(10):
        for j in range(10):
            if not ov[i,j]:
                keys += [(0,i,j)]
    
    ncols=4; nrows=3
    fig, big_axes = plt.subplots(ncols=ncols,nrows=nrows, dpi=300,figsize=(3*ncols,3*nrows))
    subject = 4
    state = ""

    big_axes = big_axes.reshape((6,2))
    ims = []
    
    for e,field in enumerate(targets): 
        
        axes = big_axes[e,:]
        
        for switch, (ax, case) in enumerate(zip(axes,['Experiment','Simulation'])):
            
            if field == 'dgf':
                vmin, vmax = (0,0.10)
                cmap='Blues'
                ncolors=5
            elif field == 'eigf':
                vmin, vmax = (0,1)
                cmap='Greys'
                ncolors=5
            else:
                vmin, vmax = (0,100)
                cmap='Reds'
                ncolors=5
            
            # Retrieve matrix structure
            mat = bounds[case].matrices[subject]
            
            # Plot matrix and keep 'image' structure for colorbars
            im = plot_matrix2(fig,ax, mat,field,state,vmin,vmax,cmap=cmap,
                    ncolors=ncolors, asterisks_coords=keys,title=case,
                    flip_vd=flip_vd)
            
            # Only store one every pair of matrices
            if switch==0:
                ims += [im]
              
    # Placing titles
    for e,(field,let) in enumerate(zip(targets,['(a)','(b)','(c)','(d)','(e)','(f)'])):
        
        ax0, ax1 = big_axes[e,:]
        left = ax0.get_position()
        right = ax1.get_position()
        
        x = (left.x0 + right.x1) / 2
        y = max(left.y1, right.y1) + 0.02
            
        fig.text(x, y,rename[field],
            ha="center", va="center",
            fontsize=12, weight='bold'
        )     

        fig.text(left.x0, y,let,
            ha="left", va="center",
            fontsize=12, weight='bold'
        )     
#    plt.tight_layout()

    # Place colorbars: Works perfectly
    for e,im in enumerate(ims):
        fig.colorbar(im, ax=big_axes[e,:], pad=0.02)

    if False:
        for i in range(5):
            # Draw a vertical separator between columns 2 and 3
            # (i.e. after column index 2)
            right_ax = big_axes[0, i+1]
            
           # x = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2
            x0 = right_ax.get_position().x0 
            x1 = right_ax.get_position().x1
            dx = x1-x0
            x = x0-0.12*dx
            
            fig.add_artist(Line2D(
                [x, x], [0.05, 0.95],
                transform=fig.transFigure,
                color='black',
                linewidth=1, 
                alpha=.5
            ))    
        
    if True: plt.savefig('./figures/representative-subject4.pdf',dpi=300, bbox_inches='tight')



# %% Directional test

if False:
    # QUESTION: How is the indexing and the data associated being plotted?
    # dummy matrix
    matrix_obj = bdl.matrix(0, 
                            nrois_ba=10,
                            nrois_vd=10, 
                            nrois_rl=1)
    # dummy frame
    
    frame = bdl.frame("", 0, "", path_to_dir = "", 
                      subject_rootname="%s",mesh_dir="",)
    
    # 1st index : L-R (untested)
    # 2nd index : V-D 
    # 3rd index:
    fig, axes = plt.subplots(ncols=2, figsize=(6,3),dpi=300)
    
    # VD test
    vd_data = np.zeros((1,10,10))
    for i in range(10):
        vd_data[0,0,:] = 5 # Ventral 
        vd_data[0,8:10,:] = 10 # Dorsal
                
    matrix_obj.load_slice(frame, 
                          'VD', 
                          vd_data, 
                          ov)
    
    plot_matrix(axes[0], matrix_obj,'VD',"",0,10,cmap='jet',
                ncolors=ncolors, asterisks_coords=keys,title="Ventro-Dorsal test")
    
    
    # BA test
    ba_data = np.zeros((1,10,10))
    for i in range(10):
        ba_data[0,:,0] = 5 # Basal 
        ba_data[0,:,8:10] = 10 # Apical
                
    matrix_obj.load_slice(frame, 
                          'BA', 
                          ba_data, 
                          ov)
    
    plot_matrix(axes[1], matrix_obj,'BA',"",0,10,cmap='jet',
                ncolors=ncolors, asterisks_coords=keys, title="Apico-Basal test")

# %% Second directional test: 
# QUESTION: How is out meshed data being processed and being plotted?
# By design, we will do the following
# Ventrodorsal..... V ~ -10    D ~ +10
# Apicobasal......  B~ -10    A ~ +10
# We'll see how is this shown in the matrix, and if it does not follow what has been stated, 
# we'll see how to make it fit it

if False:
    # matrices
    subject = 4; mtype="medium"
    mesh_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/'%(subject, mtype)
    sampled_mesh = mesh_path + "reg_anim_ready.vtu"
    
    
    M = io.read(sampled_mesh)
    xyz = M.points
    # From a view in paragraph the following information is inferred:
        # Z-axis: cephalocaudal or apicobasal axis; LOW Z => Diaphragm    HIGH Z => Apex, head, etc.
        # Y-axis: ventrodorsal axis; LOW Y => Ventral   HIGH Y => Dorsal
        # X-axis: left-to-right
        
        
    ymin = xyz[:,1].min(); ymax = xyz[:,1].max(); dy = ymax-ymin
    zmin = xyz[:,2].min(); zmax = xyz[:,2].max(); dz = zmax-zmin
    thr = 0.25
    
    # Generate ventrodorsally variable dummy data 
    dfield_vd = np.zeros_like(M.point_data['Delta Porosity'])
    dfield_vd[xyz[:,1]<ymin+thr*dy] =  -10 # Ventrally-associated data
    dfield_vd[xyz[:,1]>ymax-thr*dy] =  10 # Dorsally-associated data
    
    # Generate apicobasal dummy data
    dfield_ba = np.zeros_like(M.point_data['Delta Porosity'])
    dfield_ba[xyz[:,2]<zmin+thr*dz] =  -10 # Basal data, diaphragm
    dfield_ba[xyz[:,2]>zmax-thr*dz] =  10 # Apical data
    
    # Generate a paraview-compatible visualization
    M2 = io.Mesh(points=xyz, cells=M.cells_dict, point_data={"BA-data":dfield_ba,
                                                             "VD-data":dfield_vd,
                                                             "Mass":M.point_data['Mass']})
    M2.write(mesh_path+"orientation_test.vtu")
    
    # Trasnslate data to a numpy savefile
    voyager = {'xyz':M.points,
               'BA-data':dfield_ba,
               'VD-data':dfield_vd,
               'M':M.point_data['Mass']}
                          
    np.savez(mesh_path+"orientation_test.npz",**voyager)
    
    filename="orientation_test"
    
    # Create necessary data structures
    # Direction vectors
    ba = np.mat([0.,0.,1.]).T
    vd = np.mat([0.,1.,0.]).T
    rl = np.mat(np.cross(ba.T,vd.T)).T
    
    nrois_ba = 10
    nrois_vd = 10
    nrois_rl = 1
    data_shape = (nrois_rl,nrois_vd,nrois_ba)
    
    # Frame
    frame = bdl.frame("", subject, "", path_to_dir = mesh_path, 
                              subject_rootname="%s",mesh_dir="",)
    # Mesh handler
    mesh_obj = bdl.mesh(frame, 
                        npz_name='%s.npz'%filename,
                        vtk_name="%s.vtu"%filename,
                        minimum_binsize=5)
    
    # Load npz data    
    mesh_obj.load_npz()
    
    # Generate ids for bidirectional analysis
    mesh_obj.generate_ids(ba = ba.T, vd = vd.T, rl = rl.T, 
                          nrois_rl=nrois_rl,   
                          nrois_vd=nrois_vd, 
                          nrois_ba=nrois_ba)
        
    # Generate data structures
    mesh_obj.generate_matrix(block_ba=False)
    
    # Generate matrix data handler
    matrix_obj = bdl.matrix(mesh_obj.frame.number, 
                            nrois_ba=nrois_ba,
                            nrois_vd=nrois_vd, 
                            nrois_rl=nrois_rl)
    
    # Employ bidirectional analysis onto both fields
    for field in ["BA-data","VD-data"]:
                
        means, stds, masses, counts = mesh_obj.extract_matrix_data(field)
        # Load into object
        matrix_obj.load_slice(mesh_obj.frame, 
                              field, 
                              means, 
                              mesh_obj.matrix_validity)
    
    
    
    fig, axes = plt.subplots(ncols=2,figsize=(6,3),dpi=150)
    vmin=-10; vmax=10; cmap="jet"
    ncolors=5
    
    for ax, field in zip(axes,["BA-data","VD-data"]):
        
        plot_matrix(fig, ax, matrix_obj,field,state,vmin,vmax,cmap=cmap,
                    ncolors=ncolors, asterisks_coords=keys,title=field,
                    flip_vd=False)
        
    # Visualization succeeded; now evaluating
    # B = -10; A = 10
    # V = 10; D = -10

# %%

# Broad definitions
# Direction vectors
ba = np.mat([0.,0.,1.]).T
vd = np.mat([0.,1.,0.]).T
rl = np.mat(np.cross(ba.T,vd.T)).T

nrois_ba = 10
nrois_vd = 10
nrois_rl = 1
data_shape = (nrois_rl,nrois_vd,nrois_ba)

# group them
dirs = {"vd":vd,"ba":ba,"rl":rl}

# number of rois
nro_rois = {"rl":nrois_rl, "vd":nrois_vd, "ba":nrois_ba}
    
fields_to_load = ['dgf','eegf','eigf']

subjmesh = {5:'medium',4:'medium',2:'medium-fine',3:'medium-fine',6:'medium-fine'}
sim_path = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"

bounds = {"Experiment":bdl.bound(),
          "Simulation":bdl.bound()
          }

overall_validity = np.ones(data_shape,dtype=bool)

for subject in [2,3,4,5,6]:
    
    # Adjust path to simulated data
    mtype = subjmesh[subject]
    case = "PIG%i-%s-per/"%(subject,mtype)
    
    
    # Path to wherever the meshes are 
    mesh_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/'%(subject, mtype)
    
    
    for state, filename in zip(['Experiment','Simulation'],
                               ['reg_anim_ready', 'sim_anim_ready']):
        
        # store a point field whose integer values point towards a roi id
        key_data = []
    
        # Generate a visualization for the directional analysis
        if False:
            reg_mesh = mesh_path + 'reg_anim_ready.vtu'
            
            # evaluate both directions
            for direction in dirs:
                # compute the rois
                _, keys = roi.ISO_ROIAnalysis(reg_mesh,'Delta Porosity', nro_rois[direction], 
                                              dirs[direction])
                key_data += [keys]
           
            mesh = io.read(reg_mesh)
            mesh.point_data.update({"ROI_ID_VD":key_data[0],
                                          "ROI_ID_BA":key_data[1],
                                          "ROI_ID_RL":key_data[2],})
            
            mesh.write(mesh_path+"orientation_test.vtu")

        mesh = io.read(mesh_path+filename+'.vtu')
        # Generate old format 'npz' file
        npz_path = mesh_path+filename+".npz"
        
        voyager = {'xyz':mesh.points,
                   'dgf':mesh.point_data['Delta Porosity'],
                   'eegf':mesh.point_data['End-Expiratory Porosity'],
                   'eigf':mesh.point_data['End-Inspiratory Porosity'],
                   'M':mesh.point_data['Mass']}
                      
        np.savez(npz_path,**voyager)

        frame = bdl.frame("", subject, "", path_to_dir = mesh_path, 
                          subject_rootname="%s",mesh_dir="",)
        
        mesh_obj = bdl.mesh(frame, 
                     npz_name='%s.npz'%filename,
                     vtk_name="%s.vtu"%filename,
                     minimum_binsize=5)
    
        mesh_obj.load_npz()
        
        mesh_obj.generate_ids(ba = ba.T, vd = vd.T, rl = rl.T, 
                              nrois_rl=nrois_rl,   
                              nrois_vd=nrois_vd, 
                              nrois_ba=nrois_ba)
    
        mesh_obj.generate_matrix(block_ba=False)
    
        overall_validity = np.logical_and(overall_validity, 
                                          mesh_obj.matrix_validity)
        
        matrix_obj = bdl.matrix(mesh_obj.frame.number, 
                                nrois_ba=nrois_ba,
                                nrois_vd=nrois_vd, 
                                nrois_rl=nrois_rl)
        
        
        for field in fields_to_load:
            
            means, stds, masses, counts = mesh_obj.extract_matrix_data(field)
            
            if field in ['dgf']:
                
                if state == 'Experiment':
                    means_exp = means
                else:
                    abs_error = np.abs(means-means_exp)
                    rel_error = np.abs(abs_error/means_exp)*100
                    
                    matrix_obj.load_slice(mesh_obj.frame, 
                                          'rel_error_'+field, 
                                          rel_error, 
                                          mesh_obj.matrix_validity)
                    
                    matrix_obj.load_slice(mesh_obj.frame, 
                                          'abs_error_'+field, 
                                          abs_error, 
                                          mesh_obj.matrix_validity)
            
            
            matrix_obj.load_slice(mesh_obj.frame, 
                                  field, 
                                  means, 
                                  mesh_obj.matrix_validity)
        
        compartments = mesh_obj.extract_matrix_aeration_compartments(use_mass=True)
        
        if state == 'Experiment':
                compartments0 = compartments.copy()
                
        for comp in ['nat','pat','at','hit']:
            
            matrix_obj.load_slice(mesh_obj.frame, 
                                  comp, 
                                  np.array(compartments[comp])*100, 
                                  mesh_obj.matrix_validity)

            if state == 'Simulation':
                
                abs_error = 100*np.abs(np.array(compartments[comp])-np.array(compartments0[comp]))
                rel_error = np.abs(abs_error/np.array(compartments0[comp]))*100

                matrix_obj.load_slice(mesh_obj.frame, 
                                      'rel_error_'+comp, 
                                      rel_error, 
                                      mesh_obj.matrix_validity)
                
                matrix_obj.load_slice(mesh_obj.frame, 
                                      'abs_error_'+comp, 
                                      abs_error, 
                                      mesh_obj.matrix_validity)
        
        bounds[state].load(matrix_obj)
        
# %%
        matrix_obj = bdl.matrix(mesh_obj.frame.number, 
                                nrois_ba=nrois_ba,
                                nrois_vd=nrois_vd, 
                                nrois_rl=nrois_rl)

# %% generate the data manager for statistics

# create full list of indices
indices = [(i,j,k) for i in range(nrois_rl) for j in range(nrois_vd) for k in range(nrois_ba)]
# declare a filter

data_manager = {field:{"Experiment":{},"Simulation":{}} for field in fields_to_load+["abs_error_dgf","rel_error_dgf",'nat','pat','at','hit']}


for field in ["dgf","eegf","eigf","abs_error_dgf","rel_error_dgf",'nat','at','pat','hit']:
    
    for lung in ['Experiment','Simulation']:    
        
        if field == 'rel_error_dgf' and lung == "Experiment": continue
        if field == 'abs_error_dgf' and lung == "Experiment": continue

        b = bounds[lung]
        subjs = list(b.matrices.keys())
        
        for subj in subjs:
            
            # select the matrix
            M = b.matrices[subj]
            # retrieve data
            d, _ =  M.returnSortedData(field,"")
            # we must have a valid global validity matrix from the prev block
            d = d.reshape(data_shape)
            
            # set everything invalid as np.nan
            d[np.logical_not(overall_validity)] = np.nan
            
            # go through every index
            for i,j,k in indices:
                    
                # If blocked by mask, continue with next iterator
                if not overall_validity[i,j,k]:
                    continue
                else:
                        
                    # Create bin if it didn't exist previously
                    if (i,j,k) not in data_manager[field][lung].keys():
                        data_manager[field][lung].update({(i,j,k):[]})
                        
                    data_manager[field][lung][(i,j,k)] += [d[i,j,k]]

# %%
# trick to add overall group means as if they were subjects in a bound
lung_to_number={"Experiment":0,"Simulation":1,}
pooler = {field:{state_i:{pos:[] for pos in ["v","c","d"]} for state_i in [0,1]} for field in fields_to_load+["abs_error_dgf","rel_error_dgf",'nat','pat','at','hit']}

gb_median = bdl.bound()

for e,state in enumerate(["Experiment","Simulation"]):
           
    if not overall_validity.shape == data_shape:
        raise Exception("Shape issue!")

    # create a new frame (keyword, number, state=="")
    frame = bdl.frame(state, 
                      lung_to_number[state], 
                      "", path_to_dir = "")
    
    
    # For every field
    for field in fields_to_load+["abs_error_dgf","rel_error_dgf",'nat','pat','at','hit']:
        
        
        if field == 'rel_error_dgf' and state == "Experiment": continue
        if field == 'abs_error_dgf' and state == "Experiment": continue
        
        # Create a full matrix to hold the data
        data = np.full(data_shape,np.nan)
        # For every indexing key
        for key in indices:
            
            # Retrieve individual items
            (i,j,k) = key
            # Avoid thoses deemed invalid
            if not overall_validity[i,j,k]: continue
            # Retrieve the individual points and apply mean
            points = data_manager[field][state][key]
            data[key] = np.mean(points)
            if j<3: 
                pos="v"
            elif j<7:
                pos="c"
            else:
                pos="d"
            pooler[field][e][pos] += list(points)
            
        # If the 'State', 'Experiment' or 'Simulation' has not been created, do it
        if lung_to_number[state] not in gb_median.matrices:
            gM = bdl.matrix(frame.number, 
                            nrois_ba=nrois_ba,
                            nrois_vd=nrois_vd, 
                            nrois_rl=nrois_rl)
        
        # Load the recently computed data as a new slice within the matrix
        gM.load_slice(frame, field, data, overall_validity)
        print(field, np.mean(data[overall_validity]))
        # Save the load the matrix into the bound
        gb_median.load(gM)
        
 #   gM.plot('dgf',"", cmap='Blues',vmin=0.0,vmax=0.10) # field, state ("")
#
# %%
field='hit'
bounds['Simulation'].matrices[4].plot(field,frame.state, cmap='Blues',vmin=0.0,vmax=0.10)
bounds['Experiment'].matrices[4].plot(field,frame.state, cmap='Blues',vmin=0.0,vmax=0.10)

# %%

gb_median.matrices[0].plot('dgf',"", cmap='Blues',vmin=0.0,vmax=0.10) # field, state ("")
gb_median.matrices[1].plot('dgf',"", cmap='Blues',vmin=0.0,vmax=0.10) # field, state ("")

# %%

field = 'dgf';pos='d'
field = "nat"

x = np.asarray(np.hstack([pooler[field][0][pos] for pos in ['v','c','d']]))
y = np.asarray(np.hstack([pooler[field][1][pos] for pos in ['v','c','d']]))

# Scatter plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, alpha=0.7)

# Identity line
maxval = np.max([x.max(), y.max()])
plt.plot([0, maxval], [0, maxval],
         'k--', alpha=0.5, label='Identity')

# Linear regression
reg = sst.linregress(x, y)

slope = reg.slope
intercept = reg.intercept
rvalue = reg.rvalue
pvalue_reg = reg.pvalue
stderr = reg.stderr

# Regression line
xx = np.linspace(0, maxval, 200)
yy = intercept + slope*xx

plt.plot(xx, yy, 'r',
         label=f'y = {slope:.3f}x + {intercept:.3f}')

# Pearson correlation
pearson_r, pearson_p = sst.pearsonr(x, y)

plt.xlabel("Experimental $\Delta$GF")
plt.ylabel("Simulated $\Delta$GF")
plt.axis('equal')
plt.legend()

plt.show()

print("Linear regression")
print(f"  slope      = {slope:.5f}")
print(f"  intercept  = {intercept:.5f}")
print(f"  R²         = {rvalue**2:.5f}")
print(f"  p-value    = {pvalue_reg:.4e}")
print(f"  std. error = {stderr:.5f}")

print()

print("Pearson correlation")
print(f"  r       = {pearson_r:.5f}")
print(f"  p-value = {pearson_p:.4e}")

# %%

# %%
from scipy.stats import linregress, pearsonr

fs = 11
field ='eigf'

# Experimental and simulated pooled data
xdata = pooler[field][0]
ydata = pooler[field][1]

fig, (ax_ba, ax_scatter) = plt.subplots(
    1, 2,
    figsize=(6.0, 3.0),
    dpi=300
)

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.18,
    top=0.92,
    wspace=0.35
)

# ------------------------------------------------------------------
# Bland–Altman
# ------------------------------------------------------------------

xdata_ = np.array(xdata['c']+xdata['v']+xdata['d'])
ydata_ = np.array(ydata['c']+ydata['v']+ydata['d'])

mean_values = 0.5 * (xdata_ + ydata_)
diff_values = ydata_ - xdata_

mean_diff = np.mean(diff_values)
std_diff = np.std(diff_values, ddof=1)

loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff

ax_ba.scatter(mean_values,
              diff_values,
              marker='x',
              color='royalblue')

ax_ba.axhline(mean_diff,
              color='k',
              lw=1.2)

ax_ba.axhline(loa_upper,
              color='gray',
              ls='--',
              lw=1.0)

ax_ba.axhline(loa_lower,
              color='gray',
              ls='--',
              lw=1.0)

# Symmetric y limits
ylim = np.max(np.abs(ax_ba.get_ylim()))
ax_ba.set_ylim(-ylim, ylim)

# Annotate
x0, x1 = ax_ba.get_xlim()
y0, y1 = ax_ba.get_ylim()

ax_ba.text(x1,
           mean_diff,
           f"{mean_diff:.4f}",
           ha='right',
           va='bottom',
           alpha=0.6)

ax_ba.text(x1,
           loa_upper,
           f"{loa_upper:.4f}",
           ha='right',
           va='bottom',
           alpha=0.6)

ax_ba.text(x1,
           loa_lower,
           f"{loa_lower:.4f}",
           ha='right',
           va='bottom',
           alpha=0.6)

ax_ba.set_xlabel("Mean of Experiment and Simulation")
ax_ba.set_ylabel("Simulation − Experiment")
ax_ba.set_title("Bland–Altman", weight='bold')

ax_ba.spines['top'].set_visible(False)
ax_ba.spines['right'].set_visible(False)

# ------------------------------------------------------------------
# Scatter plot
# ------------------------------------------------------------------

reg = linregress(xdata, ydata)

pearson_r, pearson_p = pearsonr(xdata, ydata)

xmin = min(xdata.min(), ydata.min())
xmax = max(xdata.max(), ydata.max())

xfit = np.linspace(xmin, xmax, 200)
yfit = reg.slope*xfit + reg.intercept

ax_scatter.scatter(
    xdata,
    ydata,
    marker='x',
    color='royalblue'
)

ax_scatter.plot(
    xfit,
    yfit,
    color='k',
    lw=1.5,
)

ax_scatter.plot(
    [xmin, xmax],
    [xmin, xmax],
    '--',
    color='gray',
    alpha=0.5,
)

ax_scatter.set_xlim(xmin, xmax)
ax_scatter.set_ylim(xmin, xmax)

ax_scatter.set_xlabel("Experiment")
ax_scatter.set_ylabel("Simulation")
ax_scatter.set_title("Regional $\Delta$GF", weight='bold')

ax_scatter.spines['top'].set_visible(False)
ax_scatter.spines['right'].set_visible(False)

# Statistics annotation
annotation = (
    f"$r$ = {pearson_r:.3f}\n"
    f"$p$ = {pearson_p:.2e}\n"
    f"$y$ = {reg.slope:.3f}$x$ + {reg.intercept:.3f}"
)

ax_scatter.text(
    0.03,
    0.97,
    annotation,
    transform=ax_scatter.transAxes,
    va='top',
    bbox=dict(fc='white', ec='0.8')
)

# Panel labels
ax_ba.text(
    -0.15,
    1.05,
    "(a)",
    transform=ax_ba.transAxes,
    fontsize=fs+2,
    fontweight='bold'
)

ax_scatter.text(
    -0.15,
    1.05,
    "(b)",
    transform=ax_scatter.transAxes,
    fontsize=fs+2,
    fontweight='bold'
)

plt.tight_layout()

# %%

fields = ['eigf','dgf']

from scipy.stats import linregress, pearsonr
from matplotlib.lines import Line2D

colors = {
    "v": "slategrey",
    "c": "royalblue",
    "d": "midnightblue"
}

markers = {
    "v": ".",
    "c": "+",
    "d": "x"
}

renamer = {"v":"Ventral",
           "c":"Central",
           "d":"Dorsal",
           'dgf':'Delta gas fraction',
           'eigf':'End-inspiratory\ngas fraction'}


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

img_keys = {(0,0):'(a)',(0,1):'(b)',
        (1,0):'(c)',(1,1):'(d)',
        (2,0):'(e)',(2,1):'(f)',
        (3,0):'(g)',(3,1):'(h)'}

fs = 11

fig, axes = plt.subplots(
    nrows=len(fields),
    ncols=2,
    figsize=(3.5*2, 2.5*len(fields)),
    dpi=300
)

fig.subplots_adjust(
    left=0.10,
    right=0.80,
    bottom=0.08,
    top=0.95,
    wspace=0.35,
    hspace=0.50
)

if len(fields_to_load) == 1:
    axes = np.array([axes])

for ai, field in enumerate(fields):

    ax_ba = axes[ai, 0]
    ax_sc = axes[ai, 1]

    # ------------------------------------------------------------
    # reconstruct pooled vectors (region-aware)
    # ------------------------------------------------------------
    xdata = []
    ydata = []
    posdata = []

    for state_i, state in enumerate(["Experiment", "Simulation"]):

        for pos in ["v", "c", "d"]:

            xvals = pooler[field][0][pos]
            yvals = pooler[field][1][pos]

            n = min(len(xvals), len(yvals))
            xvals = np.asarray(xvals[:n])
            yvals = np.asarray(yvals[:n])

            xdata.append(xvals)
            ydata.append(yvals)
            posdata.append(np.full(n, pos))

    xdata = np.concatenate(xdata)
    ydata = np.concatenate(ydata)
    posdata = np.concatenate(posdata)

    # ============================================================
    # Scatter plot
    # ============================================================
    for pos in ["v", "c", "d"]:
        mask = posdata == pos
        ax_sc.scatter(
            xdata[mask],
            ydata[mask],
            color=colors[pos],
            marker=markers[pos],
            alpha=0.75
        )

    slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)

    xfit = np.linspace(xdata.min(), xdata.max(), 100)
    yfit = slope * xfit + intercept

    ax_sc.plot(xfit, yfit, color='k', lw=1.5)

    ax_sc.plot(
        [xdata.min(), xdata.max()],
        [xdata.min(), xdata.max()],
        '--',
        color='gray',
        alpha=0.4
    )

    ax_sc.set_title(renamer[field], weight='bold', size=fs)
    ax_sc.set_xlabel("Experiment",size=fs-1)
    ax_sc.set_ylabel("Simulation",size=fs-1)

    text = pvalue_stars(p_value)
    
    if field == 'eigf':
        ax_sc.text(
            0.03, 0.97,
            f"r = {r_value:.2f}"+text,
            transform=ax_sc.transAxes,
            va='top',
        )
    else:
        ax_sc.text(
            0.58, 0.97,
            f"r = {r_value:.2f}"+text,
            transform=ax_sc.transAxes,
            va='top',
        )

    # ============================================================
    # Bland–Altman
    # ============================================================
    mean_vals = 0.5 * (xdata + ydata)
    diff_vals = ydata - xdata

    md = np.mean(diff_vals)
    sd = np.std(diff_vals, ddof=1)

    loa_u = md + 1.96 * sd
    loa_l = md - 1.96 * sd

    for pos in ["v", "c", "d"]:
        mask = posdata == pos
        ax_ba.scatter(
            mean_vals[mask],
            diff_vals[mask],
            color=colors[pos],
            marker=markers[pos],
            alpha=0.75
        )

    ax_ba.axhline(md, color='k', lw=1.2)
    ax_ba.axhline(loa_u, color='gray', ls='--', lw=1.0)
    ax_ba.axhline(loa_l, color='gray', ls='--', lw=1.0)

    ax_ba.set_xlabel("Mean of Experiment and Simulation",size=fs-1)
    ax_ba.set_ylabel("Simulation − Experiment",size=fs-1)
    ax_ba.set_title(renamer[field], weight='bold', size=fs)

    ax_ba.tick_params(axis='y', labelsize=fs-1)
    ax_ba.tick_params(axis='x', labelsize=fs-1)       
    ax_sc.tick_params(axis='y', labelsize=fs-1)       
    ax_sc.tick_params(axis='x', labelsize=fs-1)       

    
    align='right'
    if field == 'eigf':
        ax_ba.set_xlim((0.0,1.0))
        ax_sc.set_xlim((0.0,1.0))
        ax_sc.set_ylim((0.0,1.0))
        ax_ba.set_ylim(-0.2, 0.2)

        
        y0,y1 = ax_ba.get_ylim(); dy = y1-y0; ddy=dy*0.02
        x0,x1 = ax_ba.get_xlim(); dx = x1-x0; 
        
        ddx=dx*0.17; align='right'
        ax_ba.text(x0+ddx, md-7*ddy, "%.2f"%md,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_u+ddy, "%.2f"%loa_u,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_l-4*ddy, "%.2f"%loa_l,ha=align,alpha=0.5)
    elif field == 'dgf':
        yabs=0.12
        ax_ba.set_xlim((-0.05,0.20))
        ax_sc.set_xlim((-0.10,0.2))
        ax_sc.set_ylim((-0.00,0.2))
        ax_ba.set_ylim(-yabs, yabs)

        
        y0,y1 = ax_ba.get_ylim(); dy = y1-y0; ddy=dy*0.02
        x0,x1 = ax_ba.get_xlim(); dx = x1-x0; 

        ddx=dx*0.97; align='right'
        ax_ba.text(x0+ddx, md+ddy, "%.2f"%md,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_u+ddy, "%.2f"%loa_u,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_l+ddy, "%.2f"%loa_l,ha=align,alpha=0.5)


    # ------------------------------------------------------------
    # region legend (shared style)
    # ------------------------------------------------------------
        # Region legend
    region_handles = [
            Line2D([], [], color=colors['v'], marker=markers['v'], linestyle='None', label='Ventral'),
            Line2D([], [], color=colors['c'], marker=markers['c'], linestyle='None', label='Central'),
            Line2D([], [], color=colors['d'], marker=markers['d'], linestyle='None', label='Dorsal')
    ]
    
    legend_regions = ax_sc.legend(
                handles=region_handles,
                title='Region',
                frameon=False,
                fontsize=8,
                loc='lower right')

    if field == 'eigf':
        ax_sc.legend(handles=region_handles, frameon=False, loc='lower right',fontsize=8)
    else:
        ax_sc.legend(handles=region_handles, frameon=False, loc='upper left',fontsize=8)

for i in range(2):
    for j in range(2):
        ax = axes[i,j]
        x0,x1 = ax.get_xlim(); dx=x1-x0
        y0,y1 = ax.get_ylim(); dy=y1-y0
            #ax.text(x0-0.1*dx,y1+0.08*dy,keys[(i,j)], size=fs+2,weight='bold')
        ax.text(x0-0.035*dx,y1+0.045*dy,img_keys[(i,j)], size=fs+2,weight='bold')
        
plt.savefig('./figures/reg2D-ventilation-panel.pdf',dpi=300, bbox_inches='tight')

# %%


fields = ['nat','pat','at','hit']

from scipy.stats import linregress, pearsonr
from matplotlib.lines import Line2D

colors = {
    "v": "slategrey",
    "c": "royalblue",
    "d": "midnightblue"
}

markers = {
    "v": ".",
    "c": "+",
    "d": "x"
}

renamer = {"v":"Ventral",
           "c":"Central",
           "d":"Dorsal",
           'dgf':'Delta gas fraction',
           'eigf':'End-inspiratory\ngas fraction',
           'nat':'Percentage of NAT (%)',
           'pat':'Percentage of PAT (%)',
           'at':'Percentage of AT (%)',
           'hit':'Percentage of HIT (%)'}


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

img_keys = {(0,0):'(a)',(0,1):'(b)',
        (1,0):'(c)',(1,1):'(d)',
        (2,0):'(e)',(2,1):'(f)',
        (3,0):'(g)',(3,1):'(h)'}

fs = 11

fig, axes = plt.subplots(
    nrows=len(fields),
    ncols=2,
    figsize=(3.5*2, 2.5*len(fields)),
    dpi=300
)

fig.subplots_adjust(
    left=0.10,
    right=0.80,
    bottom=0.08,
    top=0.95,
    wspace=0.35,
    hspace=0.50
)

if len(fields_to_load) == 1:
    axes = np.array([axes])

for ai, field in enumerate(fields):

    ax_ba = axes[ai, 0]
    ax_sc = axes[ai, 1]

    # ------------------------------------------------------------
    # reconstruct pooled vectors (region-aware)
    # ------------------------------------------------------------
    xdata = []
    ydata = []
    posdata = []

    for state_i, state in enumerate(["Experiment", "Simulation"]):

        for pos in ["v", "c", "d"]:

            xvals = pooler[field][0][pos]
            yvals = pooler[field][1][pos]

            n = min(len(xvals), len(yvals))
            xvals = np.asarray(xvals[:n])
            yvals = np.asarray(yvals[:n])

            xdata.append(xvals)
            ydata.append(yvals)
            posdata.append(np.full(n, pos))

    xdata = np.concatenate(xdata)*100
    ydata = np.concatenate(ydata)*100
    posdata = np.concatenate(posdata)

    # ============================================================
    # Scatter plot
    # ============================================================
    for pos in ["v", "c", "d"]:
        mask = posdata == pos
        ax_sc.scatter(
            xdata[mask],
            ydata[mask],
            color=colors[pos],
            marker=markers[pos],
            alpha=0.75
        )

    slope, intercept, r_value, p_value, _ = linregress(xdata, ydata)

    xfit = np.linspace(xdata.min(), xdata.max(), 100)
    yfit = slope * xfit + intercept

    ax_sc.plot(xfit, yfit, color='k', lw=1.5)

    ax_sc.plot(
        [xdata.min(), xdata.max()],
        [xdata.min(), xdata.max()],
        '--',
        color='gray',
        alpha=0.4
    )

    ax_sc.set_title(renamer[field], weight='bold', size=fs)
    ax_sc.set_xlabel("Experiment",size=fs-1)
    ax_sc.set_ylabel("Simulation",size=fs-1)

    text = pvalue_stars(p_value)
    
    ax_sc.text(
            0.08, 0.97,
            f"r = {r_value:.2f}"+text,
            transform=ax_sc.transAxes,
            va='top',)

    # ============================================================
    # Bland–Altman
    # ============================================================
    mean_vals = 0.5 * (xdata + ydata)
    diff_vals = ydata - xdata

    md = np.mean(diff_vals)
    sd = np.std(diff_vals, ddof=1)

    loa_u = md + 1.96 * sd
    loa_l = md - 1.96 * sd

    for pos in ["v", "c", "d"]:
        mask = posdata == pos
        ax_ba.scatter(
            mean_vals[mask],
            diff_vals[mask],
            color=colors[pos],
            marker=markers[pos],
            alpha=0.75
        )

    ax_ba.axhline(md, color='k', lw=1.2)
    ax_ba.axhline(loa_u, color='gray', ls='--', lw=1.0)
    ax_ba.axhline(loa_l, color='gray', ls='--', lw=1.0)

    ax_ba.set_xlabel("Mean of Experiment and Simulation",size=fs-1)
    ax_ba.set_ylabel("Simulation − Experiment",size=fs-1)
    ax_ba.set_title(renamer[field], weight='bold', size=fs)

    ax_ba.tick_params(axis='y', labelsize=fs-1)
    ax_ba.tick_params(axis='x', labelsize=fs-1)       
    ax_sc.tick_params(axis='y', labelsize=fs-1)       
    ax_sc.tick_params(axis='x', labelsize=fs-1)       

    
    align='right'
    ax_ba.set_xlim((0.0,100))
    ax_sc.set_xlim((0.0,100))
    ax_sc.set_ylim((0.0,100))
    ax_ba.set_ylim(-30, 30)

        
    y0,y1 = ax_ba.get_ylim(); dy = y1-y0; ddy=dy*0.02
    x0,x1 = ax_ba.get_xlim(); dx = x1-x0; 
    
    if field in ['nat','pat','hit']:
        ddx=dx*0.98; align='right'
        ax_ba.text(x0+ddx, md+1*ddy, "%.0f"%md,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_u+ddy, "%.0f"%loa_u,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_l-4*ddy, "%.0f"%loa_l,ha=align,alpha=0.5)
    else:
        ddx=dx*0.12; align='right'
        ax_ba.text(x0+ddx, md+1*ddy, "%.0f"%md,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_u+ddy, "%.0f"%loa_u,ha=align,alpha=0.5)
        ax_ba.text(x0+ddx, loa_l-4*ddy, "%.0f"%loa_l,ha=align,alpha=0.5)
        
    # ------------------------------------------------------------
    # region legend (shared style)
    # ------------------------------------------------------------
        # Region legend
    region_handles = [
            Line2D([], [], color=colors['v'], marker=markers['v'], linestyle='None', label='Ventral'),
            Line2D([], [], color=colors['c'], marker=markers['c'], linestyle='None', label='Central'),
            Line2D([], [], color=colors['d'], marker=markers['d'], linestyle='None', label='Dorsal')
    ]
    
    legend_regions = ax_sc.legend(
                handles=region_handles,
                title='Region',
                frameon=False,
                fontsize=8,
                loc='lower right')

 #   if field == 'eigf':
    ax_sc.legend(handles=region_handles, frameon=False, loc='lower right',fontsize=8)
  #  else:
   #     ax_sc.legend(handles=region_handles, frameon=False, loc='upper left',fontsize=8)
#
for i in range(4):
    for j in range(2):
        ax = axes[i,j]
        x0,x1 = ax.get_xlim(); dx=x1-x0
        y0,y1 = ax.get_ylim(); dy=y1-y0
            #ax.text(x0-0.1*dx,y1+0.08*dy,keys[(i,j)], size=fs+2,weight='bold')
        ax.text(x0-0.035*dx,y1+0.045*dy,img_keys[(i,j)], size=fs+2,weight='bold')
        
plt.savefig('./figures/reg2D-aercomp-panel.pdf',dpi=300, bbox_inches='tight')

        
plt.savefig('./figures/reg2D-aercomp-panel.pdf',dpi=300, bbox_inches='tight')
