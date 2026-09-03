# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:24:42 2025

@author: angus
"""


# %% 

import numpy as np
import meshio as io
import matplotlib.pyplot as plt
import pyvista as pv

# %%

def extract_surface(in_mesh, out_mesh,save=True, field="u"):
    
    '''
    Takes in a tetrahedral mesh and returns a surface triangular mesh.
    '''
    
    
    # Load tetrahedral mesh
    tet_mesh = pv.read(in_mesh)
     
    # Extract a surface mesh
    surf_mesh = tet_mesh.extract_surface()
     
    # Set the plotted field to be the displacement field
    surf_mesh.point_data.active_scalars_name=field
    
    # Point cloud
    points = surf_mesh.points
    
    # Displacement field
    disp_field = surf_mesh.get_array(field)
    
    # Generate normals
    norm = surf_mesh.compute_normals()
    norm.point_data.active_scalars_name="Normals"
    
    normals = surf_mesh.compute_normals()
    normal_data = normals.point_data['Normals']    
    
    
    # Extract cell dict
    ien = []
    for cell in surf_mesh.cell:
        ien += [cell.point_ids]
        
    # Write surface mesh to file
    Mesh = io.Mesh(points=points,
                   cells={"triangle":ien},
                   point_data={field:disp_field, 
                               "Normals":normal_data})
    
    if save: Mesh.write(out_mesh)
    
    return Mesh

def isolate_diaphragm(in_mesh, out_mesh, z_threshold=0.23, 
                      nz_cutoff=-0.4, save=True):
    
    '''
    Determines the surface that defines the diaphragm and generates
    the geometries for visualization and data management.
    '''

    Mesh = io.read(in_mesh)
    Mesh.point_data['Normals']
    save=True
       
    # Retrieve Z-bounds
    zmax = np.max(Mesh.points[:,2]);  zmin = np.min(Mesh.points[:,2])
       
    # Generate a threshold
    z33 = (zmax-zmin)*z_threshold + zmin
         
    # Generate a mask that only allows for the lower quarter to be seen
    lower_mask = (Mesh.points[:,2]<z33).astype(int)
    Mesh.point_data.update({"LowerLung":lower_mask})
    
    # Transform point data to cell data
    lower_cells = []
    for cell in Mesh.cells_dict['triangle']:
        lower_cells += [np.average(lower_mask[cell])]
    lower_cells =(np.array(lower_cells)>0.8).astype(int)
    
    # Update the geometry and save if required
    Mesh.cell_data.update({"Lower":[lower_cells]})    
    
    # Retrieve normals and generate a mask for normals
    normal_z = Mesh.point_data['Normals'][:,2]
    nz_mask = (normal_z < nz_cutoff).astype(int)
    
    # Joint both position and normal masks
    diaphragm_points = nz_mask*lower_mask
    
    # Transform point data to cell data
    diaphragm_cells = []
    for cell in Mesh.cells_dict['triangle']:
        diaphragm_cells += [np.average(diaphragm_points[cell])]
    diaphragm_cells =(np.array(diaphragm_cells)>0.8).astype(int)
    
    # Update the geometry
    Mesh.point_data.update({"Diaphragm":diaphragm_points})
    Mesh.cell_data.update({"Diaphragm":[diaphragm_cells]})
    
    # Export data if required
    if save: Mesh.write(out_mesh)
    
    return Mesh

# %%

# We'll interpolate intensities towards a different mesh
subject = 5
protocol = 'ARDSnet'
key = (subject,protocol)
r_root = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/"
r_mesh_path = r_root+"PIG%i/%s/MESH/"%key
r_nifti_path = r_root+"PIG%i/%s/NIFTI/"%key
r_registration_path = r_root+"PIG%i/%s/REGISTRATION/"%key

# Images to be used
exp_ct = r_nifti_path+"Exp.nii.gz"
insp_ct = r_nifti_path+"Insp.nii.gz"
exp_seg = r_nifti_path+"NEW_Mask_Exp.nii.gz"
insp_seg = r_nifti_path+"NEW_Mask_Insp.nii.gz"

# Simulation-associated paths
s_mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/%s/medium-fine/"%key
sim_tetra_mesh = s_mesh_path+"FEniCS/mesh000000.vtu"
in_simulated_ee_surf = s_mesh_path+"FEniCS/boundary_markers000000.vtu"
original_in_mesh = s_mesh_path+"temp.vtu"
final_mesh = s_mesh_path+"diaph.vtu"

# Source files
in_registered_ee_surf = r_mesh_path+"Surface_Exp.vtk"
reg_npz = np.load(r_mesh_path+"Exp_NEW.npz")

# Nifti files
filenameCPP = r_registration_path+"cpp_9000-000666.nii.gz"
filenameRef = r_nifti_path+"Exp.nii.gz"

# Point towards the deformed simulated mesh result
sim_root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/"
simulated_mesh = sim_root+"/codes/DeleteMe/MFSIMS/PIG%i/output/post/"%subject
if subject == 5:
    simulated_mesh += "full_0.750000000000.vtu"
elif subject == 4:
    simulated_mesh += "full_0.760000000000.vtu"
elif subject == 3:
    simulated_mesh += "full_0.890000000000.vtu"
else:
    simulated_mesh = None

# %%

verbose = True
nz_cutoff = -0.40
z_threshold = 0.23
dpi = 100
# We'll assume the registrated mesh over the simulated geometry is available

tet_mesh = pv.read(original_in_mesh)

# Extract a surface mesh
surf_mesh = tet_mesh.extract_surface()

# Set the plotted field to be the displacement field
surf_mesh.point_data.active_scalars_name="u"

zmax = np.max(surf_mesh.points[:,2])
zmin = np.min(surf_mesh.points[:,2])

if verbose: print("The lung ranges from Z (%.1f, %.1f)"%(zmin,zmax))

z33 = (zmax-zmin)*z_threshold + zmin
if verbose: print("The lung's basal region is at Z=%.2f"%z33)
     
# Generate a mask that only allows for the lower quarter to be seen
lower_mask = surf_mesh.points[:,2]<z33
     
norm = surf_mesh.compute_normals()
norm.point_data.active_scalars_name="Normals"
    
normals = surf_mesh.compute_normals()
normal_data = normals.point_data['Normals']
     
# Extract norm_z value in absolute value
# Generate a mask for the normal z. In Paraview, the diaphragm's norm points
# towards positive Z (inwards?). 
normal_z = normal_data[:,2]
nz_mask = normal_z < nz_cutoff
     
d_mask = np.logical_and(nz_mask,lower_mask)
     
d_points = surf_mesh.points[d_mask]
d_displacements = surf_mesh.point_data["u"][d_mask]
     
if verbose: print("Average Z displacement = %.2f"%np.mean(d_displacements[:,2])) 
    
if True:
    # Generate a visualization for the basal points
    fig,ax = plt.subplots(dpi=dpi)
    ax.scatter(d_points[:,0],d_points[:,1], s=2,alpha=0.5)
#    ax.set_title("Axial Projection of the Mesh - %s"%tag)
    ax.set_xlabel("Left to Right Position (mm)")
    ax.set_ylabel("Dorsal to Ventral Position (mm)")
    plt.show()
    
if True:
    # Generate a visualization for the basal points
    fig,ax = plt.subplots(dpi=dpi)
    ax.scatter(d_points[:,0],d_points[:,2], s=2,alpha=0.5)
#    ax.set_title("Axial Projection of the Mesh - %s"%tag)
    ax.set_xlabel("Left to Right Position (mm)")
    ax.set_ylabel("Dorsal to Ventral Position (mm)")
    plt.show()

# %% Process the registration mesh

# Extract surface
in_mesh = original_in_mesh
out_mesh = s_mesh_path+"temp_deleteme.vtu"
extract_surface(in_mesh, out_mesh)
# Isolate the diaphragm
in_mesh = s_mesh_path+"temp_deleteme.vtu"
out_mesh = final_mesh
isolate_diaphragm(in_mesh, out_mesh)

# %% Process the simulation mesh

sim_in_mesh = simulated_mesh
sim_out_mesh = s_mesh_path+"temp_sim_deleteme.vtu"
extract_surface(sim_in_mesh, sim_out_mesh)
# Isolate the diaphragm
sim_in_mesh = s_mesh_path+"temp_sim_deleteme.vtu"
sim_out_mesh = sim_in_mesh
isolate_diaphragm(sim_in_mesh, sim_out_mesh)

# %%

import functions2 as f2

out_rois = f2.give_field__through_ydirection_final(out_mesh, 10,10,'diafragma',normalize=False,
                                        campo='u')

_, _, _, _, lower_all, middle_all, upper_all = out_rois

disp_25 =  np.array(lower_all)
disp_50 = np.array(middle_all)
disp_75 = np.array(upper_all)

upper_err = disp_75-disp_50
lower_err = disp_50-disp_25


sout_rois = f2.give_field__through_ydirection_final(sim_out_mesh, 10,10,'diafragma',normalize=False,
                                        campo='u')

_, _, _, _, slower_all, smiddle_all, supper_all = sout_rois

sdisp_25 =  np.array(slower_all)
sdisp_50 = np.array(smiddle_all)
sdisp_75 = np.array(supper_all)

supper_err = sdisp_75-sdisp_50
slower_err = sdisp_50-sdisp_25


# %%
dpi=300; figsize=(5,6)
fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
#ax.errorbar(range(10),middle_all,yerr=(disp_50-disp_25,disp_75-disp_50),color='w')

yerr = [lower_err,upper_err]
syerr = [slower_err,supper_err]
xs = np.arange(10)

ax.errorbar(xs,middle_all,yerr=yerr,color='k',capsize=4,lw=1.5, label='Experiment')
ax.scatter(xs,middle_all,color="k",s=9)

ax.errorbar(xs,smiddle_all,yerr=syerr,color='r',capsize=4,lw=1.5, label='Simulation')
ax.scatter(xs,smiddle_all,color="r",s=9)


ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(10)+1,size=13)
ax.set_xlabel("ROI #",size=15)

ax.set_ylim((-12,0))
ylabs = ax.get_yticklabels()
ax.set_yticklabels(ylabs,size=13)

ax.set_ylabel('Diaphragm displacement (mm)',size=15)
ax.legend(loc='upper center')
