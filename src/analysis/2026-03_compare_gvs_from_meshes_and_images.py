# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:34:23 2026

@author: angus
"""

import os
import meshio as io
import numpy as np
import nibabel as nib
# Compare aggregated volumetric strain vs image-based volumetric strain

def tetra_volumes(points, cells):
    p0 = points[cells[:,0]]
    p1 = points[cells[:,1]]
    p2 = points[cells[:,2]]
    p3 = points[cells[:,3]]

    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0

    vol = np.abs(np.einsum('ij,ij->i', np.cross(v1, v2), v3)) / 6.0
    return vol

def compute_mask_volumes_and_gvs(nifti_path):

    img_exp_path = nifti_path + 'Exp_median.nii.gz'
    img_insp_path = nifti_path + 'Insp_median.nii.gz'
    mask_exp_path = nifti_path + 'NEW_Mask_Exp.nii.gz'
    mask_insp_path = nifti_path + 'NEW_Mask_Insp.nii.gz'

    # Load masks and images
    seg_exp = nib.load(mask_exp_path)
    seg_insp = nib.load(mask_insp_path)

    img_exp = nib.load(img_exp_path)
    img_insp = nib.load(img_insp_path)

    seg_exp_data = np.asarray(seg_exp.dataobj)
    seg_insp_data = np.asarray(seg_insp.dataobj)
    img_exp_data = np.asarray(img_exp.dataobj)
    img_insp_data = np.asarray(img_insp.dataobj)

    mask_exp = seg_exp_data > 0
    mask_insp = seg_insp_data > 0

    # Count voxels
    nvox_exp = np.count_nonzero(mask_exp)
    nvox_insp = np.count_nonzero(mask_insp)

    # Air fraction (HU -> porosity)
    seg_data_exp = -img_exp_data[mask_exp] / 1000
    seg_data_insp = -img_insp_data[mask_insp] / 1000

    seg_data_exp = np.clip(seg_data_exp, 0, 1)
    seg_data_insp = np.clip(seg_data_insp, 0, 1)

    # Voxel volume
    voxel_vol = abs(np.linalg.det(seg_exp.affine[:3,:3]))

    # Raw volumes
    exp_vol = nvox_exp * voxel_vol
    insp_vol = nvox_insp * voxel_vol

    # Air content
    exp_air = np.sum(seg_data_exp) * voxel_vol
    insp_air = np.sum(seg_data_insp) * voxel_vol
    air_vt = insp_air - exp_air

    # Strains
    gvs = (insp_vol - exp_vol) / exp_vol
    air_gvs = air_vt / exp_vol

    return (exp_vol, insp_vol, air_vt), (gvs, air_gvs)


def compute_volumes_and_gvs(mesh, jacobian_name='Jacobian Forward'):
    # geometry
    xyz = mesh.points
    u = mesh.point_data['DispField']
    phi = xyz + u

    # connectivity
    cells = mesh.cells_dict['tetra']

    # nodal Jacobian
    j = np.asarray(mesh.point_data[jacobian_name]).squeeze()

    # cell Jacobian (nodal average)
    cell_jacobian = np.mean(j[cells], axis=1)

    # volumes
    vol_xyz = tetra_volumes(xyz, cells)
    vol_phi = tetra_volumes(phi, cells)

    original_volume = np.sum(vol_xyz)
    deformed_volume = np.sum(vol_phi)

    # Jacobian-based deformed volume
    vol_phi_from_J = cell_jacobian * vol_xyz
    deformed_volume_J = np.sum(vol_phi_from_J)

    # global volumetric strain
    gvs = np.abs((original_volume - deformed_volume) / original_volume)
    gvs_J = np.abs((original_volume - deformed_volume_J) / original_volume)

    return (original_volume, deformed_volume, deformed_volume_J), (gvs, gvs_J)



# Establish paths
subject = 4
protocol = 'ARDSnet'
key = (subject,protocol)
r_root = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/"
mesh_quality = 'medium'

# Mesh path
s_mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/%s/"%key+"%s/"%mesh_quality
# Nifti file path
s_nifti_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/%s/"%key+"NIFTI/"



experiment_mesh = s_mesh_path+"reg_anim_ready.vtu"
simulation_mesh = s_mesh_path+"sim_anim_ready.vtu"


if os.path.isfile(experiment_mesh):
    print("Experiment mesh found")
else:
    print("Experiment mesh not found")
    
if os.path.isfile(simulation_mesh):
    print("Simulation mesh found")
else:
    print("Simulation mesh not found")

sim_mesh = io.read(simulation_mesh)
exp_mesh = io.read(experiment_mesh)

sim_vols, sim_gvs = compute_volumes_and_gvs(sim_mesh)
exp_vols, exp_gvs = compute_volumes_and_gvs(exp_mesh,jacobian_name='Jacobian')
img_vols, img_gvs = compute_mask_volumes_and_gvs(s_nifti_path)

img_v0, img_vinsp, img_air_vt = img_vols
img_gvs_geom, img_air_gvs = img_gvs

# %%

scale = 1e6

# unpack simulation
sim_v0, sim_vinsp, sim_vinsp_J = sim_vols
sim_gvs_geom, sim_gvs_J = sim_gvs
sim_dv = sim_vinsp - sim_v0

# unpack experimental mesh
exp_v0, exp_vinsp, exp_vinsp_J = exp_vols
exp_gvs_geom, exp_gvs_J = exp_gvs
exp_dv = exp_vinsp - exp_v0

# unpack CT
img_v0, img_vinsp, img_air_vt = img_vols
img_gvs_geom, img_air_gvs = img_gvs
img_dv = img_vinsp - img_v0

print("\n (PIG%i) GLOBAL VOLUME AND STRAIN COMPARISON"%subject)
print("-"*110)
print(f"{'Source':<22}{'Exp Vol':>12}{'Insp Vol':>12}{'ΔVolume':>12}{'Air VT':>12}{'GVS %':>10}{'Air GVS %':>10}")
print("-"*110)

print(f"{'Simulation mesh':<22}{sim_v0/scale:>12.2f}{sim_vinsp/scale:>12.2f}{sim_dv/scale:>12.2f}{'-':>12}{100*sim_gvs_geom:>9.1f}%{'-':>10}")

print(f"{'Experiment mesh':<22}{exp_v0/scale:>12.2f}{exp_vinsp/scale:>12.2f}{exp_dv/scale:>12.2f}{'-':>12}{100*exp_gvs_geom:>9.1f}%{'-':>10}")

print(f"{'CT segmentation':<22}{img_v0/scale:>12.2f}{img_vinsp/scale:>12.2f}{img_dv/scale:>12.2f}{img_air_vt/scale:>12.2f}{100*img_gvs_geom:>9.1f}%{100*img_air_gvs:>9.1f}%")

# %%
print("\nMESH JACOBIAN CONSISTENCY CHECK")
print("-"*70)
print(f"{'Source':<20}{'Geom Insp Vol':>18}{'Jacobian Insp Vol':>20}{'Diff %':>12}")
print("-"*70)

def rel_diff(a,b):
    return 100*(a-b)/b

print(f"{'Simulation':<20}{sim_vinsp/scale:>18.2f}{sim_vinsp_J/scale:>20.2f}{rel_diff(sim_vinsp,sim_vinsp_J):>11.2f}%")
print(f"{'Experiment':<20}{exp_vinsp/scale:>18.2f}{exp_vinsp_J/scale:>20.2f}{rel_diff(exp_vinsp,exp_vinsp_J):>11.2f}%")
