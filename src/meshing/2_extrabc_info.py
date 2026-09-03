# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:24:02 2025

@author: angus
"""

import os
import numpy as np
import meshio as io
import pyvista as pv
import trimesh as tri
import matplotlib.pyplot as plt
import pandas as pd
import csv

def export_cell_ids(cell_ids, save_path, file_name="cells.csv"):
    """
    Creates a CSV file with a single column 'Cell ID' and one integer per row.

    Parameters
    ----------
    cell_ids : list of int
        List of integer cell IDs.
    save_path : str
        Directory path where the CSV file will be saved.
    file_name : str
        Output CSV file name.
    """
    
    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # Full output path
    output_path = os.path.join(save_path, file_name)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cell ID"])  # header
        for value in cell_ids:
            writer.writerow([value])

    print(f"File written: {output_path}")


def extract_surface(in_mesh, out_mesh,save=True, field="u"):
    
    '''
    Takes in a tetrahedral mesh and returns a surface triangular mesh.
    '''
    
    # Load tetrahedral mesh
    tet_mesh = pv.read(in_mesh)
     
    # Extract a surface mesh
    surf_mesh = tet_mesh.extract_surface()
     
    # Set the plotted field to be the displacement field
   # surf_mesh.point_data.active_scalars_name=field
    
    # Point cloud
    points = surf_mesh.points
    
    # Displacement field
   # disp_field = surf_mesh.get_array(field)
    
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
                   point_data={"Normals":normal_data})
    
    if save: Mesh.write(out_mesh)
    
    return Mesh

def isolate_diaphragm(in_mesh, out_mesh, z_threshold=0.23, 
                      nz_cutoff=-0.4, save=True,
                      exclusion_list = [], inclusion_list=[],
                      reverse_normal=False):
    
    '''
    Determines the surface that defines the diaphragm and generates
    the geometries for visualization and data management.
    '''
    
    print("function 'isolate_diaphragm':")
    print(" >  in: %s"%in_mesh)
    print(" > out: %s"%out_mesh)
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
    if reverse_normal: normal_z *= -1
    nz_mask = (normal_z < nz_cutoff).astype(int)
    Mesh.point_data.update({'Nz_mask':nz_mask})
    Mesh.point_data.update({'Nz':normal_z})

    
    # Joint both position and normal masks
    diaphragm_points = nz_mask*lower_mask
    
    # Transform point data to cell data
    diaphragm_cells = []
    for cell in Mesh.cells_dict['triangle']:
        diaphragm_cells += [np.average(diaphragm_points[cell])]
    diaphragm_cells =(np.array(diaphragm_cells)>0.8).astype(int)
    
    # If inclusion_list is not empty:
    if len(inclusion_list)>0:
        for i in inclusion_list:
            diaphragm_cells[i] = 1
    
    # If exclusion_list is not empty:
    if len(exclusion_list)>0:
        for e in exclusion_list:
            diaphragm_cells[e] = 0
    
    # Update the geometry
    Mesh.point_data.update({"Diaphragm":diaphragm_points})
    Mesh.cell_data.update({"Diaphragm":[diaphragm_cells]})

    
    # Export data if required
    if save: Mesh.write(out_mesh)
    
    return Mesh



def mark_dorsal(in_mesh, out_mesh, visualize_inferred_plane=True,
                visualize_displaced_plane=True, dy = -12.5, save=True,
                exclusion_list = [], inclusion_list =[]):
    
    # Load the mesh
    mesh = io.read(in_mesh)
    xyz = mesh.points
    
    # Retrieve y-bounds (ventral-to-dorsal)
    # Retrieve z-bounds (apical-to-basal)
    z0 = xyz[:,2].min(); z1 = xyz[:,2].max()
    # Establish nrois (default nrois=7)
    nrois = 7
    zrois = np.linspace(z0,z1,nrois+1)
    
    # Generate a mask for the ROIs within the region
    zmask = np.logical_and(xyz[:,2]<zrois[2],xyz[:,2]>zrois[1])
    # Retrieve y-minimum point as a reference
    idmax = xyz[zmask][:,1].argmax()
    dot1 = xyz[zmask][idmax]
    
    # Generate a mask for the ROIs within the region
    zmask = np.logical_and(xyz[:,2]<zrois[-2],xyz[:,2]>zrois[-3])
    # Midvalue for the region
    # Retrieve y-minimum point as a reference
    idmax = xyz[zmask][:,1].argmax()
    dot2 = xyz[zmask][idmax]
    
    # Compute the vector connecting the two points
    v = dot2 - dot1
    
    # Normal vector to the plane (since plane is parallel to x-axis)
    n = np.array([0, -v[2], v[1]])
    
    # Normalize (optional)
    n = n / np.linalg.norm(n)
    
    # Plane equation coefficients: n_y * y + n_z * z + d = 0
    d = - (n[1] * dot1[1] + n[2] * dot1[2])
    
    print("Plane normal:", n)
    print(f"Plane equation: 0·x + {n[1]:.4f}·y + {n[2]:.4f}·z + {d:.4f} = 0")
    
    
    if visualize_inferred_plane:
        # --- Randomly sample 200 points from the mesh ---
        N = 400
        idx = np.random.choice(len(xyz), N, replace=False)
        sample = xyz[idx]
        
        # Coordinates
        z_pts = sample[:, 2]   # Z → horizontal
        y_pts = sample[:, 1]   # Y → vertical
        
        # --- Compute the line projection (plane ∩ x-parallel slice) ---
        # From the plane equation: n_y*y + n_z*z + d = 0  →  y = -(n_z*z + d)/n_y
        z_line = np.linspace(z_pts.min(), z_pts.max(), 200)
        y_line = -(n[2]*z_line + d)/n[1]
        
        # --- Plot ---
        plt.figure(figsize=(6,6))
        plt.scatter(z_pts, y_pts, s=25, alpha=0.5, color='steelblue', label='Mesh points')
        plt.plot(z_line, y_line, 'r-', linewidth=2, label='Inferred plane (Z–Y line)')
        plt.scatter([dot1[2], dot2[2]], [dot1[1], dot2[1]], color='orange', s=50, zorder=3, label='Reference points')
        
        plt.xlabel('Z (apical → basal)')
        plt.ylabel('Y (ventral → dorsal)')
        plt.title('Lung mesh projection (Z–Y) and inferred plane')
        plt.legend()
        plt.grid(True)
        
        # Adjust axes to the relevant region (avoid extra blank space)
        plt.xlim(z_pts.min(), z_pts.max())
        plt.ylim(y_pts.min(), y_pts.max())
        plt.axis('equal')
        
        plt.show()
        
    # --- 1) Compute displaced plane coefficients ---
    d_shifted = d - n[1] * dy
    
    # --- 2) Compute signed distance from each mesh point to the plane ---
    # For plane parallel to x: signed distance = (n_y*y + n_z*z + d)
    # --- 2) Compute signed distance from each mesh point to the DISPLACED plane ---
    # Plane equation: n_y*y + n_z*z + d_shifted = 0
    signed_dist_shifted = n[1] * xyz[:,1] + n[2] * xyz[:,2] + d_shifted
    
    # Points above the displaced plane (positive side)
    mask_above = signed_dist_shifted > 0
    
    # --- 3) Identify cells associated to those points ---
    cells = mesh.cells_dict["triangle"]
    mask_cells_above = np.any(mask_above[cells], axis=1).astype(int)
    
    if len(exclusion_list)>0 :
        for cell in exclusion_list:
          #  print("CELL: %i || [Exclusion] Before: %i"%(cell, mask_cells_above[cell]))
            mask_cells_above[cell]=1    
        #    print("CELL: %i || [Exclusion] After: %i"%(cell, mask_cells_above[cell]))

    
    if len(inclusion_list)>0:
        for cell in inclusion_list:
            mask_cells_above[cell]=0
    
    # --- 4) Save filtered meshes ---
    mesh.cell_data.update({"Dorsal":[mask_cells_above]})
    
    if save:
        mesh.write(out_mesh)
    
    if visualize_displaced_plane:
        # --- 5) 2D visualization (Z–Y projection) ---
        
        # Sample random points for visualization
        N = 400
        idx = np.random.choice(len(xyz), N, replace=False)
        sample = xyz[idx]
        z_pts = sample[:,2]
        y_pts = sample[:,1]
        
        # Original line (red)
        z_line = np.linspace(z_pts.min(), z_pts.max(), 200)
        y_line = -(n[2]*z_line + d)/n[1]
        
        # Displaced line (green)
        y_line_shifted = -(n[2]*z_line + d_shifted)/n[1]
        
        # Classification of sampled points (above/below displaced plane)
        signed_sample = n[1]*y_pts + n[2]*z_pts + d_shifted
        mask_above_s = signed_sample > 0
        
        plt.figure(figsize=(6,6))
        plt.scatter(z_pts[mask_above_s], y_pts[mask_above_s], s=25, color='dodgerblue', alpha=0.7, label='Points above')
        plt.scatter(z_pts[~mask_above_s], y_pts[~mask_above_s], s=25, color='salmon', alpha=0.7, label='Points below')
        
        plt.plot(z_line, y_line, 'r-', linewidth=2, label='Original plane')
        plt.plot(z_line, y_line_shifted, 'g--', linewidth=2, label=f'Displaced plane (dy={dy})')
        
        plt.scatter([dot1[2], dot2[2]], [dot1[1], dot2[1]], color='orange', s=50, zorder=3, label='Reference points')
        
        plt.xlabel('Z (apical → basal)')
        plt.ylabel('Y (ventral → dorsal)')
        plt.title('Plane displacement and point classification (Z–Y projection)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(z_pts.min(), z_pts.max())
        plt.ylim(y_pts.min(), y_pts.max())
        plt.show()
    
    return mesh

def mark_mediastinum(in_mesh, out_mesh, seeds, include_list=[], exclude_list=[],
                     ply_filename = 'surface.ply', eps = 1e-6, save=True):

    
    surface = io.read(in_mesh)
    
    # Generate arrays with relevant points/cells
    spoints = surface.points
    scells = surface.cells_dict['triangle']
    # Export .ply file for raycasting execution
    ply_out = io.Mesh(points=spoints, cells=surface.cells_dict)
    ply_out.write(mesh_path+ply_filename)
    # Manually selected sources. Must be placed somewhere in the 
    # mediastinum cavity.

    # Cell centers associated to each surface element
    cell_centers = np.array([np.average(spoints[cell],axis=0) for cell in scells])
    ply_mesh = tri.load(mesh_path+ply_filename)

    # Build accelerated ray intersector
    try:
        rmi = tri.ray.ray_pyembree.RayMeshIntersector(ply_mesh)
    except:
        print("pyembree not available: falling back to slower ray module.")
        rmi = tri.ray.ray_triangle.RayMeshIntersector(ply_mesh)
    
    is_mediastinal = np.zeros(len(ply_mesh.faces), dtype=int)
    
    
    
    for seed in seeds:
    
        # Normalize directions
        directions = cell_centers - seed
        norms = np.linalg.norm(directions, axis=1)
        directions = directions / norms[:, None]
    
        # Shift origins slightly to avoid hitting the cell itself due to numerical issues
        origins = seed + eps * directions
    
        # BVH query: only first intersection
        index_tri, index_ray = rmi.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=False
        )
    
        # Mark faces visible to the seed
        for tri_i, ray_i in zip(index_tri, index_ray):
            # The ray index corresponds to the triangle whose centroid we were aiming at
            if tri_i == ray_i:
                is_mediastinal[tri_i] += 1
    

    mediastinal = is_mediastinal>(len(seeds)-1)
    
    if len(include_list)>0:
        print("Including '%i' faces into the mediastinum"%len(include_list))
        for c in include_list:
            mediastinal[c] = 1
    
    if len(exclude_list)>0:
        print("Excluding '%i' faces from the mediastinum"%(len(exclude_list)))
        for c in exclude_list:
            mediastinal[c] = 0
    
    
    surface.cell_data.update({'Mediastinal':[mediastinal.astype(int)]})
    if save:
        surface.write(out_mesh)
    
    
data_lists = {"PIG2":{"diaphragm-parameters":{"nz_cutoff":-0.3,
                                              "z_threshold":0.30,
                                              "reverse_normal":False},
                      "dorsal-parameters":{"ny_cutoff":0.5,
                                           "y_cutoff":0.10,
                                           "reverse_normal":False,
                                           "dy":-6.0},
                      "mediastinum-parameters":{"seeds":np.array([(-10., -40., 20.),     # Point 1
                                                                  (0.,-30.,  50.),       # Point 2
                                                                  (10., -10., 50.)]),   # Point 3
                                                },
                      } ,
              "PIG3":{"diaphragm-parameters":{"nz_cutoff":-0.3,
                                              "z_threshold":0.26,
                                              "reverse_normal":True},
                      "dorsal-parameters":{"ny_cutoff":0.5,
                                           "y_cutoff":0.10,
                                           "reverse_normal":False,
                                           "dy":-7.0},
                      "mediastinum-parameters":{"seeds":np.array([(-10., -40., 20.),     # Point 1
                                                                  (0.,-30.,  50.),       # Point 2
                                                                  (10., -10., 50.)]),   # Point 3
                                                },
                      },
              "PIG4":{"diaphragm-parameters":{"nz_cutoff":-0.3,
                                              "z_threshold":0.26,
                                              "reverse_normal":False},
                      "dorsal-parameters":{"ny_cutoff":0.5,
                                           "y_cutoff":0.10,
                                           "reverse_normal":False,
                                           "dy":-7.0},
                      "mediastinum-parameters":{"seeds":np.array([(-10., -40., 20.),     # Point 1
                                                                  (0.,-30.,  50.),       # Point 2
                                                                  (10., -10., 50.)]),   # Point 3
                                                },
                      },
              "PIG5":{"diaphragm-parameters":{"nz_cutoff":-0.3,
                                              "z_threshold":0.26,
                                              "reverse_normal":False},
                      "dorsal-parameters":{"ny_cutoff":0.5,
                                           "y_cutoff":0.10,
                                           "reverse_normal":False,
                                           "dy":-7.0},
                      "mediastinum-parameters":{"seeds":np.array([(20., -25., 50.),     # Point 1
                                                                  (0.,-50.,  0.),       # Point 2
                                                                  (-20., -50., 50.)]),   # Point 3
                                                },
                     },
              "PIG6":{"diaphragm-parameters":{"nz_cutoff":-0.3,
                                              "z_threshold":0.26,
                                              "reverse_normal":True},
                      "dorsal-parameters":{"nz_cutoff":-0.3,
                                           "z_threshold":0.30,
                                           "reverse_normal":False,
                                           "dy":-4.5},
                      "mediastinum-parameters":{"seeds":np.array([(-10., -40., 20.),     # Point 1
                                                                  (0.,-30.,  50.),       # Point 2
                                                                  (10., -10., 50.)]),   # Point 3
                                                },
                      
                      },

              }



pig_number = 4
mesh_type = "medium-fine"

subject = "PIG%i"%pig_number
mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/%s/ARDSnet/%s/"%(subject,mesh_type)

mesh_filename = "FEniCS/mesh000000.vtu"
mesh_outname = "bc_info.vtu"

in_mesh = mesh_path+mesh_filename
out_mesh = mesh_path+mesh_outname

extract_surface(in_mesh, out_mesh,save=True)


try:
    inclusion_list = np.array(pd.read_csv(mesh_path+"include-diaphragm.csv")['Cell ID'])
except:
    print("No inclusion list found for the diaphragm region")
    inclusion_list = []

try:
    exclusion_list = np.array(pd.read_csv(mesh_path+"exclude-diaphragm.csv")['Cell ID'])
except:
    print("No exclusion list found for the diaphragm region")
    exclusion_list = []

nz_cutoff = data_lists[subject]["diaphragm-parameters"]['nz_cutoff']
z_threshold = data_lists[subject]['diaphragm-parameters']['z_threshold']
reverse_normal = data_lists[subject]['diaphragm-parameters']['reverse_normal']

mesh = isolate_diaphragm(out_mesh,out_mesh,save=True,
                      z_threshold=z_threshold,
                      nz_cutoff=nz_cutoff,
                      exclusion_list=exclusion_list, 
                      inclusion_list=inclusion_list,
                      reverse_normal=reverse_normal)

# %%

try:
    inclusion_list = np.array(pd.read_csv(mesh_path+"include-dorsal.csv")['Cell ID'])
except:
    print("No inclusion list found for the dorsal region")
    inclusion_list = []

try:
    exclusion_list = np.array(pd.read_csv(mesh_path+"exclude-dorsal.csv")['Cell ID'])
except:
    print("No exclusion list found for the dorsal region")
    exclusion_list = []
    
dy = data_lists[subject]['dorsal-parameters']['dy']
    
mesh = mark_dorsal(out_mesh, out_mesh, save=True, exclusion_list=exclusion_list,
                   inclusion_list=inclusion_list, dy=dy)


# %%

try:
    inclusion_list = np.array(pd.read_csv(mesh_path+"include-mediastinum.csv")['Cell ID'])
except:
    print("No inclusion list found for the mediastinum region")
    inclusion_list = []

try:
    exclusion_list = np.array(pd.read_csv(mesh_path+"exclude-mediastinum.csv")['Cell ID'])
except:
    print("No exclusion list found for the mediastinum region")
    exclusion_list = []

seeds = data_lists[subject]['mediastinum-parameters']["seeds"]

mesh = mark_mediastinum(out_mesh, out_mesh, seeds, save=True, exclude_list=exclusion_list,
                   include_list=inclusion_list)

