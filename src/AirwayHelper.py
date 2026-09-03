# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:28:28 2024

@author: angus
"""


import numpy as np
import meshio as io
import matplotlib.pyplot as plt

def terminals_from_skel_mesh(skel_path):
    '''
    Generate a list of points from a skeleton mesh
    '''
    skel = io.read(skel_path)
    points = skel.points
    distal_points = skel.point_data['distal']
    distal_point_ids = np.arange(points.shape[0])[distal_points==1]
    return points[distal_point_ids]
    
def assign_subdomains(mesh_path, terminals, export_mesh=True):
    
    # Read the mesh
    reference_mesh = io.read(mesh_path)
    ien = reference_mesh.cells_dict['tetra']
    xyz = reference_mesh.points
    
    # Store here the distances
    distances = []
    
    ### CELLS
    # Move to cells centers (coordinate triplet x,y,z for all cell center)
    cell_center = []
    for cell in ien:
        cell_center += [np.average(xyz[cell],axis=0)]
    cell_center = np.array(cell_center)
    
    # Compute distances between terminals and cell centers
    distances = []
    for p in terminals:
        # Change into numpy array
        p=np.array(p)
        # Compute distances
        distances += [np.linalg.norm(cell_center-p,axis=1)]
    
    # Compute distances
    distances = np.matrix(distances).T
    
    subdomains = np.ravel(np.argmin(distances,axis=1))
    
    if export_mesh:
        
        out = io.Mesh(xyz, cells=reference_mesh.cells_dict, cell_data={"subdomains":[subdomains]})
        filename = mesh_path.split(".vtu")[0]+"_temp.vtu"
        out.write(filename)
        
    

    # Return subdomain ids for each cell
    return subdomains, cell_center



def tetvol(points):
    ps = np.hstack([points,np.ones((4,1))])
    return np.abs(np.linalg.det(ps))/6


skel_path = "C:/Users/angus/Downloads/CORNELL-GEOM/test/AIRWAYS/skel.vtu"
mesh_path = "C:/Users/angus/Downloads/CORNELL-GEOM/test/MESH-medium/FEniCS/mesh000000.vtu"

terminals = terminals_from_skel_mesh(skel_path)
subds, cell_centers = assign_subdomains(mesh_path, terminals)

# %%

counter = []; volumes = []

mesh = io.read(mesh_path)
xyz = mesh.points
cells = mesh.cells_dict['tetra']
    
nterminals = len(terminals)
parser = np.floor(nterminals/20)

for j in range(nterminals):
    
    if j%parser == 0: print("%.0f"%(j/parser*5)+"%")

    sd_mask = subds==j
    sd_cells = cells[sd_mask]
        
    vol = 0.0
    for scell in sd_cells:
        ps = xyz[scell]
        vol += tetvol(ps)    
        
    counter += [np.count_nonzero(sd_mask)]
    volumes += [vol]

volumes = np.array(volumes)
counter = np.array(counter)
    
# %%
# Present information in terminal
print("Number of null domains: %i"%np.count_nonzero(counter==0))
print("Minimum volume: %.0f"%(np.min(volumes)))
print("Maximum volume: %.0f"%(np.max(volumes)))
print("Volume ratio: %.1f"%(np.max(volumes)/np.min(volumes)))


# %%
# Present an histogram for the volumes
plt.hist(volumes)

# %%

# Define quantiles for volume

qlow = np.quantile(volumes,0.05)
qhigh = np.quantile(volumes,0.90)

print("Lower quantile: %.1f [mm3]"%qlow)
print("Higher quantile: %.1f [mm3]"%qhigh)

# %%

# Mark for trimming/subdivision
ids = np.arange(nterminals)
below = ids[volumes<qlow]
above = ids[volumes>qhigh]

intervene_path = "C:/Users/angus/Downloads/CORNELL-GEOM/test/AIRWAYS/intervention_list.npz"

np.savez(intervene_path,below=below,above=above)
