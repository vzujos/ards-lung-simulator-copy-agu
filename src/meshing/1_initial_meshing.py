# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 17:10:45 2025

@author: angus
"""

import iso2mesh as i2m
import nibabel as nib
import numpy as np
import os
import meshio as io
import matplotlib.pyplot as plt

def tetrahedron_volume_from_array(verts):
    """
    Compute the volume of a tetrahedron from a (4, 3) array of vertices.

    Parameters:
    verts: np.ndarray
        A (4, 3) array where each row represents a vertex (x, y, z) of the tetrahedron.

    Returns:
    float
        The volume of the tetrahedron.
    """
    assert verts.shape == (4, 3), "Input must be of shape (4, 3)"
    
    a = verts[1] - verts[0]
    b = verts[2] - verts[0]
    c = verts[3] - verts[0]
    
    volume = np.abs(np.dot(a, np.cross(b, c))) / 6.0
    return volume

def determine_cell_volumes(points, cells):
    volumes = []
    for cell in cells:
        volumes += [tetrahedron_volume_from_array(points[cell])]
    return np.array(volumes)


# %% BASE DECLARATIONS

qualities = {#"coarse":{"size":24},
          #   "medium-coarse":{"size":20},
         #    "medium":{"size":16},
             "medium-fine":{"size":13}, # medium-fine is 12; will be changed to 15 for specific subjects
      #       "fine":{"size":8},
             }


# Declare base path
root = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG4/ARDSnet/"

# Reference geometry
segmentation_path = root + "NIFTI/NEW_Mask_Exp.nii.gz"

# %%
for quality in qualities:
   
    # Declare where output files will be directed
    output_root = root+"%s/"%quality
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    
    # Load nifti segmentation
    seg_nii = nib.load(segmentation_path)
    seg_img = seg_nii.dataobj
    
    # % PARAMETERS
    
    isovalue = 1 # The value to be segmented; the value of interest in the segmentation
    elem_size = qualities[quality]["size"] # Element size 
    slack = 0.5 # An amplification factor for the reference tetrahedra in the maxvol property; 
                # Tests showed that for a fixed elem size = 12, slack 0.5 delivered good quality elements
    maxvol = slack * (elem_size**3/(6*np.sqrt(2))) 
    # 6 * sqrt(2) = ratio of cube volume to regular tetrahedron volume with same edge length
    keepratio = 0.9
    opt = {'holes':None, 'regions':[],"radbound":elem_size} # Data-structure
    
    # % BINARY TO SURFACE    
    # Function 'v2s'
    # > Surface mesh generation from binary or grayscale volumetric images.
    xyz, elem, regions, holes = i2m.v2s(np.array(seg_img), isovalue,opt=opt)
    
    # Extract and apply afifne
    affine = np.array([np.abs(seg_nii.affine[i,i]) for i in range(3)])
    
    # Correct mesh (numbering scheme and scale)
    elem = elem[:,:3]-1
    xyz *= affine
    
    # Visualize (numbering scheme modified)
    i2m.plotmesh(xyz,elem+1)
    
    #  Export into out temporal folder
    out = io.Mesh(points=xyz,cells={'triangle':elem})
    #out.write(output_root+"surface_raw.off")

    
    # %% LOAD MESHLAB-ENHANCED MESH
  
for quality in qualities:
    
    # Declare where output files will be directed
    output_root = root+"%s/"%quality     # Make sure it is the one you need
    inmesh = io.read(output_root+"surface_smooth.off") # As processed in MeshLab 
    inmesh.write(output_root+"surface_smooth.vtu") # For visualization purposes in Paraview
    
    # Load the MeshLab processed mesh arrays
    xyz_ml = inmesh.points # Cloud of points
    elem_ml = inmesh.cells_dict['triangle'] # Connectivity matrix
    
    # Defining relevant properties for element quality
    elem_size = qualities[quality]["size"] # Element size 
    slack = 0.5 # An amplification factor for the reference tetrahedra in the maxvol property; 
                # Tests showed that for a fixed elem size = 12, slack 0.5 delivered good quality elements
    maxvol = slack * (elem_size**3/(6*np.sqrt(2))) 
    # 6 * sqrt(2) = ratio of cube volume to regular tetrahedron volume with same edge length
    keepratio = 0.9    
    
    # % TETRAHEDRALIZE
    try:
        points, cells, _ = i2m.s2m(xyz_ml, elem_ml+1, keepratio=keepratio, maxvol=maxvol)
    except:
        print("Tetrahedralization failed for quality '%s'"%quality)
        continue
        pass
    
    cells = cells[:,:4]-1
    
    # The boolean (True or False) below allows to save the mesh
    if True:
        out4 = io.Mesh(points=points, cells={'tetra':cells})
        out4.write(output_root+"tetrahedral_base.vtu")
    
    # % TETRAHEDRAL QUALITY ASSESSMENT
    if False:
        for quality in qualities:
                print("\nQuality: %s"%quality)
                output_root = root+"%s/"%quality    
                path = output_root+"tetrahedral_base.vtu"
                M = io.read(path)
                points = M.points
                cells = M.cells_dict['tetra']
                emetric_tetra = i2m.meshquality(points,cells+1) # meshlab-enhanced
                
                print("Number of points: %i"%points.shape[0])
                print("Number of cells: %i"%cells.shape[0])
                
                volumes = determine_cell_volumes(points,cells)
                
                v50 = np.median(volumes)
                vmax = volumes.max()
                vmin = volumes.min()
                v95 = np.quantile(volumes,0.95)
                v05 = np.quantile(volumes,0.05)
                v75 = np.quantile(volumes,0.75)
                v25 = np.quantile(volumes,0.25)
                viqr = v75 - v25
                ratio_vmaxvmin = vmax/vmin
                ratio_v95v05 = v95/v05
                
                print("Median volume (IQR): %.0f (%.0f)"%(v50,viqr))
                print("MinMax volume ratio: %.0f"%ratio_vmaxvmin)
                print("95/05 volume ratio: %.1f"%ratio_v95v05)
            
            