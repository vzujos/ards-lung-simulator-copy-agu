# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:27:37 2024

@author: angus
"""

import Matlab_to_FEniCS_bridge as mtf
import os
import numpy as np
import meshio as io
import networkx as nx
import nibabel as nib
from scipy.io import loadmat
from collections import defaultdict
import dolfin


def assign_stiffness(phi, k=80, c_tissue_max=10.0, c_tissue_min=2.0):
    # Sigmoid transition centered at 0.2, steepness controlled by k
    return c_tissue_max - (c_tissue_max - c_tissue_min) * (1 - 1 / (1 + np.exp(-k * (0.2 - phi))))


if __name__ == '__main__':

    root = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/"
    
    print("Generating paths:")
    
    # Export porosity mesh?
    export_visualization_for_porosity = True
    
    # Declare paths in a format apt for execution in WSL
    path_to_mesh = root+"MESH/"
    path_to_images = root+"NIFTI/"
    
    # Folder where the FEniCS data is going to be placed
    output_folder = "FEniCS/"
    # Input tetrahedral mesh generated following the standard pipeline
    mesh_name = "tet_lung.mat"
    mat_path = path_to_mesh+mesh_name 
    
    # Path to a surface mesh where the diaphragm has been isolated
    diaph_path = path_to_mesh+output_folder+"mesh_diaph.vtu"
    
    # Path to the airway skeleton under use
    skel_path = path_to_mesh+"skel.vtu"
    
    # Path to a reference image
    nii_path = path_to_images+"Exp.nii.gz"
    
    # Do these folders exist?
    print(" > Testing involved folders")
    for folder in [path_to_mesh, path_to_images]:
        if os.path.isdir(folder):
            print("    > [%s] Found."%(folder.split("/")[-1]))
        else:
            print("    > [%s] Not found"%(folder.split("/")[-1]))
    
    # Create folders if needed
    mtf.manage_paths(path_to_mesh,mesh_name, output_folder=output_folder)
    
    # Create a vector that will scale and localize the finite element mesh
    print(" > Generating scaling vector:")
    vec = mtf.scaling_vector(nii_path,positive_affine=True)
    
    # Standard procedure to generate the fenics mesh
    print(" > Generating FEniCS geometries:")
    mesh = mtf.mat2fenics(mat_path, path_to_mesh+output_folder, vec)
    
    # If the airway skeleton is available
    if os.path.isfile(skel_path):
        
        # Retrieve the airway terminals
        print(" > Skeleton geometry found:")
        print(" > Extracting airway terminals")
        terminals = mtf.terminals_from_skel_mesh(skel_path)
        
        # Generate the subdomains upon the FEniCS mesh
        print(" > Generating subdomains in FE mesh")
        fenics_mesh_path = path_to_mesh+output_folder+"mesh000000.vtu"
        subdomains = mtf.assign_subdomains(fenics_mesh_path, terminals, path_to_mesh+output_folder)

        # Analyze generated subdomains
        print(" > Analyzing the subdomains")
        counter = []; volumes = []
        
        # Read the mesh
        fenicsmesh = io.read(fenics_mesh_path)
        xyz = fenicsmesh.points
        cells = fenicsmesh.cells_dict['tetra']
        
        # Initialize a counter
        nterminals = len(terminals)
        
        # For each terminal
        for j in range(nterminals):
            
            # Isolate the cells belonging to each terminal
            sd_mask = subdomains==j
            sd_cells = cells[sd_mask]
            
            # Determine volume in subdomain
            vol = 0.0
            for scell in sd_cells:
                ps = xyz[scell]
                vol += mtf.tetvol(ps)    
                
            # Add the number of elements per subdomain and its volume
            counter += [np.count_nonzero(sd_mask)]
            volumes += [vol]
            
        # Transform into numpy arrays
        volumes = np.array(volumes)
        counter = np.array(counter)   
       
        # Present analysis
        print("\n > SUBDOMAIN ANALYSIS: ")
        print("     > Number of null domains: %i/%i"%(np.count_nonzero(counter==0),len(terminals)))
        print("     > Minimum volume: %.0f"%(np.min(volumes)))
        print("     > Maximum volume: %.0f"%(np.max(volumes)))
        print("     > Volume ratio: %.1f"%(np.max(volumes)/np.min(volumes)))
    
        print("> Interpolating intensities:")
        # Load raw matlab mesh
        M = loadmat(mat_path)
        # Scale the vector
        xyz = M['tetnode']*vec
        # Load the finished mesh
        fenicsmesh = io.read(fenics_mesh_path)
        
        # Interpolate the end-expiratory intensities to the fenics mesh
        intensities = mtf.interpolate_intensities(xyz, nii_path)
        
        print("Mean intensity: %.2f"%np.average(intensities))
        
        # Saturate the intensities outside the porosity range
        intensities[intensities>0.0] = 0.0
        intensities[intensities<-1000.0] = -1000.0
        
        # Determine porosity through naive means
        porosity_pd = intensities/-1000.0
        
        # Correct porosity
        porosity_pd = mtf.correct_porosity(xyz, fenicsmesh.cells_dict['tetra'], porosity_pd)
    
        # Change from point data to cell data
        porosity_cd = []
        for cell in fenicsmesh.cells_dict['tetra']:
            porosity_cd += [np.average(porosity_pd[cell])]
        
        # Export a visualization mesh for the porosity field
        if export_visualization_for_porosity:
            print(" > Exporting porosity visualization mesh")
            test_output = io.Mesh(points=fenicsmesh.points,
                                  cells=fenicsmesh.cells_dict,
                                  point_data={"HU_EE":intensities,
                                              "Porosity_EE":porosity_pd},
                                  cell_data={"Porosity_EE":[porosity_cd]})
            
            test_output.write(path_to_mesh+output_folder+"Porosity_Visualization.vtu")
    
        # Create mesh functions for k0 (initial permeability)
        k0 = dolfin.MeshFunction("double", mesh, 3) # Function for permeability
        phi0 = dolfin.MeshFunction("double",mesh,3) # Function for porosity
        c_tissue = dolfin.MeshFunction("double",mesh,3) # Function for tissue stiffness
        
        # Tissue stiffness
        c_tissue_max = 10.0
        c_tissue_min = 2.0
        # default permeability
        knormal = 10**2
      #  knormal = 10**6 # increased by me
        kmin = 1e-1
        phi0_mean = 0.50 # mean porosity at PIG4 00h Exp
        
        print(" > Generating the FEniCS mesh")
        
        # Iterate over mesh and set values
        for cell, phi in zip(dolfin.cells(mesh), porosity_cd):
            if phi > 0.1:
                k0[cell]=knormal*((phi/phi0_mean)**(2/3)) + kmin #define permeability as a porosity function
            else:
                k0[cell]=kmin
            phi0[cell] = phi
            c_tissue[cell] = assign_stiffness(phi,c_tissue_max=c_tissue_max,c_tissue_min=c_tissue_min)

        # Store to file
        mesh_file = dolfin.File(path_to_mesh+output_folder+"mesh.xml.gz")
        k0_file = dolfin.File(path_to_mesh+output_folder+"k0.xml.gz")
        phi0_file = dolfin.File(path_to_mesh+output_folder+"phi0.xml.gz")
        c_tissue_file = dolfin.File(path_to_mesh+output_folder+"c_tissue.xml.gz")

        mesh_file << mesh
        k0_file << k0
        phi0_file << phi0
        c_tissue_file << c_tissue
     
        # Final visualization of what has been computed
        permeabilityparaview = dolfin.File(path_to_mesh+output_folder+"permeabilityparaview.pvd", "compressed")
        permeabilityparaview << (k0)
        
        print("\n >>>>> END OF SCRIPT <<<<<")

