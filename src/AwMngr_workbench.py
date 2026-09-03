# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:39:30 2024

@author: angus
"""

import AirwayManager as awm
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import meshio as io

# %% Determine equivalent resistance

if __name__ == "__main__" and False:
    
    root = "C:/Users/angus/Downloads/CORNELL-NEWGEO/%s/%s/%s/"
    subject = "PIG2"
    case = "ARDSnet"
    mesh_folder = "MESH-15"
    
    target_path = root%(subject,case,mesh_folder)
    target = target_path+"skel.vtu"
    
    if os.path.isfile(target):
        print("File found")
    else:
        print("File not found")
        
    R = awm.determine_equivalent_resistance(target)
    


# %% Generate synthetic airway trees

if __name__ == "__main__" and False:
    
    path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/"
    path_to_mesh = path + "/MESH/"
    path_to_fenics = path_to_mesh + "FEniCS/"
    
    # Initial airways and reference mesh
    initial_airways = path_to_mesh + "skel.vtu"
    
    # Reference Nifti image     
    nifti_lung = path + "NIFTI/NEW_Mask_Exp.nii.gz"
    # Straight from MATLAB tetrahedral lung
    mat_lung = path_to_mesh + "/tet_lung.mat"
    # FEniCS processed tetrahedral lung
    vtu_lung = path_to_fenics+"mesh000000.vtu"

    # FEniCS processed triangle lung mesh 
    vtu_tri = path_to_fenics+"boundary_markers000000.vtu"

    # dummy vtu and npz for development purposes
    vtu_dummy = path_to_mesh+"dummy_geometry.vtu"
    npz_dummy = path_to_mesh+"dummy_geometry.npz"
    
    # Simplification of the triangle lung msh
    surf_ply = vtu_tri.split(".vtu")[0]+".ply"

    # Simplify the surface mesh for ray casting
    awm.rewrite_surface_mesh(vtu_tri, surf_ply)

    # Placeholder for the cloud of points
    npz_cloud = path_to_mesh+"cloud.npz"


    # Activate if you want to generate a newcloud of points for the airway 
    # generation scheme.
    if False:
        
        alv_density=1e-1
        cloud = awm.distribute_point_cloud(surf_ply, return_cloud=True, 
                                       vtuname = vtu_dummy, npzname=npz_dummy,
                                       alv_density=alv_density)
        np.savez(npz_cloud, cloud=cloud)
      
    else:
        
        npz = np.load(npz_cloud,allow_pickle=True)
        cloud = npz['cloud']
        
    # Read the txt and extract a finite element mesh (lines)
    TestTree = awm.initialize_tree_from_vtu(initial_airways)
    
    TestTree.set_strahler_order()

    TestTree.process_mesh(path_to_mesh+"skel_0.vtu",include_order=True)

    for gen in range(3):

        TestTree.generate_sythetic_branches(cloud, surf_ply,
                                            kmeans_coherence=True,
                                            branch_scaling_ratio=0.60)
        
        TestTree.set_strahler_order()
        TestTree.assign_synthetic_radii()
        TestTree.process_mesh(path_to_mesh+"skel_%i.vtu"%(gen+1),include_order=True)


# %% Evaluate a baseline airway mesh, trim it, and save it

if __name__ == '__main__' and True:
    
    max_trimming_attempts = 1
    execute_trimming = True
    
    path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG4/ARDSnet/"
    path_to_mesh = path + "/MESH-15/"
    
    if os.path.isdir(path_to_mesh): print("Yes")
    
    print(os.listdir(path_to_mesh))
    
    # Initial airways and reference mesh
    initial_airways = path_to_mesh + "skel.vtu"
    reference_mesh = path_to_mesh+"FEniCS/mesh000000.vtu"
    out_airways = initial_airways.split(".vtu")[0]+"_trimmed.vtu"
    
    # Load an airway tree and determine the subdomain distribution
    TestTree = awm.initialize_tree_from_vtu(initial_airways)
    terminals = TestTree.distal_points
    subdomains, counts, volumes = awm.distribute_subdomains(reference_mesh, terminals)
    
    q50 = np.median(counts)
    q20 = np.quantile(counts,0.20)
    q25 = np.quantile(counts,0.25)
    q75 = np.quantile(counts,0.75)
    
    print("\n")
    print("-"*40)
    print("| Initial subdomain analysis:"+10*" "+"|")
    print("-"*40)
    print(" > Number of null subdomains: %i/%i"%(np.count_nonzero(counts==0),len(counts)))
    print(" > Minimum elem#/subd: %i"%np.min(counts))
    print(" > Median (IQR) elem#/subd: %i (%i - %i)"%(q50,q25,q75))
    print(" > Maximum elem#/subd: %i"%np.max(counts))

    awm.determine_equivalent_resistance(initial_airways)
    
    print("*"*60)
    
    template_name = initial_airways.split(".vtu")[0]
    
    if execute_trimming:
        
        for i in range(max_trimming_attempts):
            
            print("-"*20)
            print("\nIteration #%i"%(i+1))
            if i == 0:
                in_airways = initial_airways
            else:
                in_airways = out_airways
            out_airways = template_name+"_temp.vtu"
                
            out_analysis = awm.trim_airways_with_small_subdomains(in_airways,
                                                       reference_mesh,
                                                       out_airways, 
                                                       verbose=False,
                                                       out_analysis=True,
                                                       return_out_analysis=True)
            
            _, counts, _ = out_analysis
            q50 = np.median(counts)
            q20 = np.quantile(counts,0.20)
            q25 = np.quantile(counts,0.25)
            q75 = np.quantile(counts,0.75)
            
            print("\n")
            print("-"*40)
            print("| #%i-th subdomain analysis:"%i+10*" "+"|")
            print("-"*40)
            print(" > Number of null subdomains: %i/%i"%(np.count_nonzero(counts==0),len(counts)))
            print(" > Minimum elem#/subd: %i"%np.min(counts))
            print(" > Median (IQR) elem#/subd: %i (%i - %i)"%(q50,q25,q75))
            print(" > Maximum elem#/subd: %i"%np.max(counts)) 
            
        awm.determine_equivalent_resistance(out_airways)


# %% Generate figures that compare airway trees

if __name__ == '__main__' and False:
    
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(ncols=3,figsize=(22,6),dpi=200)

    # target geometries
    # LENOVO LAPTOP
    paths = ["C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/ARDSnet/MESH/old-skeletonization/skel.vtu",
             "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/ARDSnet/MESH/skel.vtu"]
#              "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG3/ARDSnet/MESH-15/skel.vtu",

    # # MSI LAPTOP
   # paths = ["C:/Users/Agustin/Documents/TemporalData/skel-1.vtu",
   #          "C:/Users/Agustin/Documents/TemporalData/base-skel.vtu"]
    
    # x-axis control, Strahler order or airway generation "Order" "Generation"
    strat = "Generation" # Order Generation
    max_strata = 6
    
    # color control
    colors = ["tab:red","tab:blue","orange"]
    
    # boxplot control
    dx = 0.15
    x0 = -0.05
    
    mu = 1.825e-8
    # test_path =  "C:/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/BASE-GEOMETRIES/AIRWAYS/dummy-skel.vtu"
    
    # figure limits
    ylims = {"res":(1e-10,1e-4),
             "length":(0,150),
             "radius":(-1,4)}
        
    fields = ["length","radius","res"]    
     
    subjs = ["Old","New"] # PIG3
    
    # legend-related
    patches = [mpatches.Patch(color=colors[e],label=subj) for e,subj in enumerate(subjs)]
    
    data_holder = {subj:{} for subj in subjs}
    
    # generate analysis and figure
    for e,(test_path,subj,color) in enumerate(zip(paths,subjs,colors)):
        
        # load tree
        tree = awm.initialize_tree_from_vtu(test_path)
        # determine generations
        generations = np.zeros(len(tree.branches.keys()),dtype=int)
        for bid in tree.branches:
            if bid == 0:
                generations[bid] = 0
            else:
            
                branch = tree.branches[bid]
                pid = branch.parent
                nsisters = len(tree.branches[pid].children)
                if nsisters>1:
                    generations[bid] = generations[pid]+1
                else:
                    generations[bid] = generations[pid]
            
        # extract ready-to-use data
        mesh = io.read(test_path)
        #order = mesh.cell_data['order'][0]
        radius = mesh.cell_data['radius'][0]
        length = mesh.cell_data['length'][0]
        
        # adjust labels
        renamer = {"res":"Poiseuille resistance (kPa.s/mm3)",
                   "radius":"Log(Diameter)",
                   "length":"Airway length (mm)"}
        
        # determine resistance
        resistances = 8*mu*length/(np.pi*radius**4)
        
        if strat == "Generation":
            print("Using '%s' as stratificator"%strat)
            stratificator = generations
            xlabel = "Generation (-)"
            xmin = 0
        elif strat == "Order":
            print("Using '%s' as stratificator"%strat)
            stratificator = order
            xlabel = "Strahler order (-)"
            xmin = 1
        else:
            stratificator = None
            print("Invalid stratificator '%s'"%strat)
    
        # determine xlimits
        strata = np.unique(stratificator)
    
        # sort fields per strata
        sorter = {field:{s:[] for s in strata} for field in fields}
        
    
        # extract data
        for s in strata:
            mask = stratificator==s
            for field,data in zip(fields,[length,np.log(2*radius),resistances]):
                sorter[field][s] = data[mask]
        
        # determine area per strata
        sorter.update({"area":{s:np.pi*np.sum(sorter["radius"][s]**2) for s in strata}})

        data_holder[subj].update({"radius":radius,
                                  "resistance":resistances,
                                  "length":length,
                                  "generation":generations})

        # generate subplots
        for ax, field in zip(axes,fields):
            
            print("field: %s - start"%field)
            
            for s in strata:
                
                n = len(sorter[field][s])
                #ax.scatter([o-0.1]*n,sort_resistances[o],color="k")
                box = ax.boxplot(sorter[field][s], positions=[s+dx*e+x0],vert=True,showfliers=False, patch_artist=True)
            
                for b in box['boxes']:
                    b.set_alpha(0.3)
                    b.set_color(color)
                for b in box['whiskers']: b.set_alpha(0.5)
                for b in box['caps']: b.set_alpha(0.5)       
                for b in box['medians']: b.set_color(color)
    
    
            if field == "radius" and strat=="Generation":
                if max(strata) < 10:
                    ax.plot(strata, strata*-0.0438+1.3094, ls="--",color="k")
                    ax.fill_between(strata,strata*-0.0438+1.3094-0.0494,strata*-0.0438+1.3094+0.0494,color="k",alpha=0.2)

        
            if field == "length" and strat=="Generation":
                xmax = np.max(strata) if np.max(strata)<5 else 5
                xrange = np.arange(0,xmax)
                ax.plot(xrange,-7.5186*xrange+37.556, ls="--",color="k")


            yscale = "log" if field == "res" else "linear"
            ax.set_yscale(yscale)
            ax.set_xticks(strata)
            ax.set_xticklabels(["%i"%s for s in strata])
            xlims = (xmin-0.5, max_strata+0.5)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims[field])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(renamer[field])
            
            print("field: %s - end"%field)
        
        plt.subplots_adjust(hspace=0.40)
        plt.legend(handles=patches)
        



# %% Generate initial skel.vtu

if __name__ == '__main__' and False:
    
    
    path_to_mesh = "../testing-data/AIRWAY-MESHES/dev/"

    '''
    These are the meshes generated in CGAL
    texts = ['correspondance-lcc.polylines.txt', 
             'correspondance-poly.polylines.txt', 
             'correspondance-sm.polylines.txt', 
             'skel-lcc.polylines.txt', 
             'skel-poly.polylines.txt', 
             'skel-sm.polylines.txt']
    '''
    
    path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/ARDSnet/"
    path_to_mesh = path + "/MESH/"
    path_to_fenics = path_to_mesh + "FEniCS/"
    
    # Initial airways and reference mesh
    initial_airways = path_to_mesh + "skel.vtu"
    
    # Full path to the skeletonization
    skel_txt = path_to_mesh + "skel-sm.polylines.txt"
    corr_txt = path_to_mesh + "correspondance-sm.polylines.txt"
    
    # Reference Nifti image     
    nifti_lung = path + "NIFTI/NEW_Mask_Exp.nii.gz"
    # Straight from MATLAB tetrahedral lung
    mat_lung = path_to_mesh + "/tet_lung.mat"
    # FEniCS processed tetrahedral lung
    vtu_lung = path_to_fenics+"mesh000000.vtu"

    # FEniCS processed triangle lung mesh 
    vtu_tri = path_to_fenics+"boundary_markers000000.vtu"


    
    '''
    # Simplification of the triangle lung mesh
    surf_ply = vtu_tri.split(".vtu")[0]+".ply"

    # Simplify the surface mesh for ray casting
    awm.rewrite_surface_mesh(vtu_tri, surf_ply)
    
    # Activate if you want to generate a newcloud of points for the airway 
    # generation scheme.
    if False:
        alv_density=1
        cloud = awm.distribute_point_cloud(surf_ply, return_cloud=False, 
                                       vtuname = vtu_dummy, npzname=npz_dummy,
                                       alv_density=alv_density)
        
    else:
        
        npz = np.load(npz_cloud)
        cloud = npz['cloud']
    '''

    # Read the txt and extract a finite element mesh (lines)
    points, elem = awm.skel_to_mesh(skel_txt,skel_mesh=None)
    
    # Transform the skeleton mesh (traslation and scaling) using lung meshes
    points = awm.transform_skel(mat_lung, vtu_lung, nifti_lung, points, elem, 
                            outname=path_to_mesh+"raw_skel.vtu",
                            scale_airways_with_affine=True)
    
    # Retrieve topological information
    out = awm.classify_skel_nodes(points, elem)
    
    # Generate a tree using the top of the trachea as seed
    seed = out['inlet_id']
    airway_tree = awm.Tree(seed, points,elem, branch_length_threshold=1.50)
    # Initialize the skeleton analysis
    airway_tree.activate_seed() 
    # Create branches from skeletonization
    airway_tree.grow_tree() 
    # Determine radius from the skeletonization
    airway_tree.radius_from_skeletonization(skel_txt, corr_txt, nifti_lung)
    # Assign Strahler order
    #airway_tree.set_strahler_order()

    # Export mesh
    airway_tree.process_mesh(initial_airways,include_order=False)

# %% Pilot for branch trimming
# This method must be included later in the mesh generation pipeline.

if __name__ == "__main__" and False:
    
    max_trimming_attempts = 10
    
    path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG4/ARDSnet/"
    path_to_mesh = path + "/MESH-15/"
    
    if os.path.isdir(path_to_mesh): print("Yes")
    
    print(os.listdir(path_to_mesh))
    
    # Initial airways and reference mesh
    initial_airways = path_to_mesh + "skel.vtu"
    reference_mesh = path_to_mesh+"FEniCS/mesh000000.vtu"
    out_airways = path_to_mesh + "trim_skel.vtu"
    
    # Load an airway tree and determine the subdomain distribution
    TestTree = awm.initialize_tree_from_vtu(initial_airways)
    terminals = TestTree.distal_points
    subdomains, counts, volumes = awm.distribute_subdomains(reference_mesh, terminals)
    
    q50 = np.median(counts)
    q20 = np.quantile(counts,0.20)
    q25 = np.quantile(counts,0.25)
    q75 = np.quantile(counts,0.75)
    
    print("\n")
    print("-"*40)
    print("| Initial subdomain analysis:"+10*" "+"|")
    print("-"*40)
    print(" > Number of null subdomains: %i/%i"%(np.count_nonzero(counts==0),len(counts)))
    print(" > Minimum elem#/subd: %i"%np.min(counts))
    print(" > Median (IQR) elem#/subd: %i (%i - %i)"%(q50,q25,q75))
    print(" > Maximum elem#/subd: %i"%np.max(counts))
   
    template_name = initial_airways.split(".vtu")[0]
    
    for i in range(max_trimming_attempts):
        
        print("-"*20)
        print("\nIteration #%i"%(i+1))
        if i == 0:
            in_airways = initial_airways
        else:
            in_airways = out_airways
        out_airways = template_name+"_temp.vtu"
            
        out_analysis = awm.trim_airways_with_small_subdomains(in_airways,
                                                   reference_mesh,
                                                   out_airways, 
                                                   verbose=False,
                                                   out_analysis=True,
                                                   return_out_analysis=True)
        
        _, counts, _ = out_analysis
        q50 = np.median(counts)
        q20 = np.quantile(counts,0.20)
        q25 = np.quantile(counts,0.25)
        q75 = np.quantile(counts,0.75)
        
        print("\n")
        print("-"*40)
        print("| #%i-th subdomain analysis:"%i+10*" "+"|")
        print("-"*40)
        print(" > Number of null subdomains: %i/%i"%(np.count_nonzero(counts==0),len(counts)))
        print(" > Minimum elem#/subd: %i"%np.min(counts))
        print(" > Median (IQR) elem#/subd: %i (%i - %i)"%(q50,q25,q75))
        print(" > Maximum elem#/subd: %i"%np.max(counts)) 


# %% Compare two different pipelines
# MESH: Using CGAL only
# MESH-2: Using Iso2Mesh for surface, CGAL for skel
# I need to see the differences between both processing
# Iso2Mesh data is voxel-located while CGAL is localized
# according to the NIFTI image affine.

if __name__ == "__main__" and False:
    
    path_to_data = "C:/Users/angus/Downloads/CORNELLU-PIGS/"
    
    subject = "Subject-02"
    state = "02-ARDSnet-MV"
    group = "TreatmentGroup-A"
    
    target = "%s/%s/%s/"%(group,subject,state)
    
    path = path_to_data + target
    
    
        
    skel_cgal = path + "MESH/skel-sm.polylines.txt"
    skel_iso2mesh = path + "MESH-2/skel-sm.polylines.txt"
    
    points_c, elem_c = awm.skel_to_mesh(skel_cgal)
    points_i, elem_i = awm.skel_to_mesh(skel_iso2mesh)
    
    reference_image = path + "NIFTI/Exp.nii.gz"
    
    # Retrieve the image affine
    img = nib.load(reference_image)
    aff = img.affine
    diag = aff.diagonal().__abs__()[:3]
    
    def determine_bbox(points): 
        bx = np.array([points[:,0].min(),points[:,0].max()])
        by = np.array([points[:,1].min(),points[:,1].max()])
        bz = np.array([points[:,2].min(),points[:,2].max()])
        return bx, by, bz
    
    # Determine bounding boxes
    bbox_c = determine_bbox(points_c)
    bbox_i = determine_bbox(points_i)
    
    for i in range(3):
        H_i = (bbox_i[i][1]-bbox_i[i][0])*diag[i]
        H_c = (bbox_c[i][1]-bbox_c[i][0])
        print("%.1f, %.1f"%(H_i, H_c))
        
    '''
    
    Specified bounds:
         CGAL // Iso2Mesh
        140.8     145.4
        112.8     117.2
        222.6     222.0
        
        The bounds are similar but not the same.  The 
    differences may be originated in the meshing stage,
    where diferent element sizes are used
        
    '''
    
    # Scale the points ussing the affine
    scaled_points_i = points_i*diag
    
    # Plot the scaled values
    fig,ax = plt.subplots(figsize=(4,8))
    a1 = 0; a2 = 1;
    ax.scatter(points_c[:,a1],points_c[:,a2], color="r",alpha=0.2)
    ax.scatter(scaled_points_i[:,a1],scaled_points_i[:,a2], color="b",alpha=0.2)
    
    '''
    This transformation worked perfectly. A multiplication
    by the affine matrix under absolute value yields 
    well located results. 
    
    The CGAL-only proccesed values extend further than 
    the Iso2Mesh-CGAL values.
    
    '''

    print()