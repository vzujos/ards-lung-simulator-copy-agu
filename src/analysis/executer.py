# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:47:49 2024

@author: angus
"""

import vcv_lung
import supportfunctions as sf
import sys

if __name__ == "__main__":


    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
    
    
    
    mesh_packs = {"PACK-01":"MESH-25","PACK-02":"MESH-20","PACK-03":"MESH-10",
                  "PACK-04":"MESH-25","PACK-05":"MESH-20","PACK-06":"MESH-10",
                  "PACK-07":"MESH-25","PACK-08":"MESH-20","PACK-09":"MESH-10",
                  "PACK-10":"MESH-25","PACK-11":"MESH-20","PACK-12":"MESH-10",
                  "PACK-13":"MESH-25","PACK-14":"MESH-20","PACK-15":"MESH-10",
                  "PACK-16":"MESH-25","PACK-17":"MESH-20","PACK-18":"MESH-10",
                  }
    
    pack= int(sys.argv[1])
   # Simulation Codename
    packname = "PACK-01-%i"%pack
    codename = "cornell-%s"%packname # This is a name for the folder where the output is directed
    mesh_name = "" #  This should change for different states/subjects; While in development, just keep 'stable'
    case ="FEniCS" # This is the specific name for the mesh in use
    
    # Declare the path to the folder
    #path_to_mesh = "/mnt/c/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/%s/%s"%(packname,mesh_packs[packname])
    path_to_mesh = "/mnt/c/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/%s/%s"%(packname,"MESH-25")

    path_to_airway = "/mnt/c/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/%s/skel.vtu"%packname
  
    # Direct the output of this execution towards this folder
    output_to = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    
    # Checkpoint parameters
    restart_from_last_checkpoint = True
    save_checkpoints = True
    
    save_vtk = False
    
    # Processing the paths
    path_to_mesh =  sf.manage_mesh_directory(path_to_mesh,mesh_name,case)
    output_to = sf.manage_output_directory(output_to,codename,restart_from_last_checkpoint)
      
    # Quick parameter setting
    K_stiffness = 0.0168333 
    
    # Temporal management
    ncheckpoints =[2, 10, 2,  2, 4,  4]    # These will be used to save some checkpoints
   # ninternaldivs=[5, 200,10,200,50,250] # Intervals between checkpoints divisions
   # ninternaldivs=[5, 50,10,50,50,200] # Intervals between checkpoints divisions
    ninternaldivs=[5, 20, 5, 20,20,50] # Funciona para pack-main
    # ninternaldivs=[5, 20, 5, 40,30,75] # Intervals between checkpoints divisions # funciona para kappa 1e3

    #### only relevant if block_variable_permeability == True
    KK_exp = 3
    KK_factor = 1.0

    # Tolerances in the iterative cycles
    max_nit = 30
    tol_p_it = 1e-3
    tol_v_it = 1e-2
    
    # Selecting solver
    solver_type = "dict"
    solver_dict = {"nonlinear_solver":"snes",
                   "snes_solver":{"linear_solver":"mumps",
                                  "relative_tolerance":1e-6,
                                  "absolute_tolerance":1e-8,
                                  "maximum_iterations":60,
                                  "line_search":"bt",
                                  "report":True,
                                  "error_on_nonconvergence":True,
                                  "preconditioner":"default"}
                                  }
    
    # Time configuration for the volume-controlled ventilation
    ncycles = 2
    Tsyr = 0.375;
    Tpausa = 0.375;
    Texp = 1.25;
    
        #"upstream":5e-7,
                                #  "downstream":1e-4,
    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":True,
                                               "gamma":0.327,
                                               "tolerance":1e-8,
                                               "nitmax":100}}
                              
    permeability_dict = {"variable_permeability":True,
                         "permeability_file":"k0.xml.gz",
                         "KK_exp":KK_exp,
                         "KK_factor":KK_factor,
                         }
    
    porosity_dict = {"activate":True,
                     "mean":0.5,
                     "file":"phi0.xml.gz"}
    
    time_config = {"ncycles":ncycles,
                   "Tsyr":Tsyr,
                   "Tpausa":Tpausa,
                   "Texp":Texp}

    
    # Tidal volume
    vt = 0.228 # L, program takes input in mm3 so it's multiplied by 10**6 

    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = "bir2019"
    c = 1.913543 # [factor is 3.25 for VT30]
    beta = 1.075


    args = {"restart_from_last_checkpoint":restart_from_last_checkpoint,
            "save_checkpoints":save_checkpoints,
            "mesh_dir":path_to_mesh,
            "output_to":output_to,
            "ncheckpoints":ncheckpoints,
            "ninternaldivs":ninternaldivs,
            "K_stiffness":K_stiffness,
            "KK_exp":KK_exp,
            "KK_factor":KK_factor,
            "solver_type":solver_type,
            "solver_dict":solver_dict,
            "tidal_volume":vt*(10**6), # conversion from L to mm3
            "time_config":time_config,
            "constitutive_model":cm,
            "constitutive_parameters":(c,beta),
            "permeability_config":permeability_dict,
            "porosity_config":porosity_dict,
            "path_to_airway":path_to_airway,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,
            "additional_resistances":additional_resistances,
            }
    
    vcv_lung.execute_vcv_simulation(args)