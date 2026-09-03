# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:30:51 2025

@author: angus
"""

import os
import sys
sys.path.append("./calibrate/")
import numpy as np
import BOStaticCalibration as bosc


if __name__ == "__main__":
    
    # Parallel implementation    
    parallel = True
    if parallel:
        ncpus = 3#cpu_count()
    else:
        ncpus = 1
    
    # Number of initial samples in LHS
    n_samples = 15
    
    # Number of attempts
    batchsize_explore = 3
    batchsize_exploit = 3
    exploring_rounds =  6
    exploiting_rounds = 3 
    verbose = True
    
    # Execution root
    terminal_execution = True
    if terminal_execution:
        root = "/mnt/c/"    
    else:
        root = "C:/"
        
    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = root+"Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"    

    
    # Finish the mesh path
    path_to_mesh += "%s/"%case
    
    # Deliver the output files there
    output_to = root+"Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    task_name = "bayesian-static-cal-%3.3i/"
    
    # Additional folders where to obtain data from previous simulations to further 
    # feed the optimization algorithm
    optimization_folders = [root+output_to+task_name%1]

    # Path to experimental data, load matlab file and prepare data
    signal_path = root+'Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.npz'
    
    signal_configuration = {"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.375, "type": "exp"},
                                {"t_start": 0.375, "t_end": 0.470, "type": "linear"},
                                {"t_start": 0.470, "t_end": 0.750, "type": "zero"},
                                {"t_start": 0.750, "t_end": 0.810, "type": "linear"},
                                {"t_start": 0.810, "t_end": 2.000, "type": "linear"},],
                            "pressure_segments":[
                                {"t_start": 0.750, "t_end": 0.800, "type": "linear"},
                                {"t_start": 0.800, "t_end": 0.850, "type": "linear"},
                                {"t_start": 0.850, "t_end": 0.900, "type": "linear"},
                                {"t_start": 0.900, "t_end": 0.950, "type": "linear"},
                                {"t_start": 0.950, "t_end": 2.000, "type": "linear"},]
                                }
    
    
    paths = {"output_root":output_to,
             "task_name":task_name,
             "path_to_mesh":path_to_mesh,
             "path_to_airways":path_to_airway,
             "path_to_signal":signal_path,
             }
    
    # Define and present optimization parameters
    print("Number of LH samples: %i"%n_samples)
    print("Exploring phase:")
    print(" > Batch size: %i"%batchsize_explore)
    print(" > N° rounds: %i"%exploring_rounds)
    print(" > N° exploring attempts: %i"%(batchsize_explore*exploring_rounds))

    print("Exploiting phase:")
    print(" > Batch size: %i"%batchsize_exploit)
    print(" > N° rounds: %i"%exploring_rounds)
    print(" > N° exploiting attempts: %i"%(batchsize_exploit*exploiting_rounds))

    # Wave configuration
    peep = 10.8 # placeholder (cmH2O) PEEP value
    pplat = 20.6# (cmH2O)
    ppeak = 33.8 # (cmH2O)
    vt = 0.4011
    Tsyr = 0.375;
    Tpausa = 0.375;
    Texp = 1.25;
    ncycles=1
    
    # Physical parameters    
    K_cw_stiffness = 0.02771
    K_d_stiffness = K_cw_stiffness/10.
    
    # Airway constants
    expiratory_gamma = 0.200
    inspiratory_gamma = 0.200
    
    # Parenchyma configuration
    constitutive_model = "bir2019"
    c_tissue = 2.1169
    KK_exp = 5
    KK_factor = 1.0
    isotropic_prestrain = 1.03257    
    
    # Evaluate if indicated files exist
    print("Checking for the existence of some indicated files:")
    
    for file,name in zip([signal_path, path_to_airway],
                         ["Signal","Airways"]):
        if not os.path.isfile(file): 
            print(" The file '%s' was not found at %s"%(name,file))
            print(" Stop the simulation")
        else:
            print(" %s found"%name)
    
    
    npz = np.load(signal_path)
    flow = npz['flow']
    pres = npz['pressure']
    time = npz['time']
    
    wave_config = {"vt":vt,
                   "Tsyr":Tsyr,
                   "Tpausa":Tpausa,
                   "Texp":Texp,
                   "ncycles":ncycles,
                   "pplat":pplat,
                   "ppeak":ppeak,
                   "peep":peep,
                   "signal_config":{"flow":flow,
                                    "pressure":pres,
                                    "time":time,
                                    "setup":signal_configuration},
                   }

    
    parenchyma_config = {"variable_porosity":True,
                         "porosity_mean":0.5,
                         "porosity_file":"phi0.xml.gz",
                         "variable_permeability":True,
                         "permeability_file":"k0.xml",
                         "permeability_exp":KK_exp,
                         "permeability_factor":KK_factor,
                         "c_tissue":c_tissue,
                         "constitutive_model":constitutive_model,
                         "isotropic_prestrain":isotropic_prestrain
                         }
    
    peripheral_config = {"K_cw_stiffness":K_cw_stiffness,
                         "K_d_stiffness":K_d_stiffness,
                         "pedley_activate":True,
                         "pedley_expiratory_gamma":expiratory_gamma,
                         "pedley_inspiratory_gamma":inspiratory_gamma,
                         "pedley_tolerance":1e-8,
                         "pedley_nitmax":100}
    
    # continue_calibration: Boolean, to restart calibration from some intermediate point
    # restart_from: Integer, number of a calibration process
    # maximum_evaluations: Integer, maximum number of calibration processes to be checked
    # target_Ccw: Float, Chest-wall compliance to be targeted
    # target_Cl: Float, Lung compliance to be targeted
    # x0, x1, x2: Float tuple, initial guesses for C_birzle and K_stiffness
    # parallel_simulations: Boolean, execute both required simulations per step in parallel
    # neldermead_tolerance: Float, tolerance required to stop NM loop (misfit-related)
    # neldermead_nitmax: Integer, maximum number of NM iterative cycles.
    
    calibration_config = {"continue_calibration":False,
                          "save_vtk":True,
                          "ninternaldivs":[5,10,10,50,20,45],
                          "ncheckpoints":[2,10,2,2,4,4],
                          "paths":paths
                          }

    parameters = {"lhs":{"nsamples":n_samples},
                  "explore":{"batchsize":batchsize_explore,
                             "nrounds":exploring_rounds},
                  "exploit":{"batchsize":batchsize_exploit,
                             "nrounds":exploiting_rounds},
                  "bounds":{"c_tissue":(1.5,4.0),
                            "Kcw":(0.005,0.08),
                            "Kd":(0.001,0.08),
                            "alpha":(1.010,1.040)
                            },
                  "configuration":{"parallel":parallel,
                                   "verbose":verbose,
                                   "ncpus":ncpus,
                                   "seed":10,
                                   "restart_task_id":None,
                                   },
                  "paths":paths,
                  "calibration_config":calibration_config,
                  "peripheral_config":peripheral_config,
                  "wave_config":wave_config,
                  "parenchyma_config":parenchyma_config,
                  "additional_simulations":optimization_folders,
                  }

    results, logs = bosc.bayesian_optimization(parameters)
