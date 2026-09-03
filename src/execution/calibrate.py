# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:30:51 2025

@author: angus
"""

import os
import sys
sys.path.append("/home1/agustin.perez/model2/src/calibrate/")
import numpy as np
import BOCalibration as boc

parameters = {"PIG2-ARDSnet":{"Tsyr":0.435,
                           "Tpausa":0.560,
                           "Texp":1.870,
                           "Pplat":21.7,
                           "PEEP":11.0,
                           "Ppeak":26.8,
                           "Signal path":'model2/signals/PIG2-ARDSnet.npz',
                           "Target volume":0.3527,
                           },
           "PIG3-ARDSnet":{"Tsyr":0.455,
                           "Tpausa":0.435,
                           "Texp":1.100,
                           "Pplat":26.3,
                           "PEEP":11.3 ,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG3-ARDSnet.npz',
                           "Target volume":0.3870,
                           },
           "PIG4-ARDSnet":{"Tsyr":0.48,
                           "Tpausa":0.28,
                           "Texp":1.73,
                           "Pplat":18.5,
                           "PEEP": 10.7,
                           "Ppeak":24.8,
                           "Signal path":'model2/signals/PIG4-ARDSnet.npz',
                           "Target volume":0.411,
                           },
           "PIG5-ARDSnet":{"Tsyr":0.375,
                           "Tpausa":0.375,
                           "Texp":1.25,
                           "Pplat":20.6,
                           "PEEP": 10.8,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG5-ARDSnet.npz',
                           "Target volume":0.4011,
                           },
           "PIG6-ARDSnet":{"Tsyr":0.540,
                           "Tpausa":0.265,
                           "Texp":1.340,
                           "Pplat":22.7,
                           "PEEP": 10.7,
                           "Ppeak":25.7,
                           "Signal path":'model2/signals/PIG6-ARDSnet.npz',
                           "Target volume":0.2995,
                           }
           }

function_manager = {"PIG2":{"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.145, "type": "exp"},
                                {"t_start": 0.145, "t_end": 0.240, "type": "linear"},
                                {"t_start": 0.240, "t_end": 0.360, "type": "linear"},
                                {"t_start": 0.360, "t_end": 0.400, "type": "linear"},
                                {"t_start": 0.400, "t_end": 0.490, "type": "linear"},
                                {"t_start": 0.490, "t_end": 0.520, "type": "linear"},
                                {"t_start": 0.810, "t_end": 2.865, "type": "zero"},],
                                "pressure_segments":[{"t_start": 0.995, "t_end": 1.050, "type": "linear"},
                                {"t_start": 1.050, "t_end": 1.100, "type": "linear"},
                                {"t_start": 1.100, "t_end": 1.150, "type": "linear"},
                                {"t_start": 1.150, "t_end": 1.290, "type": "linear"},
                                {"t_start": 1.290, "t_end": 1.500, "type": "linear"},
                                {"t_start": 1.150, "t_end": 1.700, "type": "linear"},
                                {"t_start": 1.700, "t_end": 2.865, "type": "linear"},]
                                },
                        "PIG3":{"flow_segments": [
                                {"t_start": 0.0,   "t_end": 0.150, "type": "exp"},
                                {"t_start": 0.150, "t_end": 0.250, "type": "linear"},
                                {"t_start": 0.250, "t_end": 0.350, "type": "linear"},
                                {"t_start": 0.350, "t_end": 0.400, "type": "linear"},
                                {"t_start": 0.400, "t_end": 0.450, "type": "linear"},
                                {"t_start": 0.450, "t_end": 0.540, "type": "linear"},
                                {"t_start": 0.810, "t_end": 1.99, "type": "zero"},],
                                "pressure_segments":[
                                {"t_start": 0.890, "t_end": 0.920, "type": "linear"},
                                {"t_start": 0.920, "t_end": 0.950, "type": "linear"},
                                {"t_start": 0.950, "t_end": 1.020, "type": "linear"},
                                {"t_start": 1.020, "t_end": 1.300, "type": "linear"},
                                {"t_start": 1.300, "t_end": 1.99, "type": "linear"},]
                                },
                        "PIG4":{"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.075, "type": "linear"},
                                {"t_start": 0.075, "t_end": 0.120, "type": "linear"},
                                {"t_start": 0.120, "t_end": 0.440, "type": "linear"},
                                {"t_start": 0.440, "t_end": 0.465, "type": "linear"},
                                {"t_start": 0.465, "t_end": 0.490, "type": "linear"},
                                {"t_start": 0.490, "t_end": 0.550, "type": "linear"},
                                {"t_start": 0.550, "t_end": 0.810, "type": "zero"},
                                {"t_start": 0.810, "t_end": 2.49, "type": "linear"},],
                                "pressure_segments":[
                                {"t_start": 0.760, "t_end": 0.785, "type": "linear"},
                                {"t_start": 0.785, "t_end": 0.800, "type": "linear"},
                                {"t_start": 0.800, "t_end": 0.850, "type": "linear"},
                                {"t_start": 0.850, "t_end": 0.950, "type": "linear"},
                                {"t_start": 0.950, "t_end": 1.000, "type": "linear"},
                                {"t_start": 1.000, "t_end": 2.49, "type": "linear"},]
                                },
                        "PIG5":{"flow_segments":[
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
                                {"t_start": 0.950, "t_end": 2.000, "type": "linear"},
                                ]
                                },
                        "PIG6":{"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.075, "type": "exp"},
                                {"t_start": 0.075, "t_end": 0.470, "type": "linear"},
                                {"t_start": 0.470, "t_end": 0.485, "type": "linear"},
                                {"t_start": 0.485, "t_end": 0.510, "type": "linear"},
                                {"t_start": 0.510, "t_end": 0.570, "type": "linear"},
                                {"t_start": 0.575, "t_end": 0.810, "type": "zero"},
                                {"t_start": 0.810, "t_end": 2.145, "type": "linear"},],
                                "pressure_segments":[
                                {"t_start": 0.805, "t_end": 0.825, "type": "linear"},
                                {"t_start": 0.825, "t_end": 0.850, "type": "linear"},
                                {"t_start": 0.850, "t_end": 0.900, "type": "linear"},
                                {"t_start": 0.900, "t_end": 0.940, "type": "linear"},
                                {"t_start": 0.940, "t_end": 1.150, "type": "linear"},
                                {"t_start": 1.150, "t_end": 2.145, "type": "linear"},
                        ]}
                        
                }

if __name__ == "__main__":

    # Execution root
    terminal_execution = True
    if terminal_execution:
        root = "/mnt/c/"  
        root = "/home1/agustin.perez/"
    else:
        root = "C:/"    

    # PIG Number
    pig_number = 2
    protocol = "ARDSnet"
    codename = "PIG%i-%s"%(pig_number,protocol)
    
    # Extract data
    Tsyr = parameters[codename]["Tsyr"]
    Tpausa = parameters[codename]["Tpausa"]
    Texp = parameters[codename]["Texp"]
    pplat = parameters[codename]["Pplat"]
    peep = parameters[codename]["PEEP"] 
    ppeak = parameters[codename]["Ppeak"]
    vt = parameters[codename]["Target volume"]
    signal_path = root+parameters[codename]["Signal path"]
    

    # Cost function options
    # We'll use an open cost function that fits all of those parameters
    cost_function_options = {"include_volume_signal":True,
                             "include_pressure_signal":True,
                             "include_flux_signal":True,
                             "include_ppeak":True,
                             "include_pplat":True,
                             "include_peep":True,
                             "include_vpeak":True,
                             "w_vpeak":1.,}


    # Parameters info
    # Setting the parameters in order
    parameters_info = {"c_tissue":0,
                        "K_cw":1,
                        "K_d":2,
                        "alveolar_pressure":3,
                        "gamma_insp":4,
                        "gamma_exp":5,
                        }
    
    # We know the bounds (more or less) so we are accepting a small margin for those
    # Only the dynamic parameters (gamma*) are unknown.
    c_tissue = 10.9868
    K_cw = 0.001
    K_d = 0.001
    alveolar_pressure = 8.5
    gamma_insp = 0.20
    gamma_exp = 0.50
    bounds = {"c_tissue":(c_tissue-4.0,c_tissue+4.0),
            "K_cw":(0.001, 0.1),
            "K_d":(0.001, 0.1),
            "alveolar_pressure":(alveolar_pressure-2.0, alveolar_pressure+2.0),
            "gamma_insp":(gamma_insp-0.1, gamma_insp+0.2),
            "gamma_exp":(gamma_exp-0.2, gamma_exp+0.1),
            }
    
    # Parallel implementation    
    parallel = True
    if parallel:
        ncpus = 16 #cpu_count()
    else:
        ncpus = 1
    
    # Number of initial samples in LHS
    n_samples = len(bounds)*ncpus
    
    # Number of attempts
    batchsize_explore = ncpus
    batchsize_exploit = ncpus
    exploring_rounds =  8
    exploiting_rounds = 4 
    verbose = True
    
    # Skip LHS option
    bypass_lhs = False
    sourcing_path = "."

    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = root+"model2/geometries/PIG%i/ARDSnet/medium-coarse/"%pig_number
    path_to_airway = path_to_mesh+"skel.vtu"    

    
    # Finish the mesh path
    path_to_mesh += "%s/"%case
    
    # Deliver the output files there
    output_to ="./"
    task_name = "complete-%3.3i/"
    
    # Additional folders where to obtain data from previous simulations to further 
    # feed the optimization algorithm
    optimization_folders = []
    
    signal_configuration = function_manager["PIG%i"%pig_number]
    
    
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
    ncycles=1
    
    # Physical parameters    
    K_cw_stiffness = None
    K_d_stiffness = None
    
    # Airway constants
    expiratory_gamma = gamma_exp
    inspiratory_gamma = gamma_insp
    
    # Parenchyma configuration
    constitutive_model = "ma"
    c_tissue = None
    KK_exp = 5
    KK_factor = 1.0

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
                         "variable_permeability":False,
                         "permeability_file":"k0.xml",
                         "permeability_exp":KK_exp,
                         "permeability_factor":KK_factor,
                         "c_tissue":c_tissue,
                         "constitutive_model":constitutive_model,
                         "alveolar_pressure":alveolar_pressure,
                         "pleural_pressure_drop":10.0, # in cmH2O, transformed in BOCalibration|generate_vcv_dict
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
                          "save_vtk":False,
                          "ninternaldivs":[8,20,10,50,20,45],
                          "ncheckpoints":[2,10,2,2,4,4],
                          "paths":paths,
                          "parameters_info":parameters_info,
                          "cost_function_options":cost_function_options,
                          }

    parameters = {"lhs":{"nsamples":n_samples},
                  "explore":{"batchsize":batchsize_explore,
                             "nrounds":exploring_rounds},
                  "exploit":{"batchsize":batchsize_exploit,
                             "nrounds":exploiting_rounds},
                  "bounds":bounds,
                  "configuration":{"parallel":parallel,
                                   "verbose":verbose,
                                   "ncpus":ncpus,
                                   "seed":10,
                                   "restart_task_id":None,
                                   "bypass_lhs":{"activate":bypass_lhs,
                                                "path":sourcing_path,}
                                   },
                  "paths":paths,
                  "calibration_config":calibration_config,
                  "peripheral_config":peripheral_config,
                  "wave_config":wave_config,
                  "parenchyma_config":parenchyma_config,
                  "additional_simulations":optimization_folders,
                  }

    results, logs = boc.bayesian_optimization(parameters)
