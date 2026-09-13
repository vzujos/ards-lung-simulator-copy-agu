# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:47:49 2024

@author: angus
"""

import pcv_lung as pcv
import supportfunctions as sf
import sys
import numpy as np

if __name__ == "__main__":
    
    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
    
    # Simulation Codename
    codename = "CORNELL-PIG6-APRV" # This is a name for the folder where the output is directed
    mesh_name = "" #  This should change for different states/subjects; While in development, just keep 'stable'
    case ="FEniCS" # This is the specific name for the mesh in use
    
    
    # Declare the path to the folder
    #path_to_mesh = "/mnt/c/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/%s/%s"%(packname,mesh_packs[packname])
    path_to_mesh = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/ARDSnet/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"
  
    # Direct the output of this execution towards this folder
    output_to = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    # Path to signals
    signal_path = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/APRV/PIG6-APRV.mat"

    # Checkpoint parameters
    restart_from_last_checkpoint = False
    save_checkpoints = False
    save_vtk=True
    
    # Processing the paths
    path_to_mesh =  sf.manage_mesh_directory(path_to_mesh,mesh_name,case)
    output_to = sf.manage_output_directory(output_to,codename,restart_from_last_checkpoint)
      
    # Quick parameter setting
    K_stiffness = 0.055037  
    
    # Temporal management
    ncheckpoints = [2,4,2,2,2]    # These will be used to save some checkpoints
    ninternaldivs= [3,20,2,10,10] # Intervals between checkpoints divisions
    
    #### only relevant if block_variable_permeability == True
    KK_exp = 5
    KK_factor = 1.0
    
    # Tolerances in the iterative cycles
    max_nit = 50
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


    # Introducing a flow function to close the input signal to the prescribed flux
    from scipy.io.matlab import loadmat
    mat = loadmat(signal_path) 
    
    T =  947; cycles=1; dt = 0.0025
    t_cycle = T*dt

    measured_Paw = mat['Paw_rdata'].flatten()[:T*cycles]/10.1972
    measured_time = mat['time'].flatten()[:T*cycles]
    
    mask = measured_time<t_cycle
    min_pressure = np.min(measured_Paw[mask])
    displaced_Paw = measured_Paw - min_pressure
    
        
    # Time configuration for the volume-controlled ventilation
    ncycles = 1
    Tinsp = 2.1;
    Texp = t_cycle - Tinsp
    
    time_config = {"ncycles":ncycles,
                   "Tinsp":Tinsp,
                   "Texp":Texp,
                   "dT_ramp":0.10,
                   "dT_buffer":0.05}

    
    

    def pressure_function(time, pressure, t_cycle =T*dt ,
                          
                          t_enter_plateau = 0.10, 
                          t_leave_plateau = 2.0,
                          t_exp_intermediate = 2.1):
        # Determine the minimum pressure and zero-shift the values
        min_pressure = np.min(pressure)
        
        # Determine plateau value for pressure   
        mask = np.logical_and(time>t_enter_plateau,time<t_leave_plateau)
        mean_pressure = np.median(pressure[mask])
        print()

        # Initial positive slope
        in_slope = (mean_pressure-min_pressure)/t_enter_plateau
        def positive_slope_pressure(x): return (in_slope * x + min_pressure )       
        # Constant pressure
        def constant_pressure(x): return mean_pressure
        
        # Negative slope first region
        pos = np.argmin(np.abs(time-t_exp_intermediate))
        int_pressure = pressure[pos]
        
        out_slope1 = -(mean_pressure-int_pressure)/(t_exp_intermediate-t_leave_plateau)
        interc1 = -t_leave_plateau*out_slope1+mean_pressure
        
        def negative_slope_pressure1(x): return (out_slope1 * x + interc1)       
        
        out_slope2 = -(int_pressure-min_pressure)/(t_cycle-t_exp_intermediate)
        interc2 = -t_exp_intermediate*out_slope2+int_pressure
        def negative_slope_pressure2(x): return (out_slope2 * x + interc2)       

    
        # Definition of the flow function
        def pres_func(x, t_cycle=t_cycle):
            
            if x>t_cycle:
                x = x%t_cycle
            
            if x<t_enter_plateau:
                return positive_slope_pressure(x)
            elif x< t_leave_plateau:
                return constant_pressure(x)   
            elif x<t_exp_intermediate:
                return negative_slope_pressure1(x)
            else:
                return negative_slope_pressure2(x)
            
        return pres_func

    pf = pressure_function(measured_time, displaced_Paw)
    
    # Pressure config
    pmin = 0.0; pmax = 18.0; # in cmH2O
    pmax=pmax/10.1972 # from cmH2O to Kpa
    pressure_dict = {"pmin":pmin,
                     "pmax":pmax,
                     "activate_pfunction":True,
                     "pressure_function":pf}
    
    
    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = "bir2019"
    c = 2.13958027 # [factor is 3.25 for VT30]
    beta = 1.075

    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":True,
                                               "gamma":0.055,
                                               "tolerance":1e-8,
                                               "nitmax":100}}
    
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
            "pressure_dict":pressure_dict, # conversion from L to mm3
            "time_config":time_config,
            "constitutive_model":cm,
            "constitutive_parameters":(c,beta),
            "path_to_airway":path_to_airway,
            "additional_resistances":additional_resistances,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,

            }
    
    pcv.execute_pcv_simulation(args)
