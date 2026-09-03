# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:47:49 2024

@author: angus
"""

import vcv_lung_var_stiffness
import supportfunctions as sf
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit

# %%

def create_flow_function(time_arr, flow, t_cycle = 2.00,t_end_inspiration = 0.375, 
                  t_insp_transition = 0.470, t_end_insp_pause = 0.750, 
                  t_exp_start = 0.810):

    # Inspiration Exponential functoin
    mask = np.array(time_arr<t_end_inspiration)
    def func1(x, a, b): return -a *(1 - np.exp(-b * x))
    popt1, pcov = curve_fit(func1, time_arr[mask], flow[mask])
    # Linear decay
    mask = np.logical_and(time_arr>t_end_inspiration,time_arr<t_insp_transition)
    def func2(x, a, b): return (a * x + b)
    popt2, pcov = curve_fit(func2, time_arr[mask], flow[mask])
    # Inspiratory pause, zero flux
    def func3(x): return 0.0
    # Expiration linear function
    mask = np.logical_and(time_arr>t_end_insp_pause,time_arr<t_exp_start)
    def func4(x, a, b): return (a * x + b)
    popt4, pcov = curve_fit(func4, time_arr[mask], flow[mask])
    # Expiration linear function
    mask = np.logical_and(time_arr>t_exp_start,time_arr<t_cycle)
    def func5(x, a, b): return (a * x + b)
    popt5, pcov = curve_fit(func5, time_arr[mask], flow[mask])

    # Definition of the flow function
    def flow_func(x, tcycle=2.0):
        
        if x>tcycle:
            x = x%t_cycle
        
        if x<t_end_inspiration:
            return func1(x,*popt1)
        elif x>=t_end_inspiration and x<t_insp_transition:
            return func2(x,*popt2)
        elif x>=t_insp_transition and x<t_end_insp_pause:
            return func3(x)
        elif x>=t_end_insp_pause and x<t_exp_start:
            return func4(x,*popt4)
        elif x>=t_exp_start and x<=t_cycle:
            return func5(x,*popt5)
        else:
            return 0.0
        
    return flow_func

def create_pressure_function(time_arr, pressure, 
                             t_cycle=2.00,
                             t_plateau_end=0.75, 
                             t_break1=0.81, 
                             t_break2=1.20,
                             peep_target=5.0):
    """
    Approximate the expiration pressure decay with three linear segments.
    The plateau pressure is passed dynamically at evaluation time.
    """

    # Define each segment and fit a linear function to each
    def linear(x, a, b):
        return a * x + b

    # Segment 1: t_plateau_end → t_break1
    mask1 = (time_arr >= t_plateau_end) & (time_arr < t_break1)
    t1 = time_arr[mask1]
    p1 = pressure[mask1]
    popt1, _ = curve_fit(linear, t1, p1)

    # Segment 2: t_break1 → t_break2
    mask2 = (time_arr >= t_break1) & (time_arr < t_break2)
    t2 = time_arr[mask2]
    p2 = pressure[mask2]
    popt2, _ = curve_fit(linear, t2, p2)

    # Segment 3: t_break2 → t_cycle
    mask3 = (time_arr >= t_break2) & (time_arr <= t_cycle)
    t3 = time_arr[mask3]
    p3 = pressure[mask3]
    popt3, _ = curve_fit(linear, t3, p3)

    # Save original plateau value for scaling
    plateau_measured = pressure[time_arr >= t_plateau_end][0]

    def pressure_func(t, plateau_pressure_sim):
        t_mod = t % t_cycle  # Wrap to cycle

        # Determine scale factor
        scale = plateau_pressure_sim / plateau_measured

        if t_mod < t_plateau_end:
            return None  # Define inspiratory segment later

        elif t_plateau_end <= t_mod < t_break1:
            return linear(t_mod, *popt1) * scale

        elif t_break1 <= t_mod < t_break2:
            return linear(t_mod, *popt2) * scale

        elif t_break2 <= t_mod <= t_cycle:
            return linear(t_mod, *popt3) * scale

        else:
            return None

    return pressure_func

if __name__ == "__main__":


    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
    
    
   # Simulation Codename
    codename = "PIG5-medium-calibrated" # This is a name for the folder where the output is directed
    mesh_name = "" #  This should change for different states/subjects; While in development, just keep 'stable'
    case ="FEniCS" # This is the specific name for the mesh in use
    
    # Declare the path to the folder
    #path_to_mesh = "/mnt/c/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/%s/%s"%(packname,mesh_packs[packname])
    path_to_mesh = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/medium/"
    path_to_airway = path_to_mesh+"skel.vtu"
  
    # Direct the output of this execution towards this folder
    output_to = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    
    # Checkpoint parameters
    restart_from_last_checkpoint = False
    save_checkpoints = True
    save_vtk = True
    
    # Processing the paths
    path_to_mesh =  sf.manage_mesh_directory(path_to_mesh,mesh_name,case)
    output_to = sf.manage_output_directory(output_to,codename,restart_from_last_checkpoint)
      
    # Quick parameter setting
    K_cw_stiffness = 0.0058
    K_d_stiffness = 0.0008
    
    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = "ma"
    c = 15.6735 # [factor is 3.25 for VT30]
    stiffness_file = "c_tissue.xml.gz"
    beta = 1.075
    isotropic_prestrain = 1.0345
    
    # Temporal management
    ncheckpoints =[2, 10, 2, 10,  4,  4]    # These will be used to save some checkpoints
    ninternaldivs=[5, 20, 8,  8, 20, 50] # Funciona para pack-main

    # Permeability config
    KK_exp = 3
    KK_factor = 1.0
    permeability_dict = {"variable_permeability":False,
                         "permeability_file":"k0.xml",
                         "KK_exp":KK_exp,
                         "KK_factor":KK_factor,
                         }
    
    # Porosity configo
    porosity_config = {"activate":True, 
                       "mean":0.50, # Dummy
                       "file":"phi0.xml.gz"}

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
    ncycles = 1
    Tsyr = 0.375;
    Tpausa = 0.375;
    Texp = 1.25;
    
    time_config = {"ncycles":ncycles,
                   "Tsyr":Tsyr,
                   "Tpausa":Tpausa,
                   "Texp":Texp}

    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":True,
                                               "expiratory_gamma":1.835, # 0.327 original value
                                               "inspiratory_gamma":0.3160, # 0.327 original value
                                               "tolerance":1e-8,
                                               "nitmax":100}}
                              



    # Path to experimental data, load matlab file and prepare data
    signal_path = '/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.npz'

    # Load the experimental signal
    npz = np.load(signal_path)
    aw_pressure = npz['pressure']/10.1972 # kPa
    flow = npz['flow']*(1e6)
    time = npz['time']

    # Define flow and pressure functions
    flowfn = create_flow_function(time, flow,
                           t_cycle = 2.00,t_end_inspiration = 0.375, 
                           t_insp_transition = 0.470, t_end_insp_pause = 0.750, 
                           t_exp_start = 0.810)
    
    presfn = create_pressure_function(time, aw_pressure,t_break1 = 0.80,t_break2=0.85)

    # Generate flow function dictionary
    signal_boundary_conditions = {'activate':True,
                                  'flow_func':flowfn,
                                  'pres_func':presfn}
    
    
    
    # Tidal volume
    vt = 0.40 # L, program takes input in mm3 so it's multiplied by 10**6 



    args = {"restart_from_last_checkpoint":restart_from_last_checkpoint,
            "save_checkpoints":save_checkpoints,
            "mesh_dir":path_to_mesh,
            "output_to":output_to,
            "ncheckpoints":ncheckpoints,
            "ninternaldivs":ninternaldivs,
            "K_cw_stiffness":K_cw_stiffness,
            "K_d_stiffness":K_d_stiffness,
            "permeability_dict":permeability_dict,
            "isotropic_prestrain":isotropic_prestrain,
            "solver_type":solver_type,
            "solver_dict":solver_dict,
            "tidal_volume":vt*(10**6), # conversion from L to mm3
            "time_config":time_config,
            "porosity_config":porosity_config,
            "constitutive_model":cm,
            "constitutive_parameters":(c,beta),
            "stiffness_file":stiffness_file,
            "path_to_airway":path_to_airway,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,
            "additional_resistances":additional_resistances,
            "bc_dict":signal_boundary_conditions,
            "inspiratory_pause_stop":False,
            }
    
    vcv_lung_var_stiffness.execute_vcv_simulation(args)
