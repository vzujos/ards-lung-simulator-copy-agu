# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:15:53 2025

@author: angus
"""

import numpy as np
from scipy.optimize import curve_fit
import sys
import os
sys.path.append("./calibrate/")
from shapeCalibration import execute_vcv_calibration


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

        
if __name__ == '__main__':
    
    # Calibration type "partial-signal" or "regression"
    calibration_type = "shape-calibration"
    
    # Target values
    target_Ccw = 367.8 # placeholder (mL/cmH2O) Chest wall compliance
    target_Cl = 22.1  # placeholder (mL/cmH2O) Lung compliance
    target_peep = 10.8 # placeholder (cmH2O) PEEP value
    target_pplat = 20.6# (cmH2O)
    target_ppeak = 33.8 # (cmH2O)
    targets = np.array([target_Ccw, target_Cl, target_peep])

    # Physical parameters    
    K_cw_stiffness = 0.066980
    K_d_stiffness = K_cw_stiffness/10.
    
    expiratory_gamma = 0.200
    inspiratory_gamma = 0.200

    constitutive_model = "bir2019"
    c_tissue = 2.1506
    KK_exp = 5
    KK_factor = 1.0
    isotropic_prestrain = 1.03213
    rho = 5e-2
    
    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"    

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
    # Finish the mesh path
    path_to_mesh += "%s/"%case
    
    # Deliver the output files there
    output_to = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    task_name = "shapecal-vcv-%2.2i/"
    
    # Registration-associated geometries
    registration_tetra_geom = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/Exp_NEW.npz"
    registration_tri_geom = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/Surface_Insp.ply"
    # Nifti images
    nifti_cpp = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/cpp_9000-000666.nii.gz"
    nifti_exp = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/NEW_Mask_Exp.nii.gz"


    # Evaluate if indicated files exist
    print("Checking for the existence of some indicated files:")
    
    for file,name in zip([signal_path, path_to_airway],
                         ["Signal","Airways"]):
        if not os.path.isfile(file): 
            print(" The file '%s' was not found at %s"%(name,file))
            print(" Stop the simulation")
        else:
            print(" %s found"%name)
    
    
    # Organization of variables
    paths = {"path_to_mesh":path_to_mesh,
             "path_to_airways":path_to_airway,
             "output_to":output_to,
             "task_name":task_name,
             "registration_hq_tetra":registration_tetra_geom,
             "registration_tri":registration_tri_geom,
             "nifti_cpp":nifti_cpp,
             "nifti_exp":nifti_exp,
             }
    
    # Wave config
    vt = 0.221
    Tsyr = 0.375;
    Tpausa = 0.375;
    Texp = 1.25;
    ncycles=1
    
    wave_config = {"vt":vt,
                   "Tsyr":Tsyr,
                   "Tpausa":Tpausa,
                   "Texp":Texp,
                   "ncycles":ncycles,
                   "pplat":target_pplat,
                   "ppeak":target_ppeak,
                   "peep":target_peep,
                   "bc_dict":signal_boundary_conditions}
    
    
    x0 = (0.020, 0.0080)
    x1 = (0.023, 0.0120)
    x2 = (0.025, 0.0100)
    
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
    
    calibration_config = {"calibration_type":calibration_type,
                          "iou_rho":rho,
                          "continue_calibration":False,
                          "save_vtk":True,
                          "restart_from":None,
                          "maximum_evaluations":100,
                          "targets":targets,
                          "x0":x0,"x1":x1,"x2":x2,
                          "parallel_simulations":True,
                          "neldermead_tolerance":1e-6,
                          "neldermead_nitmax":30,
                          "ninternaldivs":[5,10,10,50,20,45],
                          "ncheckpoints":[2,10,2,2,4,4],
                          "paths":paths
                          }
           

    execute_vcv_calibration(wave_config,
                            parenchyma_config,
                            peripheral_config,
                            calibration_config)