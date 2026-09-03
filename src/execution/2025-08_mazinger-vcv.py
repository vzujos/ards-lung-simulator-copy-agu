# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:47:49 2024

@author: angus
"""
import sys
sys.path.append("/home1/agustin.perez/model2/src/")
import vcv_lung_var_stiffness
import supportfunctions as sf
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit

# %%

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

def fit_segment(time, signal, kind):
    
    def linear(x, a, b): return a * x + b
    def exp(x, a, b): return -a * (1 - np.exp(-b * x))

    if kind == "linear":
        return curve_fit(linear, time, signal)[0], linear
    elif kind == "exp":
        return curve_fit(exp, time, signal)[0], exp
    elif kind == "zero":
        return (), lambda x: 0.0
    else:
        raise ValueError(f"Unknown fit type: {kind}")


def create_flow_function(time_arr, flow, segments, t_cycle=2.0):
    segment_fits = []

    for seg in segments:
        mask = (time_arr >= seg["t_start"]) & (time_arr < seg["t_end"])
        t_seg = time_arr[mask]
        f_seg = flow[mask]

        params, fn = fit_segment(t_seg, f_seg, seg["type"])
        segment_fits.append({
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "fn": fn,
            "params": params
        })

    def flow_func(t):
        t_mod = t % t_cycle
        for seg in segment_fits:
            if seg["t_start"] <= t_mod < seg["t_end"]:
                return seg["fn"](t_mod, *seg["params"])
        return 0.0

    return flow_func


def create_pressure_function(time_arr, pressure, segments, t_cycle=2.0):
    segment_fits = []

    for seg in segments:
        mask = (time_arr >= seg["t_start"]) & (time_arr < seg["t_end"])
        t_seg = time_arr[mask]
        p_seg = pressure[mask]

        if len(p_seg) < 2:
            raise ValueError(f"Segment {seg} has too few data points for fitting.")

        params, fn = fit_segment(t_seg, p_seg, seg["type"])
        segment_fits.append({
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "fn": fn,
            "params": params
        })

    plateau_measured = pressure[time_arr >= segments[0]["t_start"]][0]

    def pressure_func(t, plateau_pressure_sim):
        t_mod = t % t_cycle
        scale = plateau_pressure_sim / plateau_measured
    
        for i, seg in enumerate(segment_fits):
            is_last = (i == len(segment_fits) - 1)
    
            if seg["t_start"] <= t_mod < seg["t_end"]:
                return seg["fn"](t_mod, *seg["params"]) * scale
    
            # Special case: t == Tcycle → t_mod == 0.0, use extrapolation from last segment
            if is_last and np.isclose(t_mod, 0.0, atol=1e-8):
                return seg["fn"](seg["t_end"], *seg["params"]) * scale
    
        raise ValueError(f"Time {t_mod:.4f} is outside all defined segments.")

    return pressure_func

if __name__ == "__main__":


    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
    
    
    # PIG Number and codename
    pig_number = 5
    protocol = "ARDSnet"
    codename = "PIG%i-%s"%(pig_number,protocol)
    
    # Declaration of variables
    root = "/home1/agustin.perez/"
    case = "FEniCS"
    path_to_mesh = root+"model2/geometries/PIG%i/ARDSnet/medium-coarse/"%pig_number
    path_to_airway = path_to_mesh+"skel.vtu"    

    
    # Checkpoint parameters
    restart_from_last_checkpoint = False
    save_checkpoints = False
    save_vtk = True
    
    # Processing the paths
    output_to = "."
      
    # Quick parameter setting
    K_cw_stiffness = 0.0050
    K_d_stiffness = 0.0010
    
    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = "bir2019"
    c = 2.0864 # [factor is 3.25 for VT30]
    stiffness_file = "c_tissue.xml.gz"
    beta = 1.075
    isotropic_prestrain = 1.0326
    
    # Temporal management
    ncheckpoints =[2, 10, 2,  2, 4, 4]    # These will be used to save some checkpoints
    ninternaldivs=[5, 20, 5, 20,20,50] # Funciona para pack-main

    # Permeability config
    KK_exp = 3
    KK_factor = 1.0
    permeability_dict = {"variable_permeability":True,
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
    Tsyr = parameters[codename]["Tsyr"]
    Tpausa = parameters[codename]["Tpausa"]
    Texp = parameters[codename]["Texp"]
    Tcycle = Tsyr+Tpausa+Texp
    
    time_config = {"ncycles":ncycles,
                   "Tsyr":Tsyr,
                   "Tpausa":Tpausa,
                   "Texp":Texp}

    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":True,
                                               "expiratory_gamma":1.0694, # 0.327 original value
                                               "inspiratory_gamma":0.77378, # 0.327 original value
                                               "tolerance":1e-8,
                                               "nitmax":100}}
                              

    # Path to experimental data, load matlab file and prepare data
    signal_path = parameters[codename]['Signal path']

    # Load the experimental signal
    npz = np.load(signal_path)
    flow = npz['flow'].flatten()*1e6
    pres = npz['pressure'].flatten()
    time = npz['time'].flatten()

    # Define flow and pressure functions
    flow_segments = function_manager['flow_segments']["PIG%i"%pig_number]
    pressure_segments = function_manager['pressure_segments']["PIG%i"%pig_number]
    
    flow_fn = create_flow_function(time, flow, flow_segments)
    pressure_fn = create_pressure_function(time, pres, pressure_segments, t_cycle=Tcycle)


    # Generate flow function dictionary
    signal_boundary_conditions = {'activate':True,
                                  'flow_func':flow_fn,
                                  'pres_func':pressure_fn}
    
    
    
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
