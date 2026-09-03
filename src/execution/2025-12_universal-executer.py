# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:47:49 2024

@author: angus
"""
import sys
sys.path.append("..")
import vcv_lung
import supportfunctions as sf
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
import argparse
# %%

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run lung simulation with configurable parameters"
    )

    parser.add_argument(
        "-pig_id",
        type=int,
        choices=[2, 3, 4, 5, 6],
        default=5,
        help="Pig ID (default: 5)"
    )

    parser.add_argument(
        "-mesh_type",
        type=str,
        choices=["coarse", "medium-coarse", "medium", "medium-fine", "fine"],
        default="medium-coarse",
        help="Mesh resolution (default: medium-coarse)"
    )

    parser.add_argument(
        "-cm",
        type=str,
        choices=["bir2019", "ma", "perez"],
        default="perez",
        help="Constitutive model (default: perez)"
    )

    return parser.parse_args()


# %%
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

word_reducer = {"coarse":"c",
                "medium-coarse":"mc",
                "medium":"m",
                "medium-fine":"mf",
                "bir2019":"bir",
                "ma":"ma",
                "perez":"per"}

physical_params = {2:{"bir2019":{"ctissue":1.2468, # Misfit = 0.016
                                 "kcw":0.1000,
                                 "kd":0.0889,
                                 "palv":10.6524,
                                 "gamma_insp":0.1612,
                                 "gamma_exp":0.4531},
                      "ma":{"ctissue":9.8050, # Misfit = 0.101
                            "kcw":0.0126,
                            "kd":0.0010,
                            "palv":8.6619,
                            "gamma_insp":0.2923,
                            "gamma_exp":0.3466},
                      "perez_pre12052026":{"ctissue": 1.2214, # 0.013 medium-fine
                               "kcw": 0.1028,
                               "kd": 0.0250,
                               "palv": 11.0451, # 11
                               "gamma_insp": 0.2484,
                               "gamma_exp": 0.4255,},
                      "perez":{"ctissue": 1.4580, # 0.027 medium-fine
                               "kcw": 0.0668,
                               "kd": 0.0790,
                               "palv": 10.8889, # 11
                               "gamma_insp": 0.2708,
                               "gamma_exp": 0.4527,}
                      },
                   3:{"bir2019":{"ctissue":1.7890, # Misfit = 0.030
                                 "kcw":0.0937,
                                 "kd":0.0212,
                                 "palv":11.1921,
                                 "gamma_insp":0.1669,
                                 "gamma_exp":0.2946},
                      "ma":{"ctissue":8.3608, # Misfit = 0.024
                            "kcw":0.0127,
                            "kd":0.0456,
                            "palv":10.9862,
                            "gamma_insp":0.2346,
                            "gamma_exp":0.2596},
                      "perez_pre12052026":{ "ctissue": 1.7615, # 0.028 medium-fine
                               "kcw": 0.0471,
                               "kd": 0.0900,
                               "palv": 11.1092,
                               "gamma_insp": 0.2003,
                               "gamma_exp": 0.3201,},
                      "perez":{ "ctissue": 1.8726, # 0.028 medium-fine
                               "kcw": 0.0433,
                               "kd": 0.0951,
                               "palv": 11.8141,
                               "gamma_insp": 0.2166,
                               "gamma_exp": 0.2948,},
                    },
                   4:{"bir2019":{"ctissue":1.4369, # Misfit = 0.010
                                 "kcw":0.0173,
                                 "kd":0.0041,
                                 "palv":10.4371,
                                 "gamma_insp":0.4417,
                                 "gamma_exp":0.6618},
                      "ma":{"ctissue":6.1465, # Misfit = 0.048
                            "kcw":0.0217,
                            "kd":0.0203,
                            "palv":10.7719,
                            "gamma_insp":0.4377,
                            "gamma_exp":0.4483},
                      "perez":{"ctissue": 1.2817, # 0.009 medium
                              "kcw": 0.0540, 
                              "kd": 0.0099,
                              "palv": 10.8658,
                              "gamma_insp": 0.4797,
                              "gamma_exp": 0.6484,}
                    },
                   5:{"bir2019":{"ctissue":1.6925, # Misfit = 0.013
                                 "kcw":0.0005,
                                 "kd":0.0058,
                                 "palv":10.5907,
                                 "gamma_insp":0.3147,
                                 "gamma_exp":0.4648},
                      "perez":{"ctissue": 1.5975, # 0.017 medium
                              "kcw": 0.0166,
                              "kd": 0.0675,
                              "palv": 11.000,
                              "gamma_insp": 0.3183,
                              "gamma_exp": 0.4433,},
                      "ma":{"ctissue":8.2465, # Misfit = 0.032
                            "kcw":0.0115,
                            "kd":0.0168,
                            "palv":9.8360,
                            "gamma_insp":0.3023,
                            "gamma_exp":0.4452},
                    },
                   6:{"bir2019":{"ctissue":1.9700, # Misfit = 0.026
                                 "kcw":0.0485,
                                 "kd":0.0083,
                                 "palv":10.5295,
                                 "gamma_insp":0.1480,
                                 "gamma_exp":0.3673},
                      "ma":{"ctissue":10.7303, # Misfit = 0.021
                            "kcw":0.0010,
                            "kd":0.1000,
                            "palv":10.7029,
                            "gamma_insp":0.1634,
                            "gamma_exp":0.3209},
                      "perez":{"ctissue": 1.6788, # 0.025 medium-fine
                              "kcw": 0.1086,
                              "kd": 0.1037,
                              "palv": 10.3918,
                              "gamma_insp": 0.1536,
                              "gamma_exp": 0.3665,}
                    },                   }
                   

# %%

if __name__ == "__main__":


    args = parse_arguments()

    pig = args.pig_id
    mesh_type = args.mesh_type
    cm = args.cm

    print(f"Running with pig_id={pig}, mesh_type={mesh_type}, cm={cm}")

    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
    
    # PIG ID and mesh quality

    # Simulation Codename
    codename = "PIG%i-%s-%s"%(pig,word_reducer[mesh_type],word_reducer[cm]) # This is a name for the folder where the output is directed
    mesh_name = "" #  This should change for different states/subjects; While in development, just keep 'stable'
    case ="FEniCS" # This is the specific name for the mesh in use
    
    # Declare the path to the folder
    #path_to_mesh = "/mnt/c/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/%s/%s"%(packname,mesh_packs[packname])
    path_to_mesh = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/%s/"%(pig,mesh_type)
    path_to_airway = path_to_mesh+"skel.vtu"
  
    # Direct the output of this execution towards this folder
    output_to = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    
    # Checkpoint parameters
    restart_from_last_checkpoint = False
    save_checkpoints = False
    save_vtk = True
    
    # Processing the paths
    path_to_mesh =  sf.manage_mesh_directory(path_to_mesh,mesh_name,case)
    output_to = sf.manage_output_directory(output_to,codename,restart_from_last_checkpoint)
      
    # Quick parameter setting
    c_tissue = physical_params[pig][cm]['ctissue'] 
    K_cw_stiffness = physical_params[pig][cm]['kcw'] 
    K_d_stiffness = physical_params[pig][cm]['kd'] 
    alveolar_pressure = physical_params[pig][cm]['palv'] 
    inspiratory_gamma = physical_params[pig][cm]['gamma_insp'] 
    expiratory_gamma = physical_params[pig][cm]['gamma_exp'] 
    
    constitutive_model_config = {"model_name":cm,
                                 "c_tissue":c_tissue,
                                 "c3":5.667e0,
                                 "k":80,
                                 "phi_transition":0.2,
                                 "stiffness_function":None,
                                 "stiffness_file":"c_tissue.xml.gz",
                                 "variable_stiffness":True,}
    
    # Temporal management
    ncheckpoints =[ 2, 10,  2, 10,  4,  4]    # These will be used to save some checkpoints
    ninternaldivs=[5, 50, 10, 60, 35, 60] # Funciona para pack-main

    # Permeability config
    KK_exp = 5
    KK_factor = 1.0
    permeability_dict = {"variable_permeability":False,
                         "permeability_file":"k0.xml",
                         "KK_exp":KK_exp,
                         "KK_factor":KK_factor,
                         }
    
    # Porosity config
    porosity_config = {"activate":True, 
                       "mean":0.50, # Dummy
                       "file":"phi0.xml.gz"}

    # Gravity config
    gravity_config = {"activate":False,
                      "rho_t":1.0e-6, # kg/mm3 # Water density
                      "rho_a":1.225e-9, # kg/mm3 # Air density
                      "K_inf":1e6, # Very high value
                      "g":9.81e3, # In mm/s2
                      }

    # Tolerances in the iterative cycles
    max_nit = 30
    tol_p_it = 1e-3
    tol_v_it = 1e-2
    
    # Selecting solver
    solver_type = "dict"
    solver_dict = {"nonlinear_solver":"snes",
                   "snes_solver":{"linear_solver":"lu",
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
    Tsyr = parameters["PIG%i-ARDSnet"%pig]["Tsyr"]
    Tpausa = parameters["PIG%i-ARDSnet"%pig]["Tpausa"]
    Texp = parameters["PIG%i-ARDSnet"%pig]["Texp"]
    Tinsp = Tsyr+Tpausa
    Tcycle = Texp + Tinsp
    
    time_config = {"ncycles":ncycles,
                   "Tsyr":Tsyr,
                   "Tpausa":Tpausa,
                   "Texp":Texp}

    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":True,
                                               "expiratory_gamma":expiratory_gamma, # 0.327 original value
                                               "inspiratory_gamma":inspiratory_gamma, # 0.327 original value
                                               "tolerance":1e-8,
                                               "nitmax":100}}

    # Path to experimental data, load matlab file and prepare data
    signal_path = '/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz'%(pig,pig)

    # Load the experimental signal
    npz = np.load(signal_path)
    aw_pressure = npz['pressure']/10.1972 # kPa
    flow = npz['flow']*(1e6)
    time = npz['time']
    
    # Retrieve segments for function construction
    flow_segments = function_manager["PIG%i"%pig]["flow_segments"]
    pres_segments = function_manager["PIG%i"%pig]["pressure_segments"]
    
    flowfn = create_flow_function(time,flow,flow_segments,t_cycle=Tcycle) 
    presfn = create_pressure_function(time,aw_pressure,pres_segments,t_cycle=Tcycle)

    # Generate flow function dictionary
    signal_boundary_conditions = {'activate':True,
                                  'flow_func':flowfn,
                                  'pres_func':presfn}
    
    # Tidal volume
    vt = 0.40 # L, program takes input in mm3 so it's multiplied by 10**6 

    solver_parameters = {"nonlinear_solver":"snes",
                         "snes_solver":{"linear_solver":"lu",
                                        "relative_tolerance":1e-6,
                                        "absolute_tolerance":1e-8,
                                        "maximum_iterations":60,
                                        "line_search":"bt",
                                        "report":True,
                                        "error_on_nonconvergence":True,
                                        "preconditioner":"default"}
                         }

    
    ida_config = {"folder":"/InverseAnalysis/",
                  "activate_gravity":True,
                  "activate_pressure_gradient":True,
                  "delta_pressure":10/10.1972,
                  "pressure_reference":0.0,
                  "solver_parameters":solver_parameters,}
    
    ida_folder = ida_config['folder']
    ida_activate_pressure_gradient = ida_config['activate_pressure_gradient']
    ida_pressure_reference = ida_config['pressure_reference']
    ida_delta_pressure = ida_config['delta_pressure']
    ida_activate_gravity = ida_config['activate_gravity']
    ida_solver_parameters = ida_config['solver_parameters']

    args = {"restart_from_last_checkpoint":restart_from_last_checkpoint,
            "save_checkpoints":save_checkpoints,
            "mesh_dir":path_to_mesh,
            "output_to":output_to,
            "ncheckpoints":ncheckpoints,
            "ninternaldivs":ninternaldivs,
            "K_cw_stiffness":K_cw_stiffness,
            "K_d_stiffness":K_d_stiffness,
            "K_m_stiffness":1e6,
            "permeability_dict":permeability_dict,
            "alveolar_pressure":alveolar_pressure,
            "ida_config":ida_config,
            "solver_type":solver_type,
            "solver_dict":solver_dict,
            "tidal_volume":vt*(10**6), # conversion from L to mm3
            "time_config":time_config,
            "porosity_config":porosity_config,
            "cm_config":constitutive_model_config,
            "path_to_airway":path_to_airway,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,
            "additional_resistances":additional_resistances,
            "bc_dict":signal_boundary_conditions,
            "inspiratory_pause_stop":False,
            "gravity_config":gravity_config,
            }
    
    vcv_lung.execute_vcv_simulation(args)
