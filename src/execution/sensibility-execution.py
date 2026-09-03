# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:11:46 2025

@author: angus
"""

shared_context = {}

import sys
import os
sys.path.append("./calibrate/")
sys.path.append("/home1/agustin.perez/model2/src/")
from multiprocessing import Pool, cpu_count, Manager, Lock
from vcv_lung import execute_vcv_simulation
import numpy as np
from scipy.interpolate import interp1d

def init_worker(wave_config, calibration_config, vcv_dict):
    
    global shared_context
    os.environ["DIJITSO_CACHE_DIR"]= f"/tmp/dijistso_cache_{os.getpid()}"
    time = wave_config['signal_config']['time']
    pressure = wave_config['signal_config']['pressure']
    flow = wave_config['signal_config']['flow']*1e6
    Tsyr = wave_config['Tsyr']
    Tpausa = wave_config['Tpausa']
    Texp = wave_config['Texp']
    Tcycle = Tsyr+Tpausa+Texp
    
    flow_segments = wave_config['signal_config']['setup']['flow_segments']
    pressure_segments = wave_config['signal_config']['setup']['pressure_segments']
    
    flow_fn = create_flow_function(time, flow, flow_segments)
    pressure_fn = create_pressure_function(time, pressure, pressure_segments, t_cycle=Tcycle)
    
    shared_context["flowfn"] = flow_fn
    shared_context["presfn"] = pressure_fn
    shared_context["vcv_config"] = vcv_dict
    shared_context["wave_config"] = wave_config
    shared_context["calibration_config"] = calibration_config
    
def generate_vcv_dictionary(wave_config,
                            parenchyma_config,
                            peripheral_config,
                            calibration_config,
                            **kwargs):
    
    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
        
   # Simulation Codename
   # codename = "CORNELL-PIG6-ARDSnet" # This is a name for the folder where the output is directed
   # mesh_name = "" #  This should change for different states/subjects; While in development, just keep 'stable'
   # case ="FEniCS" # This is the specific name for the mesh in use

    # Checkpoint parameters
    restart_from_last_checkpoint = False
    save_checkpoints = False
    save_vtk = calibration_config['save_vtk']
    
    # Path-related
    paths = calibration_config['paths']
    path_to_mesh = paths['path_to_mesh'] 
    output_to = paths['output_path']
    path_to_airway = paths['path_to_airways']

    # Processing the paths
    #path_to_mesh =  sf.manage_mesh_directory(path_to_mesh,mesh_name,case)
    #output_to = sf.manage_output_directory(output_to,codename,restart_from_last_checkpoint)
    
    permeability_dict = {"variable_permeability":parenchyma_config['variable_permeability'],
                         "permeability_file":parenchyma_config['permeability_file'],
                         "KK_exp":parenchyma_config['permeability_exp'],
                         "KK_factor":parenchyma_config['permeability_factor'],
                         }

    porosity_dict = {"activate":parenchyma_config['variable_porosity'],
                     "mean":parenchyma_config['porosity_mean'],
                     "file":parenchyma_config['porosity_file']}

    # Temporal config
    ncheckpoints = calibration_config['ncheckpoints']
    ninternaldivs = calibration_config['ninternaldivs']

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
    
    # Modifying the base resistance for the airway tree
    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":peripheral_config['pedley_activate'],
                                               "expiratory_gamma":peripheral_config['pedley_expiratory_gamma'],
                                               "inspiratory_gamma":peripheral_config['pedley_inspiratory_gamma'], # 0.327 original value
                                               "tolerance":peripheral_config['pedley_tolerance'],
                                               "nitmax":peripheral_config['pedley_nitmax']}
                              }
    
    # Configurate the different stages of the wave
    time_config = {"ncycles":wave_config['ncycles'],
                   "Tsyr":wave_config['Tsyr'],
                   "Tpausa":wave_config['Tpausa'],
                   "Texp":wave_config['Texp']}
    
    # Configure the gravity
    gravity_config = {'activate':True,
                    'rho_t':1e-6,
                    'rho_a':1.225e-9,
                    'K_inf':1e6,
                    'g':9.81e3}


    # Inverse deformation analysis
    ida_config = {'folder':'/InverseAnalysis/',
            'activate_gravity':True,
            'activate_pressure_gradient':True,
            'delta_pressure':parenchyma_config['pleural_pressure_drop']/10.1972, # Transforming 10 cmH2O to kPa
            'pressure_reference':0.0,
            'solver_parameters':solver_dict
            }

    

    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = parenchyma_config['constitutive_model']
    beta = 1.075

    constitutive_model_config = {'model_name':parenchyma_config['constitutive_model'],
                                'c_tissue':None,
                                'c3':5.667e0,
                                'k':80,
                                'phi_transition':0.2,
                                'stiffness_function':None,
                                'stiffness_file':'c_tissue.xml.gz',
                                'variable_stiffness':True}


    vt = wave_config['vt']
    K_cw_stiffness = peripheral_config['K_cw_stiffness']
    K_d_stiffness = peripheral_config['K_d_stiffness']
    c_tissue = parenchyma_config['c_tissue']
    p_alv = parenchyma_config['alveolar_pressure']

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
            "porosity_config":porosity_dict,
            "alveolar_pressure":p_alv,
            "solver_type":solver_type,
            "solver_dict":solver_dict,
            "tidal_volume":vt*(10**6), # conversion from L to mm3
            "time_config":time_config,
            "stiffness_file":None,
            "path_to_airway":path_to_airway,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,
            "additional_resistances":additional_resistances,
            "inspiratory_pause_stop":False,
            'gravity_config':gravity_config,
            'ida_config':ida_config,
            'cm_config':constitutive_model_config,
            }
        
    return args


# --- Unified simulation function ---
def run_simulation(params, cost_function_options, parameters_info, 
                   counter, lock, base_path, prefix="sim_", digits=3):
    """
    Execute a shape-calibration simulation using given (K_cw, K_d) parameters.
    This function is parallel-safe and uses a shared context for static objects.
    """

    # Generate a unique simulation path
    with lock:
        sim_id = counter.value
        counter.value += 1
    code = f"{sim_id:0{digits}d}"
    sim_path = os.path.join(base_path, f"{prefix}{code}")+"/"
    os.makedirs(sim_path, exist_ok=False)

    # Access shared static variables
    flowfn = shared_context["flowfn"]
    presfn = shared_context["presfn"]
    static_vcv_config = shared_context["vcv_config"]

    # Copy and update simulation config
    vcv_config = static_vcv_config.copy()
    
    if "c_tissue" in parameters_info:
        c_tissue = params[parameters_info["c_tissue"]]
        vcv_config['cm_config']['c_tissue'] = c_tissue
        
    if "alveolar_pressure" in parameters_info:
        p_alv = params[parameters_info["alveolar_pressure"]]
        vcv_config["alveolar_pressure"] = p_alv
        
    if "K_cw" in parameters_info:
        K_cw = params[parameters_info["K_cw"]]   
        vcv_config["K_cw_stiffness"] = K_cw
        
    if "K_d" in parameters_info:
        K_d = params[parameters_info["K_d"]]    
        vcv_config["K_d_stiffness"] = K_d
    
    if "gamma_insp" in parameters_info:
        gamma_insp = params[parameters_info["gamma_insp"]]
        vcv_config["additional_resistances"]["pedley_config"]["inspiratory_gamma"] = gamma_insp
        
    if "gamma_exp" in parameters_info:
        gamma_exp = params[parameters_info["gamma_exp"]]
        vcv_config["additional_resistances"]["pedley_config"]["expiratory_gamma"] = gamma_exp

        
    vcv_config["output_to"] = sim_path
    vcv_config["bc_dict"] = {"activate": True,
                                "flow_func": flowfn,
                                "pres_func": presfn,
                        }
    try:
        execute_vcv_simulation(vcv_config) 
        success = True
    except:
        success = False

    if success:
        print(f"(run_simulation {code}) run successfully")
    else:
        print(f"(run_simulation {code}) failed")

    return success
        
def evaluate_batch(X_batch, counter, lock, base_path, parallel, ncpus, 
                   wave_config, calibration_config, vcv_dict):
    
    cost_function_options = calibration_config['cost_function_options']
    parameters_info = calibration_config['parameters_info']
    
    pool_args = [(params, cost_function_options, parameters_info, counter, lock, base_path) for params in X_batch]

    if parallel:
        with Pool(processes=ncpus,
                  initializer=init_worker,
                  initargs=(wave_config, calibration_config, vcv_dict)) as pool:
            Y_batch = pool.starmap(run_simulation, pool_args)
    else:
        # If not parallel, still set global context manually
        init_worker(wave_config, calibration_config, vcv_dict)
        Y_batch = [run_simulation(*args) for args in pool_args]

    return np.array(Y_batch)

def sensibility_execution(parameters):
    
    # Unpack big configuration dictionaries
    wave_config = parameters['wave_config']
    calibration_config = parameters['calibration_config']
    peripheral_config = parameters['peripheral_config']
    parenchyma_config = parameters['parenchyma_config']
    
    # Cost function management
    cost_function_options = calibration_config['cost_function_options']
    parameters_info = calibration_config['parameters_info']
        
    # Parallel executions
    parallel = parameters['configuration']['parallel']
    ncpus = parameters['configuration']['ncpus']
        
    # Generate dictionary
    vcv_dict = generate_vcv_dictionary(wave_config,parenchyma_config,peripheral_config,calibration_config)
    
    Xs_manager = []; Ys_manager = []
    
    manager = Manager()
    lock = manager.Lock()
    counter = manager.Value('i',0) # Start from 0
    
    pool_args = [(params, cost_function_options, parameters_info, counter, lock, parameters['paths']['output_path']) for params in X]
    
    # We do this so we can avoid the LHS stage and go straight into the 
    # optimizaiton 

        # Evaluate the cost
    if parallel:
        print(" > Parallel execution")
        with Pool(processes=ncpus, 
                  initializer=init_worker, 
                  initargs=(wave_config, calibration_config, vcv_dict)) as pool:
             
            Y = np.array(pool.starmap(run_simulation, pool_args))

    Ys_manager = Y.copy()

    
    return Xs_manager, Ys_manager

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

