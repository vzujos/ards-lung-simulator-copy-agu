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
from scipy.stats import qmc
from scipy.optimize import curve_fit
from skopt.space import Real
from skopt import Optimizer
from multiprocessing import Pool, cpu_count, Manager, Lock
from vcv_lung_diaphragm import execute_vcv_simulation
import numpy as np
from scipy.interpolate import interp1d

def read_previous_results(source_directory):
    
    output_paths = os.listdir(source_directory)
    
    filter_out = []
    for typ in ["sh","txt","py"]:
        filter_by_type = lambda x: x.split(".")[-1]==typ
        filter_out += (list(filter(filter_by_type,output_paths)))
    
    for file in filter_out:
        if file in output_paths:
            output_paths.remove(file)
    
    # Empty list for data management
    costs = []; params = []
    
    # For every target simulation
    for output_path in output_paths:
        
        # Evaluate every folder
        for folder in  os.listdir(output_path):
            
            # Examine the folder
            folder_path = output_path+"/"+folder+"/"
            files = os.listdir(folder_path)
            
            # If the files are available
            if "cost.txt" in files and "params.txt" in files:
                
                # Read the simulation associated data
                cost = np.loadtxt(folder_path+"/cost.txt")
                param = np.loadtxt(folder_path+"/params.txt")
                
                # Store the information
                costs += [cost]
                params += [param]


    return np.array(params), np.array(costs)

def init_worker(wave_config, calibration_config, vcv_dict):
    
    global shared_context
    
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
                              
    time_config = {"ncycles":wave_config['ncycles'],
                   "Tsyr":wave_config['Tsyr'],
                   "Tpausa":wave_config['Tpausa'],
                   "Texp":wave_config['Texp']}

    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = parenchyma_config['constitutive_model']
    beta = 1.075

    vt = wave_config['vt']
    K_cw_stiffness = peripheral_config['K_cw_stiffness']
    K_d_stiffness = peripheral_config['K_d_stiffness']
    c_tissue = parenchyma_config['c_tissue']
    
    # Isotropic prestrain F_prestrain = alpha*I
    isotropic_prestrain = parenchyma_config['isotropic_prestrain']

    args = {"restart_from_last_checkpoint":restart_from_last_checkpoint,
            "save_checkpoints":save_checkpoints,
            "mesh_dir":path_to_mesh,
            "output_to":output_to,
            "ncheckpoints":ncheckpoints,
            "ninternaldivs":ninternaldivs,
            "K_cw_stiffness":K_cw_stiffness,
            "K_d_stiffness":K_d_stiffness,
            "permeability_dict":permeability_dict,
            "porosity_config":porosity_dict,
            "isotropic_prestrain":isotropic_prestrain,
            "solver_type":solver_type,
            "solver_dict":solver_dict,
            "tidal_volume":vt*(10**6), # conversion from L to mm3
            "time_config":time_config,
            "constitutive_model":cm,
            "constitutive_parameters":(c_tissue,beta),
            "path_to_airway":path_to_airway,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,
            "additional_resistances":additional_resistances,
            "inspiratory_pause_stop":False,
            }
        
    return args

def retrieve_existing_points(output_paths, verbose=False):
    
    # Empty list for data management
    costs = []; params = []; dirs = []
    
    # For every target simulation
    for output_path in output_paths:
        
        # Evaluate every folder
        for folder in  os.listdir(output_path):
            
            # Examine the folder
            folder_path = output_path+folder+"/"
            files = os.listdir(folder_path)
            
            # If the files are available
            if "cost.txt" in files and "params.txt" in files:
                
                # Read the simulation associated data
                cost = np.loadtxt(folder_path+"/cost.txt")
                param = np.loadtxt(folder_path+"/params.txt")
                if verbose: print("%s: # cost: %.3f"%(folder,cost))
                
                # Store the information
                costs += [cost]
                params += [param]
                dirs +=[folder_path+"Signals/"]

            else:
                if verbose: print("%s: dismissed"%folder)
                
    return np.array(params), np.array(costs)

def determine_misfits_from_signal(simulation_path, calibration_config, wave_config,
                                  nsamples=250, 
                                  include_volume_signal = False,
                                  include_pressure_signal = False,
                                  include_flux_signal = False,
                                  include_ppeak = False,
                                  include_pplat = False,
                                  include_peep = False,
                                  include_vpeak = False,
                                  normalize_misfit=True): 
    
    # Signal
    signal_path = calibration_config["paths"]["path_to_signal"]
    
    # Simulation parameters (mostly for normalization)
    Tsyr = wave_config["Tsyr"]
    Tpausa = wave_config["Tpausa"]
    Texp = wave_config["Texp"]
    ppeak = wave_config["ppeak"]
    pplat = wave_config["pplat"]
    peep = wave_config["peep"]
    
    # Load the experimental signal
    mat = np.load(signal_path)
    Paw = mat['pressure']; flow = mat['flow']; 
    vol = mat['volume']; time = mat['time']
    time -= time[0]
    
    # Determine the number of cycles
    Tcycle = Tsyr+Tpausa+Texp
   
    #  Load a sample calibration
    simulation_path += "/Signals/"
    stime = np.load(simulation_path+"effectivetimes.npy").flatten()
    sPaw = np.load(simulation_path+"presionestodas.npy").flatten()*10.1972
    sflow = np.load(simulation_path+"fluxes.npy").flatten()
    svol = np.load(simulation_path+"volumenes.npy").flatten()
    svol = svol - svol[0]
    
    maximum_simulated_time = np.max(stime)
    if maximum_simulated_time < (Tsyr+Tpausa):
        print("The simulation finished before inspiratory pause")
        print("Returning artificially high cost")
        return 1e4
        
    # Generate piecewise linear functions
    fsPaw  = interp1d(stime, sPaw,  kind='linear', fill_value="extrapolate")
    fsflow = interp1d(stime, sflow, kind='linear', fill_value="extrapolate")
    fsvol  = interp1d(stime, svol,  kind='linear', fill_value="extrapolate")
    
    fPaw  = interp1d(time, Paw,  kind='linear')
    fflow = interp1d(time, flow, kind='linear')
    fvol  = interp1d(time, vol,  kind='linear')
    
    # Sample  measurements
    simulated_pplat = fsPaw(Tsyr+Tpausa*(0.999)) # Plateau pressure
    simulated_ppeak = fsPaw(Tsyr*0.999) # Peak pressure
    simulated_peep  = sPaw[0] # PEEP value
    simulated_vpeak = fsvol(Tsyr+Tpausa*(0.95)) # Maximum volume
    
    # Signal VT
    signal_vt = fvol(Tsyr+Tpausa*(0.95))

    err = 1e-8
    ttime = np.linspace(0+err,Tcycle-err,nsamples)
    
    # Determine misfit
    pmisfit = 0.0; fmisfit = 0.0; vmisfit = 0.0; misfit = 0.0
    
    # Counters
    npm = 0; nfm = 0; nvm = 0
    
    # Determine misfit
    for t in ttime:
        T = np.floor(t/Tcycle).astype(int)

        if (t>T*Tcycle) and (t<(T*Tcycle+Tsyr+Tpausa)):
            # Inspiratory phases
            pmisfit += (fPaw(t)-fsPaw(t))**2
            npm += 1
        elif (t>(T*Tcycle+Tsyr+Tpausa)) and (t<(T+1)*Tcycle):
            # Expiratory phase
            fmisfit += (fflow(t)-fsflow(t))**2
            nfm += 1
        # Thoughout the whole cycle
        vmisfit += (fvol(t)-fsvol(t))**2
        nvm +=1
    
    # Normalize (Suggested: True)
    if normalize_misfit:
        pmisfit /= (npm*pplat**2)
        fmisfit /= (nfm*(signal_vt/Tsyr)**2)
        vmisfit /= (nvm*signal_vt**2)
    
    if include_pressure_signal: misfit += pmisfit
    if include_flux_signal: misfit += fmisfit
    if include_volume_signal: misfit += vmisfit
    if include_ppeak:
        misfit_ppeak = (simulated_ppeak-ppeak)**2/ppeak**2
        misfit += misfit_ppeak
        
    if include_pplat:
        misfit_pplat = (simulated_pplat-pplat)**2/pplat**2
        misfit += misfit_pplat
        
    if include_peep:
        misfit_peep = (simulated_peep-peep)**2/peep**2
        misfit += misfit_peep
    
    if include_vpeak:
        misfit_vpeak = ((simulated_vpeak-signal_vt)/signal_vt)**2
        misfit += misfit_vpeak*10                       
        
    outpath = os.path.join(simulation_path, "misfit_report.txt")
    with open(outpath, 'w') as f:
        f.write("\n")
        if include_pressure_signal: f.write("%18s %.1f (%4.1f%%)\n" % ("Pressure misfit:", pmisfit, pmisfit/misfit*100))
        if include_flux_signal:     f.write("%18s %.1f (%4.1f%%)\n" % ("Flow misfit:", fmisfit, fmisfit/misfit*100))
        if include_volume_signal:   f.write("%18s %.1f (%4.1f%%)\n" % ("Volume misfit:", vmisfit, vmisfit/misfit*100))
        if include_pplat:           f.write("%18s %.1f (%4.1f%%)\n" % ("Pplat misfit:", misfit_pplat, misfit_pplat/misfit*100))
        if include_ppeak:           f.write("%18s %.1f (%4.1f%%)\n" % ("Ppeak misfit:", misfit_ppeak, misfit_ppeak/misfit*100))
        if include_peep:            f.write("%18s %.1f (%4.1f%%)\n" % ("PEEP misfit:", misfit_peep, misfit_peep/misfit*100))
        if include_vpeak:           f.write("%18s %.1f (%4.1f%%)\n" % ("Vpeak misfit:", misfit_vpeak, misfit_vpeak/misfit*100))
        f.write("-"*30 + "\n")
        f.write("%18s %.1f\n" % ("Overall misfit:", misfit))
    
    return misfit
    
# --- Unified simulation function ---
def run_simulation(params, counter, lock, base_path, prefix="sim_", digits=3):
    """
    Execute a shape-calibration simulation using given (K_cw, K_d) parameters.
    This function is parallel-safe and uses a shared context for static objects.
    """
    # Unpack parameters
    c_tissue,K_cw, K_d,alpha = params

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
    _, beta = static_vcv_config['constitutive_parameters']
    wave_config = shared_context["wave_config"]
    calibration_config = shared_context["calibration_config"]

    # Copy and update simulation config
    vcv_config = static_vcv_config.copy()
    vcv_config["constitutive_parameters"] = (c_tissue,beta)
    vcv_config["isotropic_prestrain"] = alpha
    vcv_config["K_cw_stiffness"] = K_cw
    vcv_config["K_d_stiffness"] = K_d
    vcv_config["output_to"] = sim_path
    vcv_config["bc_dict"] = {"activate": True,
                                "flow_func": flowfn,
                                "pres_func": presfn,
                        }

    print(f"(run_simulation {code}) Running simulation for: c_tissue={c_tissue:.3f}, Kd={K_d:.5f} Kcw={K_cw:.5f}, alpha={alpha:.5f}")

    execute_vcv_simulation(vcv_config)

    try:
        cost = determine_misfits_from_signal(sim_path,
                                             calibration_config,
                                             wave_config,
                                             include_peep=True,
                                             include_pplat=True,
                                             include_vpeak=True)
        if K_cw < K_d:
            cost += 10.0 # Artificially increase cost if diaphragm is stiffer than chest-wall
    except:
        cost = 1e6  # Heavy penalty

    # Save results
    np.savetxt(os.path.join(sim_path, "params.txt"), np.array([c_tissue,K_cw,K_d,alpha]))
    with open(os.path.join(sim_path, "cost.txt"), "w") as f:
        f.write(str(cost))

    print(f"(run_simulation {code}) Cost: {cost:.5f}")
    
    return cost
        
def evaluate_batch(X_batch, counter, lock, base_path, parallel, ncpus, 
                   wave_config, calibration_config, vcv_dict):
    
    pool_args = [(params, counter, lock, base_path) for params in X_batch]

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

def bayesian_optimization(parameters):
    
    # Unpack big configuration dictionaries
    wave_config = parameters['wave_config']
    calibration_config = parameters['calibration_config']
    peripheral_config = parameters['peripheral_config']
    parenchyma_config = parameters['parenchyma_config']
    
    # Assess folders and similar stuff
    success = manage_paths(parameters)
    
    if success:
        print(parameters['paths']['output_path'])
    
    # Set seed for reproducibility
    np.random.seed(parameters['configuration']['seed'])
        
    # Define parameter bounds
    c_tissue_bounds = parameters['bounds']['c_tissue']
    K_cw_bounds = parameters['bounds']['Kcw']
    K_d_bounds = parameters['bounds']['Kd']  # Initially wide, will filter manually
    alpha_bounds = parameters['bounds']['alpha']
    
    # Parallel executions
    parallel = parameters['configuration']['parallel']
    ncpus = parameters['configuration']['ncpus']
    
    # Verbosity
    verbose = parameters['configuration']['verbose']
    
    # Skip LHS sampling and read from another place
    bypass_lhs = parameters['configuration']['bypass_lhs']['activate'] 
    sourcing_path = parameters['configuration']['bypass_lhs']['path']
    
    # Unpack parameters
    batchsize_explore = parameters['explore']['batchsize']
    exploring_rounds = parameters['explore']['nrounds']
    batchsize_exploit = parameters['exploit']['batchsize']
    exploiting_rounds = parameters['exploit']['nrounds']
    
    # Generate dictionary
    vcv_dict = generate_vcv_dictionary(wave_config,parenchyma_config,peripheral_config,calibration_config)

    # Use scipy.stats.qmc for Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=4)
    X_raw = sampler.random(n=parameters['lhs']['nsamples'])
        
    # Scale the samples to the parameter bounds
    bounds = np.array([c_tissue_bounds, K_cw_bounds, K_d_bounds, alpha_bounds])
    X_scaled = qmc.scale(X_raw, bounds[:, 0], bounds[:, 1])
        
    # Enforce constraint K_cw > K_d
    X = X_scaled
    
    Xs_manager = []; Ys_manager = []
    
    manager = Manager()
    lock = manager.Lock()
    counter = manager.Value('i',0) # Start from 0
    
    pool_args = [(params, counter, lock, parameters['paths']['output_path']) for params in X]
    
    # We do this so we can avoid the LHS stage and go straight into the 
    # optimizaiton 
    if bypass_lhs: 
        
        X_scaled, Y = read_previous_results(sourcing_path)
        X = X_scaled

    else:
        # Evaluate the cost
        if parallel:
            print(" > Parallel execution")
            with Pool(processes=ncpus, 
                      initializer=init_worker, 
                      initargs=(wave_config, calibration_config, vcv_dict)) as pool:
                Y = np.array(pool.starmap(run_simulation, pool_args))
        else:
            print(" > Serial execution:")
            Y = np.array([run_simulation(*args) for args in pool_args])
        
        print(" > Done")

    Xs_manager = X_scaled.copy()
    Ys_manager = Y.copy()
        
    # Define your parameter space
    space = [Real(c_tissue_bounds[0], c_tissue_bounds[1], name='c_tissue'),
             Real(K_cw_bounds[0], K_cw_bounds[1], name='K_cw'),
             Real(K_d_bounds[0], K_d_bounds[1], name='K_d'),
             Real(alpha_bounds[0], alpha_bounds[1], name='alpha'),]  # ensure K_d < K_cw later
        
    # Phase 1: EXPLORATION
    opt_explore = Optimizer(dimensions=space,
                            base_estimator="GP", # "GP" or "gp_hedge" or "RF"
                            acq_func="LCB",
                            acq_func_kwargs={"kappa": 3.0},  # More exploration
                            )
    
    # Tell it your initial data (e.g., from LHS or random)
    opt_explore.tell(X.tolist(), Y.tolist())
    
    # EXPLORATION STAGE
    print("EXPLORATION STAGE")
    for r in range(exploring_rounds):
        print(" Round #%i" % (r+1))
        
        Xs_explore = opt_explore.ask(n_points=batchsize_explore)
        Ys_explore = evaluate_batch(Xs_explore, counter, lock, parameters['paths']['output_path'], 
                                    parallel, ncpus, wave_config, calibration_config, vcv_dict)
        if verbose:
            for e, x in enumerate(Xs_explore):
                print("   X[%i]: (%.5f, %.5f)" % (e, x[0], x[1]))
        
        opt_explore.tell(Xs_explore, Ys_explore.tolist())
    
        Xs_manager = np.vstack([Xs_manager, np.array(Xs_explore)])
        Ys_manager = np.hstack([Ys_manager, np.array(Ys_explore)])
            
    # Phase 2: EXPLOITATION
    # Create a new optimizer, reusing the data
    opt_exploit = Optimizer(dimensions=space,
                            base_estimator="GP",
                            acq_func="PI",  # or "EI"
                            acq_func_kwargs={"xi": 0.005},  # Very greedy
                        )
        
    # Transfer data from exploratory optimizer
    opt_exploit.tell(opt_explore.Xi, opt_explore.yi)

    # EXPLOITATION STAGE
    print("EXPLOITATION STAGE")
    for r in range(exploiting_rounds):
        print(" Round #%i" % (r+1))
        
        Xs_exploit = opt_exploit.ask(n_points=batchsize_exploit)
        Ys_exploit = evaluate_batch(Xs_exploit, counter, lock, parameters['paths']['output_path'], 
                                    parallel, ncpus,wave_config, calibration_config, vcv_dict)
    
        if verbose:
            for e, x in enumerate(Xs_exploit):
                print("   X[%i]: (%.3f, %.5f, %.5f, %.3f)" % (e,x[0],x[1],x[2],x[3]))
    
        opt_exploit.tell(Xs_exploit, Ys_exploit.tolist())
    
        Xs_manager = np.vstack([Xs_manager, np.array(Xs_exploit)])
        Ys_manager = np.hstack([Ys_manager, np.array(Ys_exploit)])

    print("End of optimization")
        
    best_idx = np.argmin(Ys_manager)
    best_x = Xs_manager[best_idx]
    best_y = Ys_manager[best_idx]
        
    return (best_idx, best_x, best_y), (Xs_manager, Ys_manager)

def manage_paths(parameters):
    
    paths = parameters['paths']
    restart_task_id = parameters['configuration']['restart_task_id']
    
    # Unpack paths
    output_root = paths['output_root']
    task_name = paths['task_name']
    
    # Select the folder to send output to
    if restart_task_id is None:
        
        folders = list(map(lambda x:x+"/",os.listdir(output_root)))
        for task_id in range(999):
            test_task = task_name%task_id
            print(test_task)
            if test_task in folders:
                continue
            else:
                test_task = task_name%(task_id)
                break
        output_path = output_root+test_task
    else:
        output_path = output_root+task_name%restart_task_id
        if not os.path.isfile(output_path):
            print("Restart task path not found")
            print(" > Task folder: %s"%task_name%restart_task_id)
            return False
    
    print("Sending output to folder: %s"%output_path)

    paths.update({'output_path':output_path})
    
    return True

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
