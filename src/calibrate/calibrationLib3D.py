# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:57:14 2024

@author: angus
"""

import os
import sys
sys.path.append("/home1/agustin.perez/model2/src/")
import numpy as np
from vcv_lung import execute_vcv_simulation
from sklearn import linear_model
from multiprocessing import Pool
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

def generate_vcv_dictionary(paths,
                            wave_config,
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
    save_vtk = False
    
    # Path-related
    path_to_mesh = paths['path_to_mesh'] 
    output_to = paths['output_to']
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
    K_stiffness = peripheral_config['K_stiffness']
    c_tissue = parenchyma_config['c_tissue']
    
    # Isotropic prestrain F_prestrain = alpha*I
    isotropic_prestrain = parenchyma_config['isotropic_prestrain']
    
    # Generate flow and pressure functions dictionary
    # These functions will be manually deactivated in this implementation
    if wave_config['bc_dict'] is None:
        bc_dict = {'activate':False,
                   'flow_func':None,
                   'pressure_func':None}
    else:
        bc_dict = wave_config['bc_dict']

    args = {"restart_from_last_checkpoint":restart_from_last_checkpoint,
            "save_checkpoints":save_checkpoints,
            "mesh_dir":path_to_mesh,
            "output_to":output_to,
            "ncheckpoints":ncheckpoints,
            "ninternaldivs":ninternaldivs,
            "K_stiffness":K_stiffness,
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
            "bc_dict":bc_dict,
            "additional_resistances":additional_resistances,
            }
        
    return args


def execute_single_simulation(path,args,params,calibration_type):
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    # Manage parameters
    
    # Update dictionary
    args_a = args.copy()  
   
    # Write parameters
    np.savetxt(path+"iteration_params.txt", params)

    if calibration_type == "partial_signal":
        # Ctissue, gamma and alpha under calibration
        c_tissue, gamma, alpha = params # Unpack
        _, beta = args["constitutive_parameters"] # Reutilize beta value

        args_a["constitutive_parameters"] = (c_tissue, beta)
        args_a["additional_resistances"]["pedley_config"]["gamma"] = gamma
        args_a["output_to"] = path
        args_a["isotropic_prestrain"] = alpha
        print("Run A: output_to: %s"%args_a["output_to"])
        
    elif calibration_type == "signal-static":
        print("Executing 'signal-static' calibration")
        # Kcw, Ctissue and alpha are under study
        c_tissue, k_cw, alpha = params # Unpack
        _, beta = args["constitutive_parameters"] # Reutilize beta value
        print(" > c_tissue: %.4f"%c_tissue)
        print(" > k_cw: %.4f"%k_cw)
        print(" > alpha: %.4f"%alpha)

        args_a["constitutive_parameters"] = (c_tissue, beta)
        args_a["K_stiffness"] = k_cw
        args_a["output_to"] = path
        args_a["isotropic_prestrain"] = alpha
        print("Run A: output_to: %s"%args_a["output_to"])
                                            
    elif calibration_type == "signal-dynamic":
        # Gamma insp and gamma exp are being calibrated
        gamma_insp, gamma_exp = params
        args_a["additional_resistances"]["pedley_config"]["expiratory_gamma"] = gamma_exp
        args_a["additional_resistances"]["pedley_config"]["inspiratory_gamma"] = gamma_insp
        args_a["output_to"] = path
        print("Run A: output_to: %s"%args_a["output_to"])
        
    else:
        print("Calibration type '%s' not valid"%calibration_type)
        return False

    # Delete
   # execute_vcv_simulation(args_a)

    # 'complete lung' simulation
    try:
        execute_vcv_simulation(args_a)
        flag = True
    except:
        flag = False  

    # confirm whether the simulations succeded or crashed
    if not flag: 
        print("Convergence error when simulating the 'complete' lung")
    
    return flag
    

def execute_dual_simulations(path, args, params, null=1e-8):
    
    ''' 
    Execute two vcv simulations according to the current parameters
    '''
    
    # declare paths
    full_folder = path+"full/"; nocw_folder = path+"nocw/"
    
    # create paths
    for folder in [path, full_folder, nocw_folder]:
        if not os.path.isdir(folder): os.mkdir(folder)

    # export the parameters
    np.savetxt(path+"iteration_params.txt", params)

    # unpack parameters
    c_tissue, k_stiffness, alpha = params
    _, beta = args["constitutive_parameters"]
    
    # update dictionaries; a = full; b = no-cw
    args_a = args.copy()  
    args_a["constitutive_parameters"] = (c_tissue, beta)
    args_a["K_stiffness"] = k_stiffness
    args_a["output_to"] = full_folder
    args_a["isotropic_prestrain"] = alpha
    print("Run A: output_to: %s"%args_a["output_to"])
    
    args_b = args_a.copy()
    args_b["K_stiffness"] = null
    args_b["output_to"] = nocw_folder
    args_b["isotropic_prestrain"] = alpha
    print("Run B: output_to: %s"%args_b["output_to"])
   
    execute_vcv_simulation(args_a)

    # 'complete lung' simulation
    try:
        execute_vcv_simulation(args_a)
        flag_a = True
    except:
        flag_a = False
    
    # 'no-cw' simulation
    try:
        execute_vcv_simulation(args_b)
        flag_b = True
    except:
        flag_b = False
    
    # confirm whether the simulations succeded or crashed
    if not flag_a: 
        print("Convergence error when simulating the 'complete' lung")
    if not flag_b:
        print("Convergence error when simulating the 'no-cw' lung")
    
    # return a joint flag
    return (flag_a and flag_b)


def execute_parallel_simulations(path, args, params, null=1e-8):
    
    ''' 
    Execute two vcv simulations according to the current parameters
    '''
    
    # declare paths
    full_folder = path+"full/"; nocw_folder = path+"nocw/"
    
    # create paths
    for folder in [path, full_folder, nocw_folder]:
        if not os.path.isdir(folder): os.mkdir(folder)

    # export the parameters
    np.savetxt(path+"iteration_params.txt", params)

    # unpack parameters
    c_tissue, k_stiffness, alpha = params
    _, beta = args["constitutive_parameters"]
    
    # update dictionaries; a = full; b = no-cw
    args_a = args.copy()  
    args_a["constitutive_parameters"] = (c_tissue, beta)
    args_a["K_stiffness"] = k_stiffness
    args_a["isotropic_prestrain"] = alpha
    args_a["output_to"] = full_folder
    print("Run A: output_to: %s"%args_a["output_to"])
    
    args_b = args_a.copy()
    args_b["K_stiffness"] = null
    args_b["isotropic_prestrain"] = alpha
    args_b["output_to"] = nocw_folder
    print("Run B: output_to: %s"%args_b["output_to"])

    args = [args_a, args_b]
    
    with Pool(2) as p:
        p.map(execute_vcv_simulation,args)
    
    # return a joint flag
    return True

def determine_misfits_from_signal(signalpath,simulationpath,
                                  Tsyr,Tpausa,Texp, 
                                  ppeak, pplat, peep,vt,
                                  verbose=False,figure=False,
                                  interpolator_type='linear',
                                  nsamples = 250,
                                  normalize_misfit=True,
                                  normpres = 25,
                                  normflow = 1.0,
                                  normvol = 0.20,
                                  include_volume_signal = False,
                                  include_pressure_signal = False,
                                  include_flux_signal = False,
                                  include_ppeak = False,
                                  include_pplat = False,
                                  include_peep = False,
                                  include_vpeak = False,
                                  displace_signal = 0): 
    
    # Displace the signals 
    ds = displace_signal

    # Load the experimental signal
    mat = np.load(signalpath)
    Paw = mat['pressure']
    flow = mat['flow']
    vol = mat['volume']
    time = mat['time']
    time -= time[0]
    
    # Determine the number of cycles
    Tcycle = Tsyr+Tpausa+Texp
   
    #  Load a sample calibration
    simulationpath += "/Signals/"
    stime = np.load(simulationpath+"effectivetimes.npy").flatten()
    sPaw = np.load(simulationpath+"presionestodas.npy").flatten()*10.1972
    sflow = np.load(simulationpath+"fluxes.npy").flatten()
    svol = np.load(simulationpath+"volumenes.npy").flatten()
    svol = svol - svol[0]
        
    # Generate piecewise linear functions
    fsPaw  = interp1d(stime, sPaw,  kind=interpolator_type, fill_value="extrapolate")
    fsflow = interp1d(stime, sflow, kind=interpolator_type, fill_value="extrapolate")
    fsvol  = interp1d(stime, svol,  kind=interpolator_type, fill_value="extrapolate")
    
    fPaw  = interp1d(time, Paw,  kind=interpolator_type)
    fflow = interp1d(time, flow, kind=interpolator_type)
    fvol  = interp1d(time, vol,  kind=interpolator_type)
    
    # Sample  measurements
    simulated_pplat = fsPaw(Tsyr+Tpausa*(0.999)) # Plateau pressure
    simulated_ppeak = fsPaw(Tsyr*0.999) # Peak pressure
    simulated_peep  = sPaw[0] # PEEP value
    simulated_vpeak = fsvol(Tsyr+Tpausa*(0.95)) # Maximum volume
    
    # Signal VT
    signal_vt = fvol(Tsyr+Tpausa*(0.95))

    err = 1e-8
    ttime = np.linspace(0+err,Tcycle-err,nsamples)
    
    if figure:
        fig,axes = plt.subplots(nrows=3,figsize=(8,8),dpi=200)
        for e,ax in enumerate(axes):
            if e == 0: # Pressure
                ax.plot(ttime,fPaw(ttime),color="tab:blue")
                ax.plot(ttime,fsPaw(ttime),color="tab:orange")
                ax.fill_between(time[time<Tcycle],Paw[time<Tcycle],0,alpha=0.10,color="tab:blue")
                ax.set_ylim((0,np.ceil(Paw.max()*1.05/5.0)*5.0))
                ax.set_xticks([])
                ax.set_ylabel("Pressure (cmH2O)")
                
            if e == 1: # Flow
                ax.plot(ttime,fflow(ttime),color="tab:blue")
                ax.plot(ttime,fsflow(ttime),color="r")
                ax.fill_between(time[time<Tcycle],flow[time<Tcycle],0,alpha=0.10,color="tab:blue")
                ax.set_ylabel("Flow (L/s)")
                ax.set_xticks([])
        
            if e == 2:
                ax.plot(ttime,fvol(ttime),color="tab:blue")
                ax.plot(ttime,fsvol(ttime),color="r")
                ax.fill_between(time[time<Tcycle],vol[time<Tcycle],0,alpha=0.10,color="tab:blue")
                ax.set_ylabel("Volume (L)")
                ax.set_xlabel("Time (s)")
                
            for cy in [0]:
                
                ax.axvline(Tsyr+cy*Tcycle, ls = "--", color="k",alpha=0.25)
                ax.axvline(Tsyr+Tpausa+cy*Tcycle, ls = "--", color="k",alpha=0.25)
                ax.axvline((cy+1)*Tcycle, ls = "-", color="k",alpha=0.45)
    
    
    # Detailed misfit manager
    details = {}
    
    # Determine misfit
    pmisfit = 0.0
    fmisfit = 0.0
    vmisfit = 0.0
    misfit = 0.0
    
    # Counters
    npm = 0
    nfm = 0
    nvm = 0
    
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
        pmisfit /= (npm*normpres**2)
        fmisfit /= (nfm*normflow**2)
        vmisfit /= (nvm*normvol**2)
    
    if include_pressure_signal:
        misfit += pmisfit
        details.update({"pressure-signal":pmisfit})
        
    if include_flux_signal:
        misfit += fmisfit
        details.update({"flux-signal":fmisfit})
        
    if include_volume_signal:
        misfit += vmisfit
        details.update({"volume-signal":vmisfit})
    
    if include_ppeak:
        misfit_ppeak = (simulated_ppeak-ppeak)**2/ppeak**2
        misfit += misfit_ppeak
        details.update({"ppeak-error":misfit_ppeak})
        
    if include_pplat:
        misfit_pplat = (simulated_pplat-pplat)**2/pplat**2
        misfit += misfit_pplat
        details.update({"pplateau-error":misfit_pplat})
        
    if include_peep:
        misfit_peep = (simulated_peep-peep)**2/peep**2
        misfit += misfit_peep
        details.update({"peep-error":misfit_peep})
    
    if include_vpeak:
        misfit_vpeak = ((simulated_vpeak-signal_vt)/signal_vt)**2
        misfit += misfit_vpeak*10
        details.update({"volume-error":misfit_vpeak})
                       
        
    if verbose:
        print()
        if include_pressure_signal: print("%18s"%"Pressure misfit:"+" %.1f (%4.1f)"%(pmisfit,pmisfit/misfit*100)+"%)")
        if include_flux_signal: print("%18s"%"Flow misfit:"+" %.1f (%4.1f)"%(fmisfit,fmisfit/misfit*100)+"%)")
        if include_volume_signal: print("%18s"%"Volume misfit:"+" %.1f (%4.1f"%(vmisfit,vmisfit/misfit*100)+"%)")
        if include_pplat: print("%18s"%"Pplat misfit:"+" %.1f (%4.1f"%(misfit_pplat, misfit_pplat/misfit*100)+"%)")
        if include_ppeak: print("%18s"%"Ppeak misfit:"+" %.1f (%4.1f"%(misfit_ppeak, misfit_ppeak/misfit*100)+"%)")
        if include_peep: print("%18s"%"PEEP misfit:"+" %.1f (%4.1f"%(misfit_peep, misfit_peep/misfit*100)+"%)")
        if include_vpeak: print("%18s"%"Vpeak misfit:"+" %.1f (%4.1f"%(misfit_vpeak, misfit_vpeak/misfit*100)+"%)")
        
        print("-"*30)
        print("%18s"%"Overall misfit:"+" %.1f"%misfit)
    
    return misfit, details

def determine_respiratory_mechanics(Tsyr, Tpausa, Texp, path_to_data, model="",verbose=False,
                                    static=False):
        
    '''

    04-03-2025:
        Determine the compliance and resistance using the computational signals for a
        VC-simulation, considering the wave-form and similar stuff. 

    '''
    
    # Time management
    Tcycle = Tsyr+Tpausa+Texp
    
    # Load data
    pressures = np.load(path_to_data+"%spresionestodas.npy"%model).flatten()*10.1972 # cmH2O
    times = np.load(path_to_data+"%seffectivetimes.npy"%model).flatten() # s
    fluxes = np.load(path_to_data+"%sfluxes.npy"%model).flatten() # L/s
    volumes = np.load(path_to_data+"%svolumenes.npy"%model).flatten() # 
    volumes -= volumes[0]
        
    # Determine PEEP
    peep = pressures[0]
        
    # Determine compliance and resistance associated to the complete signal
    FV=np.zeros((fluxes.shape[0],2))
    FV[:,0]=fluxes
    FV[:,1]=volumes
    reg = linear_model.LinearRegression()
    reg.fit(FV,pressures)
    R,E=reg.coef_
    C = (1000/E)

    
    if not static:

        if verbose:   
            print("Complete signal regression:")
            print(" > C_rs,st: %.2f (mL/cmH2O)"%C)
            print(" > R_rs,st: %.2f (cmH2O/L/s)"%R)
        return C, R, peep
    
    # Determine maximum time
    max_time = max(times)
        
    # Number of simulated cycles
    Ncycles = max_time/Tcycle
        
    # Data holders
    peak_pressures = []
    plateau_pressures = []
    static_compliances = []
    airway_resistances = []

    # Correction for an incomplete cycle
    if Ncycles < 0.75:
        print("Incomplete cycle! %.2f"%Ncycles)
        #return None
    elif Ncycles < 1.05:
        Ncycles = 1
    else:
        Ncycles = np.round(Ncycles).astype(int)
            
    # Evaluate each cycle
    for i in range(Ncycles):
            
        # Isolate the cycle associated to the time
        cycle_mask = np.logical_and(times<=(i+1)*Tcycle,times>i*Tcycle)
        
        # c-prefix indicates cycle
        ctimes = times[cycle_mask]
        cfluxes = fluxes[cycle_mask]
        cvolumes = volumes[cycle_mask]
        cpressures = pressures[cycle_mask]
            
        # Zero-th time
        ctimes = ctimes - i*Tcycle 
        
        # Minimum volume in the cycle
        initial_volume = cvolumes[0]
            
        # Peak pressure detection
        T = np.argmin(np.abs(ctimes-Tsyr))
        peak_pressures += [cpressures[T]] 
        peak_volume = cvolumes[T]
            
        # Plateau pressure detection
        T = np.argmin(np.abs(ctimes-Tsyr-Tpausa))
        plateau_pressures += [cpressures[T-1]] 
            
        # Static compliance
        delta_volume = peak_volume - initial_volume
        static_compliances += [1000*delta_volume/(plateau_pressures[i]-peep)]
        
        # Airway resistances
        airway_resistances += [(peak_pressures[i]-plateau_pressures[i])/cfluxes[10]]
            
        if verbose:
            print("\nCycle number #%i:"%(i+1))
            print(" > C_stat: %.2f (mL/cmH2O)"%(static_compliances[i]))
            print(" >   R_aw: %.2f (cmH2O/L/s)"%airway_resistances[i])
            

    return np.median(static_compliances), np.median(airway_resistances), peep
    
def process_simulation_stiffness(path, wave_config, geometry_factor=1.0, static = False):

    '''
    
    Pass some arguments regarding the location of the signals
    archives and other constants needed to compute compliances.
    
    '''
    
    # Shape paths
    full_path = path+"full/Signals/"
    nocw_path = path+"nocw/Signals/"
    
    Tsyr=wave_config["Tsyr"]
    Tpausa=wave_config["Tpausa"]
    Texp=wave_config["Texp"]
    
    # Retrieve full values
    full_values = determine_respiratory_mechanics(Tsyr, Tpausa, Texp, full_path, static=static)
    nocw_values = determine_respiratory_mechanics(Tsyr, Tpausa, Texp, nocw_path, static=static)
    peep = full_values[2]
    
    # Determine simulation-sought values
    lung_compliance = nocw_values[0]
    rs_compliance = full_values[0]
    cw_compliance = (lung_compliance*rs_compliance)/(lung_compliance-rs_compliance)
    
    return np.array([cw_compliance, lung_compliance, peep])


def compute_misfit(values, targets):
    '''
    
    Compute the misfit which is a sum of squared differences between the 
    target and the current values being sought.
    
    '''
    
    misfit = 0.0
    for t,v in zip(targets,values): 
        misfit += (t-v)**2
    return misfit

def step(path, it, args, x, calibration_config, wave_config,
         parallel=True,geometry_factor=1.0):
    '''
    All the steps joined in a simple function
    '''   
    # Type of calibration 'regression' or 'partial-signal'
    calibration_type = calibration_config['calibration_type']

    # Build path
    it_path = path+"%3.3i/"%it;
    
    if not os.path.isdir(it_path): os.mkdir(it_path)
    
    if calibration_type == "regression":
        
        # Execute simulation 0
        if parallel:
            flag = execute_parallel_simulations(it_path, args, x)
        else:
            flag = execute_dual_simulations(it_path, args, x)
            
        # Determine compliances
        values = process_simulation_stiffness(it_path, wave_config,
                                              geometry_factor=geometry_factor)
            
        # Warn if any simulation diverged
        if not flag: raise Exception("Convergence error at it:%s!"%it_path[-3:])
        
        # Return the misfit
        return compute_misfit(values, calibration_config['targets']) 
        
    elif calibration_type == "partial-signal":
        
        flag = execute_single_simulation(it_path,args,x,calibration_config['calibration_type'])
                
        values = determine_misfits_from_signal(calibration_config["target_signals"],
                                               it_path,
                                               wave_config["Tsyr"],
                                               wave_config["Tpausa"],
                                               wave_config["Texp"],
                                               wave_config["ppeak"],
                                               wave_config["pplat"],
                                               wave_config["peep"],
                                               normvol = wave_config['vt'],
                                               displace_signal=calibration_config['displace_signal'])

        return values[0]
    
    elif calibration_type == "signal-static":
        
        flag = execute_single_simulation(it_path,args,x,calibration_config['calibration_type'])
                
        values = determine_misfits_from_signal(calibration_config["target_signals"],
                                               it_path,
                                               wave_config["Tsyr"],
                                               wave_config["Tpausa"],
                                               wave_config["Texp"],
                                               wave_config["ppeak"],
                                               wave_config["pplat"],
                                               wave_config["peep"],
                                               wave_config["vt"],
                                               normvol = wave_config['vt'],
                                               include_volume_signal = False,
                                               include_pressure_signal = False,
                                               include_flux_signal = False,
                                               include_ppeak = False,
                                               include_pplat = True,
                                               include_peep = True,
                                               include_vpeak = True,
                                               displace_signal=calibration_config['displace_signal'])
        return values[0]
    
    elif calibration_type == "signal-dynamic":
        
        flag = execute_single_simulation(it_path,args,x,calibration_config['calibration_type'])
                
        values = determine_misfits_from_signal(calibration_config["target_signals"],
                                               it_path,
                                               wave_config["Tsyr"],
                                               wave_config["Tpausa"],
                                               wave_config["Texp"],
                                               wave_config["ppeak"],
                                               wave_config["pplat"],
                                               wave_config["peep"],
                                               wave_config["vt"],
                                               normvol = wave_config['vt'],
                                               include_volume_signal = True,
                                               include_pressure_signal = True,
                                               include_flux_signal = True,
                                               include_ppeak = True,
                                               include_pplat = False,
                                               include_peep = False,
                                               include_vpeak = False,
                                               displace_signal=calibration_config['displace_signal'])
        return values[0]
        
    else:
        print("Error in the calibration_type: '%s'"%calibration_type)
        return None
    


# %% Nelder-Mead related

def reflect(vs,wid):
    # Compute the reflected vertex of the worst point
    nv = np.zeros_like(vs[wid]) # new vertex
    for e,v in enumerate(vs):
        if e == wid:
            nv -= v
        else:
            nv += v
    return nv

def extend(wv,rv,alpha=1.0):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that alpha allows to modify the length of the extension
    return rv + alpha*ext

def inner_contraction(wv, rv, beta=0.5):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that beta sends the new point the original simplex
    return rv - beta*ext

def outer_contraction(wv, rv, gamma=0.5):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that beta sends the new point the original simplex
    return rv + gamma*ext

def shrink(vs, sorter, delta=0.5):
    
    mid_ext = vs[sorter[0]] - vs[sorter[1]]
    new_mid = vs[sorter[1]] + delta*mid_ext
    
    wst_ext = vs[sorter[0]] - vs[sorter[2]]
    new_wst = vs[sorter[2]] + delta*wst_ext
    
    return new_mid, new_wst

def iterate(path,it,vs,zs,args,calibration_config,wave_config):
    
    # Sort the points depending on they performance
    sorter = np.argsort(zs)
    
    # Retrieve the sorted ids for each vector
    bid = sorter[0] # best
    mid = sorter[1] # mid
    wid = sorter[2] # worst
    
    # Determine reflected vector  
    rv = reflect(vs,wid)
    
    # Compute the z-value of the new vertex
    text_manager(path+"it_indexer.txt", "%3.3i - Reflect\n"%(it+1), "a")
    it+=1; rz = step(path,it,args,rv,calibration_config,wave_config)
    
    if rz < zs[bid]: # Extend
    
        # if the reflected point is better than the best point, extend
        # Determine extended point
        ev = extend(vs[wid],rv)
        # Compute the z-value for the extended point
        text_manager(path+"it_indexer.txt", "%3.3i - Extend\n"%(it+1), "a")
        it+=1; ez = step(path,it,args,ev,calibration_config,wave_config)
        
        if ez < rz:
            return (ev, ez), "Extend",it
        else:
            return (rv, rz), "Reflect",it
        
    elif rz < zs[mid]: 
        return (rv,rz), "Reflect",it
        
    elif rz < zs[wid]:
        # if the reflected point is better than the worst point, but not better than the other
        # two, then contract
        icv = inner_contraction(vs[wid], rv)
        ocv = outer_contraction(vs[wid], rv)
        text_manager(path+"it_indexer.txt", "%3.3i - Inner Contraction\n"%(it+1), "a")
        it+=1; icz = step(path,it,args,icv,calibration_config,wave_config)
        text_manager(path+"it_indexer.txt", "%3.3i - Outer Contraction\n"%(it+1), "a")
        it+=1; ocz = step(path,it,args,ocv,calibration_config,wave_config)
        
        if icz < ocz: 
            return (icv,icz), "Inner-contraction",it
        else:
            return (ocv,ocz), "Outer-contraction",it
        
    else:
        # Shrinkage
        nmv, nwv = shrink(vs,sorter)
        text_manager(path+"it_indexer.txt", "%3.3i - Shrink-1\n"%(it+1), "a")
        it+=1; nmz = step(path,it,args,nmv,calibration_config,wave_config)
        text_manager(path+"it_indexer.txt", "%3.3i - Shrink-2\n"%(it+1), "a")
        it+=1; nwz = step(path,it,args,nwv,calibration_config,wave_config)
        return (nmv,nmz,nwv,nwz), "Shrink", it


def process(vs,zs,out):
    
    if len(out) == 4:
        # implies shrinkage
        # Sort the points depending on they performance
        sorter = np.argsort(zs)
        bid = sorter[0] # best
        
        nvs = np.array([vs[bid], out[0],out[2]])
        nzs = [zs[bid],out[1],out[3]]
    
    elif len(out) == 2:
        
        sorter = np.argsort(zs)
        bid = sorter[0] # best
        mid = sorter[1]
        
        nvs = np.array([vs[bid], vs[mid],out[0]])
        nzs = [zs[bid],zs[mid],out[1]]
    else:
        raise Exception("Error at process routine")
        return None

    # sort the new points
    bid, mid, wid = np.argsort(nzs)
   
    dim = len(vs[0])
    
    if dim == 3:
        area_err = np.abs(np.linalg.det(np.hstack([nvs])))
    elif dim==2: 
        area_err = np.abs(np.linalg.det(np.hstack([nvs,np.ones((3,1))])))
    else:
        area_err = -100000.
    funct_err = np.abs((nzs[bid]-nzs[wid])/(np.abs(nzs[bid])+1e-9))
    
    err = [area_err, funct_err]
    return nvs, nzs, err


def text_manager(path,text,mode):
    
    file = open(path,mode=mode)
    file.write(text)
    file.close()
    

def manage_paths(paths, calibration_config):
    
    ''' 
    Determines the current path were the simulation data will be
    dumped, also creates the 'history' folder if it does not exist, 
    which will be used to resume the analysis and to help postprocessing
    '''
    
    # In case that we restart calibration from another point
    continue_calibration = calibration_config['continue_calibration']
    restart_from = calibration_config['restart_from']
    
    # Calibration config
    maximum_evaluations = calibration_config['maximum_evaluations']
    
    # Paths asociated data
    task_name = paths['task_name']
    output_to = paths['output_to']
    
    if not continue_calibration: 
        # We enter this branch if we are starting a new simulation
        for run_no in range(maximum_evaluations):
            # check if a folder already exists
            if not os.path.isdir(output_to+task_name%run_no):
                # if it doesn't, create and exit loop
                os.mkdir(output_to+task_name%run_no)
                break
        path = output_to+task_name%run_no
    else:
        path = output_to+task_name%restart_from
        
    if not os.path.isdir(path+"history/"): 
        print(" > Creating 'history' directory")
        os.mkdir(path+"history/")
    else:
        print(" > 'history' directory already exists")
    
    return path
    
def initialize_calibration(path,
                           vcv_dict,
                           calibration_config,
                           wave_config):
    
    # Calibration type
    calibration_type = calibration_config['calibration_type']
    
    # Retrieve initializing points
    x0 = calibration_config['x0']
    x1 = calibration_config['x1']
    x2 = calibration_config['x2']
    # Shape them as a numpy array
    xs = np.array([x0,x1,x2])
    
    # Continuing calibration boolean
    continue_calibration = calibration_config['continue_calibration']
    
    # Misfit holder
    ms = []
    
    # Parallel simulation flags
    parallel = calibration_config['parallel_simulations']
    
    # If the calibration starts from scratch
    if not continue_calibration:
        
        # Execute every simulation
        for it,x in enumerate(xs):
            mode = "w" if it==0 else "a" 
            text_manager(path+"it_indexer.txt", "%3.3i - Initial\n"%it, mode)

            # Complete step to compute a misfit
            ms += [step(path, it, vcv_dict, x, calibration_config, wave_config,
                        parallel=parallel,)]

        history = {"simplex":[xs.copy()],
                   "misfits":[np.array(ms)],
                   "actions":["Start"],
                   "area_err":[],
                   "func_err":[],
                   }
        np.savez(path+"history/calibration_results.npz",**history)

    else:
                
        old = np.load(path+"history/calibration_results.npz")
            
        history = {"simplex":old["simplex"],
                   "misfits":old["misfits"],
                   "actions":old["actions"],
                   "area_err":old["area_err"],
                   "func_err":old["func_err"],
                }

        simulations = os.listdir(path)
        for folder in ["history","images","it_indexer.txt"]:
            if folder in simulations: simulations.remove(folder)
        
        it = int(simulations[-1])
        xs = history["simplex"][0]
        ms = history["misfits"][0]
    
    nm_state = (it, xs, ms, history)
    return nm_state
        
def neldermead_iterations(path, nm_state, args, 
                          calibration_config, wave_config):
        
    err = [1,1]
    cycles = 0
    
    tol = calibration_config["neldermead_tolerance"]
    nitmax = calibration_config["neldermead_nitmax"]
    
    #Unpack Nelder-Mead state
    it, xs, ms, history = nm_state

    # Build targets

    # =============================================================================
    # Nelder-Mead loop
    # =============================================================================
    
    # Stopping criteria is variation within successive points being below tolerance
    # and reaching a maximum number of iterations within this loop.
    
    while((err[0] > tol and err[1] > tol) or (cycles < nitmax) ):
        
        # Declare the current iteration path    
        out, action, it = iterate(path,it,xs,ms,args,calibration_config,wave_config)
        xs, ms, err = process(xs,ms,out)
        
        history["simplex"] += [xs.copy()]
        history["misfits"] += [np.array(ms)]
        history["actions"] += [action]    
        history["area_err"] += [err[0]]    
        history["func_err"] += [err[1]]    
        
        np.savez(path+"history/calibration_results.npz",**history)
        
        cycles += 1


def execute_vcv_calibration(paths,
                            wave_config,
                            parenchyma_config,
                            peripheral_config,
                            calibration_config):
                        
    ''' 
    Calls the different steps than comprehend a VCV calibration 
    '''
   
    path = manage_paths(paths, calibration_config)

    vcv_dict = generate_vcv_dictionary(paths, 
                                       wave_config,
                                       parenchyma_config,
                                       peripheral_config,
                                       calibration_config) 
    
    nm_state = initialize_calibration(path,vcv_dict,calibration_config,wave_config)
    
    neldermead_iterations(path, nm_state, vcv_dict,
                          calibration_config, wave_config)
        
# %%


if __name__ == '__main__':
        
    print("Debugging...")
    
    # Calibration type "partial-signal" or "regression"
    calibration_type = "partial-signal"
    
    # Target values
    target_Ccw = 367.8 # placeholder (mL/cmH2O) Chest wall compliance
    target_Cl = 22.1  # placeholder (mL/cmH2O) Lung compliance
    target_peep = 10.5 # placeholder (cmH2O) PEEP value
    targets = np.array([target_Ccw, target_Cl, target_peep])
    target_pplat = 20.6# (cmH2O)
    target_ppeak = 34.1 # (cmH2O)
    target_peep = 11.1 # (cmH2O)
    
    # Physical parameters    
    K_stiffness = 0.0168333
    
    expiratory_gamma = 0.200
    inspiratory_gamma = 0.200

    constitutive_model = "bir2019"
    c_tissue = 2.5
    KK_exp = 5
    KK_factor = 1.0
    isotropic_prestrain = 1.0
    
    
    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = "/home1/agustin.perez/test-geo/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"
    path_to_mesh += "%s/"%case
    output_to = "/home1/agustin.perez/model2/output/"
    task_name = "calibration-vcv-%2.2i/"
    path_to_signals = "/home1/agustin.perez/model2/signals/PIG5-ARDSnet.mat"

    # Organization of variables
    paths = {"path_to_mesh":path_to_mesh,
             "path_to_airways":path_to_airway,
             "output_to":output_to,
             "task_name":task_name,
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
                   "peep":target_peep}
    
    
    x0 = (3.00, 0.010, 1.020)
    x1 = (2.50, 0.013, 1.050)
    x2 = (2.00, 0.012, 1.025)
    
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
    
    peripheral_config = {"K_stiffness":K_stiffness,
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
                          "target_signals":path_to_signals,
                          "continue_calibration":False,
                          "restart_from":None,
                          "maximum_evaluations":100,
                          "targets":targets,
                          "x0":x0,"x1":x1,"x2":x2,
                          "parallel_simulations":True,
                          "neldermead_tolerance":1e-6,
                          "neldermead_nitmax":30,
                          "ninternaldivs":[5,10,5,10,20,45],
                          "ncheckpoints":[2,10,2,2,4,4],
                          }
           

    execute_vcv_calibration(paths,
                            wave_config,
                            parenchyma_config,
                            peripheral_config,
                            calibration_config,
                            )
    
    
        
