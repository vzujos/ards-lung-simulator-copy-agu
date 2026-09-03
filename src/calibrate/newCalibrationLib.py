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
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.optimize import curve_fit

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
    
    # Constitutive model
    cm = parenchyma_config["constitutive_model"]

    # Isotropic prestrain F_prestress = alpha*I
    isotropic_prestrain = parenchyma_config['isotropic_prestrain']

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
    
    # Generate flow function dictionary
    flow_dict = {'activate':False,
                 'flow_func':None}
    
    # Time configuration for the volume-controlled ventilation

    
    # Modifying the base resistance for the airway tree
    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":peripheral_config['pedley_activate'],
                                               "gamma":peripheral_config['pedley_gamma'], # 0.327 original value
                                               "tolerance":peripheral_config['pedley_tolerance'],
                                               "nitmax":peripheral_config['pedley_nitmax']}
                              }
                              
    time_config = {"ncycles":wave_config['ncycles'],
                   "Tsyr":wave_config['Tsyr'],
                   "Tpausa":wave_config['Tpausa'],
                   "Texp":wave_config['Texp']}

    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    beta = 1.075

    vt = wave_config['vt']
    K_stiffness = peripheral_config['K_stiffness']
    c_tissue = parenchyma_config['c_tissue']

    # Generate flow and pressure functions dictionary
    # These functions will be manually deactivated in this implementation
    bc_dict = {'activate':False,
               'flow_func':None,
               'pressure_func':None}

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
            "flow_dict":flow_dict,
            "bc_dict":bc_dict,
            "additional_resistances":additional_resistances,
            }
        
    return args

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
    c_tissue, k_stiffness = params
    _, beta = args["constitutive_parameters"]
    
    # update dictionaries; a = full; b = no-cw
    args_a = args.copy()  
    args_a["constitutive_parameters"] = (c_tissue, beta)
    args_a["K_stiffness"] = k_stiffness
    args_a["output_to"] = full_folder
    print("Run A: output_to: %s"%args_a["output_to"])
    
    args_b = args_a.copy()
    args_b["K_stiffness"] = null
    args_b["output_to"] = nocw_folder
    print("Run B: output_to: %s"%args_b["output_to"])
    
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
    c_tissue, k_stiffness = params
    _, beta = args["constitutive_parameters"]
    
    # update dictionaries; a = full; b = no-cw
    args_a = args.copy()  
    args_a["constitutive_parameters"] = (c_tissue, beta)
    args_a["K_stiffness"] = k_stiffness
    args_a["output_to"] = full_folder
    print("Run A: output_to: %s"%args_a["output_to"])
    
    args_b = args_a.copy()
    args_b["K_stiffness"] = null
    args_b["output_to"] = nocw_folder
    print("Run B: output_to: %s"%args_b["output_to"])

    args = [args_a, args_b]
    
    with Pool(2) as p:
        p.map(execute_vcv_simulation,args)
    
    # return a joint flag
    return True

def execute_single_simulation(path, args, params, null=1e-8, parallel=False):
    
    ''' 
    Execute two vcv simulations according to the current parameters
    '''
    
    # declare paths
    folder = path+"full/"
    
    # create paths
    if not os.path.isdir(folder): os.mkdir(folder)

    # export the parameters
    np.savetxt(path+"iteration_params.txt", params)

    # unpack parameters
    cargs = args.copy()
    cargs["additional_resistances"]["pedley_config"]["gamma"] = params[0]
    
    # update dictionaries; a = full; b = no-cw
    cargs["output_to"] = folder
    print("Sending output to: %s"%folder)
    
    execute_vcv_simulation(cargs)
    
    return True


def equation_of_motion_regression(fileflujos,filepresiones,filevolumenes,
                                  verbose=False):
    
    '''
    Regression used to determine the static compliance and the rs resistance
    from the signal data of a simulation.
    '''
    # IN: Pressures in cmH2O
    # IN: Volumes in L
    # IN: Fluxes in L/s
    
    flujos=np.asarray((fileflujos)) # L/s
    presiones=np.asarray((filepresiones)) # cmH2O
    volumenes=np.asarray((filevolumenes))  # L
    
    FV=np.zeros((flujos.shape[0],2))
    FV[:,0]=flujos
    FV[:,1]=volumenes
    reg = linear_model.LinearRegression()
    reg.fit(FV,presiones)
    R,E=reg.coef_
    C = (1000*1/E)
    if verbose:
        print('Resistence (R) = %.2f (cmH2O-L/s)'%R)
        print('Compliance (C) = %.2f (mL/cmH2O)'%C)        
        print('---------------')

    return C, R

def determine_static_respiratory_mechanics(Tsyr, Tpausa, Texp, path_to_data, model="",verbose=False):
        
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
    #C,R = equation_of_motion_regression(fluxes,pressures,volumes)
    #print("Full signal:")
    #print(" > C_rs,st: %.2f (mL/cmH2O)"%C)
    #print(" > R_rs,st: %.2f (cmH2O/L/s)"%R)
        
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
            

    return np.median(static_compliances), np.median(airway_resistances)
'''
def describe_simulation(path_to_data, geometry_factor=1.0, peep=0.0, dt=0.01,
                        verbose=False, model="", delta_V=True, Tsyr=0.58,
                        Tpause=0.32,Texp=1.25):
    
    pressures = np.load(path_to_data+"%spresionestodas.npy"%model)*10.1972 # cmH2O
    times = np.load(path_to_data+"%seffectivetimes.npy"%model) # s
    fluxes = np.load(path_to_data+"%sfluxes.npy"%model)*geometry_factor # L/s
    volumes = np.load(path_to_data+"%svolumenes.npy"%model)*geometry_factor # L
    
    pressures = pressures.flatten(); times = times.flatten()
    fluxes = fluxes.flatten();    volumes = volumes.flatten()
    
    if delta_V: volumes -= volumes[0]
    
    # Determine Peak Pressures
    peak_pressure = []; plateau_pressure = []; max_flux = []
    
    # Determine end time
    endtime = np.max(times)
    
    Tinsp = Tsyr+Tpause
    Tcycle = Tinsp+Texp
    
    if endtime > Tsyr:
        pos = np.argmin(np.abs(times-Tsyr))
        peak_pressure += [pressures[pos]]
    if endtime > Tinsp:
        pos = np.argmin(np.abs(times-(Tinsp-dt)))
        plateau_pressure += [pressures[pos]]
    if endtime > (Tsyr+Tcycle):
        pos = np.argmin(np.abs(times-(Tsyr+Tcycle)))
        peak_pressure += [pressures[pos]]
    if endtime > (Tinsp+Tcycle):
        pos = np.argmin(np.abs(times-((Tinsp+Tcycle)-dt)))
        plateau_pressure += [pressures[pos]]
            
    from scipy.integrate import trapezoid
    
    mean_pressure = trapezoid(pressures,times)/endtime
    if endtime > Tsyr:
        pos = np.argmin(np.abs(times-(Tsyr-dt)))
        in_flux = fluxes[pos]
    
    if endtime > Tinsp:
        startpos = np.argmin(np.abs(times-(Tinsp-dt)))
        endpos = np.argmin(np.abs(times-(Tcycle-dt)))
        reducedfluxes = np.abs(fluxes[startpos:endpos])
        max_flux += [np.max(reducedfluxes)]
    if endtime > (Tsyr+Tcycle):
        startpos = np.argmin(np.abs(times-((Tsyr+Tcycle)-dt)))
        endpos = np.argmin(np.abs(times-(2*Tcycle-dt)))
        reducedfluxes = np.abs(fluxes[startpos:endpos])
        max_flux += [np.max(reducedfluxes)]
    
    if endtime > Tinsp:
        startpos = 0
        endpos = np.argmin(np.abs(times-(Tinsp-dt)))
        reducedfluxes = np.abs(fluxes[startpos:endpos]) 
        reducedtimes = times[startpos:endpos]
    
    volume = trapezoid(reducedfluxes,reducedtimes)

    stat_compliance = volume/(np.mean(plateau_pressure)-peep)*1000
    dyn_compliance = volume/(np.mean(peak_pressure)-peep)*1000
    
    pdrop_resistance = (np.mean(peak_pressure)-np.mean(plateau_pressure))/in_flux
    
    reg_compliance, reg_resistance = equation_of_motion_regression(fluxes,
                                                                   pressures,
                                                                   volumes)
    
    mask = np.logical_and(times>Tinsp, times<Tcycle)
    
    st = times[mask]
    sv = volumes[mask]
    
    # propose a shape for the function
    def func(t,tau,a,b):
        return a*np.exp(-t*tau) + b
    
    # fit the curve
    popt, pcov = curve_fit(func,st,sv) 
    tau = popt[0]

    if verbose:
        line = "-"*40
        print(line)
        print("%25s"%"Simulated time: "+"%.2f (s)"%endtime)
        print(line)
        print("%25s"%"Tidal volume: "+"%.2f (L)"%(volume))
        print("%25s"%"Inlet flow: "+"%.2f (L/s)"%(in_flux))
    
        print("")
        print("%25s"%"Peak pressure: "+"%.2f (cmH2O)"%(np.mean(peak_pressure)))
        print("%25s"%"Plateau pressure: "+"%.2f (cmH2O)"%(np.mean(plateau_pressure)))
        print("%25s"%"Mean pressure: "+"%.2f (cmH2O)"%mean_pressure)
        print("%25s"%"Peak Expiratory Flow: "+"%.2f (L/s)"%np.max(max_flux))
        print(line)
        print("%25s"%"Static Compliance: "+"%.1f (mL/cmH2O)"%(stat_compliance))
        print("%25s"%"Dynamic Compliance: "+"%.1f (mL/cmH2O)"%(dyn_compliance))
        print(line)
        print("%25s"%"Regr. Compliance (RS): "+"%.2f (mL/cmH2O)"%(reg_compliance))
        print("%25s"%"Regr. Resistance (RS): "+"%.2f (cmH2O/L.s)"%(reg_resistance))
        print("%25s"%"(PD) Resistance (RS): "+"%.2f (cmH2O/L.s)"%(pdrop_resistance))
        print(line)
        print("%25s"%"Exp. time constant: "+"%.2f (1/s)"%(tau))
        print(line)

    return [endtime, volume, in_flux, np.mean(peak_pressure), 
            np.mean(plateau_pressure), mean_pressure, np.max(max_flux), 
            stat_compliance, dyn_compliance, reg_compliance,
            reg_resistance, pdrop_resistance, tau]
'''
    
def process_simulation_stiffness(path, wave_config, geometry_factor=1.0):

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
    
    # Retrieve relevant values
    #full_values = describe_simulation(full_path, geometry_factor=geometry_factor,
    #                                  Tsyr=Tsyr,Tpause=Tpausa,Texp=Texp)
    
    #nocw_values = describe_simulation(nocw_path, geometry_factor=geometry_factor,
     #                                 Tsyr=Tsyr,Tpause=Tpausa,Texp=Texp)
     
    full_values = determine_static_respiratory_mechanics(Tsyr, Tpausa, Texp, full_path)
    nocw_values = determine_static_respiratory_mechanics(Tsyr, Tpausa, Texp, nocw_path)
    
    lung_compliance = nocw_values[0]
    rs_compliance = full_values[0]
    cw_compliance = (lung_compliance*rs_compliance)/(lung_compliance-rs_compliance)
    
    return np.array([cw_compliance, lung_compliance])

'''
def process_simulation_gamma(path, geometry_factor=1.0):
    ''
    
    Pass some arguments regarding the location of the signals
    archives and other constants needed to compute compliances.
    
    ''
    
    # Shape paths
    full_path = path+"full/Signals/"

    # Retrieve relevant values
    full_values = describe_simulation(full_path, geometry_factor=geometry_factor)
    peak_pressure = full_values[3]
    tau = full_values[-1]
    
    return [peak_pressure, tau]
'''
# %%


def compute_misfit(values, targets):
    '''
    
    Compute the misfit which is a sum of squared differences between the 
    target and the current values being sought.
    
    '''
    
    misfit = 0.0
    for t,v in zip(targets,values): 
        misfit += (t-v)**2
    return misfit

def step(path, it, args, x, targets, wave_config,
         parallel=True,geometry_factor=1.0):
    '''
    All the steps joined in a simple function
    '''    
    # Build path
    it_path = path+"%3.3i/"%it;
    
    if not os.path.isdir(it_path): os.mkdir(it_path)
    
    

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
    return compute_misfit(values, targets)

# %% Nelder-Mead related

def reflect(vs,wid):
    # Compute the reflected vertex of the worst point
    nv = np.zeros(2) # new vertex
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

def iterate(path,it,vs,zs,tgts,args,wave_config):
    
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
    it+=1; rz = step(path,it,args,rv,tgts,wave_config)
    
    if rz < zs[bid]: # Extend
    
        # if the reflected point is better than the best point, extend
        # Determine extended point
        ev = extend(vs[wid],rv)
        # Compute the z-value for the extended point
        text_manager(path+"it_indexer.txt", "%3.3i - Extend\n"%(it+1), "a")
        it+=1; ez = step(path,it,args,ev,tgts,wave_config)
        
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
        it+=1; icz = step(path,it,args,icv,tgts,wave_config)
        text_manager(path+"it_indexer.txt", "%3.3i - Outer Contraction\n"%(it+1), "a")
        it+=1; ocz = step(path,it,args,ocv,tgts,wave_config)
        
        if icz < ocz: 
            return (icv,icz), "Inner-contraction",it
        else:
            return (ocv,ocz), "Outer-contraction",it
        
    else:
        # Shrinkage
        nmv, nwv = shrink(vs,sorter)
        text_manager(path+"it_indexer.txt", "%3.3i - Shrink-1\n"%(it+1), "a")
        it+=1; nmz = step(path,it,args,nmv,tgts,wave_config)
        text_manager(path+"it_indexer.txt", "%3.3i - Shrink-2\n"%(it+1), "a")
        it+=1; nwz = step(path,it,args,nwv,tgts,wave_config)
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
    
    area_err = np.abs(np.linalg.det(np.hstack([nvs,np.ones((3,1))])))
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
    
    # Retrieve initializing points
    x0 = calibration_config['x0']
    x1 = calibration_config['x1']
    x2 = calibration_config['x2']
    # Shape them as a numpy array
    xs = np.array([x0,x1,x2])
    
    # Continuing calibration boolean
    continue_calibration = calibration_config['continue_calibration']
    
    # Build targets
    target_Ccw = calibration_config['target_Ccw']
    target_Cl = calibration_config['target_Cl']
    targets = np.array([target_Ccw,target_Cl])
    
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
            ms += [step(path, it, vcv_dict, x, targets, wave_config,
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
    target_Ccw = calibration_config['target_Ccw']
    target_Cl = calibration_config['target_Cl']
    targets = np.array([target_Ccw,target_Cl])    

    # =============================================================================
    # Nelder-Mead loop
    # =============================================================================
    
    # Stopping criteria is variation within successive points being below tolerance
    # and reaching a maximum number of iterations within this loop.
    
    while((err[0] > tol and err[1] > tol) or (cycles < nitmax) ):
        
        # Declare the current iteration path    
        out, action, it = iterate(path,it,xs,ms,targets,args,wave_config)
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
    
    # Target values
    target_Ccw = 367.8 # (mL/cmH2O) Chest wall compliance
    target_Cl = 22.1  # (mL/cmH2O) Lung compliance
    
    # Physical parameters    
    K_stiffness = 0.0168333
    gamma=0.200
    c_tissue = 2.13958027
    KK_exp = 5
    KK_factor = 1.0
    isotropic_prestrain = 1.0
    cm = "bir2019"

    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = "/home3/agustin.perez/test-geo/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"
    path_to_mesh += "%s/"%case
    output_to = "/home3/agustin.perez/model2/output/"
    task_name = "calibration-vcv-%2.2i/"

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
                   "ncycles":ncycles}
    
    
    x0 = (1.913543*1.02, 0.0168333*0.90)
    x1 = (1.913543*0.96, 0.0168333*1.2)
    x2 = (1.913543*1.04, 0.0168333*1.5)
    
    parenchyma_config = {"variable_porosity":True,
                         "porosity_mean":0.5,
                         "porosity_file":"phi0.xml.gz",
                         "variable_permeability":True,
                         "permeability_file":"k0.xml",
                         "permeability_exp":KK_exp,
                         "permeability_factor":KK_factor,
                         "c_tissue":c_tissue,
                         "constitutive_model":cm,
                         "isotropic_prestrain":isotropic_prestrain
                         }
    
    peripheral_config = {"K_stiffness":K_stiffness,
                         "pedley_activate":True,
                         "pedley_gamma":gamma,
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
                          "restart_from":None,
                          "maximum_evaluations":100,
                          "target_Ccw":target_Ccw,
                          "target_Cl":target_Cl,
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
                            
    
        
