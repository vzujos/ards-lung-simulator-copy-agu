# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:57:10 2025

@author: angus
"""

import numpy as np
import os
from sklearn import linear_model
import sys

sys.path.append("/home1/agustin.perez/model2/src/calibrate/")

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


if __name__ == "__main__":
    
    
    foldername = sys.argv[1]
    
    origin = os.path.abspath(".")
    print(origin) 
    path = os.path.join(os.path.abspath(".."),foldername)+"/"
    
    from calibrate import wave_config
    
    out = process_simulation_stiffness(path, wave_config )
    
    c_cw, c_l, peep = out
    
    print(" Ccw: %.1f (mL/cmH2O)"%c_cw)
    print("  CL: %.1f (mL/cmH2O)"%c_l)
    print("PEEP: %.1f (cmH2O)"%c_cw)
