# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:04:03 2024

@author: angus
"""

import calibrationLib as cl
import numpy as np
import os

# =============================================================================
# Preliminaries
# =============================================================================

# 1) Define target values
tgt_Ccw = 53.5 # (mL/cmH2O) Chest wall compliance
tgt_Cl = 39.7  # (mL/cmH2O) Lung compliance
targets = np.array([tgt_Ccw, tgt_Cl])

# 2) Define the geometry and other related parameters
geometry_factor = 8.0 # symmetry compensation


# 3) Stopping criteria for the internal loop
nitmax = 50
tol = 1e-6

# 4) Resolve destination of the files
# set accordingly to the device being used for computation
output_to = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
output_to = "/mnt/c/"+output_to[3:]
task_name = "calibration-task-%2.2i/"
args = cl.generate_octave_dictionary(vt=0.2/8.,K_perm=(5,5))

#
continue_calibration = False
restart_from = 0

if not continue_calibration: 
    # We enter this branch if we are starting a new simulation
    for run_no in range(100):
        # check if a folder already exists
        if not os.path.isdir(output_to+task_name%run_no):
            # if it doesn't, create and exit loop
            os.mkdir(output_to+task_name%run_no)
            break
    path = output_to+task_name%run_no
else:
    path = output_to+task_name%restart_from

# generate a 'history' folder to store history arrays
if not os.path.isdir(path+"history/"): os.mkdir(path+"history/")

# =============================================================================
# Simulation initialization
# =============================================================================

# Define three starting points for the nelder-mead algorithm
x0 = (0.50, 0.088) # c_birzle, k_cw
x1 = (0.60, 0.095) # c_birzle, k_cw
x2 = (0.55, 0.09) # c_birzle, k_cw

xs = np.array([x0,x1,x2])

ms = []

if not continue_calibration:
    # Execute every simulation
    for it,x in enumerate(xs):
        mode = "w" if it==0 else "a" 
        cl.text_manager(path+"it_indexer.txt", "%3.3i - Initial\n"%it, mode)

        # Complete step to compute a misfit
        ms += [cl.step(path, it, args, x)]

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

    
    
err = [1,1]
cycles = 0

# =============================================================================
# Nelder-Mead loop
# =============================================================================

# Stopping criteria is variation within successive points being below tolerance
# and reaching a maximum number of iterations within this loop.

while((err[0] > tol and err[1] > tol) or (cycles < nitmax) ):
    
    # Declare the current iteration path    
    out, action, it = cl.iterate(path,it,xs,ms)
    xs, ms, err = cl.process(xs,ms,out)
    
    history["simplex"] += [xs.copy()]
    history["misfits"] += [np.array(ms)]
    history["actions"] += [action]    
    history["area_err"] += [err[0]]    
    history["func_err"] += [err[1]]    
    
    np.savez(path+"history/calibration_results.npz",**history)
    
    cycles += 1