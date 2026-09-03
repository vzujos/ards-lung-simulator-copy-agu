# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:56:40 2025

@author: angus
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def retrieve_existing_points(sourcing_path):

    output_paths = os.listdir(sourcing_path)
    
    filter_out = []
    for typ in ["sh","txt","py"]:
        filter_by_type = lambda x: x.split(".")[-1]==typ
        filter_out += (list(filter(filter_by_type,output_paths)))
    
    for file in filter_out:
        if file in output_paths:
            output_paths.remove(file)    

    # Empty list for data management
    costs = []; params = []; dirs = []
    
    # For every target simulation
    for output_path in output_paths:
        
        output_path = sourcing_path + output_path
        
        # Evaluate every folder
        for folder in  os.listdir(output_path):
            
            if folder in ["history","it_indexer.txt"]: 
                continue
            
            # Examine the folder
            folder_path = output_path+"/"+folder+"/"
            print(folder_path)
            files = os.listdir(folder_path)
            
            # If the files are available
            if "cost.txt" in files and "params.txt" in files:
                
                # Read the simulation associated data
                cost = np.loadtxt(folder_path+"/cost.txt")
                param = np.loadtxt(folder_path+"/params.txt")
                print("%s: # cost: %.3f"%(folder,cost))
                
                # Store the information
                costs += [cost]
                params += [param]
                dirs +=[folder_path+"Signals/"]

            else:
                print("%s: dismissed"%folder)
                
    return np.array(params), np.array(costs)

# %%


#Xs, Ys = retrieve_existing_points(output_paths)

# %%

if __name__ == "__main__":
    
    root = "/home1/agustin.perez/"
        
    output_paths = [os.listdir(".")]
    
    filter_out = []
    for typ in ["sh","txt","py"]:
        filter_by_type = lambda x: x.split(".")[-1]==typ
        filter_out += (list(filter(filter_by_type,output_paths)))
    
    for file in filter_out:
        if file in output_paths:
            output_paths.remove(file)
        
# %%        
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
                print("%s: # cost: %.3f"%(folder,cost))
                
                # Store the information
                costs += [cost]
                params += [param]
                dirs +=[folder_path+"Signals/"]
            else:
                print("%s: dismissed"%folder)
            
# %%

best_id = np.argmin(costs)
sel_path = dirs[best_id]
sel_params = params[best_id]
sel_cost = costs[best_id]

print("The best cost was found to be %.3f"%sel_cost)
print("The selected parameters are:")
for txt,par in zip(["c_tissue","K_cw","K_d","alpha"],sel_params):
    print(" > %8s: %.4f"%(txt,par))
print("Path: %s"%sel_path)

