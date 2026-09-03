# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 19:56:49 2025

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# %%

if True:

    manager = {"PIG2-ARDSnet":{"Tsyr":0.435,
                               "Tpausa":0.580,
                               "Texp":1.850,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG2/PIG2-ARDSnet.npz',
                               "Target volume":0.3527,
                               },
               "PIG3-ARDSnet":{"Tsyr":0.455,
                               "Tpausa":0.435,
                               "Texp":1.100,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG3/PIG3-ARDSnet.npz',
                               "Target volume":0.3870,
                               },
               "PIG4-ARDSnet":{"Tsyr":0.48,
                               "Tpausa":0.28,
                               "Texp":1.73,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG4/PIG4-ARDSnet.npz',
                               "Target volume":0.411,
                               },
               "PIG5-ARDSnet":{"Tsyr":0.375,
                               "Tpausa":0.375,
                               "Texp":1.25,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.npz',
                               "Target volume":0.4011,
                               },
               "PIG6-ARDSnet":{"Tsyr":0.540,
                               "Tpausa":0.265,
                               "Texp":1.340,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/PIG6-ARDSnet.npz',
                               "Target volume":0.2995,
                               },               
               "Template":{"Tsyr":None,
                               "Tpausa":None,
                               "Texp":None,
                               "Signal path":None,
                               "Target volume":None,
                               },
               }
    


# %%

path_to_cal = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/calibration-vcv-01/"
os.listdir(path_to_cal)

ids = []; actions = [];
with open(path_to_cal+'it_indexer.txt', 'r') as file:
    for line in file:
        split = line.split(" ")
        ids += [split[0]]
        actions += [split[2][:-1]]
    
# %% Read the calibration results file
cal_path = path_to_cal+"history/calibration_results.npz"
npz = np.load(cal_path)
misfits = npz['misfits']
simplex = npz['simplex']
acts = npz['actions']

# %% Open the id file and test params against finished simplex data

# id organizer
id_organizer = {}

for id_,action in zip(ids,actions):
    
    id_path = path_to_cal + id_
    os.listdir(id_path)
    params = []
    with open(id_path+'/iteration_params.txt', 'r') as file:
        for line in file:
            params += [float(line)]
    
    # Used to compare parameters to calibration results 'simplex' items
    tol = 1e-8
    # Vectorize the parameters
    params = np.array(params)
    # Save the appereances of this combination of parameters here
    appereances = []
    for nelder_it,simpl in enumerate(simplex):
        diff = np.linalg.norm(simpl-params,axis=1)
        where = np.where(diff<tol)[0]
        if len(where)>0:
            if len(where)==1:
                appereances += [(nelder_it, where[0])]
    
    id_organizer.update({id_:{"action":action,
                              "appereances":appereances,
                              "params":params}})
    
# %%

subj_id = "PIG6-ARDSnet"
Tsyr = manager[subj_id]['Tsyr']
Tpausa = manager[subj_id]['Tpausa']
Texp = manager[subj_id]['Texp']
signal_path = manager[subj_id]['Signal path']
signals = np.load(signal_path)
spressure = signals['pressure']
svolume = signals['volume']
sflow = signals['flow']
stime = signals['time']

id_ = ids[-2]
sim_path = path_to_cal+id_+"/Signals/"

volume = np.load(sim_path+"volumenes.npy")
volume -= volume[0]
pressure = np.load(sim_path+"presionestodas.npy").flatten()*10.1972
time = np.load(sim_path+"effectivetimes.npy")
flow = -np.load(sim_path+"fluxes.npy")
flow2 = np.load(sim_path+"prescribedfluxes.npy") 

fig,axes = plt.subplots(nrows=3,figsize=(8,8))

ax=axes[0]
ax.plot(time,pressure)
ax.plot(stime,spressure,color="r")
ax.fill_between(time,np.zeros_like(pressure),pressure,alpha=0.2)
ax.set_xticks([])
ax.set_ylabel("Pressure (cmH2O)")

ax=axes[1]
ax.plot(time,flow)
ax.plot(stime,sflow,color="r")

ax.fill_between(time,flow,0,alpha=0.2)
ax.set_xticks([])
ax.set_ylabel("Flow (L/s)")

ax=axes[2]
ax.plot(time,volume)
ax.plot(stime,svolume,color="r")
ax.fill_between(time,volume,0,alpha=0.2)
ax.set_ylabel("Volume (L)")

for ax in axes:
    ax.axvline(0.0,ls="--",color="k",alpha=0.2)
    ax.axvline(Tsyr,ls="--",color="k",alpha=0.2)
    ax.axvline(Tsyr+Tpausa,ls="--",color="k",alpha=0.2)
    ax.axvline(Tsyr+Tpausa+Texp,ls="--",color="k",alpha=0.2)

plt.tight_layout()


#plt.plot(time,flow2)