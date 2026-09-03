#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:56:33 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d



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
                                  displace_signal=0): 
    
    # Displace signal
    ds = displace_signal    
    
    # Load the experimental signal
    mat = loadmat(signalpath)
    Paw = mat['Paw_rdata'].flatten()[ds:]
    flow = mat['flow_rdata'].flatten()[ds:]
    vol = mat['volume'].flatten()[ds:]
    time = mat['time'].flatten()[ds:]
    
    if ds>0:
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
    fsPaw = interp1d(stime,sPaw,kind=interpolator_type) # Simulated
    fsflow = interp1d(stime,sflow,kind=interpolator_type) # Simulated
    fsvol = interp1d(stime,svol,kind=interpolator_type) # Simulated
    
    fPaw = interp1d(time,Paw,kind=interpolator_type) # Signal
    fflow = interp1d(time,flow,kind=interpolator_type) # Signal
    fvol = interp1d(time,vol,kind=interpolator_type) # Signal
    
    # Sample  measurements
    simulated_pplat = fsPaw(Tsyr+Tpausa*(0.999)) # Plateau pressure
    simulated_ppeak = fsPaw(Tsyr*0.999) # Peak pressure
    simulated_peep = sPaw[0] # PEEP value
    simulated_vpeak = fsvol(Tsyr+Tpausa*(0.999)) # Maximum volume
    print("Simulated PEEP: %.1f"%simulated_peep)
    print(sPaw[:10])
    err = 1e-2
    ttime = np.linspace(0+err,Tcycle-err,nsamples)
    
    if figure:
        fig,axes = plt.subplots(nrows=3,figsize=(8,8),dpi=200)
        for e,ax in enumerate(axes):
            if e == 0: # Pressure
                ax.plot(ttime,fPaw(ttime),color="tab:blue")
                ax.plot(ttime,fsPaw(ttime),color="tab:red")
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
        misfit_vpeak = ((simulated_vpeak-vt)/vt)**2
        misfit += misfit_vpeak*10
        details.update({"volume-error":misfit_vpeak})
                       
        
    if verbose:
        print()
        if include_pressure_signal: print("%18s"%"Pressure misfit:"+" %.2f (%4.1f)"%(pmisfit,pmisfit/misfit*100)+"%)")
        if include_flux_signal: print("%18s"%"Flow misfit:"+" %.2f (%4.1f)"%(fmisfit,fmisfit/misfit*100)+"%)")
        if include_volume_signal: print("%18s"%"Volume misfit:"+" %.2f (%4.1f"%(vmisfit,vmisfit/misfit*100)+"%)")
        if include_pplat: print("%18s"%"Pplat misfit:"+" %.2f (%4.1f"%(misfit_pplat, misfit_pplat/misfit*100)+"%)")
        if include_ppeak: print("%18s"%"Ppeak misfit:"+" %.2f (%4.1f"%(misfit_ppeak, misfit_ppeak/misfit*100)+"%)")
        if include_peep: print("%18s"%"PEEP misfit:"+" %.2f (%4.1f"%(misfit_peep, misfit_peep/misfit*100)+"%)")
        if include_vpeak: print("%18s"%"Vpeak misfit:"+" %.2f (%4.1f"%(misfit_vpeak, misfit_vpeak/misfit*100)+"%)")
        
        print("-"*30)
        print("%18s"%"Overall misfit:"+" %.1f"%misfit)
    
    return misfit, details


# %%





# %%

if __name__ == "__main__" and True: # Static
    

    database = {"PIG5":{"ARDSnet":{"signal_vt":0.221, # in Liters
                                   "image_vt":None,
                                   "Tsyr":0.375, 
                                   "Tpausa":0.375,
                                   "Texp":1.25,
                                   "Pplat":20.91, # cmH2O
                                   "Ppeak":33.98, # cmH2O
                                   "PEEP":10.50, # cmH2O
                                   "C_rs_reg":None, # Regression RS compliance
                                   "C_cw_reg":None, # Regression CW compliance
                                   "C_L_reg":None, # Regression L compliance
                                   "C_rs_stat":None, # Static compliance obtained from measurements,
                                   "R_rs_reg":None, # Regression resistance term
                                   "R_rs_formula":None, # RS resistance from formula
                                   "signal_path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.mat'
                                   },
                        "Template":{"signal_vt":None, # in Liters
                                   "image_vt":None,
                                   "Tsyr":None, 
                                   "Tpausa":None,
                                   "Texp":None,
                                   "Pplat":None, # cmH2O
                                   "Ppeak":None, # cmH2O
                                   "PEEP":None, # cmH2O
                                   "C_rs_reg":None, # Regression RS compliance
                                   "C_cw_reg":None, # Regression CW compliance
                                   "C_L_reg":None, # Regression L compliance
                                   "C_rs_stat":None, # Static compliance obtained from measurements,
                                   "R_rs_reg":None, # Regression resistance term
                                   "R_rs_formula":None, # RS resistance from formula
                                   "signal_path":None,
                                   },
                        "APRV":{},
                        }
    
                }

    
# %%

if __name__ == "__main__":

    evaluation_type = "static-signal"
    
    if evaluation_type == "static-signal":
        
        include_pressure_signal = (False,1.0)
        include_flux_signal = (False,1.0)
        include_volume_signal = (False,1.0)
        include_ppeak = (False,1.0)
        include_peep = (True,1.0)
        include_pplat= True,
        include_vpeak=True,




    vt = 0.221
    Tsyr = 0.375
    Tpausa = 0.375
    Texp = 1.25
    Tcycle = Texp + Tsyr + Tpausa
    target_pplat = 20.91# (cmH2O)
    target_ppeak = 33.98 # (cmH2O)
    target_peep = 10.5 # (cmH2O)
    vt = 0.221 # L
    
    displace_signal = 0
    
    nsamples=100
    exepts = ['newCalibrationLib.py','output.txt','reporter.py','__pycache__','calibrationLib3D.py',
    'errors.txt','history', 'it_indexer.txt']
    simulationpath = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/pig3-compliance-cal/'
    folders = sorted(os.listdir(simulationpath))
    for tag in exepts:
        if tag in folders: folders.remove(tag)
        
    folders.remove(folders[-1])
    signalpath = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.mat'

    
    misfits = {"overall":[],
               "pressure":[],
               "flow":[],
               "volume":[],
               "pplat":[],
               "ppeak":[],
               "peep":[],
               "novol":[]}
                
    
    for cid in folders:
        
        block_cid = int(cid)
        
        if block_cid<40 or False:
            out = determine_misfits_from_signal(signalpath,simulationpath+cid+"/full/",
                                                  Tsyr,Tpausa,Texp,
                                                  target_ppeak, target_pplat, target_peep,
                                                  0.221,
                                                  verbose=False,
                                                  figure=False,
                                                  include_pressure_signal=False,
                                                  include_flux_signal=False,
                                                  include_volume_signal=False,
                                                  include_ppeak=False,
                                                  include_peep=True,
                                                  include_pplat=True,
                                                  include_vpeak=True,
                                                  interpolator_type='linear',
                                                  nsamples=nsamples,
                                                  displace_signal=displace_signal)
            
            mis, det = out
            misfits["overall"] += [mis]
            misfits["pplat"] += [det['pplateau-error']]
            misfits["peep"] += [det['peep-error']]
            misfits["novol"] += [det['volume-error']*10]

        
    # Generate a plot
    fig,ax = plt.subplots(dpi=200)
    
    # Individual lines
    ax.plot(misfits["overall"],label="Overall",color="k",ls="-",alpha=0.75)
    ax.plot(misfits["novol"],label="Volume",color="tab:red",alpha=0.5,ls="-") 
    ax.plot(misfits["pplat"],label="Plateau",color="tab:green",alpha=0.5,ls="-")
    ax.plot(misfits["peep"],label="PEEP",color="tab:pink",alpha=0.5,ls="-") 

    ylim = ax.get_ylim()
    ylim_ = ylim[1]
    ax.set_ylim((0,ylim_))
    
    ax.legend()
    ax.set_xlabel("Iteration ")
    ax.set_ylabel("Misfit(-)")

# %%
pos = np.argmin(misfits['novol'])
determine_misfits_from_signal(signalpath,
                              simulationpath+"%3.3i/full/"%pos,
                              Tsyr,Tpausa,Texp,
                              target_ppeak, target_pplat, target_peep,
                              vt,
                              verbose=True,figure=True,
                              include_volume_signal=False,
                              include_pressure_signal=False,
                              include_flux_signal=False,
                              include_ppeak=False,
                              include_peep=True,
                              include_pplat=True,
                              include_vpeak=True,
                              interpolator_type='linear',
                              nsamples=nsamples)

# %%



if __name__ == "__main__" and False: # Dynamic
     
    vt = 0.221
    Tsyr = 0.51-0.055
    Tpausa = 0.39+0.045
    Texp = 1.1
    Tcycle = Texp + Tsyr + Tpausa
    target_pplat = 20.91# (cmH2O)
    target_ppeak = 33.98 # (cmH2O)
    target_peep = 10.5 # (cmH2O)
    vt = 0.221 # L
    
    displace_signal = 0
    
    nsamples=100
    exepts = ['newCalibrationLib.py','output.txt','reporter.py','__pycache__','calibrationLib3D.py',
    'errors.txt','history', 'it_indexer.txt']
    simulationpath = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/data/dyn3/'
    folders = sorted(os.listdir(simulationpath))
    for tag in exepts:
        if tag in folders: folders.remove(tag)
        
    folders.remove(folders[-1])
    signalpath = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG3/PIG3-ARDSnet.mat'

    
    misfits = {"overall":[],
               "pressure":[],
               "flow":[],
               "volume":[],
               "pplat":[],
               "ppeak":[],
               "peep":[],
               "novol":[]}
                
    
    for cid in folders:
        
        block_cid = int(cid)
        
        if block_cid<40 or False:
            out = determine_misfits_from_signal(signalpath,simulationpath+cid+"",
                                                  Tsyr,Tpausa,Texp,
                                                  target_ppeak, target_pplat, target_peep,
                                                  0.221,
                                                  verbose=False,
                                                  figure=False,
                                                  include_pressure_signal=True,
                                                  include_flux_signal=True,
                                                  include_volume_signal=True,
                                                  include_ppeak=True,
                                                  include_peep=False,
                                                  include_pplat=False,
                                                  include_vpeak=False,
                                                  interpolator_type='linear',
                                                  nsamples=nsamples,
                                                  displace_signal=displace_signal)
            
            mis, det = out
            misfits["overall"] += [mis]

            misfits["pressure"] += [det['pressure-signal']]
            misfits["flow"] += [det['flux-signal']]
            misfits["volume"] += [det['volume-signal']]
            misfits["ppeak"] += [det['ppeak-error']]
#            misfits["pplat"] += [det['pplateau-error']]
#            misfits["peep"] += [det['peep-error']]
#            misfits["novol"] += [det['volume-error']*10]
        

        
    # Generate a plot
    fig,ax = plt.subplots(dpi=200)
    
    # Individual lines
    ax.plot(misfits["overall"],label="Overall",color="k",ls="-",alpha=0.75)
    ax.plot(misfits["pressure"],label="Pressure signal",color="tab:orange", alpha=0.5,ls="-")
    ax.plot(misfits["volume"],label="Volume signal",color="tab:blue", alpha=0.5,ls="-")
    ax.plot(misfits["flow"],label="Flow signal",color="tab:cyan",alpha=0.5,ls="-") 
#    ax.plot(misfits["novol"],label="Volume",color="tab:red",alpha=0.5,ls="-") 
#    ax.plot(misfits["pplat"],label="Plateau",color="tab:green",alpha=0.5,ls="-") 
    ax.plot(misfits["ppeak"],label="Peak",color="tab:purple",alpha=0.5,ls="-") 
#    ax.set_xticks([0,1,2,3,4])
    
#    ax.plot(misfits["peep"],label="PEEP",color="tab:pink",alpha=0.5,ls="-") 
    ylim = ax.get_ylim()
    ylim_ = 0.025 #ylim[1]
    ax.set_ylim((0,ylim_))
    
    ax.legend()
    ax.set_xlabel("Iteration ")
    ax.set_ylabel("Misfit(-)")

# %%


pos = np.argmin(misfits['overall'])
#pos = 0
determine_misfits_from_signal(signalpath,
                              simulationpath+"%3.3i"%pos,
                              Tsyr,Tpausa,Texp,
                              target_ppeak, target_pplat, target_peep,
                              vt,
                              verbose=True,figure=True,
                              include_volume_signal=True,
                              include_pressure_signal=True,
                              include_flux_signal=True,
                              include_ppeak=True,
                              include_peep=False,
                              include_pplat=False,
                              include_vpeak=False,
                              interpolator_type='linear',
                              nsamples=nsamples)

