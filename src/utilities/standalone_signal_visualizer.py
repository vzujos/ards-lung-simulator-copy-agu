# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:01:28 2024

@author: angus
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']

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


def determine_misfits_from_signal(simulation_path, calibration_config, wave_config,
                                  nsamples=250, 
                                  include_volume_signal = False, w_volume_signal=1.0,
                                  include_pressure_signal = False, w_pressure_signal=1.0,
                                  include_flux_signal = False, w_flux_signal=1.0,
                                  include_ppeak = False, w_ppeak=1.0,
                                  include_pplat = False, w_pplat=1.0,
                                  include_peep = False, w_peep=1.0,
                                  include_vpeak = False, w_vpeak=1.0,
                                  normalize_misfit=True,): 
                                                                    
    
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
    sflow = -np.load(simulation_path+"fluxes.npy").flatten()
    svol = np.load(simulation_path+"volumenes.npy").flatten()
    svol = svol - svol[0]
    
    maximum_simulated_time = np.max(stime)
    if maximum_simulated_time < (Tsyr+Tpausa):
        print("The simulation finished before inspiratory pause")
        print("Returning artificially high cost")
        return 999.9
        
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
    
    # Determine signal misfit
    for t in ttime:
        T = np.floor(t/Tcycle).astype(int)
        
        # Skip these measurements if beyond the simulated time
        if t>maximum_simulated_time:
            continue
            
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
    
    if include_pressure_signal: misfit += pmisfit*w_pressure_signal
    if include_flux_signal: misfit += fmisfit*w_flux_signal
    if include_volume_signal: misfit += vmisfit*w_volume_signal
    if include_ppeak:
        misfit_ppeak = (simulated_ppeak-ppeak)**2/ppeak**2
        misfit += misfit_ppeak*w_ppeak
        
    if include_pplat:
        misfit_pplat = (simulated_pplat-pplat)**2/pplat**2
        misfit += misfit_pplat*w_pplat
        
    if include_peep:
        misfit_peep = (simulated_peep-peep)**2/peep**2
        misfit += misfit_peep*w_peep
    
    if include_vpeak:
        misfit_vpeak = ((simulated_vpeak-signal_vt)/signal_vt)**2
        misfit += misfit_vpeak*w_vpeak                       
    
    return misfit
def compute_errors_interpolated(sim_time, sim_signal,
                                exp_time, exp_signal,
                                t0, t1, N=500,
                                normalize=None,
                                mape_tol=1e-3):
    """
    Compute MAE, RMSE, and MAPE between simulated and experimental signals by:
      1. Interpolating both signals (piecewise linear).
      2. Sampling N points uniformly in [t0, t1].
      3. Applying optional normalization.
      4. Ignoring points where experimental < mape_tol for MAPE.

    Parameters
    ----------
    sim_time, exp_time : array-like
        Time arrays for simulation and experiment.
    sim_signal, exp_signal : array-like
        Signal values.
    t0, t1 : float
        Relevant time window.
    N : int
        Number of interpolation samples.
    normalize : str or None
        None, "range", "std", "max", "mean", "zscore".
    mape_tol : float
        Ignore MAPE contributions where exp < mape_tol.

    Returns
    -------
    mae : float
    rmse : float
    mape : float
    t : array
        Sample times.
    sim_interp : array
        Interpolated simulated values.
    exp_interp : array
        Interpolated experimental values.
    """

    # --- Convert to np arrays ---
    sim_time = np.asarray(sim_time)
    exp_time = np.asarray(exp_time)
    sim_signal = np.asarray(sim_signal)
    exp_signal = np.asarray(exp_signal)

    # --- Create sampling grid ---
    t = np.linspace(t0, t1, N)

    # --- Linear interpolation ---
    sim_interp = np.interp(t, sim_time, sim_signal)
    exp_interp = np.interp(t, exp_time, exp_signal)

    # --- Normalization ---
    if normalize is None:
        sim_n, exp_n = sim_interp, exp_interp

    elif normalize == "range":
        rng = np.max(exp_interp) - np.min(exp_interp)
        sim_n, exp_n = sim_interp / rng, exp_interp / rng

    elif normalize == "std":
        sigma = np.std(exp_interp)
        sim_n, exp_n = sim_interp / sigma, exp_interp / sigma

    elif normalize == "max":
        m = np.max(np.abs(exp_interp))
        sim_n, exp_n = sim_interp / m, exp_interp / m

    elif normalize == "mean":
        mu = np.mean(exp_interp)
        sim_n, exp_n = sim_interp / mu, exp_interp / mu

    elif normalize == "zscore":
        mu = np.mean(exp_interp)
        sigma = np.std(exp_interp)
        sim_n = (sim_interp - mu) / sigma
        exp_n = (exp_interp - mu) / sigma

    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    # --- MAE & RMSE ---
    mae = np.mean(np.abs(sim_n - exp_n))
    rmse = np.sqrt(np.mean((sim_n - exp_n)**2))

    # --- MAPE with tolerance filter ---
    mask = np.abs(exp_interp) >= mape_tol

    if np.any(mask):
        mape = np.mean(
            np.abs((sim_interp[mask] - exp_interp[mask]) / exp_interp[mask])
        ) * 100
    else:
        mape = np.nan  # no valid points for MAPE

    return mae, rmse, mape, t, sim_interp, exp_interp




include_signals = True
signal_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz"



sim_db = {"PIG2":{"bir":"sim_040",
                  "ma":"sim_069"},
          "PIG3":{"bir":"sim_079",
                  "ma":"sim_038"},
          "PIG4":{"bir":"sim_178",
                  "ma":"sim_043"},
          "PIG5":{"bir":"sim_285",
                  "ma":"sim_167"},
          "PIG6":{"bir":"sim_125",
                  "ma":"sim_120"},
          
          }



if True:
    
    simulation = 6
    cm = 'ma'
    renamer = {'bir':'Yeoh-type constitutive model',
               'ma':"Exponential constitutive model"}
    
    root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/"
    #root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    subj = "PIG%i"%simulation
    root += "%s/%s/"%(subj, sim_db[subj][cm])
    #root += "PIG5-mc-mediastinum-3"


    codename = "PIG%i-ARDSnet"%simulation
    Tsyr = parameters[codename]['Tsyr']
    Tpausa = parameters[codename]['Tpausa']
    Texp = parameters[codename]['Texp']
    Tinsp = Tsyr+Tpausa
    Tcycle = Texp+Tinsp
    
    fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(6,6),dpi=300)

    if os.path.isdir(root): 
        print("Root folder found")
    else:
        print("Root folder not found")
        
    sim_path = root+"PIG%i"%simulation
    sim_path = root
    if os.path.isdir(sim_path): 
        print("Simulation folder found")
    else:
        print("Simulation folder not found")
            
    
    path_to_data = sim_path+"/Signals/"
    model = ""
    lw = 1.5
    c = "tab:blue"

    # Load data
    pressures = np.load(path_to_data+"%spresionestodas.npy"%model).flatten()*10.1972 # cmH2O
    times = np.load(path_to_data+"%seffectivetimes.npy"%model).flatten() # s
    fluxes = -np.load(path_to_data+"%sfluxes.npy"%model).flatten() # L/s
    fluxes[0] = fluxes[1]
    volumes = np.load(path_to_data+"%svolumenes.npy"%model).flatten() # 
    volumes -= volumes[0]
    
    # Plot situation
    axes[0].plot(times[:-1],fluxes[:-1],alpha=1.0,color=c,label="Simulation",lw=lw)
    xmin,xmax = axes[0].get_xlim()
    axes[0].axhline(0.0,xmin=xmin,xmax=xmax,color="r",ls="--",lw=1.0, alpha=0.5)
    axes[0].set_xticks([])
    axes[0].set_ylabel("Flux (L/s)")
    ymin,ymax = axes[0].get_ylim()
    axes[0].axvspan(0.0, Tinsp,  color='tab:blue', alpha=0.2)

            
    axes[1].plot(times,volumes,alpha=1.0,color=c,lw=lw)
    axes[1].axhline(0.0,xmin=xmin,xmax=xmax,color="r",ls="--",lw=1.0, alpha=0.5)
    axes[1].set_xticks([])
    axes[1].set_ylabel("Volume (L)")
        
    axes[2].plot(times,pressures,alpha=1.0,color=c,lw=lw)
    axes[2].axhline(0.0,xmin=xmin,xmax=xmax,color="r",ls="--",lw=1.0, alpha=0.5)
    axes[2].set_ylabel("Pressure (cmH2O)")
    axes[2].set_xlabel("Time (s)")
    ymin,ymax = axes[2].get_ylim()
    axes[2].axvspan(Tinsp, Tcycle, ymin=ymin, ymax=ymax, color='tab:blue', alpha=0.2)

    
    for ax in axes:
        ax.axvline(0.0,color="k",ls="--",alpha=0.5)
        ax.axvline(Tsyr,color="k",ls="--",alpha=0.5)
        ax.axvline(Tinsp,color="k",ls="--",alpha=0.5)
        ax.axvline(Tcycle,color="k",ls="--",alpha=0.5)
    
    
    if include_signals:
        
        mat = np.load(signal_path%(simulation,simulation))
        
        measured_vol = mat['volume'].flatten()
        measured_flow = mat['flow'].flatten()
        measured_Paw = mat['pressure'].flatten()
        measured_time = mat['time'].flatten()
        
        col = 'k'
        axes[0].plot(measured_time,measured_flow, alpha=1.0,color=col,label="Signal",lw=lw)
        axes[1].plot(measured_time,measured_vol , alpha=1.0,color=col,lw=lw)
        axes[2].plot(measured_time,measured_Paw , alpha=1.0,color=col,lw=lw)
    
    axes[0].legend()
    
    plt.suptitle("PIG%i - %s"%(simulation,renamer[cm]), weight='bold')
    plt.tight_layout()

# %%

mae, rmse, mape, t, sim_interp, exp_interp = compute_errors_interpolated(times, fluxes,
                                                                    measured_time, measured_flow,
                                                                    Tinsp, Tcycle, N=500,
                                                                    normalize=None)

print("\n(MAE)  Flux: %.3f (L/s)"%mae)
print("(RMSE) Flux: %.3f (L/s)"%rmse)
print("(MAPE) Flux: %.1f "%mape+"(%)")

_, nrmse, _, _, _, _ = compute_errors_interpolated(times, fluxes,
                                                   measured_time, measured_flow,
                                                   Tinsp, Tcycle, N=500,
                                                   normalize='range')
print("(NRMSE) Flux: %.1f "%(nrmse*100)+"(%)")


print("\n"+"-"*40+"\n")
mae, rmse, mape, t, sim_interp, exp_interp = compute_errors_interpolated(times, pressures,
                                                                    measured_time, measured_Paw,
                                                                    0, Tinsp, N=500,
                                                                    normalize=None)

print("(MAE)  Airway pressure: %.2f (cmH2O)"%mae)
print("(RMSE) Airway pressure: %.2f (cmH2O)"%rmse)
print("(MAPE) Airway pressure: %.1f "%mape+"(%)")


_, nrmse, _, _, _, _ = compute_errors_interpolated(times, pressures,
                                                   measured_time, measured_Paw,
                                                   Tinsp, Tcycle, N=500,
                                                   normalize='range')
print("(NRMSE) Airway pressure: %.1f "%(nrmse*100)+"(%)")


# %%






cm = 'ma'
root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/"

database = {'bir':{'costs':[],
                   'flux':{'MAE':[],
                           'RMSE':[],
                           'MAPE':[],
                           'NRMSE':[],
                           },
                   'pressure':{'MAE':[],
                               'RMSE':[],
                               'MAPE':[],
                               'NRMSE':[],
                               },
                   },
            'ma':{'costs':[],
                  'flux':{'MAE':[],
                          'RMSE':[],
                          'MAPE':[],
                          'NRMSE':[],
                          },
                  'pressure':{'MAE':[],
                              'RMSE':[],
                              'MAPE':[],
                              'NRMSE':[],
                              },
                                   },
            }


for cm in ['bir','ma']:

    for subject in [2,3,4,5,6]:
        
        root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/"

        subj = "PIG%i"%subject
        root += "%s/%s/"%(subj, sim_db[subj][cm])
        sim_path = root
        path_to_data = sim_path+"/Signals/"

        codename = "PIG%i-ARDSnet"%subject
        Tsyr = parameters[codename]['Tsyr']
        Tpausa = parameters[codename]['Tpausa']
        Texp = parameters[codename]['Texp']
        Tinsp = Tsyr+Tpausa
        Tcycle = Texp+Tinsp
    
    
        wave_config = {"Tsyr":Tsyr,
                       "Tpausa":Tpausa,
                       "Texp":Texp,
                       "ppeak":parameters[codename]['Ppeak'],
                       "pplat":parameters[codename]['Pplat'],
                       "peep":parameters[codename]['PEEP']}
        
        calibration_config = {"paths":{"path_to_signal":signal_path%(subject,subject)}}

        database[cm]['costs'] += [determine_misfits_from_signal(sim_path, calibration_config, wave_config,
                                                                include_volume_signal = True,
                                                                include_pressure_signal = True, 
                                                                include_flux_signal = True,
                                                                include_ppeak = True, 
                                                                include_pplat = True,
                                                                include_peep = True,
                                                                include_vpeak = True,)]

        # Load data
        pressures = np.load(path_to_data+"%spresionestodas.npy"%model).flatten()*10.1972 # cmH2O
        times = np.load(path_to_data+"%seffectivetimes.npy"%model).flatten() # s
        fluxes = -np.load(path_to_data+"%sfluxes.npy"%model).flatten() # L/s
        fluxes[0] = fluxes[1]
        volumes = np.load(path_to_data+"%svolumenes.npy"%model).flatten() # 
        volumes -= volumes[0]
        
        
        mat = np.load(signal_path%(subject,subject))
            
        measured_vol = mat['volume'].flatten()
        measured_flow = mat['flow'].flatten()
        measured_Paw = mat['pressure'].flatten()
        measured_time = mat['time'].flatten()
        
        mae, rmse, mape, t, sim_interp, exp_interp = compute_errors_interpolated(times, fluxes,
                                                                                measured_time, measured_flow,
                                                                                Tinsp, Tcycle, N=500,
                                                                                normalize=None)
    
    
        _, nrmse, _, _, _, _ = compute_errors_interpolated(times, fluxes,
                                                           measured_time, measured_flow,
                                                           Tinsp, Tcycle, N=500,
                                                           normalize='range')
    
        database[cm]['flux']['MAE'] += [mae]
        database[cm]['flux']['RMSE'] += [rmse]
        database[cm]['flux']['MAPE'] += [mape]
        database[cm]['flux']['NRMSE'] += [nrmse*100]
    
        mae, rmse, mape, t, sim_interp, exp_interp = compute_errors_interpolated(times, pressures,
                                                                            measured_time, measured_Paw,
                                                                            0, Tinsp, N=500,
                                                                            normalize=None)
    
    
        _, nrmse, _, _, _, _ = compute_errors_interpolated(times, pressures,
                                                           measured_time, measured_Paw,
                                                           Tinsp, Tcycle, N=500,
                                                           normalize='range')
        
        database[cm]['pressure']['MAE'] += [mae]
        database[cm]['pressure']['RMSE'] += [rmse]
        database[cm]['pressure']['MAPE'] += [mape]
        database[cm]['pressure']['NRMSE'] += [nrmse*100]
        
# %%

fig,axes=plt.subplots(ncols=4,figsize=(12,6),dpi=300)

colors = ['tab:blue','tab:orange']
cms = ['bir','ma']
D = 0.3

boxplot = True
scatter = True

for e,(cm,col) in enumerate(zip(cms,colors)):
    
    ax = axes[0]
    x0 = 1.0+(e-0.5)*D; x1 = 2.0+(e-0.5)*D
    
    if scatter:
        ax.scatter([x0]*5, database[cm]['flux']['MAE'],color=col)
        ax.scatter([x1]*5, database[cm]['flux']['RMSE'], color=col)
        
    if boxplot:
        for f, field in enumerate(['MAE','RMSE']):
            box = ax.boxplot(database[cm]['flux'][field],positions=[x0+f],vert=True,showfliers=False, patch_artist=True)
            box['medians'][0].set_color('k')
            box['medians'][0].set_alpha(1.0)
            box['medians'][0].set_linewidth(lw)
            box['boxes'][0].set_alpha(0.5)
            box['boxes'][0].set_color(col)
            box['whiskers'][0].set_linewidth(lw)
            box['whiskers'][1].set_linewidth(lw)
            box['caps'][0].set_linewidth(lw)
            box['caps'][1].set_linewidth(lw)
    
    ax.set_ylabel("Flux (L/s)")
    ax.set_xticks([1,2])
    ax.set_xticklabels(["MAE", "RMSE"])
    ax.set_xlim((0,3))
    ax.set_ylim(0.0,0.12)
    ax.set_yticks([0.0, 0.04, 0.08, 0.12])
    

    ax = axes[1]
    
    if scatter:
        ax.scatter([x0]*5, database[cm]['pressure']['MAE'],color=col)
        ax.scatter([x1]*5, database[cm]['pressure']['RMSE'], color=col)

    if boxplot:
        for f, field in enumerate(['MAE','RMSE']):
            box = ax.boxplot(database[cm]['pressure'][field],positions=[x0+f],vert=True,showfliers=False, patch_artist=True)
            box['medians'][0].set_color('k')
            box['medians'][0].set_alpha(1.0)
            box['medians'][0].set_linewidth(lw)
            box['boxes'][0].set_alpha(0.5)
            box['boxes'][0].set_color(col)
            box['whiskers'][0].set_linewidth(lw)
            box['whiskers'][1].set_linewidth(lw)
            box['caps'][0].set_linewidth(lw)
            box['caps'][1].set_linewidth(lw)

    ax.set_ylabel("Pressure ($cmH_{2}O$)")
    ax.set_xticks([1,2])
    ax.set_xticklabels(["MAE", "RMSE"])
    ax.set_ylim(0.0,4.0)
    ax.set_yticks([0.0, 1.0, 2.0, 3.0,4.0])

    ax.set_xlim((0,3))

    ax = axes[2]
    
    if scatter:
        ax.scatter([x0]*5, database[cm]['flux']['NRMSE'], color=col)
        ax.scatter([x1]*5, database[cm]['pressure']['NRMSE'], color=col)
    
    if boxplot:
        for f, field in enumerate(['flux','pressure']):
            box = ax.boxplot(database[cm][field]['NRMSE'],positions=[x0+f],vert=True,showfliers=False, patch_artist=True)
            box['medians'][0].set_color('k')
            box['medians'][0].set_alpha(1.0)
            box['medians'][0].set_linewidth(lw)
            box['boxes'][0].set_alpha(0.5)
            box['boxes'][0].set_color(col)
            box['whiskers'][0].set_linewidth(lw)
            box['whiskers'][1].set_linewidth(lw)
            box['caps'][0].set_linewidth(lw)
            box['caps'][1].set_linewidth(lw)
        
    ax.set_xticks([1,2])
    ax.set_xticklabels(["Flux", "Pressure"])
    ax.set_ylabel("NRSME (%)")
    ax.set_xlim((0,3))
    ax.set_ylim(0.0, 25.0)
    ax.set_yticks([0.0, 5, 10, 15, 20, 25])
    
    ax = axes[3]
    
    if scatter:
        ax.scatter([x0]*5, database[cm]['costs'], color=col)
    
    if boxplot:
        box = ax.boxplot(database[cm]['costs'],positions=[x0],vert=True,showfliers=False, patch_artist=True,)
        box['medians'][0].set_color('k')
        box['medians'][0].set_alpha(1.0)
        box['medians'][0].set_linewidth(lw)
        box['boxes'][0].set_alpha(0.5)
        box['boxes'][0].set_color(col)
        box['whiskers'][0].set_linewidth(lw)
        box['whiskers'][1].set_linewidth(lw)
        box['caps'][0].set_linewidth(lw)
        box['caps'][1].set_linewidth(lw)
        
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("Cost value (-)")
    ax.set_xlim((0,2))
    ax.set_ylim(0.0,0.12)
    ax.set_yticks([0.0, 0.04, 0.08, 0.12])
     
    
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    

ax.legend()

from matplotlib.patches import Patch

# Legend elements
legend_elements = [ Patch(facecolor='tab:blue',   edgecolor='none', label='Yeoh-type'),
    Patch(facecolor='tab:orange', edgecolor='none', label='Exponential') ]

ax.legend(
    handles=legend_elements,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=True)

plt.show()

plt.tight_layout()


